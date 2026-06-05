#!/usr/bin/env python
"""Ask Qwen-VL for a qualitative visual comparison of source/reference/robot motion."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence


DEFAULT_API_KEY_FILE = Path(r"F:\LLM-pepper\qw_LLM.txt")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OUTPUT_DIR = Path("data/llm/llm_visual_edit/qwen_vl")


SYSTEM_PROMPT = """你是人形机器人动作模仿的视觉评审。

你会看到三组按时间顺序排列的图片：
第一组是原始视频中的人类动作；
第二组是生成的参考动作 GIF；
第三组是机器人执行动作 GIF。

请只根据图片本身判断动作，不要使用文件名，不要输出 JSON，不要输出字段表，不要输出分数、参数、阈值或控制指令。

评价时请忽略灰色/黑色、背景、地面、光照、渲染方式、机器人外形和人体外形的差异，只比较动作本身。

判断方向时，以画面中的方向为准，不要纠结人体或机器人自己的左手/右手。如果原视频中手臂指向画面右侧，参考动作或机器人动作也大致指向画面右侧，就认为方向基本一致。轻微向上抬、肘部弯曲或伸展不足可以作为动作不足描述，但不要因此直接说方向完全相反。

请用自然语言分段回答：

原视频动作：
描述人类动作。

参考动作：
描述参考动作 GIF。

参考动作不足：
只和原视频比较，说参考动作哪里不像。

机器人执行：
描述机器人执行 GIF。

机器人相对原视频的不足：
先和原视频比较，说机器人哪里不像。

机器人相对参考动作的不足：
再和参考动作 GIF 比较，说机器人执行哪里没有跟上参考动作。
"""


def read_api_key(path: Path) -> str:
    if os.environ.get("DASHSCOPE_API_KEY"):
        return os.environ["DASHSCOPE_API_KEY"].strip()
    if os.environ.get("QWEN_API_KEY"):
        return os.environ["QWEN_API_KEY"].strip()
    if not path.exists():
        raise FileNotFoundError(
            f"API key file not found: {path}. Set DASHSCOPE_API_KEY instead."
        )
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def extract_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse a JSON object from model response.")


def save_resized_jpeg(image: Image.Image, path: Path, max_width: int, quality: int) -> None:
    image = image.convert("RGB")
    if image.width > max_width:
        height = int(round(image.height * (max_width / image.width)))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    image.save(path, format="JPEG", quality=quality)


def sample_gif_frames(
    gif_path: Path,
    output_dir: Path,
    *,
    num_frames: int,
    max_width: int,
    jpeg_quality: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(gif_path) as gif:
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    if not frames:
        raise RuntimeError(f"No frames found in GIF: {gif_path}")

    indices = [
        round((i + 1) * (len(frames) - 1) / (num_frames + 1))
        for i in range(num_frames)
    ]
    output_paths: list[Path] = []
    for i, frame_index in enumerate(indices, start=1):
        out_path = output_dir / f"frame_{i:02d}_idx_{frame_index}.jpg"
        save_resized_jpeg(frames[frame_index], out_path, max_width, jpeg_quality)
        output_paths.append(out_path)
    return output_paths


def collect_frame_paths(frame_dir: Path, num_frames: int) -> list[Path]:
    paths = sorted(frame_dir.glob("*.jpg"))[:num_frames]
    if len(paths) != num_frames:
        raise RuntimeError(f"Expected {num_frames} jpg frames in {frame_dir}, found {len(paths)}.")
    return paths


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_qwen_vl(
    *,
    api_key: str,
    base_url: str,
    model: str,
    source_frames: list[Path],
    reference_frames: list[Path] | None,
    robot_frames: list[Path],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "下面第一组图片是原始视频中的人类动作，按时间顺序排列。"}
    ]
    for path in source_frames:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

    if reference_frames is not None:
        user_content.append(
            {
                "type": "text",
                "text": "下面第二组图片是生成的参考动作 GIF，按时间顺序排列。",
            }
        )
        for path in reference_frames:
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
            )

    user_content.append(
        {
            "type": "text",
            "text": "下面第三组图片是机器人执行动作 GIF，按时间顺序排列。",
        }
    )
    for path in robot_frames:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen-VL API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen-VL API request failed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare source/reference/robot frames, then ask Qwen-VL for qualitative shortcomings."
    )
    parser.add_argument("--source-frames-dir", type=Path, required=True)
    parser.add_argument("--reference-frames-dir", type=Path, default=None)
    parser.add_argument("--reference-gif", type=Path, default=None)
    parser.add_argument("--robot-frames-dir", type=Path, default=None)
    parser.add_argument("--robot-gif", type=Path, default=None)
    parser.add_argument("--motion-id", default="motion")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-thinking")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-output", action="store_true", help="Try to parse and save a JSON report.")
    parser.add_argument("--allow-non-json", action="store_true")
    args = parser.parse_args()

    if args.robot_frames_dir is None and args.robot_gif is None:
        raise ValueError("Set either --robot-frames-dir or --robot-gif.")
    if args.reference_frames_dir is not None and args.reference_gif is not None:
        raise ValueError("Set at most one of --reference-frames-dir or --reference-gif.")

    started = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.motion_id / started
    output_dir.mkdir(parents=True, exist_ok=True)

    source_frames = collect_frame_paths(args.source_frames_dir, args.num_frames)
    reference_frames = None
    if args.reference_frames_dir is not None:
        reference_frames = collect_frame_paths(args.reference_frames_dir, args.num_frames)
    elif args.reference_gif is not None:
        reference_frames = sample_gif_frames(
            args.reference_gif,
            output_dir / "reference_frames",
            num_frames=args.num_frames,
            max_width=args.max_width,
            jpeg_quality=args.jpeg_quality,
        )

    if args.robot_frames_dir is not None:
        robot_frames = collect_frame_paths(args.robot_frames_dir, args.num_frames)
    else:
        robot_frames = sample_gif_frames(
            args.robot_gif,
            output_dir / "robot_frames",
            num_frames=args.num_frames,
            max_width=args.max_width,
            jpeg_quality=args.jpeg_quality,
        )

    metadata_path = output_dir / f"{args.model}_visual_edit_metadata.json"
    raw_response_path = output_dir / f"{args.model}_raw_response.json"
    raw_text_path = output_dir / f"{args.model}_raw_text.txt"
    suggestions_path = output_dir / f"{args.model}_visual_qualitative_report.txt"

    metadata = {
        "model": args.model,
        "motion_id": args.motion_id,
        "source_frames": [str(path) for path in source_frames],
        "reference_gif": str(args.reference_gif) if args.reference_gif else None,
        "reference_frames": [str(path) for path in reference_frames] if reference_frames else None,
        "robot_gif": str(args.robot_gif) if args.robot_gif else None,
        "robot_frames": [str(path) for path in robot_frames],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Saved metadata: {metadata_path}")
        print("Dry run only. Qwen-VL API was not called.")
        return 0

    api_key = read_api_key(args.api_key_file)
    response = call_qwen_vl(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        source_frames=source_frames,
        reference_frames=reference_frames,
        robot_frames=robot_frames,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    raw_response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Qwen-VL response shape: {response}") from exc

    raw_text_path.write_text(content, encoding="utf-8")

    suggestions_path.write_text(content, encoding="utf-8")

    json_path = None
    if args.json_output:
        json_path = output_dir / f"{args.model}_visual_qualitative_report.json"
        try:
            suggestions = extract_json_object(content)
            json_path.write_text(
                json.dumps(suggestions, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            if not args.allow_non_json:
                raise
            json_path = None

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved raw response: {raw_response_path}")
    print(f"Saved raw text: {raw_text_path}")
    print(f"Saved visual qualitative report text: {suggestions_path}")
    if json_path is not None:
        print(f"Saved visual qualitative report JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
