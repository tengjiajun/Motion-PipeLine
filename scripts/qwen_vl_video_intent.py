#!/usr/bin/env python
"""Sample video frames and ask Qwen-VL to infer the visible action intent."""

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


DEFAULT_API_KEY_FILE = Path(r"F:\LLM-pepper\qw_LLM.txt")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OUTPUT_DIR = Path("data/llm/llm_visual_intent/qwen_vl")


SYSTEM_PROMPT = """You are a visual action-intent analyzer for humanoid robot
motion imitation.

You receive 4 sampled frames from one source video. Infer only the visible human
action intent. Do not use the video filename. Do not propose robot control edits.

Output JSON only:
{
  "visual_intent": {
    "action_label": "...",
    "action_description": "...",
    "confidence": 0.0,
    "visible_body_parts": ["..."],
    "key_evidence": ["..."],
    "uncertainty": "..."
  }
}
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


def sample_video_frames(
    video_path: Path,
    frame_dir: Path,
    *,
    num_frames: int,
    max_width: int,
    jpeg_quality: int,
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to sample video frames.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count: {video_path}")

    frame_dir.mkdir(parents=True, exist_ok=True)
    indices = [
        round((i + 1) * (total_frames - 1) / (num_frames + 1))
        for i in range(num_frames)
    ]

    output_paths: list[Path] = []
    for i, frame_index in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(
                frame,
                (max_width, int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        out_path = frame_dir / f"frame_{i:02d}_idx_{frame_index}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        output_paths.append(out_path)

    cap.release()

    if len(output_paths) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames, but sampled {len(output_paths)} frames."
        )

    return output_paths


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_qwen_vl(
    *,
    api_key: str,
    base_url: str,
    model: str,
    frame_paths: list[Path],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Infer the visible human action intent from these sampled "
                "frames. The frames are in temporal order. Ignore any filename."
            ),
        }
    ]
    for frame_path in frame_paths:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame_path)},
            }
        )

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
        description="Sample 4 frames from a video and ask Qwen-VL for action intent."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--motion-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default="qwen-vl-plus-latest")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-non-json", action="store_true")
    args = parser.parse_args()

    motion_id = args.motion_id or args.video.stem
    output_dir = args.output_dir / motion_id
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sample_video_frames(
        args.video,
        frame_dir,
        num_frames=args.num_frames,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
    )

    started = time.strftime("%Y%m%d_%H%M%S")
    metadata_path = output_dir / f"{started}_{args.model}_frame_metadata.json"
    raw_response_path = output_dir / f"{started}_{args.model}_raw_response.json"
    raw_text_path = output_dir / f"{started}_{args.model}_raw_text.txt"
    intent_path = output_dir / f"{started}_{args.model}_visual_intent.json"

    metadata = {
        "video": str(args.video),
        "motion_id": motion_id,
        "model": args.model,
        "num_frames": args.num_frames,
        "frames": [str(path) for path in frame_paths],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(f"Saved sampled frames: {frame_dir}")
        print(f"Saved metadata: {metadata_path}")
        print("Dry run only. Qwen-VL API was not called.")
        return 0

    api_key = read_api_key(args.api_key_file)
    response = call_qwen_vl(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        frame_paths=frame_paths,
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

    try:
        intent = extract_json_object(content)
        intent_path.write_text(
            json.dumps(intent, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        if not args.allow_non_json:
            raise
        intent_path = None

    print(f"Saved sampled frames: {frame_dir}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved raw response: {raw_response_path}")
    print(f"Saved raw text: {raw_text_path}")
    if intent_path is not None:
        print(f"Saved visual intent JSON: {intent_path}")
    else:
        print("Model response was saved, but no JSON intent was parsed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
