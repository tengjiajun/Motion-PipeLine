#!/usr/bin/env python
"""Compare robot GIF motions against sampled source-video frames using Qwen-VL."""

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
DEFAULT_OUTPUT_DIR = Path("data/llm/llm_visual_match/qwen_vl")


SYSTEM_PROMPT = """You are a visual evaluator for humanoid robot imitation.

You receive:
1. Four source-video frames showing the target human action.
2. Four sampled frames for each candidate robot GIF.

Decide which candidate robot motion best corresponds to the source action.
Focus on coarse action intent and visible body-part movement, not visual style,
camera, lighting, or robot appearance.

Do not use file names to infer the answer. If all candidates look nearly static
or ambiguous, say so and keep confidence low.

Output JSON only:
{
  "source_action_summary": "...",
  "best_match": {
    "candidate_id": "...",
    "confidence": 0.0,
    "reason": "..."
  },
  "candidate_scores": [
    {
      "candidate_id": "...",
      "score": 0.0,
      "matching_evidence": ["..."],
      "mismatch_evidence": ["..."]
    }
  ],
  "uncertainty": "..."
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


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_qwen_vl(
    *,
    api_key: str,
    base_url: str,
    model: str,
    source_frames: list[Path],
    candidates: dict[str, list[Path]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The next four images are SOURCE frames in temporal order. "
                "Infer the human action from them."
            ),
        }
    ]

    for path in source_frames:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

    for candidate_id, frame_paths in candidates.items():
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"The next four images are candidate {candidate_id} robot "
                    "frames in temporal order."
                ),
            }
        )
        for path in frame_paths:
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
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


def parse_candidate_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    candidate_id, path = value.split("=", 1)
    return candidate_id.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask Qwen-VL which robot GIF best matches source-video frames."
    )
    parser.add_argument(
        "--source-frames-dir",
        type=Path,
        required=True,
        help="Directory containing the sampled source frames.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in id=path form. Can be repeated.",
    )
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
    parser.add_argument("--allow-non-json", action="store_true")
    args = parser.parse_args()

    source_frames = sorted(args.source_frames_dir.glob("*.jpg"))[: args.num_frames]
    if len(source_frames) != args.num_frames:
        raise RuntimeError(
            f"Expected {args.num_frames} source jpg frames in {args.source_frames_dir}, "
            f"found {len(source_frames)}."
        )

    started = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / started
    sampled_dir = output_dir / "sampled_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, list[Path]] = {}
    candidate_sources: dict[str, str] = {}
    for candidate_arg in args.candidate:
        candidate_id, gif_path = parse_candidate_arg(candidate_arg)
        if not gif_path.exists():
            raise FileNotFoundError(gif_path)
        candidate_sources[candidate_id] = str(gif_path)
        candidates[candidate_id] = sample_gif_frames(
            gif_path,
            sampled_dir / candidate_id,
            num_frames=args.num_frames,
            max_width=args.max_width,
            jpeg_quality=args.jpeg_quality,
        )

    metadata_path = output_dir / f"{args.model}_match_metadata.json"
    raw_response_path = output_dir / f"{args.model}_raw_response.json"
    raw_text_path = output_dir / f"{args.model}_raw_text.txt"
    match_path = output_dir / f"{args.model}_visual_match.json"

    metadata = {
        "model": args.model,
        "source_frames": [str(path) for path in source_frames],
        "candidate_sources": candidate_sources,
        "sampled_candidate_frames": {
            key: [str(path) for path in value] for key, value in candidates.items()
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(f"Saved metadata: {metadata_path}")
        print(f"Saved sampled candidate frames: {sampled_dir}")
        print("Dry run only. Qwen-VL API was not called.")
        return 0

    api_key = read_api_key(args.api_key_file)
    response = call_qwen_vl(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        source_frames=source_frames,
        candidates=candidates,
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
        match = extract_json_object(content)
        match_path.write_text(
            json.dumps(match, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        if not args.allow_non_json:
            raise
        match_path = None

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved raw response: {raw_response_path}")
    print(f"Saved raw text: {raw_text_path}")
    if match_path is not None:
        print(f"Saved visual match JSON: {match_path}")
    else:
        print("Model response was saved, but no JSON match was parsed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
