#!/usr/bin/env python
"""Call Qwen-Max to edit one H1 reference motion from evaluation summaries.

Default behavior is intentionally single-motion: the script reads one
``motion_id`` from ``three_layer_summary.json`` and asks Qwen-Max for a
robot-level H1 reference edit plan for that motion only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_JSON = Path(
    "data/metrics/three_layer_summary_fromw1_v2/three_layer_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("data/llm/llm_edits/qwen_max_h1_reference")
DEFAULT_API_KEY_FILE = Path(r"F:\LLM-pepper\qw_LLM.txt")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


SYSTEM_PROMPT = """You are a humanoid robot H1 reference motion editor.
You receive the three-layer evaluation for exactly one motion plus optional
user and visual-model evaluations after watching the source video, the H1
reference GIF, and the robot execution GIF.

The editable target is the robot-level H1 reference motion, not the human SMPL
or canonical source motion. H1 reference motion contains root pose and 19-DoF
joint references before environment execution. Your task is to propose high-level
H1 reference edit operators. Do not output raw joint angles, quaternions, arrays,
or npy/npz values.

Strict rules:
1. Output JSON only. Do not output Markdown or any text outside JSON.
2. Do not infer action semantics from motion_id or file names.
3. Treat user evaluation as the highest-level task preference, then use the
   visual qualitative evaluation to identify visible shortcomings. Keep edits
   physically conservative and compatible with the automatic metrics.
4. If the source layer needs manual review, do not propose semantic enhancement.
   Only propose source_review or conservative repair.
5. If the H1 reference is not smooth, prefer smooth_reference, time_scale,
   reduce_reference_speed, stabilize_lower_body, stabilize_root, stabilize_torso.
6. If execution reports falling, excessive base tilt, abnormal base z, or large
   tracking error, reduce speed/amplitude and stabilize lower body/root/torso.
7. strength must be a number from 0.0 to 1.0. Be conservative; normally do not
   exceed 0.6.

8. For spatial user requests such as "raise the hand" or "move the hand down",
   use move_keybody or adjust_limb_extension. Do not choose individual joint
   names or numeric joint deltas.

Output JSON schema:
{
  "model_role": "h1_reference_motion_editor",
  "motion_id": "...",
  "summary": {
    "main_failure_modes": ["..."],
    "user_feedback_interpretation": "...",
    "recommended_next_step": "..."
  },
  "status": "source_review_needed | h1_reference_edit_needed | execution_failed | usable",
  "priority": "high | medium | low",
  "edits": [
    {
      "type": "time_scale | smooth_reference | reduce_reference_speed | reduce_reference_amplitude | move_keybody | adjust_limb_extension | stabilize_lower_body | stabilize_root | stabilize_torso | increase_hold_duration | source_review",
      "target": "whole_body | upper_body | lower_body | root | torso | left_hand | right_hand | left_elbow | right_elbow | arms | legs",
      "direction": "none | up | down | forward | backward | left | right | inward | outward",
      "segment": "all | active | early | middle | late",
      "strength": 0.0,
      "reason": "..."
    }
  ],
  "expected_effect": "..."
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


def load_single_motion(summary_json: Path, motion_id: str) -> dict[str, Any]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    for motion in data.get("motions", []):
        if motion.get("motion_id") == motion_id:
            return motion
    available = ", ".join(m.get("motion_id", "?") for m in data.get("motions", []))
    raise ValueError(f"motion_id not found: {motion_id}. Available: {available}")


def load_h1_reference_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import numpy as np

    data = np.load(str(path), allow_pickle=False)
    required = ["fps", "num_frames", "h1_joint_names", "root_pos_ref", "h1_dof_pos_ref"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path} is not a valid h1_reference_motion.npz, missing: {missing}")

    fps = int(data["fps"])
    num_frames = int(data["num_frames"])
    root_pos = data["root_pos_ref"]
    dof = data["h1_dof_pos_ref"]
    return {
        "path": str(path),
        "format": "h1_reference_motion_v1",
        "fps": fps,
        "num_frames": num_frames,
        "duration_s": float(num_frames / max(fps, 1)),
        "joint_count": int(dof.shape[1]),
        "joint_names": [str(x) for x in data["h1_joint_names"].tolist()],
        "root_position_range_m": {
            "x": [float(root_pos[:, 0].min()), float(root_pos[:, 0].max())],
            "y": [float(root_pos[:, 1].min()), float(root_pos[:, 1].max())],
            "z": [float(root_pos[:, 2].min()), float(root_pos[:, 2].max())],
        },
    }


def collect_user_evaluation(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.no_user_eval:
        return None

    user_eval = {
        "satisfaction_score_1_to_5": args.user_score,
        "satisfied": args.user_satisfied,
        "problems": args.user_problem,
        "desired_change": args.user_desired_change,
        "freeform_comment": args.user_comment,
    }

    if any(value not in (None, "") for value in user_eval.values()):
        return user_eval

    print("User evaluation is required before generating the edit plan.")
    print("Press Enter to leave an item blank.")
    score = input("Satisfaction score 1-5: ").strip()
    satisfied = input("Satisfied? yes/no/unknown: ").strip().lower()
    problem = input("Main visible problem: ").strip()
    desired_change = input("Desired change: ").strip()
    comment = input("Additional comment: ").strip()

    parsed_score: int | None
    try:
        parsed_score = int(score) if score else None
    except ValueError:
        parsed_score = None

    if satisfied in {"yes", "y", "true", "1"}:
        parsed_satisfied: bool | None = True
    elif satisfied in {"no", "n", "false", "0"}:
        parsed_satisfied = False
    else:
        parsed_satisfied = None

    return {
        "satisfaction_score_1_to_5": parsed_score,
        "satisfied": parsed_satisfied,
        "problems": problem or None,
        "desired_change": desired_change or None,
        "freeform_comment": comment or None,
    }


def build_single_motion_prompt(
    motion: dict[str, Any],
    user_evaluation: dict[str, Any] | None,
    visual_evaluation: str | None,
    h1_reference_metadata: dict[str, Any] | None,
    h1_reference_gif: Path | None,
) -> str:
    payload = {
        "instruction": (
            "Analyze this single motion only. Propose conservative H1 robot "
            "reference edit operators based on the automatic metrics, the "
            "user's manual evaluation, and optional visual qualitative "
            "evaluation. The editable target is h1_reference_motion.npz, not "
            "human canonical/SMPL data. Do not edit npy/npz directly and do "
            "not output raw joint values."
        ),
        "motion": motion,
        "h1_reference": h1_reference_metadata,
        "h1_reference_gif": str(h1_reference_gif) if h1_reference_gif else None,
        "user_evaluation": user_evaluation,
        "visual_evaluation": visual_evaluation,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> Any:
    """Parse JSON directly, or recover a fenced/embedded JSON object."""
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


def call_qwen(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
        raise RuntimeError(f"Qwen API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen API request failed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call qwen-max on one motion from a three-layer summary."
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--motion-id",
        required=True,
        help="Motion id to evaluate, for example beckon_001.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional custom single-motion prompt file. If set, summary JSON is not read.",
    )
    parser.add_argument(
        "--h1-reference-npz",
        type=Path,
        default=None,
        help="Optional h1_reference_motion.npz for this motion; metadata is included in the prompt.",
    )
    parser.add_argument(
        "--h1-reference-gif",
        type=Path,
        default=None,
        help="Optional H1 reference GIF path; image bytes are not uploaded by this text-only script.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default="qwen-max")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--user-score",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="User satisfaction score after watching execution, 1 worst and 5 best.",
    )
    parser.add_argument(
        "--user-satisfied",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the user is satisfied with this motion.",
    )
    parser.add_argument("--user-problem", default=None)
    parser.add_argument("--user-desired-change", default=None)
    parser.add_argument("--user-comment", default=None)
    parser.add_argument(
        "--visual-report",
        type=Path,
        default=None,
        help="Optional qualitative visual report text from Qwen-VL.",
    )
    parser.add_argument(
        "--no-user-eval",
        action="store_true",
        help="Skip manual user evaluation. Use only for debugging or ablations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and save the single-motion prompt without calling Qwen.",
    )
    parser.add_argument(
        "--allow-non-json",
        action="store_true",
        help="Do not fail if the model response cannot be parsed as JSON.",
    )
    args = parser.parse_args()

    if args.input is not None:
        prompt = args.input.read_text(encoding="utf-8")
    else:
        motion = load_single_motion(args.summary_json, args.motion_id)
        user_evaluation = collect_user_evaluation(args)
        visual_evaluation = (
            args.visual_report.read_text(encoding="utf-8")
            if args.visual_report is not None
            else None
        )
        h1_reference_metadata = load_h1_reference_metadata(args.h1_reference_npz)
        prompt = build_single_motion_prompt(
            motion,
            user_evaluation,
            visual_evaluation,
            h1_reference_metadata,
            args.h1_reference_gif,
        )

    output_dir = args.output_dir / args.motion_id
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y%m%d_%H%M%S")
    prompt_path = output_dir / f"{started}_{args.model}_input_prompt.json"
    raw_response_path = output_dir / f"{started}_{args.model}_raw_response.json"
    raw_text_path = output_dir / f"{started}_{args.model}_raw_text.txt"
    edit_plan_path = output_dir / f"{started}_{args.model}_edit_plan.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    if args.dry_run:
        print(f"Saved input prompt: {prompt_path}")
        print("Dry run only. Qwen API was not called.")
        return 0

    api_key = read_api_key(args.api_key_file)
    response = call_qwen(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
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
        raise RuntimeError(f"Unexpected Qwen API response shape: {response}") from exc

    raw_text_path.write_text(content, encoding="utf-8")

    try:
        edit_plan = extract_json_object(content)
        edit_plan_path.write_text(
            json.dumps(edit_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        if not args.allow_non_json:
            raise
        edit_plan_path = None

    print(f"Saved input prompt: {prompt_path}")
    print(f"Saved raw response: {raw_response_path}")
    print(f"Saved raw text: {raw_text_path}")
    if edit_plan_path is not None:
        print(f"Saved edit plan JSON: {edit_plan_path}")
    else:
        print("Model response was saved, but no JSON edit plan was parsed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
