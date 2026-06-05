import argparse
from pathlib import Path

import numpy as np


def frame_has_person(frame):
    return bool(frame.get("result"))


def load_payload(path):
    return np.load(str(path), allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser(description="Drop frames with no detected person from an AlphaPose SMPL npy.")
    parser.add_argument("--input", required=True, help="Source npy to compact.")
    parser.add_argument(
        "--reference",
        default="",
        help="Optional reference npy used to decide which frames are missing. Defaults to --input.",
    )
    parser.add_argument("--output", default="", help="Output compact npy path.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    ref_path = Path(args.reference).resolve() if args.reference else input_path
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_compact.npy").resolve()
    )

    payload = load_payload(input_path)
    ref_payload = load_payload(ref_path)

    keep_mask = np.array([frame_has_person(frame) for frame in ref_payload["frames"]], dtype=bool)
    if keep_mask.shape[0] != len(payload["frames"]):
        raise ValueError("Reference and input frame counts do not match.")
    if not np.any(keep_mask):
        raise ValueError("Reference contains no valid frames to keep.")

    compact = dict(payload)
    compact["frames"] = [payload["frames"][i] for i in np.flatnonzero(keep_mask)]
    compact["compact_meta"] = {
        "source_frames": int(len(payload["frames"])),
        "kept_frames": int(np.count_nonzero(keep_mask)),
        "dropped_frames": int(np.count_nonzero(~keep_mask)),
        "reference": str(ref_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), compact, allow_pickle=True)
    print(output_path)
    print(compact["compact_meta"])


if __name__ == "__main__":
    main()
