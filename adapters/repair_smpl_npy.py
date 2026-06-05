import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np


JOINT_FIELDS = {
    "pred_xyz_jts_24": 24,
    "pred_xyz_jts_24_struct": 24,
    "pred_xyz_jts_29": 29,
    "pred_uvd_jts": 29,
    "kp_score": 29,
}

FRAME_FIELDS = [
    "pred_shape",
    "pred_phi",
    "pred_theta_mats",
    "transl",
    "cam_root",
    "cam_trans",
    "bbox_score",
    "box",
    "crop_box",
]

ROOT_FIELDS = {"transl", "cam_root"}


def pick_person(frame_result):
    if not frame_result:
        return None
    return max(frame_result, key=lambda item: float(item.get("bbox_score", 0.0)))


def nearest_valid_index(valid_mask, index):
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No valid person frames found in the input npy.")
    nearest = np.argmin(np.abs(valid_indices - index))
    return int(valid_indices[nearest])


def interp_nan_columns(array):
    array = np.asarray(array, dtype=np.float32)
    flat = array.reshape(array.shape[0], -1)
    t = np.arange(array.shape[0], dtype=np.float32)

    for col in range(flat.shape[1]):
        values = flat[:, col]
        valid = np.isfinite(values)
        if not np.any(valid):
            flat[:, col] = 0.0
            continue
        if np.count_nonzero(valid) == 1:
            flat[:, col] = values[valid][0]
            continue
        flat[:, col] = np.interp(t, t[valid], values[valid]).astype(np.float32)

    return flat.reshape(array.shape)


def smooth_series(array, window):
    if window <= 1:
        return array
    if window % 2 == 0:
        window += 1

    pad = window // 2
    flat = array.reshape(array.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(flat)
    for col in range(flat.shape[1]):
        smoothed[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return smoothed.reshape(array.shape)


def normalize_phi(array):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return array / norms


def project_rotmats(array):
    mats = array.reshape(-1, 3, 3)
    repaired = np.empty_like(mats)
    for idx, mat in enumerate(mats):
        u, _, vh = np.linalg.svd(mat)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1.0
            rot = u @ vh
        repaired[idx] = rot.astype(np.float32)
    return repaired.reshape(array.shape)


def detect_spikes(data, valid_mask, z_scale=4.0, bridge_scale=1.5):
    data = np.asarray(data, dtype=np.float32)
    spikes = np.zeros(data.shape[:2], dtype=bool)

    # Root-center to reduce false positives from whole-body translation.
    centered = data.copy()
    if data.shape[1] > 1:
        centered = centered - centered[:, :1, :]

    velocity = np.linalg.norm(np.diff(centered, axis=0), axis=-1)

    for joint_idx in range(data.shape[1]):
        joint_vel = velocity[:, joint_idx]
        finite = np.isfinite(joint_vel)
        if np.count_nonzero(finite) < 3:
            continue

        median = np.median(joint_vel[finite])
        mad = np.median(np.abs(joint_vel[finite] - median))
        robust_sigma = 1.4826 * mad
        threshold = max(median + z_scale * robust_sigma, median * 3.0, 0.05)

        for frame_idx in range(1, data.shape[0] - 1):
            if not (valid_mask[frame_idx - 1, joint_idx] and valid_mask[frame_idx, joint_idx] and valid_mask[frame_idx + 1, joint_idx]):
                continue

            prev_joint = centered[frame_idx - 1, joint_idx]
            cur_joint = centered[frame_idx, joint_idx]
            next_joint = centered[frame_idx + 1, joint_idx]

            if not (
                np.all(np.isfinite(prev_joint))
                and np.all(np.isfinite(cur_joint))
                and np.all(np.isfinite(next_joint))
            ):
                continue

            midpoint = 0.5 * (prev_joint + next_joint)
            deviation = float(np.linalg.norm(cur_joint - midpoint))
            bridge = float(np.linalg.norm(next_joint - prev_joint))

            if deviation > threshold and deviation > bridge * bridge_scale:
                spikes[frame_idx, joint_idx] = True

    return spikes


def detect_vector_spikes(data, valid_mask, z_scale=4.0, bridge_scale=1.5, min_threshold=0.05):
    data = np.asarray(data, dtype=np.float32)
    spikes = np.zeros(data.shape[0], dtype=bool)
    if data.shape[0] < 3:
        return spikes

    velocity = np.linalg.norm(np.diff(data, axis=0), axis=-1)
    finite = np.isfinite(velocity) & valid_mask[:-1] & valid_mask[1:]
    if np.count_nonzero(finite) < 3:
        return spikes

    median = np.median(velocity[finite])
    mad = np.median(np.abs(velocity[finite] - median))
    robust_sigma = 1.4826 * mad
    threshold = max(median + z_scale * robust_sigma, median * 3.0, min_threshold)

    for frame_idx in range(1, data.shape[0] - 1):
        if not (valid_mask[frame_idx - 1] and valid_mask[frame_idx] and valid_mask[frame_idx + 1]):
            continue

        prev_value = data[frame_idx - 1]
        cur_value = data[frame_idx]
        next_value = data[frame_idx + 1]
        if not (
            np.all(np.isfinite(prev_value))
            and np.all(np.isfinite(cur_value))
            and np.all(np.isfinite(next_value))
        ):
            continue

        midpoint = 0.5 * (prev_value + next_value)
        deviation = float(np.linalg.norm(cur_value - midpoint))
        bridge = float(np.linalg.norm(next_value - prev_value))

        if deviation > threshold and deviation > bridge * bridge_scale:
            spikes[frame_idx] = True

    return spikes


def gather_track(payload):
    frames = payload["frames"]
    dominant = [pick_person(frame["result"]) for frame in frames]
    present = np.array([person is not None for person in dominant], dtype=bool)
    first_person = dominant[np.flatnonzero(present)[0]]

    arrays = {}
    for field in list(JOINT_FIELDS.keys()) + FRAME_FIELDS:
        sample = np.asarray(first_person[field], dtype=np.float32)
        arrays[field] = np.full((len(frames),) + sample.shape, np.nan, dtype=np.float32)

    idx_series = np.full(len(frames), np.nan, dtype=np.float32)

    for frame_idx, person in enumerate(dominant):
        if person is None:
            continue
        for field in arrays:
            arrays[field][frame_idx] = np.asarray(person[field], dtype=np.float32)
        idx_series[frame_idx] = float(person.get("idx", 0))

    return dominant, present, arrays, idx_series


def repair_track(payload, min_conf, spike_z, bridge_scale, smooth_window, frame_bad_ratio):
    dominant, present, arrays, idx_series = gather_track(payload)
    num_frames = len(payload["frames"])

    kp_score = arrays["kp_score"].squeeze(-1)
    conf24 = kp_score[:, :24]
    conf29 = kp_score[:, :29]

    valid24 = present[:, None] & np.isfinite(conf24)
    valid29 = present[:, None] & np.isfinite(conf29)

    invalid24 = ~valid24 | (conf24 < min_conf)
    invalid29 = ~valid29 | (conf29 < min_conf)

    spike24 = detect_spikes(arrays["pred_xyz_jts_24"], ~invalid24, z_scale=spike_z, bridge_scale=bridge_scale)
    spike29 = detect_spikes(arrays["pred_xyz_jts_29"], ~invalid29, z_scale=spike_z, bridge_scale=bridge_scale)

    invalid24 |= spike24
    invalid29 |= spike29

    root_invalid = {}
    for field in ROOT_FIELDS:
        if field in arrays:
            root_invalid[field] = (~present) | detect_vector_spikes(
                arrays[field].reshape(num_frames, -1),
                present,
                z_scale=spike_z,
                bridge_scale=bridge_scale,
                min_threshold=0.05,
            )

    repaired = {}
    repaired["kp_score"] = interp_nan_columns(
        np.where(invalid29[..., None], np.nan, arrays["kp_score"])
    )
    repaired["kp_score"] = np.clip(repaired["kp_score"], 0.0, 1.0)

    for field, joint_count in JOINT_FIELDS.items():
        joint_invalid = invalid24 if joint_count == 24 else invalid29
        repaired[field] = interp_nan_columns(
            np.where(joint_invalid[..., None], np.nan, arrays[field])
        )
        if field != "kp_score" and smooth_window > 1:
            repaired[field] = smooth_series(repaired[field], smooth_window)

    frame_invalid = (~present) | (np.mean(invalid24, axis=1) >= frame_bad_ratio)

    for field in FRAME_FIELDS:
        if field == "pred_shape":
            valid_shape = present & np.all(np.isfinite(arrays[field].reshape(num_frames, -1)), axis=1)
            if np.any(valid_shape):
                median_shape = np.nanmedian(arrays[field][valid_shape], axis=0).astype(np.float32)
                repaired[field] = np.repeat(median_shape[None, ...], num_frames, axis=0)
            else:
                repaired[field] = np.zeros_like(arrays[field], dtype=np.float32)
            continue

        field_invalid = frame_invalid
        if field in root_invalid:
            field_invalid = root_invalid[field]

        repaired[field] = interp_nan_columns(
            np.where(field_invalid[:, None], np.nan, arrays[field].reshape(num_frames, -1))
        ).reshape(arrays[field].shape)
        if field not in {"bbox_score", "box", "crop_box"} and smooth_window > 1:
            repaired[field] = smooth_series(repaired[field], smooth_window)
        if field == "pred_phi":
            repaired[field] = normalize_phi(repaired[field]).astype(np.float32)
        if field == "pred_theta_mats":
            repaired[field] = project_rotmats(repaired[field]).astype(np.float32)

    repaired_idx = interp_nan_columns(idx_series[:, None]).reshape(-1)
    repaired_idx = np.rint(repaired_idx).astype(np.int32)

    out_payload = deepcopy(payload)
    out_payload["format"] = "alphapose_smpl_v1_repaired"
    out_payload["repair_meta"] = {
        "min_conf": float(min_conf),
        "spike_z": float(spike_z),
        "bridge_scale": float(bridge_scale),
        "smooth_window": int(smooth_window),
        "frame_bad_ratio": float(frame_bad_ratio),
        "missing_frames": int(np.count_nonzero(~present)),
        "joint24_repaired": int(np.count_nonzero(invalid24)),
        "joint29_repaired": int(np.count_nonzero(invalid29)),
        "root_repaired": {
            field: int(np.count_nonzero(mask & present))
            for field, mask in root_invalid.items()
        },
        "shape_fixed_to_sequence_median": True,
    }

    valid_template_mask = present
    for frame_idx in range(num_frames):
        template_idx = nearest_valid_index(valid_template_mask, frame_idx)
        template_person = deepcopy(dominant[template_idx])
        template_person["idx"] = int(repaired_idx[frame_idx])

        for field in JOINT_FIELDS:
            template_person[field] = repaired[field][frame_idx].astype(np.float32)
        for field in FRAME_FIELDS:
            value = repaired[field][frame_idx]
            if field == "bbox_score":
                template_person[field] = float(np.asarray(value).reshape(-1)[0])
            else:
                template_person[field] = np.asarray(value, dtype=np.float32)

        out_payload["frames"][frame_idx]["result"] = [template_person]

    return out_payload


def main():
    parser = argparse.ArgumentParser(description="Repair AlphaPose SMPL npy by interpolation.")
    parser.add_argument("--input", required=True, help="Input AlphaPose SMPL npy.")
    parser.add_argument("--output", default="", help="Output repaired npy path.")
    parser.add_argument("--min-conf", type=float, default=0.55, help="Confidence threshold below which a joint is repaired.")
    parser.add_argument("--spike-z", type=float, default=4.0, help="Robust spike threshold in MAD units.")
    parser.add_argument("--bridge-scale", type=float, default=1.5, help="How much larger the deviation must be than the neighbor bridge.")
    parser.add_argument("--smooth-window", type=int, default=3, help="Optional moving-average smoothing window after interpolation.")
    parser.add_argument("--frame-bad-ratio", type=float, default=0.35, help="Interpolate whole-frame SMPL params when this fraction of 24 joints is repaired.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_repaired.npy").resolve()
    )

    payload = np.load(str(input_path), allow_pickle=True).item()
    repaired = repair_track(
        payload=payload,
        min_conf=args.min_conf,
        spike_z=args.spike_z,
        bridge_scale=args.bridge_scale,
        smooth_window=args.smooth_window,
        frame_bad_ratio=args.frame_bad_ratio,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), repaired, allow_pickle=True)
    print(output_path)
    print(repaired["repair_meta"])


if __name__ == "__main__":
    main()
