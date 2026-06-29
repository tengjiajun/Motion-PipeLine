#!/usr/bin/env python
"""Local review UI for comparing source, H1 reference, and execution motion."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

VIDEO_STEM_ALIASES = {
    "point_left_001": "video_test_001",
}

MOTION_LABELS = {
    "beckon_001": "招手靠近",
    "bow_001": "鞠躬",
    "nod_gesture_001": "点头",
    "point_left_001": "指向左侧",
    "point_right_001": "指向右侧",
    "raise_hand_001": "举手",
    "turn_point_001": "转身指向",
    "wave_both_001": "双手挥动",
    "wave_right_001": "右手挥动",
    "welcome_001": "欢迎动作",
}

FROMW1_VERSIONS = (
    {
        "id": "original",
        "label": "Original",
        "reference_roots": ("gifs/pkl_gifs/fromw1_pkl_original",),
        "execution_roots": (
            "gifs/RoBoJuDo_H1_gifs/robojudo_pkl_original_gifs",
            "results/robojudo/robojudo_fromw1_pkl_original",
        ),
    },
    {
        "id": "canonical_v2",
        "label": "Canonical v2",
        "reference_roots": (
            "gifs/h1_reference_gifs/fromw1_pkl_canonical_v2",
            "gifs/pkl_gifs/fromw1_pkl_canonical_v2",
        ),
        "execution_roots": (
            "gifs/RoBoJuDo_H1_gifs/robojudo_pkl_canonical_v2_gifs",
            "results/robojudo/robojudo_fromw1_pkl_canonical_v2",
        ),
    },
    {
        "id": "llm_v3",
        "label": "LLM v3",
        "motion_ids": ("point_left_001",),
        "reference_roots": ("gifs/pkl_gifs/point_left_001_llm_v3",),
        "execution_roots": ("results/robojudo/point_left_001_llm_v3",),
    },
)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def media_url(path: Path | None, data_root: Path) -> str | None:
    if path is None:
        return None
    relative = relative_posix(path, data_root)
    return "/media/" + quote(relative, safe="/")


def discover_motion_ids(data_root: Path) -> list[str]:
    motion_ids: set[str] = set()
    alphapose_root = data_root / "alphapose_raw"
    if alphapose_root.exists():
        motion_ids.update(path.name for path in alphapose_root.iterdir() if path.is_dir())

    videos_root = data_root / "videos"
    reverse_aliases = {value: key for key, value in VIDEO_STEM_ALIASES.items()}
    if videos_root.exists():
        for path in videos_root.iterdir():
            if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                motion_ids.add(reverse_aliases.get(path.stem, path.stem))

    return sorted(motion_ids)


def find_source_video(data_root: Path, motion_id: str) -> Path | None:
    videos_root = data_root / "videos"
    stems = (motion_id, VIDEO_STEM_ALIASES.get(motion_id, motion_id))
    for stem in stems:
        for extension in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            candidate = videos_root / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def asset_score(path: Path, motion_id: str, preferred_terms: tuple[str, ...]) -> tuple[int, float]:
    name = path.stem.lower()
    score = 0
    if name.startswith(motion_id.lower()):
        score += 50
    if motion_id.lower() in name:
        score += 30
    for term in preferred_terms:
        if term in path.as_posix().lower():
            score += 5
    return score, path.stat().st_mtime


def find_best_asset(
    data_root: Path,
    roots: tuple[str, ...],
    motion_id: str,
    *,
    extensions: set[str],
    preferred_terms: tuple[str, ...] = (),
) -> Path | None:
    candidates: list[Path] = []
    for relative_root in roots:
        root = data_root / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions and motion_id.lower() in path.stem.lower():
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: asset_score(path, motion_id, preferred_terms))


def infer_version(text: str) -> str | None:
    normalized = text.lower().replace("\\", "/")
    llm_match = re.search(r"llm[_-]?v(\d+)", normalized)
    if llm_match:
        return f"llm_v{llm_match.group(1)}"
    if "canonical_v2" in normalized or "canonical-motion-v2" in normalized:
        return "canonical_v2"
    if "fromw1_v2" in normalized:
        return "canonical_v2"
    if "original" in normalized:
        return "original"
    return None


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def motion_from_text(text: str, motion_ids: list[str]) -> str | None:
    lowered = text.lower()
    for motion_id in motion_ids:
        if motion_id.lower() in lowered:
            return motion_id
    return None


def supporting_metadata_text(path: Path) -> str:
    chunks = [path.as_posix()]
    for metadata in path.parent.glob("*metadata.json"):
        try:
            chunks.append(read_text(metadata))
        except OSError:
            pass
    return "\n".join(chunks)


def load_visual_reports(data_root: Path, motion_ids: list[str]) -> dict[tuple[str, str | None], dict[str, Any]]:
    report_root = data_root / "llm" / "llm_visual_edit"
    index: dict[tuple[str, str | None], dict[str, Any]] = {}
    if not report_root.exists():
        return index

    patterns = ("*visual_qualitative_report_decoded.txt", "*visual_qualitative_report.txt")
    report_paths: list[Path] = []
    for pattern in patterns:
        report_paths.extend(report_root.rglob(pattern))
    report_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for path in report_paths:
        context = supporting_metadata_text(path)
        motion_id = motion_from_text(context, motion_ids)
        if motion_id is None:
            continue
        version = infer_version(context)
        key = (motion_id, version)
        if key not in index:
            index[key] = {
                "text": read_text(path).strip(),
                "source": relative_posix(path, data_root),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
    return index


def load_edit_plans(data_root: Path, motion_ids: list[str]) -> dict[tuple[str, str | None], dict[str, Any]]:
    plan_root = data_root / "llm" / "llm_edits"
    index: dict[tuple[str, str | None], dict[str, Any]] = {}
    if not plan_root.exists():
        return index

    paths = list(plan_root.rglob("*edit_plan_decoded.json"))
    paths.extend(plan_root.rglob("*edit_plan.json"))
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for path in paths:
        try:
            plan = json.loads(read_text(path))
        except (json.JSONDecodeError, OSError):
            continue
        context = path.as_posix() + "\n" + json.dumps(plan, ensure_ascii=False)
        version = infer_version(context) or "canonical_v2"
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        source = relative_posix(path, data_root)

        batch_motions = plan.get("motions")
        if isinstance(batch_motions, list):
            for motion_plan in batch_motions:
                if not isinstance(motion_plan, dict):
                    continue
                motion_id = str(motion_plan.get("motion_id", ""))
                if motion_id not in motion_ids:
                    continue
                single_plan = {
                    "motion_id": motion_id,
                    "summary": plan.get("global_summary", {}),
                    **motion_plan,
                }
                key = (motion_id, version)
                if key not in index:
                    index[key] = {
                        "data": single_plan,
                        "source": source,
                        "updated_at": updated_at,
                    }
            continue

        motion_id = str(plan.get("motion_id") or motion_from_text(context, motion_ids) or "")
        if motion_id in motion_ids:
            key = (motion_id, version)
            if key not in index:
                index[key] = {
                    "data": plan,
                    "source": source,
                    "updated_at": updated_at,
                }
    return index


def lookup_llm_item(
    index: dict[tuple[str, str | None], dict[str, Any]],
    motion_id: str,
    version: str,
) -> dict[str, Any] | None:
    return index.get((motion_id, version)) or index.get((motion_id, None))


def discover_fromw1_version_specs(data_root: Path, motion_ids: list[str]) -> list[dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for source_spec in FROMW1_VERSIONS:
        registry[str(source_spec["id"])] = {
            "id": source_spec["id"],
            "label": source_spec["label"],
            "motion_ids": set(source_spec.get("motion_ids", motion_ids)),
            "reference_roots": set(source_spec["reference_roots"]),
            "execution_roots": set(source_spec["execution_roots"]),
        }

    scan_roots = (
        ("gifs/h1_reference_gifs", "reference_roots"),
        ("gifs/pkl_gifs", "reference_roots"),
        ("gifs/RoBoJuDo_H1_gifs", "execution_roots"),
        ("results/robojudo", "execution_roots"),
    )
    for relative_parent, role in scan_roots:
        parent = data_root / relative_parent
        if not parent.exists():
            continue
        for directory in (path for path in parent.iterdir() if path.is_dir()):
            version_id = infer_version(directory.name)
            if version_id is None:
                continue
            entry = registry.setdefault(
                version_id,
                {
                    "id": version_id,
                    "label": version_id.replace("_", " ").upper(),
                    "motion_ids": set(motion_ids),
                    "reference_roots": set(),
                    "execution_roots": set(),
                },
            )
            entry[role].add(relative_posix(directory, data_root))
            detected_motion = motion_from_text(directory.name, motion_ids)
            if version_id.startswith("llm_v") and detected_motion:
                if entry["motion_ids"] == set(motion_ids):
                    entry["motion_ids"] = set()
                entry["motion_ids"].add(detected_motion)

    def sort_key(spec: dict[str, Any]) -> tuple[int, int]:
        version_id = str(spec["id"])
        if version_id == "original":
            return 0, 0
        if version_id == "canonical_v2":
            return 1, 0
        match = re.fullmatch(r"llm_v(\d+)", version_id)
        if match:
            return 2, int(match.group(1))
        return 3, 0

    specs: list[dict[str, Any]] = []
    for entry in sorted(registry.values(), key=sort_key):
        specs.append(
            {
                **entry,
                "motion_ids": tuple(sorted(entry["motion_ids"])),
                "reference_roots": tuple(sorted(entry["reference_roots"])),
                "execution_roots": tuple(sorted(entry["execution_roots"])),
            }
        )
    return specs


def build_fromw1_versions(
    data_root: Path,
    motion_ids: list[str],
    visual_reports: dict[tuple[str, str | None], dict[str, Any]],
    edit_plans: dict[tuple[str, str | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    versions: list[dict[str, Any]] = []
    for spec in discover_fromw1_version_specs(data_root, motion_ids):
        allowed = set(spec.get("motion_ids", motion_ids))
        motions: list[dict[str, Any]] = []
        for motion_id in motion_ids:
            if motion_id not in allowed:
                continue
            source = find_source_video(data_root, motion_id)
            reference = find_best_asset(
                data_root,
                spec["reference_roots"],
                motion_id,
                extensions={".gif", ".mp4"},
                preferred_terms=("h1_reference", "h1"),
            )
            execution = find_best_asset(
                data_root,
                spec["execution_roots"],
                motion_id,
                extensions={".gif", ".mp4"},
                preferred_terms=("h2h", "execution", "robojudo"),
            )
            if not any((source, reference, execution)):
                continue
            version_id = str(spec["id"])
            motions.append(
                {
                    "id": motion_id,
                    "label": MOTION_LABELS.get(motion_id, motion_id),
                    "source": media_entry(source, data_root, "原始视频"),
                    "reference": media_entry(reference, data_root, "H1 参考动作"),
                    "execution": media_entry(execution, data_root, "实际执行"),
                    "visual_review": lookup_llm_item(visual_reports, motion_id, version_id),
                    "edit_plan": lookup_llm_item(edit_plans, motion_id, version_id),
                }
            )
        if motions:
            versions.append({"id": spec["id"], "label": spec["label"], "motions": motions})
    return versions


def exbody_candidates(data_root: Path, motion_id: str, *, execution: bool) -> Path | None:
    roots = (
        ("results/exbody", "gifs/exbody_execution", "gifs/exbody_results")
        if execution
        else ("gifs/exbody_reference", "gifs/exbody", "exbody_inputs")
    )
    preferred = ("execution", "result") if execution else ("reference", "retarget")
    return find_best_asset(
        data_root,
        roots,
        motion_id,
        extensions={".gif", ".mp4"},
        preferred_terms=preferred,
    )


def build_exbody_versions(
    data_root: Path,
    motion_ids: list[str],
    visual_reports: dict[tuple[str, str | None], dict[str, Any]],
    edit_plans: dict[tuple[str, str | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    motions: list[dict[str, Any]] = []
    for motion_id in motion_ids:
        source = find_source_video(data_root, motion_id)
        reference = exbody_candidates(data_root, motion_id, execution=False)
        execution = exbody_candidates(data_root, motion_id, execution=True)
        motions.append(
            {
                "id": motion_id,
                "label": MOTION_LABELS.get(motion_id, motion_id),
                "source": media_entry(source, data_root, "原始视频"),
                "reference": media_entry(reference, data_root, "H1 参考动作"),
                "execution": media_entry(execution, data_root, "实际执行"),
                "visual_review": lookup_llm_item(visual_reports, motion_id, "current"),
                "edit_plan": lookup_llm_item(edit_plans, motion_id, "current"),
            }
        )
    return [{"id": "current", "label": "Current", "motions": motions}]


def media_entry(path: Path | None, data_root: Path, label: str) -> dict[str, Any]:
    if path is None:
        return {"label": label, "available": False, "url": None, "path": None, "kind": None}
    suffix = path.suffix.lower()
    return {
        "label": label,
        "available": True,
        "url": media_url(path, data_root),
        "path": relative_posix(path, data_root),
        "kind": "video" if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"} else "image",
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def build_catalog(data_root: Path) -> dict[str, Any]:
    motion_ids = discover_motion_ids(data_root)
    visual_reports = load_visual_reports(data_root, motion_ids)
    edit_plans = load_edit_plans(data_root, motion_ids)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "backends": [
            {
                "id": "fromw1",
                "label": "FRoM-W1",
                "versions": build_fromw1_versions(data_root, motion_ids, visual_reports, edit_plans),
            },
            {
                "id": "exbody",
                "label": "ExBody",
                "versions": build_exbody_versions(data_root, motion_ids, visual_reports, edit_plans),
            },
        ],
    }


def feedback_directory(data_root: Path, backend: str, motion_id: str, version: str) -> Path:
    safe = re.compile(r"^[A-Za-z0-9_.-]+$")
    for value in (backend, motion_id, version):
        if not safe.fullmatch(value):
            raise ValueError("Invalid feedback selector.")
    return data_root / "feedback" / backend / motion_id / version


def load_feedback_history(data_root: Path, backend: str, motion_id: str, version: str) -> list[dict[str, Any]]:
    directory = feedback_directory(data_root, backend, motion_id, version)
    if not directory.exists():
        return []
    history: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*_user_feedback.json"), reverse=True):
        try:
            item = json.loads(read_text(path))
        except (json.JSONDecodeError, OSError):
            continue
        item["file"] = relative_posix(path, data_root)
        history.append(item)
    return history[:20]


def find_catalog_motion(
    catalog: dict[str, Any],
    backend_id: str,
    version_id: str,
    motion_id: str,
) -> dict[str, Any] | None:
    for backend in catalog["backends"]:
        if backend["id"] != backend_id:
            continue
        for version in backend["versions"]:
            if version["id"] != version_id:
                continue
            for motion in version["motions"]:
                if motion["id"] == motion_id:
                    return motion
    return None


class MotionReviewServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], data_root: Path):
        self.data_root = data_root.resolve()
        self.catalog = build_catalog(self.data_root)
        super().__init__(server_address, MotionReviewHandler)

    def refresh_catalog(self) -> dict[str, Any]:
        self.catalog = build_catalog(self.data_root)
        return self.catalog


class MotionReviewHandler(BaseHTTPRequestHandler):
    server: MotionReviewServer

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stdout.write(
            f"[{self.log_date_time_string()}] {self.client_address[0]} {format_string % args}\n"
        )

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, relative_path: str) -> None:
        path = (STATIC_ROOT / relative_path).resolve()
        if not path_is_within(path, STATIC_ROOT.resolve()) or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, relative_path: str) -> None:
        path = (self.server.data_root / unquote(relative_path)).resolve()
        if not path_is_within(path, self.server.data_root) or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
                end = min(int(match.group(2)), end)
            if start > end or start >= file_size:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with path.open("rb") as file_handle:
            file_handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = file_handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static("index.html")
            return
        if parsed.path.startswith("/static/"):
            self.send_static(parsed.path.removeprefix("/static/"))
            return
        if parsed.path == "/api/catalog":
            query = parse_qs(parsed.query)
            catalog = self.server.refresh_catalog() if query.get("refresh") == ["1"] else self.server.catalog
            self.send_json(catalog)
            return
        if parsed.path == "/api/feedback":
            query = parse_qs(parsed.query)
            try:
                history = load_feedback_history(
                    self.server.data_root,
                    query.get("backend", [""])[0],
                    query.get("motion_id", [""])[0],
                    query.get("version", [""])[0],
                )
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"items": history})
            return
        if parsed.path.startswith("/media/"):
            self.send_media(parsed.path.removeprefix("/media/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/feedback":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 128 * 1024:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            backend = str(payload.get("backend", "")).strip()
            version = str(payload.get("version", "")).strip()
            motion_id = str(payload.get("motion_id", "")).strip()
            comment = str(payload.get("comment", "")).strip()
            if not comment:
                raise ValueError("用户建议不能为空。")
            if len(comment) > 4000:
                raise ValueError("用户建议不能超过 4000 个字符。")

            catalog_motion = find_catalog_motion(self.server.catalog, backend, version, motion_id)
            if catalog_motion is None:
                raise ValueError("当前动作、版本或后端不存在，请刷新页面后重试。")

            now = datetime.now()
            directory = feedback_directory(self.server.data_root, backend, motion_id, version)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{now.strftime('%Y%m%d_%H%M%S_%f')}_user_feedback.json"
            record = {
                "created_at": now.isoformat(timespec="seconds"),
                "backend": backend,
                "version": version,
                "motion_id": motion_id,
                "comment": comment,
                "assets": {
                    "source": catalog_motion["source"]["path"],
                    "h1_reference": catalog_motion["reference"]["path"],
                    "execution": catalog_motion["execution"]["path"],
                },
                "llm_context": {
                    "visual_review_source": (
                        catalog_motion["visual_review"]["source"]
                        if catalog_motion.get("visual_review")
                        else None
                    ),
                    "edit_plan_source": (
                        catalog_motion["edit_plan"]["source"]
                        if catalog_motion.get("edit_plan")
                        else None
                    ),
                },
                "status": "pending_llm_iteration",
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            record["file"] = relative_posix(path, self.server.data_root)
            self.send_json({"saved": record}, HTTPStatus.CREATED)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self.send_json({"error": f"保存反馈失败: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local motion iteration review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--open", action="store_true", help="Open the UI in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_dir.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    server = MotionReviewServer((args.host, args.port), data_root)
    url = f"http://{args.host}:{args.port}"
    print(f"Motion review UI: {url}")
    print(f"Data root: {data_root}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping motion review UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
