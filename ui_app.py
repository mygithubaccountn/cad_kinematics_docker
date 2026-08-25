#!/usr/bin/env python3
"""Mini local UI to pick a STEP, run the pipeline, and preview in the viewer.

  ./start_ui.sh
  → http://127.0.0.1:8787/
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
VIEWER_DIR = ROOT / "viewer"
UPLOAD_DIR = ROOT / "uploads"
OUT_ROOT = ROOT / "out"
STATE_PATH = ROOT / "out" / "_ui_state.json"
LOG_PATH = ROOT / "out" / "_ui_run.log"

PORT = int(os.environ.get("UI_PORT", "8787"))
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "8765"))
# Right after a run starts, there's a narrow window where _proc (this
# process's own subprocess handle) may not be assigned yet — the background
# thread that does `_proc = subprocess.Popen(...)` hasn't been scheduled —
# and the child may still be inside run_with_freecad.sh's bash wrapper,
# before `exec` replaces it with the actual `pipeline.py` process, so pgrep
# won't find it either. A status poll landing in that window looks exactly
# like an already-dead orphan. Give a run this long to prove it started
# before orphan-finalization is allowed to declare it failed.
STARTUP_GRACE_SECONDS = 10.0
STAGE_ORDER = ("ingest", "features", "joints", "hierarchy", "package", "meshes", "validate")
DEFAULT_PROGRESS_STAGES = ("ingest", "features", "joints", "hierarchy", "package", "meshes", "validate")

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_viewer_proc: subprocess.Popen | None = None
_godot_proc: subprocess.Popen | None = None
GODOT_LOG_PATH = ROOT / "out" / "_ui_godot_launch.log"


def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "run"


def _read_state() -> dict:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"status": "idle", "log_path": str(LOG_PATH)}


def _write_state(data: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(data, indent=2))


def _append_log(line: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")
        f.flush()


def _subprocess_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _progress_from_manifest(out_dir: Path) -> dict:
    stages: dict = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            stages = (json.loads(manifest_path.read_text()) or {}).get("stages") or {}
        except Exception:
            pass
    # Prefer kinematics path for progress unless meshes already running/done.
    order = list(DEFAULT_PROGRESS_STAGES)
    if (stages.get("meshes") or {}).get("status") == "done" or (
        stages.get("meshes") or {}
    ).get("status") == "running":
        order = list(STAGE_ORDER)
    done: list[str] = []
    current = "starting"
    for stage in order:
        st = stages.get(stage) or {}
        if st.get("status") == "done":
            done.append(stage)
        elif current == "starting":
            current = stage
    if len(done) == len(order):
        current = "finished"
    return {
        "done": done,
        "current": current,
        "total": len(order),
        "done_count": len(done),
        # Raw per-stage manifest entries (status/meta/updated_at) so the UI can
        # show live detail (e.g. "5 joints selected") without re-deriving it.
        "stages": stages,
    }


def _perf_spans(out_dir: Path) -> list[dict]:
    """Read the pipeline's own performance_summary.json (already written by
    pipeline.py) for real per-stage timing — purely a read of an existing
    artifact, no timing logic lives in the UI."""
    p = out_dir / "performance_summary.json"
    if not p.is_file():
        return []
    try:
        return (json.loads(p.read_text()) or {}).get("spans") or []
    except Exception:
        return []


def _recorded_source(out_dir: Path) -> str | None:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            src = (json.loads(manifest_path.read_text()) or {}).get("source")
            if src:
                return str(src)
        except Exception:
            pass
    robot = out_dir / "robot.json"
    if robot.is_file():
        try:
            src = (json.loads(robot.read_text()).get("meta") or {}).get("source")
            if src:
                return str(src)
        except Exception:
            pass
    return None


def _resolve_out_dir(step: Path, name: str, *, allow_overwrite: bool) -> tuple[str, Path, str | None]:
    """Pick output folder; auto-suffix when reusing name for a different STEP."""
    out_dir = OUT_ROOT / name
    step_s = str(step.resolve())
    prev = _recorded_source(out_dir)
    if prev and prev != step_s and not allow_overwrite:
        suffix = time.strftime("%m%d_%H%M")
        name = _safe_name(f"{name}_{suffix}")
        out_dir = OUT_ROOT / name
        return name, out_dir, prev
    return name, out_dir, None


def _proc_running() -> bool:
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return True
    return False


def _orphan_pipeline_for(out_dir: str | None) -> bool:
    """True if a pipeline.py child is still running for this out folder (after UI restart)."""
    if not out_dir:
        return False
    try:
        r = subprocess.run(
            ["pgrep", "-lf", "pipeline.py"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        needle = f"--out {out_dir}"
        return needle in (r.stdout or "")
    except Exception:
        return False


def _maybe_finalize_orphan(st: dict) -> dict:
    """If pipeline finished while UI was detached, sync viewer and close state."""
    if st.get("status") != "running" or _proc_running():
        return st
    started_at = st.get("started_at")
    if started_at and time.time() - started_at < STARTUP_GRACE_SECONDS:
        return st
    out = st.get("out")
    if not out or _orphan_pipeline_for(out):
        return st
    out_dir = Path(out)
    prog = _progress_from_manifest(out_dir)
    if prog.get("current") != "finished" or not (out_dir / "robot.json").is_file():
        st["status"] = "error"
        st["error"] = "Pipeline durdu ama validate tamamlanmadı"
        st["progress"] = prog
        _write_state(st)
        return st
    try:
        _sync_viewer(out_dir)
        _ensure_viewer_server()
        summary = _summarize_out(out_dir)
        st = {
            **st,
            "status": "done",
            "ended_at": time.time(),
            "exit_code": 0,
            "summary": summary,
            "viewer_url": f"http://127.0.0.1:{VIEWER_PORT}/",
            "progress": prog,
            "orphan": False,
        }
        _write_state(st)
    except Exception as e:
        st["status"] = "error"
        st["error"] = str(e)
        _write_state(st)
    return st


def _live_status() -> dict:
    st = _read_state()
    running = _proc_running()
    out = st.get("out")
    if not running and st.get("status") == "running" and _orphan_pipeline_for(out):
        running = True
        st["orphan"] = True
    st["running"] = running
    if st.get("status") == "running":
        if running:
            if out:
                st["progress"] = _progress_from_manifest(Path(out))
        else:
            started_at = st.get("started_at")
            in_startup_grace = bool(
                started_at and time.time() - started_at < STARTUP_GRACE_SECONDS
            )
            if not in_startup_grace:
                st = _maybe_finalize_orphan(st)
                if st.get("status") == "running":
                    st["status"] = "stale"
                    st.setdefault("error", "Süreç kesildi veya UI yeniden başlatıldı")
    return st


def _list_steps() -> list[dict]:
    roots = [Path.home() / "Desktop", ROOT / "uploads", ROOT / "fixtures"]
    seen: set[str] = set()
    items: list[dict] = []
    for base in roots:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".step", ".stp"}:
                continue
            # skip deep noise / huge trees beyond 3 levels from Desktop
            try:
                rel = p.relative_to(base)
                if len(rel.parts) > 3:
                    continue
            except ValueError:
                pass
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "path": key,
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "folder": str(base),
                }
            )
    items.sort(key=lambda x: (-Path(x["path"]).stat().st_mtime, x["name"]))
    return items[:80]


def _sync_viewer(out_dir: Path) -> None:
    robot = out_dir / "robot.json"
    meshes = out_dir / "meshes"
    if not robot.is_file():
        raise FileNotFoundError(f"missing {robot}")
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    (VIEWER_DIR / "meshes").mkdir(parents=True, exist_ok=True)
    shutil.copy2(robot, VIEWER_DIR / "robot.json")
    dbg = out_dir / "debug_overlay.json"
    if dbg.is_file():
        shutil.copy2(dbg, VIEWER_DIR / "debug_overlay.json")
    for old in (VIEWER_DIR / "meshes").glob("*.glb"):
        old.unlink()
    if meshes.is_dir():
        for glb in meshes.glob("*.glb"):
            shutil.copy2(glb, VIEWER_DIR / "meshes" / glb.name)


def _list_runs() -> list[dict]:
    """Existing out/*/robot.json runs for one-click preview (no recompute)."""
    items: list[dict] = []
    if not OUT_ROOT.is_dir():
        return items
    for d in sorted(OUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        robot = d / "robot.json"
        if not robot.is_file():
            continue
        meta: dict = {"name": d.name, "out": str(d), "mtime": d.stat().st_mtime}
        manifest_src = _recorded_source(d)
        if manifest_src:
            meta["source"] = manifest_src
        try:
            r = json.loads(robot.read_text())
            meta["links"] = len(r.get("links", []))
            meta["joints"] = len(r.get("joints", []))
            if not meta.get("source"):
                meta["source"] = (r.get("meta") or {}).get("source")
        except Exception:
            pass
        val = d / "validation_report.json"
        if val.is_file():
            try:
                v = json.loads(val.read_text())
                meta["validation_ok"] = v.get("ok")
                meta["confidence"] = v.get("overall_confidence")
                meta["warnings"] = len(v.get("warnings") or [])
            except Exception:
                pass
        items.append(meta)
    return items[:40]


def _summarize_out(out_dir: Path) -> dict:
    summary: dict = {}
    try:
        robot = json.loads((out_dir / "robot.json").read_text())
        summary["links"] = len(robot.get("links", []))
        summary["joints"] = len(robot.get("joints", []))
        summary["movable"] = sum(
            1 for j in robot.get("joints", []) if j.get("type") != "fixed"
        )
        summary["frame"] = robot.get("frame")
        summary["base"] = robot.get("base_link")
        summary["source"] = (robot.get("meta") or {}).get("source")
    except Exception:
        pass
    try:
        val = json.loads((out_dir / "validation_report.json").read_text())
        summary["validation_ok"] = val.get("ok")
        summary["confidence"] = val.get("overall_confidence")
        summary["warnings"] = list(val.get("warnings") or [])
        m = val.get("metrics") or {}
        summary["fk_ok"] = m.get("fk_motion_all_ok")
        summary["chain_errors"] = m.get("chain_sanity_errors")
        summary["godot_ok"] = m.get("godot_runtime_ok")
    except Exception:
        pass
    summary["perf"] = _perf_spans(out_dir)
    return summary


def _ensure_viewer_server() -> None:
    global _viewer_proc
    # If something already serves viewer port, leave it
    try:
        import urllib.request

        urllib.request.urlopen(f"http://127.0.0.1:{VIEWER_PORT}/robot.json", timeout=0.4)
        return
    except Exception:
        pass
    if _viewer_proc and _viewer_proc.poll() is None:
        return
    _viewer_proc = subprocess.Popen(
        [sys_executable(), "-m", "http.server", "--bind", "0.0.0.0", str(VIEWER_PORT)],
        cwd=str(VIEWER_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sys_executable() -> str:
    return os.environ.get("UI_PYTHON", "python3")


def _launch_godot(out_dir: Path) -> None:
    """Fire-and-forget: hand off to the existing run_godot_test.sh, which
    already does sync + asset-import + launch. The UI never talks to Godot
    or the pipeline directly, only this one script."""
    global _godot_proc
    with _lock:
        if _godot_proc and _godot_proc.poll() is None:
            raise RuntimeError("Godot zaten açılıyor/açık")
        GODOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = GODOT_LOG_PATH.open("w", encoding="utf-8")
        _godot_proc = subprocess.Popen(
            [str(ROOT / "run_godot_test.sh"), str(out_dir)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_subprocess_env(),
        )


def _run_pipeline(
    step: Path,
    name: str,
    out_dir: Path,
    run_id: str,
    *,
    from_stage: str | None = None,
    remesh: bool = False,
    force: bool = False,
    final_meshes: bool = False,
) -> None:
    global _proc
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    _write_state(
        {
            "status": "running",
            "run_id": run_id,
            "step": str(step),
            "name": name,
            "out": str(out_dir),
            "started_at": time.time(),
            "viewer_url": f"http://127.0.0.1:{VIEWER_PORT}/",
            "log_path": str(LOG_PATH),
            "from_stage": from_stage,
            "remesh": remesh,
            "force": force,
            "final_meshes": final_meshes,
            "preview_only": False,
            "summary": None,
            "progress": _progress_from_manifest(out_dir),
        }
    )
    _append_log(f"=== run_id {run_id}")
    _append_log(f"=== STEP {step}")
    _append_log(f"=== out  {out_dir}")
    _append_log(f"=== name {name}")
    stop_progress = threading.Event()

    def _progress_loop() -> None:
        while not stop_progress.wait(2.0):
            cur = _read_state()
            if cur.get("run_id") != run_id or cur.get("status") != "running":
                break
            cur["progress"] = _progress_from_manifest(out_dir)
            _write_state(cur)

    progress_thread = threading.Thread(target=_progress_loop, daemon=True)
    progress_thread.start()
    cmd = [str(ROOT / "run_with_freecad.sh"), "run", str(step), "--out", str(out_dir), "--name", name]
    if from_stage:
        cmd.extend(["--from-stage", from_stage])
        _append_log(f"=== from-stage {from_stage}")
    if final_meshes:
        cmd.append("--final-meshes")
        _append_log("=== final-meshes")
    if remesh:
        cmd.append("--remesh")
        _append_log("=== remesh (force tessellate)")
    else:
        _append_log("=== mesh cache (skip if topology fresh)")
    if force:
        cmd.append("--force")
        _append_log("=== force")
    try:
        _proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_subprocess_env(),
            # Heavy STEP imports can run 10+ minutes. Without this, the child
            # shares this process's session/process group — if the terminal
            # that launched ui_app.py closes (or this process dies for any
            # other reason), the OS can SIGHUP/kill the whole group, taking
            # a nearly-finished import down with it. start_new_session
            # detaches the child so it survives independently; the existing
            # orphan-recovery logic (_orphan_pipeline_for / pgrep) already
            # finds and reconciles with it on the next status poll or UI
            # restart regardless of session, so nothing else needs to change.
            start_new_session=True,
        )
        assert _proc.stdout is not None
        for line in _proc.stdout:
            _append_log(line.rstrip("\n"))
        code = _proc.wait()
        stop_progress.set()
        progress_thread.join(timeout=1.0)
        if code != 0:
            _write_state(
                {
                    **_read_state(),
                    "run_id": run_id,
                    "status": "error",
                    "exit_code": code,
                    "ended_at": time.time(),
                    "progress": _progress_from_manifest(out_dir),
                }
            )
            _append_log(f"=== FAILED exit={code}")
            return
        _sync_viewer(out_dir)
        _ensure_viewer_server()
        summary = _summarize_out(out_dir)
        _append_log(
            f"=== OK links={summary.get('links')} joints={summary.get('joints')} "
            f"validation={summary.get('validation_ok')} conf={summary.get('confidence')}"
        )
        _write_state(
            {
                "run_id": run_id,
                "status": "done",
                "step": str(step),
                "name": name,
                "out": str(out_dir),
                "started_at": _read_state().get("started_at"),
                "ended_at": time.time(),
                "exit_code": 0,
                "summary": summary,
                "viewer_url": f"http://127.0.0.1:{VIEWER_PORT}/",
                "log_path": str(LOG_PATH),
                "from_stage": from_stage,
                "remesh": remesh,
                "force": force,
                "final_meshes": final_meshes,
                "preview_only": False,
                "progress": _progress_from_manifest(out_dir),
            }
        )
    except Exception as e:
        stop_progress.set()
        _append_log(f"=== ERROR {e}")
        _write_state(
            {
                **_read_state(),
                "run_id": run_id,
                "status": "error",
                "error": str(e),
                "ended_at": time.time(),
            }
        )
    finally:
        stop_progress.set()
        with _lock:
            _proc = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # quieter
        if args and str(args[0]).startswith("GET /api"):
            return
        super().log_message(fmt, *args)

    def _json(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/steps":
            self._json(200, {"steps": _list_steps()})
            return
        if parsed.path == "/api/runs":
            self._json(200, {"runs": _list_runs()})
            return
        if parsed.path == "/api/status":
            self._json(200, _live_status())
            return
        if parsed.path == "/api/log":
            qs = urllib.parse.parse_qs(parsed.query)
            offset = int((qs.get("offset") or ["0"])[0])
            text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.is_file() else ""
            chunk = text[offset:]
            self._json(200, {"offset": offset + len(chunk), "text": chunk, "total": len(text)})
            return
        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/upload":
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self._json(400, {"error": "multipart required"})
                return
            raw = _read_multipart_file(self)
            if raw is None:
                self._json(400, {"error": "file missing"})
                return
            filename, data = raw
            if Path(filename).suffix.lower() not in {".step", ".stp"}:
                self._json(400, {"error": "only .step / .stp"})
                return
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / Path(filename).name
            dest.write_bytes(data)
            self._json(200, {"path": str(dest.resolve()), "name": dest.name})
            return

        if parsed.path == "/api/run":
            try:
                payload = json.loads(self._read_body().decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "bad json"})
                return
            with _lock:
                if _proc is not None and _proc.poll() is None:
                    self._json(409, {"error": "already running"})
                    return
            st = _read_state()
            if st.get("status") == "running" and _orphan_pipeline_for(st.get("out")):
                self._json(409, {"error": f"Zaten çalışıyor: {st.get('out')}"})
                return
            step = Path(payload.get("path", "")).expanduser()
            if not step.is_file():
                self._json(400, {"error": f"STEP not found: {step}"})
                return
            name = _safe_name(str(payload.get("name") or step.stem))
            allow_overwrite = bool(payload.get("allow_overwrite"))
            name, out_dir, prev_source = _resolve_out_dir(step.resolve(), name, allow_overwrite=allow_overwrite)
            from_stage = payload.get("from_stage") or None
            if from_stage in ("", "full", None):
                from_stage = None
            remesh = bool(payload.get("remesh"))
            force = bool(payload.get("force"))
            final_meshes = bool(payload.get("final_meshes"))
            run_id = uuid.uuid4().hex[:12]
            t = threading.Thread(
                target=_run_pipeline,
                args=(step.resolve(), name, out_dir, run_id),
                kwargs={
                    "from_stage": from_stage,
                    "remesh": remesh,
                    "force": force,
                    "final_meshes": final_meshes,
                },
                daemon=True,
            )
            t.start()
            resp: dict = {"ok": True, "name": name, "out": str(out_dir), "run_id": run_id}
            if prev_source:
                resp["renamed"] = True
                resp["previous_source"] = prev_source
                resp["note"] = f"Farklı STEP için yeni klasör: {name}"
            self._json(200, resp)
            return

        if parsed.path == "/api/preview":
            try:
                payload = json.loads(self._read_body().decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "bad json"})
                return
            if _proc_running() or _orphan_pipeline_for(_read_state().get("out")):
                self._json(409, {"error": "Pipeline çalışırken preview yapılamaz — bitmesini bekle"})
                return
            out_dir = Path(payload.get("out", "")).expanduser()
            if not out_dir.is_dir() or not (out_dir / "robot.json").is_file():
                self._json(400, {"error": f"run not found: {out_dir}"})
                return
            try:
                _sync_viewer(out_dir)
                _ensure_viewer_server()
                summary = _summarize_out(out_dir)
                manifest_src = _recorded_source(out_dir)
                if manifest_src:
                    summary["source"] = manifest_src
                _write_state(
                    {
                        "status": "done",
                        "run_id": f"preview-{uuid.uuid4().hex[:8]}",
                        "out": str(out_dir),
                        "name": out_dir.name,
                        "step": summary.get("source"),
                        "summary": summary,
                        "viewer_url": f"http://127.0.0.1:{VIEWER_PORT}/",
                        "preview_only": True,
                    }
                )
                self._json(
                    200,
                    {
                        "ok": True,
                        "url": f"http://127.0.0.1:{VIEWER_PORT}/",
                        "summary": summary,
                    },
                )
            except Exception as e:
                self._json(500, {"error": str(e)})
            return

        if parsed.path == "/api/open_viewer":
            try:
                _ensure_viewer_server()
                self._json(200, {"url": f"http://127.0.0.1:{VIEWER_PORT}/"})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return

        if parsed.path == "/api/open_godot":
            try:
                payload = json.loads(self._read_body().decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "bad json"})
                return
            out_dir = Path(payload.get("out", "")).expanduser()
            if not out_dir.is_dir() or not (out_dir / "robot.json").is_file():
                self._json(400, {"error": f"run not found: {out_dir}"})
                return
            try:
                _launch_godot(out_dir)
                self._json(200, {"ok": True})
            except Exception as e:
                self._json(409, {"error": str(e)})
            return

        self._json(404, {"error": "not found"})


def _read_multipart_file(handler: Handler) -> tuple[str, bytes] | None:
    """Minimal multipart parser for a single file field named 'file'."""
    ctype = handler.headers.get("Content-Type", "")
    m = re.search(r"boundary=(.+)", ctype)
    if not m:
        return None
    boundary = m.group(1).strip().encode()
    body = handler._read_body()
    parts = body.split(b"--" + boundary)
    for part in parts:
        if b"Content-Disposition" not in part or b'name="file"' not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        fm = re.search(br'filename="([^"]+)"', header)
        if not fm:
            continue
        name = fm.group(1).decode("utf-8", errors="replace")
        return name, data
    return None


def main() -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_viewer_server()
    bind_host = os.environ.get("UI_BIND", "127.0.0.1")
    server = ThreadingHTTPServer((bind_host, PORT), Handler)
    print(f"CAD Robot Test UI → http://127.0.0.1:{PORT}/")
    print(f"3D Viewer         → http://127.0.0.1:{VIEWER_PORT}/")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
        if _viewer_proc and _viewer_proc.poll() is None:
            _viewer_proc.terminate()


if __name__ == "__main__":
    main()
