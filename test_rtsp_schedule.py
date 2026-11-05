#!/usr/bin/env python3
"""
test_rtsp_schedule.py — Unit test for direct-RTSP frame grabbing + the
"next 24/48 images" baseline capture plan.

Covers:
  - grab_rtsp_frame() failure paths (empty URL, unreachable stream) return None
    instead of raising
  - start_rtsp_schedule() refuses when the source isn't RTSP, the URL/zone is
    missing, the interval is wrong, or the baseline store has no room
  - a running plan grabs its first frame immediately, persists its progress,
    and stops on request
  - resume_rtsp_schedule() finishes a plan left over from a restart

Runs offline: the model is never loaded (frame capture + baseline saving are
stubbed) and all state lives in a temp dir.

Usage:
  python test_rtsp_schedule.py
"""
import importlib.util
import json
import tempfile
import time
from pathlib import Path

from PIL import Image


def _spec(dpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, dpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Harness:
    """Point the server at a temp dir and stub the camera + baseline store."""

    def __init__(self, srv, tmpdir: Path):
        self.srv = srv
        self.saved = []
        self.grabs = 0
        tmpdir.mkdir(parents=True, exist_ok=True)
        srv.RTSP_SCHEDULE_PATH = tmpdir / "rtsp_schedule.json"
        srv.BASELINE_DIR = tmpdir
        # Frame capture always "works" and returns a tiny image.
        self._img = Image.new("RGB", (8, 8), "blue")

        def fake_fetch():
            self.grabs += 1
            return self._img

        def fake_add(img):
            self.saved.append(img)
            return tmpdir / f"baseline_{len(self.saved):02d}.jpg"

        srv.fetch_snapshot = fake_fetch
        srv.add_baseline_image = fake_add
        srv._baseline_files = lambda: [Path(p) for p in self.saved]

        srv.config["source_type"] = "rtsp"
        srv.config["rtsp_url"] = "rtsp://user:secret@10.0.0.9:554/s1"
        srv.config["camera"] = "front"
        srv.config["zones"] = {
            "front": {"x": 0, "y": 0, "w": 4, "h": 4,
                      "ref": {"w": 8, "h": 8}},
        }
        srv.MAX_BASELINES = 64
        self._reset()

    def _reset(self):
        self.saved.clear()
        self.grabs = 0
        self.srv.rtsp_schedule_stop.clear()
        self.srv.rtsp_schedule_state.update({
            "active": False, "stopped": False, "minutes": 60, "total": 0,
            "captured": 0, "failed": 0, "started_at": None, "next_at": None,
            "last_error": None,
        })

    def wait_for(self, pred, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.02)
        return False

    def state(self):
        return self.srv.rtsp_schedule_status()

    def persisted(self):
        with open(self.srv.RTSP_SCHEDULE_PATH) as f:
            return json.load(f)


def main():
    srv = _spec("server.py", "srv")
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got!r} want={want!r}")
        if not ok:
            failures.append(name)

    def check_true(name, cond, detail=""):
        ok = bool(cond)
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else ' — ' + detail}")
        if not ok:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        # ---- frame grabbing fails cleanly --------------------------------
        check("grab: empty url", srv.grab_rtsp_frame(""), None)
        check("grab: unreachable stream",
              srv.grab_rtsp_frame("rtsp://127.0.0.1:1/nope"), None)
        check("redact: hides credentials",
              srv._redact_url("rtsp://admin:hunter2@10.0.0.9:554/s1"),
              "rtsp://***@10.0.0.9:554/s1")

        h = _Harness(srv, Path(td))

        # ---- refusals ----------------------------------------------------
        srv.config["source_type"] = "frigate"
        check("refuse: frigate source", srv.start_rtsp_schedule(60).get("ok"), False)
        srv.config["source_type"] = "rtsp"

        srv.config["rtsp_url"] = ""
        check("refuse: no rtsp url", srv.start_rtsp_schedule(60).get("ok"), False)

        srv.config["rtsp_url"] = "rtsp://user:secret@10.0.0.9:554/s1"
        srv.config["camera"] = ""
        check("refuse: no camera", srv.start_rtsp_schedule(60).get("ok"), False)

        srv.config["camera"] = "front"
        srv.config["zones"] = {}
        check("refuse: no zone", srv.start_rtsp_schedule(60).get("ok"), False)

        srv.config["zones"] = {"front": {"x": 0, "y": 0, "w": 4, "h": 4,
                                         "ref": {"w": 8, "h": 8}}}
        check("refuse: bad interval", srv.start_rtsp_schedule(15).get("ok"), False)

        srv.MAX_BASELINES = 10
        check("refuse: no room for 24", srv.start_rtsp_schedule(60).get("ok"), False)
        srv.MAX_BASELINES = 64

        # ---- a plan runs, reports progress, and stops --------------------
        start = srv.start_rtsp_schedule(60)
        check("start: accepted", start.get("ok"), True)
        check("start: 24 images for hourly", start.get("total"), 24)
        check_true("start: first frame captured immediately",
                   h.wait_for(lambda: len(h.saved) == 1), f"saved={len(h.saved)}")
        check_true("start: progress is persisted",
                   h.wait_for(lambda: h.persisted().get("captured") == 1))
        st = h.state()
        check("start: plan is active", st["active"], True)
        check("start: next slot scheduled", bool(st["next_at"]), True)

        stop = srv.stop_rtsp_schedule()
        check("stop: reports it was active", stop.get("stopped"), True)
        check_true("stop: loop exits",
                   h.wait_for(lambda: h.state()["active"] is False))
        check_true("stop: persisted as stopped",
                   h.persisted().get("stopped") is True)
        check("stop: nothing extra captured", len(h.saved), 1)

        # ---- resume an unfinished plan after a restart -------------------
        h._reset()
        h.saved.append("already saved")       # pretend 1 image landed pre-restart
        srv.rtsp_schedule_state.update({
            "active": True, "stopped": False, "minutes": 30,
            "total": 2, "captured": 1, "failed": 0,
            "started_at": "2026-01-01T00:00:00",
            "next_at": "2026-01-01T00:30:00",    # long overdue -> grab now
            "last_error": None,
        })
        srv._persist_rtsp_schedule()
        srv.rtsp_schedule_state["active"] = False   # in-memory state is clean
        srv.rtsp_schedule_state["total"] = 0

        srv.resume_rtsp_schedule()
        check_true("resume: remaining frame captured",
                   h.wait_for(lambda: h.state()["captured"] == 2),
                   f"state={h.state()}")
        check_true("resume: plan finishes and deactivates",
                   h.wait_for(lambda: h.state()["active"] is False))
        check("resume: total reached", h.state()["remaining"], 0)

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        return 1
    print("All RTSP schedule tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
