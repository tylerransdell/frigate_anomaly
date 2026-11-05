#!/usr/bin/env python3
"""
test_kick_start.py — Unit test for the engine's first-sample kick-start.

The detection loop used to wait a full sampling interval before its first look,
which is minutes (or an hour, when a duration filter does the real work). Now a
fresh engine run samples after FIRST_CYCLE_DELAY instead, then falls back to the
configured cadence.

Covers:
  - _sample_delay() returns the kick-start delay only on the first cycle of a
    non-duration run whose interval is comfortably slower than the kick-start
  - an already-fast interval keeps its own cadence (no kick-start)
  - the kick-start never exceeds the configured interval
  - mid-duration-window sampling is unchanged
  - the real detection_loop() actually sleeps FIRST_CYCLE_DELAY first and the
    configured interval afterwards (fake clock, stubbed model)

Usage:
  python test_kick_start.py
"""
import importlib.util

import numpy as np
from PIL import Image


def _spec(dpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, dpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTime:
    """Stand-in for the `time` module: records sleeps, never actually sleeps."""

    def __init__(self):
        self.sleeps = []
        self.on_sleep = None

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.on_sleep:
            self.on_sleep(seconds)

    def time(self):
        return 1000.0

    def perf_counter(self):
        return 1000.0


def main():
    srv = _spec("server.py", "srv")
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got!r} want={want!r}")
        if not ok:
            failures.append(name)

    check("delay constant", srv.FIRST_CYCLE_DELAY, 5.0)
    check("kick-start threshold", srv.FIRST_CYCLE_MIN_INTERVAL, 20.0)

    # ---- _sample_delay() ------------------------------------------------
    d = srv._sample_delay
    check("600s interval, first cycle -> kick-start", d(600, False, 0, 3, True), (5.0, True))
    check("60s interval, first cycle -> kick-start", d(60, False, 0, 1, True), (5.0, True))
    check("20s interval, first cycle -> kick-start", d(20, False, 0, 1, True), (5.0, True))
    check("15s interval is already fast -> no kick-start",
          d(15, False, 0, 1, True), (15, False))
    check("5s interval -> no kick-start", d(5, False, 0, 1, True), (5, False))
    check("later cycles keep the interval", d(600, False, 0, 3, False), (600, False))
    check("mid-window: interval already fits, no kick-start",
          d(600, True, 3600, 3, True), (600, False))
    check("mid-window: speeds up to fit min_samples",
          d(600, True, 30, 10, False), (3.0, False))
    check("kick-start never exceeds the interval",
          d(20, False, 0, 1, True)[0] <= 20, True)

    # ---- the real loop, on a fake clock --------------------------------
    ft = FakeTime()
    srv.time = ft
    srv.config.update({
        "sampling_interval": 600, "duration_filter": 0, "min_samples": 3,
        "threshold": 0.93, "camera": "kickcam",
        "anomaly_dedupe_threshold": 0.87, "anomaly_dedupe_max_per_unique": 1,
    })
    srv._active_zone = lambda: {"x": 0, "y": 0, "w": 4, "h": 4,
                                "ref": {"w": 8, "h": 8}}
    srv.fetch_snapshot = lambda: Image.new("RGB", (8, 8), "gray")
    srv.get_embedding = lambda img: np.zeros(3, dtype=np.float32)
    srv.baseline_bank = np.ones((1, 3), dtype=np.float32)
    srv.evaluate_against_baselines = lambda *a, **k: (0.97, 90.0, False)
    srv.add_snippet = lambda *a, **k: None

    def run_loop(n_sleeps, duration_active=False):
        """Run detection_loop() until it has slept n_sleeps times."""
        ft.sleeps.clear()
        before = srv.perf_stats["detections"]
        srv.duration_active = duration_active
        srv.duration_anomaly_count = 0
        srv.duration_sample_count = 0
        srv.engine_running = True

        def on_sleep(_seconds):
            if len(ft.sleeps) >= n_sleeps:
                srv.engine_running = False

        ft.on_sleep = on_sleep
        srv.detection_loop()
        return list(ft.sleeps), srv.perf_stats["detections"] - before

    # Slow interval: first sample fast, then the configured cadence.
    sleeps, detections = run_loop(2)
    check("loop: first wait is the kick-start", sleeps[0], 5.0)
    check("loop: second wait is the interval", sleeps[1], 600.0)
    check("loop: a detection ran after the kick-start", detections, 1)

    # Fast interval: unchanged behaviour (no kick-start).
    srv.config["sampling_interval"] = 10
    sleeps, _ = run_loop(2)
    check("loop: fast interval keeps its own cadence", sleeps, [10.0, 10.0])

    # Long duration window already running: keep the window's cadence.
    srv.config["sampling_interval"] = 600
    srv.config["duration_filter"] = 3600
    sleeps, _ = run_loop(2, duration_active=True)
    check("loop: mid-window first wait stays the interval", sleeps[0], 600.0)

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        return 1
    print("All kick-start tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
