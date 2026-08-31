#!/usr/bin/env python3
"""
test_anomaly_dedupe.py — Unit test for the saved-anomaly dedupe cluster logic.

Covers the configurable "max samples per unique anomaly":
  - default (1) -> only one sample per anomaly event (the old behavior)
  - 0 -> treated as 1
  - N>1 -> keeps up to N samples from the same event, then stops; a distinct
    event is still saved while one event is capped
  - cluster membership is transitive (a slowly-drifting event stays one event)

Runs entirely offline: the model isn't loaded (load_anomalies is stubbed),
the anomaly store is written to a temp dir, and embeddings are synthetic.

Usage:
  python test_anomaly_dedupe.py
"""
import importlib.util
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _spec(dpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, dpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(v: np.ndarray) -> np.ndarray:
    return (np.asarray(v, dtype=np.float32) /
            np.linalg.norm(v)).astype(np.float32)


class DedupeHarness:
    """A tiny stand-in for the server's anomaly store: temp dir + bank."""

    def __init__(self, srv, tmpdir: Path):
        self.srv = srv
        srv.ANOMALIES_DIR = tmpdir            # offline store
        srv.load_anomalies = lambda: None     # never touch the real model
        srv.config["max_anomalies"] = 100     # never trim during these tests
        srv.config["anomaly_dedupe_threshold"] = 0.9
        srv.config["anomaly_dedupe_max_per_unique"] = 1
        self.rows = []                        # embeddings saved so far

    def set_quota(self, n):
        self.srv.config["anomaly_dedupe_max_per_unique"] = n

    def set_threshold(self, t):
        self.srv.config["anomaly_dedupe_threshold"] = t

    def save(self, emb: np.ndarray) -> bool:
        """Mirror what the server does: refresh bank, try to save, track."""
        self.srv.anomaly_bank = (np.stack(self.rows).astype(np.float32)
                                 if self.rows else None)
        img = Image.new("RGB", (16, 16), "red")
        saved = self.srv.maybe_save_anomaly(img, emb)
        if saved:
            self.rows.append(np.asarray(emb, dtype=np.float32))
        return saved


def main():
    srv = _spec("server.py", "srv")
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
        if not ok:
            failures.append(name)

    # ---- cluster size is transitive ------------------------------------
    v = _norm([1, 0, 0])     # drift chain at equal 41.4° steps:
    m = _norm([0.75, np.sqrt(1 - 0.75 ** 2), 0])   # cos(v,m) = 0.75
    w = _norm([0.125, np.sqrt(1 - 0.125 ** 2), 0])  # cos(m,w) = 0.75, cos(v,w) = 0.125
    other = _norm([0, 0, 1])
    bank = np.stack([v, m, w, other]).astype(np.float32)
    check("cluster transitivity (v)", srv._cluster_size_around(bank, 0, 0.6), 3)
    check("cluster transitivity (w)", srv._cluster_size_around(bank, 2, 0.6), 3)
    check("unrelated vector is its own cluster",
          srv._cluster_size_around(bank, 3, 0.6), 1)

    # ---- default quota = 1: one sample per event ------------------------
    with tempfile.TemporaryDirectory() as td:
        h = DedupeHarness(srv, Path(td))
        check("quota=1: first save of event", h.save(v), True)
        check("quota=1: duplicate event rejected", h.save(v), False)
        check("quota=1: store has exactly one item", len(h.rows), 1)

    # ---- quota = 0 means 1 ----------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        h = DedupeHarness(srv, Path(td))
        h.set_quota(0)
        check("quota=0: first save of event", h.save(v), True)
        check("quota=0: duplicate event rejected (=1)", h.save(v), False)

    # ---- quota = 3: keeps 3 per event, per-event independently -------------
    with tempfile.TemporaryDirectory() as td:
        h = DedupeHarness(srv, Path(td))
        h.set_quota(3)
        check("quota=3: save #1 of event", h.save(v), True)
        check("quota=3: save #2 of event", h.save(v), True)
        check("quota=3: save #3 of event", h.save(v), True)
        check("quota=3: save #4 of event rejected", h.save(v), False)
        # A different unique event gets its OWN quota, independent of v's.
        check("quota=3: distinct event saved #1", h.save(other), True)
        check("quota=3: distinct event saved #2", h.save(other), True)
        check("quota=3: distinct event saved #3", h.save(other), True)
        check("quota=3: distinct event #4 rejected", h.save(other), False)
        check("quota=3: two events -> 6 items", len(h.rows), 6)

    # ---- quota applies to the whole transitive cluster ------------------
    # v -> m -> w drift chain: cos(v,m)=0.75, cos(m,w)=0.75, cos(v,w)=0.125,
    # so at threshold 0.6 all three are ONE event (transitively), while
    # each is a distinct event at threshold 0.9.
    with tempfile.TemporaryDirectory() as td:
        h = DedupeHarness(srv, Path(td))
        h.set_quota(2)
        h.set_threshold(0.6)
        check("cluster quota: drift sample v", h.save(v), True)
        check("cluster quota: drift sample m", h.save(m), True)
        check("cluster quota: drifted member w rejected (cluster full)",
              h.save(w), False)

    with tempfile.TemporaryDirectory() as td:
        h = DedupeHarness(srv, Path(td))
        h.set_quota(3)
        h.set_threshold(0.6)
        check("cluster quota 3: v", h.save(v), True)
        check("cluster quota 3: m", h.save(m), True)
        check("cluster quota 3: w (transitively same event)",
              h.save(w), True)
        check("cluster quota 3: another w rejected (3 reached)",
              h.save(w), False)

    # ---- same chain at a HIGH threshold = three separate unique events ---
    with tempfile.TemporaryDirectory() as td:
        h = DedupeHarness(srv, Path(td))
        h.set_quota(1)
        h.set_threshold(0.9)
        check("high threshold: v is event A", h.save(v), True)
        check("high threshold: m is event B (not v's duplicate)",
              h.save(m), True)
        check("high threshold: w is event C", h.save(w), True)
        check("high threshold: another v rejected (A full)",
              h.save(v), False)

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        return 1
    print("All dedupe tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())