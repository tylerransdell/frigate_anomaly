#!/usr/bin/env python3
"""
test_anomaly_meta.py — Unit test for anomaly metadata + MQTT alert cooldown.

Covers:
  - saving an anomaly records its capture time and the ORIGINAL similarity it
    scored against the baselines at capture time
  - that original score never changes afterwards (it is not recomputed when the
    baseline library or the anomaly store changes)
  - removing / trimming an anomaly drops its metadata
  - images that predate the metadata index get their timestamp backfilled and
    original_sim = None (unknown, not invented)
  - publish_alert() sends one alert then stays quiet for mqtt_alert_cooldown
    seconds; 0 disables the throttle

Runs entirely offline: the model is never loaded (load_anomalies is stubbed),
the anomaly store is a temp dir, and MQTT is a fake client.

Usage:
  python test_anomaly_meta.py
"""
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _spec(dpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, dpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeMqtt:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=1):
        self.published.append((topic, payload))


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

    def index_of(path):
        p = Path(path) / "index.json"
        return json.loads(p.read_text()) if p.exists() else {}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        srv.ANOMALIES_DIR = tmp
        srv.load_anomalies = lambda: None      # never touch the real model
        srv.anomaly_bank = None
        srv.config["max_anomalies"] = 128
        srv.config["anomaly_dedupe_threshold"] = 0.87
        srv.config["anomaly_dedupe_max_per_unique"] = 1
        srv.config["camera"] = "attic"

        img = Image.new("RGB", (16, 16), "red")
        emb = _norm([1, 0, 0])

        # ---- capture stamps time + permanent original score --------------
        saved = srv.maybe_save_anomaly(img, emb, similarity=0.812345, threshold=0.93)
        check("save: anomaly stored", saved, True)
        meta = index_of(td).get("anomaly_001.jpg") or {}
        check("save: original_sim recorded", meta.get("original_sim"), 0.8123)
        check("save: threshold recorded", meta.get("threshold"), 0.93)
        check("save: camera recorded", meta.get("camera"), "attic")
        check_true("save: created_at recorded", bool(meta.get("created_at")))
        check_true("save: created_at is an ISO timestamp",
                   "T" in (meta.get("created_at") or ""))

        first_stamp = meta.get("created_at")

        # ---- the original score is permanent ----------------------------
        emb2 = _norm([0, 1, 0])
        srv.maybe_save_anomaly(img, emb2, similarity=0.5555, threshold=0.93)
        idx = index_of(td)
        check("permanent: second anomaly recorded",
              idx.get("anomaly_002.jpg", {}).get("original_sim"), 0.5555)
        check("permanent: first anomaly unchanged",
              idx.get("anomaly_001.jpg", {}).get("original_sim"), 0.8123)
        check("permanent: first timestamp unchanged",
              idx.get("anomaly_001.jpg", {}).get("created_at"), first_stamp)

        # ---- dashboard payload exposes both similarities ----------------
        srv.anomaly_bank = np.stack([emb, emb2]).astype(np.float32)
        items = srv.anomaly_metrics()
        check("api: two anomalies listed", len(items), 2)
        check("api: original_sim present", items[0].get("original_sim"), 0.8123)
        check_true("api: store_sim present",
                   items[0].get("store_sim") is not None)
        check_true("api: created_at present", bool(items[0].get("created_at")))

        # ---- removing an anomaly drops its metadata ---------------------
        check("remove: anomaly deleted", srv.remove_anomaly_by_index(1), True)
        idx = index_of(td)
        check_true("remove: metadata dropped",
                   "anomaly_001.jpg" not in idx, f"index={idx}")
        check_true("remove: other metadata kept", "anomaly_002.jpg" in idx)

        # ---- trimming drops the oldest metadata too ---------------------
        srv.config["max_anomalies"] = 1
        srv.maybe_save_anomaly(img, _norm([0, 0, 1]), similarity=0.4, threshold=0.93)
        idx = index_of(td)
        check("trim: only the newest metadata survives", len(idx), 1)
        check_true("trim: newest entry kept", "anomaly_003.jpg" in idx, f"index={idx}")

        # ---- legacy image without metadata ------------------------------
        path = tmp / "anomaly_009.jpg"
        Image.new("RGB", (8, 8), "green").save(path)
        srv.config["max_anomalies"] = 128
        idx = srv.load_anomaly_index()
        legacy = idx.get("anomaly_009.jpg") or {}
        check("legacy: original_sim unknown", legacy.get("original_sim"), None)
        check_true("legacy: timestamp backfilled from mtime",
                   bool(legacy.get("created_at")))

    # ---- MQTT alert cooldown -------------------------------------------
    fake = FakeMqtt()
    srv.mqtt_client = fake
    srv.config["mqtt_topic"] = "frigate/anomaly/alert"
    srv.config["mqtt_alert_cooldown"] = 60
    srv._last_alert_at = 0.0

    check("alert: first publishes", srv.publish_alert({"n": 1}), True)
    check("alert: second suppressed", srv.publish_alert({"n": 2}), False)
    check("alert: only one publish reached MQTT", len(fake.published), 1)

    srv._last_alert_at = time.time() - 61         # cooldown elapsed
    check("alert: publishes again after cooldown",
          srv.publish_alert({"n": 3}), True)
    check("alert: two publishes total", len(fake.published), 2)

    srv.config["mqtt_alert_cooldown"] = 0         # throttle off
    check("alert: cooldown 0 publishes", srv.publish_alert({"n": 4}), True)
    check("alert: cooldown 0 publishes again", srv.publish_alert({"n": 5}), True)
    check("alert: four publishes total", len(fake.published), 4)

    srv.mqtt_client = None
    check("alert: no client -> not published", srv.publish_alert({"n": 6}), False)

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED: {failures}")
        return 1
    print("All anomaly metadata + alert cooldown tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
