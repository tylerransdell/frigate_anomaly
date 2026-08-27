#!/usr/bin/env python3
"""
Frigate Anomaly Detector - Full PWA Server
===========================================
Serves a Progressive Web App for configuring and monitoring
DINOv2-based visual anomaly detection on Frigate NVR cameras.

Features:
  - PWA frontend with zone selection, model knobs, and live dashboard
  - DINOv2 ViT-S/14 via OpenVINO (iGPU optimized)
  - Configurable sampling (interval or MQTT trigger)
  - Duration filter with rapid re-sampling
  - MQTT alert publishing (no auth)
  - Minimum sample count for alert confirmation

Run:
  python server.py                    # default port 8080
  python server.py --port 9000        # custom port 9000

  Open http://localhost:8080 in your browser
"""

import json
import time
import threading
import base64
import re
import logging
import logging.handlers
import argparse
from pathlib import Path
from io import BytesIO
from datetime import datetime

import requests
import numpy as np
import openvino as ov
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify, send_file, Response
import paho.mqtt.client as mqtt

# ============================================================================
# LOGGING (console + rotating file so crashes/stalls leave a trace)
# ============================================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "server.log"

_log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_format)

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter(_log_format))
_logger_root = logging.getLogger()
_logger_root.addHandler(_file_handler)

log = logging.getLogger("anomaly")

# Flask/Werkzeug access logs also go to the same file.
flask_logger = logging.getLogger("werkzeug")
flask_logger.addHandler(_file_handler)

class _AccessLogFilter(logging.Filter):
    """Suppress routine Werkzeug GET access log noise (the front end polls
    /api/dashboard, /api/baseline/list and thumbnails every few seconds).

    Werkzeug lines look like:  IP - - [date] "GET /path HTTP/1.1" 200 -
    A GET is only logged when the response was significant (an error status,
    4xx/5xx). Non-GET (state-changing) requests and any error-level record
    (e.g. unhandled exception traces) always pass through.
    """
    # method, path, and status appear anywhere in the line.
    _METHOD_STATUS = re.compile(r'"([A-Z]+)\s+\S+[^"]*"\s+(\d{3})')

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO:
            return True
        if not isinstance(record.msg, str):
            return True
        msg = record.getMessage()
        m = self._METHOD_STATUS.search(msg)
        if not m:
            return True
        method, status = m.group(1), int(m.group(2))
        # State-changing (non-GET) requests always log.
        if method != "GET":
            return True
        # GET: only significant (error) responses get logged; polling 2xx/3xx don't.
        return status >= 400

flask_logger.addFilter(_AccessLogFilter())

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
BASELINE_DIR = BASE_DIR / "baselines"
MODEL_CACHE = BASE_DIR / "model_cache"
BASELINE_DIR.mkdir(exist_ok=True)
MODEL_CACHE.mkdir(exist_ok=True)

# ============================================================================
# DEFAULT CONFIG
# ============================================================================
DEFAULT_CONFIG = {
    "frigate_url": "http://192.168.1.100:5000",
    "camera": "",
    "zone": {"x": 0, "y": 0, "w": 224, "h": 224},
    "model_size": "half",          # "small" (112), "half" (224) or "full" (518)
    "scale": 1.0,                  # 0.5, 1.0, or 2.0
    "threshold": 0.85,
    "similarity_method": "mean_top3",   # "nearest" (1 closest baseline) or "mean_top3" (mean of top 3)
    "neighbors": 3,                # top-k closest baselines averaged when using "mean_top3"
    "sampling_interval": 60,       # seconds (min 3; how often to sample)
    "duration_filter": 0,          # seconds, 0 = disabled
    "min_samples": 5,              # min anomalous samples to confirm; 0/null -> 1
    "mqtt_server": "",
    "mqtt_port": 1883,
    "mqtt_topic": "frigate/anomaly/alert",
}

# ============================================================================
# GLOBAL STATE
# ============================================================================
config = dict(DEFAULT_CONFIG)
engine_running = False
engine_thread = None
mqtt_client = None

# Model state
compiled_model = None
loaded_model_size = None      # model_size string ("small"/"half"/"full") holds compiled_model
preprocess = None

# Baseline library (1..40 actual images saved under baselines/)
baseline_paths: list[Path] = []
baseline_bank = None          # np.ndarray (N, embed_dim), L2-normalized rows
MAX_BASELINES = 40

# Model-loading state (load runs in background so it never wedges the server)
model_loading = False          # True while a background model load is in progress
model_loading_lock = threading.Lock()
model_error = None             # non-None message if the last load attempt failed

# Dashboard state (last 8 detections)
recent_snippets = []
snippet_lock = threading.Lock()

# Performance / runtime stats (updated by detection loop)
perf_stats = {
    "detections": 0,
    "anomalies": 0,
    "fetch_last_ms": 0.0,
    "fetch_avg_ms": 0.0,
    "infer_last_ms": 0.0,
    "infer_avg_ms": 0.0,
    "total_last_ms": 0.0,
    "total_avg_ms": 0.0,
    "last_sim": None,
    "last_status": None,
    "last_detection_at": None,
    "cumulative_infer_ms": 0.0,
    "cumulative_total_ms": 0.0,
}
perf_lock = threading.Lock()

# Duration filter state
duration_active = False
duration_start = 0
duration_anomaly_count = 0
duration_sample_count = 0

# ============================================================================
# CONFIG PERSISTENCE
# ============================================================================
def load_config():
    global config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            saved = json.load(f)
        config.update(saved)
        log.info("Config loaded from %s", CONFIG_PATH)

def save_config():
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config saved to %s", CONFIG_PATH)

# ============================================================================
# RESOLUTION HELPERS
# ============================================================================
RESOLUTION_MAP = {"full": 518, "half": 224, "small": 112}

def get_image_size():
    return RESOLUTION_MAP.get(config.get("model_size", "half"), 224)

def get_comparison_k() -> int:
    """Number of nearest baselines to score against.

    "nearest" -> 1 (single closest baseline); "mean_top3" -> 3 (mean of the
    three closest, clamping down to however many baselines exist).
    """
    method = config.get("similarity_method", "mean_top3")
    if method == "nearest":
        return 1
    return int(config.get("neighbors", 3) or 3)

# ============================================================================
# MODEL LOADING (with OpenVINO IR caching)
# ============================================================================
def load_model():
    """
    Loads DINOv2 ViT-S/14 and compiles for iGPU via OpenVINO.
    Caches the converted IR to disk so subsequent runs are instant.
    On failure, records model_error and re-raises so callers can surface it.
    """
    global model_loading, model_error
    try:
        _load_model_impl()
        model_error = None
    except Exception as e:
        model_error = f"Model load failed: {e}. The model downloads via torch.hub (facebookresearch/dinov2); check network access."
        log.exception("Model load failed: %s", e)
        raise
    finally:
        model_loading = False


def _load_model_impl():
    """Load DINOv2 via torch.hub (no HF token needed), int8 quantize with NNCF,
    compile for GPU via OpenVINO, cache the quantized IR to disk."""
    global compiled_model, preprocess, loaded_model_size

    image_size = get_image_size()
    loaded_model_size = config.get("model_size", "half")
    ir_dir = MODEL_CACHE / f"ir_{image_size}"
    ir_xml = ir_dir / "model.xml"

    core = ov.Core()

    if ir_xml.exists():
        log.info("Loaded cached OpenVINO IR from %s", ir_dir)
        compiled_model = core.compile_model(str(ir_xml), "GPU")
    else:
        log.info("Loading DINOv2 via torch.hub and quantizing to int8 (first run, may take a few min)...")
        import nncf

        # torch.hub pulls the public, non-gated facebookresearch/dinov2 weights.
        pt_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        pt_model.eval()

        example_input = torch.randn(1, 3, image_size, image_size)
        ov_model = ov.convert_model(pt_model, example_input=example_input)
        ov_model.reshape([1, 3, image_size, image_size])

        # int8 quantization (NNCF) — the OpenVINO-native quantization path.
        def _transform(item):
            return item

        calib = [torch.randn(1, 3, image_size, image_size) for _ in range(8)]
        quantized_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calib, _transform),
            model_type="transformer",
        )

        ir_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantized_model, str(ir_xml))
        log.info("Quantized IR saved to %s", ir_dir)

        compiled_model = core.compile_model(quantized_model, "GPU")

    log.info("Model compiled on GPU at %dx%d", image_size, image_size)

    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def start_model_load_async():
    """Launch a model load in a background thread. Reloads when the requested
    model size differs from the one currently compiled in memory."""
    if not model_loading_lock.acquire(blocking=False):
        return
    try:
        if model_loading:
            return
        if compiled_model is not None and loaded_model_size == config.get("model_size", "half"):
            return
        model_loading = True
        threading.Thread(target=load_model, daemon=True).start()
    finally:
        model_loading_lock.release()

    log.info("Model load started in background thread")


def ready_to_autostart() -> str | None:
    """Return a reason the engine can't auto-start, or None if it can.

    Pre-requires: a Frigate URL, a camera, valid zone coords, a cached
    quantized model, and a baseline image all present."""
    if not config.get("frigate_url"):
        return "no frigate URL configured"
    if not config.get("camera"):
        return "no camera"
    z = config.get("zone") or {}
    if not (z.get("w", 0) > 0 and z.get("h", 0) > 0):
        return "no zone coordinates"
    if int(z.get("w", 0) * z.get("h", 0)) < 1:
        return "zone has zero area"

    ir = MODEL_CACHE / f"ir_{get_image_size()}" / "model.xml"
    if not ir.exists():
        return "model not downloaded/quantized yet"
    if not any(BASELINE_DIR.glob("baseline_*.jpg")):
        return "no baseline image captured"

    return None


def maybe_autostart():
    """If all prerequisites are present, load model + baseline and start the engine."""
    global engine_running
    reason = ready_to_autostart()
    if reason:
        log.info("Auto-start skipped: %s", reason)
        return

    try:
        load_model()
        load_baselines()
        if baseline_bank is None or len(baseline_bank) == 0:
            log.warning("Auto-start: baseline bank empty, skipping engine")
            return
        setup_mqtt()
        engine_running = True
        engine_thread = threading.Thread(target=detection_loop, daemon=True)
        engine_thread.start()
        log.info("Engine auto-started (all prerequisites satisfied)")
    except Exception as e:
        log.exception("Auto-start failed: %s", e)


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================
def get_embedding(image: Image.Image) -> np.ndarray:
    """Extract CLS token embedding from a PIL Image.

    Returns a flat 1-D, L2-normalized (dim,) vector so cosine similarity can
    be computed as a simple normalized dot product.
    """
    inputs = preprocess(image).unsqueeze(0).numpy()
    result = compiled_model([inputs])
    # torch.hub DINOv2 outputs the CLS embedding as a (1, dim) array; flatten it.
    vec = np.asarray(result[0]).astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec

# ============================================================================
# FRIGATE API HELPERS
# ============================================================================
def fetch_frigate_snapshot() -> Image.Image | None:
    """Pull latest.jpg from Frigate, crop to zone, apply scale."""
    url = f"{config['frigate_url']}/api/{config['camera']}/latest.jpg"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        log.error("Failed to fetch snapshot: %s", e)
        return None

    # Crop to zone
    z = config["zone"]
    if z["w"] > 0 and z["h"] > 0:
        img = img.crop((z["x"], z["y"], z["x"] + z["w"], z["y"] + z["h"]))

    # Apply scale
    scale = config.get("scale", 1.0)
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img

def fetch_frigate_cameras(frigate_url: str) -> list[str]:
    """Get list of camera names from Frigate config API."""
    try:
        resp = requests.get(f"{frigate_url}/api/config", timeout=5)
        resp.raise_for_status()
        cfg = resp.json()
        return list(cfg.get("cameras", {}).keys())
    except Exception as e:
        log.error("Failed to fetch cameras: %s", e)
        return []

# ============================================================================
# BASELINE MANAGEMENT
# ============================================================================
# A "baseline bank" of up to 40 actual images. Detection scores against the
# mean of the closest neighbors in this bank instead of a single baseline.

def _baseline_files() -> list[Path]:
    """All baseline image files on disk, sorted by name for stable ordering."""
    return sorted(BASELINE_DIR.glob("baseline_*.jpg"))

def _next_baseline_index() -> int:
    """Next available 1-based index (skips names like baseline_A.jpg)."""
    nums = []
    for p in _baseline_files():
        m = re.match(r"baseline_(\d+)\.jpg", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums, default=0) + 1)

def load_baselines():
    """Scan baseline dir, recompute embeddings with the currently loaded model,
    and rebuild the in-memory bank (N x D, L2-normalized rows)."""
    global baseline_paths, baseline_bank
    files = _baseline_files()
    baseline_paths = list(files)
    if not files:
        baseline_bank = None
        log.info("No baseline images found")
        return
    if compiled_model is None:
        load_model()
    embs = [get_embedding(Image.open(p).convert("RGB")) for p in files]
    baseline_bank = np.stack(embs).astype(np.float32)
    log.info("Baseline bank loaded: %d image(s)", len(files))

def add_baseline_image(img: Image.Image) -> Path | None:
    """Persist a snapshot as a new baseline and update the bank.
    Returns the new path, or None if the bank is already full."""
    if len(_baseline_files()) >= MAX_BASELINES:
        log.warning("Baseline bank full (%d), refusing to add", MAX_BASELINES)
        return None
    if compiled_model is None:
        load_model()
    path = BASELINE_DIR / f"baseline_{_next_baseline_index():02d}.jpg"
    img.save(path)
    # Rebuild the bank from disk (sorted) so in-memory order always matches the
    # on-disk filename order used by _baseline_metrics / _baseline_files. An
    # incremental append here would drift out of order after a removal +
    # re-add and produce confidence on the wrong image until a restart.
    load_baselines()
    log.info("Baseline added: %s (bank now %d)", path.name, len(baseline_paths))
    return path

def remove_baseline_by_index(index: int) -> bool:
    """Delete the baseline at the given 1-based index (in _baseline_files order)."""
    files = _baseline_files()
    if index < 1 or index > len(files):
        return False
    path = files[index - 1]
    try:
        path.unlink()
        log.info("Baseline removed: %s", path.name)
    except OSError as e:
        log.error("Failed to remove baseline %s: %s", path, e)
        return False
    load_baselines()
    return True

def capture_baseline_image() -> bool:
    """Capture current zone as a new baseline image."""
    if compiled_model is None:
        load_model()
    img = fetch_frigate_snapshot()
    if img is None:
        return False
    return add_baseline_image(img) is not None

def _baseline_metrics() -> dict[str, dict]:
    """Per-baseline nearest-neighbor confidence.

    For each baseline image, the nearest neighbor is the *other* baseline it is
    most similar to inside the bank. Confidence = that cosine similarity. A very
    high confidence means the baseline is effectively a duplicate of another;
    low confidence means it's genuinely distinct.
    """
    if baseline_bank is None or len(baseline_bank) == 0:
        return {}
    files = _baseline_files()
    n = len(baseline_bank)
    metrics = {}
    if n == 1:
        # Nothing else to compare against — trivially its own reference.
        metrics[files[0].name] = {"nn_sim": 1.0, "confidence": 100.0}
        return metrics
    for i in range(n):
        sims = np.dot(baseline_bank[i], baseline_bank.T)  # sim to every baseline
        sims = np.delete(sims, i)                          # exclude self
        nn_sim = float(np.max(sims))
        metrics[files[i].name] = {"nn_sim": nn_sim, "confidence": nn_sim * 100.0}
    return metrics

def evaluate_against_baselines(embedding: np.ndarray, bank: np.ndarray,
                               threshold: float = 0.85, k: int = 3):
    """Score a test embedding against the baseline bank.

    Similarity = mean cosine of the k closest baselines (or all of them if
    fewer k exist). Maps to a 0-100% confidence and an anomaly flag.
    Returns (mean_similarity, confidence_percent, is_anomaly).
    """
    if bank is None or len(bank) == 0:
        return 0.0, 0.0, True
    actual_k = min(max(int(k), 1), len(bank))
    sims = np.dot(bank, embedding.T).flatten()
    top_k = np.sort(sims)[-actual_k:]
    mean_sim = float(np.mean(top_k))

    min_floor, max_ceiling = 0.65, 0.98
    confidence = (mean_sim - min_floor) / (max_ceiling - min_floor)
    confidence = float(np.clip(confidence, 0.0, 1.0) * 100.0)
    is_anomaly = mean_sim < threshold
    return mean_sim, confidence, is_anomaly

# ============================================================================
# MQTT SETUP
# ============================================================================
def setup_mqtt():
    """Connect to MQTT broker for alert publishing."""
    global mqtt_client

    if not config.get("mqtt_server"):
        log.warning("No MQTT server configured, alerts disabled")
        return

    mqtt_client = mqtt.Client(
        client_id="frigate-anomaly",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )

    def on_connect(client, userdata, flags, rc, properties=None):
        log.info("MQTT connected to %s:%s", config["mqtt_server"], config["mqtt_port"])

    mqtt_client.on_connect = on_connect

    try:
        mqtt_client.connect(config["mqtt_server"], config["mqtt_port"], keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        log.error("MQTT connection failed: %s", e)
        mqtt_client = None

def publish_alert(payload: dict):
    """Publish alert to MQTT topic."""
    if mqtt_client and config.get("mqtt_topic"):
        try:
            mqtt_client.publish(
                config["mqtt_topic"],
                json.dumps(payload),
                qos=1,
            )
            log.info("Alert published to %s", config["mqtt_topic"])
        except Exception as e:
            log.error("MQTT publish failed: %s", e)

# ============================================================================
# DASHBOARD HELPERS
# ============================================================================
def add_snippet(image: Image.Image, similarity: float, is_anomaly: bool):
    """Add a detection result to the dashboard (keeps last 8)."""
    # Encode image as base64 data URI
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=60)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    snippet = {
        "image_url": data_uri,
        "similarity": similarity,
        "is_anomaly": bool(is_anomaly),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    with snippet_lock:
        recent_snippets.insert(0, snippet)
        while len(recent_snippets) > 8:
            recent_snippets.pop()

# ============================================================================
# DETECTION ENGINE (Background Thread)
# ============================================================================
def detection_loop():
    """
    Main detection loop. Schedules interval sampling, and during an active
    duration filter speeds up sampling so enough samples fit in the window.
    """
    global engine_running, duration_active, duration_start
    global duration_anomaly_count, duration_sample_count

    log.info("Detection engine started")

    while engine_running:
        try:
            # ---- SAMPLING DELAY ----
            # 3s is the fastest sampling allowed (floor for any interval).
            min_samples = config.get("min_samples") or 1        # 0/null -> 1
            base_interval = max(3.0, float(config.get("sampling_interval", 60) or 60))

            if duration_active:
                duration = float(config.get("duration_filter", 0) or 0)
                # If the normal interval already fits min_samples inside the
                # duration window, keep it. Otherwise speed up so that
                # min_samples samples fit, but never faster than 3s.
                if duration >= min_samples * base_interval:
                    delay = base_interval
                else:
                    delay = max(3.0, duration / min_samples)
            else:
                delay = base_interval

            time.sleep(delay)

            if not engine_running:
                break

            # ---- FETCH & PROCESS ----
            t_cycle = time.perf_counter()
            t0 = time.perf_counter()
            img = fetch_frigate_snapshot()
            fetch_ms = (time.perf_counter() - t0) * 1000.0
            if img is None:
                continue

            if baseline_bank is None or len(baseline_bank) == 0:
                log.warning("No baseline bank set, skipping")
                continue

            t0 = time.perf_counter()
            embedding = get_embedding(img)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            similarity, confidence, is_anomaly = evaluate_against_baselines(
                embedding, baseline_bank,
                threshold=config.get("threshold", 0.85),
                k=get_comparison_k(),
            )
            total_ms = (time.perf_counter() - t_cycle) * 1000.0

            # ---- PERF STATS & DETECTION LOGGING ----
            with perf_lock:
                perf_stats["detections"] += 1
                if is_anomaly:
                    perf_stats["anomalies"] += 1
                perf_stats["fetch_last_ms"] = fetch_ms
                perf_stats["infer_last_ms"] = infer_ms
                perf_stats["total_last_ms"] = total_ms
                perf_stats["fetch_avg_ms"] = (perf_stats["fetch_avg_ms"] * (perf_stats["detections"] - 1) + fetch_ms) / perf_stats["detections"]
                perf_stats["infer_avg_ms"] = (perf_stats["infer_avg_ms"] * (perf_stats["detections"] - 1) + infer_ms) / perf_stats["detections"]
                perf_stats["total_avg_ms"] = (perf_stats["total_avg_ms"] * (perf_stats["detections"] - 1) + total_ms) / perf_stats["detections"]
                perf_stats["cumulative_infer_ms"] += infer_ms
                perf_stats["cumulative_total_ms"] += total_ms
                perf_stats["last_sim"] = round(float(similarity), 4)
                perf_stats["last_status"] = "ANOMALY" if is_anomaly else "NORMAL"
                perf_stats["last_detection_at"] = datetime.now().strftime("%H:%M:%S")

            log.info(
                "DETECT cam=%s sim=%.4f conf=%.1f%% status=%s thresh=%.2f fetch=%.1fms infer=%.1fms total=%.1fms",
                config.get("camera"), similarity, confidence,
                "ANOMALY" if is_anomaly else "NORMAL",
                config.get("threshold", 0.85),
                fetch_ms, infer_ms, total_ms,
            )

            # ---- DURATION FILTER LOGIC ----
            duration = config.get("duration_filter", 0)

            if duration > 0:
                if is_anomaly and not duration_active:
                    # Start duration window
                    duration_active = True
                    duration_start = time.time()
                    duration_anomaly_count = 1
                    duration_sample_count = 1
                    log.info("Duration filter started (%.0fs window)", duration)
                    add_snippet(img, similarity, is_anomaly)
                    continue

                elif duration_active:
                    duration_sample_count += 1
                    if is_anomaly:
                        duration_anomaly_count += 1

                    elapsed = time.time() - duration_start

                    if elapsed >= duration:
                        # Duration window complete — evaluate
                        min_samples = config.get("min_samples") or 1   # 0/null -> 1
                        if duration_anomaly_count >= min_samples:
                            # CONFIRMED ANOMALY
                            log.info(
                                "🚨 ALERT: %d/%d samples anomalous over %.0fs",
                                duration_anomaly_count, duration_sample_count, elapsed
                            )

                            payload = {
                                "camera": config["camera"],
                                "similarity": round(similarity, 4),
                                "anomaly_count": duration_anomaly_count,
                                "sample_count": duration_sample_count,
                                "duration": round(elapsed, 1),
                                "timestamp": datetime.now().isoformat(),
                            }
                            publish_alert(payload)

                        else:
                            log.info(
                                "Duration window ended: %d/%d anomalous (need %d), no alert",
                                duration_anomaly_count, duration_sample_count, min_samples
                            )

                        # Reset duration state
                        duration_active = False
                        duration_anomaly_count = 0
                        duration_sample_count = 0

                    add_snippet(img, similarity, is_anomaly)
                    continue
            else:
                # No duration filter — simple threshold check
                if is_anomaly:
                    log.info("⚠ Anomaly detected (sim: %.3f)", similarity)
                    payload = {
                        "camera": config["camera"],
                        "similarity": round(similarity, 4),
                        "timestamp": datetime.now().isoformat(),
                    }
                    publish_alert(payload)

            add_snippet(img, similarity, is_anomaly)

        except Exception as e:
            log.exception("Unexpected error in detection loop: %s", e)
            time.sleep(1.0)

    log.info("Detection engine stopped")

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({**config, "engine_running": engine_running,
                    "model_ready": compiled_model is not None,
                    "model_loading": model_loading,
                    "model_error": model_error})

@app.route("/api/config", methods=["POST"])
def post_config():
    global config
    config.update(request.json)
    save_config()
    return jsonify({"status": "Config saved"})

@app.route("/api/cameras")
def get_frigate_cameras():
    """Fetch list of cameras from Frigate's config endpoint."""
    # Strip trailing slashes to prevent URL formatting issues
    frigate_url = request.args.get("frigate_url", "").rstrip('/')
    
    if not frigate_url:
        return jsonify({"error": "frigate_url required"}), 400
    
    try:
        # Frigate's config endpoint contains the camera definitions
        resp = requests.get(f"{frigate_url}/api/config", timeout=5)
        
        # Check if Frigate returned HTML (like a login page) instead of JSON
        if 'text/html' in resp.headers.get('Content-Type', ''):
            return jsonify({
                "error": "Frigate returned an HTML page. Check URL and ensure Authentication is DISABLED in Frigate settings."
            }), 401
            
        resp.raise_for_status()
        config_data = resp.json()
        
        # Extract camera names from the 'cameras' dictionary keys
        camera_names = list(config_data.get("cameras", {}).keys())
        
        if not camera_names:
            return jsonify({"error": "No cameras found in Frigate config."}), 404
            
        return jsonify({"cameras": camera_names})
        
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Cannot connect to {frigate_url}. Check IP and port."}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/api/baseline/capture", methods=["POST"])
def api_capture_baseline():
    if not config.get("camera"):
        return jsonify({"error": "No camera configured"}), 400
    if len(_baseline_files()) >= MAX_BASELINES:
        return jsonify({"error": "Baseline bank full (40/40). Delete baseline images before adding more."}), 409
    try:
        if capture_baseline_image():
            return jsonify({"status": "ok", "count": len(_baseline_files())})
        return jsonify({"error": "Failed to capture"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/baseline/list")
def api_baseline_list():
    """List saved baselines with thumbnails + nearest-neighbor confidence.

    confidence = similarity to that baseline's closest neighbor in the bank.
    Too close (near 100%) => likely a duplicate candidate. Recomputes from the
    current bank on every request, so it updates as baselines are added/removed."""
    files = _baseline_files()
    # Build the bank lazily if the model is already compiled (no forced load
    # on a polled endpoint — numbers show as soon as the engine's model is
    # available). A single baseline is trivially 100% + nn_sim 1.0.
    if baseline_bank is None and compiled_model is not None:
        load_baselines()
    if baseline_bank is not None and len(baseline_bank) == len(files):
        metrics = _baseline_metrics()
    elif len(files) == 1:
        metrics = {files[0].name: {"nn_sim": 1.0, "confidence": 100.0}}
    else:
        metrics = {}
    items = [
        {
            "index": idx,
            "name": p.name,
            "image_url": f"/api/baseline/image/{idx}",
            "confidence": metrics.get(p.name, {}).get("confidence"),
        }
        for idx, p in enumerate(files, start=1)
    ]
    return jsonify({"baselines": items, "count": len(items), "max": MAX_BASELINES})

@app.route("/api/baseline/image/<int:index>")
def api_baseline_image(index: int):
    files = _baseline_files()
    if index < 1 or index > len(files):
        return jsonify({"error": "No baseline"}), 404
    return send_file(files[index - 1], mimetype="image/jpeg")

@app.route("/api/baseline/image")
def api_latest_baseline_image():
    """Latest saved baseline (used by the older single-selector preview)."""
    files = _baseline_files()
    if files:
        return send_file(files[-1], mimetype="image/jpeg")
    return jsonify({"error": "No baseline"}), 404

@app.route("/api/baseline/add", methods=["POST"])
def api_add_baseline():
    """Add an image (e.g. picked from the live dashboard) as a new baseline."""
    if len(_baseline_files()) >= MAX_BASELINES:
        return jsonify({"error": "Baseline bank full (40/40). Delete baseline images before adding more."}), 409
    data = request.get_json(silent=True) or {}
    image_url = data.get("image_url", "")
    try:
        if image_url.startswith("data:image"):
            _, b64 = image_url.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        else:
            return jsonify({"error": "Unsupported image format"}), 400
        path = add_baseline_image(img)
        if path is None:
            return jsonify({"error": "Baseline bank full (40/40). Delete baseline images before adding more."}), 409
        return jsonify({"status": "ok", "count": len(_baseline_files())})
    except Exception as e:
        log.exception("Failed to add baseline")
        return jsonify({"error": str(e)}), 500

@app.route("/api/baseline/remove/<int:index>", methods=["POST"])
def api_remove_baseline(index: int):
    if remove_baseline_by_index(index):
        return jsonify({"status": "ok", "count": len(_baseline_files())})
    return jsonify({"error": "Invalid baseline index"}), 400

@app.route("/api/engine/start", methods=["POST"])
def api_start_engine():
    global engine_running, engine_thread

    if engine_running:
        return jsonify({"running": True, "status": "Already running"})

    if not config.get("camera"):
        return jsonify({"running": False, "error": "No camera configured"}), 400

    # Reload the model if the requested model_size changed since it was last
    # compiled in memory (small/half/full all need their own IR).
    if compiled_model is None or loaded_model_size != config.get("model_size", "half"):
        if model_loading:
            return jsonify({"running": False, "loading": True,
                            "status": "Model is still loading..."}), 202
        start_model_load_async()
        return jsonify({"running": False, "loading": True,
                        "status": "Model loading in background..."}), 202

    # Recompute baseline embeddings for the currently loaded model size.
    load_baselines()
    if baseline_bank is None or len(baseline_bank) == 0:
        return jsonify({"running": False, "error": "No baseline. Capture one first."}), 400

    # Setup MQTT
    setup_mqtt()

    # Start engine thread
    engine_running = True
    engine_thread = threading.Thread(target=detection_loop, daemon=True)
    engine_thread.start()

    return jsonify({"running": True, "status": "Engine started"})

@app.route("/api/engine/stop", methods=["POST"])
def api_stop_engine():
    global engine_running
    engine_running = False
    if mqtt_client:
        mqtt_client.loop_stop()
    return jsonify({"running": False, "status": "Engine stopped"})

@app.route("/api/snapshot")
def proxy_snapshot():
    """Proxy Frigate snapshot endpoint for the web UI."""
    # .strip() removes hidden spaces/newlines
    frigate_url = request.args.get("frigate_url", "").strip().rstrip('/')
    camera_name = request.args.get("camera_name", "").strip()

    if not frigate_url or not camera_name:
        return jsonify({"error": "Missing URL or Camera Name"}), 400

    target_url = f"{frigate_url}/api/{camera_name}/latest.jpg"
    log.info("Proxying snapshot: %s", target_url)

    try:
        headers = {"User-Agent": "curl/7.81.0", "Accept": "image/*"}
        resp = requests.get(target_url, headers=headers, timeout=10)

        # CRITICAL FIX: If Frigate returns JSON (the 46-byte error),
        # we MUST return a 500 error to the browser so it stops trying to render it as an image.
        if 'image' not in resp.headers.get('Content-Type', ''):
            log.error("Frigate returned JSON error for %s: %s", target_url, resp.text)
            return jsonify({"error": "Frigate returned an error. See server terminal."}), 500

        return Response(resp.content, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache'})

    except Exception as e:
        log.error("Snapshot proxy exception for %s: %s", target_url, e)
        return jsonify({"error": str(e)}), 500

def _json_safe(obj):
    """Recursively convert numpy scalars/arrays/dates to JSON-native types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.route("/api/dashboard")
def api_dashboard():
    with snippet_lock:
        snippets = list(recent_snippets)
    with perf_lock:
        perf = dict(perf_stats)
    payload = {
        "engine_running": engine_running,
        "snippets": snippets,
        "model_ready": compiled_model is not None,
        "model_loading": model_loading,
        "model_error": model_error,
        "perf": perf,
    }
    return jsonify(_json_safe(payload))

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frigate Anomaly Detector Server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    load_config()
    # threaded=True: serve request concurrently so a slow outbound call
    # (snapshot proxy / camera fetch / model load) never wedges the server.
    log.info("Starting Frigate Anomaly Detector PWA on http://0.0.0.0:%d", args.port)
    maybe_autostart()
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
