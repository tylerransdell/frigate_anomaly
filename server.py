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
from sklearn.metrics.pairwise import cosine_similarity
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
    "model_size": "half",          # "half" (224) or "full" (518)
    "scale": 1.0,                  # 0.5, 1.0, or 2.0
    "threshold": 0.85,
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
preprocess = None
baseline_embedding = None
baseline_image_path = None

# Model-loading state (load runs in background so it never wedges the server)
model_loading = False          # True while a background model load is in progress
model_loading_lock = threading.Lock()
model_error = None             # non-None message if the last load attempt failed

# Dashboard state (last 3 detections)
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
RESOLUTION_MAP = {"full": 518, "half": 224}

def get_image_size():
    return RESOLUTION_MAP.get(config.get("model_size", "half"), 224)

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
    global compiled_model, preprocess

    image_size = get_image_size()
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
    """
    Launch a model load in a background thread so the Flask worker isn't wedged
    by a long download/50 overlap. Guarded so it runs at most once at a time.
    """
    if compiled_model is not None or model_loading:
        return

    if not model_loading_lock.acquire(blocking=False):
        return
    try:
        if compiled_model is not None or model_loading:
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
    if not (BASELINE_DIR / "baseline_A.jpg").exists():
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
        if baseline_embedding is None and not load_existing_baseline():
            log.warning("Auto-start: baseline file missing or failed to load")
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

    Returns a flat 1-D (dim,) vector. Consumers wrap it as [embedding] to get
    the (1, dim) 2-D shape expected by sklearn's cosine_similarity.
    """
    inputs = preprocess(image).unsqueeze(0).numpy()
    result = compiled_model([inputs])
    # torch.hub DINOv2 outputs the CLS embedding as a (1, dim) array; flatten it.
    return np.asarray(result[0]).astype(np.float32).reshape(-1)

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
def capture_baseline_image() -> bool:
    """Capture current zone as baseline image and compute embedding."""
    global baseline_embedding, baseline_image_path

    if compiled_model is None:
        load_model()

    img = fetch_frigate_snapshot()
    if img is None:
        return False

    # Save baseline image
    baseline_image_path = BASELINE_DIR / "baseline_A.jpg"
    img.save(baseline_image_path)

    # Compute embedding
    baseline_embedding = get_embedding(img)
    log.info("Baseline captured and embedding computed")
    return True

def load_existing_baseline() -> bool:
    """Load baseline from disk if it exists."""
    global baseline_embedding, baseline_image_path
    baseline_image_path = BASELINE_DIR / "baseline_A.jpg"
    if not baseline_image_path.exists():
        return False
    if compiled_model is None:
        load_model()
    img = Image.open(baseline_image_path).convert("RGB")
    baseline_embedding = get_embedding(img)
    log.info("Existing baseline loaded from disk")
    return True

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
    """Add a detection result to the dashboard (keeps last 3)."""
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
        while len(recent_snippets) > 3:
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

            if baseline_embedding is None:
                log.warning("No baseline set, skipping")
                continue

            t0 = time.perf_counter()
            embedding = get_embedding(img)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            similarity = cosine_similarity([embedding], [baseline_embedding])[0][0]
            is_anomaly = similarity < config.get("threshold", 0.85)
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
                "DETECT cam=%s sim=%.4f status=%s thresh=%.2f fetch=%.1fms infer=%.1fms total=%.1fms",
                config.get("camera"), similarity,
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
    try:
        if capture_baseline_image():
            return jsonify({"status": "ok"})
        return jsonify({"error": "Failed to capture"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/baseline/image")
def api_baseline_image():
    path = BASELINE_DIR / "baseline_A.jpg"
    if path.exists():
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "No baseline"}), 404

@app.route("/api/engine/start", methods=["POST"])
def api_start_engine():
    global engine_running, engine_thread

    if engine_running:
        return jsonify({"running": True, "status": "Already running"})

    if not config.get("camera"):
        return jsonify({"running": False, "error": "No camera configured"}), 400

    # Load the model in the background when needed, so the request returns at
    # once and the server stays responsive during the (potentially long) load.
    if compiled_model is None:
        if model_loading:
            return jsonify({"running": False, "loading": True,
                            "status": "Model is still loading..."}), 202
        start_model_load_async()
        return jsonify({"running": False, "loading": True,
                        "status": "Model loading in background..."}), 202

    # Load or capture baseline
    if baseline_embedding is None:
        if not load_existing_baseline():
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
