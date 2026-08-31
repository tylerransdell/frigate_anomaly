# Anomaly Detection for Frigate

**version 0.0 — proof of concept.** Don't get attached.

An ugly yet powerful tool for zero-shot anomaly detection on Frigate camera feeds.

This is a pragmatic utility bringing Zero-Shot Anomaly Detection directly to local Frigate streams using OpenVINO and Intel iGPUs. Define "normal" structurally and let the model handle the rest.

---

## Why Zero-Shot Anomaly Detection?

Traditional classifiers break down when data is scarce or the environment introduces unpredictable edge cases.

| Approach | How It Works | What It Triggers On |
| :--- | :--- | :--- |
| **Classification** | Matches input against well-defined, and specifically labeled/trained categories. | **Any normal** objects/events. |
| **ZSAD (This Tool)** | Takes known normal baselines and measures structural deviation. | **ANY unknown event** outside the normal baseline. |

* No training overhead: Skip dataset curation and manual labeling.
* Open-world defense: Traditional classifiers only catch what they were explicitly trained to see. ZSAD flags any structural deviation from normal geometry.
* Lighting resilient: Powered by DINOv2 feature embeddings, it handles natural illumination shifts and sliding shadows without false alarms.

---

## What It Does

* Frigate API Integration: Pulls snapshots directly from your local Frigate server (no auth, local LAN use only).
* Flexible UI Cropping & Zones: Draw any rectangle for a region or use fixed scaling. The server automatically handles image resizing to fit the model input.
* Robust k-NN Scoring: Score against your single nearest neighbor (k=1) or the mean of the top 3 (k=3) across up to 64 baseline images.
* Automated Batch Baselines: One-button pull for 1, 24 (hourly), or 48 (every 30 mins) baseline snapshots straight from Frigate's past recordings. Gracefully handles recordings < 24hr.
* Smart Filtering: Configurable sample and duration filters to manage certainty and cut down on transient noise.
* Global Anomaly Store: Saves up to 128 unique anomalies in a dedicated global store (deduped by similarity threshold). Includes a max samples per unique anomaly setting (default 1, 0 = 1) so you can keep multiple frames from the same event when you want one to become a baseline later.
* Easy Management: Easily remove baselines or promote them directly from the live or history dashboards.
* OpenVINO GPU Only: Runs the DINOv2 vision model strictly through OpenVINO GPU — simple, fixed, and fully automatic.
* Profiles & Persistence: Save settings and baselines into profiles and load them via drop-down. Configurations persist cleanly to local disk (config.json, baselines/, anomalies/).

**Won't fix** — deliberate proof-of-concept scoping:
* The UI is what ships: No external REST API for third-party control agents or Home Assistant core integration. 
* Global-token inference: Optimized for zero-shot anomaly detection and scene tampering, not for microscopic local inspections (like surface scratches or dents).
* No Frigate Auth: Designed for trusted local network setups behind Frigate's local API.

---

## Throughput & Performance

* Meteor Lake iGPU: 
  * Half Model (224px): ~35ms per inference.
  * Small Model (112px): ~15ms per inference.
* Stability Safeguard: Live detection is throttled to a minimum interval (max 1 check per 3 seconds).

---

## Quickstart (4 Steps)
```
git clone https://github.com/tylerransdell/frigate_anomaly && cd frigate_anomaly
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 server.py                # web UI at http://<host-ip>:8080
# or, if port 8080 is in use:
python3 server.py --port 9090
```
Note: On the first run, the script compiles the OpenVINO IR model and caches it under model_cache/. This takes a minute or two; subsequent launches are instantaneous.

---

## A Photo

![dino.png](dino.png)

---

## How It Decides (k-NN & Resolution Matching)

1. Embeddings: Captured baseline images and live snapshots are translated into compact vector representations of scene geometry using DINOv2.
2. Scoring: Live snapshots are scored against your baseline library using k-nearest-neighbors (k=1 or k=3).
3. Thresholding & Filtering: If the similarity score drops below your Anomaly Threshold, it hits a provisional candidate state. It must then pass your sampling and duration filters before finally being confirmed and logged/alerted as a true anomaly.
4. Resolution Harmonization: Live snapshots pull from Frigate's detect stream, while batch baseline pulls grab from recordings (main stream). If resolutions differ, batch frames are automatically downscaled to match your live crop framing.

---

## Using the UI

1. Open the web UI via your server's local network address and port (e.g., http://192.168.1.x:8080).
2. Enter your Frigate API URL (e.g., http://192.168.1.y:5000 — no auth, keep it on your LAN).
3. Click Fetch Cameras, then select your camera.
4. Load a camera snapshot to view the stream.
5. Choose your Model Size (Small 112 / Half 224) and Crop scale (or use Free mode to draw custom bounding boxes).
6. Capture baselines—individually or by batch-pulling 24/48 hours of historical context.
7. Select your similarity method (Nearest 1 vs. Mean of top 3) and set your Anomaly Threshold.
8. Review sampling/duration filters and hit Start Engine.

Detection will start automatically if the detector has the information it needs to detect, allowing the script to be cleanly wrapped as a service.

---

## License

[MIT](LICENSE)
