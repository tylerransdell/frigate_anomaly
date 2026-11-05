# Anomaly Detection for Frigate

**version 0.1.0 — proof of concept.** Don't get attached.

An ugly yet powerful tool for zero-shot anomaly detection on Frigate camera feeds — or on any camera over RTSP.

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
* Direct RTSP Cameras: No Frigate handy? Select the RTSP source, paste a stream URL, and grab full-resolution frames straight off the camera (uses ffmpeg, credentials stay on the server).
* Flexible UI Cropping & Zones: Draw any rectangle for a region or use fixed scaling. The server automatically handles image resizing to fit the model input.
* Robust k-NN Scoring: Score against your single nearest neighbor (k=1) or the mean of the top 3 (k=3) across up to 64 baseline images.
* Automated Batch Baselines: From Frigate, one-button pull of the **previous** 24 (hourly) or 48 (every 30 mins) snapshots out of its recordings — graceful when recordings are < 24hr. From an RTSP camera, start a stoppable background plan that captures the **next** 24 (hourly) or 48 (every 30 mins) images going forward.
* Smart Filtering: Configurable sample and duration filters to manage certainty and cut down on transient noise.
* Alert Cooldown: Publish one MQTT alert, then stay quiet for a configurable window (default 60s, in seconds — up to days). Escalation belongs in your alerting system.
* Global Anomaly Store: Saves up to 128 unique anomalies in a dedicated global store (deduped by similarity threshold). Every saved anomaly keeps its capture date/time and the original similarity it scored against the baselines **at that moment** — permanent, because baselines change but that number shouldn't. A separate live number shows how close it is to the other stored anomalies. Includes a max samples per unique anomaly setting (default 1, 0 = 1) so you can keep multiple frames from the same event when you want one to become a baseline later.
* Easy Management: Easily remove baselines or promote them directly from the live or history dashboards.
* OpenVINO GPU Only: Runs the DINOv2 vision model strictly through OpenVINO GPU — simple, fixed, and fully automatic.
* Profiles & Persistence: Save settings and baselines into profiles and load them via drop-down. Configurations persist cleanly to local disk (config.json, baselines/, anomalies/, rtsp_schedule.json).

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
* Kick-start: a freshly started engine takes its first sample ~5s in (whenever your interval is ≥20s), then follows the configured cadence. A 10-minute — or 1-hour — interval no longer leaves the dashboard blank until the first slot comes around.
* RTSP mode: every frame is a fresh ffmpeg connection, so a grab costs ~0.1–2s depending on the camera's keyframe interval. Sampling every 60s+ is the sane setting; sub-10s intervals will spend most of their time waiting on the camera.

---

## Quickstart

```bash
git clone https://github.com/tylerransdell/frigate_anomaly && cd frigate_anomaly
mkdir -p data
docker compose up -d --build
# UI at http://<host-ip>:8080
docker compose logs -f          # follow the detector
```

First start builds the image and int8-quantizes DINOv2 (a minute or two, needs internet); after that it is instant. Needs an Intel GPU — the model runs through OpenVINO's GPU plugin, with no CPU fallback.

---

## Docker: storage & networking

`./data` holds everything the app writes — settings, baselines, anomalies, profiles, the compiled model cache and logs — so rebuilds and upgrades keep your setup.

The compose file ships with `network_mode: host` — the container reaches Frigate, your cameras and MQTT by LAN address and serves the UI on `:8080` — and `privileged: true` for iGPU (`/dev/dri`) access. Both can be tightened with the commented `devices:` / `group_add:` / `ports:` / `user:` alternatives at the bottom of the file.

Prefer to keep Frigate off the host network? Give both containers their own bridge network and talk to Frigate privately on its open port 5000:

```bash
docker network create dockernet
docker network connect dockernet frigate
```

…then drop `network_mode: host` from this service in favour of `networks: [dockernet]` plus `ports: ["8080:8080"]`, and point the Frigate URL at `http://frigate:5000`.

Bare metal instead? `pip install -r requirements.txt && python3 server.py` — add `ffmpeg` for RTSP sources and the Intel GPU runtime on the host, both of which the image bundles.

---

## Migrating from the bare-metal install

Stop the old server first (systemd unit, tmux, whatever runs it — it owns `:8080`), then move its state into the volume:

```bash
mkdir -p data
mv config.json baselines anomalies profiles model_cache logs rtsp_schedule.json data/ 2>/dev/null
docker compose up -d --build
```

`./data` is exactly the layout the container expects, so settings, baselines, saved anomalies (with their capture scores and timestamps) and the compiled model all carry over. The redirect is only there because `rtsp_schedule.json` may not exist yet.

---

## A Photo

![dino.png](dino.png)

*Outdated — live dashboard & anomaly store do one-click baseline upgrade now. Still just as ugly.*

---

## How It Decides (k-NN & Resolution Matching)

1. Embeddings: Captured baseline images and live snapshots are translated into compact vector representations of scene geometry using DINOv2.
2. Scoring: Live snapshots are scored against your baseline library using k-nearest-neighbors (k=1 or k=3).
3. Thresholding & Filtering: If the similarity score drops below your Anomaly Threshold, it hits a provisional candidate state. It must then pass your sampling and duration filters before finally being confirmed and logged/alerted as a true anomaly.
4. Resolution Harmonization: Live snapshots pull from Frigate's detect stream, while batch baseline pulls grab from recordings (main stream). If resolutions differ, batch frames are automatically downscaled to match your live crop framing.

---

## Using the UI

1. Open the web UI via your server's local network address and port (e.g., http://192.168.1.x:8080).
2. Under **Camera Connection**, pick your Source:
   * **Frigate** — enter your Frigate API URL (e.g., http://192.168.1.y:5000 — no auth, keep it on your LAN), click Fetch Cameras, then select the camera.
   * **RTSP Camera** — paste the stream URL (e.g., `rtsp://user:pass@192.168.1.z:554/Streaming/Channels/101`), give it a Camera Name of your choosing, and hit **Grab Test Frame** to prove the connection.
3. Click **Load Camera Snapshot** to load a frame, and draw your detection zone on it.
4. Choose your Model Size (Small 112 / Half 224) and Crop scale (or use Free mode to draw custom bounding boxes).
5. Capture baselines: individually, from the live dashboard/anomaly store, or in bulk —
   * Frigate: **Previous 24/48** pulls historical snapshots out of Frigate's recordings.
   * RTSP: **Next 24/48** starts a stoppable background plan that captures one frame every hour (24) or 30 minutes (48) from now on. Progress and a Stop button live right under the buttons.
6. Select your similarity method (Nearest 1 vs. Mean of top 3) and set your Anomaly Threshold — start around **0.93**; the sampling/duration filters keep transient noise out, so a tighter cutoff won't drown you in alerts.
7. Review sampling/duration filters, the MQTT alert cooldown, and hit Start Engine.

Detection will start automatically if the detector has the information it needs to detect, allowing the script to be cleanly wrapped as a service.

---

## License

[MIT](LICENSE)
