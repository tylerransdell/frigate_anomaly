# 🦖 frigate_anomaly

**version 0.0 — proof of concept.** Don't get attached.

A dumb little tool that runs **zero shot anomaly detection** on a Frigate camera feed. It grabs a picture from your camera, compares it to a "normal" baseline that you set, and lets you know (via logs, and optionally MQTT) when something looks weird.

This is the proof of concept version. This will likely be only minimally maintained as proof that zero shot anomaly detection is extremely easy.

---

## ? WHY

When you already have data, there are better options. Zero‑shot is what you use when data is limited or non‑existent.

One educated guess for confidence and you suddenly have a smart detector. No training, no labeling, no dataset prep. Just define “normal” once and let the model handle everything else.

---

## What it does / doesn't

✅ **Does:**
- Pulls snapshots straight from the Frigate API (no auth, local use).
- Lets you pick a camera, draw a zone, and capture a baseline.
- Runs the DINOv2 vision model through **OpenVINO only** (iGPU/CPU — no CUDA drama).
- Gives you basic filtering and interval sampling, speeding up whenever an anomaly is actually brewing.
- Includes a tiny dashboard so you can see what it's flagging.
- Runs as **one instance** — single camera, single baseline, zero swag.

❌ **Doesn't (yet):**
- No multi-camera, no docker-compose, no auth, no Home Assistant integration.
- No GPU-vendor hopping. It's **OpenVINO or nothing**.
- Nothing fancy at all, honestly. It's v0.0. Manage your expectations.

---

## Quickstart (seriously, 4 steps)

```bash
# 1. Clone it
git clone https://github.com/tylerransdell/frigate_anomaly && cd frigate_anomaly

# 2. Make a venv & install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run it
python3 server.py                # web UI at http://localhost:8080
# or, if 8080 is taken:
python3 server.py --port 9090
```

That's it. On the first run it converts and compiles the model (OpenVINO), which can take a minute or two — be patient. After that, it's fast.

---

## 🧪 A Photo

![dino.png](dino.png)

---

## 🐎 Throughput

About **35ms per inference** on a **Meteor Lake** iGPU with the **half (224) model**. That's fast enough to sample every couple of seconds without breaking a sweat console shows inference and fetch times.

---

## 🕹️ Using It

1. Open the web UI (usually `http://host:8080`).
2. Enter your Frigate URL, like `http://192.168.1.x:5000` — **no auth yet**, so just make sure it's reachable on your LAN.
3. Click **Fetch Cameras**, then pick your camera.
4. Click **Load Camera Snapshot** so you can actually see what the camera sees.
5. Tweak **Model Area** (Half 224 / Full 518) and **Crop** (0.5× / 1.0× / 2.0×) to your taste.
6. **Click-drag** a box on the image to select the zone you care about.
7. Click **Capture Baseline** — that's your "everything is normal" picture.
8. Review the **Sampling & Duration** filters, and flip them on if you want.
9. Hit **Start Engine** and let it do its thing.
10. There are also MQTT alerts. Let me know if they work or I'll get around to it. 

---

## ✋ Fine Print & Caveats

- `server.py` compiles the OpenVINO IR once and caches it under `model_cache/`. If you delete that folder, it'll rebuild itself.
- Your settings live in `config.json` (git-ignored), and your baseline lives in `baselines/` (also git-ignored). Both are just plain files on disk.
- **Honest warning:** there's no auth, no HTTPS, nothing. Run it on a trusted LAN and don't port-forward it.

---

## Roadmap (maybe)

Multiple detectors. Polish. Static container. Things like that.

## License

MIT. Do whatever you want, just don't blame me for anything.
