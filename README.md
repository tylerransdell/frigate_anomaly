# 🦖 Anomaly Detection for Frigate

**version 0.0 — proof of concept.** Don't get attached.

A dumb little tool that runs **zero shot anomaly detection** on a Frigate camera feed. It grabs a picture from your camera, compares it to "normal" baselines that you set, and lets you know (via logs, and optionally MQTT) when something looks weird.

The current point of this project isn't to be a long-term solution. It's a thing to play with while you figure out if **zero-shot anomaly detection is actually useful in the surveillance space**. Treat it as a playground: if ZSAD turns out to be valuable here, great — this proves it can be done quick and easy. If not, you've lost 20 minutes.

This is very easy. You just need an Intel GPU or iGPU. You pick a camera, a region, and you can even pull 48 baseline images over the previous 24 hours in one button press. Then, start the engine.

---

## ? WHY

When you already have data, there are better options. Zero‑shot is what you use when data is limited or non‑existent.

No training, no labeling, no dataset prep. Just define “normal” once and let the model handle everything else.

---

## What it does / doesn't

✅ **Does:**
- Pulls snapshots straight from the Frigate API (no auth, local use).
- Lets you pick a camera, draw a zone, and capture baselines.
- Keeps a **baseline library of up to 64 images**, so your "normal" is more than one lucky snapshot — grab a 24-hour set from recordings in one click.
- Everything lives in the UI — no editing JSON or YAML by hand.
- Models: pick **Small (112)** or **Half (224)**; the old full (518) model is gone.
- Scoring: choose k-NN with **k=1 (nearest)** or **k=3 (mean of top 3)**.
- Sample + duration filters improve accuracy and fit the whole thing to the real-world problem.
- Auto-pull baselines from recordings: grab **1 now**, **24 (past 24 hours, hourly)**, or **48 (every 30 min)** at once. Only have 18 hours recorded? It pulls what it can — it doesn't choke.
- Tracks a running **"lowest observed similarity"** to help tune (resets when the config changes).
- Saves up to **128 unique anomalies** as a **global store** — not bound to any profile, zone, camera, or boundary. Deduped by a configurable similarity threshold against previously saved ones, and a **max samples per unique anomaly** (default 1, 0 = 1) lets you keep a few frames from the *same* event when you want one to become a baseline later.
- Easy to remove baselines or promote them from the live/history dashboards.
- Runs the DINOv2 vision model through **OpenVINO only** (iGPU/CPU — no CUDA drama).
- Saves **profiles** (baselines + settings) and loads them from a drop-down.

**Won't fix** — deliberate proof-of-concept scoping. This is a *campaign*, not a *maintain*:
- The **UI is what ships**. No HTTP API for third parties; not written for live external control by an agent or Home Assistant.
- The **inference is global-token-only** by design — good for zero-shot anomaly detection and tampering, not for finding dents/scratches/tears. No per-region/local modeling.
- **OpenVINO only** — no GPU-vendor hopping (I can't test others).
- **Ugly UI.** It is what it is. It works.
- Proof-of-concept codebase: stable if used as intended, less stable if you click fast. Maybe you have to refresh a couple times when changing setups.

**Not yet** (could happen, no commitment):
- No multi-camera, no docker-compose/container — it's a script. Wrap it in systemd if you want it to survive reboots.
- **No Frigate auth** — maybe not possible? Tried and failed.

---

## Quickstart (seriously, 4 steps)

```bash
# 1. Clone it
git clone https://github.com/tylerransdell/frigate_anomaly && cd frigate_anomaly

# 2-3. Make a venv & install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Run it
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

About **35ms per inference** on a **Meteor Lake** iGPU with the **half (224) model** — dropping to roughly **15ms** on the **small (112) model**. There is also fetch time. Currently detected is hardcoded no faster than 1 per 3s as a stability hedge.

---

## 🎯 How It Decides (k-NN)

This is textbook **k-nearest-neighbors** on the baseline library:

1. Every captured baseline is turned into an embedding — a compact vector fingerprint of the scene.
2. Every live snapshot becomes an embedding too, then gets scored against your whole **baseline library (up to 64 images)**.
3. Pick the method: score against your **single nearest neighbor** (k=1) or the **mean of the top 3 nearest** (k=3 / "mean of top 3").
4. That similarity hits your **Anomaly Threshold** — under it, it's an anomaly; at or above it, normal.

The baseline dashboard previews each image's confidence (its own nearest-neighbor similarity), so you can spot duplicates or outliers in your "normal" set before they skew detection.

**Live vs. recording resolution:** live snapshots come from Frigate's **detect stream** (whatever you've set there), while the 24h/48h batch pulls its set straight off **recordings** (Frigate's main-stream grabs). If those are the **same resolution, great — nothing to do**. If they differ, the batch is **high-quality downscaled to the live snapshot's resolution** before the zone is cut, so every baseline covers exactly the same scene framing you boxed. (Run Frigate detection at full resolution and batch == live automatically.) This still works extremely well and honestly seems to work fine even is aspect ratio of record and detect are a little off.

---

## 🕹️ Using It

1. Open the web UI (usually `http://host:8080`).
2. Enter your Frigate URL, like `http://192.168.1.x:5000` — **no auth**, so just make sure it's reachable on your LAN.
3. Click **Fetch Cameras**, then pick your camera.
4. Click **Load Camera Snapshot** so you can actually see what the camera sees. 
5. Tweak **Model Area** (Small 112 / Half 224) and **Crop** (0.25× / 0.5× / 1.0× / 2.0× / 3.0× / **Free**) to your taste.
6. Pick the zone any way you like. With a numeric Crop Scale the box is pre-sized and you click-drag to move it around. With **Free** you can draw any rectangle on the image and the server resizes the crop to the model input (112 or 224) for you.
7. Click **Capture Baseline** — that's your "everything is normal" picture (up to 64). There are buttons to capture 24 or 48 at once over the last day.
8. Pick a **Similarity Method** — **Nearest 1** scores against your single closest baseline, or **Mean of top 3** scores against the average of your 3 closest (or however many you've saved). Same **Anomaly Threshold** line applies to whichever you choose.
9. Review the **Sampling & Duration** filters, and flip them on if you want.
10. Hit **Start Engine** and let it do its thing.
11. There are also MQTT alerts. Let me know if they work or I'll get around to it. 

---

## ♻️ Persistence & Startup

This is a standalone script, not a daemon yet. To keep it running across reboots, wrap `server.py` in a **system service** (e.g. a systemd unit) for now. A containerized (Docker) build may show up at some point, but it's **not** here today.

The inference engine will fire up automatically on startup, but only once everything it needs has been saved. Your prerequisites are:

- **Server** (Frigate URL)
- **Camera**
- **Detect area**
- **At least one baseline image** (the library holds up to 64)

Once all four are configured and saved, the engine starts on launch and does its thing.

---

## ✋ Fine Print & Caveats

- `server.py` compiles the OpenVINO IR once and caches it under `model_cache/`. If you delete that folder, it'll rebuild itself.
- Your settings live in `config.json` (git-ignored), and your baselines live in `baselines/` (also git-ignored). Both are just plain files on disk. Up to 64 baseline images, compared by your chosen method — they persist across restarts. Saved **anomalies live in `anomalies/`** as a separate global store, also git-ignored — they persist across restarts and are never touched by profiles or config.
- **Honest warning:** there's no auth, no HTTPS, nothing. Run it on a trusted LAN and don't port-forward it.

---

## License

MIT. Do whatever you want, just don't blame me for anything.
