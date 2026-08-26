mkdir frigate-anomaly && cd frigate-anomaly
mkdir -p templates static baselines model_cache
pip install -r requirements.txt

# Create your local config from the template (contains your camera/IP, so it's git-ignored)
cp config.example.json config.json

