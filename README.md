# sky-hack-mvp
End-to-end pipeline to score ORD flight departure difficulty—EDA, calibrated model, daily ranking/classes, and reproducible outputs.

cat > README.md <<'EOF'
# Flight Difficulty Score (FDS) — ORD

**Team:** The Wingmen  
**Authors:** ALOK_SINGH, YASH_KUMAR_GUPTA

End-to-end pipeline to score the **difficulty** of ORD departures and rank flights for daily ops triage.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd fds_ord
python -m src.run \
  --data_dir ./data \
  --out_dir ./out \
  --your_name "ALOK_SINGH" \
  --class_quantiles 0.2 0.7 \
  --seed 17
