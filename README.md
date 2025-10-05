# fds_ord

Feature-driven scoring for ORD departures. We flag A14 departures (>=15 minutes late) by blending
turn-time slack, passenger mix, SSR load, and bag transfer signals into a calibrated probability and
simple daily tiers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data inputs

Drop the raw CSV exports in `data/` with the exact filenames:

- `Flight Level Data.csv`
- `PNR+Flight+Level+Data.csv`
- `PNR Remark Level Data.csv`
- `Bag+Level+Data.csv`
- `Airports Data.csv`

## Run the pipeline

```bash
python -m src.run \
  --data_dir ./data \
  --out_dir ./out \
  --your_name "First Last" \
  --class_quantiles 0.2 0.7 \
  --seed 17
```

## Outputs (`out/`)

- `test_<your_name>.csv` — one flight per row with calibrated `fds_raw`, per-day z-score, rank, and
  `daily_class`. Key drivers (`sched_turn_slack`, `transfer_bag_ratio`, `load_factor`, `ssr_rate`) are
  carried along for triage.
- `feature_importance.csv` — absolute logistic coefficients to show which signals dominated.
- `eda_summary.md` — markdown recap of delay rates, tight turns, mix ratios, and regression effect.
- `*_png` charts — histogram, tight-turn bar chart, transfer bag violin, and load-factor scatter.

## Adjusting class bands

`--class_quantiles LOW HIGH` controls the breakpoints used for the daily classes. Lower `LOW`
produces a narrower “Difficult” band; higher `HIGH` shrinks “Easy”. Both must be in `(0,1)` and
increasing.

## Known limitations / future work

- Weather, ATC initiatives, and runway-level congestion are not modeled yet.
- Model is tuned for a rolling ~2 week ORD departure horizon; other airports need retraining.
- Inputs assume ORD-specific extracts; broader scope will require schema extensions.
