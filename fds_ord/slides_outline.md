# Problem & Objective
- Predict A14 (>=15 min late) departures so ops can preemptively resource choke flights.

# Data & Grain
- Flight-day level with joins to PNR, remarks, bag transfers, airport metadata (ORD exports).

# Feature Engineering
- sched_turn_slack = scheduled_ground_time_minutes − minimum_turn_minutes.
- load_factor = total_pax ÷ total_seats (clip 0–1).
- transfer_bag_ratio = bags_transfer ÷ bags_total (0 if no bags).
- ssr_rate = ssr_count ÷ total_pax (0 if no pax).
- pct_basic_econ / pct_children / stroller_rate share passenger mix.

# EDA Highlights
- Avg dep delay ~A14 risk = X%; tight turns (turn_sched ≤ turn_min) count Y; transfer bag ratio avg Z; load factor mean/median/p95; SSR rate impact per logistic.

# Model & Calibration
- Calibrated LogisticRegression (5-fold CV) → Brier score ~B, ROC ~R; sigmoid calibration keeps probability honesty.

# Score Normalization, Daily Rank & Classes
- fds_raw = calibrated probability, fds_z = per-day z-score; dense rank per dep_date; quantile bands for Difficult/Medium/Easy using --class_quantiles.

# Top Insights
- Difficult flights cluster on ORD–XXX red-eyes, heavy transfer_bag_ratio >0.4, low sched_turn_slack <15 min.

# Ops Playbook
- Pad turns <15 min or pre-stage cleaners for flagged flights.
- Rebalance SSR-heavy departures with extra gate support.
- Prioritize bag runners for transfer ratio >30% on Difficult class flights.
