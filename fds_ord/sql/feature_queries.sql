WITH
pnr_agg AS (
    SELECT
        pf.company_id,
        pf.flight_number,
        pf.flight_day,
        COUNT(*) AS total_pax,
        SUM(CASE WHEN COALESCE(pf.basic_economy_flag, pf.basic_economy_pax, 0) IN (1, '1', 'Y', 'YES', 'TRUE') THEN 1 ELSE 0 END) AS basic_economy_pax,
        SUM(CASE WHEN COALESCE(pf.is_child, pf.child_flag, 0) IN (1, '1', 'Y', 'YES', 'TRUE') THEN 1 ELSE 0 END) AS is_child_count,
        SUM(CASE WHEN COALESCE(pf.lap_child, pf.lap_child_flag, 0) IN (1, '1', 'Y', 'YES', 'TRUE') THEN 1 ELSE 0 END) AS lap_child_count,
        SUM(CASE WHEN COALESCE(pf.stroller_flag, pf.stroller_indicator, 0) IN (1, '1', 'Y', 'YES', 'TRUE') THEN 1 ELSE 0 END) AS stroller_users
    FROM pnr_flight pf
    GROUP BY pf.company_id, pf.flight_number, pf.flight_day
),
ssr_agg AS (
    SELECT
        pr.company_id,
        pr.flight_number,
        pr.flight_day,
        COUNT(*) AS ssr_count
    FROM pnr_remarks pr
    GROUP BY pr.company_id, pr.flight_number, pr.flight_day
),
bag_agg AS (
    SELECT
        b.company_id,
        b.flight_number,
        b.flight_day,
        SUM(COALESCE(b.bag_count, b.bags, 0)) AS bags_total,
        SUM(COALESCE(b.transfer_bag_count, b.transfer_bags, 0)) AS bags_transfer
    FROM bags b
    GROUP BY b.company_id, b.flight_number, b.flight_day
),
airport_ranked AS (
    SELECT
        a.station_code,
        CASE WHEN UPPER(COALESCE(a.iso_country_code, a.country_iso_code, 'US')) <> 'US' THEN 1 ELSE 0 END AS is_international,
        ROW_NUMBER() OVER (PARTITION BY a.station_code ORDER BY a.station_code) AS rn
    FROM airports a
),
airport_flags AS (
    SELECT
        station_code AS scheduled_arrival_station_code,
        is_international
    FROM airport_ranked
    WHERE rn = 1
),
feat AS (
    SELECT
        f.company_id,
        f.flight_number,
        f.flight_day,
        f.scheduled_departure_station_code,
        f.scheduled_arrival_station_code,
        f.scheduled_departure_datetime_local,
        f.actual_departure_datetime_local,
        f.scheduled_ground_time_minutes AS turn_sched,
        f.minimum_turn_minutes AS turn_min,
        (COALESCE(f.scheduled_ground_time_minutes, 0) - COALESCE(f.minimum_turn_minutes, 0)) AS sched_turn_slack,
        f.total_seats,
        pa.total_pax,
        pa.basic_economy_pax,
        pa.is_child_count,
        pa.lap_child_count,
        pa.stroller_users,
        COALESCE(pa.total_pax, 0) / NULLIF(f.total_seats, 0) AS load_factor,
        COALESCE(pa.basic_economy_pax, 0) / NULLIF(pa.total_pax, 0) AS pct_basic_econ,
        COALESCE(pa.is_child_count, 0) / NULLIF(pa.total_pax, 0) AS pct_children,
        COALESCE(pa.stroller_users, 0) / NULLIF(pa.total_pax, 0) AS stroller_rate,
        ba.bags_total,
        ba.bags_transfer,
        COALESCE(ba.bags_transfer, 0) / NULLIF(ba.bags_total, 0) AS transfer_bag_ratio,
        sa.ssr_count,
        COALESCE(sa.ssr_count, 0) / NULLIF(pa.total_pax, 0) AS ssr_rate,
        af.is_international,
        EXTRACT(HOUR FROM f.scheduled_departure_datetime_local) AS dep_hour,
        EXTRACT(EPOCH FROM (f.actual_departure_datetime_local - f.scheduled_departure_datetime_local)) / 60.0 AS dep_delay_min,
        CASE WHEN EXTRACT(EPOCH FROM (f.actual_departure_datetime_local - f.scheduled_departure_datetime_local)) / 60.0 >= 15 THEN 1 ELSE 0 END AS is_high_delay
    FROM flights f
    LEFT JOIN pnr_agg pa
        ON pa.company_id = f.company_id
       AND pa.flight_number = f.flight_number
       AND pa.flight_day = f.flight_day
    LEFT JOIN ssr_agg sa
        ON sa.company_id = f.company_id
       AND sa.flight_number = f.flight_number
       AND sa.flight_day = f.flight_day
    LEFT JOIN bag_agg ba
        ON ba.company_id = f.company_id
       AND ba.flight_number = f.flight_number
       AND ba.flight_day = f.flight_day
    LEFT JOIN airport_flags af
        ON af.scheduled_arrival_station_code = f.scheduled_arrival_station_code
)
SELECT *
FROM feat;
