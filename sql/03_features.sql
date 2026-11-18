-- Month-level building blocks (window function: LAG for the previous statement).
-- Assumption: the payment made in month t is paid toward the statement of month t-1.
--   pay_ratio   = share of previous bill paid, capped to [0, 1]; NULL when the previous bill is <= 0.
--   delay_months= max(status, 0): statuses -2 / -1 / 0 all mean "no delay"; 1..9 = months of delay.
--   utilization = max(bill, 0) / credit limit (a negative bill is a credit balance -> 0).
CREATE OR REPLACE VIEW v_monthly_enriched AS
SELECT id, t, limit_bal, default_next_month, pay_status, bill_amt, pay_amt,
       GREATEST(pay_status, 0)                       AS delay_months,
       CASE WHEN pay_status >= 1 THEN 1 ELSE 0 END   AS is_delinquent,
       GREATEST(bill_amt, 0) / NULLIF(limit_bal, 0)  AS utilization,
       CASE WHEN LAG(bill_amt) OVER w > 0
            THEN LEAST(GREATEST(pay_amt / LAG(bill_amt) OVER w, 0), 1)
       END                                           AS pay_ratio
FROM v_monthly
WINDOW w AS (PARTITION BY id ORDER BY t);

-- Customer-level features via window aggregates over each customer's own months only.
-- The full-partition frame is repeated on every row; we keep the newest month (t = 6).
CREATE OR REPLACE VIEW v_model_features AS
WITH w AS (
    SELECT e.id, e.t,
           AVG(e.pay_ratio)                 OVER p_all   AS avg_pay_ratio,
           REGR_SLOPE(e.pay_ratio, e.t)     OVER p_all   AS pay_ratio_trend,   -- >0: paying a larger share over time
           AVG(e.pay_ratio)                 OVER p_last3 AS pay_ratio_last3,
           SUM(e.is_delinquent)             OVER p_all   AS n_months_delinquent,
           SUM(e.is_delinquent)             OVER p_last3 AS n_delinquent_last3,
           MAX(e.delay_months)              OVER p_all   AS max_delay,
           REGR_SLOPE(e.delay_months, e.t)  OVER p_all   AS delay_trend,       -- >0: delays getting worse
           AVG(e.utilization)               OVER p_all   AS avg_utilization,
           e.utilization                                 AS utilization_latest,
           REGR_SLOPE(e.utilization, e.t)   OVER p_all   AS utilization_trend
    FROM v_monthly_enriched e
    WINDOW p_all   AS (PARTITION BY e.id ORDER BY e.t
                       ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
           p_last3 AS (PARTITION BY e.id ORDER BY e.t
                       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
)
SELECT c.id, c.sex, c.education, c.marriage, c.limit_bal,
       c.bill_amt1, c.bill_amt2, c.bill_amt3, c.bill_amt4, c.bill_amt5, c.bill_amt6,
       c.pay_amt1, c.pay_amt2, c.pay_amt3, c.pay_amt4, c.pay_amt5, c.pay_amt6,
       c.age,
       c.pay_1, c.pay_2, c.pay_3, c.pay_4, c.pay_5, c.pay_6,
       w.avg_pay_ratio, w.pay_ratio_trend, w.pay_ratio_last3,
       w.n_months_delinquent::INT AS n_months_delinquent,
       w.n_delinquent_last3::INT  AS n_delinquent_last3,
       w.max_delay, w.delay_trend,
       w.avg_utilization, w.utilization_latest, w.utilization_trend,
       c.default_next_month
FROM v_clean c
JOIN w ON w.id = c.id AND w.t = 6;
