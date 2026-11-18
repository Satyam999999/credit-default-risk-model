-- v_clean: fixed-rule cleaning (no statistics learned from data, so safe before any split).
--   * drop exact duplicates (same values in every column except id; keep lowest id)
--   * education outside 1-4 -> 4 (other); marriage outside 1-3 -> 3 (other)
CREATE OR REPLACE VIEW v_clean AS
WITH ranked AS (
    SELECT r.*,
           ROW_NUMBER() OVER (
               PARTITION BY limit_bal, sex, education, marriage, age,
                            pay_1, pay_2, pay_3, pay_4, pay_5, pay_6,
                            bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                            pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6,
                            default_next_month
               ORDER BY id) AS rn
    FROM raw_credit r
)
SELECT id, limit_bal, sex,
       CASE WHEN education IN (1, 2, 3, 4) THEN education ELSE 4 END AS education,
       CASE WHEN marriage  IN (1, 2, 3)    THEN marriage  ELSE 3 END AS marriage,
       age,
       pay_1, pay_2, pay_3, pay_4, pay_5, pay_6,
       bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
       pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6,
       default_next_month
FROM ranked
WHERE rn = 1;

-- v_monthly: one row per customer-month. t = 1 (oldest, Apr) ... 6 (newest, Sep); UCI column k -> t = 7 - k.
CREATE OR REPLACE VIEW v_monthly AS
SELECT c.id, c.limit_bal, c.default_next_month,
       m.t, m.pay_status, m.bill_amt, m.pay_amt
FROM v_clean c
CROSS JOIN LATERAL (VALUES
    (1, c.pay_6::INT, c.bill_amt6::DOUBLE PRECISION, c.pay_amt6::DOUBLE PRECISION),
    (2, c.pay_5::INT, c.bill_amt5::DOUBLE PRECISION, c.pay_amt5::DOUBLE PRECISION),
    (3, c.pay_4::INT, c.bill_amt4::DOUBLE PRECISION, c.pay_amt4::DOUBLE PRECISION),
    (4, c.pay_3::INT, c.bill_amt3::DOUBLE PRECISION, c.pay_amt3::DOUBLE PRECISION),
    (5, c.pay_2::INT, c.bill_amt2::DOUBLE PRECISION, c.pay_amt2::DOUBLE PRECISION),
    (6, c.pay_1::INT, c.bill_amt1::DOUBLE PRECISION, c.pay_amt1::DOUBLE PRECISION)
) AS m(t, pay_status, bill_amt, pay_amt);
