-- Data-quality checks on the RAW table; mirrors creditrisk.cleaning.data_quality_report.
-- Pairing: payment in month k is paid toward the statement of the older month k+1.
SELECT 'total rows' AS check_name, COUNT(*) AS n_rows FROM raw_credit
UNION ALL SELECT 'default rows (target = 1)', COUNT(*) FROM raw_credit WHERE default_next_month = 1
UNION ALL SELECT 'education outside 1-4', COUNT(*) FROM raw_credit WHERE education NOT IN (1,2,3,4)
UNION ALL SELECT 'marriage outside 1-3', COUNT(*) FROM raw_credit WHERE marriage NOT IN (1,2,3)
UNION ALL SELECT 'any pay status = -2', COUNT(*) FROM raw_credit
    WHERE -2 IN (pay_1, pay_2, pay_3, pay_4, pay_5, pay_6)
UNION ALL SELECT 'any pay status = 0', COUNT(*) FROM raw_credit
    WHERE 0 IN (pay_1, pay_2, pay_3, pay_4, pay_5, pay_6)
UNION ALL SELECT 'any pay status >= 8', COUNT(*) FROM raw_credit
    WHERE GREATEST(pay_1, pay_2, pay_3, pay_4, pay_5, pay_6) >= 8
UNION ALL SELECT 'duplicate rows (excluding first copy)', COUNT(*) FROM (
    SELECT ROW_NUMBER() OVER (
        PARTITION BY limit_bal, sex, education, marriage, age,
                     pay_1, pay_2, pay_3, pay_4, pay_5, pay_6,
                     bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                     pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6,
                     default_next_month ORDER BY id) AS rn
    FROM raw_credit) d WHERE rn > 1
UNION ALL SELECT 'any negative bill', COUNT(*) FROM raw_credit
    WHERE LEAST(bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6) < 0
UNION ALL SELECT 'any bill > 1.5x limit', COUNT(*) FROM raw_credit
    WHERE GREATEST(bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6) > 1.5 * limit_bal
UNION ALL SELECT 'any payment > limit', COUNT(*) FROM raw_credit
    WHERE GREATEST(pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6) > limit_bal
UNION ALL SELECT 'delay flagged but previous bill <= 0', COUNT(*) FROM raw_credit
    WHERE (pay_1 >= 1 AND bill_amt2 <= 0) OR (pay_2 >= 1 AND bill_amt3 <= 0) OR (pay_3 >= 1 AND bill_amt4 <= 0)
       OR (pay_4 >= 1 AND bill_amt5 <= 0) OR (pay_5 >= 1 AND bill_amt6 <= 0)
UNION ALL SELECT 'status 0 but zero payment vs positive previous bill', COUNT(*) FROM raw_credit
    WHERE (pay_1 = 0 AND pay_amt1 = 0 AND bill_amt2 > 0) OR (pay_2 = 0 AND pay_amt2 = 0 AND bill_amt3 > 0)
       OR (pay_3 = 0 AND pay_amt3 = 0 AND bill_amt4 > 0) OR (pay_4 = 0 AND pay_amt4 = 0 AND bill_amt5 > 0)
       OR (pay_5 = 0 AND pay_amt5 = 0 AND bill_amt6 > 0)
UNION ALL SELECT 'delay jumps from <=0 to >=2 in one month', COUNT(*) FROM raw_credit
    WHERE (pay_1 >= 2 AND pay_2 <= 0) OR (pay_2 >= 2 AND pay_3 <= 0) OR (pay_3 >= 2 AND pay_4 <= 0)
       OR (pay_4 >= 2 AND pay_5 <= 0) OR (pay_5 >= 2 AND pay_6 <= 0)
UNION ALL SELECT 'age outside 21-79', COUNT(*) FROM raw_credit WHERE age NOT BETWEEN 21 AND 79;
