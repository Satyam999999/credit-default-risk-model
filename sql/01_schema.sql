-- Raw table: loaded 1:1 from the UCI file, no cleaning. PAY_0 in the UCI file is stored as pay_1.
-- *_1 = most recent month (Sep 2005) ... *_6 = oldest (Apr 2005).
DROP TABLE IF EXISTS raw_credit CASCADE;
CREATE TABLE raw_credit (
    id                  INTEGER PRIMARY KEY,
    limit_bal           DOUBLE PRECISION NOT NULL,
    sex                 SMALLINT,
    education           SMALLINT,
    marriage            SMALLINT,
    age                 SMALLINT,
    pay_1 SMALLINT, pay_2 SMALLINT, pay_3 SMALLINT, pay_4 SMALLINT, pay_5 SMALLINT, pay_6 SMALLINT,
    bill_amt1 DOUBLE PRECISION, bill_amt2 DOUBLE PRECISION, bill_amt3 DOUBLE PRECISION,
    bill_amt4 DOUBLE PRECISION, bill_amt5 DOUBLE PRECISION, bill_amt6 DOUBLE PRECISION,
    pay_amt1  DOUBLE PRECISION, pay_amt2  DOUBLE PRECISION, pay_amt3  DOUBLE PRECISION,
    pay_amt4  DOUBLE PRECISION, pay_amt5  DOUBLE PRECISION, pay_amt6  DOUBLE PRECISION,
    default_next_month  SMALLINT NOT NULL CHECK (default_next_month IN (0, 1))
);
