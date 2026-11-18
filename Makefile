PY?=MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python3
FILE?=data/default_of_credit_card_clients.xls
SOURCE?=file
FAST?=
test:        ; $(PY) -m pytest -q
db-up:       ; docker compose up -d db
db-load:     ; $(PY) -m creditrisk.db load --path $(FILE)
db-dq:       ; $(PY) -m creditrisk.db dq
leakage:     ; $(PY) -m creditrisk.leakage_checks --source $(SOURCE) --path $(FILE)
imbalance:   ; $(PY) -m creditrisk.imbalance_study --source $(SOURCE) --path $(FILE) $(FAST)
cv:          ; $(PY) -m creditrisk.run_experiments --source $(SOURCE) --path $(FILE) --models lr,lr_binned,rf,xgb,lgbm $(FAST)
interpret:   ; $(PY) -m creditrisk.interpret --source $(SOURCE) --path $(FILE) --models lr,lr_binned,rf,xgb,lgbm --shap-model lgbm $(FAST)
report:      ; $(PY) -m creditrisk.report
all: test leakage imbalance cv interpret report
smoke:       ; $(PY) -m creditrisk.run_experiments --source synthetic --models lr,rf --fast --out /tmp/smoke   # plumbing only
