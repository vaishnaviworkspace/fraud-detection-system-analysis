# Transaction Risk & Anomaly Analysis

A fraud detection system with a business-analysis layer: a supervised model for known fraud patterns, an Isolation Forest for novel anomalies, a cost-based threshold tool, a risk register, and a Streamlit dashboard.

## Run
See RUN_IN_VSCODE.md for step-by-step VS Code instructions.
```
pip install -r requirements.txt
python train.py          # trains on synthetic data, writes model.joblib and metrics.json
streamlit run app.py
```
Use your own data: `python train.py --csv data.csv` (columns: amount, amount_z, txn_1h, foreign, new_device, hour, night, merchant_risk, is_fraud).

## Results (synthetic data, 3% fraud, 6,000 test rows)
ROC-AUC 0.96, PR-AUC 0.81. At review above 45 and block above 90: about 86% of fraud caught, 29% precision, 7% of transactions sent to review.
Synthetic data was deliberately made noisy (stealthy fraud, genuine large purchases) so results are not unrealistically perfect. Re-test on a real dataset before quoting numbers.

## Web dashboard
`python export_dashboard.py` writes `dashboard.html`: a standalone, shareable dashboard (live stream, anomaly charts, cost curve, risk matrix and register, business analysis) fed by the trained model's real scores and per-signal explanations. Re-run it after retraining.

## What to explain in interviews
- **Two detectors:** supervised model catches known patterns; Isolation Forest flags unusual behaviour without labels.
- **Business trade-off:** thresholds are chosen by total cost (missed fraud + customer friction + analyst time), not accuracy alone.
- **Risk management:** risk register with likelihood x impact ratings, owners and mitigations.
- **Limits:** synthetic data, placeholder cost values, no drift monitoring yet.

## Next steps
Swap in the Kaggle credit card fraud dataset, add SHAP explanations, add drift monitoring, deploy on Streamlit Community Cloud.
