"""python train.py [--csv your_data.csv]  -> trains models, saves model.joblib and metrics.json"""
import argparse, json, joblib, pandas as pd
from sklearn.inspection import permutation_importance
from core import train, FEATURES, evaluate

ap = argparse.ArgumentParser(); ap.add_argument('--csv'); a = ap.parse_args()
df = pd.read_csv(a.csv) if a.csv else None  # needs FEATURES columns + is_fraud
art = train(df)
te = art['test']
pi = permutation_importance(art['clf'], te[FEATURES], te.is_fraud, scoring='average_precision', n_repeats=5, random_state=0)
art['metrics']['importance'] = dict(sorted(zip(FEATURES, pi.importances_mean.round(4).tolist()), key=lambda x: -x[1]))
art['metrics']['at_review45_block90'] = {k: float(v) for k, v in evaluate(te, 45, 90).items()}
joblib.dump(art, 'model.joblib')
json.dump(art['metrics'], open('metrics.json', 'w'), indent=2)
print(json.dumps(art['metrics'], indent=2))
