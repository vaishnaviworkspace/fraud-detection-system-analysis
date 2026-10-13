"""python export_dashboard.py -> dashboard.html: the web dashboard fed by your trained model's real scores."""
import json, joblib, numpy as np
from core import FEATURES, CATS

art = joblib.load('model.joblib'); clf = art['clf']
te = art['test'].sample(2000, random_state=1).reset_index(drop=True)
cust = art['test'][art['test'].is_fraud == 0].groupby('customer_id').amount.mean().round(0).to_dict()
te['avg'] = te.customer_id.map(cust).fillna(te.amount.median())
base = clf.predict_proba(te[FEATURES])[:, 1] * 100
neutral = [{'amount_z': 0, 'amount': te.avg}, {'txn_1h': 0}, {'foreign': 0}, {'new_device': 0}, {'hour': 12, 'night': 0}, {'merchant_risk': 0}]
contrib = []
for g in neutral:                       # how many score points disappear if this signal were normal
    x = te[FEATURES].copy()
    for k, v in g.items(): x[k] = v
    contrib.append(np.clip(base - clf.predict_proba(x)[:, 1] * 100, 0, None).round().astype(int))
rows = [[r.customer_id, r.category, round(r.amount), int(r.hour), int(r.txn_1h), int(r.foreign), int(r.new_device),
         int(r.is_fraud), int(r.merchant_risk), round(r.risk, 1), round(r.anomaly, 2), [int(c[i]) for c in contrib]]
        for i, r in enumerate(te.itertuples())]
html = open('dashboard_template.html').read().replace('/*DATA*/[]', json.dumps(rows, separators=(',', ':'))) \
    .replace('/*CUST*/{}', json.dumps({k: float(v) for k, v in cust.items()}, separators=(',', ':')))
open('dashboard.html', 'w').write(html); print('wrote dashboard.html', len(html) // 1024, 'KB')
