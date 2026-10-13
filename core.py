"""Shared logic: synthetic data, model training, decision bands, cost model."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

CATS = {'Grocery': 0, 'Fuel': 0, 'Dining': 0, 'Utilities': 0, 'Travel': 3,
        'Electronics': 8, 'Gift cards': 12, 'Crypto': 14}
FEATURES = ['amount', 'amount_z', 'txn_1h', 'foreign', 'new_device', 'hour', 'night', 'merchant_risk']
FRICTION, REVIEW_COST = 150, 60  # INR per wrongly blocked customer / per manual review (placeholders)


def make_data(n=20000, fraud_rate=0.03, seed=42):
    """Synthetic card transactions. Replace with a real dataset for production claims."""
    r = np.random.default_rng(seed)
    nc = 500
    avg, sd = r.uniform(800, 6800, nc), r.uniform(300, 1800, nc)
    cid = r.integers(0, nc, n)
    fraud = r.random(n) < fraud_rate
    obvious = fraud & (r.random(n) < .6)      # clear fraud patterns
    stealth = fraud & ~obvious                # fraud that mimics normal behaviour
    cats = np.array(list(CATS))
    cat = np.where(fraud & (r.random(n) < .5), r.choice(cats[4:], n), r.choice(cats, n))
    base = np.maximum(30, avg[cid] + (r.random(n) + r.random(n) + r.random(n) - 1.5) * sd[cid] * 1.6)
    legit_big = ~fraud & (r.random(n) < .05)  # genuine large purchases
    amount = np.where(legit_big, avg[cid] * r.uniform(2, 6, n), base)
    amount = np.where(obvious, np.where(r.random(n) < .3, r.uniform(20, 100, n), avg[cid] * r.uniform(2.5, 9.5, n)), amount)
    amount = np.where(stealth, base * r.uniform(1.0, 1.8, n), amount)
    night_legit = r.random(n) < .06
    hour = np.where(obvious, np.where(r.random(n) < .6, r.integers(0, 5, n), r.integers(0, 24, n)),
           np.where(stealth | night_legit, np.where(r.random(n) < .3, r.integers(0, 5, n), r.integers(8, 23, n)), r.integers(8, 23, n)))
    vel_legit = np.floor(r.random(n) * r.random(n) * 4) + (r.random(n) < .04) * r.integers(2, 4, n)
    txn = np.where(obvious, r.integers(2, 8, n), np.where(stealth, vel_legit + (r.random(n) < .3), vel_legit)).astype(int)
    foreign = np.where(obvious, r.random(n) < .7, np.where(stealth, r.random(n) < .25, r.random(n) < .04)).astype(int)
    newdev = np.where(obvious, r.random(n) < .75, np.where(stealth, r.random(n) < .3, r.random(n) < .07)).astype(int)
    df = pd.DataFrame({
        'customer_id': ['C%04d' % i for i in cid], 'category': cat, 'amount': amount.round(2),
        'amount_z': (amount - avg[cid]) / sd[cid],
        'txn_1h': txn,
        'foreign': foreign,
        'new_device': newdev,
        'hour': hour, 'is_fraud': fraud.astype(int)})
    df['night'] = (df.hour < 5).astype(int)
    df['merchant_risk'] = df.category.map(CATS)
    return df


def train(df=None, seed=42):
    """Supervised model (known fraud patterns) + Isolation Forest (novel anomalies)."""
    df = make_data() if df is None else df
    tr, te = train_test_split(df, test_size=.3, stratify=df.is_fraud, random_state=seed)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=.1, class_weight='balanced', random_state=seed)
    clf.fit(tr[FEATURES], tr.is_fraud)
    iso = IsolationForest(n_estimators=200, contamination=.03, random_state=seed).fit(tr[FEATURES])
    raw = -iso.score_samples(tr[FEATURES])
    lo, hi = raw.min(), raw.max()
    art = {'clf': clf, 'iso': iso, 'iso_range': (lo, hi)}
    te = score(art, te)
    m = {'roc_auc': roc_auc_score(te.is_fraud, te.risk / 100), 'pr_auc': average_precision_score(te.is_fraud, te.risk / 100),
         'iso_roc_auc': roc_auc_score(te.is_fraud, te.anomaly), 'train_rows': len(tr), 'test_rows': len(te),
         'fraud_rate': float(df.is_fraud.mean())}
    art['test'], art['metrics'] = te, m
    return art


def score(art, df):
    df = df.copy()
    df['risk'] = art['clf'].predict_proba(df[FEATURES])[:, 1] * 100
    lo, hi = art['iso_range']
    df['anomaly'] = ((-art['iso'].score_samples(df[FEATURES]) - lo) / (hi - lo)).clip(0, 1)
    return df


def decide(risk, review, block):
    return np.where(risk >= block, 'Block', np.where(risk >= review, 'Review', 'Approve'))


def evaluate(df, review, block):
    d = decide(df.risk, review, block)
    fraud, legit = df.is_fraud == 1, df.is_fraud == 0
    flagged = d != 'Approve'
    missed = df.amount[fraud & ~flagged].sum()
    fp_blocks, reviews = int((legit & (d == 'Block')).sum()), int((d == 'Review').sum())
    tp = int((fraud & flagged).sum())
    return {'recall': tp / max(1, fraud.sum()), 'precision': tp / max(1, flagged.sum()),
            'review_rate': reviews / len(df), 'prevented': df.amount[fraud & flagged].sum(), 'missed': missed,
            'false_blocks': fp_blocks, 'reviews': reviews, 'cost': missed + fp_blocks * FRICTION + reviews * REVIEW_COST}


RISKS = pd.DataFrame([
    ['Card-testing bursts of small purchases', 4, 3, 'Velocity rule + auto-block for rapid small charges', 'Fraud ops'],
    ['Account takeover', 3, 5, 'Device and geo checks, step-up authentication', 'Security'],
    ['False declines lose good customers', 4, 4, 'Review band before hard block; weekly decline-rate tracking', 'Product'],
    ['Model drift as fraud tactics change', 3, 4, 'Monthly retraining and threshold review with analyst labels', 'Data science'],
    ['Privacy and compliance gaps (PCI DSS, DPDP Act)', 2, 5, 'Data minimisation, access logs, audit trail', 'Compliance'],
    ['Review queue backlog', 3, 3, 'Prioritise queue by score; SLA alerts', 'Fraud ops']],
    columns=['Risk', 'Likelihood (1-5)', 'Impact (1-5)', 'Mitigation', 'Owner'])
RISKS['Rating'] = RISKS['Likelihood (1-5)'] * RISKS['Impact (1-5)']
