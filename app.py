import numpy as np, pandas as pd, altair as alt, streamlit as st, joblib, os
from core import *

st.set_page_config(page_title='Transaction Risk & Anomaly Analysis', layout='wide')

@st.cache_resource
def load():
    return joblib.load('model.joblib') if os.path.exists('model.joblib') else train()
art = load(); te = art['test']

st.title('Transaction Risk & Anomaly Analysis')
st.sidebar.header('Decision thresholds')
review = st.sidebar.slider('Review above (risk score)', 5, 90, 45)
block = st.sidebar.slider('Block above (risk score)', 10, 99, 90)
if block <= review: st.sidebar.warning('Block threshold should be higher than review.')
m = evaluate(te, review, block)
c = st.columns(5)
c[0].metric('Fraud caught (recall)', f"{m['recall']:.0%}"); c[1].metric('Precision', f"{m['precision']:.0%}")
c[2].metric('Sent to review', f"{m['review_rate']:.1%}"); c[3].metric('Loss prevented', f"₹{m['prevented']:,.0f}")
c[4].metric('Total cost', f"₹{m['cost']:,.0f}")

t1, t2, t3, t4, t5 = st.tabs(['Anomaly analysis', 'Risk management', 'Score a transaction', 'Score a CSV', 'Business analysis'])

with t1:
    a, b = st.columns(2)
    d = te.assign(Outcome=np.where(te.is_fraud == 1, 'Fraud', 'Legitimate'))
    a.subheader('Risk score distribution')
    a.altair_chart(alt.Chart(d).mark_bar(opacity=.8).encode(
        alt.X('risk:Q', bin=alt.Bin(maxbins=25), title='Risk score'), alt.Y('count():Q', scale=alt.Scale(type='sqrt')),
        color='Outcome:N').properties(height=300), use_container_width=True)
    b.subheader('Amount by hour of day')
    d['Decision'] = decide(d.risk, review, block)
    b.altair_chart(alt.Chart(d.sample(min(3000, len(d)), random_state=1)).mark_circle(size=40, opacity=.6).encode(
        alt.X('hour:Q'), alt.Y('amount:Q', scale=alt.Scale(type='log')),
        color=alt.Color('Decision:N', scale=alt.Scale(domain=['Approve', 'Review', 'Block'], range=['#2E8B6A', '#C98A00', '#C92A4B'])),
        tooltip=['customer_id', 'category', 'amount', 'risk']).properties(height=300), use_container_width=True)
    st.subheader('Flag rate by category')
    cat = d.groupby('category').apply(lambda x: (x.Decision != 'Approve').mean(), include_groups=False).sort_values(ascending=False)
    st.bar_chart(cat)
    st.subheader('Novel-pattern detector (Isolation Forest)')
    st.caption('Flags unusual transactions without using fraud labels. Useful for new attack types the supervised model has not seen.')
    st.dataframe(d.sort_values('anomaly', ascending=False).head(15)[['customer_id', 'category', 'amount', 'hour', 'risk', 'anomaly', 'is_fraud']], use_container_width=True)

with t2:
    ths = range(10, 91, 5)
    cost = pd.DataFrame({'Review threshold': list(ths), 'Total cost': [evaluate(te, t, min(99, t + 30))['cost'] for t in ths]})
    st.subheader('Threshold trade-off')
    st.caption('Cost = fraud missed + ₹%d per wrongly blocked customer + ₹%d per review (placeholders to validate with Finance). Block = review + 30.' % (FRICTION, REVIEW_COST))
    st.altair_chart(alt.Chart(cost).mark_line(point=True).encode(x='Review threshold:Q', y='Total cost:Q'), use_container_width=True)
    st.subheader('Risk register')
    st.dataframe(RISKS.sort_values('Rating', ascending=False), use_container_width=True, hide_index=True)
    st.subheader('Model quality')
    st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in art['metrics'].items() if k != 'importance'})

with t3:
    x = st.columns(4)
    amt = x[0].number_input('Amount (₹)', 1.0, 1e6, 25000.0); typ = x[1].number_input('Customer typical spend (₹)', 100.0, 1e5, 3000.0)
    hr = x[2].slider('Hour', 0, 23, 2); cat_ = x[3].selectbox('Category', list(CATS))
    y = st.columns(3)
    v = y[0].slider('Transactions in last hour', 0, 10, 4); fo = y[1].checkbox('Foreign country', True); nd = y[2].checkbox('New device', True)
    row = pd.DataFrame([{'amount': amt, 'amount_z': (amt - typ) / (typ * .25), 'txn_1h': v, 'foreign': int(fo), 'new_device': int(nd),
                         'hour': hr, 'night': int(hr < 5), 'merchant_risk': CATS[cat_]}])
    s = score(art, row).iloc[0]
    st.metric('Risk score', f"{s.risk:.0f} / 100", decide(pd.Series([s.risk]), review, block)[0])
    st.progress(min(1.0, s.risk / 100)); st.caption(f'Novelty (anomaly) score: {s.anomaly:.2f}')

with t4:
    st.write('Upload a CSV with columns: ' + ', '.join(FEATURES))
    st.download_button('Download sample CSV', te[FEATURES].head(50).to_csv(index=False), 'sample.csv')
    f = st.file_uploader('CSV file', type='csv')
    if f:
        u = pd.read_csv(f); miss = [c for c in FEATURES if c not in u]
        if miss: st.error('Missing columns: ' + ', '.join(miss))
        else:
            u = score(art, u); u['decision'] = decide(u.risk, review, block)
            st.dataframe(u.sort_values('risk', ascending=False), use_container_width=True)
            st.download_button('Download scored CSV', u.to_csv(index=False), 'scored.csv')

with t5:
    st.markdown("""
**Problem.** Fraud costs money and trust, but blocking too aggressively loses good customers. Goal: catch most fraud while keeping false declines and analyst workload manageable.

**Success measures.** Recall ≥ 85%, precision ≥ 40%, under 8% of transactions in review, decision under 200 ms.

**Stakeholders.** Fraud operations (review queue), customers (fewer wrongful declines), Finance (fraud loss), Compliance (audit trail, PCI DSS, data privacy), Product (checkout friction).

**Requirements (MoSCoW).** *Must:* real-time scoring, explainable score, approve/review/block bands, adjustable thresholds with cost shown. *Should:* per-customer baseline, analyst feedback loop. *Could:* device graph, step-up authentication. *Won't (v1):* automated chargeback disputes.

**Process.** Transaction → feature checks → risk score → approve / review / block → analyst label → monthly retraining and threshold review.
""")
