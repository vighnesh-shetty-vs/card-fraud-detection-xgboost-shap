import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import altair as alt
import time
from sklearn.model_selection import train_test_split

# --- 1. System Setup ---
st.set_page_config(page_title="Fintech Risk Command Center", page_icon="🏦", layout="wide")

@st.cache_resource
def load_enterprise_model():
    df = pd.read_csv('creditcard.csv')
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    model = xgb.XGBClassifier(n_estimators=150, scale_pos_weight=ratio, learning_rate=0.06)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_test, y_test

model, explainer, X_test, y_test = load_enterprise_model()

# --- 2. Session State Initialization ---
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Time", "Amount", "Risk %", "Action", "Risk Factor", "SHAP", "Fraudulent", "Actual"])

# --- 3. Header & Metrics ---
st.title("🏦 Fintech Risk Command Center")
st.markdown("#### Real-Time Fraud Mitigation & Audit Trail")


# --- 4. The Monitoring Engine ---
@st.fragment
def start_monitoring():
    # Metrics live inside the fragment so its reruns can update them
    m1, m2, m3, m4 = st.columns(4)
    v_total = m1.empty()
    v_fraud = m2.empty()
    v_savings = m3.empty()
    v_status = m4.empty()
    v_total.metric("Session Volume", "0")
    v_fraud.metric("Flagged Threats", "0")
    v_savings.metric("Fraud Value Blocked", "0.00")
    v_status.metric("System Health", "READY")

    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.subheader("📡 Live Transaction Stream")
        table_placeholder = st.empty()
    
    with col_right:
        st.subheader("📈 Threat Pattern Analysis")
        chart_placeholder = st.empty()
        pattern_placeholder = st.empty()

    # Move Archive to a specific container that updates with the fragment
    st.divider()
    st.subheader("📂 Complete Transaction Archive")
    st.caption("Full audit trail for regulatory compliance (Yes/No flags for business review).")
    archive_placeholder = st.empty()

    if st.button("🚀 DEPLOY RISK AGENT", type="primary", use_container_width=True):
        # Create a demonstration pool
        pool_f = X_test[y_test == 1].sample(20)
        pool_n = X_test[y_test == 0].sample(200)
        sim_pool = pd.concat([pool_f, pool_n]).sample(frac=1)

        for i in range(len(sim_pool)):
            row = sim_pool.iloc[[i]]
            start = time.time()
            pred = model.predict(row)[0]
            prob = model.predict_proba(row)[0][1]
            latency = (time.time() - start) * 1000

            # Risk factor = feature with the largest SHAP push towards fraud for this transaction.
            # V1-V28 are anonymised PCA components in the source dataset, so they keep their names.
            shap_row = explainer.shap_values(row)[0]
            top_idx = int(np.argmax(shap_row))
            reason = row.columns[top_idx]
            amt = float(row['Amount'].iloc[0])
            actual = int(y_test.loc[row.index[0]])

            # --- Business Logic: Yes/No Conversion ---
            is_fraud_label = "Yes" if pred == 1 else "No"

            new_entry = {
                "Time": i,
                "Amount": amt,
                "Risk %": f"{prob:.1%}",
                "Action": "🚫 BLOCK" if pred == 1 else "✅ CLEAR",
                "Risk Factor": reason if pred == 1 else "N/A",
                "SHAP": round(float(shap_row[top_idx]), 2) if pred == 1 else None,
                "Fraudulent": is_fraud_label,
                "Actual": "Fraud" if actual == 1 else "Legit",
                "IsFraudNumeric": int(pred) # Hidden for charting
            }
            
            st.session_state.history_df = pd.concat([pd.DataFrame([new_entry]), st.session_state.history_df]).reset_index(drop=True)
            
            # Update Metrics
            history = st.session_state.history_df
            total_tx = len(history)
            fraud_tx = history[history['Fraudulent'] == "Yes"].shape[0]
            # Only true positives count as prevented losses; false alarms don't.
            caught = history[(history['Fraudulent'] == "Yes") & (history['Actual'] == "Fraud")]
            saved = caught['Amount'].sum()

            v_total.metric("Session Volume", f"{total_tx}")
            v_fraud.metric("Flagged Threats", f"{fraud_tx}", delta=f"{len(caught)} confirmed fraud", delta_color="off")
            v_savings.metric("Fraud Value Blocked", f"{saved:,.2f}")
            v_status.metric("System Health", "ONLINE", delta=f"{latency:.1f}ms")

            # Update Live Table
            table_placeholder.dataframe(
                history.head(10).drop(columns=['IsFraudNumeric']).style.map(
                    lambda x: 'color: #ff4b4b; font-weight: bold' if x == "🚫 BLOCK" else 'color: #00cc96', 
                    subset=['Action']
                ), use_container_width=True, hide_index=True
            )

            # Update Archive (Now Visible!)
            archive_placeholder.dataframe(history.drop(columns=['IsFraudNumeric']), use_container_width=True, hide_index=True)

            # Update Charts
            trend_chart = alt.Chart(history.sort_values("Time")).mark_line(color="#ff4b4b").encode(
                x='Time:Q', y=alt.Y('IsFraudNumeric:Q', title='Fraud Detected')
            ).properties(height=180)
            chart_placeholder.altair_chart(trend_chart, use_container_width=True)

            if fraud_tx > 0:
                pattern_df = history[history['Fraudulent'] == "Yes"]['Risk Factor'].value_counts().reset_index()
                pattern_df.columns = ['Factor', 'Count']
                pattern_chart = alt.Chart(pattern_df).mark_bar(color='#1E88E5').encode(
                    x='Count:Q', y=alt.Y('Factor:N', sort='-x')
                ).properties(height=180)
                pattern_placeholder.altair_chart(pattern_chart, use_container_width=True)

            time.sleep(0.1)

start_monitoring()