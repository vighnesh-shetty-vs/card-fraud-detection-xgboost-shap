"""Offline evaluation of the fraud model on the FULL held-out test set.

Same data, split and hyper-parameters as app.py. Writes reports/metrics.md.
Usage: python evaluate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

OUT = Path("reports")


def main() -> None:
    df = pd.read_csv("creditcard.csv")
    X, y = df.drop(["Class", "Time"], axis=1), df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    model = xgb.XGBClassifier(n_estimators=150, scale_pos_weight=ratio, learning_rate=0.06, random_state=42)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    rows = []
    for t in [0.5, 0.8, 0.9, 0.95]:
        pred = p >= t
        tp = int((pred & (y_te == 1)).sum()); fp = int((pred & (y_te == 0)).sum()); fn = int((~pred & (y_te == 1)).sum())
        caught_amt = X_te.loc[pred & (y_te == 1), "Amount"].sum(); fraud_amt = X_te.loc[y_te == 1, "Amount"].sum()
        rows.append({"threshold": t, "frauds caught": f"{tp} / {tp + fn}", "recall": f"{tp / (tp + fn):.1%}",
                     "precision": f"{tp / max(tp + fp, 1):.1%}", "false alarms": fp,
                     "fraud value caught": f"{caught_amt / fraud_amt:.1%}"})
    # Baseline: a rule a bank might start with, block every transaction above the 99th amount percentile
    amt_cut = X_tr["Amount"].quantile(0.99)
    rule = X_te["Amount"] > amt_cut
    tp = int((rule & (y_te == 1)).sum()); fp = int((rule & (y_te == 0)).sum())
    rows.append({"threshold": f"rule: amount > {amt_cut:,.0f}", "frauds caught": f"{tp} / {int(y_te.sum())}",
                 "recall": f"{tp / y_te.sum():.1%}", "precision": f"{tp / max(tp + fp, 1):.1%}", "false alarms": fp,
                 "fraud value caught": f"{X_te.loc[rule & (y_te == 1), 'Amount'].sum() / X_te.loc[y_te == 1, 'Amount'].sum():.1%}"})
    table = pd.DataFrame(rows)

    OUT.mkdir(exist_ok=True)
    lines = [
        "# Model evaluation (full test set)", "",
        f"Test set: {len(y_te):,} transactions, {int(y_te.sum())} frauds ({y_te.mean():.3%}).", "",
        f"- **PR-AUC (average precision): {average_precision_score(y_te, p):.3f}** "
        f"(a random model scores {y_te.mean():.4f}; this is the headline metric for rare-event problems)",
        f"- ROC-AUC: {roc_auc_score(y_te, p):.4f} (flattered by the 56,000+ easy legitimate transactions, so it is reported but not relied on)", "",
        "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns),
        *["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)], "",
        "The app blocks at the default 0.5 threshold. Raising the threshold trades recall for fewer false alarms; "
        "the right point depends on the cost of a missed fraud vs an annoyed customer.",
    ]
    (OUT / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
