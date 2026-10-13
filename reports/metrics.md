# Model evaluation (full test set)

Test set: 56,962 transactions, 98 frauds (0.172%).

- **PR-AUC (average precision): 0.859** (a random model scores 0.0017; this is the headline metric for rare-event problems)
- ROC-AUC: 0.9736 (flattered by the 56,000+ easy legitimate transactions, so it is reported but not relied on)

| threshold | frauds caught | recall | precision | false alarms | fraud value caught |
|---|---|---|---|---|---|
| 0.5 | 83 / 98 | 84.7% | 72.8% | 31 | 64.0% |
| 0.8 | 80 / 98 | 81.6% | 83.3% | 16 | 63.1% |
| 0.9 | 80 / 98 | 81.6% | 89.9% | 9 | 63.1% |
| 0.95 | 77 / 98 | 78.6% | 91.7% | 7 | 60.6% |
| rule: amount > 1,012 | 1 / 98 | 1.0% | 0.2% | 599 | 17.0% |

The app blocks at the default 0.5 threshold. Raising the threshold trades recall for fewer false alarms; the right point depends on the cost of a missed fraud vs an annoyed customer.