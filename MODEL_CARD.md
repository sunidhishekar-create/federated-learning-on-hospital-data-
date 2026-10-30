# Model Card: Hospital Readmission MLP (centralized and federated)

## Model details
- **Type:** feed-forward neural network (PyTorch), 182 → 64 → 32 → 1, ReLU, dropout 0.2,
  13,825 parameters. Output is a logit; the probability is the sigmoid of the logit.
- **Task:** predict readmission within 30 days of discharge for diabetic inpatients.
- **Two training modes, same architecture and hyperparameters:**
  1. **Centralized:** one model trained on the union of the five hospitals' training splits.
  2. **Federated:** FedAvg with Flower, 8 rounds, 1 local epoch per round, five clients
     (one per simulated hospital). Raw data never leaves a client. (<Name 2>)
- **Training setup:** Adam, learning rate 0.001, batch size 64, `BCEWithLogitsLoss` with
  `pos_weight` = negatives / positives (computed on the combined data for centralized
  training, on each client's own data for federated training). Decision threshold 0.5.
- **Training budget:** 8 epochs (centralized) / 8 rounds (federated), selected on the pooled
  validation sets, because longer training overfits (see `notebooks/03_epoch_selection.ipynb`).
- **Seeds:** results are reported over `eval_seeds` = 42 to 46.
- **Code:** `src/model.py`, `src/centralized.py`, `src/federated/`.

## Intended use
Course project comparing centralized and federated training under non-IID hospital data.
**Not for clinical decisions.** The model is mediocre (ROC-AUC 0.68), was trained on US data
from 1999 to 2008, and has not been validated on any real hospital.

## Training and evaluation data
See `DATA_CARD.md`. Diabetes 130-US Hospitals (UCI, CC BY 4.0). 99,340 encounters after
cleaning, 182 features. Global test set: 19,773 encounters (20% of patients, split by patient).
Five **simulated** hospitals hold the remaining 79,567 encounters, with 10% of each hospital
held out as its validation set. Hospitals differ in positive rate (4.8% to 24.8%), age mix and
size (9,026 to 23,965 encounters).

## Performance: global test set
| Metric | Centralized (mean ± std, 5 seeds) | Federated (mean ± std) |
|---|---|---|
| ROC-AUC | 0.6778 ± 0.0008 | TBD |
| Recall | 0.6217 ± 0.0475 | TBD |
| Precision | 0.1767 ± 0.0070 | TBD |
| F1 | 0.2746 ± 0.0032 | TBD |
| Accuracy | 0.6268 ± 0.0338 | TBD |
| Training time | 19.4 ± 1.3 s | TBD |
| Communication | none | TBD (expected about 4.4 MB over 8 rounds) |

ROC-AUC is the headline metric because it does not depend on the threshold. Recall and
accuracy shift by ±0.03 to 0.05 between seeds because the 0.5 threshold interacts with class
weights, so single runs of these should not be compared. Precision is low because only 11.4%
of cases are positive.

## Performance by hospital (validation sets, mean ± std over 5 seeds)
| Hospital | Positive rate | ROC-AUC (centralized) | Recall (centralized) | ROC-AUC (federated) | Recall (federated) |
|---|---|---|---|---|---|
| 0 | 22.4% | 0.619 ± 0.007 | 0.591 ± 0.060 | TBD | TBD |
| 1 | 6.8% | 0.698 ± 0.006 | 0.668 ± 0.054 | TBD | TBD |
| 2 | 24.8% | 0.625 ± 0.004 | 0.654 ± 0.038 | TBD | TBD |
| 3 | 8.4% | 0.623 ± 0.019 | 0.525 ± 0.045 | TBD | TBD |
| 4 | 4.8% | 0.665 ± 0.009 | 0.600 ± 0.043 | TBD | TBD |

Per-site ROC-AUC gap (max minus min), centralized: 0.083 ± 0.009. Federated: TBD.

## Fairness considerations
- Fairness is assessed **per hospital**, with ROC-AUC and recall as main metrics. Precision
  and F1 mostly reflect each hospital's base rate. Hospital 1 is served best by the
  centralized model; its lead over hospitals 0, 2 and 3 is large relative to seed noise, but
  borderline relative to validation-set sampling noise (single-run bootstrap intervals clear
  of hospitals 0 and 2 by only 0.002).
- Hospital 3's validation set has 56 positive cases, so its estimates are the least reliable.
- **Not assessed:** performance by race, gender or age group. Race and gender are model
  inputs. Any real use would need that analysis.
- Whether federated training narrows or widens the per-site gap: TBD.

## Limitations and risks
- Hospitals are simulated from one population. The skew is artificial and understates real
  differences between sites.
- Data from 1999 to 2008, US only, diabetic inpatients only. Readmissions to other hospitals
  are not observed, so labels are incomplete.
- Low precision means most flagged patients would not be readmitted.
- No differential privacy or secure aggregation, so the federated setup shown here gives no
  formal privacy guarantee.
- Hyperparameters were not tuned beyond the epoch count.

## Reproduce
See `README.md`: `python -m src.prepare`, `python -m src.centralized`,
`python -m src.multiseed`, then the federated run described there.
