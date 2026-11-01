# Model Card: Hospital Readmission MLP (centralized and federated)

## Model details
- **Type:** feed-forward neural network (PyTorch), 182 → 64 → 32 → 1, ReLU, dropout 0.2,
  13,825 parameters. Output is a logit; the probability is the sigmoid of the logit.
- **Task:** predict readmission within 30 days of discharge for diabetic inpatients.
- **Two training modes, same architecture and hyperparameters:**
  1. **Centralized:** one model trained on the union of the five hospitals' training splits.
  2. **Federated:** FedAvg with Flower 1.42, 8 rounds, 1 local epoch per round, five clients
     (one per simulated hospital). The clients run as five isolated Docker containers, or in
     one process with Flower's simulation runtime. Raw data never leaves a client; only model
     weights and metric values are exchanged. (Sonia Sharma)
- **Training setup:** Adam, learning rate 0.001, batch size 64, `BCEWithLogitsLoss` with
  `pos_weight` = negatives / positives (computed on the combined data for centralized
  training, on each client's own data for federated training). Decision threshold 0.5.
  The optimizer is rebuilt in every local training call, so Adam's state is not carried
  across federated rounds.
- **Training budget:** 8 epochs (centralized) / 8 rounds of 1 local epoch (federated),
  selected on the pooled validation sets, because longer training overfits (see
  `notebooks/03_epoch_selection.ipynb`).
- **Initialization:** both modes call `torch.manual_seed(seed)` before building the model,
  so they start from identical weights.
- **Seeds:** results are reported over `eval_seeds` = 42 to 46.
- **Code:** `src/model.py`, `src/centralized.py`, `src/federated/`, `docker-compose.yml`.

## Intended use
Course project comparing centralized and federated training under non-IID hospital data.
**Not for clinical decisions.** The model is mediocre (ROC-AUC 0.67 to 0.68), was trained on
US data from 1999 to 2008, and has not been validated on any real hospital.

## Training and evaluation data
See `DATA_CARD.md`. Diabetes 130-US Hospitals (UCI, CC BY 4.0). 99,340 encounters after
cleaning, 182 features. Global test set: 19,773 encounters (20% of patients, split by patient).
Five **simulated** hospitals hold the remaining 79,567 encounters, with 10% of each hospital
held out as its validation set. Hospitals differ in positive rate (4.8% to 24.8%), age mix and
size (9,026 to 23,965 encounters).

## Performance: global test set
Mean ± std over 5 seeds, all on the same Mac (Apple Silicon, CPU only), so times are comparable.

| Metric | Centralized | Federated (simulation) | Federated (Docker, 6 containers) |
|---|---|---|---|
| ROC-AUC | 0.6777 ± 0.0023 | 0.6696 ± 0.0012 | 0.6699 ± 0.0012 |
| Recall | 0.6229 ± 0.0396 | 0.6405 ± 0.0562 | 0.6386 ± 0.0597 |
| Precision | 0.1768 ± 0.0057 | 0.1709 ± 0.0072 | 0.1716 ± 0.0091 |
| F1 | 0.2750 ± 0.0031 | 0.2692 ± 0.0041 | 0.2697 ± 0.0057 |
| Accuracy | 0.6271 ± 0.0281 | 0.6047 ± 0.0418 | 0.6067 ± 0.0464 |
| Training time | 5.8 ± 0.5 s | 33.8 ± 8.4 s | 115.7 ± 20.4 s (216.8 s end to end) |
| Communication | none | 4.49 MB of weights (6.73 MB incl. evaluation broadcast) | 12.3 ± 0.1 MB measured on the network |

- **Accuracy cost of federation.** Federated − centralized ROC-AUC is −0.0081, 95% CI
  [−0.0132, −0.0030], p = 0.001 (paired bootstrap, 1,000 draws, averaged over seeds;
  `src/compare.py`). The cost is small but real.
- **Recall.** The higher federated recall (+0.018, p = 0.002) is a threshold effect: the averaged
  model predicts positive more often. It is not a better model.
- **Docker vs simulation.** Same code; the two agree to within 0.0006 ROC-AUC per seed.
- **Communication.** The weight volume matches the expected 4.4 MB: 56 KB × 10 transfers per
  round × 8 rounds. Docker adds app delivery in round 1, HTTP overhead, polling and heartbeats.
- **Time.** Federated training is slower because of coordination overhead, not extra computation.
- **Windows baseline.** Sunidhi's original centralized run on Windows gives ROC-AUC
  0.6778 ± 0.0008 and 19.4 ± 1.3 s (`results/centralized_seeds.json`).

ROC-AUC is the headline metric because it does not depend on the threshold. Recall and
accuracy shift by ±0.03 to 0.05 between seeds because the 0.5 threshold interacts with class
weights, so single runs of these should not be compared. Precision is low because only 11.4%
of cases are positive.

## Performance by hospital (validation sets, mean ± std over 5 seeds)
Same Mac for both modes; federated values from the simulation runs.

| Hospital | Positive rate | ROC-AUC (centralized) | Recall (centralized) | ROC-AUC (federated) | Recall (federated) | Federated − centralized ROC-AUC [95% CI], p |
|---|---|---|---|---|---|---|
| 0 | 22.4% | 0.616 ± 0.007 | 0.591 ± 0.056 | 0.600 ± 0.002 | 0.590 ± 0.071 | −0.016 [−0.034, +0.002], 0.078 |
| 1 | 6.8% | 0.697 ± 0.006 | 0.663 ± 0.029 | 0.690 ± 0.004 | 0.676 ± 0.037 | −0.006 [−0.025, +0.009], 0.478 |
| 2 | 24.8% | 0.627 ± 0.007 | 0.651 ± 0.032 | 0.602 ± 0.002 | 0.652 ± 0.043 | **−0.025** [−0.040, −0.010], 0.001 |
| 3 | 8.4% | 0.619 ± 0.015 | 0.521 ± 0.027 | 0.644 ± 0.004 | 0.571 ± 0.049 | +0.025 [−0.003, +0.056], 0.078 |
| 4 | 4.8% | 0.665 ± 0.008 | 0.584 ± 0.076 | 0.655 ± 0.003 | 0.577 ± 0.066 | −0.010 [−0.029, +0.009], 0.338 |

Per-site ROC-AUC gap (max minus min, mean ± std over seeds):
- centralized: 0.086 ± 0.009 (0.083 ± 0.009 on Windows);
- federated: 0.090 ± 0.004 (simulation) and 0.089 ± 0.002 (Docker).

The federated − centralized difference in the gap is +0.010, 95% CI [−0.049, +0.046], p = 0.75.

## Fairness considerations
- **How fairness is measured.** Fairness is assessed **per hospital**, with ROC-AUC and recall
  as main metrics. Precision and F1 mostly reflect each hospital's base rate. Per-hospital tests
  use a Bonferroni threshold of 0.01 (5 hospitals).
- **The gap does not change.** Federated training neither widens nor narrows the per-site gap
  compared with centralized training (p = 0.75). The worst-off hospital is hospital 0 in both,
  and its ROC-AUC does not change significantly (−0.016, p = 0.56).
- **Hospital 2 loses under federation.** Hospital 2 is the only one measurably worse off under
  FedAvg (−0.025, p = 0.001), and this survives age adjustment. It has the highest
  readmission rate (24.8%), so its local class weight differs most from the global one. That is
  a likely cause, but it has not been tested.
- **Hospital 1 is served best** by both the centralized and the federated model. Its lead over hospitals 0, 2 and 3 is large
  relative to seed noise but close to validation-set sampling noise.
- **Simulation artifact.** The partitioning creates hospital-specific age-to-label shortcuts.
  In hospital 3, only 0.7% of 50-69 year olds are readmitted. A model trained on hospital 3 alone
  exploits this (ROC-AUC 0.889 on its own data, 0.556 on the global test set). Per-site results
  were therefore also checked with an age-adjusted ROC-AUC, which only compares patients in
  the same age band (`results/comparison.json`).
- **Reason to join.** Compared with each hospital training alone, the federated model is
  better on the global test set for every hospital (+0.005 to +0.113 ROC-AUC, all p ≤ 0.05).
  On a hospital's own patients, after age adjustment, it is not significantly different.
- **Small validation sets.** Hospital 3's validation set has 56 positive cases, so its
  estimates are the least reliable.
- **Not assessed:** performance by race, gender or age group as protected attributes. Race and
  gender are model inputs. Any real use would need that analysis.

## Limitations and risks
- Hospitals are simulated from one population. The skew is artificial and understates real
  differences between sites, and the partitioning creates age shortcuts (see above).
- Data from 1999 to 2008, US only, diabetic inpatients only. Readmissions to other hospitals
  are not observed, so labels are incomplete.
- Low precision means most flagged patients would not be readmitted.
- No differential privacy or secure aggregation, so the federated setup shown here gives no
  formal privacy guarantee. Model weights can still leak information about training data.
- The Docker deployment runs all containers on one machine. Real network latency between
  hospitals is not modeled.
- Hyperparameters were not tuned beyond the epoch count. The same budget is used for every
  hospital. Only FedAvg was tried.

## Reproduce
See `README.md`:
- **Data and centralized baseline:** `python -m src.prepare`, `python -m src.centralized`,
  `python -m src.multiseed`.
- **Federated in Docker:** `python docker/run_experiment.py`.
- **Federated simulation:** `python -m src.federated.run`.
- **Local-only baseline and comparison:** `python -m src.local_only`, then `python -m src.compare`.
