# Data Card: Simulated Hospital Readmission Data

## Source
[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008),
UCI Machine Learning Repository. License: CC BY 4.0. Original study: Strack et al., "Impact
of HbA1c Measurement on Hospital Readmission Rates", BioMed Research International, 2014.
Records are inpatient diabetic encounters at 130 US hospitals and delivery networks,
1999-2008. The raw file has 101,766 encounters and 50 columns.

## Task
Binary classification: is the patient readmitted within 30 days (`readmitted == "<30"`)?
11.4% of cleaned encounters are positive.

## Preprocessing
- Dropped `weight` (97% missing) and `payer_code` (40% missing); `?` treated as missing.
- Dropped encounters ending in death or hospice (discharge codes 11, 13, 14, 19, 20, 21)
  and 3 rows with unknown gender: 101,766 to 99,340 rows.
- ICD-9 diagnosis codes grouped into broad categories; `medical_specialty` reduced to the top
  10 values plus "Other"; admission, discharge and source codes treated as categories.
- Two constant columns removed. Age brackets converted to midpoints.
- One-hot encoding and standard scaling fitted on the training pool only: 182 features.

## Splits
- Global test set: 20% of patients (19,773 encounters), split by `patient_nbr` so no patient
  appears in both train and test.
- Remaining 79,567 encounters form the pool, divided among five simulated hospitals;
  each holds out 10% of its own patients for validation.

## Simulated hospitals
The dataset has no hospital identifier, so the five hospitals are **simulated**: patients are
assigned by age band and readmission label using a Dirichlet draw (alpha = 1.0, seed 42),
which creates feature skew (age mix), label skew (4.8% to 24.8% positive) and quantity skew
(9,026 to 23,965 encounters). See `results/hospital_partition.csv`. The skew is artificial and
not an estimate of how real hospitals differ.

## Sensitive attributes and risks
Age, race and gender are present. Race and gender are used as model inputs; they are not
used to form the hospitals. No fairness analysis by race or gender was done, only per
hospital. Data are de-identified but concern health records.

## Limitations
- Data are from 1999 to 2008 in the US. Practice, coding and populations have changed.
- Only diabetic inpatient encounters, so results do not transfer to other patients.
- One patient can have several encounters. This is handled in the splits, but encounters
  within a patient are correlated.
- Readmission to a different hospital is not observed, so labels are incomplete.
- Simulated hospitals share one population and differ only through our assignment.

## Intended use
Course project comparing centralized and federated training. **Not for clinical
decisions.**