# CONTEXT.md — cotton scanner session memory

Status file, not a diary. Session ke shuru mein sabse pehle ye parho.

---

## 1. Maqsad

Cotton 2025 timeseries model (`FAO/cotton/timeseries_model/model_v1/best_rf_classifier.joblib`)
ko validation AOIs par test karna, masle dhoondhna, phir theek karna.
Shak: model rice ko bhi cotton pakar leta hai. Orchards bhi cotton jaisi NDVI curve dete hain.

---

## 2. Model kya hai

- RandomForest, 250 trees, depth 30. Binary: cotton=1, non-cotton=4.
- 35 features = 8-din ke NDVI composites, **2025-04-01 se 2025-12-29**.
- Training data: 22,243 rows, `FAO/cotton/training_data_parquet/cotton_2025_training_data_v2.csv`.
- Held-out test (training log): accuracy 0.93, cotton recall 0.78, precision 0.89.

## 3. Ab tak kya mila

**BUG-1 (confirmed, bara masla): inference window ek step shifted tha.**
Purani pipeline (`FAO/cotton/scripts/FAO_All_Model_Execution_Pipeline_v3.0 (2).ipynb`)
mein `inference_start_date = 2025-04-02`. STAC stack ke bands 04-01 se 12-31 tak (36) hain,
to filter 04-01 wala band gira deta hai aur model ko **04-09 .. 12-31** deta hai — jabki
training **04-01 .. 12-29** thi. RF features positionally parhta hai, to har date apne
calendar point se 8 din aage compare hoti rahi. Aakhri band (12-29→12-31) sirf 2 din ka
composite hai, yaani shor.

Layyah test AOI (150k pixels) par asar:

| features | cotton |
|---|---|
| aligned 04-01..12-29 | 4.63% |
| shifted 04-09..12-31 | 7.57% |

Yaani shift ne cotton area **63% barha diya**.

**BUG-2: validation contamination.** 23 training cluster AOIs mein se 2 validation AOIs
ko cut karte hain — `Baba-Fareed-2` (train cluster 10) aur `Al-Moiz-2-1` (train cluster 12).
In do ke numbers optimistic honge. Faran-1, Layyah-1, Baba-Fareed-1 saaf hain.

**Ground truth sirf cotton hai** (`predicted=2`, positives only). To in se **recall** milta
hai, precision nahi. Precision ke liye rice / fall-maize / sugarcane scans aur orchard mask
ko known non-cotton ke taur par use kar rahe hain.

## 4. Kya kiya

- `cropstack` (dawood-labs/cropstack) mein **cotton crop config** add ki, sahi dates ke sath.
- `cropstack/ndvi_pipeline.py` mein guard add kiya: agar inference window model ke
  `n_features_in_` se match na kare to run **rukta** hai. BUG-1 dobara chup ke nahi ho sakta.
- Validation data GCS se local: `FAO/cotton/validation_data/`.
- 5 per-mill AOIs: `FAO/cotton/validation_data/aois/*.gpkg`.

## 5. Abhi kya chal raha hai

`scripts/run_validation.sh` — 5 AOIs sequential, output `FAO/cotton/validation_runs/`.
Order: al_moiz_2_1, baba_fareed_1, baba_fareed_2, faran_1, layyah_1.
Box par sirf 2 core hain, isliye sequential.

## 6. Agla qadam

1. Runs khatam hone par per-mill recall + reference-crop false positive rate.
2. Rice / cane / orchard confusion ka pixel-level analysis.
3. Orchard mask lagana.

## 7. Rules

- Roman Urdu mein baat karni hai.
- Commits `dawoodahamd.spsc@gmail.com` ke naam se, Claude ke naam se nahi.
- Domain faisla (kaunsa threshold, kaunsi acreage sahi) khud nahi karna — user se poochna.
