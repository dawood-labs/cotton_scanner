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

## 5. Validation ka tareeqa

Ground truth cotton-only hai, is se sirf **recall** milta hai. Precision ke liye
rice / fall-maize / sugarcane scans aur orchard mask ko known non-cotton maan kar
**per-crop commission** nikalte hain: us crop ke kitne pixels model ne cotton keh diya.
Dono cheezein sieved classification raster par, 10 m grid par.

6 AOIs: paanch mill wale + `layyah_orchards`. Aakhri wala khud banaya — Layyah district
ke andar wo 0.24 deg square jahan orchard blocks aur surveyed cotton dono sab se zyada
hain (1005 orchard feats / 6,470 acres, 560 cotton feats / 1,301 acres). Paanch mill AOIs
mein itne orchards nahi thay ke wo confusion test ho sake.

Pehle do AOIs ke numbers (date fix ke baad), % = us class ke kitne pixels cotton mape:

| AOI | cotton (recall) | rice | sugarcane | fall maize | orchard |
|---|---|---|---|---|---|
| Al-Moiz-2-1 | 89.6 | — | 13.0 | — | 5.0 |
| Baba-Fareed-1 | 81.8 | 7.2 | 1.9 | 1.1 | — |

Al-Moiz-2-1 training se contaminated hai (BUG-2), is ka 89.6 optimistic hai.

## 6. Agla qadam

1. Baqi 4 AOIs ke scores.
2. `ab_window_shift.py` — purane shifted window se A/B, taake bug ki qeemat inhi
   scores mein dikhe.
3. Reference curves ka plot: rice / orchard cotton ke oopar baithti hai ya alag hai.
   Isi se tay hoga ke masla signal mein hai ya training set mein.
4. Orchard mask lagane ka faisla — user se poochna hai.

## 7. Rules

- Roman Urdu mein baat karni hai.
- Commits `dawoodahamd.spsc@gmail.com` ke naam se, Claude ke naam se nahi.
- Domain faisla (kaunsa threshold, kaunsi acreage sahi) khud nahi karna — user se poochna.
