# CONTEXT.md — cotton scanner session memory

Status file, not a diary. Session ke shuru mein sabse pehle ye parho.

---

## 1. Maqsad

Cotton 2025 timeseries model ko behtar karna. Static model abhi bana hi nahi — user ne
kaha timeseries pehle, jo phir bhi na sudhre wo static ke liye chhod dena.

## 2. Model v1

RandomForest, binary cotton=1 / non-cotton=4, 35 features = 8-din NDVI composites
**2025-04-01 se 2025-12-29**. File: `FAO/cotton/timeseries_model/model_v1/best_rf_classifier.joblib`.

## 3. Do bugs, dono theek

**BUG-1: inference window ek step shifted tha.** Purani notebook pipeline mein
`inference_start_date = 2025-04-02`, jo 04-01 band gira kar model ko 04-09..12-31 deta tha.
RF features positionally parhta hai, to har date apne calendar point se 8 din aage. Layyah
test AOI par cotton 4.63% se 7.57% — **63% zyada area**. cropstack mein cotton config sahi
dates ke saath add ki, aur guard bhi: `ndvi_training_dates` match na kare to run rukta hai.
7 test likhe, suite 78/78.

**BUG-2: training aur validation AOIs overlap karte hain.** Al-Moiz-2-1 aur Baba-Fareed-2
training clusters 12 aur 10 ko cut karte hain. Un do ke recall optimistic hain.

## 4. Validation, 6 AOIs (v1, date fix ke baad)

% = us class ke kitne pixels cotton mape gaye.

| AOI | cotton (recall) | rice | sugarcane | fall maize | orchard |
|---|---|---|---|---|---|
| Al-Moiz-2-1 | 89.6 | — | 13.0 | — | 5.0 |
| Baba-Fareed-1 | 81.8 | 7.2 | 1.9 | 1.1 | — |
| Baba-Fareed-2 | 73.1 | 20.0 | 3.5 | 0.5 | — |
| Faran-1 (Sindh) | 57.3 | 9.9 | 0.7 | — | 5.8 |
| Layyah-1 | 76.1 | 6.6 | 1.9 | — | — |
| Layyah-orchards | 70.8 | — | 3.8 | — | 1.3 |

Ground truth cotton-only hai, is se sirf recall milta hai. Precision ke liye rice/cane/
maize scans aur orchard mask ko known non-cotton maana.

**Orchard ka dar ghalat nikla.** `layyah_orchards` AOI khud banaya, us mein 7,661 acres
orchards hain — un mein se sirf **1.3%** cotton mapa. Faran-1 ka 5.8% shor hai, wahan
orchards sirf 561 acres thay.

## 5. Label audit — ek claim wapas liya

Pehle phenology se laga ke 47% cotton clusters rice hain. **Wo ghalat tha.** Surveyed
pixels par classifier train kiya (macro F1 0.890, yaani crops waqai alag hain), us ka
faisla ulta hai:

| training label | cotton | rice | cane | maize | orchard |
|---|---|---|---|---|---|
| cotton (272) | 259 | 7 | 5 | 0 | 1 |
| non-cotton (878) | 139 | 146 | 139 | 72 | 382 |

Cotton labels theek hain. **Asli ghalti ye ke 139 non-cotton clusters cotton lagte hain**,
49 par yaqeen 0.6+. Cotton ke examples negative class mein daale gaye — recall ka masla,
aur recall hi sab se kamzor hai. `aoi_3_cotton` cluster 199 pakka ghalat: classifier 100%
yaqeen se orchard kehta hai, p_cotton 0.000.

## 6. Training set v2

GEE NDVI `gs://farmdar_data_catalog/fao_cotton_training_data_2` mein para tha — 23 AOIs.
Is se ghanton ka STAC download bach gaya. 14 kaam ki AOIs utaari (7.58 GB).

**Domain shift check pehle kiya**: ek hi Faran zameen par GEE vs STAC cotton curve ka
farq 0.026 mean absolute, sugarcane 0.065. Shor se kam. GEE safe hai.

`labelled_pixels.parquet`: **1,142,992 rows, 252 grids.** Sab polygons se labelled.
cane 436k / rice 322k / orchard 253k / maize 76k / cotton 55k, plus purani non-cotton
rows label 4 ke taur par (jo cotton jaisi lagti hain wo nikal di gayin).

Codes: cotton=1, rice=2, cane=3, other=4, fall_maize=5, orchard=6. Cotton hamesha 1.

## 7. Abhi kya chal raha hai

`train_chain.sh`: v2 train (search skip, v1 ke params) → curves par v1/v2 compare →
cached tiles se v2 ke maps → un maps ko score.

## 8. Raftaar ke faisle

2 core hain. Optuna search 12 trials × 3 folds tqareeban ek ghanta le raha tha, is liye
pehle model ke liye skip kar diya. Har class 45,000 rows par cap — cane cotton se aath
guna tha aur wo rows sirf waqt kha rahi thin.

Sab kuch `gs://farmdar_data_catalog/fao_cotton_scanner_cache/` mein cache hota hai. Raw
Sentinel tiles wahi cheez hain jo mehngi hain; model ya window badalne par dobara download
nahi karna parta.

## 9. Khula sawal — user se poochna hai

Faran-1 ka recall 57.3%, baqi 73 se 90. Sindh mein cotton pehle boya jata hai. Agar v2 ise
theek na kar saka to Punjab aur Sindh ke alag models chahiye ya nahi — **ye domain faisla
hai, khud nahi karna.**

## 10. Rules

- Roman Urdu, chhota aur seedha.
- Commits `dawoodahamd.spsc@gmail.com` ke naam se, Claude ke naam se nahi.
- Domain faisla user ka hai.
- `git add -A` mat karna jab agents kaam kar rahe hon — ek dafa aadha likha file commit ho gaya tha.
