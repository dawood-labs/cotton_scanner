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

## 7. v2 ka nateeja — map level, sieve ke baad

| AOI | recall v1→v2 | rice | cane | orchard |
|---|---|---|---|---|
| Al-Moiz-2-1 | 89.6 → **92.0** | — | 13.0 → **9.9** | 5.0 → **4.1** |
| Baba-Fareed-1 | 81.8 → **83.2** | 7.2 → **4.1** | 1.9 → 3.0 | — |
| Baba-Fareed-2 | 73.1 → **81.6** | 20.0 → **12.3** | 3.5 → 6.9 | — |
| Faran-1 | 57.3 → **78.6** | 9.9 → 21.6 | 0.7 → 1.9 | 5.8 → 13.3 |
| Layyah-1 | 76.1 → **81.6** | 6.6 → **4.6** | 1.9 → 5.1 | — |
| Layyah-orchards | 70.8 → **78.5** | — | 3.8 → 4.4 | 1.3 → 1.8 |

Recall har jagah barha, sab se kam ab 78.5 (tha 57.3). Rice teen jagah kam hua.
Faran akela ulta chala.

Model v2 held-out (67 grids jo training mein thay hi nahi): cotton precision 0.924,
recall 0.872. Rice precision 0.692 — crops mein sab se kamzor.

## 8. 'other' class — do experiments, dono ka jawab

- **Nikal dena: ghalat.** Recall +2.3 lekin rice +1.7, cane +0.9, orchard +2.0.
  Ganda hone ke bawajood wo class apna kaam kar rahi hai.
- **Saaf karna: bemani.** Cleaner ne 17,305 mein se 9,561 rows apni asli class mein
  bhej deen (849 cotton, 1,619 rice, 6,089 orchard). Nateeja: pooled recall 83.1 → 83.3,
  rice 12.0 → 12.3. Sab shor ke andar. **v2 sada 'other' ke saath hi rahega.**

## 9. Faran / Sindh — faisla user ka, lekin ab jeet ka raasta maujood hai

`validation_runs/threshold_sweep.csv` — 6 AOIs, 7 cuts. Faran-1 par:

| cut | recall | rice | orchard |
|---|---|---|---|
| 0.5 | 73.5 | 15.4 | 6.4 |
| **0.6** | **67.6** | **10.5** | **4.7** |
| 0.7 | 60.1 | 7.1 | 2.5 |

v1 wahan 59.4 recall / 11.9 rice deta tha. **v2 at cut 0.6 = 67.6 / 10.5 — dono taraf
behtar.** Koi trade-off nahi, sidha faida. Tajweez: Sindh mein cut 0.6, Punjab mein
argmax. **Number user ne dena hai.**

Sweep ke numbers argmax se neeche hain kyunki argmax cotton chun leta hai chahe uski
probability 0.5 se kam ho, agar baqi sab us se bhi kam hon.

## 10. Orchard mask

v2 maps par: Faran-1 1.3%, Layyah-orchards 1.2%, Al-Moiz 0.0% (wahan sirf 21 blocks).
Chhota hai lekin muft, aur wo acres pakke ghalat hain. Lagana chahiye.

## 12. Raftaar ke faisle

2 core hain. Optuna search 12 trials × 3 folds tqareeban ek ghanta le raha tha, is liye
pehle model ke liye skip kar diya. Har class 45,000 rows par cap — cane cotton se aath
guna tha aur wo rows sirf waqt kha rahi thin.

Sab kuch `gs://farmdar_data_catalog/fao_cotton_scanner_cache/` mein cache hota hai. Raw
Sentinel tiles wahi cheez hain jo mehngi hain; model ya window badalne par dobara download
nahi karna parta.

## 11. Hard negative mining — try kiya, kaam nahi aaya

Har class ka aadha quota un rows se bhara jinhein model ghalat karta hai. Pehli nazar
mein jeet lagi: cane 7.0 → 4.0, orchard 6.8 → 4.7, rice 12.0 → 11.0. Lekin recall bhi
83.1 → 78.4. Koi bhi shy model yahi karta hai.

Barabar recall par sweep karke dekha (`validation_runs/matched_recall.csv`):

| recall | v2 | mined |
|---|---|---|
| 75% | 3.91 | 3.93 |
| 83% | 7.14 | 7.46 |
| 85% | 7.91 | 8.64 |

Curve hili hi nahi, sirf us par jagah badli. Ek cheez hui: mined har recall par **cane
par behtar** aur **rice par kharab** hai. User ne kaha rice priority hai, cane static
se nikal jayega — is liye **v2 rakha, mined nahi**.

## 12. Kahan tak pahuncha

- cropstack ka cotton config ab **model_v2** ki taraf ishara karta hai. 81 tests pass.
- cropstack ab model ka apna `model_card.json` parhta hai (35 dates) aur usay config ki
  list par tarjeeh deta hai — kyunki config us waqt purani hoti hai jab model retrain
  hota hai, aur wahi waqt hai jab guard chahiye.
- Sab kuch `gs://farmdar_data_catalog/fao_cotton_scanner_cache/` mein cache hai.

## 13. Layyah district — poora run, end to end

| | |
|---|---|
| district | 1,555,070 acres |
| cotton mila | **117,826 acres** (7.6%) |
| polygons | 46,543 |
| kul waqt | **136 min** |

Phase timings — aur yahi asal seekh hai:

| phase | waqt | hissa |
|---|---|---|
| STAC download (79 tiles, 4 workers) | 111 min | **82%** |
| RF inference (2 core) | 24 min | 18% |
| mosaic + sieve | 7 sec | ~0 |
| vector + export | 14 sec | ~0 |

**Bottleneck CPU nahi, download hai.** Inference 20 sec/tile, download 5.5 min/tile.
Download workers se bandha hai aur workers RAM se — `per_tile_gib = stac_tile_memory_gib
x (tile_deg/0.1)^2`. Layyah par 4 workers ne budget ka 97% bhar rakha tha (5.8 of 6.0 GiB).

Do guna core = 11 min ka faida. Do guna RAM = download aadha. **RAM chahiye, core nahi.**

Meri ghalti: tuning ko "khali CPU" samajh kar download ke saath chalaya. Us ne RAM li,
jis se download ka budget ghata aur workers 4 par ruke.

Model card guard production mein chala: log mein `Model card at model_card.json pins 35
training dates; using those`.

## 14. Ghotki — chhoti tiles ka experiment

0.07 deg tiles + 8 workers. RAM per tile rakbe ke murabba se ghatti hai, to aadhi RAM,
dugne workers. Khatra: tiles 79 se ~120 ho jayengi aur har tile ka apna fixed kharcha hai.
Layyah (6,293 km2) aur Ghotki (4,757 km2) qareeb hain, to per-km2 muqabla saaf hoga.

## 15. Pod restart ka sabaq

Pod dobara ban gaya (disk `nvme2n1` -> `nvme3n1`), aur `setsid nohup` waale background
waiters bhi mar gaye. Layyah bach gaya kyunki wo mukammal ho chuka tha. **Lambi chain
par bharosa mat karo** — har qadam ke baad state disk par honi chahiye, aur cropstack ka
`run_mode=resume` hi asli bachao hai.

## 16. Agla kaam

1. Poore Layyah district par end-to-end run. Ab tak sab chhote AOIs par hua hai.
2. Hyperparameter tuning — shuru ki thi, rok di kyunki mined run zyada qeemti tha.
   `--n-trials 14` se chal jayegi, ~110 min.
3. Static model (user banayega) — cane usi se nikalna hai.
4. Sindh ka threshold: `validation_runs/threshold_sweep.csv` mein poori curve hai.
   Faran-1 par cut 0.6 = 67.6 recall / 10.5 rice, jabke v1 = 59.4 / 11.9. Dono taraf
   behtar. **Number user ne dena hai.**

## 14. Rules

- Roman Urdu, chhota aur seedha.
- Commits `dawoodahamd.spsc@gmail.com` ke naam se, Claude ke naam se nahi.
- Domain faisla user ka hai.
- `git add -A` mat karna jab agents kaam kar rahe hon — ek dafa aadha likha file commit ho gaya tha.
