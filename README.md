# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

> Yazılım Geliştirme Dersi — 2. Proje  
> Kocaeli Üniversitesi

## İçindekiler

1. [Proje Tanımı](#proje-tanımı)
2. [Veri Setleri](#veri-setleri)
3. [Kurulum](#kurulum)
4. [Kullanım](#kullanım)
5. [Yazılım Mimarisi](#yazılım-mimarisi)
6. [Modeller](#modeller)
7. [Deneysel Sonuçlar](#deneysel-sonuçlar)
8. [İstatistiksel Analiz](#istatistiksel-analiz)
9. [Açıklanabilirlik Modülü](#açıklanabilirlik-modülü)
10. [Parametre Analizi](#parametre-analizi)
11. [Görseller](#görseller)

---

## Proje Tanımı

Bu projede zaman serisi anomali tespiti için iki farklı modelleme yaklaşımı karşılaştırılmaktadır:

- **Derin Öğrenme Modelleri** (LSTM, GRU, 1D-CNN): Yüksek doğruluk potansiyeline sahip ancak kararlarını açıklayamayan *black-box* modeller
- **Probabilistic Automata**: PAA + SAX dönüşümleri üzerine kurulu, her kararı olasılıksal geçişlerle açıklayabilen yorumlanabilir model

Modeller yalnızca performans değil; gürültüye dayanıklılık, unseen veri davranışı ve açıklanabilirlik kriterleri açısından da analiz edilmektedir.

**Araştırma sorusu:** *Farklı modelleme yaklaşımları, zaman serisi verileri üzerinde farklı veri koşulları altında nasıl davranmaktadır ve bu davranışlar istatistiksel olarak anlamlı mıdır?*

---

## Veri Setleri

### SKAB (Skoltech Anomaly Benchmark)
- Kaynak: `data/raw/SKAB/valve1/` ve `valve2/` klasörleri
- Tüm `.csv` dosyaları birleştirildi; `source_group` (valve1/valve2) ve `source_file` takip sütunları eklendi
- `source_file`, klasör önekiyle benzersizleştirildi (`valve1/0.csv`, `valve2/0.csv`) — böylece iki klasördeki aynı adlı dosyalar GroupKFold'da çakışmaz ve veri sızıntısı önlenir
- Hedef değişken: `anomaly` (0=normal, 1=anomali)
- Model girdisi dışı sütunlar: `datetime`, `changepoint`, `source_group`, `source_file`
- Değerlendirme: **StratifiedGroupKFold** (5 fold, `source_file` grup değişkeni)

### BATADAL (Battle of the Attack Detection ALgorithms)
- Kaynak: `data/raw/BATADAL/BATADAL_dataset04.csv` (Training Dataset 2)
- Hedef değişken: `ATT_FLAG` (-999 → 0 normal, diğer → 1 saldırı/anomali); veri dosyasında doğrulandı
- Anomali oranı ~%5.2 (219 / 4177 kayıt)
- 43 sensör/sistem değişkeni model girdisi; zaman sütunu (`DATETIME`) model girdisi dışı tutuldu
- Değerlendirme: **Zaman sıralı bölme** — %60 eğitim / %20 doğrulama / %20 test

---

## Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Veri setlerini aşağıdaki konumlara koy:
# data/raw/SKAB/valve1/*.csv
# data/raw/SKAB/valve2/*.csv
# data/raw/BATADAL/BATADAL_dataset04.csv
```

---

## Kullanım

```bash
# Tüm modelleri çalıştır (DL + Automata)
python main.py --dataset both --model all --output-dir results

# Sadece DL modelleri
python main.py --dataset both --model dl --output-dir results

# Sadece Automata
python main.py --dataset both --model automata --output-dir results

# Parametre analizi (window_size x alphabet_size)
python main.py --param-analysis --dataset both --output-dir results

# Tüm görselleri üret (DL + Automata: CM/ROC/PR, state diagram, heatmap, parametre grafikleri)
python generate_plots.py --batadal-dir results --skab-dir results --results-dir results --dataset both

# İstatistiksel testleri çalıştır
python run_statistical_tests.py --batadal-dir results --skab-dir results --output results/statistical_tests.json

# Birim testleri çalıştır
python -m pytest -q
```

---

## Yazılım Mimarisi

```
yazlab2proje2/
├── config/
│   └── config.yaml          # Merkezi konfigürasyon (hard-coded değer yok)
├── src/
│   ├── config.py             # Config dataclass'ları
│   ├── data/
│   │   ├── loader.py         # SKAB + BATADAL veri yükleyiciler
│   │   └── preprocessor.py  # Normalizasyon, PCA, bölme, GroupKFold
│   ├── models/
│   │   ├── lstm_model.py     # PyTorch LSTM
│   │   ├── gru_model.py      # PyTorch GRU
│   │   ├── cnn_model.py      # PyTorch 1D-CNN
│   │   ├── trainer.py        # Eğitim döngüsü, loglama, sonuç kaydetme
│   │   └── _train_utils.py   # Paylaşılan eğitim fonksiyonu
│   ├── automata/
│   │   ├── paa.py            # Piecewise Aggregate Approximation
│   │   ├── sax.py            # Symbolic Aggregate approXimation
│   │   ├── levenshtein.py    # Edit distance + unseen pattern yönetimi
│   │   ├── automata.py       # Probabilistic Automata ana modeli
│   │   └── explainability.py # Açıklanabilirlik modülü
│   ├── experiments/
│   │   ├── runner.py         # Deney koşturucu (DL + Automata)
│   │   └── noise.py          # Gaussian gürültü ekleme
│   ├── evaluation/
│   │   ├── metrics.py        # Accuracy, Precision, Recall, F1
│   │   └── statistical.py    # Wilcoxon + McNemar testleri
│   └── visualization/
│       ├── plots.py          # DL görsel fonksiyonları
│       └── automata_plots.py # Automata görsel fonksiyonları
├── tests/
│   ├── conftest.py           # Ortak pytest fixture'ları (config)
│   ├── test_data.py          # Veri pipeline birim testleri
│   └── test_automata.py      # PAA, SAX, Levenshtein birim testleri (31 test)
├── main.py                   # Ana giriş noktası
├── generate_plots.py         # Görsel üretici (DL + Automata)
└── run_statistical_tests.py  # McNemar istatistiksel testler
```

**Temel tasarım kararları:**
- Tüm parametreler `config/config.yaml`'da — hiçbir yerde hard-coded değer yok
- Scaler ve PCA yalnızca train verisiyle `fit` edilir (data leakage önleme)
- SAX/PAA sözlüğü ve otomata geçiş olasılıkları yalnızca train verisiyle oluşturulur
- SKAB'da `source_file` grup değişkeni klasör önekli tutulur; aynı dosyanın kayıtları hem train hem test'te yer almaz
- DL modelleri 5 PCA bileşeni, Automata modeli PC1 (1 bileşen) kullanır
- Sınıf dengesizliği: DL modellerinde `pos_weight = n_neg / n_pos` ile ağırlıklı BCELoss

---

## Modeller

### Derin Öğrenme Modelleri

| Parametre | Değer |
|-----------|-------|
| Epoch üst sınırı | 50 |
| Batch size | 32 |
| Early stopping patience | 5 (val_loss) |
| Random seed | 42, 123, 2026, 7, 999 |
| Sequence length | 30 |
| PCA bileşen sayısı | 5 |
| Optimizör | Adam |

### Probabilistic Automata

| Parametre | Değer |
|-----------|-------|
| Window size | 4 (varsayılan) |
| Alphabet size | 3 (varsayılan) |
| PCA bileşen sayısı | 1 (PC1) |
| Smoothing | Laplace (α=1e-6) |
| Unseen yönetimi | Levenshtein edit distance |

**Akış:** Ham seri → PAA → SAX → Sliding Window → Pattern (State) → Geçiş Olasılıkları → Anomali Skoru

---

## Deneysel Sonuçlar

### BATADAL Sonuçları

| Model | Senaryo | F1 (mean ± std) | Accuracy | Precision | Recall |
|-------|---------|-----------------|----------|-----------|--------|
| LSTM | Original | 0.5799 ± 0.0711 | 0.8715 | 0.4719 | 0.8425 |
| LSTM | Noisy | 0.5462 ± 0.0986 | 0.8352 | 0.4141 | 0.9075 |
| GRU | Original | 0.5704 ± 0.0572 | 0.8486 | 0.4045 | 0.9850 |
| GRU | Noisy | 0.5616 ± 0.0464 | 0.8452 | 0.3958 | 0.9800 |
| CNN | Original | 0.4666 ± 0.0899 | 0.7717 | 0.3120 | 0.9575 |
| CNN | Noisy | 0.4270 ± 0.0840 | 0.7352 | 0.2788 | 0.9500 |
| **Automata** | Original | 0.1522 | 0.6268 | 0.0972 | 0.3500 |
| **Automata** | Noisy | 0.1935 | 0.6411 | 0.1233 | 0.4500 |
| **Automata** | Unseen | 0.1522 | 0.6268 | 0.0972 | 0.3500 |

### SKAB Sonuçları (5-Fold Ortalama)

| Model | Senaryo | F1 (mean ± std) | Accuracy | Precision | Recall |
|-------|---------|-----------------|----------|-----------|--------|
| LSTM | Original | **0.9066 ± 0.0272** | 0.9369 | 0.9383 | 0.8794 |
| LSTM | Noisy | 0.9022 ± 0.0191 | 0.9350 | 0.9437 | 0.8674 |
| GRU | Original | 0.8974 ± 0.0289 | 0.9296 | 0.9187 | 0.8803 |
| GRU | Noisy | 0.9005 ± 0.0325 | 0.9316 | 0.9180 | 0.8868 |
| CNN | Original | 0.8957 ± 0.0452 | 0.9279 | 0.9118 | 0.8836 |
| CNN | Noisy | 0.8937 ± 0.0458 | 0.9264 | 0.9082 | 0.8826 |
| **Automata** | Original | 0.4777 ± 0.0193 | 0.4010 | 0.3432 | 0.7886 |
| **Automata** | Noisy | 0.4768 ± 0.0165 | 0.3947 | 0.3413 | 0.7944 |
| **Automata** | Unseen | 0.4777 ± 0.0193 | 0.4010 | 0.3432 | 0.7886 |

### Veri Setleri Arası Karşılaştırma

DL modelleri SKAB'da belirgin şekilde daha iyi performans göstermektedir (F1 ~0.90 vs ~0.58). BATADAL'da sınıf dengesizliği (~%5.2 anomali) ve yüksek boyutlu sensör verisi (43 özellik → 5 PCA bileşeni) zorluğu artırmaktadır. Automata modeli PC1 tek bileşenle çalıştığından BATADAL'da bilgi kaybı daha yüksektir.

### Gürültü Etkisi Analizi

Gaussian gürültü (std=0.1) eklendiğinde:
- **DL modelleri:** F1 büyük ölçüde korunur — SKAB'da değişim ~0.00–0.01 (GRU gürültüyle hafifçe iyileşir bile), BATADAL'da ~0.01–0.04 düşüş. Yüksek dayanıklılık.
- **Automata:** BATADAL'da F1 hafif artış (0.1522 → 0.1935), SKAB'da minimal değişim — gürültü bazı durumlarda anomali skorunu değiştiriyor.

Genel olarak her iki model tipi de Gaussian gürültüye karşı dayanıklıdır.

### Unseen Veri Davranışı

Automata modelinde test sırasında eğitim SAX sözlüğünde bulunmayan pattern'larla karşılaşıldığında Levenshtein edit distance ile en yakın bilinen pattern bulunur:
- BATADAL: Unseen oranı ~%2.9 (76 state)
- SKAB: Unseen oranı ~%0.3 (≈35 state)

Unseen senaryo ile original senaryo arasında performans farkı gözlemlenmemiştir — Levenshtein eşleştirme mekanizması her tahminde etkin çalıştığından görülmemiş pattern'lar en yakın bilinen state üzerinden işlenir.

---

## İstatistiksel Analiz

Model farklarının istatistiksel anlamlılığı **McNemar testi** ile değerlendirilmiştir (örnek bazlı karşılaştırma, seed=42).

### BATADAL

| Karşılaştırma | Original p-değeri | Noisy p-değeri |
|---------------|-------------------|----------------|
| LSTM vs GRU | p < 0.0001 ✓ | p < 0.0001 ✓ |
| LSTM vs CNN | p = 0.0055 ✓ | p < 0.0001 ✓ |
| GRU vs CNN | p = 0.0002 ✓ | p = 0.0001 ✓ |

### SKAB

| Karşılaştırma | Original p-değeri | Noisy p-değeri |
|---------------|-------------------|----------------|
| LSTM vs GRU | p < 0.0001 ✓ | p < 0.0001 ✓ |
| LSTM vs CNN | p < 0.0001 ✓ | p < 0.0001 ✓ |
| GRU vs CNN | p = 0.0022 ✓ | p < 0.0001 ✓ |

**Yorum:** Hem BATADAL hem SKAB'da tüm model çiftleri istatistiksel olarak anlamlı farklılık göstermektedir (p < 0.05). SKAB'da en yüksek F1'i LSTM (0.9066) elde etmekle birlikte üç DL modeli de yüksek ve birbirine yakın performans sergilemektedir; McNemar testi bu yakın farkların dahi örnek bazında anlamlı olduğunu göstermektedir.

> **Not:** Wilcoxon testi n=5 seed ile minimum p=0.0625 üretemediğinden (matematiksel sınır), daha güçlü olan McNemar testi tercih edilmiştir.

---

## Açıklanabilirlik Modülü

Automata modeli her karar için aşağıdaki bilgileri üretmektedir:

```json
{
  "time_step": 5,
  "state": "abc",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "distance": 1,
  "transitions": {
    "aab->abc": 0.72,
    "abc->bcc": 0.15
  },
  "transition_probability": 0.15,
  "path_probability": 0.108,
  "state_anomaly_rate": 0.412,
  "anomaly_score": 0.55,
  "probability": 0.108,
  "decision": "anomaly",
  "justification": "Dusuk olasilikli path tespit edildi (beklenmeyen gecis)",
  "confidence": 0.108
}
```

**Geçiş olasılıkları** frekans tabanlı öğrenilir: `P(Si → Sj) = Geçiş Sayısı / Toplam Çıkış Sayısı` (Laplace α=1e-6 ile).

**Path probability:** Gözlemlenen yerel geçişlerin çarpımı — `path_probability = P(prev → current) × P(current → next)`. Düşük path olasılığı, model tarafından beklenmeyen davranış olarak yorumlanır.

**Güven skoru** doğrudan geçiş olasılıklarından türetilir: `confidence = path_probability`. Düşük path olasılığı → düşük güven → beklenmeyen (anomali) davranış (örnekte 0.72 × 0.15 = 0.108).

**Anomali skoru** iki sinyal birleştirilerek hesaplanır:
- State anomali oranı (eğitimden öğrenilir): `0.6 × state_anomaly_rate`
- Geçiş beklenmedikliği: `0.4 × (1 - transition_probability)`

Metin (insan-okur) formatı `format_decision()` ile de üretilir:

```
[SYSTEM DECISION]
Time Step: t = 5
Previous State: "aab"
Incoming Pattern: "adc"
Status: Unseen
Nearest Pattern: "abc" (distance = 1)

Transitions:
  aab->abc : 0.72
  abc->bcc : 0.15

Path Probability: 0.72 * 0.15 = 0.108

Decision:
  Low probability path detected
  Result: ANOMALY
  Confidence Score: 0.108 (Low)
```

Açıklamalar deterministik ve yeniden üretilebilirdir; modelin iç hesaplamalarıyla tutarlıdır.

---

## Parametre Analizi

Automata modeli için window_size × alphabet_size kombinasyonları BATADAL verisi üzerinde test edilmiştir:

| window_size | alphabet_size | F1 | State Sayısı | Geçiş Yoğunluğu |
|-------------|---------------|----|--------------|-----------------|
| 3 | 3 | 0.2045 | 26 | 0.1050 |
| 4 | 3 | 0.1522 | 76 | 0.0260 |
| 5 | 4 | 0.2237 | 298 | 0.0042 |
| 5 | 5 | **0.2362** | 367 | 0.0032 |
| 6 | 4 | 0.1964 | 304 | 0.0038 |

**Gözlemler:**
- Alphabet size artışı → daha fazla state → geçiş yoğunluğu düşer
- En iyi F1: ws=5, as=5 (0.2362) — orta-geniş pencere ile orta alfabe en iyi dengeyi sağlıyor
- Yüksek state sayısı modeli daha açıklayıcı yapar ancak seyrek geçiş matrisine yol açar

---

## Görseller

### Model Karşılaştırmaları

**BATADAL — DL vs Automata**
![DL vs Automata BATADAL](plots/batadal/dl_vs_automata_BATADAL.png)

**SKAB — DL vs Automata**
![DL vs Automata SKAB](plots/skab/dl_vs_automata_SKAB.png)

**Tüm Metrikler — BATADAL (LSTM/GRU/CNN/Automata)**
![All Metrics BATADAL](plots/batadal/all_metrics_BATADAL.png)

**Tüm Metrikler — SKAB**
![All Metrics SKAB](plots/skab/all_metrics_SKAB.png)

**Cross-Dataset Karşılaştırma**
![Cross Dataset](plots/cross_dataset_comparison.png)

---

### Confusion Matrix

**LSTM — BATADAL (Original)**
![CM LSTM BATADAL](plots/batadal/cm_BATADAL_LSTM_original.png)

**GRU — SKAB (Original)**
![CM GRU SKAB](plots/skab/cm_SKAB_GRU_original.png)

---

### ROC Eğrileri

**LSTM — BATADAL**
![ROC LSTM BATADAL](plots/batadal/roc_BATADAL_LSTM_original.png)

**GRU — SKAB**
![ROC GRU SKAB](plots/skab/roc_SKAB_GRU_original.png)

---

### Automata State Diagram

**BATADAL** — Düğüm rengi anomali oranını gösterir (yeşil=normal, kırmızı=anomali)
![State Diagram BATADAL](plots/batadal/state_diagram_BATADAL.png)

**SKAB**
![State Diagram SKAB](plots/skab/state_diagram_SKAB.png)

---

### Transition Probability Heatmap

**BATADAL**
![Heatmap BATADAL](plots/batadal/transition_heatmap_BATADAL.png)

**SKAB**
![Heatmap SKAB](plots/skab/transition_heatmap_SKAB.png)

---

### Parametre Duyarlılık Grafikleri

**BATADAL — F1 Heatmap (Window Size × Alphabet Size)**
![Param F1 BATADAL](plots/batadal/param_f1_heatmap_BATADAL.png)

**BATADAL — State Sayısı Heatmap**
![Param States BATADAL](plots/batadal/param_states_heatmap_BATADAL.png)

**BATADAL — Parametre Duyarlılık Çizgi Grafiği**
![Param Sensitivity BATADAL](plots/batadal/param_sensitivity_BATADAL.png)

---

### Gürültü Etkisi

**BATADAL**
![Noise BATADAL](plots/batadal/noise_comparison_BATADAL.png)

**SKAB**
![Noise SKAB](plots/skab/noise_comparison_SKAB.png)

---

## Ekip

| Kişi | Görev |
|------|-------|
| Kerem Ünal | DL modelleri (LSTM, GRU, CNN), veri pipeline, görselleştirme, istatistiksel testler |
| Efekan Tosmak | Probabilistic Automata, açıklanabilirlik modülü, parametre analizi |
