# Model Dokümantasyonu - Telco Customer Churn Prediction

## İçindekiler
1. [Genel Bakış](#genel-bakış)
2. [Model Açıklamaları](#model-açıklamaları)
3. [Performans Karşılaştırması](#performans-karşılaştırması)
4. [Sınıf Dengesizliği Çözümü](#sınıf-dengesizliği-çözümü)
5. [Hiperparametre Optimizasyonu](#hiperparametre-optimizasyonu)
6. [Sonuç ve Değerlendirme](#sonuç-ve-değerlendirme)

---

## Genel Bakış

Bu projede Telco müşteri kaybı (churn) tahmini için **10 farklı makine öğrenmesi modeli** eğitilmiş ve karşılaştırılmıştır.

- **Veri Seti:** 7.043 müşteri, 30 feature (encoding sonrası)
- **Train/Test Oranı:** %80 / %20 (5.634 / 1.409)
- **Sınıf Dağılımı:** ~%74 No Churn, ~%26 Churn (dengesiz veri)
- **Değerlendirme Metriği:** F1-Score (ana), ROC-AUC, Recall (destekleyici)

---

## Model Açıklamaları

### 1. Logistic Regression (Lojistik Regresyon)

**Kategori:** Doğrusal Model

Lojistik Regresyon, ikili sınıflandırma için en temel ve yorumlanabilir modellerden biridir. Sigmoid fonksiyonu kullanarak giriş özelliklerinin doğrusal kombinasyonunu 0-1 arasında bir olasılığa dönüştürür.

- **Avantajları:** Hızlı eğitim, yüksek yorumlanabilirlik, düşük overfitting riski
- **Dezavantajları:** Doğrusal olmayan ilişkileri yakalayamaz
- **Parametreler:** `C=1.0, max_iter=1000, class_weight='balanced'`

### 2. Decision Tree (Karar Ağacı)

**Kategori:** Ağaç Tabanlı Model

Karar ağaçları, veriyi özellik eşik değerlerine göre hiyerarşik olarak bölerek sınıflandırma yapar. Her düğüm bir karar kuralı, her yaprak bir sınıf etiketi içerir.

- **Avantajları:** Kolay yorumlanabilirlik, özellik etkileşimlerini yakalama
- **Dezavantajları:** Overfitting eğilimi yüksek, kararsız (küçük veri değişikliklerine duyarlı)
- **Parametreler:** `max_depth=8, min_samples_split=10, class_weight='balanced'`

### 3. Random Forest (Rastgele Orman)

**Kategori:** Ensemble (Topluluk) — Bagging

Birden fazla karar ağacını rastgele alt veri kümeleri ve rastgele özellik alt kümeleri üzerinde eğitir. Tahminleri oylama (voting) ile birleştirir. Bu yaklaşım, tek ağacın overfitting sorununu azaltır.

- **Avantajları:** Yüksek doğruluk, overfitting direnci, feature importance
- **Dezavantajları:** Yavaş tahmin, düşük yorumlanabilirlik
- **Parametreler:** `n_estimators=200, max_depth=15, class_weight='balanced'`

### 4. Gradient Boosting

**Kategori:** Ensemble — Boosting

Ağaçları sıralı olarak eğitir; her yeni ağaç, önceki ağaçların hatalarını düzeltmeye çalışır. Gradient descent optimizasyonu ile kayıp fonksiyonunu minimize eder.

- **Avantajları:** Yüksek doğruluk, çeşitli kayıp fonksiyonları desteği
- **Dezavantajları:** Yavaş eğitim, hiperparametre hassasiyeti, class_weight desteği yok
- **Parametreler:** `n_estimators=200, learning_rate=0.1, max_depth=5`

### 5. AdaBoost (Adaptive Boosting)

**Kategori:** Ensemble — Boosting

Zayıf sınıflandırıcıları (genellikle sığ ağaçlar) iteratif olarak eğitir. Her iterasyonda yanlış sınıflandırılan örneklere daha fazla ağırlık verir.

- **Avantajları:** Basit implementasyon, overfitting direnci
- **Dezavantajları:** Gürültülü veriye hassas, sınıf dengesizliğinde zayıf
- **Parametreler:** `n_estimators=200, learning_rate=0.1`

### 6. XGBoost (Extreme Gradient Boosting)

**Kategori:** Ensemble — Gelişmiş Boosting

Gradient Boosting'in optimize edilmiş versiyonudur. L1/L2 regularizasyon, paralel işleme, ağaç budama ve eksik değer yönetimi gibi gelişmiş özellikler sunar. Kaggle yarışmalarında en sık kullanılan algoritmadır.

- **Avantajları:** En yüksek doğruluk potansiyeli, hızlı eğitim, regularizasyon
- **Dezavantajları:** Hiperparametre sayısı fazla, yorumlanması zor
- **Parametreler:** `n_estimators=200, max_depth=3, learning_rate=0.05, scale_pos_weight=2.83` *(RandomizedSearchCV ile optimize edildi)*

### 7. LightGBM (Light Gradient Boosting Machine)

**Kategori:** Ensemble — Gelişmiş Boosting

Microsoft tarafından geliştirilen, histogram tabanlı bölme ve yaprak bazlı (leaf-wise) büyüme stratejisi kullanan gradient boosting framework'üdür. XGBoost'a göre daha hızlı eğitim süresi sunar.

- **Avantajları:** Çok hızlı eğitim, düşük bellek kullanımı, kategorik özellik desteği
- **Dezavantajları:** Küçük veri setlerinde overfitting riski, yaprak sayısı hassasiyeti
- **Parametreler:** `n_estimators=300, max_depth=5, num_leaves=15, learning_rate=0.01, is_unbalance=True` *(RandomizedSearchCV ile optimize edildi)*

### 8. SVM (Support Vector Machine)

**Kategori:** Çekirdek Tabanlı Model

Veri noktalarını en geniş marjin ile ayıran hiper düzlemi bulur. RBF çekirdeği ile doğrusal olmayan sınırları da öğrenebilir. Yüksek boyutlu uzaylarda etkilidir.

- **Avantajları:** Yüksek boyutlu veride güçlü, kernel trick ile esneklik
- **Dezavantajları:** Büyük veri setlerinde yavaş, hiperparametre hassasiyeti
- **Parametreler:** `kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'`

### 9. KNN (K-Nearest Neighbors)

**Kategori:** Instance-Based (Tembel Öğrenme)

Yeni bir veri noktasını, eğitim verisindeki en yakın K komşusunun sınıfına göre sınıflandırır. Model eğitimi yapmaz; tüm hesaplama tahmin zamanında gerçekleşir.

- **Avantajları:** Basit mantık, parametresiz öğrenme, doğrusal olmayan sınırlar
- **Dezavantajları:** Büyük veri setlerinde yavaş tahmin, boyut laneti, ölçeklemeye hassas
- **Parametreler:** `n_neighbors=7, weights='distance'`

### 10. MLP Neural Network (Çok Katmanlı Algılayıcı)

**Kategori:** Yapay Sinir Ağı

Giriş, gizli ve çıkış katmanlarından oluşan ileri beslemeli yapay sinir ağıdır. Geri yayılım (backpropagation) ile eğitilir. Karmaşık doğrusal olmayan ilişkileri öğrenebilir.

- **Avantajları:** Karmaşık örüntüleri yakalama, otomatik özellik çıkarımı
- **Dezavantajları:** Çok veri gerektirir, kara kutu, eğitim süresi uzun
- **Parametreler:** `hidden_layers=(128,64,32), early_stopping=True, learning_rate='adaptive'`

---

## Performans Karşılaştırması

### Genel Sıralama (F1-Score'a göre)

| Sıra | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV-F1 | Süre |
|------|-------|----------|-----------|--------|----------|---------|-------|------|
| 🥇 | **LightGBM** | 0.7488 | 0.5174 | **0.7941** | **0.6266** | 0.8439 | **0.6281** | 0.22s |
| 🥈 | **XGBoost** | 0.7445 | 0.5119 | **0.8048** | 0.6258 | **0.8446** | 0.6273 | 0.10s |
| 🥉 | **SVM** | 0.7459 | 0.5141 | 0.7807 | 0.6200 | 0.8249 | 0.6190 | 4.82s |
| 4 | Logistic Regression | 0.7388 | 0.5052 | 0.7834 | 0.6143 | 0.8418 | 0.6278 | 0.03s |
| 5 | Random Forest | 0.7800 | 0.5766 | 0.6444 | 0.6086 | 0.8356 | 0.6044 | 0.45s |
| 6 | Decision Tree | 0.7353 | 0.5009 | 0.7594 | 0.6036 | 0.8116 | 0.5948 | 0.03s |
| 7 | MLP Neural Network | **0.7942** | **0.6329** | 0.5348 | 0.5797 | 0.8443 | 0.5779 | 1.01s |
| 8 | Gradient Boosting | 0.7921 | 0.6278 | 0.5321 | 0.5760 | 0.8296 | 0.5697 | 2.61s |
| 9 | KNN | 0.7587 | 0.5478 | 0.5214 | 0.5342 | 0.7868 | 0.5291 | 0.00s |
| 10 | AdaBoost | 0.7842 | 0.6699 | 0.3690 | 0.4759 | 0.8408 | 0.4866 | 0.90s |

### Metrik Açıklamaları

| Metrik | Açıklama | Bu Projede Önemi |
|--------|----------|------------------|
| **Accuracy** | Tüm doğru tahminlerin oranı | Dengesiz veride yanıltıcı olabilir |
| **Precision** | Churn denen müşterilerden gerçekten churn olanların oranı | Yanlış alarm maliyeti |
| **Recall** | Gerçek churn müşterilerinden yakalanabilenlerin oranı | Kaçırılan müşteri maliyeti |
| **F1-Score** | Precision ve Recall'un harmonik ortalaması | **Ana karar metriğimiz** |
| **ROC-AUC** | Sınıflandırma eşiğinden bağımsız genel performans | Modelin ayırt edici gücü |
| **CV-F1** | 5-Fold Cross-Validation F1 ortalaması | Genelleme performansı |

### Model Kategorilerine Göre Karşılaştırma

#### Boosting Modelleri (En Başarılı Kategori)
```
LightGBM  ████████████████████████████████ 0.6266 (🥇)
XGBoost   ███████████████████████████████▉ 0.6258 (🥈)
Grad.Boost████████████████████████████▊    0.5760
AdaBoost  ████████████████████████▍        0.4759
```

#### Doğrusal & Çekirdek Modelleri
```
SVM       ███████████████████████████████▋ 0.6200 (🥉)
Log. Reg. ███████████████████████████████▍ 0.6143
```

#### Ağaç Tabanlı Modeller
```
Rnd Forest████████████████████████████████ 0.6086
Dec. Tree ██████████████████████████████▊  0.6036
```

#### Diğer Modeller
```
MLP NN    █████████████████████████████▋   0.5797
KNN       ███████████████████████████▎     0.5342
```

### Recall vs Precision Trade-off Analizi

Churn tahmininde **Recall** kritiktir çünkü kaçırılan her churn müşterisi doğrudan gelir kaybıdır.

| Kategori | Modeller | Recall | Precision | Yorum |
|----------|----------|--------|-----------|-------|
| **Yüksek Recall** | XGBoost, LightGBM, LR, SVM | 0.78-0.80 | 0.50-0.52 | Churn'ü iyi yakalar, bazı false alarm |
| **Dengeli** | Random Forest, Decision Tree | 0.64-0.76 | 0.50-0.58 | Orta düzey her iki metrikte |
| **Yüksek Precision** | AdaBoost, MLP, GB | 0.37-0.53 | 0.63-0.67 | Az false alarm ama çok churn kaçırır |

---

## Sınıf Dengesizliği Çözümü

Veri setinde sınıf dağılımı dengesizdir (~%74 No Churn vs ~%26 Churn). Bu dengesizlik, modellerin çoğunluk sınıfına (No Churn) yanlı öğrenmesine neden olur.

### Uygulanan Yöntem: `class_weight='balanced'`

Bu parametre, azınlık sınıfına (Churn) otomatik olarak daha yüksek ağırlık verir:

```
weight_churn = n_samples / (2 × n_churn) ≈ 2.83
weight_no_churn = n_samples / (2 × n_no_churn) ≈ 0.68
```

| Model | Yöntem |
|-------|--------|
| Logistic Regression | `class_weight='balanced'` |
| Decision Tree | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| SVM | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight=2.83` |
| LightGBM | `is_unbalance=True` |
| Gradient Boosting | Desteklemiyor (varsayılan) |
| AdaBoost | Desteklemiyor (varsayılan) |
| KNN | Desteklemiyor (`weights='distance'` ile kısmi çözüm) |
| MLP | Desteklemiyor (varsayılan) |

### Etki Analizi

`class_weight` uygulaması öncesi ve sonrası (Logistic Regression örneği):

| Durum | Recall | Precision | F1-Score |
|-------|--------|-----------|----------|
| `class_weight` yok | 0.5588 | 0.6593 | 0.6049 |
| `class_weight='balanced'` | **0.7834** | 0.5052 | **0.6143** |

Recall %40 arttı — daha fazla churn müşterisi yakalanıyor.

---

## Hiperparametre Optimizasyonu

XGBoost ve LightGBM için **RandomizedSearchCV** (30 iterasyon, 5-Fold CV) uygulandı.

### XGBoost - Optimize Edilen Parametreler

| Parametre | Arama Aralığı | Bulunan Değer |
|-----------|---------------|---------------|
| n_estimators | [100, 200, 300] | 200 |
| max_depth | [3, 5, 7, 9] | 3 |
| learning_rate | [0.01, 0.05, 0.1, 0.2] | 0.05 |
| subsample | [0.7, 0.8, 0.9, 1.0] | 0.9 |
| colsample_bytree | [0.7, 0.8, 0.9, 1.0] | 1.0 |
| min_child_weight | [1, 3, 5] | 1 |
| gamma | [0, 0.1, 0.2] | 0 |

### LightGBM - Optimize Edilen Parametreler

| Parametre | Arama Aralığı | Bulunan Değer |
|-----------|---------------|---------------|
| n_estimators | [100, 200, 300] | 300 |
| max_depth | [3, 5, 7, -1] | 5 |
| learning_rate | [0.01, 0.05, 0.1, 0.2] | 0.01 |
| subsample | [0.7, 0.8, 0.9, 1.0] | 0.9 |
| colsample_bytree | [0.7, 0.8, 0.9, 1.0] | 1.0 |
| num_leaves | [15, 31, 63] | 15 |
| min_child_samples | [5, 10, 20] | 20 |

---

## Sonuç ve Değerlendirme

### Neden LightGBM Seçildi?

1. **En yüksek F1-Score (0.6266):** Precision-Recall dengesinde en iyi performans
2. **Yüksek Recall (0.7941):** Churn müşterilerinin %79'unu yakalıyor
3. **Kararlı CV performansı (0.6281 ± 0.008):** Düşük varyans = güvenilir genelleme
4. **Hızlı eğitim (0.22s):** Üretim ortamında yeniden eğitim kolaylığı
5. **ROC-AUC (0.8439):** Güçlü ayırt edici yetenek

### XGBoost Neden Çok Yakın?

XGBoost (F1=0.6258) neredeyse eşdeğer performans gösterdi. Hatta **en yüksek Recall (0.8048)** ve **en yüksek ROC-AUC (0.8446)** bu modele ait. LightGBM'in CV-F1'deki küçük üstünlüğü (0.6281 vs 0.6273) ve daha düşük varyansı seçimi belirledi.

### Başarısız Kalan Modeller

| Model | Sorun |
|-------|-------|
| **AdaBoost** (F1=0.4759) | Sınıf dengesizliğini yönetemiyor, Recall çok düşük (0.37) |
| **KNN** (F1=0.5342) | Yüksek boyutlu veride "curse of dimensionality" etkisi |
| **Gradient Boosting** (F1=0.5760) | class_weight desteği yok, azınlık sınıfını yetersiz öğreniyor |

### Gelecek İyileştirme Önerileri

1. **Feature Engineering:** Tenure grupları, aylık/toplam ücret oranı gibi türetilmiş özellikler
2. **Stacking/Blending:** Top 3 modeli birleştiren ensemble yaklaşımı
3. **Threshold Tuning:** 0.5 yerine optimal karar eşiği belirleme
4. **Bayesian Optimization:** Hiperparametre aramasında daha verimli yöntem
