# Telco Customer Churn Prediction

Telco musteri kayip (churn) tahmini icin uctan uca makine ogrenmesi projesi.

## Proje Hakkinda

Bu proje, bir telekomunikasyon sirketinin musterilerinin hizmeti birakip birakmayacagini tahmin etmek icin gelistirilmistir. Proje asagidaki adimlari kapsar:

1. **Veri On Isleme** - Eksik veri, aykiri deger, encoding, olcekleme
2. **Model Gelistirme** - 10 farkli ML modelinin egitimi ve karsilastirmasi
3. **API Servisi** - FastAPI tabanli REST API
4. **Web Arayuzu** - Modern dark-mode tahmin arayuzu
5. **Docker** - Konteyner destegi

## Veri Seti

- **Kaynak:** IBM Telco Customer Churn Dataset
- **Boyut:** 7.043 musteri, 21 ozellik
- **Hedef:** `Churn` (Evet/Hayir)

## Model Performansi

10 farkli model egitilmis ve karsilastirilmistir (detaylar icin [MODEL_DOKUMANTASYON.md](MODEL_DOKUMANTASYON.md)):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **LightGBM** | 0.7488 | 0.5174 | **0.7941** | **0.6266** | 0.8439 |
| **XGBoost** | 0.7445 | 0.5119 | **0.8048** | 0.6258 | **0.8446** |
| SVM | 0.7459 | 0.5141 | 0.7807 | 0.6200 | 0.8249 |
| Logistic Regression | 0.7388 | 0.5052 | 0.7834 | 0.6143 | 0.8418 |
| Random Forest | 0.7800 | 0.5766 | 0.6444 | 0.6086 | 0.8356 |
| Decision Tree | 0.7353 | 0.5009 | 0.7594 | 0.6036 | 0.8116 |
| MLP Neural Network | 0.7942 | 0.6329 | 0.5348 | 0.5797 | 0.8443 |
| Gradient Boosting | 0.7921 | 0.6278 | 0.5321 | 0.5760 | 0.8296 |
| KNN | 0.7587 | 0.5478 | 0.5214 | 0.5342 | 0.7868 |
| AdaBoost | 0.7842 | 0.6699 | 0.3690 | 0.4759 | 0.8408 |

**En iyi model:** LightGBM (F1-Score bazinda, class_weight + RandomizedSearchCV ile optimize edilmistir)

## Proje Yapisi

```
YZTA/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Ham veri seti
├── telco_churn_preprocessed.csv           # Islenmis veri
├── veri_on_isleme.py                      # Veri on isleme scripti
├── model_gelistirme.py                    # Model egitim scripti
├── app.py                                 # FastAPI servisi
├── requirements.txt                       # Python bagimliliklari
├── Dockerfile                             # Docker yapilandirmasi
├── README.md                              # Bu dosya
├── models/
│   ├── best_model.pkl                     # Egitilmis model
│   ├── scaler.pkl                         # StandardScaler
│   ├── preprocessing_info.json            # On isleme bilgileri
│   └── model_metadata.json                # Model metrikleri
├── templates/
│   └── index.html                         # Web arayuzu
├── static/
│   └── style.css                          # CSS stilleri
└── grafikler/                             # EDA ve model grafikleri
    ├── 01_eksik_veri_haritasi.png
    ├── 02_kategorik_dagilimlar.png
    ├── ...
    ├── 09_roc_curves.png
    ├── 10_feature_importance.png
    └── 11_cv_karsilastirma.png
```

## Kurulum

### Gereksinimler
- Python 3.10+
- pip

### Adimlar

```bash
# 1. Depoyu klonla
git clone <repo-url>
cd YZTA

# 2. Bagimliliklari yukle
pip install -r requirements.txt

# 3. (Opsiyonel) Modeli yeniden egit
python model_gelistirme.py

# 4. API'yi baslat
python app.py
```

Servis http://localhost:8000 adresinde calisacaktir.

## API Kullanimi

### Saglik Kontrolu
```bash
curl http://localhost:8000/health
```

### Model Bilgileri
```bash
curl http://localhost:8000/model-info
```

### Tahmin (Predict)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 56.95,
    "TotalCharges": 683.4
  }'
```

**Ornek Cikti:**
```json
{
  "prediction": "No Churn",
  "churn_probability": 0.2341,
  "no_churn_probability": 0.7659,
  "confidence": "76.6%",
  "risk_level": "Dusuk"
}
```

### Swagger UI (Interaktif Dokumantasyon)
```
http://localhost:8000/docs
```

### Web Arayuzu
```
http://localhost:8000/
```

## Docker ile Calistirma

```bash
# Image olustur
docker build -t telco-churn-api .

# Container baslat
docker run -p 8000:8000 telco-churn-api
```

## Teknolojiler

- **Python 3.12** - Ana programlama dili
- **scikit-learn** - Makine ogrenmesi
- **XGBoost** - Gradient boosting (optimize edilmis)
- **LightGBM** - Hizli gradient boosting
- **FastAPI** - Web framework / REST API
- **Pandas / NumPy** - Veri isleme
- **Matplotlib / Seaborn** - Gorsellestirme
- **Jinja2** - HTML template
- **Docker** - Konteyner (opsiyonel)

## Veri On Isleme Adimlari

1. `TotalCharges` bosluk degerleri sayisala cevrilerek 0 ile dolduruldu
2. `customerID` sutunu kaldirildi
3. Ikili kategorik degiskenler Label Encoding ile donusturuldu
4. Cok sinifli kategorik degiskenler One-Hot Encoding ile donusturuldu
5. Aykiri degerler IQR yontemiyle sinirlandirildi
6. Sayisal degiskenler StandardScaler ile olceklendirildi
