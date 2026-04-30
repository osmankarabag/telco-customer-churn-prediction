"""
Telco Müşteri Kaybı (Churn) - FastAPI Servisi
==============================================
Bu servis, eğitilmiş makine öğrenmesi modeli üzerinden müşteri kaybı tahmini yapar.

Endpoint'ler:
  GET  /           -> Web arayüzü (HTML Form)
  GET  /health     -> Servis sağlık kontrolü
  GET  /model-info -> Model performans bilgileri
  POST /predict    -> Müşteri kaybı tahmini
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import json
import numpy as np
import pandas as pd
import os

# ============================================================
# YAPILANDIRMA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ============================================================
# MODEL VE PIPELINE YUKLEME
# ============================================================
print("[*] Model ve preprocessing pipeline yukleniyor...")

# En iyi model
with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Scaler
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Preprocessing bilgileri
with open(os.path.join(MODEL_DIR, 'preprocessing_info.json'), 'r', encoding='utf-8') as f:
    preprocessing_info = json.load(f)

# Model metadata
with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r', encoding='utf-8') as f:
    model_metadata = json.load(f)

# Optimal threshold
OPTIMAL_THRESHOLD = model_metadata.get('optimal_threshold', 0.5)

print(f"[OK] Model yuklendi: {model_metadata['best_model']}")
print(f"[OK] Optimal threshold: {OPTIMAL_THRESHOLD}")
print(f"[OK] F1-Score: {model_metadata['metrics']['f1_score']}")

# ============================================================
# FASTAPI UYGULAMASI
# ============================================================
app = FastAPI(
    title="📡 Telco Müşteri Kaybı Tahmin API'si",
    description="""
## 🎯 Nedir Bu API?
Bu servis, bir telekomunikasyon şirketinin müşterilerinin **abonelikten ayrılıp ayrılmayacağını** tahmin eden makine öğrenmesi modelini sunar.

## 📋 Kullanılabilir Endpoint'ler
| Method | Yol | Açıklama |
|--------|-----|----------|
| `GET` | `/` | Web arayüzü (HTML Form) |
| `GET` | `/health` | Servis sağlık kontrolü |
| `GET` | `/model-info` | Model bilgileri ve metrikleri |
| `POST` | `/predict` | **Müşteri kaybı tahmini** |

## ⚡ Hızlı Başlangıç
`POST /predict` endpoint'ine müşteri bilgilerini gönderin ve anında tahmin alın!

## 📊 Risk Seviyeleri
- 🟢 **Düşük Risk**: Churn olasılığı < %30
- 🟡 **Orta Risk**: Churn olasılığı %30-%60 arası
- 🔴 **Yüksek Risk**: Churn olasılığı > %60
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Telco Churn Tahmin Sistemi",
        "email": "destek@telcochurn.com"
    },
    license_info={
        "name": "MIT Lisansı"
    }
)

# Static dosyalar ve template'ler
static_dir = os.path.join(BASE_DIR, 'static')
templates_dir = os.path.join(BASE_DIR, 'templates')
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# ============================================================
# PYDANTIC MODELLERI
# ============================================================
class CustomerInput(BaseModel):
    """Müşteri bilgileri giriş modeli — tüm alanlar zorunludur"""

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }

    gender: str = Field(..., description="👤 Cinsiyet — Kabul edilen değerler: `Male` (Erkek) veya `Female` (Kadın)", example="Male")
    SeniorCitizen: int = Field(..., description="👴 Yaşlı müşteri mi? — `0` = Hayır, `1` = Evet", example=0)
    Partner: str = Field(..., description="💑 Eşi/partneri var mı? — `Yes` (Evet) veya `No` (Hayır)", example="Yes")
    Dependents: str = Field(..., description="👨‍👧 Bakmakla yükümlü olduğu kişi var mı? — `Yes` veya `No`", example="No")
    tenure: int = Field(..., description="📅 Müşteri kaç aydır abone? (0-72 arası tam sayı)", example=12)
    PhoneService: str = Field(..., description="📞 Telefon servisi kullanıyor mu? — `Yes` veya `No`", example="Yes")
    MultipleLines: str = Field(..., description="📱 Birden fazla hattı var mı? — `Yes`, `No` veya `No phone service` (Telefon servisi yok)", example="No")
    InternetService: str = Field(..., description="🌐 İnternet servis türü — `DSL`, `Fiber optic` (Fiber optik) veya `No` (Yok)", example="DSL")
    OnlineSecurity: str = Field(..., description="🔒 Çevrimiçi güvenlik hizmeti var mı? — `Yes`, `No` veya `No internet service` (İnternet servisi yok)", example="Yes")
    OnlineBackup: str = Field(..., description="☁️ Çevrimiçi yedekleme hizmeti var mı? — `Yes`, `No` veya `No internet service`", example="No")
    DeviceProtection: str = Field(..., description="🛡️ Cihaz koruma hizmeti var mı? — `Yes`, `No` veya `No internet service`", example="No")
    TechSupport: str = Field(..., description="🛠️ Teknik destek hizmeti var mı? — `Yes`, `No` veya `No internet service`", example="No")
    StreamingTV: str = Field(..., description="📺 TV yayın akışı hizmeti var mı? — `Yes`, `No` veya `No internet service`", example="No")
    StreamingMovies: str = Field(..., description="🎬 Film yayın akışı hizmeti var mı? — `Yes`, `No` veya `No internet service`", example="No")
    Contract: str = Field(..., description="📄 Sözleşme türü — `Month-to-month` (Aylık), `One year` (1 Yıllık) veya `Two year` (2 Yıllık)", example="Month-to-month")
    PaperlessBilling: str = Field(..., description="📧 Kağıtsız (e-fatura) kullanıyor mu? — `Yes` veya `No`", example="Yes")
    PaymentMethod: str = Field(..., description="💳 Ödeme yöntemi — `Electronic check` (Elektronik çek), `Mailed check` (Posta çeki), `Bank transfer (automatic)` (Banka transferi otomatik), `Credit card (automatic)` (Kredi kartı otomatik)", example="Electronic check")
    MonthlyCharges: float = Field(..., description="💰 Aylık ücret (TL/USD cinsinden)", example=56.95)
    TotalCharges: float = Field(..., description="💵 Toplam ücret — müşterinin bugüne kadar ödediği toplam tutar", example=683.4)


class PredictionOutput(BaseModel):
    """Tahmin sonucu çıkış modeli"""
    prediction: str = Field(..., description="🎯 Tahmin sonucu — `Churn` (Müşteri ayrılacak) veya `No Churn` (Müşteri kalacak)")
    churn_probability: float = Field(..., description="📊 Müşterinin ayrılma olasılığı (0.0 ile 1.0 arasında — örn: 0.52 = %52)")
    no_churn_probability: float = Field(..., description="📊 Müşterinin kalma olasılığı (0.0 ile 1.0 arasında)")
    confidence: str = Field(..., description="✅ Modelin tahminine olan güven yüzdesi (örn: '72.5%')")
    risk_level: str = Field(..., description="⚠️ Müşteri kaybı risk seviyesi — `Dusuk` (< %30), `Orta` (%30-%60), `Yuksek` (> %60)")


# ============================================================
# ON ISLEME FONKSIYONU
# ============================================================
def preprocess_input(customer: CustomerInput) -> pd.DataFrame:
    """
    Ham musteri verisini model icin uygun formata donusturur.
    Ayni pipeline: Binary Encoding -> One-Hot Encoding -> Outlier Capping -> Scaling
    """
    # Veriyi dict'e cevir
    data = customer.model_dump()

    # 1. Binary Label Encoding
    binary_mappings = preprocessing_info['binary_mappings']
    binary_fields = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for col in binary_fields:
        mapping = binary_mappings[col]
        value = data[col]
        if value in mapping:
            data[col] = mapping[value]
        else:
            data[col] = 0  # Bilinmeyen deger icin varsayilan

    # 2. Multi-class One-Hot Encoding
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']

    # Once tum feature sutunlarini olustur (0 ile)
    feature_columns = preprocessing_info['feature_columns']
    row = {col: 0 for col in feature_columns}

    # Binary encoded degerleri koy
    for col in binary_fields:
        row[col] = data[col]

    # Sayisal degerleri koy
    row['SeniorCitizen'] = data['SeniorCitizen']
    row['tenure'] = data['tenure']
    row['MonthlyCharges'] = data['MonthlyCharges']
    row['TotalCharges'] = data['TotalCharges']

    # Feature Engineering
    tenure = data['tenure']
    monthly = data['MonthlyCharges']
    total = data['TotalCharges']
    row['AvgMonthlyCharge'] = total / (tenure + 1)
    row['ChargeRatio'] = monthly / (row['AvgMonthlyCharge'] + 0.01)
    if tenure <= 12: row['tenure_group'] = 0
    elif tenure <= 24: row['tenure_group'] = 1
    elif tenure <= 48: row['tenure_group'] = 2
    elif tenure <= 60: row['tenure_group'] = 3
    else: row['tenure_group'] = 4
    svc = sum(1 for s in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if data.get(s) == 'Yes')
    row['TotalServices'] = svc
    row['HasSecurity'] = sum(1 for s in ['OnlineSecurity','TechSupport','DeviceProtection'] if data.get(s) == 'Yes')
    row['HasStreaming'] = sum(1 for s in ['StreamingTV','StreamingMovies'] if data.get(s) == 'Yes')
    row['IsNewCustomer'] = 1 if tenure <= 12 else 0
    row['HasFamily'] = 1 if data.get('Partner') == 'Yes' or data.get('Dependents') == 'Yes' else 0
    contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
    row['ContractRisk'] = contract_risk.get(data.get('Contract', ''), 1)
    row['tenure_x_Monthly'] = tenure * monthly

    # One-Hot encoded sutunlari ayarla
    for col in multi_cols:
        value = data[col]
        col_name = f"{col}_{value}"
        if col_name in row:
            row[col_name] = 1

    # DataFrame'e cevir
    df = pd.DataFrame([row], columns=feature_columns)

    # 3. Outlier Capping
    outlier_bounds = preprocessing_info.get('outlier_bounds', {})
    for col, bounds in outlier_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds['lower'], upper=bounds['upper'])

    # 4. StandardScaler
    numerical_cols = preprocessing_info.get('numerical_cols', ['tenure', 'MonthlyCharges', 'TotalCharges'])
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["🖥️ Web Arayüzü"])
async def home(request: Request):
    """
    🏠 **Ana Sayfa — Web Arayüzü**

    Müşteri bilgilerini doldurabileceğiniz interaktif HTML formunu döndürür.
    Tarayıcıdan `http://localhost:8000` adresine giderek kullanabilirsiniz.
    """
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_name": model_metadata['best_model'],
            "accuracy": model_metadata['metrics']['accuracy'],
        }
    )


@app.get("/health", tags=["⚙️ Sistem"])
async def health_check():
    """
    💚 **Servis Sağlık Kontrolü**

    API'nin çalışır durumda olup olmadığını kontrol eder.
    Dönen `status: healthy` değeri, servisin sorunsuz çalıştığını gösterir.
    """
    return {
        "durum": "sağlıklı",
        "model_yuklendi": model is not None,
        "model_adi": model_metadata['best_model'],
        "versiyon": "1.0.0",
        "aciklama": "Servis çalışıyor ve tahmin yapmaya hazır."
    }


@app.get("/model-info", tags=["📊 Model Bilgileri"])
async def model_info():
    """
    📈 **Model Bilgileri ve Performans Metrikleri**

    Eğitilmiş makine öğrenmesi modelinin detaylarını döndürür:
    - Hangi algoritmanın seçildiği
    - Doğruluk (accuracy), F1-Score, AUC gibi metrikler
    - Karşılaştırılan tüm modeller
    - Eğitim/test veri boyutları
    """
    return {
        "en_iyi_model": model_metadata['best_model'],
        "performans_metrikleri": model_metadata['metrics'],
        "tum_modeller": model_metadata['all_models'],
        "toplam_ozellik_sayisi": model_metadata['total_features'],
        "egitim_veri_boyutu": model_metadata['train_size'],
        "test_veri_boyutu": model_metadata['test_size'],
        "aciklama": "Daha yüksek F1-Score ve AUC değeri, daha iyi model performansını gösterir."
    }


@app.post("/predict", response_model=PredictionOutput, tags=["🎯 Churn Tahmini"])
async def predict(customer: CustomerInput):
    """
    🔮 **Müşteri Kaybı (Churn) Tahmini**

    Girilen müşteri bilgilerini alır, ön işleme uygular ve eğitilmiş model ile tahmin üretir.

    ### 📥 Giriş
    Müşteriye ait 19 adet bilgi (demografik, sözleşme, hizmet bilgileri)

    ### 📤 Çıkış
    | Alan | Açıklama |
    |------|----------|
    | `prediction` | `Churn` (ayrılacak) veya `No Churn` (kalacak) |
    | `churn_probability` | Ayrılma olasılığı (0.0–1.0) |
    | `no_churn_probability` | Kalma olasılığı (0.0–1.0) |
    | `confidence` | Modelin güven yüzdesi |
    | `risk_level` | `Dusuk` / `Orta` / `Yuksek` |

    ### ⚠️ Risk Seviyeleri
    - 🟢 **Düşük** → Churn olasılığı < %30 → Müşteri muhtemelen kalacak
    - 🟡 **Orta** → Churn olasılığı %30–%60 → İzlenmeli, önlem alınabilir
    - 🔴 **Yüksek** → Churn olasılığı > %60 → Acil aksiyon önerilir
    """
    try:
        # On isleme
        processed_data = preprocess_input(customer)

        # Tahmin (optimal threshold ile)
        probabilities = model.predict_proba(processed_data)[0]
        churn_prob = float(probabilities[1])
        no_churn_prob = float(probabilities[0])
        prediction = 1 if churn_prob >= OPTIMAL_THRESHOLD else 0

        # Risk seviyesi
        if churn_prob < 0.3:
            risk_level = "Dusuk"
        elif churn_prob < 0.6:
            risk_level = "Orta"
        else:
            risk_level = "Yuksek"

        # Guven
        confidence = f"{max(churn_prob, no_churn_prob) * 100:.1f}%"

        return PredictionOutput(
            prediction="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(churn_prob, 4),
            no_churn_probability=round(no_churn_prob, 4),
            confidence=confidence,
            risk_level=risk_level
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Tahmin hatasi: {str(e)}"}
        )


# ============================================================
# CALISTIRMA
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n[*] Telco Churn Prediction API baslatiliyor...")
    print("[*] Swagger UI: http://localhost:8000/docs")
    print("[*] Web Arayuzu: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
