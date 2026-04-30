"""Feature Engineering modulu - Telco Churn"""
import pandas as pd
import numpy as np

def create_features(df):
    """Ham veri uzerinde feature engineering uygular. Encoding ONCESI cagrilmali."""
    # 1. Ortalama aylik ucret
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # 2. Ucret degisim orani
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['AvgMonthlyCharge'] + 0.01)
    
    # 3. Tenure grubu
    bins = [0, 12, 24, 48, 60, 73]
    labels = [0, 1, 2, 3, 4]
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True).astype(int)
    
    # 4. Toplam servis sayisi
    svc_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies']
    df['TotalServices'] = 0
    for c in svc_cols:
        if c in df.columns:
            df['TotalServices'] += (df[c] == 'Yes').astype(int)
    
    # 5. Guvenlik paketi
    sec_cols = ['OnlineSecurity','TechSupport','DeviceProtection']
    df['HasSecurity'] = 0
    for c in sec_cols:
        if c in df.columns:
            df['HasSecurity'] += (df[c] == 'Yes').astype(int)
    
    # 6. Eglence paketi
    df['HasStreaming'] = 0
    for c in ['StreamingTV','StreamingMovies']:
        if c in df.columns:
            df['HasStreaming'] += (df[c] == 'Yes').astype(int)
    
    # 7. Yeni musteri
    df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
    
    # 8. Aile bagliligi
    partner_val = (df['Partner'] == 'Yes') if df['Partner'].dtype == object else (df['Partner'] == 1)
    dep_val = (df['Dependents'] == 'Yes') if df['Dependents'].dtype == object else (df['Dependents'] == 1)
    df['HasFamily'] = (partner_val | dep_val).astype(int)
    
    # 9. Sozlesme risk skoru
    if df['Contract'].dtype == object:
        contract_map = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
        df['ContractRisk'] = df['Contract'].map(contract_map).fillna(1).astype(int)
    
    # 10. Etkilesim terimi
    df['tenure_x_Monthly'] = df['tenure'] * df['MonthlyCharges']
    
    return df

# Feature engineering sonrasi eklenen numerik kolonlar
ENGINEERED_NUMERICAL = ['AvgMonthlyCharge', 'ChargeRatio', 'tenure_x_Monthly']
ENGINEERED_OTHER = ['tenure_group', 'TotalServices', 'HasSecurity', 
                     'HasStreaming', 'IsNewCustomer', 'HasFamily', 'ContractRisk']
