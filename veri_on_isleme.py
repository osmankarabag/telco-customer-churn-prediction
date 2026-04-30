"""
Telco Customer Churn - Veri Ön İşleme Script'i
================================================
Bu script aşağıdaki veri ön işleme adımlarını gerçekleştirir:
1. Veri Keşfi (EDA)
2. Eksik Veri Analizi ve İşleme
3. Veri Tipi Dönüşümleri
4. Aykırı Değer Analizi ve İşleme (IQR Yöntemi)
5. Kategorik Verilerin Dönüştürülmesi (Label Encoding & One-Hot Encoding)
6. Özellik Ölçeklendirme (StandardScaler)
7. İşlenmiş Verinin Kaydedilmesi
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışmak için
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import warnings
import os

warnings.filterwarnings('ignore')

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# Çıktı dizini
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, 'grafikler')
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("   TELCO CUSTOMER CHURN - VERİ ÖN İŞLEME RAPORU")
print("=" * 70)

# ============================================================
# 1. VERİ YÜKLEME VE KEŞİF
# ============================================================
print("\n" + "=" * 70)
print("  1. VERİ YÜKLEME VE KEŞİF")
print("=" * 70)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))

print(f"\n📊 Veri Seti Boyutu: {df.shape[0]} satır x {df.shape[1]} sütun")
print(f"\n📋 Sütunlar:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n📝 İlk 5 Satır:")
print(df.head().to_string())

print(f"\n📈 Veri Tipleri:")
print(df.dtypes.to_string())

print(f"\n📊 Temel İstatistikler (Sayısal):")
print(df.describe().to_string())

print(f"\n[>>] Temel Istatistikler (Kategorik):")
print(df.describe(include='object').to_string())

# 2. EKSİK VERI ANALIZI VE ISLEME
# ============================================================
print("\n" + "=" * 70)
print("  2. EKSIK VERI ANALIZI VE ISLEME")
print("=" * 70)

# --- 2a: Gizli bosluk (string) kontrolu ---
print("\n[>>] TotalCharges gizli bosluk kontrolu:")
bosluk_degerler = df[df['TotalCharges'] == ' ']
print(f"   Bosluk string degeri sayisi: {len(bosluk_degerler)}")
if len(bosluk_degerler) > 0:
    print(f"   Tenure degerleri: {bosluk_degerler['tenure'].tolist()}")
    print(f"   -> Tenure=0 olan musteriler henuz odeme yapmamis.")

# TotalCharges'i sayisala cevir (bosluklar NaN olacak)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# --- 2b: Gercek NaN analizi ---
print(f"\n[A] Gercek (NaN) Eksik Veri Tablosu:")
nan_sayisi = df.isnull().sum()
nan_goster = pd.DataFrame({
    'Eksik Sayisi': nan_sayisi,
    'Eksik Orani (%)': (nan_sayisi / len(df) * 100).round(4)
})
nan_goster_filtre = nan_goster[nan_goster['Eksik Sayisi'] > 0]
if len(nan_goster_filtre) > 0:
    print(nan_goster_filtre.to_string())
else:
    print("   Gercek NaN bulunamadi.")

# --- 2c: Anlamsal (semantik) eksik veri analizi ---
# 'No internet service' / 'No phone service' = o hizmet alinmiyor = mantiksal eksiklik
print(f"\n[B] Anlamsal/Ortulu Eksik Veri Analizi:")
print("   ('No internet service' / 'No phone service' degerleri)")
print("-" * 60)

internet_bagli_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
telefon_bagli_cols  = ['MultipleLines']

semantik_eksik = {}
for col in internet_bagli_cols:
    sayi = (df[col] == 'No internet service').sum()
    oran = sayi / len(df) * 100
    semantik_eksik[col] = {'sayi': int(sayi), 'oran': oran, 'neden': 'No internet service'}
    print(f"   {col:<22} -> 'No internet service': {sayi:5d} ({oran:.1f}%)")

for col in telefon_bagli_cols:
    sayi = (df[col] == 'No phone service').sum()
    oran = sayi / len(df) * 100
    semantik_eksik[col] = {'sayi': int(sayi), 'oran': oran, 'neden': 'No phone service'}
    print(f"   {col:<22} -> 'No phone service'  : {sayi:5d} ({oran:.1f}%)")

toplam_semantik = sum(v['sayi'] for v in semantik_eksik.values())
print(f"\n   Toplam anlamsal eksik deger sayisi: {toplam_semantik}")

# --- 2d: Birlesmis eksiklik matrisi (NaN=1, Semantik=2, Tam=0) ---
df_missing_map = df.isnull().astype(int).copy()
for col, info in semantik_eksik.items():
    mask = df[col] == info['neden']
    df_missing_map[col] = df_missing_map[col].where(~mask, 2)

# --- Grafik ---
print(f"\n[>>] Eksik veri haritasi olusturuluyor...")

fig = plt.figure(figsize=(22, 9))
gs  = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.1], wspace=0.35)
ax_heat = fig.add_subplot(gs[0])
ax_bar  = fig.add_subplot(gs[1])

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

cmap3 = ListedColormap(['#f0f0f0', '#e63946', '#f4a261'])
im = ax_heat.imshow(df_missing_map.values, aspect='auto',
                     cmap=cmap3, vmin=0, vmax=2, interpolation='none')

ax_heat.set_xticks(range(len(df_missing_map.columns)))
ax_heat.set_xticklabels(df_missing_map.columns, rotation=55, ha='right', fontsize=8)
ax_heat.set_yticks([])
ax_heat.set_title('Eksik Veri Isi Haritasi\n(Gri=Tam  Kirmizi=NaN  Turuncu=Anlamsal Eksik)',
                   fontsize=12, fontweight='bold')
ax_heat.set_xlabel('Sutunlar', fontsize=10)
ax_heat.set_ylabel('Gozlemler (7043 satir)', fontsize=10)

legend_h = [
    mpatches.Patch(color='#f0f0f0', label='Tam Veri', ec='gray'),
    mpatches.Patch(color='#e63946', label='Gercek NaN (TotalCharges: 11)'),
    mpatches.Patch(color='#f4a261', label='Anlamsal Eksik (No service)'),
]
ax_heat.legend(handles=legend_h, loc='upper right', fontsize=8, framealpha=0.9)

# Bar grafigi
eksik_toplam_dict = {}
for col in df.columns:
    nan_c = int(df[col].isnull().sum())
    sem_c = semantik_eksik.get(col, {}).get('sayi', 0)
    if nan_c + sem_c > 0:
        eksik_toplam_dict[col] = {'nan': nan_c, 'semantik': sem_c}

toplam_ser = pd.Series({c: v['nan'] + v['semantik'] for c, v in eksik_toplam_dict.items()})
toplam_ser = toplam_ser.sort_values(ascending=True)
pct_ser = toplam_ser / len(df) * 100

bar_colors = ['#e63946' if eksik_toplam_dict[c]['nan'] > 0 and eksik_toplam_dict[c]['semantik'] == 0
              else '#f4a261' for c in toplam_ser.index]

bars = ax_bar.barh(toplam_ser.index, pct_ser.values,
                    color=bar_colors, edgecolor='white', height=0.6)
ax_bar.set_xlabel('Eksik Oran (%)', fontsize=10)
ax_bar.set_title('Sutun Bazinda Eksiklik\n(NaN + Anlamsal)', fontsize=12, fontweight='bold')
ax_bar.tick_params(axis='y', labelsize=8)

for bar, col in zip(bars, toplam_ser.index):
    val_pct = pct_ser[col]
    val_adet = toplam_ser[col]
    ax_bar.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f'{val_pct:.1f}%  ({val_adet})',
                va='center', fontsize=7.5, fontweight='bold')
ax_bar.set_xlim(0, pct_ser.max() * 1.6)

legend_b = [mpatches.Patch(color='#e63946', label='Gercek NaN'),
            mpatches.Patch(color='#f4a261', label='Anlamsal Eksik')]
ax_bar.legend(handles=legend_b, fontsize=8, loc='lower right')

plt.suptitle('Eksik Veri Analizi - Orijinal Veri Seti (Gercek NaN + Anlamsal)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(PLOT_DIR, '01_eksik_veri_haritasi.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   [OK] Grafik kaydedildi: grafikler/01_eksik_veri_haritasi.png")

# === KNN Imputer: sadece TotalCharges NaN'larini doldur ===
print(f"\n[>>] Eksik Deger Isleme (KNN Imputer, k=5):")
n_eksik_once = int(df['TotalCharges'].isnull().sum())
if n_eksik_once > 0:
    sayisal_knn = ['tenure', 'MonthlyCharges', 'TotalCharges']
    imputer = KNNImputer(n_neighbors=5)
    df[sayisal_knn] = imputer.fit_transform(df[sayisal_knn])
    n_eksik_sonra = int(df['TotalCharges'].isnull().sum())
    print(f"   [OK] TotalCharges: {n_eksik_once} eksik deger KNN (k=5) ile dolduruldu.")
    print(f"        Kullanilan ozellikler: {sayisal_knn}")
    print(f"        Doldurma sonrasi eksik sayisi: {n_eksik_sonra}")
else:
    print(f"   [INFO] TotalCharges'ta eksik deger bulunmadi.")

print(f"   [INFO] Anlamsal eksikler ('No internet/phone service') kategorik")
print(f"         degisken olarak korunuyor; One-Hot Encoding ile islenecek.")
print(f"\n   Eksik veri kontrolu (islem sonrasi): {df.isnull().sum().sum()} eksik deger")

# ============================================================
# 3. VERİ TİPİ DÖNÜŞÜMLERİ
# ============================================================
print("\n" + "=" * 70)
print("  3. VERİ TİPİ DÖNÜŞÜMLERİ")
print("=" * 70)

# SeniorCitizen: 0/1 → kategorik olarak düşünülebilir ama zaten sayısal
print(f"\n🔧 SeniorCitizen sütunu zaten 0/1 (sayısal) formatında.")
print(f"   Değer dağılımı: {dict(df['SeniorCitizen'].value_counts())}")

# TotalCharges: string → float (zaten yukarıda yapıldı)
print(f"\n🔧 TotalCharges sütunu float64'e dönüştürüldü.")
print(f"   Veri tipi: {df['TotalCharges'].dtype}")

print(f"\n📊 Güncellenmiş Veri Tipleri:")
sayisal = df.select_dtypes(include=[np.number]).columns.tolist()
kategorik = df.select_dtypes(include=['object']).columns.tolist()
print(f"   Sayısal sütunlar ({len(sayisal)}): {sayisal}")
print(f"   Kategorik sütunlar ({len(kategorik)}): {kategorik}")

# ============================================================
# 4. KATEGORİK VERİ ANALİZİ
# ============================================================
print("\n" + "=" * 70)
print("  4. KATEGORİK VERİ ANALİZİ")
print("=" * 70)

for col in kategorik:
    if col == 'customerID':
        continue
    print(f"\n📋 {col}:")
    print(f"   Benzersiz değerler: {df[col].nunique()}")
    vc = df[col].value_counts()
    for val, count in vc.items():
        pct = count / len(df) * 100
        print(f"   - {val}: {count} ({pct:.1f}%)")

# Kategorik değişkenlerin bar grafikleri
kat_cols = [col for col in kategorik if col != 'customerID']
n_cols = 3
n_rows = (len(kat_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(kat_cols):
    sns.countplot(data=df, x=col, hue='Churn', ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='Churn', fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Kategorik Degiskenlerin Churn ile Iliskisi', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_kategorik_dagilimlar.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n   📊 Grafik kaydedildi: grafikler/02_kategorik_dagilimlar.png")

# ============================================================
# 5. AYKIRI DEĞER ANALİZİ VE İŞLEME (IQR YÖNTEMİ)
# ============================================================
print("\n" + "=" * 70)
print("  5. AYKIRI DEGER ANALİZİ VE İŞLEME (IQR YÖNTEMİ)")
print("=" * 70)

sayisal_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

print(f"\n📊 Sayısal Değişkenler İçin Aykırı Değer Analizi:")
print("-" * 65)
print(f"{'Sütun':<18} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Alt Sınır':>10} {'Üst Sınır':>10} {'Aykırı':>8}")
print("-" * 65)

aykiri_bilgi = {}
for col in sayisal_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    aykiri_sayi = ((df[col] < alt_sinir) | (df[col] > ust_sinir)).sum()
    aykiri_bilgi[col] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'alt_sinir': alt_sinir, 'ust_sinir': ust_sinir,
        'aykiri_sayi': aykiri_sayi
    }
    print(f"{col:<18} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} {alt_sinir:>10.2f} {ust_sinir:>10.2f} {aykiri_sayi:>8}")

# Boxplot grafikleri
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(sayisal_cols):
    sns.boxplot(data=df, y=col, ax=axes[i], color='#5B9BD5', width=0.4)
    axes[i].set_title(f'{col}\n(Aykiri: {aykiri_bilgi[col]["aykiri_sayi"]})', fontsize=11, fontweight='bold')
    axes[i].axhline(y=aykiri_bilgi[col]['ust_sinir'], color='red', linestyle='--', alpha=0.7, label='Ust Sinir')
    axes[i].axhline(y=aykiri_bilgi[col]['alt_sinir'], color='red', linestyle='--', alpha=0.7, label='Alt Sinir')
    axes[i].legend(fontsize=8)

plt.suptitle('Sayisal Degiskenlerde Aykiri Deger Analizi (IQR)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_aykiri_deger_boxplot.png'), dpi=150)
plt.close()
print(f"\n   📊 Grafik kaydedildi: grafikler/03_aykiri_deger_boxplot.png")

# Aykırı değerleri sınır değerlerine çekme (Capping / Winsorization)
print(f"\n🔧 Aykırı Değer İşleme (Capping/Winsorization):")
for col in sayisal_cols:
    info = aykiri_bilgi[col]
    if info['aykiri_sayi'] > 0:
        onceki = df[col].copy()
        df[col] = df[col].clip(lower=info['alt_sinir'], upper=info['ust_sinir'])
        degisen = (onceki != df[col]).sum()
        print(f"   ✅ {col}: {degisen} aykırı değer sınır değerlerine çekildi.")
    else:
        print(f"   ℹ️  {col}: Aykırı değer bulunmadı, işlem yapılmadı.")

# İşlem sonrası boxplot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(sayisal_cols):
    sns.boxplot(data=df, y=col, ax=axes[i], color='#70AD47', width=0.4)
    axes[i].set_title(f'{col} (Islem Sonrasi)', fontsize=11, fontweight='bold')

plt.suptitle('Aykiri Deger Isleme Sonrasi Boxplot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_aykiri_deger_sonrasi.png'), dpi=150)
plt.close()
print(f"   📊 Grafik kaydedildi: grafikler/04_aykiri_deger_sonrasi.png")

# ============================================================
# 6. KATEGORİK VERİLERİN DÖNÜŞTÜRÜLMESİ
# ============================================================
print("\n" + "=" * 70)
print("  6. KATEGORİK VERİLERİN DÖNÜŞTÜRÜLMESI")
print("=" * 70)

# customerID'yi kaldır (model için gerekli değil)
print(f"\n🔧 customerID sütunu kaldırıldı (model için gerekli değil).")
df_processed = df.drop('customerID', axis=1)

# Binary (İkili) kategorik değişkenler → Label Encoding
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

print(f"\n📋 Label Encoding (İkili Değişkenler):")
le = LabelEncoder()
for col in binary_cols:
    onceki = df_processed[col].unique()
    df_processed[col] = le.fit_transform(df_processed[col])
    sonraki = df_processed[col].unique()
    print(f"   {col}: {onceki} → {sonraki}")

# Çok sınıflı kategorik değişkenler → One-Hot Encoding
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']

print(f"\n📋 One-Hot Encoding (Çok Sınıflı Değişkenler):")
for col in multi_cols:
    print(f"   {col}: {df_processed[col].unique().tolist()}")

df_encoded = pd.get_dummies(df_processed, columns=multi_cols, drop_first=True, dtype=int)

print(f"\n   ✅ One-Hot Encoding uygulandı.")
print(f"   Encoding öncesi sütun sayısı: {df_processed.shape[1]}")
print(f"   Encoding sonrası sütun sayısı: {df_encoded.shape[1]}")
print(f"\n   Yeni oluşan sütunlar:")
yeni_sutunlar = [col for col in df_encoded.columns if col not in df_processed.columns]
for s in yeni_sutunlar:
    print(f"   - {s}")

# ============================================================
# 7. ÖZELLİK ÖLÇEKLENDİRME (StandardScaler)
# ============================================================
print("\n" + "=" * 70)
print("  7. ÖZELLİK ÖLÇEKLENDİRME (StandardScaler)")
print("=" * 70)

olceklenecek_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

print(f"\n📊 Ölçekleme Öncesi İstatistikler:")
print(df_encoded[olceklenecek_cols].describe().to_string())

scaler = StandardScaler()
df_encoded[olceklenecek_cols] = scaler.fit_transform(df_encoded[olceklenecek_cols])

print(f"\n📊 Ölçekleme Sonrası İstatistikler:")
print(df_encoded[olceklenecek_cols].describe().to_string())
print(f"\n   ✅ {olceklenecek_cols} sütunları StandardScaler ile ölçeklendirildi.")

# Dağılım grafikleri
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(olceklenecek_cols):
    sns.histplot(data=df_encoded, x=col, kde=True, ax=axes[i], color='#5B9BD5', bins=30)
    axes[i].set_title(f'{col} (Olceklenmis)', fontsize=11, fontweight='bold')
    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)

plt.suptitle('Olcekleme Sonrasi Dagilimlar', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '05_olcekleme_sonrasi.png'), dpi=150)
plt.close()
print(f"   📊 Grafik kaydedildi: grafikler/05_olcekleme_sonrasi.png")

# ============================================================
# 8. KORELASYON ANALİZİ
# ============================================================
print("\n" + "=" * 70)
print("  8. KORELASYON ANALİZİ")
print("=" * 70)

# Churn ile en yüksek korelasyona sahip özellikler
korelasyon = df_encoded.corr()['Churn'].sort_values(ascending=False)
print(f"\n📊 Churn ile En Yüksek Korelasyona Sahip Değişkenler:")
print("-" * 45)
for feat, corr in korelasyon.items():
    if feat != 'Churn':
        bar = "█" * int(abs(corr) * 30)
        sign = "+" if corr > 0 else "-"
        print(f"   {feat:<40} {sign}{abs(corr):.4f} {bar}")

# Korelasyon ısı haritası (sayısal + önemli sütunlar)
fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(df_encoded.corr(), annot=False, cmap='RdBu_r', center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Korelasyon Isi Haritasi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_korelasyon_haritasi.png'), dpi=150)
plt.close()
print(f"\n   📊 Grafik kaydedildi: grafikler/06_korelasyon_haritasi.png")

# ============================================================
# 9. İŞLENMİŞ VERİYİ KAYDETME
# ============================================================
print("\n" + "=" * 70)
print("  9. İŞLENMİŞ VERİYİ KAYDETME")
print("=" * 70)

output_file = os.path.join(OUTPUT_DIR, 'telco_churn_preprocessed.csv')
df_encoded.to_csv(output_file, index=False)
print(f"\n   ✅ İşlenmiş veri kaydedildi: telco_churn_preprocessed.csv")
print(f"   Satır sayısı: {df_encoded.shape[0]}")
print(f"   Sütun sayısı: {df_encoded.shape[1]}")

print(f"\n   📋 Son Sütun Listesi:")
for i, col in enumerate(df_encoded.columns, 1):
    print(f"   {i:2d}. {col} (tip: {df_encoded[col].dtype})")

# ============================================================
# 10. ÖZET RAPOR
# ============================================================
print("\n" + "=" * 70)
print("  ÖZET RAPOR")
print("=" * 70)

print(f"""
  📊 Orijinal veri seti: 7043 satır x 21 sütun
  📊 İşlenmiş veri seti: {df_encoded.shape[0]} satır x {df_encoded.shape[1]} sütun

  🔧 Yapılan İşlemler:
     1. ✅ TotalCharges boşluk değerleri (11 adet) → KNN Imputer (k=5) ile dolduruldu
     2. ✅ TotalCharges string → float64 dönüşümü yapıldı
     3. ✅ Aykırı değer analizi (IQR yöntemi) yapıldı ve capping uygulandı
     4. ✅ İkili kategorik değişkenler Label Encoding ile dönüştürüldü
        ({', '.join(binary_cols)})
     5. ✅ Çok sınıflı kategorik değişkenler One-Hot Encoding ile dönüştürüldü
        ({', '.join(multi_cols)})
     6. ✅ Sayısal değişkenler StandardScaler ile ölçeklendirildi
        ({', '.join(olceklenecek_cols)})
     7. ✅ customerID sütunu kaldırıldı

  📂 Çıktı Dosyaları:
     - telco_churn_preprocessed.csv (işlenmiş veri)
     - grafikler/01_eksik_veri_haritasi.png
     - grafikler/02_kategorik_dagilimlar.png
     - grafikler/03_aykiri_deger_boxplot.png
     - grafikler/04_aykiri_deger_sonrasi.png
     - grafikler/05_olcekleme_sonrasi.png
     - grafikler/06_korelasyon_haritasi.png
""")

print("=" * 70)
print("  VERİ ÖN İŞLEME TAMAMLANDI!")
print("=" * 70)
