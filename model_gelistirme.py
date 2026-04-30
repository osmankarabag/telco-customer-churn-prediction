"""
Telco Churn - Gelismis Model Gelistirme v3
Feature Selection + Optuna HP Tuning + CatBoost
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna
from feature_engineering import create_features, ENGINEERED_NUMERICAL
import pickle
import json
import os
import warnings
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams['font.family'] = 'DejaVu Sans'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR = os.path.join(BASE_DIR, 'grafikler')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("   TELCO CHURN - GELISMIS MODEL v3 (FE + FS + Optuna + CatBoost)")
print("=" * 70)

# === 1. VERI YUKLEME ===
print("\n" + "=" * 70)
print("  1. VERI YUKLEME VE FEATURE ENGINEERING")
print("=" * 70)
df = pd.read_csv(os.path.join(BASE_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
print(f"\n   [INFO] Orijinal: {df.shape[0]} satir x {df.shape[1]} sutun")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# KNN Imputer ile eksik TotalCharges degerlerini doldur (fillna(0) yerine)
if df['TotalCharges'].isnull().sum() > 0:
    _knn_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    _imputer = KNNImputer(n_neighbors=5)
    df[_knn_cols] = _imputer.fit_transform(df[_knn_cols])
    print(f"   [KNN] TotalCharges eksik degerleri KNN (k=5) ile dolduruldu.")
df = df.drop('customerID', axis=1)

# Feature Engineering
df = create_features(df)
print(f"   [OK] Feature Engineering: Yeni feature'lar eklendi")

preprocessing_info = {}

# Binary Encoding
binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
binary_mappings = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    binary_mappings[col] = dict(zip([str(c) for c in le.classes_.tolist()],
                                     [int(v) for v in le.transform(le.classes_).tolist()]))
preprocessing_info['binary_mappings'] = binary_mappings

# One-Hot Encoding
multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
              'Contract','PaymentMethod']
multi_categories = {}
for col in multi_cols:
    multi_categories[col] = sorted(df[col].unique().tolist())
preprocessing_info['multi_categories'] = multi_categories
df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)

feature_columns = [col for col in df.columns if col != 'Churn']
print(f"   [OK] Encoding oncesi: {len(feature_columns)} feature")

# Outlier + Scaling
numerical_cols = ['tenure','MonthlyCharges','TotalCharges'] + list(ENGINEERED_NUMERICAL)
outlier_bounds = {}
for col in numerical_cols:
    if col not in df.columns: continue
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = float(Q1 - 1.5*IQR), float(Q3 + 1.5*IQR)
    df[col] = df[col].clip(lower=lo, upper=hi)
    outlier_bounds[col] = {'lower': lo, 'upper': hi}
preprocessing_info['outlier_bounds'] = outlier_bounds

scaler = StandardScaler()
scale_cols = [c for c in numerical_cols if c in df.columns]
df[scale_cols] = scaler.fit_transform(df[scale_cols])
preprocessing_info['numerical_cols'] = scale_cols

# === 2. TRAIN/TEST & FEATURE SELECTION ===
print("\n" + "=" * 70)
print("  2. TRAIN/TEST SPLIT VE FEATURE SELECTION")
print("=" * 70)
X = df[feature_columns]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
churn_ratio = y_train.mean()
scale_pos = (1-churn_ratio)/churn_ratio

# Feature Selection (LightGBM kullanarak)
print("\n   [>>] Feature Selection uygulaniyor...")
selector_model = LGBMClassifier(random_state=42, n_estimators=100, class_weight='balanced', verbose=-1)
selector_model.fit(X_train, y_train)

# SelectFromModel ile ortalama onemin uzerindeki feature'lari sec
selector = SelectFromModel(selector_model, prefit=True, threshold='0.75*mean')
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_feature_indices = selector.get_support(indices=True)
selected_feature_columns = [feature_columns[i] for i in selected_feature_indices]
preprocessing_info['feature_columns'] = selected_feature_columns

print(f"   [OK] Orijinal Feature Sayisi: {len(feature_columns)}")
print(f"   [OK] Secilen Feature Sayisi: {len(selected_feature_columns)}")
print(f"   [OK] Elenen Feature Sayisi: {len(feature_columns) - len(selected_feature_columns)}")

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f: pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'preprocessing_info.json'), 'w', encoding='utf-8') as f:
    json.dump(preprocessing_info, f, ensure_ascii=False, indent=2)

# === 3. OPTUNA ILE HIPERPARAMETRE OPTIMIZASYONU ===
print("\n" + "=" * 70)
print("  3. OPTUNA ILE DERIN HIPERPARAMETRE OPTIMIZASYONU")
print("=" * 70)

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'scale_pos_weight': scale_pos,
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': False
    }
    model = CatBoostClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()

n_trials = 25

print(f"\n   [>>] LightGBM Optuna Tuning ({n_trials} trials)...")
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=n_trials)
best_lgb_params = study_lgb.best_params
best_lgb_params['class_weight'] = 'balanced'
best_lgb_params['random_state'] = 42
best_lgb_params['verbose'] = -1
print(f"   [OK] LightGBM Best CV F1: {study_lgb.best_value:.4f}")

print(f"\n   [>>] XGBoost Optuna Tuning ({n_trials} trials)...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=n_trials)
best_xgb_params = study_xgb.best_params
best_xgb_params['scale_pos_weight'] = scale_pos
best_xgb_params['random_state'] = 42
best_xgb_params['eval_metric'] = 'logloss'
best_xgb_params['verbosity'] = 0
print(f"   [OK] XGBoost Best CV F1: {study_xgb.best_value:.4f}")

print(f"\n   [>>] CatBoost Optuna Tuning ({n_trials} trials)...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=n_trials)
best_cat_params = study_cat.best_params
best_cat_params['auto_class_weights'] = 'Balanced'
best_cat_params['random_seed'] = 42
best_cat_params['verbose'] = False
print(f"   [OK] CatBoost Best CV F1: {study_cat.best_value:.4f}")

# === 4. MODEL EGITIMI ===
print("\n" + "=" * 70)
print("  4. MODEL EGITIMI")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight='balanced', n_jobs=-1),
    'XGBoost (Optuna)': XGBClassifier(**best_xgb_params),
    'LightGBM (Optuna)': LGBMClassifier(**best_lgb_params),
    'CatBoost (Optuna)': CatBoostClassifier(**best_cat_params)
}

results = {}

for name, model in models.items():
    print(f"\n   [>>] {name} egitiliyor...")
    t0 = time.time()
    model.fit(X_train_selected, y_train)
    tt = time.time() - t0
    yp = model.predict(X_test_selected)
    ypr = model.predict_proba(X_test_selected)[:, 1]
    cv = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='f1', n_jobs=-1)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, yp),
        'precision': precision_score(y_test, yp),
        'recall': recall_score(y_test, yp),
        'f1_score': f1_score(y_test, yp),
        'roc_auc': roc_auc_score(y_test, ypr),
        'cv_f1_mean': cv.mean(),
        'cv_f1_std': cv.std(),
        'train_time': tt,
        'y_pred': yp,
        'y_proba': ypr
    }
    r = results[name]
    print(f"      {tt:.2f}s | F1:{r['f1_score']:.4f} | Rec:{r['recall']:.4f} | AUC:{r['roc_auc']:.4f} | CV:{r['cv_f1_mean']:.4f}")

# Stacking Ensemble
print(f"\n   [>>] Stacking Ensemble (LGBM + CatBoost + XGBoost)...")
t0 = time.time()
stacking = StackingClassifier(
    estimators=[
        ('lgbm', LGBMClassifier(**best_lgb_params)),
        ('cat', CatBoostClassifier(**best_cat_params)),
        ('xgb', XGBClassifier(**best_xgb_params))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5, n_jobs=-1
)
stacking.fit(X_train_selected, y_train)
tt = time.time() - t0
yp = stacking.predict(X_test_selected)
ypr = stacking.predict_proba(X_test_selected)[:, 1]
cv = cross_val_score(stacking, X_train_selected, y_train, cv=5, scoring='f1', n_jobs=-1)

results['Stacking Ensemble'] = {
    'model': stacking, 'accuracy': accuracy_score(y_test, yp), 'precision': precision_score(y_test, yp),
    'recall': recall_score(y_test, yp), 'f1_score': f1_score(y_test, yp), 'roc_auc': roc_auc_score(y_test, ypr),
    'cv_f1_mean': cv.mean(), 'cv_f1_std': cv.std(), 'train_time': tt, 'y_pred': yp, 'y_proba': ypr
}
r = results['Stacking Ensemble']
print(f"      {tt:.2f}s | F1:{r['f1_score']:.4f} | Rec:{r['recall']:.4f} | AUC:{r['roc_auc']:.4f} | CV:{r['cv_f1_mean']:.4f}")

# === 5. THRESHOLD OPTIMIZATION ===
print("\n" + "=" * 70)
print("  5. THRESHOLD OPTIMIZATION")
print("=" * 70)

best_name = max(results, key=lambda x: results[x]['f1_score'])
best_proba = results[best_name]['y_proba']
thresholds = np.arange(0.2, 0.7, 0.01)
f1s = [f1_score(y_test, (best_proba >= t).astype(int)) for t in thresholds]
opt_thresh = thresholds[np.argmax(f1s)]
opt_f1 = max(f1s)
opt_pred = (best_proba >= opt_thresh).astype(int)
opt_rec = recall_score(y_test, opt_pred)
opt_prec = precision_score(y_test, opt_pred)

print(f"\n   [*] Model: {best_name}")
print(f"   [*] Varsayilan (0.50): F1={results[best_name]['f1_score']:.4f}")
print(f"   [*] Optimal  ({opt_thresh:.2f}): F1={opt_f1:.4f} | Prec={opt_prec:.4f} | Rec={opt_rec:.4f}")

preprocessing_info['optimal_threshold'] = float(opt_thresh)
with open(os.path.join(MODEL_DIR, 'preprocessing_info.json'), 'w', encoding='utf-8') as f:
    json.dump(preprocessing_info, f, ensure_ascii=False, indent=2)

# === 6. SONUCLAR VE METADATA ===
print("\n" + "=" * 70)
print("  6. MODEL KARSILASTIRMA (F1 sirali)")
print("=" * 70)
print(f"\n   {'Model':<25} {'F1':>7} {'Rec':>7} {'Prec':>7} {'AUC':>7} {'CV-F1':>7}")
print("   " + "-" * 65)
for name, r in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
    m = " ***" if name == best_name else ""
    print(f"   {name:<25} {r['f1_score']:>7.4f} {r['recall']:>7.4f} {r['precision']:>7.4f} {r['roc_auc']:>7.4f} {r['cv_f1_mean']:>7.4f}{m}")

best_result = results[best_name]
with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'wb') as f: pickle.dump(best_result['model'], f)
print(f"\n   [OK] Model kaydedildi: {best_name}")

metadata = {
    'best_model': best_name,
    'optimal_threshold': float(opt_thresh),
    'optimal_f1_with_threshold': round(opt_f1, 4),
    'metrics': {k: round(best_result[k], 4) for k in ['accuracy','precision','recall','f1_score','roc_auc','cv_f1_mean','cv_f1_std']},
    'all_models': {n: {k: round(r[k],4) for k in ['accuracy','precision','recall','f1_score','roc_auc','cv_f1_mean','cv_f1_std']}
                   | {'train_time_seconds': round(r['train_time'],3)} for n,r in results.items()},
    'feature_columns': selected_feature_columns,
    'train_size': int(len(X_train_selected)), 'test_size': int(len(X_test_selected)), 'total_features': int(len(selected_feature_columns)),
    'improvements': ['feature_engineering', 'feature_selection', 'optuna_tuning', 'catboost', 'stacking_ensemble', 'threshold_optimization']
}
with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# === 7. GRAFIKLER ===
print("\n" + "=" * 70)
print("  7. GRAFIKLER CIZILIYOR...")
print("=" * 70)
model_names = list(results.keys())
cmap = plt.cm.tab10

# 7a. Karsilastirma
fig, ax = plt.subplots(figsize=(14,7))
x = np.arange(len(model_names)); w = 0.15
for i,(met,lab) in enumerate(zip(['accuracy','precision','recall','f1_score','roc_auc'],
                                  ['Accuracy','Precision','Recall','F1','AUC'])):
    vals = [results[m][met] for m in model_names]
    ax.bar(x+i*w, vals, w, label=lab)
ax.set_xticks(x+w*2); ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_title('Model Performans Karsilastirmasi', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right'); ax.set_ylim(0,1.15); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'07_model_karsilastirma.png'),dpi=150); plt.close()

# 7b. Threshold
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(thresholds, f1s, 'b-', linewidth=2, label='F1-Score')
ax.axvline(x=opt_thresh, color='r', linestyle='--', linewidth=2, label=f'Optimal={opt_thresh:.2f}')
ax.set_xlabel('Threshold'); ax.set_ylabel('F1-Score')
ax.set_title(f'Threshold Optimization - {best_name}', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'13_threshold_optimization.png'),dpi=150); plt.close()

print(f"\n{classification_report(y_test, opt_pred, target_names=['No Churn','Churn'])}")
print("=" * 70)
print("   GELISMIS MODEL GELISTIRME v3 TAMAMLANDI!")
print("=" * 70)
