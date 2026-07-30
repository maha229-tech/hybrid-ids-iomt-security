# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# %%
# Charger le dataset
df = pd.read_csv("iot_complete_realistic.csv")

print("=== Aperçu du dataset ===")
print(f"Shape: {df.shape}")
print(f"Colonnes: {list(df.columns)}")
print(f"Types de données:")
print(df.dtypes.value_counts())

# Identifier les colonnes non-numériques
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Colonnes non-numériques: {non_numeric}")

# %%
# === Prétraitement du dataset COMPLET ===
# Remplacement des valeurs manquantes
df.fillna(0, inplace=True)

# === ENCODAGE DE TOUTES LES COLONNES CATÉGORIELLES ===
label_encoders = {}

# 1. Colonnes avec LabelEncoder (ordinales ou avec beaucoup de valeurs uniques)
label_encode_cols = ["patient_id", "severity_level"]  # severity_level ajouté
for col in label_encode_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"✅ LabelEncoder appliqué à {col}: {len(le.classes_)} classes")

# 2. Colonnes réseau avec LabelEncoder si présentes
network_cols = ["id.orig_p", "id.resp_p"]
for col in network_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"✅ LabelEncoder appliqué à {col}: {len(le.classes_)} classes")

# Sauvegarde des LabelEncoders
joblib.dump(label_encoders, "label_encoders_instant.pkl")

# 3. One-hot encoding pour les colonnes catégorielles nominales
categorical_cols = ["proto", "service", "activity_type", "fall_detected"]
available_categorical = [c for c in categorical_cols if c in df.columns]

if available_categorical:
    print(f"✅ One-hot encoding pour: {available_categorical}")
    df_encoded = pd.get_dummies(df, columns=available_categorical, drop_first=False)
    print(f"Shape après one-hot: {df_encoded.shape}")
else:
    df_encoded = df.copy()
    print("⚠️ Aucune colonne pour one-hot encoding trouvée")

# %%
# === VÉRIFICATION FINALE DES TYPES ===
# S'assurer qu'il ne reste aucune colonne non-numérique (sauf celles à ignorer)
ignore_cols = ["timestamp", "Attack_type"]  # Colonnes à ignorer
remaining_non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
remaining_non_numeric = [col for col in remaining_non_numeric if col not in ignore_cols]

if remaining_non_numeric:
    print(f"⚠️ Colonnes non-numériques restantes: {remaining_non_numeric}")
    # Les encoder automatiquement
    for col in remaining_non_numeric:
        if df_encoded[col].nunique() > 2:
            # Beaucoup de valeurs uniques → LabelEncoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            print(f"✅ LabelEncoder automatique pour {col}")
        else:
            # Peu de valeurs → Mapping manuel
            unique_vals = df_encoded[col].unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"✅ Mapping automatique pour {col}: {mapping}")

# Re-sauvegarder les encoders mis à jour
joblib.dump(label_encoders, "label_encoders_instant.pkl")

# %%
# === NORMALISATION ===
# Identifier les colonnes numériques à normaliser
numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
cols_to_exclude = ["patient_id", "Attack_encoded", "timestamp"]
cols_to_exclude = [col for col in cols_to_exclude if col in numerical_cols]
numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]

print(f"✅ Colonnes à normaliser ({len(numerical_cols)}): {numerical_cols[:10]}...")

scaler = RobustScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
joblib.dump(scaler, "scaler_instant.pkl")

print("✅ Normalisation terminée")

# %%
# === Préparation des données pour XGBoost ===
ignore_cols_final = ["patient_id", "timestamp", "Attack_type"]
target_col = "Attack_encoded"

# Vérifier que la colonne cible existe
if target_col not in df_encoded.columns:
    print("❌ Colonne Attack_encoded manquante, création...")
    # Créer Attack_encoded à partir de Attack_type
    attack_le = LabelEncoder()
    df_encoded[target_col] = attack_le.fit_transform(df_encoded["Attack_type"])
    joblib.dump(attack_le, "attack_type_encoder.pkl")

feature_cols = [col for col in df_encoded.columns if col not in ignore_cols_final + [target_col]]

print(f"✅ Features sélectionnées ({len(feature_cols)}): {feature_cols[:10]}...")

# Définir X et y
X = df_encoded[feature_cols].values.astype(np.float32)  # Force conversion en float
y = df_encoded[target_col].values.astype(np.int32)      # Force conversion en int

# Vérification finale des types
print(f"✅ X shape: {X.shape}, dtype: {X.dtype}")
print(f"✅ y shape: {y.shape}, dtype: {y.dtype}")
print(f"✅ Valeurs uniques dans y: {np.unique(y)}")

# Sauvegarde de la liste des features
joblib.dump(feature_cols, "feature_columns_instant.pkl")

# %%
# === SPLIT TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Répartition des données pour XGBoost:")
print(f"X_train: {X_train.shape} y_train: {y_train.shape}")
print(f"X_test: {X_test.shape} y_test: {y_test.shape}")

# %%
# === ENTRAÎNEMENT XGBOOST ===
print(f"\n✅ Entraînement du modèle XGBoost...")

num_classes = len(np.unique(y))
print(f"Nombre de classes: {num_classes}")

# Paramètres optimisés pour éviter l'overfitting
params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': num_classes,
    'n_estimators': 100,        # Réduit pour éviter overfitting
    'learning_rate': 0.1,
    'max_depth': 4,             # Réduit pour éviter overfitting
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,           # Régularisation L1
    'reg_lambda': 0.1,          # Régularisation L2
    'random_state': 42,
    'verbosity': 1
}

model_xgb = xgb.XGBClassifier(**params)

try:
    model_xgb.fit(X_train, y_train)
    print("✅ Modèle XGBoost entraîné avec succès.")
    
    # Sauvegarde du modèle
    joblib.dump(model_xgb, "xgboost_instant_detector.pkl")
    print("✅ Modèle sauvegardé.")
    
except Exception as e:
    print(f"❌ Erreur pendant l'entraînement: {e}")
    print("Vérification des données:")
    print(f"X_train contient des NaN: {np.isnan(X_train).any()}")
    print(f"X_train contient des Inf: {np.isinf(X_train).any()}")
    print(f"Types dans X_train: {np.unique([type(x).__name__ for x in X_train.flatten()[:100]])}")
    raise

# %%
# === ÉVALUATION ===
print(f"\n✅ Évaluation du modèle sur les données de test:")

try:
    y_pred_probs = model_xgb.predict_proba(X_test)
    y_pred = model_xgb.predict(X_test)
    
    # Créer les mappings pour les labels
    if "Attack_type" in df_encoded.columns:
        unique_attacks = df_encoded["Attack_type"].unique()
        unique_encoded = df_encoded[df_encoded["Attack_type"].isin(unique_attacks)][target_col].unique()
        
        # Créer le mapping
        label_mapping = {}
        for attack in unique_attacks:
            encoded_val = df_encoded[df_encoded["Attack_type"] == attack][target_col].iloc[0]
            label_mapping[encoded_val] = attack
    else:
        # Mapping par défaut
        label_mapping = {i: f"Class_{i}" for i in range(num_classes)}
    
    # Conversion pour le rapport
    y_true_labels = [label_mapping[i] for i in y_test]
    y_pred_labels = [label_mapping[i] for i in y_pred]
    
    print(f"\n✅ Rapport de classification :")
    print(classification_report(y_true_labels, y_pred_labels, digits=4))
    
    print(f"\n✅ Matrice de confusion :")
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print(cm)
    
    # Analyse des performances par classe
    print(f"\n✅ Analyse détaillée:")
    from sklearn.metrics import classification_report
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"{class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    overall_accuracy = report['accuracy']
    print(f"\nAccuracy globale: {overall_accuracy:.4f}")
    
    if overall_accuracy > 0.95:
        print("⚠️ ATTENTION: Accuracy très élevée, possible overfitting ou data leakage")
    elif overall_accuracy < 0.7:
        print("⚠️ ATTENTION: Accuracy faible, dataset potentiellement trop difficile")
    else:
        print("✅ Accuracy dans une plage réaliste")
        
except Exception as e:
    print(f"❌ Erreur pendant l'évaluation: {e}")
    raise