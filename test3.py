# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Conv1D, MaxPooling1D, 
                                   Input, Attention, BatchNormalization, 
                                   Bidirectional, GlobalAveragePooling1D,
                                   MultiHeadAttention, LayerNormalization, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import joblib

# Configuration pour reproductibilité
tf.random.set_seed(42)
np.random.seed(42)

# %%
# Charger le dataset généré avec le nouveau générateur
df = pd.read_csv("iot_complete_realistic.csv")

# Inspection du dataset
print("=== Aperçu du dataset ===")
print(f"Shape: {df.shape}")
print(f"Colonnes: {list(df.columns)}")
print(df.head())

print("\n=== Informations générales ===")
print(df.info())

print("\n=== Statistiques sur les valeurs manquantes ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.any() else "Aucune valeur manquante.")

# Statistiques descriptives pour variables numériques
print("\n=== Statistiques descriptives (premières 5 colonnes numériques) ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
print(df[numeric_cols].describe())

print("\n=== Répartition des types d'attaques ===")
attack_counts = df['Attack_type'].value_counts()
print(attack_counts)

# %%
# Prétraitement amélioré
print("\n=== Prétraitement du dataset ===")
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

categorical_columns = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_columns:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("Dataset trié par timestamp.")

# %%
# Feature Engineering
print("=== Feature Engineering ===")

if all(col in df.columns for col in ['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia']):
    df['blood_pressure_ratio'] = df['blood_pressure_sys'] / (df['blood_pressure_dia'] + 1)
    df['cardiovascular_stress'] = (df['heart_rate'] * df['blood_pressure_sys']) / 10000

if all(col in df.columns for col in ['spo2_level', 'respiration_rate']):
    df['respiratory_efficiency'] = df['spo2_level'] * df['respiration_rate'] / 100

if all(col in df.columns for col in ['eeg_alpha_power', 'eeg_beta_power']):
    df['eeg_alpha_beta_ratio'] = df['eeg_alpha_power'] / (df['eeg_beta_power'] + 0.001)

if all(col in df.columns for col in ['fwd_pkts_tot', 'bwd_pkts_tot', 'flow_duration']):
    df['total_packets'] = df['fwd_pkts_tot'] + df['bwd_pkts_tot']
    df['packets_per_second'] = df['total_packets'] / (df['flow_duration'] + 1)
    df['fwd_bwd_packet_ratio'] = df['fwd_pkts_tot'] / (df['bwd_pkts_tot'] + 1)

if all(col in df.columns for col in ['fwd_pkts_per_sec', 'bwd_pkts_per_sec']):
    df['packet_rate_imbalance'] = abs(df['fwd_pkts_per_sec'] - df['bwd_pkts_per_sec'])

if all(col in df.columns for col in ['latitude', 'longitude']):
    df['geo_distance_origin'] = np.sqrt(df['latitude']**2 + df['longitude']**2)

if "timestamp" in df.columns:
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

print("Feature engineering terminé.")

# %%
# Encodage
label_encoders = {}
label_encode_cols = ["patient_id", "severity_level"]
for col in label_encode_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

network_cols = ["id.orig_p", "id.resp_p"]
for col in network_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

categorical_cols = ["proto", "service", "activity_type", "fall_detected"]
available_categorical = [c for c in categorical_cols if c in df.columns]

if available_categorical:
    df_encoded = pd.get_dummies(df, columns=available_categorical, drop_first=False)
else:
    df_encoded = df.copy()

# %%
# Encodage de la cible
if "Attack_encoded" not in df_encoded.columns:
    attack_le = LabelEncoder()
    df_encoded["Attack_encoded"] = attack_le.fit_transform(df_encoded["Attack_type"])
    joblib.dump(attack_le, "attack_label_encoder.pkl")

# %%
# Normalisation
numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
cols_to_exclude = ["patient_id", "Attack_encoded", "timestamp"]
cols_to_exclude = [col for col in cols_to_exclude if col in numerical_cols]
numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]

scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
joblib.dump(scaler, "feature_scaler.pkl")

# %%
# Features finales
ignore_cols = ["patient_id", "timestamp", "Attack_type"]
target_col = "Attack_encoded"
feature_cols = [col for col in df_encoded.columns if col not in ignore_cols + [target_col]]

joblib.dump(feature_cols, "feature_columns.pkl")

X = df_encoded[feature_cols].values.astype("float32")
y = df_encoded[target_col].values

# %%
# Séquences
def create_sequences_with_overlap(X, y, seq_len=30, overlap=0.6):
    step = max(1, int(seq_len * (1 - overlap)))
    X_seq, y_seq = [], []
    for start in range(0, len(X) - seq_len + 1, step):
        X_seq.append(X[start:start+seq_len])
        y_seq.append(y[start+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

seq_len = 30
X_seq, y_seq_cat = create_sequences_with_overlap(X, y, seq_len=seq_len, overlap=0.6)

# === AJOUT === Sauvegarde des séquences
np.save("X_seq.npy", X_seq)
np.save("y_seq_cat.npy", y_seq_cat)
print("✅ Séquences sauvegardées (X_seq.npy, y_seq_cat.npy)")

# %%
# Conversion en one-hot
num_classes = len(np.unique(y_seq_cat))
y_seq_onehot = tf.keras.utils.to_categorical(y_seq_cat, num_classes=num_classes)
n_features = X_seq.shape[2]

# %%
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq_onehot, test_size=0.2, random_state=42, stratify=y_seq_cat
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42, stratify=np.argmax(y_train, axis=1)
)

# %%
# SMOTE
X_train_2d = X_train.reshape(X_train.shape[0], -1)
y_train_1d = np.argmax(y_train, axis=1)
unique, counts = np.unique(y_train_1d, return_counts=True)
min_samples = min(counts)
k_neighbors = min(5, max(1, min_samples - 1))

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train_2d, y_train_1d)

X_train_smote = X_train_smote.reshape(-1, seq_len, n_features)
y_train_smote = tf.keras.utils.to_categorical(y_train_smote, num_classes=num_classes)

# %%
# Poids de classes
y_integers = np.argmax(y_train_smote, axis=1)
weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_integers), y=y_integers)
class_weights = dict(zip(np.unique(y_integers), weights))

# %%
# Modèle
def create_residual_block(inputs, filters, kernel_size=3, dropout_rate=0.3):
    conv = Conv1D(filters, kernel_size, padding='same', activation='relu')(inputs)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv1D(filters, kernel_size, padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    if inputs.shape[-1] == filters:
        shortcut = inputs
    else:
        shortcut = Conv1D(filters, 1, padding='same')(inputs)
    output = Add()([conv, shortcut])
    return Dropout(dropout_rate)(output)

inputs = Input(shape=(seq_len, n_features), name='input_layer')
conv_out = create_residual_block(inputs, 128)
conv_out = MaxPooling1D(pool_size=2)(conv_out)
conv_out = create_residual_block(conv_out, 256)
conv_out = MaxPooling1D(pool_size=2)(conv_out)
conv_out = create_residual_block(conv_out, 128)

lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(inputs)
lstm_out = LayerNormalization()(lstm_out)
lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(lstm_out)
lstm_out = LayerNormalization()(lstm_out)

attention_out = MultiHeadAttention(num_heads=8, key_dim=64)(lstm_out, lstm_out)
attention_out = LayerNormalization()(attention_out)
attention_out = Add()([lstm_out, attention_out])
attention_out = Dropout(0.4)(attention_out)

conv_pooled = GlobalAveragePooling1D()(conv_out)
attention_pooled = GlobalAveragePooling1D()(attention_out)

from tensorflow.keras.layers import Concatenate
merged = Concatenate()([conv_pooled, attention_pooled])

dense1 = Dense(512, activation="relu")(merged)
dense1 = BatchNormalization()(dense1)
dense1 = Dropout(0.5)(dense1)
dense2 = Dense(256, activation="relu")(dense1)
dense2 = BatchNormalization()(dense2)
dense2 = Dropout(0.5)(dense2)
dense3 = Dense(128, activation="relu")(dense2)
dense3 = Dropout(0.4)(dense3)

outputs = Dense(num_classes, activation="softmax", name='output_layer')(dense3)

model = Model(inputs=inputs, outputs=outputs, name='Advanced_CNN_LSTM_Attention')
model.compile(loss="categorical_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              metrics=["accuracy"])

# %%
# Callbacks
early_stop = EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True, verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=8, min_lr=1e-6, verbose=1, mode='max')
checkpoint = ModelCheckpoint('best_model_advanced.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# %%
# Entraînement
history = model.fit(
    X_train_smote, y_train_smote,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# === AJOUT === Sauvegarde du modèle entraîné
model.save("advanced_cnn_lstm_attention.keras")
print("✅ Modèle sauvegardé sous advanced_cnn_lstm_attention.keras")

# %%
# Chargement du meilleur modèle
try:
    model = tf.keras.models.load_model('best_model_advanced.h5')
    print("Meilleur modèle chargé.")
except:
    print("Utilisation du modèle actuel.")

# %%
# Évaluation
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Perte sur test: {test_loss:.4f}")
print(f"Précision sur test: {test_accuracy:.4f}")

# %%
print("\n=== Sauvegarde des objets ===")
try:
    joblib.dump(label_encoders, "label_encoders.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")

    # Sauvegarde du label encoder de la cible seulement s’il existe
    if 'attack_le' in locals():
        joblib.dump(attack_le, "attack_label_encoder.pkl")
        print("attack_label_encoder.pkl sauvegardé.")
    else:
        print("⚠️ Pas de nouvel attack_le détecté (colonne Attack_encoded déjà présente).")

    print("✅ Tous les objets nécessaires ont été sauvegardés.")

except Exception as e:
    print(f"Erreur lors de la sauvegarde: {e}")

