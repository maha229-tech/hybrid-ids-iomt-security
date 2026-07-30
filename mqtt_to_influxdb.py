#!/usr/bin/env python3
"""
Consumer MQTT hybride suivant exactement les étapes des codes d'entraînement.
- XGBoost pour détection instantanée (étapes de code_xgboost.py)  
- CNN-LSTM pour analyse séquentielle (étapes de test3.py)
"""

import json
import time
import signal
import sys
from collections import deque
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, RobustScaler
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import warnings
warnings.filterwarnings('ignore')

# Configuration
BROKER = "localhost"
PORT = 1883
TOPIC = "iot/medical/sensors"
SEQUENCE_LENGTH = 30
ALERT_THRESHOLD = 0.7

# Configuration InfluxDB
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "ewclmn94SI9OZpSqUMb0KXyS4JzyU0TJhpkWfAB-jx8SboEugj2OnjQ8HiOFnlE3kV4DeCsrrlYCHwckSTTAfA=="
INFLUXDB_ORG = "Ibn tofail university"
INFLUXDB_BUCKET = "iot_health"

# Variables globales
running = True
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
influx_client = None
write_api = None
stats = {
    'messages_received': 0,
    'instant_detections': 0,
    'sequence_analyses': 0,
    'alerts_confirmed': 0,
    'alerts_rejected': 0,
    'influxdb_writes': 0,
    'influxdb_errors': 0
}

def handle_sigterm(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)

def initialize_influxdb():
    """Initialise la connexion InfluxDB"""
    global influx_client, write_api
    try:
        influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        query_api = influx_client.query_api()
        query = f'from(bucket:"{INFLUXDB_BUCKET}") |> range(start: -1m) |> limit(n:1)'
        query_api.query(query=query)
        print("✅ Connexion InfluxDB établie")
        return True
    except Exception as e:
        print(f"⚠️ Erreur connexion InfluxDB: {e}")
        return False

def write_to_influxdb(data_point):
    """Écrit un point de données dans InfluxDB"""
    global stats
    try:
        if write_api is None:
            print("❌ write_api is None")
            return False
        
        print(f"🔍 Tentative d'écriture: {data_point}")  # DEBUG
        write_api.write(bucket=INFLUXDB_BUCKET, record=data_point)
        stats['influxdb_writes'] += 1
        print(f"✅ Écriture réussie #{stats['influxdb_writes']}")  # DEBUG
        return True
        
    except Exception as e:
        stats['influxdb_errors'] += 1
        print(f"❌ Erreur InfluxDB #{stats['influxdb_errors']}: {e}")  # TOUJOURS afficher
        return False

class ModelLoader:
    def __init__(self):
        self.xgb_model = None
        self.cnn_lstm_model = None
        self.scaler_xgb = None
        self.scaler_cnn = None
        self.label_encoders_xgb = {}
        self.label_encoders_cnn = {}
        self.attack_encoder = None
        self.feature_columns_xgb = []
        self.feature_columns_cnn = []
        self.attack_mapping = {}
        
    def load_all_models(self):
        """Charge tous les modèles et artefacts nécessaires"""
        try:
            print("Chargement des modèles...")
            
            # Modèle XGBoost
            try:
                self.xgb_model = joblib.load("xgboost_instant_detector.pkl")
                print("✅ Modèle XGBoost chargé")
            except Exception as e:
                print(f"⚠️ Erreur XGBoost: {e}")
                
            # Modèle CNN-LSTM
            try:
                self.cnn_lstm_model = tf.keras.models.load_model("best_model_advanced.h5")
                print("✅ Modèle CNN-LSTM-Attention chargé")
            except Exception as e:
                print(f"⚠️ Erreur CNN-LSTM: {e}")
                
            # Artefacts XGBoost
            try:
                self.scaler_xgb = joblib.load("scaler_instant.pkl")
                self.label_encoders_xgb = joblib.load("label_encoders_instant.pkl")
                self.feature_columns_xgb = joblib.load("feature_columns_instant.pkl")
                print("✅ Artefacts XGBoost chargés")
            except Exception as e:
                print(f"⚠️ Erreur artefacts XGBoost: {e}")
                
            # Artefacts CNN-LSTM  
            try:
                self.scaler_cnn = joblib.load("feature_scaler.pkl")
                self.label_encoders_cnn = joblib.load("label_encoders.pkl")
                self.feature_columns_cnn = joblib.load("feature_columns.pkl")
                print("✅ Artefacts CNN-LSTM chargés")
            except Exception as e:
                print(f"⚠️ Erreur artefacts CNN-LSTM: {e}")
                
            # Attack encoder
            try:
                self.attack_encoder = joblib.load("attack_label_encoder.pkl")
                print("✅ Attack encoder chargé")
            except:
                try:
                    self.attack_encoder = joblib.load("attack_type_encoder.pkl")
                    print("✅ Attack encoder alternatif chargé")
                except:
                    print("🔧 Création d'un encoder par défaut...")
                    from sklearn.preprocessing import LabelEncoder
                    self.attack_encoder = LabelEncoder()
                    self.attack_encoder.classes_ = np.array(["Normal", "DoS/DDoS", "Eavesdropping", "Injection", "Selective Forwarding", "Sinkhole", "MitM"])
                    print("✅ Encoder par défaut créé")
                    
            # Mapping des attaques
            if self.attack_encoder:
                self.attack_mapping = {i: attack for i, attack in enumerate(self.attack_encoder.classes_)}
            else:
                self.attack_mapping = {0: "Normal", 1: "DoS/DDoS", 2: "Eavesdropping", 3: "Injection", 4: "Selective Forwarding", 5: "Sinkhole", 6: "MitM"}
                
            print("🎯 Chargement des modèles terminé")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False

class XGBoostProcessor:
    """Applique exactement les étapes de code_xgboost.py"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        
    def preprocess_for_xgboost(self, message_data):
        """Prétraitement EXACT du code_xgboost.py"""
        try:
            # Créer DataFrame
            df = pd.DataFrame([message_data])
            
            # === PRÉTRAITEMENT DU DATASET COMPLET ===
            # Remplacement des valeurs manquantes
            df.fillna(0, inplace=True)
            
            # === ENCODAGE DE TOUTES LES COLONNES CATÉGORIELLES ===
            # 1. Colonnes avec LabelEncoder (ordinales ou avec beaucoup de valeurs uniques)
            label_encode_cols = ["patient_id", "severity_level"]
            for col in label_encode_cols:
                if col in df.columns and col in self.model_loader.label_encoders_xgb:
                    le = self.model_loader.label_encoders_xgb[col]
                    df[col] = df[col].astype(str)
                    # Gérer valeurs inconnues
                    unknown_mask = ~df[col].isin(le.classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = le.classes_[0]
                    df[col] = le.transform(df[col])

            # 2. Colonnes réseau avec LabelEncoder si présentes
            network_cols = ["id.orig_p", "id.resp_p"]
            for col in network_cols:
                if col in df.columns and col in self.model_loader.label_encoders_xgb:
                    le = self.model_loader.label_encoders_xgb[col]
                    df[col] = df[col].astype(str)
                    unknown_mask = ~df[col].isin(le.classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = le.classes_[0]
                    df[col] = le.transform(df[col])

            # 3. One-hot encoding COMPLET pour toutes les valeurs possibles
            df = self.create_complete_onehot_encoding(df)

            # 4. === VÉRIFICATION FINALE DES TYPES ===
            ignore_cols = ["timestamp", "Attack_type"]
            remaining_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            remaining_non_numeric = [col for col in remaining_non_numeric if col not in ignore_cols]

            if remaining_non_numeric:
                for col in remaining_non_numeric:
                    if col in self.model_loader.label_encoders_xgb:
                        le = self.model_loader.label_encoders_xgb[col]
                        df[col] = df[col].astype(str)
                        unknown_mask = ~df[col].isin(le.classes_)
                        if unknown_mask.any():
                            df.loc[unknown_mask, col] = le.classes_[0]
                        df[col] = le.transform(df[col])
                    else:
                        if df[col].nunique() > 2:
                            unique_vals = df[col].unique()
                            mapping = {val: i for i, val in enumerate(unique_vals)}
                            df[col] = df[col].map(mapping)
                        else:
                            unique_vals = df[col].unique()
                            mapping = {val: i for i, val in enumerate(unique_vals)}
                            df[col] = df[col].map(mapping)

            # === NORMALISATION ===
            # S'assurer que toutes les colonnes attendues existent
            df = self.align_with_training_features(df, is_xgboost=True)
            
            # Normaliser seulement les colonnes qui étaient normalisées à l'entraînement
            if hasattr(self.model_loader.scaler_xgb, 'feature_names_in_'):
                expected_cols = list(self.model_loader.scaler_xgb.feature_names_in_)
                available_cols = [col for col in expected_cols if col in df.columns]
                if available_cols:
                    df[available_cols] = self.model_loader.scaler_xgb.transform(df[available_cols])
            else:
                # Fallback : utiliser la logique d'origine
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cols_to_exclude = ["patient_id", "Attack_encoded", "timestamp"]
                cols_to_exclude = [col for col in cols_to_exclude if col in numerical_cols]
                numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]
                if numerical_cols:
                    df[numerical_cols] = self.model_loader.scaler_xgb.transform(df[numerical_cols])

            return df
            
        except Exception as e:
            print(f"❌ Erreur preprocessing XGBoost: {e}")
            return None
            
    def create_complete_onehot_encoding(self, df):
        """Crée TOUTES les colonnes one-hot possibles basées sur l'entraînement"""
        try:
            # Définir toutes les valeurs possibles pour chaque colonne
            # (basé sur les données d'entraînement)
            possible_values = {
                "proto": ["TCP", "UDP"],
                "service": ["MQTT", "HTTP", "HTTPS", "DNS", "COAP"],
                "activity_type": ["resting", "walking", "running"],
                "fall_detected": [0, 1]
            }
            
            for col in ["proto", "service", "activity_type", "fall_detected"]:
                if col in df.columns:
                    current_value = df[col].iloc[0]
                    
                    # Créer TOUTES les colonnes possibles
                    for value in possible_values[col]:
                        new_col = f"{col}_{value}"
                        if current_value == value:
                            df[new_col] = 1
                        else:
                            df[new_col] = 0
                    
                    # Supprimer la colonne originale
                    df = df.drop(columns=[col])
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur one-hot complet: {e}")
            return df
            
    def align_with_training_features(self, df, is_xgboost=True):
        """Aligne le DataFrame avec les features d'entraînement"""
        try:
            # Obtenir les features attendues
            if is_xgboost and self.model_loader.feature_columns_xgb:
                expected_features = set(self.model_loader.feature_columns_xgb)
            elif not is_xgboost and self.model_loader.feature_columns_cnn:
                expected_features = set(self.model_loader.feature_columns_cnn)
            else:
                return df
                
            current_features = set(df.columns)
            
            # Ajouter les colonnes manquantes avec des valeurs 0
            missing_features = expected_features - current_features
            for feature in missing_features:
                df[feature] = 0
                
            # Supprimer les colonnes en trop (garder les colonnes importantes)
            keep_cols = {'timestamp', 'Attack_type', 'Attack_encoded', 'patient_id'}
            extra_features = current_features - expected_features - keep_cols
            df = df.drop(columns=[col for col in extra_features if col in df.columns], errors='ignore')
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur alignement features: {e}")
            return df
            
    def predict_xgboost(self, processed_df):
        """Prédiction avec XGBoost"""
        try:
            if self.model_loader.xgb_model is None or processed_df is None:
                return None
                
            # === PRÉPARATION DES DONNÉES POUR XGBOOST ===
            ignore_cols_final = ["patient_id", "timestamp", "Attack_type"]
            target_col = "Attack_encoded"
            feature_cols = [col for col in processed_df.columns if col not in ignore_cols_final + [target_col]]
            
            # Utiliser les features sauvegardées si disponibles
            if self.model_loader.feature_columns_xgb:
                available_features = [col for col in self.model_loader.feature_columns_xgb if col in processed_df.columns]
                if available_features:
                    feature_cols = available_features

            if not feature_cols:
                return None

            X = processed_df[feature_cols].values.astype(np.float32)
            
            # Prédiction
            prediction = self.model_loader.xgb_model.predict(X)[0]
            probabilities = self.model_loader.xgb_model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            attack_name = self.model_loader.attack_mapping.get(prediction, f"Unknown_{prediction}")
            
            result = {
                'prediction': attack_name,
                'prediction_id': int(prediction),
                'confidence': float(confidence),
                'probabilities': [float(p) for p in probabilities],
                'is_alert': confidence > ALERT_THRESHOLD and prediction != 0
            }
            
            stats['instant_detections'] += 1
            return result
            
        except Exception as e:
            print(f"❌ Erreur prédiction XGBoost: {e}")
            return None

class CNNLSTMProcessor:
    """Applique exactement les étapes de test3.py"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        
    def preprocess_for_cnn_lstm(self, message_data):
        """Prétraitement EXACT du test3.py"""
        try:
            # Créer DataFrame
            df = pd.DataFrame([message_data])
            
            # Prétraitement amélioré
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

            # Feature Engineering
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

            # Encodage
            label_encode_cols = ["patient_id", "severity_level"]
            for col in label_encode_cols:
                if col in df.columns and col in self.model_loader.label_encoders_cnn:
                    le = self.model_loader.label_encoders_cnn[col]
                    df[col] = df[col].astype(str)
                    unknown_mask = ~df[col].isin(le.classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = le.classes_[0]
                    df[col] = le.transform(df[col])

            network_cols = ["id.orig_p", "id.resp_p"]
            for col in network_cols:
                if col in df.columns and col in self.model_loader.label_encoders_cnn:
                    le = self.model_loader.label_encoders_cnn[col]
                    df[col] = df[col].astype(str)
                    unknown_mask = ~df[col].isin(le.classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = le.classes_[0]
                    df[col] = le.transform(df[col])

            # One-hot encoding COMPLET pour CNN-LSTM
            df = self.create_complete_onehot_encoding_cnn(df)

            # Aligner avec les features d'entraînement CNN-LSTM
            df = self.align_with_training_features_cnn(df)

            # Normalisation
            if hasattr(self.model_loader.scaler_cnn, 'feature_names_in_'):
                expected_cols = list(self.model_loader.scaler_cnn.feature_names_in_)
                available_cols = [col for col in expected_cols if col in df.columns]
                if available_cols:
                    df[available_cols] = self.model_loader.scaler_cnn.transform(df[available_cols])
            else:
                # Fallback
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cols_to_exclude = ["patient_id", "Attack_encoded", "timestamp"]
                cols_to_exclude = [col for col in cols_to_exclude if col in numerical_cols]
                numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]
                if numerical_cols:
                    df[numerical_cols] = self.model_loader.scaler_cnn.transform(df[numerical_cols])

            return df
            
        except Exception as e:
            print(f"❌ Erreur preprocessing CNN-LSTM: {e}")
            return None
            
    def create_complete_onehot_encoding_cnn(self, df):
        """Crée TOUTES les colonnes one-hot possibles pour CNN-LSTM"""
        try:
            # Même logique que XGBoost mais pour CNN-LSTM
            possible_values = {
                "proto": ["TCP", "UDP"],
                "service": ["MQTT", "HTTP", "HTTPS", "DNS", "COAP"],
                "activity_type": ["resting", "walking", "running"],
                "fall_detected": [0, 1]
            }
            
            for col in ["proto", "service", "activity_type", "fall_detected"]:
                if col in df.columns:
                    current_value = df[col].iloc[0]
                    
                    # Créer TOUTES les colonnes possibles
                    for value in possible_values[col]:
                        new_col = f"{col}_{value}"
                        if current_value == value:
                            df[new_col] = 1
                        else:
                            df[new_col] = 0
                    
                    # Supprimer la colonne originale
                    df = df.drop(columns=[col])
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur one-hot complet CNN-LSTM: {e}")
            return df
            
    def align_with_training_features_cnn(self, df):
        """Aligne le DataFrame avec les features d'entraînement CNN-LSTM"""
        try:
            if not self.model_loader.feature_columns_cnn:
                return df
                
            expected_features = set(self.model_loader.feature_columns_cnn)
            current_features = set(df.columns)
            
            # Ajouter les colonnes manquantes avec des valeurs 0
            missing_features = expected_features - current_features
            for feature in missing_features:
                df[feature] = 0
                
            # Supprimer les colonnes en trop (garder les colonnes importantes)
            keep_cols = {'timestamp', 'Attack_type', 'Attack_encoded', 'patient_id'}
            extra_features = current_features - expected_features - keep_cols
            df = df.drop(columns=[col for col in extra_features if col in df.columns], errors='ignore')
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur alignement features CNN-LSTM: {e}")
            return df
            
    def predict_sequence(self, sequence_data):
        """Analyse séquentielle avec CNN-LSTM"""
        try:
            if self.model_loader.cnn_lstm_model is None or len(sequence_data) < SEQUENCE_LENGTH:
                return None
                
            # Prendre les dernières SEQUENCE_LENGTH données
            recent_data = sequence_data[-SEQUENCE_LENGTH:]
            
            # Features finales
            ignore_cols = ["patient_id", "timestamp", "Attack_type"]
            target_col = "Attack_encoded"
            
            X_seq = []
            for processed_df in recent_data:
                feature_cols = [col for col in processed_df.columns if col not in ignore_cols + [target_col]]
                
                # Utiliser les features sauvegardées
                if self.model_loader.feature_columns_cnn:
                    available_features = [col for col in self.model_loader.feature_columns_cnn if col in processed_df.columns]
                    if available_features:
                        feature_cols = available_features
                
                if feature_cols:
                    X_seq.append(processed_df[feature_cols].values[0])
                    
            if len(X_seq) != SEQUENCE_LENGTH:
                return None
                
            X_seq = np.array([X_seq], dtype=np.float32)
            
            # Prédiction
            prediction_probs = self.model_loader.cnn_lstm_model.predict(X_seq, verbose=0)[0]
            prediction = np.argmax(prediction_probs)
            confidence = max(prediction_probs)
            
            attack_name = self.model_loader.attack_mapping.get(prediction, f"Unknown_{prediction}")
            
            result = {
                'prediction': attack_name,
                'prediction_id': int(prediction),
                'confidence': float(confidence),
                'probabilities': [float(p) for p in prediction_probs],
                'sequence_length': len(recent_data),
                'is_alert': confidence > ALERT_THRESHOLD and prediction != 0
            }
            
            stats['sequence_analyses'] += 1
            return result
            
        except Exception as e:
            print(f"❌ Erreur analyse séquentielle: {e}")
            return None

def correlate_results(instant_result, sequence_result, timestamp, message_data):
    """
    Corrèle les résultats XGBoost (instantané) et CNN-LSTM (séquentiel) pour 
    déterminer une décision finale avec une confiance combinée.

    Args:
        instant_result (dict): Résultat du modèle instantané (XGBoost), 
                               doit contenir 'prediction' et 'confidence'.
        sequence_result (dict): Résultat du modèle séquentiel (CNN-LSTM), 
                               doit contenir 'prediction' et 'confidence'.
        timestamp (datetime): Horodatage de la donnée traitée.
        message_data (dict): Données brutes du message (pour le patient_id).

    Returns:
        dict: Dictionnaire de corrélation avec la décision finale et la confiance.
    """

    try:
        # Initialisation du dictionnaire de corrélation
        correlation = {
            "timestamp": timestamp,
            "patient_id": message_data.get("patient_id", "unknown"),
            "xgb_pred": instant_result.get("prediction") if instant_result else None,
            "xgb_confidence": instant_result.get("confidence") if instant_result else None,
            "cnn_pred": sequence_result.get("prediction") if sequence_result else None,
            "cnn_confidence": sequence_result.get("confidence") if sequence_result else None,
            "final_decision": None,
            "confidence": 0.0
        }

        # --- Logique de Corrélation ---

        # Cas 1: Le modèle séquentiel n'a pas encore de résultat (trop peu de points)
        if not sequence_result:
            if instant_result:
                # La décision repose uniquement sur le modèle instantané
                correlation["final_decision"] = instant_result["prediction"]
                correlation["confidence"] = instant_result["confidence"]
            return correlation

        # Cas 2: Les deux modèles ont un résultat (vérification implicite que instant_result est aussi présent)
        if instant_result and sequence_result:
            
            # Sous-cas 2a: Accord entre les deux modèles
            if instant_result["prediction"] == sequence_result["prediction"]:
                # Renforcement de la confiance par la moyenne
                correlation["final_decision"] = instant_result["prediction"]
                correlation["confidence"] = (instant_result["confidence"] + sequence_result["confidence"]) / 2
                return correlation

            # Sous-cas 2b: Divergence -> choisir la prédiction la plus confiante
            else:
                if sequence_result["confidence"] >= instant_result["confidence"]:
                    correlation["final_decision"] = sequence_result["prediction"]
                    correlation["confidence"] = sequence_result["confidence"]
                else:
                    correlation["final_decision"] = instant_result["prediction"]
                    correlation["confidence"] = instant_result["confidence"]

        # Cas de figure où seul le résultat instantané est présent (pour couverture)
        elif instant_result:
             correlation["final_decision"] = instant_result["prediction"]
             correlation["confidence"] = instant_result["confidence"]


        return correlation

    except Exception as e:
        # Gestion des erreurs de logique ou d'accès aux clés
        print(f"❌ Erreur lors de la corrélation des résultats : {e}")
        return None

def create_sensor_data_point(message_data, correlation=None, timestamp=None):
    """
    Crée un point de données pour InfluxDB en respectant les conventions:
    - TAGS: Données catégorielles et d'indexation (patient_id, décisions).
    - FIELDS: Données numériques à tracer (métriques médicales, scores et confiances).
    """
    try:
        # Initialisation du Point
        point = Point("sensor_data") 
        
        # Horodatage
        if timestamp:
            point = point.time(timestamp)

        # === TAGS (Indexation et Décisions) ===
        point = point.tag("patient_id", message_data.get('patient_id', 'unknown'))
        point = point.tag("attack_type", message_data.get('Attack_type', 'unknown'))
        point = point.tag("severity_level", message_data.get('severity_level', 'unknown'))
        
        # Décision finale (TAG: catégorie)
        if correlation and correlation.get('final_decision'):
             point = point.tag("final_decision", correlation.get('final_decision', 'NORMAL'))
        else:
             point = point.tag("final_decision", 'WAITING') 

        # === FIELDS (Métriques Numériques - Médicales + Scores + Confiances) ===
        medical_fields = [
            'heart_rate',
            'spo2_level',
            'body_temperature',
            'ecg_signal',
            'respiration_rate',
            'blood_pressure_sys',
            'blood_pressure_dia',
            'blood_glucose',
            'eeg_alpha_power',
            'eeg_beta_power',
            'emg_signal_strength',
            'step_count'
        ] # Le score est une métrique numérique à tracer
    
        
        # 1. Ajout des champs médicaux/scores
        for field in medical_fields:
            if field in message_data and message_data[field] is not None:
                try:
                    point = point.field(field, float(message_data[field]))
                except ValueError:
                    print(f"⚠️ Valeur non-numérique pour le champ médical: {field}")
                    pass 

        # 2. Ajout des Confiances des Modèles (FIELDS)
        if correlation:
            # Confiance finale (FIELD numérique pour le tracé)
            point = point.field("final_confidence", float(correlation.get('confidence', 0.0)))

            if correlation.get("xgb_pred"):
                point = point.tag("xgb_prediction", correlation["xgb_pred"]) # Prediction: TAG
                point = point.field("xgb_confidence", float(correlation.get("xgb_confidence", 0.0))) # Confidence: FIELD

            if correlation.get("cnn_pred"):
                point = point.tag("cnn_lstm_prediction", correlation["cnn_pred"]) # Prediction: TAG
                point = point.field("cnn_confidence", float(correlation.get("cnn_confidence", 0.0))) # Confidence: FIELD

        return point

    except Exception as e:
        print(f"❌ Erreur création point InfluxDB: {e}")
        return None


    
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connecté au broker MQTT")
        client.subscribe(TOPIC)
        print(f"📡 Abonné au topic: {TOPIC}")
    else:
        print(f"❌ Erreur de connexion: {rc}")

def on_message(client, userdata, msg):
    """Callback pour traitement des messages MQTT"""
    global stats, sequence_buffer
    
    try:
        # Parser le message JSON
        message_data = json.loads(msg.payload.decode())
        stats['messages_received'] += 1
        
        # Obtenir les processeurs
        model_loader, xgb_processor, cnn_processor = userdata
        
        # 1. DÉTECTION INSTANTANÉE (étapes XGBoost)
        processed_xgb = xgb_processor.preprocess_for_xgboost(message_data)
        instant_result = xgb_processor.predict_xgboost(processed_xgb)
        
        # 2. PRÉPARATION POUR CNN-LSTM (étapes test3.py)
        processed_cnn = cnn_processor.preprocess_for_cnn_lstm(message_data)
        if processed_cnn is not None:
            sequence_buffer.append(processed_cnn)
        
        # 3. ANALYSE SÉQUENTIELLE si buffer plein
        sequence_result = None
        if len(sequence_buffer) >= SEQUENCE_LENGTH:
            sequence_result = cnn_processor.predict_sequence(list(sequence_buffer))
        
        # 4. CORRÉLATION DES RÉSULTATS
        timestamp = message_data.get('timestamp', datetime.now().isoformat())
        correlation = correlate_results(instant_result, sequence_result, timestamp, message_data)
        
        # 5. STOCKAGE INFLUXDB
        if influx_client:
            sensor_point = create_sensor_data_point(message_data, correlation)
            if sensor_point:
                write_to_influxdb(sensor_point)
        
        # 6. AFFICHAGE DES ALERTES
        if correlation and correlation['final_decision'] != 'NORMAL':
            print(f"\n🚨 ALERTE DÉTECTÉE:")
            print(f"   Patient: {correlation['patient_id']}")
            print(f"   Décision: {correlation['final_decision']}")
            print(f"   Confiance: {correlation['confidence']:.3f}")
            
            if instant_result:
                print(f"   XGBoost: {instant_result['prediction']} ({instant_result['confidence']:.3f})")
            if sequence_result:
                print(f"   CNN-LSTM: {sequence_result['prediction']} ({sequence_result['confidence']:.3f})")
                
        # 7. STATS PÉRIODIQUES
        if stats['messages_received'] % 50 == 0:
            print(f"\n📊 Stats: {stats['messages_received']} msgs | "
                  f"{stats['instant_detections']} XGBoost | "
                  f"{stats['sequence_analyses']} CNN-LSTM | "
                  f"{stats['alerts_confirmed']} confirmées | "
                  f"{stats['alerts_rejected']} rejetées")
                  
    except Exception as e:
        print(f"❌ Erreur traitement message: {e}")

def main():
    """Fonction principale"""
    print("🚀 Consumer Hybride IoT - Exact Preprocessing")
    print("=" * 50)
    
    # Initialiser InfluxDB
    influx_connected = initialize_influxdb()
    if influx_connected:
        print(f"💾 InfluxDB configuré - Bucket: {INFLUXDB_BUCKET}")
    
    # Charger les modèles
    model_loader = ModelLoader()
    if not model_loader.load_all_models():
        print("❌ Échec chargement des modèles")
        return
        
    # Initialiser les processeurs
    xgb_processor = XGBoostProcessor(model_loader)
    cnn_processor = CNNLSTMProcessor(model_loader)
    
    # Configurer MQTT
    client = mqtt.Client(client_id="hybrid_iot_consumer_exact")
    client.user_data_set((model_loader, xgb_processor, cnn_processor))
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print(f"🔗 Connexion au broker MQTT {BROKER}:{PORT}")
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        
        print("🎯 Consumer démarré - Preprocessing exact des codes d'entraînement")
        print("   - XGBoost: code_xgboost.py")
        print("   - CNN-LSTM: test3.py")
        print("   - Corrélation des résultats")
        print("   - Appuyez sur Ctrl+C pour arrêter")
        
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        
        if influx_client:
            influx_client.close()
        
        print("\n📈 Statistiques finales:")
        print(f"   Messages reçus: {stats['messages_received']}")
        print(f"   Détections XGBoost: {stats['instant_detections']}")
        print(f"   Analyses CNN-LSTM: {stats['sequence_analyses']}")
        print(f"   Alertes confirmées: {stats['alerts_confirmed']}")
        print(f"   Alertes rejetées: {stats['alerts_rejected']}")
        if influx_connected:
            print(f"   Écritures InfluxDB: {stats['influxdb_writes']}")
            print(f"   Erreurs InfluxDB: {stats['influxdb_errors']}")
        print("👋 Consumer arrêté")

if __name__ == "__main__":
    main()