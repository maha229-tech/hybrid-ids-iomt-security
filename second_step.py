# second_step.py (version modifiée pour meilleure séparabilité des classes)
"""
Générateur de dataset IoT complet (modifié pour améliorer la séparabilité des classes).
But: conserver réalisme tout en rendant chaque Attack_type statistiquement plus distinct.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_complete_realistic_dataset(
    out_csv="iot_complete_realistic.csv",
    n_rows=40000,
    n_patients=500,
    seed=123
):
    random.seed(seed)
    np.random.seed(seed)
    
    # Définitions des classes d'attaque
    attack_types = ["Normal", "DoS/DDoS", "Eavesdropping", "Injection", 
                    "Selective Forwarding", "Sinkhole", "MitM"]
    # Poids raisonnablement équilibrés pour éviter un bias fort
    weights = [0.25, 0.14, 0.12, 0.12, 0.12, 0.12, 0.13]
    weights = np.array(weights) / np.sum(weights)
    
    # Fonctions de génération de base
    def generate_base_medical_data():
        return {
            "heart_rate": np.clip(np.random.normal(75, 12), 40, 140),
            "spo2_level": np.clip(np.random.normal(97, 1.5), 70, 100),
            "body_temperature": np.clip(np.random.normal(36.8, 0.35), 34.0, 41.0),
            "ecg_signal": np.clip(np.random.normal(0.9, 0.15), 0.05, 6.0),
            "respiration_rate": np.clip(np.random.normal(16, 2.5), 8, 40),
            "blood_pressure_sys": np.clip(np.random.normal(120, 15), 80, 220),
            "blood_pressure_dia": np.clip(np.random.normal(80, 10), 50, 130),
            "blood_glucose": np.clip(np.random.normal(100, 20), 40, 400),
            "eeg_alpha_power": np.clip(np.random.normal(10, 1.8), 2, 30),
            "eeg_beta_power": np.clip(np.random.normal(20, 3.5), 5, 50),
            "emg_signal_strength": np.clip(np.random.normal(0.5, 0.2), 0.01, 3.0),
            "fall_detected": np.random.choice([0, 1], p=[0.985, 0.015]),
            "step_count": max(0, int(np.random.poisson(50))),
            "ambient_temperature": np.clip(np.random.normal(22, 4), 5, 45),
            "stress_level_index": np.clip(np.random.uniform(1, 10), 1, 10)
        }
    
    def generate_base_network_data():
        # lognormal pour paquets, exponential pour durée — valeurs de base "calmes"
        base_fwd = max(1, int(np.random.lognormal(3.0, 0.8)))
        base_bwd = max(0, int(np.random.lognormal(2.0, 0.9)))
        base_duration = max(0.001, np.random.exponential(1.5))
        return {
            "fwd_pkts_tot": base_fwd,
            "bwd_pkts_tot": base_bwd,
            "flow_duration": base_duration,
            "flow_entropy": np.clip(np.random.normal(3.0, 0.6), 0.1, 8.0),
            "anomaly_score": np.clip(np.random.beta(2.0, 8.0), 0.0, 1.0),
            "id.orig_p": np.random.randint(1025, 65000),
            "id.resp_p": np.random.choice([80, 443, 53, 5683, 8080, 1883])
        }
    
    def generate_location_data():
        # Région Maroc (approx), on garde même plage
        return {
            "latitude": np.random.uniform(30.0, 36.5),
            "longitude": np.random.uniform(-9.8, -1.0)
        }
    
    # Plages d'anomaly_score séparées par type pour réduire le chevauchement
    anomaly_ranges = {
        "Normal": (0.0, 0.20),
        "Eavesdropping": (0.20, 0.40),
        "DoS/DDoS": (0.45, 0.75),
        "Selective Forwarding": (0.50, 0.78),
        "Injection": (0.70, 0.95),
        "Sinkhole": (0.60, 0.90),
        "MitM": (0.80, 1.0)
    }
    
    def apply_attack_modifications(data, attack_type, severity="medium"):
        """
        Modifie data en fonction de attack_type en insistant sur des signaux
        distinctifs (réseau vs biomédical), et applique une plage d'anomaly_score dédiée.
        """
        # severity -> facteur d'intensité (contrôlé)
        sev_map = {"low": 0.7, "medium": 1.0, "high": 1.6}
        factor = sev_map.get(severity, 1.0)
        
        # Valeur de base pour anomaly_score selon type (plage spécifique)
        a_min, a_max = anomaly_ranges.get(attack_type, (0.0, 1.0))
        # tirer un score dans la plage, mais tenir compte de la sévérité
        base_as = np.clip(np.random.uniform(a_min, a_max) * (0.9 + 0.2 * (factor-1)), 0, 1)
        data["anomaly_score"] = base_as
        
        # Effets par attaque (sélection de modifications plus distinctives)
        if attack_type == "Normal":
            # Légères variations autour des valeurs de base — pas d'altération majeure
            data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.8, 1.1), 0.1, 8.0)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(0.8, 1.2))
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(0.8, 1.3))
            
        elif attack_type == "DoS/DDoS":
            # Explosion du forward traffic, durée courte, entropy plus faible (flux répétitif)
            multiplier = np.random.uniform(10.0 * factor, 30.0 * factor)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * multiplier)
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(0.05, 0.3))
            data["flow_duration"] = max(0.0005, data["flow_duration"] * np.random.uniform(0.5, 1.2))
            data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.1, 0.6), 0.1, 5.0)
            # biomédical généralement inchangé pour DoS
            data["body_temperature"] *= np.random.uniform(0.99, 1.01)
            
        elif attack_type == "Injection":
            # Corruption biomédicale marquée — altère les signaux vitaux
            corruption = np.random.choice(["mild","moderate","severe"], p=[0.25,0.45,0.30])
            if corruption == "mild":
                data["heart_rate"] = min(220, data["heart_rate"] * np.random.uniform(1.1, 1.3) * factor)
                data["blood_glucose"] = min(500, data["blood_glucose"] * np.random.uniform(1.2, 1.6))
                data["spo2_level"] = np.clip(data["spo2_level"] - np.random.uniform(0,3), 0, 100)
            elif corruption == "moderate":
                data["heart_rate"] = np.random.uniform(120, 190)
                data["body_temperature"] = np.random.uniform(38.5, 41.0)
                data["blood_glucose"] = np.random.uniform(180, 350)
                data["spo2_level"] = np.random.uniform(75, 92)
                data["ecg_signal"] = np.random.uniform(2.0, 4.0)
            else: # severe
                data["heart_rate"] = np.random.uniform(160, 240)
                data["spo2_level"] = np.random.uniform(55, 85)
                data["body_temperature"] = np.random.uniform(39.0, 42.0)
                data["blood_glucose"] = np.random.uniform(300, 600)
                data["ecg_signal"] = np.random.uniform(3.5, 8.0)
            # réseau souvent normal mais anomaly_score élevé (car injection visible dans biomédical)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(0.9, 1.2))
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(0.9, 1.2))
            
        elif attack_type == "Eavesdropping":
            # Flux très long (écoute), débit en backward faible, entropy élevé (observations longues)
            data["flow_duration"] = data["flow_duration"] * np.random.uniform(4.0, 15.0) * factor
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(0.05, 0.4))
            data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(1.5, 2.8), 0.2, 8.0)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(0.8, 1.6))
            # biomédical légèrement modifié parfois (ex: readings sporadiques à 0)
            if np.random.random() < 0.08:
                data["spo2_level"] = 0
            
        elif attack_type == "Selective Forwarding":
            # Perte sélective de paquets — backward fortement réduit, quelques capteurs mis à 0
            selectivity = np.random.uniform(0.5, 0.95) * factor
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * (1 - selectivity))
            data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.3, 0.7), 0.1, 6.0)
            # suppression aléatoire de capteurs (simuler pertes partielles)
            if np.random.random() < 0.45:
                data["spo2_level"] = 0
            if np.random.random() < 0.35:
                data["ecg_signal"] = 0
            if np.random.random() < 0.25:
                data["eeg_alpha_power"] = 0
            
        elif attack_type == "Sinkhole":
            # Absorption du trafic: forward gonflé, backward nul, durée augmentée modestement
            absorption = np.random.uniform(0.75, 0.98)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(3.0, 12.0) * factor)
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * (1 - absorption))
            data["flow_duration"] = data["flow_duration"] * np.random.uniform(1.5, 4.0)
            data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.2, 0.9), 0.1, 6.0)
            # biomédical pas fortement impacté
            if np.random.random() < 0.03:
                data["step_count"] = 0
            
        elif attack_type == "MitM":
            # Manipulation bidirectionnelle: mélange d'altérations réseau et biomédical
            if np.random.random() < 0.6:
                # subtle: petites manipulations en biomédical + légère hausse paquets
                data["heart_rate"] = data["heart_rate"] * np.random.uniform(0.95, 1.15) * factor
                data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(1.2, 2.2))
                data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(1.2, 2.2))
                data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.8, 1.4), 0.1, 8.0)
            else:
                # obvious: modifications plus marquées
                data["heart_rate"] = data["heart_rate"] * np.random.uniform(1.1, 1.4) * factor
                data["body_temperature"] = data["body_temperature"] * np.random.uniform(1.02, 1.08)
                data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(2.0, 6.0))
                data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(2.0, 6.0))
                data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.6, 1.8), 0.1, 8.0)
        
        # Re-clipping pour maintenir réalisme
        data["heart_rate"] = np.clip(data["heart_rate"], 30, 300)
        data["spo2_level"] = np.clip(data.get("spo2_level", 0), 0, 100)
        data["body_temperature"] = np.clip(data["body_temperature"], 32, 45)
        data["blood_glucose"] = np.clip(data["blood_glucose"], 30, 1000)
        data["ecg_signal"] = np.clip(data["ecg_signal"], 0, 10)
        data["blood_pressure_sys"] = np.clip(data["blood_pressure_sys"], 60, 300)
        data["fwd_pkts_tot"] = max(1, int(data["fwd_pkts_tot"]))
        data["bwd_pkts_tot"] = max(0, int(data["bwd_pkts_tot"]))
        data["anomaly_score"] = np.clip(data["anomaly_score"], 0, 1)
        data["flow_entropy"] = np.clip(data["flow_entropy"], 0.1, 8.0)
        
        return data
    
    def add_derived_features(data):
        duration = data["flow_duration"]
        fwd_pkts = data["fwd_pkts_tot"]
        bwd_pkts = data["bwd_pkts_tot"]
        # éviter division par zéro en ajoutant epsilon
        eps = 1e-6
        derived = {
            "fwd_pkts_per_sec": fwd_pkts / (duration + eps),
            "bwd_pkts_per_sec": bwd_pkts / (duration + eps),
            "packet_ratio": (fwd_pkts + 1) / (bwd_pkts + 1),
            "down_up_ratio": (bwd_pkts + 1) / (fwd_pkts + 1),
            "packet_rate_diff": (fwd_pkts - bwd_pkts) / (duration + eps),
            "payload_bytes_per_second": max(1.0, np.random.lognormal(7.5, 1.5))
        }
        return derived
    
    # Génération des données
    data_rows = []
    patient_ids = [f"P{i:04d}" for i in range(1, n_patients + 1)]
    
    print(f"Génération de {n_rows} lignes (seed={seed})...")
    
    for i in range(n_rows):
        if i % 5000 == 0:
            print(f"Progression: {i}/{n_rows} lignes")
        
        # Choix de l'attaque et sévérité
        attack_type = np.random.choice(attack_types, p=weights)
        severity = np.random.choice(["low", "medium", "high"], p=[0.3, 0.55, 0.15])
        
        row_data = {}
        row_data.update(generate_base_medical_data())
        row_data.update(generate_base_network_data())
        row_data.update(generate_location_data())
        
        # Application des modifications spécifiques à l'attaque
        row_data = apply_attack_modifications(row_data, attack_type, severity)
        
        # Ajout des features dérivées
        row_data.update(add_derived_features(row_data))
        
        # Métadonnées / contextuelles
        row_data.update({
            "patient_id": np.random.choice(patient_ids),
            "timestamp": datetime(2025, 1, 1) + timedelta(seconds=i * 2),
            "Attack_type": attack_type,
            "severity_level": severity,
            "proto": np.random.choice(["TCP", "UDP"], p=[0.72, 0.28]),
            "service": np.random.choice(["MQTT", "HTTP", "HTTPS", "DNS", "COAP"], 
                                       p=[0.45, 0.18, 0.18, 0.09, 0.10]),
            "activity_type": np.random.choice(["resting", "walking", "running"], 
                                             p=[0.52, 0.30, 0.18])
        })
        
        data_rows.append(row_data)
    
    # Création DataFrame
    df = pd.DataFrame(data_rows)
    
    # Encodage Attack -> Attack_encoded (fixe pour reproductibilité)
    attack_to_code = {attack: i for i, attack in enumerate(attack_types)}
    df["Attack_encoded"] = df["Attack_type"].map(attack_to_code)
    
    # Nettoyage final: remplissage, clipping, types entiers pour colonnes logiques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    int_cols = ["heart_rate", "spo2_level", "respiration_rate", "blood_pressure_sys", 
                "blood_pressure_dia", "blood_glucose", "fall_detected", "step_count",
                "fwd_pkts_tot", "bwd_pkts_tot", "id.orig_p", "id.resp_p"]
    
    for col in int_cols:
        if col in df.columns:
            # safeguard: convert via round then int to avoid overflow from floats extrêmes
            df[col] = df[col].round(0).astype(int)
    
    # Sauvegarde CSV
    df.to_csv(out_csv, index=False)
    
    print(f"\n✅ Dataset généré: {out_csv}")
    print(f"Lignes: {len(df)}, Colonnes: {len(df.columns)}")
    print(f"Colonnes: {list(df.columns)}")
    print("\nRépartition des classes:")
    print(df["Attack_type"].value_counts(normalize=False).sort_index())
    print("\nAnalyse anomaly_score par classe:")
    print(df.groupby("Attack_type")["anomaly_score"].agg(['min','mean','max']).round(3))
    
    return df

if __name__ == "__main__":
    # Paramètres par défaut — tu peux ajuster n_rows et seed si besoin
    df = generate_complete_realistic_dataset(
        n_rows=40000,
        n_patients=500,
        seed=42
    )
