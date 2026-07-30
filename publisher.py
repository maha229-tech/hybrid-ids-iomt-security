#!/usr/bin/env python3
"""
Publisher MQTT corrigé pour générer des données cohérentes avec le dataset d'entraînement.
Génère les mêmes colonnes et plages de valeurs que second_step.py.
VERSION RÉALISTE : 85% trafic normal, attaques contextuelles courtes
"""

import time
import random
import json
import numpy as np
from datetime import datetime, timezone
import signal
import sys
import paho.mqtt.client as mqtt

# ---------- Config fixe ----------
BROKER = "localhost"
PORT = 1883
TOPIC = "iot/medical/sensors"

# Liste des patients (même format que le dataset)
PATIENTS = [f"P{str(i).zfill(4)}" for i in range(1, 501)]  # P0001 → P0500

# Types d'attaques (identiques au dataset)
ATTACK_TYPES = ["Normal", "DoS/DDoS", "Eavesdropping", "Injection",
               "Selective Forwarding", "Sinkhole", "MitM"]

# Distribution RÉALISTE : 85% Normal
ATTACK_WEIGHTS_LIST = [0.85, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02]

# Niveaux de sévérité
SEVERITY_LEVELS = ["low", "medium", "high"]
SEVERITY_WEIGHTS = [0.3, 0.5, 0.2]

# Durées réalistes
RUN_MIN_ATTACK = 5
RUN_MAX_ATTACK = 40
RUN_MIN_NORMAL = 50
RUN_MAX_NORMAL = 300
BASE_INTERVAL = 1.0

running = True

def handle_sigterm(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)

# MQTT client
client = mqtt.Client(client_id="realistic_iot_publisher")
try:
    client.connect(BROKER, PORT, keepalive=60)
    print(f"Connecté au broker MQTT {BROKER}:{PORT}")
except Exception as e:
    print("Erreur connexion MQTT :", e)
    sys.exit(1)

# ---------- Helpers ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

# --- Convertisseur JSON (fix int64/float64) ---
def json_converter(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def generate_base_medical_data():
    """Génère les données biomédicales de base réalistes"""
    return {
        "heart_rate": int(np.clip(np.random.normal(75, 15), 40, 120)),
        "spo2_level": int(np.clip(np.random.normal(97, 2), 85, 100)),
        "body_temperature": round(float(np.clip(np.random.normal(36.8, 0.4), 35.0, 38.0)), 6),
        "ecg_signal": round(float(np.clip(np.random.normal(0.9, 0.2), 0.1, 2.0)), 6),
        "respiration_rate": int(np.clip(np.random.normal(16, 3), 8, 30)),
        "blood_pressure_sys": int(np.clip(np.random.normal(120, 20), 90, 160)),
        "blood_pressure_dia": int(np.clip(np.random.normal(80, 15), 60, 100)),
        "blood_glucose": int(np.clip(np.random.normal(100, 25), 60, 180)),
        "eeg_alpha_power": round(float(np.clip(np.random.normal(10, 2), 5, 20)), 6),
        "eeg_beta_power": round(float(np.clip(np.random.normal(20, 5), 10, 40)), 6),
        "emg_signal_strength": round(float(np.clip(np.random.normal(0.5, 0.3), 0.1, 2.0)), 6),
        "fall_detected": int(np.random.choice([0, 1], p=[0.98, 0.02])),
        "step_count": max(0, int(np.random.poisson(50))),
        "ambient_temperature": round(float(np.clip(np.random.normal(22, 5), 10, 35)), 6),
        "stress_level_index": round(float(np.clip(np.random.uniform(1, 10), 1, 10)), 6)
    }

def generate_base_network_data():
    """Génère les données réseau de base réalistes"""
    return {
        "fwd_pkts_tot": max(1, int(np.random.lognormal(3, 1))),
        "bwd_pkts_tot": max(0, int(np.random.lognormal(2, 1.5))),
        "flow_duration": max(0.001, round(float(np.random.exponential(2)), 6)),
        "flow_entropy": round(float(np.clip(np.random.normal(3, 1), 0.5, 7.0)), 6),
        "anomaly_score": round(float(np.clip(np.random.beta(2, 8), 0, 1)), 6),
        "id.orig_p": int(np.random.randint(1025, 65000)),
        "id.resp_p": int(np.random.choice([80, 443, 53, 5683, 8080, 1883]))
    }

def generate_location_data():
    """Génère les données géospatiales (région Maroc)"""
    return {
        "latitude": round(float(np.random.uniform(33.5, 35.0)), 6),
        "longitude": round(float(np.random.uniform(-7.7, -6.5)), 6)
    }

def apply_attack_modifications(data, attack_type, severity="medium"):
    """Applique les modifications selon le type d'attaque - identique à second_step.py"""
    
    # Facteurs de sévérité
    severity_factors = {"low": 0.5, "medium": 1.0, "high": 1.8, "none": 0.0}
    factor = severity_factors.get(severity, 1.0)
    
    if attack_type == "Normal":
        # Variation minimale pour le trafic normal
        data["anomaly_score"] = np.clip(data["anomaly_score"], 0, 0.25)
        
    elif attack_type == "DoS/DDoS":
        # DoS avec gradient de sévérité
        if severity == "low":
            multiplier = np.random.uniform(3, 8)
            anomaly_range = (0.3, 0.6)
        elif severity == "high":
            multiplier = np.random.uniform(20, 50)
            anomaly_range = (0.8, 1.0)
        else:  # medium
            multiplier = np.random.uniform(8, 25)
            anomaly_range = (0.5, 0.85)
            
        data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * multiplier)
        data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(0.1, 0.4))
        data["flow_entropy"] = np.clip(data["flow_entropy"] * np.random.uniform(0.2, 0.8), 0.1, 8)
        data["anomaly_score"] = np.random.uniform(*anomaly_range)
        
    elif attack_type == "Injection":
        # Injection avec altération des données biomédicales
        corruption_level = np.random.choice(["mild", "moderate", "severe"], p=[0.3, 0.4, 0.3])
        
        if corruption_level == "mild":
            # Altération subtile
            data["heart_rate"] = min(140, data["heart_rate"] * np.random.uniform(1.1, 1.3))
            data["body_temperature"] = min(39, data["body_temperature"] * np.random.uniform(1.02, 1.06))
            data["blood_glucose"] = min(200, data["blood_glucose"] * np.random.uniform(1.2, 1.6))
            data["anomaly_score"] = np.random.uniform(0.4, 0.7)
            
        elif corruption_level == "moderate":
            data["heart_rate"] = np.random.uniform(130, 180)
            data["body_temperature"] = np.random.uniform(38.5, 40.5)
            data["blood_glucose"] = np.random.uniform(200, 350)
            data["ecg_signal"] = np.random.uniform(2.0, 4.0)
            data["spo2_level"] = np.random.uniform(80, 92)
            data["anomaly_score"] = np.random.uniform(0.6, 0.9)
            
        else:  # severe
            data["heart_rate"] = np.random.uniform(180, 250)
            data["spo2_level"] = np.random.uniform(60, 85)
            data["body_temperature"] = np.random.uniform(40, 42)
            data["blood_glucose"] = np.random.uniform(300, 500)
            data["ecg_signal"] = np.random.uniform(4.0, 8.0)
            data["blood_pressure_sys"] = np.random.uniform(160, 220)
            data["anomaly_score"] = np.random.uniform(0.8, 1.0)
            
    elif attack_type == "Eavesdropping":
        # Eavesdropping avec patterns temporels
        listening_type = np.random.choice(["passive", "active"])
        
        if listening_type == "passive":
            data["flow_duration"] = data["flow_duration"] * np.random.uniform(3, 8)
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(0.1, 0.4))
            data["flow_entropy"] = data["flow_entropy"] * np.random.uniform(1.3, 2.2)
            data["anomaly_score"] = np.random.uniform(0.2, 0.6)
        else:  # active
            data["flow_duration"] = data["flow_duration"] * np.random.uniform(8, 20)
            data["flow_entropy"] = data["flow_entropy"] * np.random.uniform(1.8, 3.5)
            data["anomaly_score"] = np.random.uniform(0.5, 0.8)
            
    elif attack_type == "Selective Forwarding":
        # Forwarding sélectif avec perte de paquets
        selectivity = np.random.uniform(0.4, 0.9)
        
        data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * (1 - selectivity))
        data["flow_entropy"] = data["flow_entropy"] * np.random.uniform(0.3, 0.7)
        data["anomaly_score"] = np.random.uniform(0.4, 0.8)
        
        # Suppression occasionnelle de capteurs
        if np.random.random() < 0.3:
            data["spo2_level"] = 0
        if np.random.random() < 0.2:
            data["ecg_signal"] = 0
        if np.random.random() < 0.15:
            data["eeg_alpha_power"] = 0
            
    elif attack_type == "Sinkhole":
        # Sinkhole avec absorption de trafic
        absorption_rate = np.random.uniform(0.7, 0.95)
        
        data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(4, 12))
        data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * (1 - absorption_rate))
        data["flow_duration"] = data["flow_duration"] * np.random.uniform(2, 6)
        data["flow_entropy"] = data["flow_entropy"] * np.random.uniform(0.2, 0.8)
        data["anomaly_score"] = np.random.uniform(0.5, 0.9)
        
    elif attack_type == "MitM":
        # Man-in-the-Middle avec manipulation bidirectionnelle
        manipulation_level = np.random.choice(["subtle", "obvious"])
        
        if manipulation_level == "subtle":
            data["heart_rate"] = data["heart_rate"] * np.random.uniform(0.95, 1.15)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(1.2, 2.0))
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(1.2, 2.0))
            data["flow_entropy"] = data["flow_entropy"] * np.random.uniform(0.8, 1.4)
            data["anomaly_score"] = np.random.uniform(0.3, 0.65)
        else:  # obvious
            data["heart_rate"] = data["heart_rate"] * np.random.uniform(1.15, 1.4)
            data["body_temperature"] = data["body_temperature"] * np.random.uniform(1.02, 1.08)
            data["fwd_pkts_tot"] = int(data["fwd_pkts_tot"] * np.random.uniform(2, 6))
            data["bwd_pkts_tot"] = int(data["bwd_pkts_tot"] * np.random.uniform(2, 6))
            data["flow_entropy"] = data["flow_entropy"] * np.random.uniform(0.6, 1.8)
            data["anomaly_score"] = np.random.uniform(0.6, 0.95)
    
    # Re-clipping après modifications
    data["heart_rate"] = int(np.clip(data["heart_rate"], 30, 300))
    data["spo2_level"] = int(np.clip(data["spo2_level"], 0, 100))
    data["body_temperature"] = round(float(np.clip(data["body_temperature"], 32, 45)), 6)
    data["blood_glucose"] = int(np.clip(data["blood_glucose"], 30, 600))
    data["ecg_signal"] = round(float(np.clip(data["ecg_signal"], 0, 10)), 6)
    data["blood_pressure_sys"] = int(np.clip(data["blood_pressure_sys"], 60, 250))
    data["fwd_pkts_tot"] = max(1, data["fwd_pkts_tot"])
    data["bwd_pkts_tot"] = max(0, data["bwd_pkts_tot"])
    data["anomaly_score"] = round(float(np.clip(data["anomaly_score"], 0, 1)), 6)
    data["flow_entropy"] = round(float(np.clip(data["flow_entropy"], 0.1, 8.0)), 6)
    
    return data

def add_derived_features(data):
    """Calcule les features dérivées à partir des données de base - identique à second_step.py"""
    duration = data["flow_duration"]
    fwd_pkts = data["fwd_pkts_tot"]
    bwd_pkts = data["bwd_pkts_tot"]
    
    derived = {
        "fwd_pkts_per_sec": round(float(fwd_pkts / duration), 6),
        "bwd_pkts_per_sec": round(float(bwd_pkts / duration), 6),
        "packet_ratio": round(float((fwd_pkts + 1) / (bwd_pkts + 1)), 6),
        "down_up_ratio": round(float((bwd_pkts + 1) / (fwd_pkts + 1)), 6),
        "packet_rate_diff": round(float((fwd_pkts - bwd_pkts) / duration), 6),
        "payload_bytes_per_second": round(float(np.random.lognormal(8, 2)), 6)
    }
    
    return derived

def select_contextual_attack(previous_attack):
    """Sélectionne une attaque avec logique contextuelle"""
    
    # Si dernière action = Eavesdropping → attaque ciblée
    if previous_attack == "Eavesdropping":
        return random.choice(["Injection", "MitM"])
    
    # Sinon attaque aléatoire (exclure Normal)
    attacks = ["DoS/DDoS", "Eavesdropping", "Injection", 
               "Selective Forwarding", "Sinkhole", "MitM"]
    weights = [0.30, 0.20, 0.20, 0.15, 0.08, 0.07]
    
    return np.random.choice(attacks, p=weights)

def select_realistic_severity(attack_type):
    """Sévérité selon le type d'attaque"""
    
    # Eavesdropping commence toujours par reconnaissance légère
    if attack_type == "Eavesdropping":
        return random.choice(["low", "medium"])
    
    # DoS escalade souvent
    elif attack_type == "DoS/DDoS":
        return np.random.choice(["medium", "high"], p=[0.4, 0.6])
    
    # Injection/MitM peuvent être sévères
    elif attack_type in ["Injection", "MitM"]:
        return np.random.choice(["medium", "high"], p=[0.5, 0.5])
    
    # Autres attaques distribution normale
    else:
        return np.random.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])

def generate_realistic_scenario(block_size=1000):
    """Génère un scénario d'attaque réaliste avec contexte"""
    runs = []
    remaining = block_size
    last_attack = None
    
    while remaining > 0:
        # Décision : Normal ou Attaque (85% Normal)
        if random.random() < 0.85 or remaining < RUN_MIN_ATTACK:
            # TRAFIC NORMAL
            run_len = min(random.randint(RUN_MIN_NORMAL, RUN_MAX_NORMAL), remaining)
            runs.append(("Normal", "none", run_len))
            last_attack = None
        else:
            # ATTAQUE
            attack_type = select_contextual_attack(last_attack)
            severity = select_realistic_severity(attack_type)
            run_len = min(random.randint(RUN_MIN_ATTACK, RUN_MAX_ATTACK), remaining)
            
            runs.append((attack_type, severity, run_len))
            last_attack = attack_type
            remaining -= run_len
            
            # Pause après attaque (comportement réaliste)
            if remaining > 0:
                pause_len = min(random.randint(30, 100), remaining)
                runs.append(("Normal", "none", pause_len))
                remaining -= pause_len
                last_attack = None
            continue
        
        remaining -= run_len
    
    return runs

def build_message(attack_type, severity):
    """Construire un message complet avec la même structure que second_step.py"""
    
    # Correction : Normal n'a pas de severity
    if attack_type == "Normal":
        severity = "low"
    
    # Métadonnées et informations contextuelles
    message = {
        "patient_id": random.choice(PATIENTS),
        "timestamp": now_iso(),
        "Attack_type": attack_type,
        "severity_level": severity,
        "proto": np.random.choice(["TCP", "UDP"], p=[0.7, 0.3]),
        "service": np.random.choice(["MQTT", "HTTP", "HTTPS", "DNS", "COAP"], 
                                  p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        "activity_type": np.random.choice(["resting", "walking", "running"], 
                                        p=[0.5, 0.3, 0.2])
    }
    
    # Génération des données de base
    message.update(generate_base_medical_data())
    message.update(generate_base_network_data())
    message.update(generate_location_data())
    
    # Application des modifications d'attaque
    message = apply_attack_modifications(message, attack_type, severity)
    
    # Ajout des features dérivées
    message.update(add_derived_features(message))
    
    # Encodage des classes (comme dans le dataset original)
    attack_to_code = {
        "Normal": 0,
        "DoS/DDoS": 1, 
        "Eavesdropping": 2,
        "Injection": 3,
        "Selective Forwarding": 4,
        "Sinkhole": 5,
        "MitM": 6
    }
    message["Attack_encoded"] = attack_to_code.get(attack_type, 0)
    
    return message

def main_loop():
    """Boucle principale du publisher"""
    print(f"Publisher REALISTE demarre -> mqtt://{BROKER}:{PORT}")
    print(f"Topic: {TOPIC}")
    print("Scenarios realistes : 85% Normal, attaques contextuelles courtes")
    print("Compatible avec le dataset d'entrainement (second_step.py)")
    
    block_size = 1000
    message_count = 0
    
    try:
        while running:
            runs = generate_realistic_scenario(block_size)
            for attack_type, severity, run_len in runs:
                # Affichage amélioré
                if attack_type != "Normal":
                    print(f">>> ATTAQUE: {attack_type} ({severity}) - {run_len} messages")
                elif run_len > 100:
                    print(f">>> Normal - {run_len} messages")
                
                for i in range(run_len):
                    if not running:
                        break
                        
                    message = build_message(attack_type, severity)
                    
                    # Publication MQTT
                    client.publish(TOPIC, json.dumps(message, default=json_converter))
                    message_count += 1
                    
                    if message_count % 100 == 0:
                        print(f"Messages publies: {message_count}")
                    
                    time.sleep(BASE_INTERVAL)
                    
                if not running:
                    break
                    
    except KeyboardInterrupt:
        print("Interruption utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        client.disconnect()
        print(f"Publisher arrete. Total messages: {message_count}")

if __name__ == "__main__":
    # Initialisation des graines pour reproductibilité partielle
    random.seed(int(time.time()))
    np.random.seed(int(time.time()) % 2**32)
    main_loop()