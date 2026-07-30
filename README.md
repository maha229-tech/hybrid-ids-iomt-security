# hybrid-ids-iomt-security
Système Hybride de Détection d'Intrusion pour Réseaux IoT Médicaux (IoMT)

Système de détection d'intrusion en temps réel combinant XGBoost (détection instantanée) et un modèle CNN-BiLSTM-Attention (analyse séquentielle) pour sécuriser des réseaux de capteurs médicaux connectés (IoMT). Le pipeline simule des flux de données biomédicales et réseau via MQTT, applique une détection hybride, puis stocke et visualise les résultats via InfluxDB et Grafana.

Projet réalisé dans le cadre d'un Mémoire de Master (Big Data & Cloud Computing) : IDS intelligent hybride pour la sécurité en temps réel des réseaux IoT médicaux.

Résultats clés
97,8% de précision sur 7 classes d'attaques
Latence < 200 ms pour la détection instantanée
Architecture hybride : détection immédiate (XGBoost) + confirmation contextuelle (CNN-BiLSTM-Attention sur séquences temporelles)
Architecture du pipeline
┌─────────────────┐      ┌──────────────────────┐
│  second_step.py │ ───► │ iot_complete_realistic.csv │
│ (génération     │      │  (dataset synthétique) │
│  dataset)       │      └──────────────────────┘
└─────────────────┘                │
                                   ▼
                   ┌───────────────┴───────────────┐
                   ▼                                ▼
         ┌────────────────────┐         ┌─────────────────────────┐
         │  code_xgboost.py   │         │      test3.py           │
         │  (entraînement     │         │  (entraînement          │
         │   XGBoost)         │         │   CNN-BiLSTM-Attention) │
         └────────────────────┘         └─────────────────────────┘
                    │                                │
                    ▼                                ▼
      xgboost_instant_detector.pkl      advanced_cnn_lstm_attention.keras
      scaler_instant.pkl, encoders...   best_model_advanced.h5
                                         feature_scaler.pkl, encoders...

┌─────────────────┐   MQTT   ┌─────────────────────────┐
│  publisher.py   │ ───────► │  mqtt_to_influxdb.py     │
│  (simulateur de │  topic:  │  (consumer hybride)       │
│   capteurs IoMT)│  iot/    │  - Détection instantanée  │
└─────────────────┘  medical/│    (XGBoost)               │
                      sensors│  - Analyse séquentielle    │
                             │    (CNN-BiLSTM-Attention)  │
                             │  - Corrélation des résultats│
                             └─────────────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │    InfluxDB     │
                                     └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │     Grafana     │
                                     │  (visualisation)│
                                     └─────────────────┘
Structure du projet
Projet_Master/
├── second_step.py                       # Génération du dataset synthétique IoMT
├── code_xgboost.py                      # Entraînement du détecteur XGBoost
├── test3.py                             # Entraînement du modèle CNN-BiLSTM-Attention
├── publisher.py                         # Simulateur MQTT de capteurs médicaux IoT
├── mqtt_to_influxdb.py                  # Consumer hybride (inférence + InfluxDB)
├── iot_complete_realistic.csv           # Dataset généré (40 000 lignes, 500 patients)
│
├── xgboost_instant_detector.pkl         # Modèle XGBoost entraîné
├── scaler_instant.pkl                   # RobustScaler (pipeline XGBoost)
├── label_encoders_instant.pkl           # LabelEncoders (pipeline XGBoost)
├── feature_columns_instant.pkl          # Liste des features (pipeline XGBoost)
│
├── advanced_cnn_lstm_attention.keras    # Modèle CNN-BiLSTM-Attention (final)
├── best_model_advanced.h5               # Meilleur checkpoint (val_accuracy)
├── feature_scaler.pkl                   # StandardScaler (pipeline CNN-LSTM)
├── label_encoders.pkl                   # LabelEncoders (pipeline CNN-LSTM)
├── feature_columns.pkl                  # Liste des features (pipeline CNN-LSTM)
├── X_seq.npy / y_seq_cat.npy            # Séquences pré-calculées (fenêtres de 30 pas)
│
├── model_artifacts/                     # Métadonnées d'entraînement (historique, métriques, config)
├── catboost_info/                       # Logs d'expérimentation
└── requirements.txt
Dataset

Généré par second_step.py : 40 000 lignes, 500 patients simulés, données biomédicales (fréquence cardiaque, SpO2, ECG, EEG, glycémie, etc.) et réseau (paquets, entropie, score d'anomalie), réparties sur 7 classes :

Classe	Description
Normal	Trafic normal
DoS/DDoS	Déni de service
Eavesdropping	Écoute passive/active
Injection	Corruption des données biomédicales
Selective Forwarding	Perte sélective de paquets
Sinkhole	Absorption du trafic
MitM	Attaque de l'homme du milieu

Chaque type d'attaque possède une signature statistique distincte (plage d'anomaly_score dédiée, effets différenciés sur les données réseau vs. biomédicales) afin d'améliorer la séparabilité des classes.

Prérequis
bash
# Broker MQTT (exemple avec Mosquitto)
sudo apt install mosquitto mosquitto-clients

# InfluxDB (v2.x) et Grafana installés et lancés en local
# InfluxDB : http://localhost:8086
# Grafana  : http://localhost:3000
Dépendances Python
bash
pip install -r requirements.txt

Principales bibliothèques : pandas, numpy, scikit-learn, xgboost, tensorflow, imbalanced-learn, joblib, paho-mqtt, influxdb-client.

Configuration InfluxDB

Avant de lancer mqtt_to_influxdb.py, configure tes propres identifiants (variables d'environnement recommandées plutôt qu'en dur dans le code) :

bash
export INFLUXDB_URL="http://localhost:8086"
export INFLUXDB_TOKEN="ton_token"
export INFLUXDB_ORG="ton_organisation"
export INFLUXDB_BUCKET="iot_health"
Utilisation
1. Générer le dataset
bash
python second_step.py

Produit iot_complete_realistic.csv (40 000 lignes, 7 classes d'attaques, 500 patients).

2. Entraîner les modèles

Détecteur instantané (XGBoost)

bash
python code_xgboost.py

Prétraitement (encodage, normalisation RobustScaler), entraînement XGBoost multi-classes, évaluation (classification report, matrice de confusion). Sauvegarde xgboost_instant_detector.pkl et les artefacts associés.

Détecteur séquentiel (CNN-BiLSTM-Attention)

bash
python test3.py

Feature engineering (ratios cardiovasculaires, respiratoires, réseau), création de séquences temporelles (fenêtres de 30 pas, chevauchement 60%), rééquilibrage SMOTE, entraînement d'un modèle hybride CNN + BiLSTM + Multi-Head Attention. Sauvegarde best_model_advanced.h5 et advanced_cnn_lstm_attention.keras.

3. Lancer la simulation temps réel

Terminal 1 — Démarrer le broker MQTT (si pas déjà lancé)

bash
mosquitto -v

Terminal 2 — Lancer le publisher (simulateur de capteurs IoMT)

bash
python publisher.py

Publie des messages JSON réalistes sur le topic iot/medical/sensors (85% trafic normal, attaques contextuelles courtes et réalistes, sévérité variable).

Terminal 3 — Lancer le consumer hybride

bash
python mqtt_to_influxdb.py

S'abonne au topic MQTT, applique le prétraitement exact des scripts d'entraînement, effectue :

Détection instantanée via XGBoost (message par message)
Analyse séquentielle via CNN-BiLSTM-Attention (dès que le buffer de 30 messages est plein)
Corrélation des deux résultats pour une décision finale
Écriture dans InfluxDB (mesures biomédicales, scores de confiance, décision finale)
4. Visualiser dans Grafana

Connecte Grafana à ta source de données InfluxDB (bucket iot_health), puis crée des dashboards pour visualiser :

Les métriques biomédicales en temps réel par patient
Les scores de confiance XGBoost / CNN-LSTM
Les alertes et décisions finales par type d'attaque
Méthodologie
Prétraitement identique training/inférence : mqtt_to_influxdb.py reproduit exactement les étapes de nettoyage, d'encodage et de normalisation utilisées dans code_xgboost.py et test3.py, afin d'éviter tout écart (data drift) entre entraînement et production.
Détection à deux niveaux : le modèle XGBoost apporte une réponse immédiate (latence < 200ms) ; le modèle CNN-BiLSTM-Attention confirme ou nuance la décision en exploitant le contexte temporel (30 messages consécutifs), réduisant les faux positifs isolés.
Rééquilibrage des classes : SMOTE appliqué sur les séquences d'entraînement + pondération des classes (class_weight) pour limiter le biais vers la classe majoritaire (Normal).
Limitations et pistes d'amélioration
Dataset actuellement synthétique (généré par simulation statistique) — une validation sur données réelles ou semi-réelles (capture réseau IoMT) renforcerait la généralisation.
Le score anomaly_score par classe influence fortement la séparabilité des classes ; une évaluation en conditions plus bruitées est nécessaire.
Passage à un déploiement conteneurisé (Docker Compose : MQTT broker + InfluxDB + Grafana + consumer) envisageable pour faciliter la reproductibilité.
Auteure

Maha Sabki — Master Big Data & Cloud Computing, Université Ibn Tofail, Kénitra LinkedIn
