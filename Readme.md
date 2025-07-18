# 👁️ YOLOv8 Streamlit Person Detection & Tracking

Projet de détection, suivi et comptage de personnes en temps réel à l'aide de **YOLOv8**, avec une interface **Streamlit** interactive et la gestion de **multi-flux vidéo**.

---

## ✅ Fonctionnalités actuelles

- 🎯 Détection de personnes en temps réel avec **YOLOv8** (Ultralytics).
- 📦 Interface Streamlit simple et rapide (`app.py` ou `app_multi_streamlit.py`).
- 🧠 Suivi par ID (tracking).
- 🪵 Log automatique des détections dans `people_log.csv`.
- 📁 Organisation propre du projet.
- 🖼️ Support d’images et de vidéos depuis `/images` et `/videos`.
- 🔁 Gestion de **multi-flux** vidéo simultanés (avec `multi_stream_yolo_logger.py`).

---

## 📈 Améliorations prévues

- 🔢 Ajout du **compteur Entrée/Sortie** par zone.
- 🧮 Calcul des personnes présentes à un instant T.
- 📊 Affichage de **métriques dynamiques** (Streamlit) : nbre total, seuil, alertes.
- 🗂️ Système de **configuration JSON** par salle/flux.
- 🧵 Intégration de **flux simulés** (ex : 10 images par salle).
- 🌐 Intégration Kafka, ZeroMQ ou RabbitMQ pour gérer les flux en parallèle.
- 🎨 UI plus fluide, colorée et “mignonne” (émoticônes, onglets, couleurs pastel).
- 🛑 Ajout d’un bouton **“Stop”** dans l’interface pour arrêter proprement un flux.

---

## 🛠️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/votre-utilisateur/yolov8-streamlit-detection-tracking.git
cd yolov8-streamlit-detection-tracking
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> 📦 Tu peux aussi utiliser `packages.txt` si `requirements.txt` n’est pas à jour.

---

## 🚀 Exécution

### Mode mono-flux (app.py)

```bash
streamlit run app.py
```

### Mode multi-flux (app_multi_streamlit.py)

```bash
streamlit run app_multi_streamlit.py
```

### Mode script (logger)

```bash
python multi_stream_yolo_logger.py
```

> Les vidéos doivent être placées dans le dossier `videos/`.

---

## 📁 Structure du projet

```bash
.
├── app.py                          # Interface Streamlit principale
├── app_multi_streamlit.py         # Version multi-flux Streamlit
├── helper.py                      # Fonctions utilitaires (YOLO, logs, affichage)
├── multi_stream_yolo_logger.py    # Exécution parallèle multi-flux
├── settings.py                    # Paramètres généraux du projet
├── people_log.csv                 # CSV des détections
├── weights/                       # Modèle YOLOv8 (ex: yolov8n.pt)
├── images/                        # Images pour test
├── videos/                        # Vidéos pour test
├── logs/                          # Logs du système
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
```

---

## 🧪 Exemple de log (`people_log.csv`)

| timestamp           | stream_id | person_count |
|---------------------|-----------|--------------|
| 2025-07-18 11:20:12 | cam1      | 3            |
| 2025-07-18 11:20:14 | cam2      | 2            |

---

## 🙌 Contributions

Pour contribuer :

1. Fork ce repo
2. Crée une branche (`feature/ma-fonctionnalité`)
3. Commit puis push
4. Fais une Pull Request 💡

---

## 🧠 Auteurs

- 👨‍💻 Projet réalisé par [TON NOM]
- 📅 Dernière mise à jour : 18/07/2025

---

## 🔐 Licence

Ce projet est open-source sous licence MIT.
