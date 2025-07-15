#  VisionProject – Comptage de personnes en temps réel via webcam

## Objectif

Ce projet a pour but de créer un système de **détection et de comptage de personnes** entrant ou sortant d'une salle à l’aide d’une **webcam**, en utilisant des techniques de **vision par ordinateur (OpenCV)**. Le nombre de personnes détectées est ensuite affiché sur un **tableau de bord interactif** mis à jour en temps réel.

---

## Fonctionnalités

- Détection en temps réel de personnes via webcam
- Suivi de leur mouvement à travers une ligne virtuelle (entrée/sortie)
- Comptage dynamique du nombre de personnes présentes
- Détection du dépassement de capacité maximale
- Tableau de bord Streamlit mis à jour automatiquement :
  - Nombre actuel de personnes
  - Alerte de dépassement
  - Historique du comptage (optionnel)

---

## Technologies utilisées

| Composant         | Lib utilisée       |
|-------------------|--------------------|
| Vision par ordi   | OpenCV, cvlib, YOLOv5 (optionnel) |
| Dashboard         | Streamlit          |
| Tracking (option) | centroid tracking ou DeepSort |
| Communication     | JSON ou SQLite local |

---

##  Structure du projet

