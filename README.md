# Property Sales Preisprognose

Ein Regressionsproblem basierend auf dem "Property Sales" Datensatz von Milwaukee Open Data, um den Verkaufspreis von Immobilien zu prognostizieren.
Dabei liegt besonders die EDA, Pipelineerstellung sowie Hyperparameteroptimierung mit sklearn und die Darstellung und Tests der Modellleistungen im Vordergrund.

---

## Projektbestandteile

- **Exploration & Visualisierung**
  - Datenimport und Zusammenführung (2022, 2023, 2024)
  - Explorative Datenanalyse inklusive aussagekräftiger Visualisierungen (Pandas Profiling, Scatterplots, Korrelationsmatrix)
  - Ableitung: Vorhersage des Verkaufspreises

- **Transformation & Pipeline**
  - Feature Engineering: Extraktion von Adress-Informationen, Saisonalität und Baualter
  - Datenaufbereitung: Behandlung fehlender Werte, Transformation (Winsorisierung, Log-Transformation) und Skalierung
  - Aufbau von Preprocessing-Pipelines für numerische und kategoriale Features

- **Modellierung & Evaluation**
  - Untersuchung verschiedener Modelle (ElasticNet, KNN, Decision Tree)
  - Hyperparameteroptimierung mittels RandomizedSearchCV und Kreuzvalidierung
  - Separater Test (z.B. 2022-Daten) zur Validierung der Prognose

- **Ergebnisdarstellung**
  - Visualisierung der Vorhersagen (Residualplots, Scatterplots, Lernkurven)
  - Aggregation von Ergebnissen zu KPIs (z.B. MAE, R²)

---

## Verzeichnisstruktur

| Ordner/Datei                      | Inhalt/Information                                                 |
|-----------------------------------|---------------------------------------------------------------------|
| **input/**                        | CSV-Daten (2022, 2023, 2024)                                          |
| **output/**                       | Generierte Visualisierungen & Berichte (z.B. Lernkurven, Plots)       |
| **scripts/**                      | Module: `eda.py` & `models.py`                                        |
| `01_EDA.ipynb`                    | Notebook zur Exploration & Feature Engineering                      |
| `02_MODELS.ipynb`                 | Notebook zur Modellierung, Hyperparameteroptimierung & Evaluation     |
| `dockerfile` & `docker-compose.yml`| Containerisierung des Projekts                                      |
| `requirements.txt`                | Python-Abhängigkeiten                                               |
| `.gitignore`                      | Dateien/Ordner, die nicht versioniert werden sollen                  |

---

## Installation

1. **Docker (VM / Container):**
   - `docker-compose up --build`

2. **Optional (lokale Installation):**
   - Python 3.11.11 installieren
   - `pip install -r requirements.txt`

---

## Nutzung

- **Datenexploration & Feature Engineering:**  
  Führe `01_EDA.ipynb` aus, um den Datensatz zu analysieren und mithilfe der Erkenntnisse die Modellierung vorzubereiten.

- **Modellierung & Evaluation:**  
  Starte `02_MODELS.ipynb` für:
  - Preprocessing und Pipeline-Aufbau
  - Modelltraining mit Hyperparameteroptimierung und Kreuzvalidierung
  - Darstellung der Prognoseergebnisse und Modellleistungen (inkl. KPI-Aggregation MAE, R2)

- **Reproduzierbarkeit**
Das Projekt ist komplett reproduzierbar, da alle Skripte und Pipelines feste Random States, vordefinierte Parameterräume sowie klar strukturierte und dokumentierte Verarbeitungs- und Modellierungsschritte verwenden. Zudem sorgen ein versioniertes Git-Repository und eine Docker-Umgebung dafür, dass alle Abhängigkeiten und Umgebungsvariablen dokumentiert und konstant bleiben.

---

## Hinweise

- **Nutzung GenAI:**  
  Im Rahmen dieser Arbeit wurde GenAI, Chat-GPT4o, verwendet um mehr Effizienz zu gewinnen u.a. um Code zu formatieren, verständlicher zu kommentieren, usw.
- **Datenherkunft:**  
  Nutzung des "Property Sales" Datensatzes von Milwaukee Open Data zur Prognose des Verkaufspreises.
- **Datenfilterung:**  
  Fokus auf relevante Immobilienarten (z.B. "Residential", "Condominium") zur Vermeidung von Verzerrungen - in EDA etnschieden.
- **Ergebnisse:**  
  Die finale Modellperformance wird über Metriken wie MAE und R² bewertet und durch Visualisierungen unterstützt.

