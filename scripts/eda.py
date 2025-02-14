#!/usr/bin/env python
"""
Module: eda_scripts.py
Pfad: /scripts/eda_scripts.py

Dieses Modul enthält Funktionen zur Verarbeitung von:
1. Adressdaten (z.B. Extraktion von Hausnummer und Straßentyp)
2. Datumsangaben (Konvertierung in Date-Objekte, Extraktion von Jahr und Jahreszeit)
3. Alter-Berechnung (Umrechnung von Year_Built in Alter, basierend auf 2024)
4. Imputation fehlender Werte basierend auf den spezifischen Strategien
5. Gruppierung seltener Kategorien (Zusammenfassen von Kategorien, die weniger als einen bestimmten Prozentsatz ausmachen, zu "Other")
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##############################
# Adress-Verarbeitung
##############################

def extract_street_number(address):
    """
    Extrahiert die Hausnummer aus der Adresse.
    Unterstützt sowohl einzelne Zahlen als auch Bereiche (z.B. "10611-10613").
    Bei einem Bereich wird der Durchschnittswert zurückgegeben.
    
    Beispiele:
      "9220 N 107TH ST" -> 9220
      "10611-10613 W WABASH AV" -> 10612
    """
    tokens = address.split()
    if tokens:
        token = tokens[0]
        if '-' in token:
            parts = token.split('-')
            try:
                numbers = [int(p) for p in parts if p.isdigit()]
                if numbers:
                    return int(sum(numbers) / len(numbers))
            except ValueError:
                return None
        else:
            try:
                return int(token)
            except ValueError:
                return None
    return None

def extract_street_type(address, known_types=None):
    """
    Extrahiert den Straßentyp aus dem Hauptteil der Adresse.
    Der Hauptteil (alles vor dem ersten Komma) wird in Tokens zerlegt,
    und das letzte Token wird als potenzieller Straßentyp angenommen.
    Ist dieses Token nicht in der Liste bekannter Typen, wird "Other" zurückgegeben.
    
    Beispiele:
      "9220 N 107TH ST"  -> "ST"
      "400 N BROADWAY"    -> "Other"
    """
    if known_types is None:
        known_types = ['ST', 'CT', 'AVE', 'AV', 'RD', 'BLVD', 'LN', 'DR', 'WAY', 'PL', 'TER']
    main_part = address.split(',')[0].strip()
    tokens = main_part.split()
    if tokens:
        candidate = tokens[-1].strip().upper()
        if candidate in known_types:
            return candidate
        else:
            return "Other"
    return None

##############################
# Datums-Verarbeitung
##############################

def convert_to_date(date_series):
    """
    Wandelt eine Datumsspalte in reine Python Date-Objekte (ohne Zeit) um.
    
    Parameter:
      - date_series: Eine Pandas-Serie mit Datumsangaben (als String oder Timestamp).
    
    Rückgabe:
      - Eine Serie von Python date-Objekten.
    """
    return pd.to_datetime(date_series).dt.date

def extract_year_from_date(date_series):
    """
    Extrahiert das Jahr aus einer Datumsspalte.
    Intern wird zunächst convert_to_date aufgerufen, sodass auch String-Daten korrekt verarbeitet werden.
    
    Parameter:
      - date_series: Eine Pandas-Serie mit Datumsangaben.
    
    Rückgabe:
      - Eine Serie mit dem Jahr (als Integer).
    """
    dates = convert_to_date(date_series)
    return dates.apply(lambda x: x.year)

def extract_season_from_date(date_series):
    """
    Bestimmt für jedes Datum die zugehörige Jahreszeit.
    Intern wird zunächst convert_to_date aufgerufen.
    
    Zuordnung:
      - Winter: Monate 12, 1, 2
      - Frühling: Monate 3, 4, 5
      - Sommer: Monate 6, 7, 8
      - Herbst: Monate 9, 10, 11
    
    Parameter:
      - date_series: Eine Pandas-Serie mit Datumsangaben.
    
    Rückgabe:
      - Eine Serie mit der Jahreszeit (als String).
    """
    dates = convert_to_date(date_series)
    return dates.apply(lambda x: 'Win' if x.month in [12, 1, 2]
                                 else ('Spr' if x.month in [3, 4, 5]
                                       else ('Sum' if x.month in [6, 7, 8]
                                             else 'Aut')))

##############################
# Alter-Berechnung
##############################

def get_age_year(date_built_series, date_sold_series):
    """
    Berechnet elementweise das Alter einer Immobilie basierend auf dem Baujahr und dem Verkaufsjahr.
    Falls das Verkaufsjahr kleiner als das Baujahr ist, wird das Alter auf 0 gesetzt.
    
    Parameter:
      - date_built_series: Pandas-Serie, die das Baujahr enthält (als numerisch oder als Datum/Str).
      - date_sold_series: Pandas-Serie, die das Verkaufsjahr enthält (als numerisch oder als Datum/Str).
    
    Rückgabe:
      - Eine Pandas-Serie, in der das Alter (Sale_year - Year_Built) berechnet wird, 
        wobei negative Werte auf 0 gesetzt werden.
        
    Beispiel:
      1980 -> 2024 - 1980 = 44; falls ein Verkaufsjahr vor dem Baujahr liegt, wird 0 zurückgegeben.
    """
    # Konvertiere beide Serien in numerische Werte (falls noch nicht der Fall)
    built_year = pd.to_numeric(date_built_series, errors="coerce")
    sold_year = pd.to_numeric(date_sold_series, errors="coerce")
    
    # Elementweise Differenz berechnen
    age = sold_year - built_year
    # Negative Werte auf 0 setzen (z. B. falls Verkaufsjahr < Baujahr)
    age = age.clip(lower=0)
    return age

##############################
# Imputation fehlender Werte
##############################

def impute_missing_values(df):
    """
    Imputiert fehlende Werte im DataFrame df basierend auf den folgenden Strategien:
      - 'Style': fehlende Werte werden mit dem Modus aufgefüllt.
      - 'Extwall': 
           Für Condominiums (PropType == 'Condominium') werden fehlende Werte mit "Not Applicable" ersetzt.
           Für andere Gruppen werden fehlende Werte mit dem Modus der jeweiligen Gruppe aufgefüllt.
      - Numerische Spalten (Stories, Rooms, FinishedSqft, Bdrms, Building_age):
           Fehlende Werte werden mit dem Median aufgefüllt.
    
    Parameter:
      - df: Pandas DataFrame mit den entsprechenden Spalten.
    
    Rückgabe:
      - df: Der DataFrame mit imputierten Werten.
    """
    # Style: Fehlende Werte mit dem Modus auffüllen
    if df['Style'].isnull().sum() > 0:
        df['Style'] = df['Style'].fillna(df['Style'].mode()[0])
    
    # Extwall: getrennte Behandlung
    condo_mask = df['PropType'] == 'Condominium'
    non_condo_mask = ~condo_mask

    # Wenn 'Extwall' ein kategorischer Datentyp ist, füge "Not Applicable" hinzu, falls noch nicht vorhanden
    if pd.api.types.is_categorical_dtype(df['Extwall']):
        if "Not Applicable" not in df['Extwall'].cat.categories:
            df['Extwall'] = df['Extwall'].cat.add_categories("Not Applicable")
    
    # Für Condominiums: fehlende Werte als "Not Applicable" setzen
    df.loc[condo_mask & df['Extwall'].isnull(), 'Extwall'] = "Not Applicable"
    
    # Für die anderen Gruppen: fehlende Werte mit dem Modus auffüllen
    if df.loc[non_condo_mask, 'Extwall'].isnull().sum() > 0:
        mode_extwall = df.loc[non_condo_mask, 'Extwall'].mode()[0]
        df.loc[non_condo_mask & df['Extwall'].isnull(), 'Extwall'] = mode_extwall
    
    # Numerische Spalten: Stories, Rooms, FinishedSqft, Bdrms, Building_age
    numeric_cols = ['Stories', 'Rooms', 'FinishedSqft', 'Bdrms', 'Building_age']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

##############################
# Anpassung der Datentypen
##############################

def adjust_dtypes(df, col_cat, col_num):
    """
    Passt die Datentypen im DataFrame df an.
    
    Parameter:
      - df: Pandas DataFrame, der die zu konvertierenden Spalten enthält.
      - col_cat: Liste von Spaltennamen, die in den Datentyp 'category' umgewandelt werden sollen.
      - col_num: Liste von Spaltennamen, die in numerische Spalten (mittels pd.to_numeric) konvertiert werden sollen.
      
    Rückgabe:
      - df: Der DataFrame mit den angepassten Datentypen.
    
    Beispiel:
      col_categories = ["PropType", "District", "Style", "Extwall", "StrTyp", "Sale_season", "Sale_year"]
      col_numerical = ["Stories", "Rooms", "FinishedSqft", "Units", "Bdrms", "Fbath", "Hbath", "Lotsize", "Sale_price", "Building_age", "StrNr"]
      df = adjust_dtypes(df, col_categories, col_numerical)
    """
    # Kategoriale Spalten konvertieren
    for col in col_cat:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # Numerische Spalten konvertieren
    for col in col_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


##############################
# Gruppierung seltener Kategorien
##############################

def group_rare_categories(series, threshold=0.05, new_value="Other"):
    """
    Gruppiert seltene Kategorien in einer Pandas-Serie zu new_value, wenn deren Anteil
    unter dem angegebenen Schwellenwert liegt.
    
    Parameter:
      - series: Pandas-Serie mit kategorialen Werten.
      - threshold: Schwellenwert (z.B. 0.05 für 5%), unter dem Kategorien als selten gelten.
      - new_value: Der Wert, der für seltene Kategorien gesetzt wird (Standard: "Other").
      
    Rückgabe:
      - Eine Pandas-Serie, bei der seltene Kategorien ersetzt wurden, 
        als 'category'-Datentyp.
    
    Beispiel:
      grouped_series = group_rare_categories(df['PropType'], threshold=0.05, new_value="Other")
    """
    if not pd.api.types.is_categorical_dtype(series):
        series = series.astype('category')
        print("Converted series to 'category' dtype.")
    
    freq = series.value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    series_grouped = series.apply(lambda x: new_value if x in rare_categories else x)
    return series_grouped.astype('category')

##############################
# PCA-Transformation
##############################

def perform_pca(df, min_variance, target_variable):
    """
    Führt eine PCA auf den Features eines bereits vorverarbeiteten DataFrames durch.
    Der DataFrame sollte numerische Variablen (winsorized, log-transformiert, robust gescaled)
    sowie kategoriale Variablen enthalten. Die kategorialen Variablen werden per One-Hot-Encoding
    in numerische Features überführt. Die Zielvariable wird vom Feature-Set getrennt.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Der vorverarbeitete DataFrame inklusive der Zielvariable.
    min_variance : float
        Der minimale kumulative Varianzanteil (z. B. 0.90 für 90 %), der durch die PCA-Komponenten erklärt werden soll.
    target_variable : str
        Der Name der Zielvariable, die vom Feature-Set getrennt werden soll.
    
    Returns:
    --------
    X : np.array
        Das PCA-transformierte Feature-Array.
    Y : pd.Series
        Die Zielvariable.
    """
    
    # Erstelle eine Kopie des DataFrames, um das Original nicht zu verändern
    df_pc = df.copy()
    
    # Zielvariable extrahieren und vom Feature-Set entfernen
    Y = df_pc[target_variable]
    X_df = df_pc.drop(columns=[target_variable])
    
    # One-Hot-Encoding der kategorialen Variablen (Typ 'category' oder 'object')
    categorical_cols = X_df.select_dtypes(include=['category', 'object']).columns.tolist()
    X_df_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    
    # Konvertiere in ein NumPy-Array
    X = X_df_encoded.values
    
    # ----------------------------------------------
    # Initiale PCA: Bestimme den kumulativen Varianzanteil
    # ----------------------------------------------
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X)
    
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Visualisierung der kumulativen erklärten Varianz
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Anzahl der Komponenten')
    plt.ylabel('Kumulative erklärte Varianz')
    plt.title('PCA: Kumulative erklärte Varianz')
    plt.grid(True)
    plt.show()
    
    # Ermittlung der optimalen Anzahl an Komponenten
    n_components = np.argmax(cumulative_variance >= min_variance) + 1
    print(f"Anzahl der Komponenten zur Erklärung von {int(min_variance * 100)}% der Varianz: {n_components}")
    
    # ----------------------------------------------
    # Durchführung der PCA mit der optimalen Anzahl an Komponenten
    # ----------------------------------------------
    pca_opt = PCA(n_components=n_components)
    X_pca = pca_opt.fit_transform(X)
    
    return X_pca, Y