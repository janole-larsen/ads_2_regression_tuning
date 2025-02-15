import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
import shap

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Begrenzung von Ausreißern in numerischen Daten anhand definierter Quantile.
    Parameter:
      - lower_quantile (float): Unteres Quantil (Standard: 0.01)
      - upper_quantile (float): Oberes Quantil (Standard: 0.99)
    Arbeitet ausschließlich mit NumPy-Arrays.
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def fit(self, X, y=None):
        X_np = np.asarray(X)
        self.lower_bounds_ = np.quantile(X_np, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X_np, self.upper_quantile, axis=0)
        return self
    
    def transform(self, X):
        X_np = np.asarray(X).copy()
        return np.clip(X_np, a_min=self.lower_bounds_, a_max=self.upper_bounds_)

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Wendet eine log(1+x)-Transformation auf angegebene numerische Spalten an.
    Erwartet, dass der Input ein NumPy-Array ist und dass die Spalten als Integer-Indizes angegeben werden.
    """
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_np = np.asarray(X).copy()
        if self.columns is None:
            cols = range(X_np.shape[1])
        else:
            if isinstance(self.columns[0], str):
                raise ValueError("Bei NumPy-Arrays müssen die Spalten als Integer-Indizes angegeben werden.")
            cols = self.columns
        for idx in cols:
            X_np[:, idx] = np.log1p(np.clip(X_np[:, idx], a_min=0, a_max=None))
        return X_np

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Gruppiert seltene Kategorien in einem DataFrame spaltenweise zu einem neuen Wert (Standard: "Other").
    Falls der Input kein DataFrame ist, wird er in einen DataFrame konvertiert und am Ende wieder in ein NumPy-Array umgewandelt.
    Parameter:
      - threshold (float): Schwellenwert (Standard: 0.05)
      - new_value (str): Neuer Wert für seltene Kategorien (Standard: "Other")
    """
    def __init__(self, threshold=0.05, new_value="Other"):
        self.threshold = threshold
        self.new_value = new_value
        
    def fit(self, X, y=None):
        # Konvertiere X in DataFrame, falls nötig
        if not isinstance(X, pd.DataFrame):
            if X.ndim == 1:
                X = pd.Series(X)
            else:
                X = pd.DataFrame(X)
        self.rare_categories_ = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.rare_categories_[col] = freq[freq < self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        # Konvertiere X in DataFrame, falls nötig
        is_input_df = isinstance(X, pd.DataFrame)
        if not is_input_df:
            if X.ndim == 1:
                X = pd.Series(X)
            else:
                X = pd.DataFrame(X)
        X = X.copy()
        for col in X.columns:
            rares = self.rare_categories_.get(col, [])
            X[col] = X[col].apply(lambda x: self.new_value if x in rares else x)
        if is_input_df:
            return X
        else:
            return X.values

class CustomAddressTransformer(BaseEstimator, TransformerMixin):
    """
    Extrahiert aus der 'Address'-Spalte Features wie Hausnummer (StrNr) und Straßentyp (StrTyp).
    Erwartet als Input ein DataFrame mit einer 'Address'-Spalte.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = pd.DataFrame()
        X_transformed['StrNr'] = X['Address'].apply(self.extract_street_number)
        X_transformed['StrTyp'] = X['Address'].apply(self.extract_street_type)
        return X_transformed

    def extract_street_number(self, address):
        try:
            token = address.split()[0]
            if '-' in token:
                parts = token.split('-')
                numbers = [int(p) for p in parts if p.isdigit()]
                return int(np.mean(numbers))
            else:
                return int(token)
        except Exception:
            return np.nan

    def extract_street_type(self, address):
        try:
            main_part = address.split(',')[0]
            tokens = main_part.split()
            candidate = tokens[-1].upper()
            known_types = ['ST', 'CT', 'AVE', 'AV', 'RD', 'BLVD', 'LN', 'DR', 'WAY', 'PL', 'TER']
            return candidate if candidate in known_types else "Other"
        except Exception:
            return np.nan

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Führt Feature Engineering auf den Rohdaten durch:
      - Aus 'Address' werden 'StrNr' und 'StrTyp' extrahiert.
      - Aus 'Sale_date' werden 'Sale_year' und 'Sale_season' erzeugt.
      - 'Building_age' wird als Differenz zwischen 'Sale_year' und 'Year_Built' berechnet.
      - Nicht benötigte Spalten ('Address', 'Sale_date', 'Year_Built') werden entfernt.
    Erwartet als Input ein DataFrame.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        if 'Address' in X_df.columns:
            X_df['StrNr'] = X_df['Address'].apply(self.extract_street_number)
            X_df['StrTyp'] = X_df['Address'].apply(self.extract_street_type)
        if 'Sale_date' in X_df.columns:
            X_df['Sale_date'] = pd.to_datetime(X_df['Sale_date'], errors='coerce')
            X_df['Sale_year'] = X_df['Sale_date'].dt.year
            X_df['Sale_season'] = X_df['Sale_date'].dt.month.apply(self.month_to_season)
        if 'Year_Built' in X_df.columns and 'Sale_year' in X_df.columns:
            X_df['Building_age'] = X_df['Sale_year'] - X_df['Year_Built']
        cols_to_drop = [col for col in ['Address', 'Sale_date', 'Year_Built'] if col in X_df.columns]
        X_df.drop(columns=cols_to_drop, inplace=True)
        return X_df

    def extract_street_number(self, address):
        try:
            token = address.split()[0]
            if '-' in token:
                parts = token.split('-')
                numbers = [int(p) for p in parts if p.isdigit()]
                return int(np.mean(numbers))
            else:
                return int(token)
        except Exception:
            return np.nan

    def extract_street_type(self, address):
        try:
            main_part = address.split(',')[0]
            tokens = main_part.split()
            candidate = tokens[-1].upper()
            known_types = ['ST', 'CT', 'AVE', 'AV', 'RD', 'BLVD', 'LN', 'DR', 'WAY', 'PL', 'TER']
            return candidate if candidate in known_types else "Other"
        except Exception:
            return np.nan

    def month_to_season(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
        
def plot_parameter_interactions(df, col_mae, col_mae_std, view="model"):
    """
    Erstellt Plots basierend auf Parametern und Zielvariablen.
    
    Abhängig vom Parameter "view" werden unterschiedliche Spalten gefiltert:
    
    - view="model":
      Es werden alle Spalten betrachtet, die "param_model__regressor__" im Namen enthalten
      plus die Zielvariablen (col_mae und col_mae_std).
      
    - view="preprocess" (alternativ "preprocessing"):
      Es werden alle Spalten betrachtet, die "param_preprocessing__" im Namen enthalten
      plus die Zielvariablen (col_mae und col_mae_std).
      
    Für beide Views werden folgende Plots erstellt:
    
    1. Heatmap:
       - Es werden die numerischen Spalten (inkl. col_mae, col_mae_std) herangezogen.
       - Vor der Darstellung werden die Spaltennamen um den filter_str gekürzt.
    
    2. Balkendiagramme:
       - Für jede nicht-numerische (kategoriale) Spalte im gefilterten DataFrame wird der Mittelwert von
         col_mae pro Kategorie berechnet.
       - Zusätzlich werden Fehlerbalken anhand des Mittelwerts von col_mae_std angezeigt.
       - In den Plot-Titeln wird der Spaltenname um den filter_str gekürzt.
    
    Parameter:
      - df: DataFrame mit Parametern und Zielvariablen.
      - col_mae: Name der Spalte für den mittleren MAE (z. B. "mean_test_MAE").
      - col_mae_std: Name der Spalte für die Standardabweichung des MAE (z. B. "std_test_MAE").
      - view: "model" oder "preprocess" (alternativ "preprocessing").
    """

    # Filter je nach View
    if view == "model":
        filter_str = "param_model__regressor__"
        title_view = "Modell-Parameter"
    elif view in ["preprocess", "preprocessing"]:
        filter_str = "param_preprocessing__"
        title_view = "Preprocessing-Parameter"
    else:
        raise ValueError("Der Parameter 'view' muss 'model' oder 'preprocess'/'preprocessing' sein.")
    
    # Filter: Spalten, die filter_str enthalten oder die Zielvariablen sind.
    cols_to_use = [col for col in df.columns if (filter_str in col) or (col in [col_mae, col_mae_std])]
    df_subset = df[cols_to_use].copy()

        # 1. Heatmap: Plot der numerischen Spalten
    num_cols = df_subset.select_dtypes(include=[np.number])
    if not num_cols.empty:
        # Kürze Spaltennamen: Entferne filter_str und nimm nur die letzten 15 Zeichen.
        def shorten_col(x):
            x_new = x.replace(filter_str, "") if filter_str in x else x
            return x_new[-15:]
        num_cols_renamed = num_cols.rename(columns=shorten_col)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(num_cols_renamed.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Korrelation zwischen numerischen {title_view} und {col_mae}, {col_mae_std}")
        plt.show()
    else:
        print("Keine numerischen Spalten im gefilterten DataFrame gefunden.")


    # 2. Balkendiagramme: Für jede kategoriale Spalte (nicht numerisch und nicht Zielvariablen)
    cat_cols = [col for col in df_subset.columns 
                if (not pd.api.types.is_numeric_dtype(df_subset[col])) and (col not in [col_mae, col_mae_std])]
    
    for col in cat_cols:
        df_subset[col] = df_subset[col].astype(str).astype("category")
        group_stats = df_subset.groupby(col, observed=False).agg(
            mean_mae=(col_mae, "mean"),
            mean_std=(col_mae_std, "mean")
        ).reset_index().copy()
        
        # Kürze den Spaltennamen für den Plot (falls filter_str enthalten ist)
        short_name = col.replace(filter_str, "") if filter_str in col else col
        
        plt.figure(figsize=(6, 4))
        sns.barplot(data=group_stats, x=col, y="mean_mae", errorbar=None)
        plt.errorbar(x=np.arange(len(group_stats)), y=group_stats["mean_mae"],
                     yerr=group_stats["mean_std"], fmt='none', c='black', capsize=5)
        
        # Kürze die Gruppennamen auf die ersten 10 Zeichen
        current_labels = group_stats[col].astype(str).tolist()
        short_labels = [label[:10] for label in current_labels]
        plt.xticks(ticks=np.arange(len(short_labels)), labels=short_labels, rotation=45)
        
        plt.title(f"Mean {col_mae} ± Mean {col_mae_std} pro Kategorie von {short_name} ({title_view})")
        plt.tight_layout()
        plt.show()

def scatter_plot(dict_best_params, X_train, y_train, X_test, y_test, full_pipeline, kategorien=None, save_path=None):
    """
    Scatter-Plot: Wahre vs. Vorhergesagte Werte

    Parameter:
    - dict_best_params: Dictionary mit den besten Parametern pro Modell
    - X_train, y_train, X_test, y_test: Trainings- und Testdaten
    - full_pipeline: Vorverarbeitungspipeline bzw. Modell-Pipeline
    - kategorien (optional): z.B. X_test['PropType']
    - save_path (optional): Pfad zum Speichern des Plots
    """
    models = list(dict_best_params.keys())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, sharex=True, sharey=True, figsize=(6 * n_models, 5))
    axes = axes if n_models > 1 else [axes]

    data = {}
    for m in models:
        full_pipeline.set_params(**dict_best_params[m])
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        if kategorien is not None:
            df['Kategorie'] = kategorien
        data[m] = df

    # Globale Achsengrenzen für y_test und y_pred
    all_y_pred = pd.concat([df['y_pred'] for df in data.values()])
    all_y_test = pd.concat([df['y_test'] for df in data.values()])
    global_min = min(all_y_pred.min(), all_y_test.min())
    global_max = max(all_y_pred.max(), all_y_test.max())

    handles, labels = None, None
    for i, (ax, m) in enumerate(zip(axes, models)):
        df = data[m]
        if kategorien is not None:
            if i == 0:
                sns.scatterplot(x='y_pred', y='y_test', hue='Kategorie', data=df, ax=ax)
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
            else:
                sns.scatterplot(x='y_pred', y='y_test', hue='Kategorie', data=df, ax=ax, legend=False)
        else:
            sns.scatterplot(x='y_pred', y='y_test', data=df, ax=ax)
        
        # Diagonale Referenzlinie y = x
        ax.plot([global_min, global_max], [global_min, global_max], ls='--', c='r')
        ax.set(xlabel='Vorhergesagte Werte', ylabel='Wahre Werte', title=f'Modell: {m}',
               xlim=(global_min, global_max), ylim=(global_min, global_max))
    
    if kategorien is not None:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def residuen_plot(dict_best_params, X_train, y_train, X_test, y_test, full_pipeline, kategorien=None, save_path=None):
    """
    Residualanalyse
    
    Parameter:
    - dict_best_params: Dictionary mit den besten Parametern pro Modell
    - X_train, y_train, X_test, y_test: Trainings- und Testdaten
    - full_pipeline: Vorverarbeitungspipeline bzw. Modell-Pipeline
    - kategorien (optional): z.B. X_test['PropType']
    - save_path (optional): Pfad zum Speichern des Plots
    """
    models = list(dict_best_params.keys())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, sharex=True, sharey=True, figsize=(6 * n_models, 5))
    axes = axes if n_models > 1 else [axes]
    
    data = {}
    for m in models:
        full_pipeline.set_params(**dict_best_params[m])
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        if kategorien is not None:
            df['Kategorie'] = kategorien
        df['Residual'] = df['y_test'] - df['y_pred']
        data[m] = df

    # Globale Achsengrenzen
    all_preds = pd.concat([df['y_pred'] for df in data.values()])
    global_min_pred, global_max_pred = all_preds.min(), all_preds.max()
    all_residuals = pd.concat([df['Residual'] for df in data.values()])
    global_min_res, global_max_res = all_residuals.min(), all_residuals.max()

    handles, labels = None, None
    for i, (ax, m) in enumerate(zip(axes, models)):
        df = data[m]
        if kategorien is not None:
            if i == 0:
                sns.scatterplot(x='y_pred', y='Residual', hue='Kategorie', data=df, ax=ax)
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
            else:
                sns.scatterplot(x='y_pred', y='Residual', hue='Kategorie', data=df, ax=ax, legend=False)
        else:
            sns.scatterplot(x='y_pred', y='Residual', data=df, ax=ax)
        ax.axhline(0, ls='--', c='r')
        ax.set(xlabel='Vorhergesagte Werte', ylabel='Residuen', title=f'Modell: {m}',
               xlim=(global_min_pred, global_max_pred), ylim=(global_min_res, global_max_res))
    
    if kategorien is not None:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_learning_curve_r2(dict_result_rndm, sort_ascending=True, smooth=True, window=10, save_path=None):
    """
    Visualisiert pro Modell (eine Spalte pro Modell) den Verlauf von R² (mean_test_R2)
    über die Hyperparameter-Iteration, wobei alle Graphen dieselbe y-Achse teilen.
    """
    model_names = list(dict_result_rndm.keys())
    n_models = len(model_names)

    # Globale y-Grenzen ermitteln
    global_min = float('inf')
    global_max = float('-inf')
    for model_name in model_names:
        df_sorted = dict_result_rndm[model_name].dropna(subset=['mean_test_R2']).sort_values(
            by='mean_test_R2', ascending=sort_ascending).reset_index(drop=True)
        y = df_sorted['mean_test_R2'].values
        yerr = df_sorted['std_test_R2'].values
        global_min = min(global_min, (y - yerr).min())
        global_max = max(global_max, (y + yerr).max())

    # sharey=True sorgt für eine gemeinsame y-Achse
    fig, axes = plt.subplots(nrows=1, ncols=n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        df_sorted = dict_result_rndm[model_name].sort_values(by='mean_test_R2', ascending=sort_ascending).reset_index(drop=True)
        x = np.arange(len(df_sorted))
        y = df_sorted['mean_test_R2'].values
        yerr = df_sorted['std_test_R2'].values

        ax = axes[i]
        if smooth and len(x) >= window:
            y_smooth = pd.Series(y).rolling(window, min_periods=1, center=True).mean().values
            yerr_smooth = pd.Series(yerr).rolling(window, min_periods=1, center=True).mean().values
            ax.plot(x, y_smooth, 'b-', label='Geglättete R²')
            ax.fill_between(x, y_smooth - yerr_smooth, y_smooth + yerr_smooth, color='blue', alpha=0.2)
        else:
            ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3, label='R²')
        
        ax.set_xlabel(f"Sortierte Iterationen")
        ax.set_title(f"{model_name}\nLernkurve R²")
        ax.set_ylim(global_min, global_max)
        ax.legend()
        # Nur der erste Plot zeigt die y-Achsenbeschriftung
        if i == 0:
            ax.set_ylabel("Mean Test R²")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()



def get_top_records_per_model(results_dict: dict, top_n_per_model: int, kpi: str) -> pd.DataFrame:
    """
    Übergibt ein Dictionary mit Ergebnissen pro Modell und liefert einen DataFrame mit den Top-N Einträgen,
    sortiert nach der angegebenen KPI. Dabei wird:
      - Internes Rename-Mapping genutzt,
      - MAE-Werte (als negative Werte) in positive Zahlen umgerechnet,
      - R2-Werte in Prozent umgerechnet,
      - Alle numerischen Spalten auf ganze Zahlen gerundet,
      - Die 'Regressor'-Spalte als erste Spalte gesetzt.
    
    Parameter:
      - results_dict: Dictionary mit Modellnamen als Keys und DataFrames als Values.
      - top_n_per_model: Anzahl der Top-Einträge pro Modell.
      - kpi: KPI als String ("R2" oder "MAE"), nach der sortiert wird.
      
    Rückgabe:
      - Ein DataFrame, das alle transformierten Top-N Einträge der Modelle enthält.
    """
    # Internes Rename-Mapping
    rename_mapping = {
        "mean_test_R2": "R2_mean_%",
        "std_test_R2": "R2_std_%",
        "mean_test_MAE": "MAE_mean",
        "std_test_MAE": "MAE_std",
        "mean_test_MedAE": "MedAE_mean", 
        "std_test_MedAE": "MedAE_std",
        #"split0_test_MAE": "MAE_0",
        #"split1_test_MAE": "MAE_1",
        #"split2_test_MAE": "MAE_2",
        #"split3_test_MAE": "MAE_3",
        #"split4_test_MAE": "MAE_4",
    }
    
    sort_col = f"{kpi}_mean"
    if kpi.upper() in ["MAE", "MEDAE"]:
        ascending = True  # Niedrigere Fehler sind besser
        sort_key = lambda x: x.abs()
    else:
        ascending = False  # Bei R2: höhere Werte sind besser
        sort_key = None

    dfs = []
    for model, df in results_dict.items():
        df = df.copy()
        # Nur die im Mapping definierten Spalten beibehalten und umbenennen
        cols = [col for col in rename_mapping if col in df.columns]
        df = df[cols].rename(columns=rename_mapping)
        # Modellnamen als Spalte hinzufügen
        df["Regressor"] = model
        
        # Sortierung nach KPI, sofern vorhanden
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, key=sort_key, ascending=ascending) if sort_key \
                    else df.sort_values(by=sort_col, ascending=ascending)
        
        top_df = df.head(top_n_per_model).copy()
        
        # Transformation: MAE in positive Werte umwandeln, R2 in Prozent umrechnen
        for col in top_df.columns:
            if col.startswith("MAE_"):
                top_df[col] = top_df[col].abs()
            if col.startswith("MedAE_"):
                top_df[col] = top_df[col].abs()
            elif col.startswith("R2_"):
                top_df[col] = top_df[col] * 100
        
        # Alle numerischen Spalten auf ganze Zahlen runden
        numeric_cols = top_df.select_dtypes(include=["number"]).columns
        top_df[numeric_cols] = top_df[numeric_cols].round(0).astype(int)
        
        # Setze 'Regressor' Spalte an den Anfang
        if "Regressor" in top_df.columns:
            cols_order = ["Regressor"] + [col for col in top_df.columns if col != "Regressor"]
            top_df = top_df[cols_order]
        
        dfs.append(top_df)
    
    return pd.concat(dfs, ignore_index=True)



def plot_save_shap(params, X_train, y_train, X_test, full_pipeline, path_save_graph,
                   nsamples=100, background_size=100):
    """
    Setzt die Pipeline-Parameter, trainiert das Modell, berechnet SHAP-Werte,
    formatiert die x-Achse wissenschaftlich (mit wissenschaftlicher Notation) und speichert den SHAP Summary-Plot.

    Args:
      params (dict): Beste Modellparameter (z.B. dict_best_params["KNN"])
      X_train (pd.DataFrame): Trainingsdaten
      y_train (pd.Series): Zielwerte der Trainingsdaten
      X_test (pd.DataFrame): Testdaten für die SHAP-Analyse
      full_pipeline: Die gesamte Pipeline
      path_save_graph (str): Pfad inkl. Dateiname zum Speichern des Plots
      nsamples (int, optional): Anzahl Samples für SHAP-Berechnung (Default=100)
      background_size (int, optional): Größe des Hintergrund-Datensatzes (Default=100)
    """
    # Pipeline konfigurieren und trainieren
    full_pipeline.set_params(**params)
    full_pipeline.fit(X_train, y_train)
    
    # Hintergrund-Datensatz wählen
    background = X_train.sample(background_size, random_state=42)
    
    # Wrapper-Funktion, um sicherzustellen, dass der Input als DataFrame vorliegt
    def predict_wrapper(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=X_train.columns)
        return full_pipeline.predict(X)
    
    # SHAP KernelExplainer initialisieren und SHAP-Werte berechnen
    explainer = shap.KernelExplainer(predict_wrapper, background)
    shap_values = explainer.shap_values(X_test, nsamples=nsamples)
    
    # SHAP Summary-Plot erstellen, aber nicht direkt anzeigen
    shap.summary_plot(shap_values, X_test, show=False)
    
    # x-Achse wissenschaftlich skalieren (wissenschaftliche Notation)
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Plot speichern
    plt.savefig(path_save_graph, bbox_inches='tight')
    plt.show()
    plt.close()


def evaluate_models_new_data(dict_best_params, full_pipeline, X, y, X_2022, y_2022, RANDOM_STATE):
    """
    Führt eine 5-fache Cross-Validation auf einem Trainingssplit der Daten (X, y) durch und evaluiert
    anschließend das finale Modell auf den neuen Daten (X_2022, y_2022).
    
    Zunächst wird X, y in einen Trainings- und Testsplit aufgeteilt, wobei mit Hilfe von stratify=X['PropType']
    sichergestellt wird, dass die Verteilung der 'PropType'-Eigenschaft in beiden Splits erhalten bleibt.
    
    Dabei werden alle Preprocessing-Schritte (z. B. Feature Engineering, Imputation, Skalierung,
    Encoding, PCA etc.) der Pipeline auch auf die neuen Daten angewendet.
    
    Parameter:
      - dict_best_params: Dictionary, das für jedes Modell die besten Parameter enthält.
      - full_pipeline: Die komplette Pipeline (Preprocessing und Modellierung).
      - X: Feature-Matrix der Originaldaten (z. B. 2023/2024).
      - y: Zielvariable der Originaldaten.
      - X_2022: Feature-Matrix der neuen Daten (z. B. 2022).
      - y_2022: Zielvariable der neuen Daten.
      - RANDOM_STATE: Integer, der den Zufallsstatus für reproduzierbare Splits definiert.
      
    Rückgabe:
      - results_df: DataFrame, der die Evaluierungsergebnisse (Durchschnittswerte und Standardabweichungen
        der CV-Metriken sowie Ergebnisse für die neuen Daten) für jedes Modell enthält.
    """
    # Aufteilen in Trainings- und Testdaten unter Beibehaltung der Verteilung der 'PropType'-Eigenschaft
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=X['PropType']
    )
    
    # Definiere die Cross-Validation-Strategie auf dem Trainingssplit
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    for model, best_params in dict_best_params.items():
        # Setze die besten Parameter in der Gesamt-Pipeline
        full_pipeline.set_params(**best_params)
        
        # Führe Cross-Validation auf den Trainingsdaten (X_train, y_train) durch
        cv_results = cross_validate(
            full_pipeline, X_train, y_train,
            cv=cv,
            scoring={
                'mae': 'neg_mean_absolute_error',
                'medae': 'neg_median_absolute_error',
                'r2': 'r2'
            }
        )
        
        # Berechne die durchschnittlichen Scores und Standardabweichungen
        mean_mae = -cv_results['test_mae'].mean()
        std_mae  = cv_results['test_mae'].std()
        mean_medae = -cv_results['test_medae'].mean()
        std_medae  = cv_results['test_medae'].std()
        mean_r2 = cv_results['test_r2'].mean()
        std_r2  = cv_results['test_r2'].std()
        
        # Trainiere die Pipeline auf den kompletten Trainingsdaten (X_train, y_train)
        full_pipeline.fit(X_train, y_train)
        
        # Wende das finale Modell auf die neuen Daten (X_2022, y_2022) an
        y_pred_2022 = full_pipeline.predict(X_2022)
        mae_2022 = mean_absolute_error(y_2022, y_pred_2022)
        medae_2022 = median_absolute_error(y_2022, y_pred_2022)
        r2_2022 = r2_score(y_2022, y_pred_2022)
        
        # Speichere die Ergebnisse für das aktuelle Modell
        results.append({
            'model': model,
            'cv_mae_mean': mean_mae,
            'cv_mae_std': std_mae,
            'cv_medae_mean': mean_medae,
            'cv_medae_std': std_medae,
            'cv_r2_mean': mean_r2,
            'cv_r2_std': std_r2,
            '2022_mae': mae_2022,
            '2022_medae': medae_2022,
            '2022_r2': r2_2022
        })
    
    # Erstelle einen DataFrame mit den Ergebnissen
    results_df = pd.DataFrame(results)
    return results_df

