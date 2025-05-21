"""
Prognose-Pipeline für Baulandpreise im Kanton Zürich
---------------------------------------------------

Dieses Skript lädt, bereinigt und analysiert Daten zu Baulandpreisen in Zürcher Gemeinden.
Es beinhaltet Machine-Learning-Modelle (Random Forest, XGBoost) und Zeitreihenanalyse (ARIMA).
Zusätzlich werden GEO-Visualisierungen mit Schweizer Open Data erstellt.

Hauptfunktionen:
- Datenerfassung aus verschiedenen Quellen
- Datenbereinigung und Vorverarbeitung
- Feature-Engineering und Zeitreihenanalyse
- Machine Learning mit Random Forest und XGBoost
- ARIMA-Modell für Zeitreihenprognosen
- GEO-Visualisierung der Ergebnisse
- Modellbewertung und Interpretation
"""

# 📚 Notwendige Bibliotheken importieren
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.parse
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import shap
from statsmodels.tsa.arima.model import ARIMA
import geopandas as gpd
import contextily as ctx
from swissparl import get_data

# Konstanten
START_JAHR = 2012
END_JAHR = 2023

# ----------------------------
# DATENLADUNG UND -REINIGUNG
# ----------------------------

def daten_laden_und_bereinigen():
    """
    Lädt und bereinigt alle Datensätze aus GitHub-Repositories.
    Gibt ein Dictionary mit bereinigten DataFrames zurück.
    """
    daten = {}
    
    # 1. Gemeinden nach Bezirk
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/main/Daten/Gemeinden%20nach%20Bezirk%20Kt%20Z%C3%BCrich.xlsx'
    bezirk = pd.read_excel(url)
    
    # Bevölkerungszahlen bereinigen
    bezirk['Anzahl Einwohner (Dez 2018)_clean'] = (
        bezirk['Anzahl Einwohner (Dez 2018)']
        .astype(str)
        .str.replace(' ', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace('.', '', regex=False)
    )
    bezirk['Anzahl Einwohner (Dez 2018)_clean'] = pd.to_numeric(
        bezirk['Anzahl Einwohner (Dez 2018)_clean'], errors='coerce')
    bezirk.rename(columns={'Gebiet_Name': 'Gemeinde'}, inplace=True)
    daten['bezirk'] = bezirk

    # 2. Preise unbebautes Land
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/main/Daten/Preise_unbebautes_Wohnland_Z%C3%BCrich_gepoolt3Jahre.xlsx'
    unbebaut = pd.read_excel(url)
    unbebaut.columns = unbebaut.columns.str.strip()
    unbebaut.dropna(subset=['Durchschnitt'], inplace=True)
    unbebaut = unbebaut.drop(['Q25', 'Q75'], axis=1)
    daten['unbebaut'] = unbebaut

    # 3. Fahrzeitdaten
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/main/Daten/2017_PM_Bodenpreismodell_gerundet.csv'
    fahrzeit = pd.read_csv(url)
    fahrzeit = fahrzeit.drop(['wohngebiet', 'totbev', 'steuerfuss', 'X25.', 'X50.', 'X75.'], axis=1)
    fahrzeit.rename(columns={'bfs': 'BFS_NR', 'Fahrzeit': 'fahrzeit'}, inplace=True)
    daten['fahrzeit'] = fahrzeit

    # 4. Arbeitslosenquote
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/main/Daten/Arbeitslosenanteil%20an%20Bev%C3%B6lkerung%2015-64%20Jahre%20%5B%25%5D%20nach%20Gemeinde.csv'
    arbeitslosenanteil = pd.read_csv(url)
    arbeitslosenanteil = arbeitslosenanteil.drop(
        ['THEMA_NAME', 'SET_NAME', 'SUBSET_NAME', 'INDIKATOR_ID', 'Unnamed: 11', 'EINHEIT_LANG', 'INDIKATOR_NAME'], axis=1)
    arbeitslosenanteil.rename(columns={
        'GEBIET_NAME': 'Gemeinde',
        'INDIKATOR_JAHR': 'Jahr',
        'INDIKATOR_VALUE': 'Arbeitslosenanteil',
        'EINHEIT_KURZ': 'Einheit'
    }, inplace=True)
    daten['arbeitslosenanteil'] = arbeitslosenanteil

    # 5. Steuerfüsse
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/refs/heads/main/Daten/kanton_zuerich_stf_timeseries.csv'
    steuerfuss = pd.read_csv(url)
    steuerfuss = steuerfuss.drop(steuerfuss.columns[4:15], axis=1)
    steuerfuss.rename(columns={
        'BFSNR': 'BFS_NR',
        'GDE_NAME': 'Gemeinde',
        'STF_O_KIRCHE1': 'Steuerfuss_ohne_Kirche',
        'JUR_PERS': 'Juristische_Personen',
        'YEAR': 'Jahr'
    }, inplace=True)
    daten['steuerfuss'] = steuerfuss

    # 6. Baulandpreise
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/refs/heads/main/Daten/Baulandpreis%20Median%20nach%20Gemeinde.csv'
    bauland = pd.read_csv(url)
    bauland = bauland.drop(
        ['THEMA_NAME', 'SET_NAME', 'SUBSET_NAME', 'INDIKATOR_ID', 'Unnamed: 11', 'EINHEIT_LANG', 'INDIKATOR_NAME'], axis=1)
    bauland.rename(columns={
        'GEBIET_NAME': 'Gemeinde',
        'INDIKATOR_VALUE': 'Baulandpreis_Median',
        'INDIKATOR_JAHR': 'Jahr',
        'EINHEIT_KURZ': 'Einheit'
    }, inplace=True)
    daten['bauland'] = bauland

    # 7. Kriminalitätsstatistiken
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/refs/heads/main/Daten/Anzahl%20Straftaten%20nach%20Tatbestand%20und%20Bezirken.csv'
    straftaten = pd.read_csv(url)
    straftaten = straftaten.drop(
        ['Gesetz_Nummer', 'Gesetz_Abk', 'Reihenfolge_Haupttitel', "Haupttitel", "Artikel", "Tatbestand"], axis=1)
    agg_cols = ["Straftaten_total", "Straftaten_vollendet", "Straftaten_versucht", "Häufigkeitszahl"]
    straftaten_summiert = straftaten.groupby(
        ["Ausgangsjahr", "Bezirk_BFS_Nr", "Bezirksname"], as_index=False)[agg_cols].sum()
    straftaten_summiert.rename(columns={
        'Ausgangsjahr': 'Jahr',
        'Bezirk_BFS_Nr': 'Bezirk_BFS_NR'
    }, inplace=True)
    daten['straftaten'] = straftaten_summiert

    # 8. Bevölkerungsdichte
    url = 'https://raw.githubusercontent.com/Bina-G10/BINA-G10/main/Daten/Bev%C3%B6lkerungsdichte%20Einwohner%20pro%202%20km%20nach%20Gemeinde.csv'
    encoded_url = urllib.parse.quote(url, safe=":/%")
    bevölkerungsdichte = pd.read_csv(encoded_url)
    bevölkerungsdichte = bevölkerungsdichte.drop(
        ['SET_NAME', 'SUBSET_NAME', 'THEMA_NAME', 'INDIKATOR_ID', 'INDIKATOR_NAME',
         'EINHEIT_LANG', 'Unnamed: 11'], axis=1)
    bevölkerungsdichte.rename(columns={
        'GEBIET_NAME': 'Gemeinde',
        'INDIKATOR_VALUE': 'Bevölkerungsdichte',
        'INDIKATOR_JAHR': 'Jahr',
        'EINHEIT_KURZ': 'Einheit'
    }, inplace=True)
    daten['bevölkerungsdichte'] = bevölkerungsdichte

    return daten

# ----------------------------
# GEO-DATEN UND VISUALISIERUNG
# ----------------------------

def lade_geo_daten():
    """
    Lädt Geodaten für den Kanton Zürich von der Swiss Open Data Plattform.
    Gibt einen GeoDataFrame mit Gemeindegrenzen zurück.
    """
    # Gemeindegeometrien vom Bundesamt für Statistik
    gemeinden_gdf = gpd.read_file("https://www.bfs.admin.ch/bfsstatic/dam/assets/16944104/master")
    
    # Auf Kanton Zürich filtern (BFS-NR 1)
    zh_gemeinden = gemeinden_gdf[gemeinden_gdf['kanton'] == '01']
    
    return zh_gemeinden

def erstelle_geo_visualisierung(gemeinden_gdf, preise_df, jahr=2023, title="Baulandpreise im Kanton Zürich"):
    """
    Erstellt eine geografische Visualisierung der Baulandpreise.
    
    Args:
        gemeinden_gdf: GeoDataFrame mit Gemeindegrenzen
        preise_df: DataFrame mit Preisdaten
        jahr: Jahr für die Darstellung
        title: Titel der Karte
    """
    # Daten zusammenführen
    merged = gemeinden_gdf.merge(
        preise_df[preise_df['Jahr'] == jahr],
        left_on='gemname',
        right_on='Gemeinde',
        how='left'
    )
    
    # Karte erstellen
    fig, ax = plt.subplots(figsize=(12, 10))
    merged.plot(
        column='Baulandpreis_Median',
        cmap='YlOrRd',
        linewidth=0.5,
        edgecolor='gray',
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey"}
    )
    
    # Basiskarte hinzufügen (Swiss Open Data)
    ctx.add_basemap(ax, crs=gemeinden_gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.CH)
    
    # Titel und Legende
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    plt.tight_layout()
    
    return fig

# ----------------------------
# ARIMA-MODELL
# ----------------------------

def erstelle_arima_modell(zeitreihe, p=1, d=1, q=1):
    """
    Erstellt und trainiert ein ARIMA-Modell für die gegebenen Zeitreihendaten.
    
    Args:
        zeitreihe: Pandas Series mit Zeitreihendaten
        p: AR-Ordnung
        d: Differenzierungsordnung
        q: MA-Ordnung
    
    Returns:
        Trainiertes ARIMA-Modell und Prognose
    """
    # Modell erstellen und trainieren
    modell = ARIMA(zeitreihe, order=(p, d, q))
    ergebnis = modell.fit()
    
    # Prognose für die nächsten 3 Jahre
    prognose = ergebnis.get_forecast(steps=3)
    konfidenzintervall = prognose.conf_int()
    
    return ergebnis, prognose, konfidenzintervall

def visualisiere_arima_ergebnis(zeitreihe, prognose, konfidenzintervall, title="ARIMA-Prognose"):
    """
    Visualisiert die ARIMA-Ergebnisse und Prognose.
    """
    plt.figure(figsize=(12, 6))
    
    # Historische Daten
    plt.plot(zeitreihe.index, zeitreihe, label='Historische Daten')
    
    # Prognose
    prognose_index = pd.date_range(
        start=zeitreihe.index[-1] + pd.DateOffset(years=1),
        periods=3,
        freq='Y'
    )
    plt.plot(prognose_index, prognose.predicted_mean, 'r--', label='Prognose')
    
    # Konfidenzintervall
    plt.fill_between(
        prognose_index,
        konfidenzintervall.iloc[:, 0],
        konfidenzintervall.iloc[:, 1],
        color='pink',
        alpha=0.3
    )
    
    plt.title(title)
    plt.xlabel('Jahr')
    plt.ylabel('Baulandpreis (Median)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    return plt

# ----------------------------
# HAUPTFUNKTIONEN
# ----------------------------

def main():
    # 1. Daten laden und bereinigen
    print("Lade und bereinige Daten...")
    daten_dict = daten_laden_und_bereinigen()
    
    # 2. Geodaten laden
    print("Lade Geodaten...")
    zh_gemeinden = lade_geo_daten()
    
    # 3. Daten integrieren
    print("Integriere Daten...")
    master_df = integrate_data(daten_dict)
    
    # 4. Geo-Visualisierung erstellen
    print("Erstelle GEO-Visualisierung...")
    geo_fig = erstelle_geo_visualisierung(zh_gemeinden, master_df)
    geo_fig.savefig('baulandpreise_zuerich.png')
    print("GEO-Visualisierung gespeichert als 'baulandpreise_zuerich.png'")
    
    # 5. ARIMA-Modell für ausgewählte Gemeinde
    print("\nErstelle ARIMA-Modell...")
    beispiel_gemeinde = "Zürich"
    gemeinde_daten = master_df[master_df['Gemeinde'] == beispiel_gemeinde]
    zeitreihe = gemeinde_daten.set_index('Jahr')['Baulandpreis_Median']
    
    # ARIMA-Modell trainieren
    arima_ergebnis, prognose, konfidenz = erstelle_arima_modell(zeitreihe)
    print(arima_ergebnis.summary())
    
    # ARIMA-Ergebnisse visualisieren
    arima_plot = visualisiere_arima_ergebnis(
        zeitreihe, 
        prognose, 
        konfidenz,
        title=f"ARIMA-Prognose der Baulandpreise für {beispiel_gemeinde}"
    )
    arima_plot.savefig('arima_prognose.png')
    print("ARIMA-Visualisierung gespeichert als 'arima_prognose.png'")
    
    # 6. Machine-Learning-Modelle trainieren
    print("\nTrainiere Machine-Learning-Modelle...")
    X, y, X_agg = prepare_features_target(master_df)
    
    # Random Forest
    rf_modell, rf_r2, rf_rmse = train_random_forest(X, y)
    
    # XGBoost
    xgb_modell, xgb_r2, xgb_rmse = train_xgboost(X, y)
    
    # 7. Feature-Importance visualisieren
    print("\nErstelle Feature-Importance-Diagramme...")
    plot_feature_importance(rf_modell, X.columns.tolist(), "Random Forest Feature Importance")
    plot_feature_importance(xgb_modell, X.columns.tolist(), "XGBoost Feature Importance")
    
    # 8. SHAP-Werte für Modellinterpretation
    print("\nBerechne SHAP-Werte...")
    plot_shap_values(rf_modell, X, X.columns.tolist())
    
    # 9. Prognosen für zukünftige Jahre
    print("\nErstelle Prognosen...")
    X_2023 = master_df.loc[master_df['Jahr'] == 2023, X.columns]
    pred_2025 = rf_modell.predict(X_2023)
    pred_2030 = rf_modell.predict(X_2023)
    
    # Prognose-DataFrame erstellen
    prognosen = master_df.loc[master_df['Jahr'] == 2023, ['Gemeinde', 'Baulandpreis_Median']].copy()
    prognosen['Pred_2025'] = pred_2025
    prognosen['Pred_2030'] = pred_2030
    prognosen['Δ2025-2023'] = prognosen['Pred_2025'] - prognosen['Baulandpreis_Median']
    prognosen['Δ2030-2025'] = prognosen['Pred_2030'] - prognosen['Pred_2025']
    
    # Top 10 Gemeinden nach Prognose
    print("\nTop 10 Gemeinden nach prognostizierten Preisen 2025:")
    print(prognosen.sort_values('Pred_2025', ascending=False).head(10))
    
    # 10. Geo-Visualisierung der Prognosen
    print("\nErstelle GEO-Visualisierung der Prognosen...")
    prognosen_geo = zh_gemeinden.merge(
        prognosen,
        left_on='gemname',
        right_on='Gemeinde',
        how='left'
    )
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Aktuelle Preise
    prognosen_geo.plot(
        column='Baulandpreis_Median',
        cmap='YlOrRd',
        legend=True,
        ax=ax[0],
        missing_kwds={"color": "lightgrey"}
    )
    ax[0].set_title("Baulandpreise 2023")
    ax[0].set_axis_off()
    
    # Prognostizierte Preise
    prognosen_geo.plot(
        column='Pred_2025',
        cmap='YlOrRd',
        legend=True,
        ax=ax[1],
        missing_kwds={"color": "lightgrey"}
    )
    ax[1].set_title("Prognostizierte Baulandpreise 2025")
    ax[1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig('prognose_vergleich.png')
    print("GEO-Visualisierung der Prognosen gespeichert als 'prognose_vergleich.png'")

if __name__ == "__main__":
    main()