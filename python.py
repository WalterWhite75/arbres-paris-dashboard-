import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# Configuration Plotly globale 
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.family = "Roboto, Arial, sans-serif"
pio.templates["plotly_white"].layout.title.font.size = 20
pio.templates["plotly_white"].layout.title.font.color = "#2e7d32"
pio.templates["plotly_white"].layout.xaxis.title.font.size = 14
pio.templates["plotly_white"].layout.yaxis.title.font.size = 14
pio.templates["plotly_white"].layout.xaxis.tickfont.size = 12
pio.templates["plotly_white"].layout.yaxis.tickfont.size = 12

#  Ajout import & chargement GeoJSON pour choroplèthes
import json
import urllib.request

geojson_url = "https://opendata.paris.fr/explore/dataset/arrondissements/download/?format=geojson&timezone=Europe/Berlin&lang=fr"
with urllib.request.urlopen(geojson_url) as response:
    paris_geojson = json.load(response)


# 1️) Données


file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'arbres-paris-dashboard', 'arbres-plantes-par-projet.xlsx')
print("Chemin du fichier Excel utilisé :", file_path)
df = pd.read_excel(file_path)

if "Nombre d'arbres plantés" in df.columns:
    df.rename(columns={"Nombre d'arbres plantés": 'nombre_arbres'}, inplace=True)

if 'Nom du projet' in df.columns:
    df['année'] = df['Nom du projet'].str.extract(r'(20\d{2})')
    df['année'] = pd.to_numeric(df['année'], errors='coerce').astype('Int64')

def extract_lat_lon(coord):
    try:
        lat, lon = coord.split(',')
        return float(lat.strip()), float(lon.strip())
    except:
        return None, None

df['latitude'], df['longitude'] = zip(*df['geo_point_2d ( coordonnées ) '].map(extract_lat_lon))
df = df.dropna(subset=['année', 'nombre_arbres'])

# Création d'une version agrégée "Paris Centre" (1er, 2e, 3e, 4e arrondissements)
paris_centre_df = df[df['Arrondissement'].isin([1, 2, 3, 4])].copy()
if not paris_centre_df.empty:
    paris_centre_grouped = paris_centre_df.groupby('année', as_index=False)['nombre_arbres'].sum()
    paris_centre_grouped['Arrondissement'] = 'Paris Centre'
    paris_centre_grouped['is_paris_centre_agg'] = True
    df['is_paris_centre_agg'] = False
    df = pd.concat([df, paris_centre_grouped], ignore_index=True)

# Colonne Arrondissement_mod : copie directe sauf pour Paris Centre agrégé
df['Arrondissement_mod'] = df['Arrondissement']

print("Colonnes :", df.columns.tolist())
print(df.head())

# 2️) Analyse simple

annees = df.groupby('année')['nombre_arbres'].sum()

X = np.array([int(a) for a in annees.index]).reshape(-1, 1)
y = annees.values
model = LinearRegression().fit(X, y)

# Modèle polynomial d'ordre 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)

# Calcul RMSE pour comparaison
y_pred_linear = model.predict(X)
y_pred_poly = poly_model.predict(X_poly)

rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

print(f"RMSE linéaire : {rmse_linear:.2f}")
print(f"RMSE polynomial (ordre 2) : {rmse_poly:.2f}")

next_year = int(max(annees.index)) + 1
next_year_prediction = model.predict(np.array([[next_year]]))
print(f"Prévision pour {next_year} : {int(next_year_prediction[0])} arbres plantés")

# 3️) Dashboard


# Utilisation du dossier assets externe sur le bureau
ASSETS_DIR = os.path.join(os.path.expanduser('~'), 'Desktop', 'arbres-paris-dashboard', 'assets')
print("Dossier assets utilisé :", ASSETS_DIR)
app = Dash(__name__, assets_folder=ASSETS_DIR)

# Définition d'un index_string personnalisé pour styles globaux modernes
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <!-- Import Google Fonts Roboto -->
        <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,600&display=swap" rel="stylesheet">
        <style>
            html, body {
                background: #f4f6f8 !important;
                font-family: 'Roboto', Arial, sans-serif !important;
                margin: 0; padding: 0;
                color: #222;
            }
            h1, h2, h3, h4, h5 {
                font-family: 'Roboto', Arial, sans-serif !important;
                color: #2e7d32;
                font-weight: 600;
                margin-top: 0.7em;
                margin-bottom: 0.4em;
            }
            h1 { font-size: 2.2em; }
            h2 { font-size: 1.5em; }
            h3 { font-size: 1.2em; }
            p {
                font-family: 'Roboto', Arial, sans-serif !important;
                color: #333;
                font-size: 1.06em;
                margin-top: 0.3em;
                margin-bottom: 1em;
            }
            /* Tabs modern style */
            .dash-tabs {
                background: #fff !important;
                border-radius: 12px !important;
                box-shadow: 0 2px 8px 0 rgba(60,72,88,0.08);
                padding: 0.5em 0.5em 0 0.5em;
                margin-bottom: 18px;
            }
            .dash-tab {
                background: #fff !important;
                color: #2e7d32 !important;
                border-radius: 8px 8px 0 0 !important;
                border: none !important;
                margin-right: 6px !important;
                transition: background 0.2s;
                font-weight: 500;
                font-family: 'Roboto', Arial, sans-serif !important;
            }
            .dash-tab--selected {
                background: #fff !important;
                border-bottom: 3px solid #2e7d32 !important;
                color: #2e7d32 !important;
                font-weight: 600 !important;
                box-shadow: none !important;
            }
            .dash-tab:hover {
                background: #f4f6f8 !important;
                color: #256026 !important;
            }
            /* KPI container modern */
            #kpi-container {
                gap: 24px;
            }
            #kpi-container > div {
                background: #fff;
                box-shadow: 0 2px 10px 0 rgba(60,72,88,0.10);
                border-radius: 14px;
                padding: 20px 30px;
                margin: 8px 0;
                min-width: 160px;
            }
            #kpi-container h3 {
                color: #2e7d32 !important;
                font-family: 'Roboto', Arial, sans-serif !important;
                font-weight: 600 !important;
            }
            /* Dropdown modern */
            .Select-control, .Select-menu-outer {
                background: #fff !important;
                border-radius: 8px !important;
                border: 1px solid #b2dfdb !important;
                box-shadow: 0 1px 6px 0 rgba(60,72,88,0.06);
            }
            /* Header modern */
            .header-modern {
                background: #fff;
                box-shadow: 0 2px 10px 0 rgba(60,72,88,0.10);
                border-radius: 16px;
                padding: 18px 32px;
                margin-bottom: 32px;
                display: grid;
                grid-template-columns: 80px 1fr 80px;
                align-items: center;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


unique_years = sorted([int(a) for a in df['année'].dropna().unique()])
annee_options = [{'label': str(a), 'value': a} for a in unique_years]

# 4️) Fonctions graphiques + KPI


def compute_kpis(selected_years, selected_arrondissements):
    dff = df[df['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement_mod'].isin(selected_arrondissements)]
    total_arbres = int(dff['nombre_arbres'].sum())
    total_projets = dff.shape[0]
    total_arrondissements = dff['Arrondissement_mod'].nunique()
    moyenne_par_projet = round(dff['nombre_arbres'].mean(), 1) if total_projets > 0 else 0
    return total_arbres, total_projets, total_arrondissements, moyenne_par_projet

# Nouvelle fonction make_annee_bar prenant en compte le filtre arrondissement
def make_annee_bar(selected_years, selected_arrondissements):
    # Filtrer par arrondissement si sélectionné
    dff = df.copy()
    if selected_arrondissements:
        dff = dff[dff['Arrondissement_mod'].isin(selected_arrondissements)]
    grouped = dff.groupby('année')['nombre_arbres'].sum()
    x = [int(a) for a in grouped.index]
    y = list(grouped.values)
    colors = ['#2e7d32' if (int(xi) in (selected_years or [])) else '#9CCC65' for xi in x]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
    fig.update_layout(
        title="Nombre d'arbres plantés par année",
        xaxis_title='Année',
        yaxis_title="Nombre d'arbres",
        template="plotly_white",
        title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# Réintroduction de la fonction make_arrondissement_figure standard
def make_arrondissement_figure(selected_years, selected_arrondissements):
    # Fallback : si selected_years est vide ou None, prend toutes les années
    if not selected_years:
        selected_years = unique_years
    dff = df[df['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement_mod'].isin(selected_arrondissements)]
    arr_data = dff.groupby('Arrondissement_mod')['nombre_arbres'].sum().sort_values(ascending=False)
    if arr_data.empty:
        # Figure vide avec message
        fig = go.Figure()
        fig.update_layout(
            title="Aucune donnée disponible pour la sélection",
            template="plotly_white",
            title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        return fig
    fig = px.bar(
        x=arr_data.index.astype(str),
        y=arr_data.values,
        labels={'x': 'Arrondissement', 'y': "Nombre d'arbres"},
        title=f"Répartition par arrondissement ({', '.join(map(str, selected_years))})"
    )
    fig.update_layout(
        template="plotly_white",
        title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# ➕ Cartes choroplèthes
def make_choropleth_total(selected_years):
    # Fallback : si selected_years est vide ou None, prend toutes les années
    if not selected_years:
        selected_years = unique_years
    dff = df[df['année'].isin(selected_years)]
    grouped = dff.groupby('Arrondissement_mod')['nombre_arbres'].sum().reset_index()
    # Retirer Paris Centre car pas dans le GeoJSON
    grouped = grouped[grouped['Arrondissement_mod'] != "Paris Centre"]
    if grouped.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Aucune donnée disponible",
            margin={"r":0,"t":40,"l":0,"b":0},
            template="plotly_white",
            title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
        )
        return fig
    fig = px.choropleth_mapbox(
        grouped,
        geojson=paris_geojson,
        locations='Arrondissement_mod',
        featureidkey="properties.c_ar",
        color='nombre_arbres',
        color_continuous_scale="Greens",
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 48.8566, "lon": 2.3522},
        opacity=0.6,
        labels={'nombre_arbres': 'Nombre d’arbres'}
    )
    fig.update_layout(
        title="Carte choroplèthe - Total d’arbres plantés par arrondissement",
        margin={"r":0,"t":40,"l":0,"b":0},
        template="plotly_white",
        title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
    )
    return fig

def make_choropleth_tcam():
    # Créer un DataFrame tcam_df_mod avec Arrondissement_mod comme clé
    tcam_df_mod = []
    for arr in sorted(df['Arrondissement_mod'].unique(), key=lambda x: (0 if x=="Paris Centre" else x)):
        # Exclure Paris Centre de la carte
        if arr == "Paris Centre":
            continue
        grouped = df[df['Arrondissement_mod'] == arr].groupby('année')['nombre_arbres'].sum()
        tcam = calculate_tcam(grouped)
        if tcam is not None:
            tcam_df_mod.append({'Arrondissement_mod': arr, 'TCAM': tcam * 100})
    tcam_df_mod = pd.DataFrame(tcam_df_mod)
    fig = px.choropleth_mapbox(
        tcam_df_mod,
        geojson=paris_geojson,
        locations='Arrondissement_mod',
        featureidkey="properties.c_ar",
        color='TCAM',
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 48.8566, "lon": 2.3522},
        opacity=0.6,
        labels={'TCAM': 'TCAM (%)'}
    )
    fig.update_layout(
        title="Carte choroplèthe - TCAM (%) par arrondissement",
        margin={"r":0,"t":40,"l":0,"b":0},
        template="plotly_white",
        title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
    )
    return fig


# + Fonction utilitaire TCAM

def calculate_tcam(grouped_data):
    if len(grouped_data) < 2:
        return None
    start_year = grouped_data.index.min()
    end_year = grouped_data.index.max()
    start_value = grouped_data.loc[start_year]
    end_value = grouped_data.loc[end_year]
    years = end_year - start_year
    if start_value <= 0 or years == 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1

# 5️) Layout

arr_mod_uniques = df['Arrondissement_mod'].unique()

# Création de la figure TCAM avec Paris Centre inclus et tri personnalisé
tcam_list = []
for arr in sorted(df['Arrondissement_mod'].unique(), key=lambda x: (0 if x == "Paris Centre" else x)):
    grouped = df[df['Arrondissement_mod'] == arr].groupby('année')['nombre_arbres'].sum()
    tcam = calculate_tcam(grouped)
    if tcam is not None:
        tcam_list.append({'Arrondissement_mod': arr, 'TCAM': tcam * 100})
    else:
        # Inclure quand même Paris Centre avec TCAM = 0 si données insuffisantes
        if arr == "Paris Centre":
            tcam_list.append({'Arrondissement_mod': arr, 'TCAM': 0.0})

tcam_df = pd.DataFrame(tcam_list)

# Création de la figure TCAM avec Paris Centre inclus et tri personnalisé
tcam_df['Arrondissement_mod'] = tcam_df['Arrondissement_mod'].astype(str)

# Ordre personnalisé : Paris Centre d'abord, puis arrondissements numériques
order = ['Paris Centre'] + sorted([x for x in tcam_df['Arrondissement_mod'].unique() if x.isdigit()], key=int)

fig_tcam = px.bar(
    tcam_df,
    x='Arrondissement_mod',
    y='TCAM',
    title='TCAM par arrondissement (%)',
    labels={'TCAM': 'TCAM (%)'},
    category_orders={'Arrondissement_mod': order}
)
fig_tcam.update_layout(
    xaxis_tickangle=-45,
    xaxis={'tickmode': 'array', 'tickvals': order, 'ticktext': order}
)


# Section introductive
intro_section = html.Div([
    html.H2("Plantations d’arbres à Paris", style={'color': '#2e7d32', 'marginBottom': '10px'}),
    html.P([
        "Bienvenue sur le dashboard dédié aux plantations d’arbres à Paris. ",
        "Ce dashboard permet de visualiser les efforts de végétalisation dans la capitale, ",
        "d’analyser les tendances, de cartographier les projets, de réaliser des prévisions, ",
        "et de comparer les dynamiques entre arrondissements. ",
        "Naviguez via les onglets pour explorer : ",
        html.Br(),
        "1) Vue d’ensemble → 2) Cartographie → 3) Prévisions → 4) Comparaisons."
    ],
        style={'fontSize': '18px', 'color': '#333', 'margin': '0 auto 25px', 'maxWidth': '900px'}
    ),
    html.P(
        "À noter : les 1er, 2e, 3e et 4e arrondissements sont également regroupés sous l'entité « Paris Centre » pour refléter leur fusion administrative.",
        style={'fontSize': '16px', 'color': '#555', 'margin': '0 auto 10px', 'maxWidth': '800px'}
    )
], style={'marginBottom': '30px', 'textAlign': 'center'})

app.layout = html.Div(
    [
        html.Div([
            html.Img(
                src=app.get_asset_url('logo-paris.jpg'),
                style={
                    'height': '65px',
                    'width': 'auto',
                    'objectFit': 'contain',
                    'justifySelf': 'start'
                },
                title="Ville de Paris"
            ),
        html.H1(
            "Dashboard – Plantations d’arbres à Paris",
            style={
                'textAlign': 'center',
                'color': '#2e7d32',
                'margin': '0',
                'fontFamily': 'Roboto, Arial, sans-serif',
                'fontWeight': 600
            }
        ),
        html.Div(style={'width': '80px'})
        ], className="header-modern"),

        # Ajout de la section introductive
        intro_section,

        html.Div(id='kpi-container', style={
            'display': 'flex',
            'justifyContent': 'space-around',
            'margin': '20px 0',
            'textAlign': 'center',
            'gap': '24px'
        }),

        html.Div([
            html.Button("Exporter les données", id="export-button", style={
                'fontFamily': 'Roboto, Arial, sans-serif',
                'background': '#2e7d32',
                'color': '#fff',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 22px',
                'fontWeight': 500,
                'boxShadow': '0 2px 6px 0 rgba(60,72,88,0.07)',
                'cursor': 'pointer'
            }),
            dcc.Download(id="download-dataframe-csv")
        ], style={'textAlign': 'center', 'margin': '10px 0'}),

        dcc.Tabs(
            value='vue-globale',
            style={'margin': '20px'},
            children=[
                dcc.Tab(
                    label='1) Vue d’ensemble',
                    value='vue-globale',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'border': 'none',
                        'borderRadius': '10px 10px 0 0',
                        'marginRight': '5px',
                        'fontFamily': 'Roboto, Arial, sans-serif',
                        'fontWeight': 500,
                        'transition': 'background 0.2s'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'fontWeight': '600',
                        'borderBottom': '3px solid #2e7d32',
                        'color': '#2e7d32',
                        'borderRadius': '10px 10px 0 0',
                        'boxShadow': 'none'
                    },
                    children=[
                        html.Div([
                            html.H3("Contexte général", style={'color': '#2e7d32', 'fontFamily': 'Roboto, Arial, sans-serif', 'fontWeight': 600}),
                            html.P(
                                "Depuis plusieurs années, la Ville de Paris intensifie ses efforts de végétalisation. "
                                "Ce dashboard permet d'explorer l'évolution des plantations d'arbres à travers le temps "
                                "et les arrondissements. Commencez par observer la tendance annuelle globale, puis affinez votre exploration "
                                "grâce aux filtres par année et arrondissement.",
                                style={'color': '#555', 'fontSize': '16px', 'marginBottom': '20px'}
                            )
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px',
                            'marginBottom': '22px'
                        }),
                        html.Div([
                            html.H2("Évolution annuelle", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600,
                                'marginBottom': '12px'
                            }),
                            # Nouveau dropdown arrondissement pour filtrer le bar chart annuel
                            html.Div([
                                dcc.Dropdown(
                                    id='annee-bar-arrondissement-dropdown',
                                    options=[{'label': str(a), 'value': a} for a in sorted(df['Arrondissement_mod'].unique(), key=lambda x: (0 if x=="Paris Centre" else int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else x))],
                                    value=[],
                                    multi=True,
                                    clearable=True,
                                    placeholder="Filtrer par arrondissement",
                                    style={
                                        'backgroundColor': '#fff',
                                        'borderRadius': '8px',
                                        'boxShadow': '0 1px 6px 0 rgba(60,72,88,0.06)',
                                        'marginBottom': '18px'
                                    }
                                )
                            ]),
                            dcc.Graph(id='annee-bar', figure=make_annee_bar([int(max(annees.index))], [])),
                            html.P(
                                "Cette section présente la tendance annuelle globale des plantations d’arbres à Paris. "
                                "Elle sert de point de départ pour filtrer l’ensemble du dashboard selon l’année sélectionnée.",
                                style={'color': '#555', 'fontSize': '16px', 'marginBottom': '20px'}
                            ),
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px',
                            'marginBottom': '22px'
                        }),
                        html.Div([
                            html.H2("Filtrer par année", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600
                            }),
                            dcc.Dropdown(
                                id='year-dropdown',
                                options=annee_options,
                                value=[int(max(annees.index))],
                                multi=True,
                                clearable=False,
                                style={
                                    'backgroundColor': '#fff',
                                    'borderRadius': '8px',
                                    'boxShadow': '0 1px 6px 0 rgba(60,72,88,0.06)',
                                    'marginBottom': '18px'
                                }
                            ),
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px',
                            'marginBottom': '22px'
                        }),
                        html.Div([
                            html.H2("Répartition par arrondissement", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600
                            }),
                            dcc.Graph(id='arrondissement-graph'),
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px'
                        }),
                    ]
                ),
                dcc.Tab(
                    label='2) Cartographie',
                    value='cartes',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'border': 'none',
                        'borderRadius': '10px 10px 0 0',
                        'marginRight': '5px',
                        'fontFamily': 'Roboto, Arial, sans-serif',
                        'fontWeight': 500,
                        'transition': 'background 0.2s'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'fontWeight': '600',
                        'borderBottom': '3px solid #2e7d32',
                        'color': '#2e7d32',
                        'borderRadius': '10px 10px 0 0',
                        'boxShadow': 'none'
                    },
                    children=[
                        html.Div([
                            html.H2("Choroplèthe - Total d’arbres", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600
                            }),
                            dcc.Graph(id='choropleth-total-graph'),
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px',
                            'marginBottom': '22px'
                        }),
                        html.Div([
                            html.H2("Choroplèthe - TCAM (%)", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600
                            }),
                            dcc.Graph(id='choropleth-tcam-graph', figure=make_choropleth_tcam()),
                            dcc.Graph(id='tcam-graph', figure=fig_tcam),
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px'
                        }),
                    ]
                ),
                dcc.Tab(
                    label='3) Prévisions',
                    value='predictions',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'border': 'none',
                        'borderRadius': '10px 10px 0 0',
                        'marginRight': '5px',
                        'fontFamily': 'Roboto, Arial, sans-serif',
                        'fontWeight': 500,
                        'transition': 'background 0.2s'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'fontWeight': '600',
                        'borderBottom': '3px solid #2e7d32',
                        'color': '#2e7d32',
                        'borderRadius': '10px 10px 0 0',
                        'boxShadow': 'none'
                    },
                    children=[
                        html.Div([
                            html.H2("Prévisions globales et par arrondissement", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600
                            }),
                            dcc.RadioItems(
                                id='forecast-view-mode',
                                options=[
                                    {'label': 'Global', 'value': 'global'},
                                    {'label': 'Arrondissements', 'value': 'arr'},
                                    {'label': 'Global + Arrondissements', 'value': 'both'}
                                ],
                                value='both',
                                inline=True,
                                labelStyle={'marginRight': '15px', 'fontWeight': '500'},
                                style={'textAlign': 'center', 'marginBottom': '15px'}
                            ),
                            dcc.Slider(
                                id='year-slider',
                                min=int(min(annees.index)),
                                max=int(max(annees.index)) + 10,
                                step=1,
                                value=int(max(annees.index)) + 1,
                                marks={year: str(year) for year in range(int(min(annees.index)), int(max(annees.index)) + 11, 2)},
                            ),
                            html.Div(id='forecast-text', style={'textAlign': 'center', 'marginTop': '10px'}),
                            dcc.Dropdown(
                                id='forecast-arr-dropdown',
                                options=[{'label': str(a), 'value': a} for a in sorted(df['Arrondissement_mod'].unique(), key=lambda x: (0 if x=="Paris Centre" else int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else x))],
                                value=[],
                                multi=True,
                                placeholder="Sélectionnez un ou plusieurs arrondissements",
                                style={
                                    'backgroundColor': '#fff',
                                    'borderRadius': '8px',
                                    'boxShadow': '0 1px 6px 0 rgba(60,72,88,0.06)',
                                    'marginBottom': '18px'
                                }
                            ),
                            dcc.Graph(id='forecast-combined-graph')
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px',
                            'marginBottom': '22px'
                        }),
                    ]
                ),
                dcc.Tab(
                    label='4) Comparaisons',
                    value='comparaisons',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'border': 'none',
                        'borderRadius': '10px 10px 0 0',
                        'marginRight': '5px',
                        'fontFamily': 'Roboto, Arial, sans-serif',
                        'fontWeight': 500,
                        'transition': 'background 0.2s'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#fff',
                        'fontWeight': '600',
                        'borderBottom': '3px solid #2e7d32',
                        'color': '#2e7d32',
                        'borderRadius': '10px 10px 0 0',
                        'boxShadow': 'none'
                    },
                    children=[
                        html.Div([
                            html.H2("Comparaison entre arrondissements", style={
                                'color': '#2e7d32',
                                'fontFamily': 'Roboto, Arial, sans-serif',
                                'fontWeight': 600
                            }),
                            dcc.Dropdown(
                                id='compare-arrondissements-dropdown',
                                options=[{'label': str(a), 'value': a} for a in sorted(df['Arrondissement_mod'].unique(), key=lambda x: (0 if x=="Paris Centre" else int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else x))],
                                value=[17, 10] if 17 in df['Arrondissement_mod'].unique() and 10 in df['Arrondissement_mod'].unique() else [],
                                multi=True,
                                clearable=True,
                                style={
                                    'backgroundColor': '#fff',
                                    'borderRadius': '8px',
                                    'boxShadow': '0 1px 6px 0 rgba(60,72,88,0.06)',
                                    'marginBottom': '18px'
                                }
                            ),
                            dcc.Graph(id='compare-arrondissements-graph'),
                        ], style={
                            'backgroundColor': '#fff',
                            'boxShadow': '0 2px 8px 0 rgba(60,72,88,0.07)',
                            'borderRadius': '14px',
                            'padding': '18px 24px'
                        }),
                    ]
                ),
            ]
        ),
    ],
    style={
        'backgroundColor': '#f4f6f8',
        'padding': '28px',
        'maxWidth': '1200px',
        'margin': '0 auto'
    }
)

# 6️) Callbacks

# Synchronisation sélection année depuis clic sur le bar chart
@app.callback(
    Output('year-dropdown', 'value'),
    Input('annee-bar', 'clickData'),
    State('year-dropdown', 'value'),
    prevent_initial_call=True
)
def sync_year_from_click(clickData, current):
    if not clickData or 'points' not in clickData:
        raise PreventUpdate
    clicked_year = int(clickData['points'][0]['x'])
    return [clicked_year]

# Mise à jour du bar chart selon la sélection d'années ET du nouveau dropdown arrondissement
@app.callback(
    Output('annee-bar', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('annee-bar-arrondissement-dropdown', 'value')]
)
def update_annee_bar(selected_years, selected_arrondissements):
    return make_annee_bar(selected_years or [], selected_arrondissements or [])


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-button", "n_clicks"),
    State("year-dropdown", "value"),
    State("annee-bar-arrondissement-dropdown", "value"),
    prevent_initial_call=True
)
def export_filtered_data(n_clicks, selected_years, selected_arrondissements):
    dff = df.copy()
    if selected_years:
        dff = dff[dff['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement_mod'].isin(selected_arrondissements)]
    return dcc.send_data_frame(dff.to_csv, "arbres_filtrés.csv", index=False)

@app.callback(
    Output('arrondissement-graph', 'figure'),
    Input('year-dropdown', 'value'),
    Input('annee-bar-arrondissement-dropdown', 'value')
)
def update_arrondissement_graph(selected_years, selected_arrondissements):
    # Fallback : si selected_years est vide ou None, prend toutes les années
    if not selected_years:
        selected_years = unique_years
    return make_arrondissement_figure(selected_years, selected_arrondissements)


# Callback pour la carte choroplèthe totale
@app.callback(
    Output('choropleth-total-graph', 'figure'),
    Input('year-dropdown', 'value')
)
def update_choropleth_total(selected_years):
    # Fallback : si selected_years est vide ou None, prend toutes les années
    if not selected_years:
        selected_years = unique_years
    return make_choropleth_total(selected_years)

@app.callback(
    Output('kpi-container', 'children'),
    Input('year-dropdown', 'value'),
    Input('annee-bar-arrondissement-dropdown', 'value')
)
def update_kpis(selected_years, selected_arrondissements):
    total_arbres, total_projets, total_arrondissements, moyenne_par_projet = compute_kpis(selected_years, selected_arrondissements)
    return [
        html.Div([
            html.H3("Total arbres"), html.P(f"{total_arbres}")
        ]),
        html.Div([
            html.H3("Projets"), html.P(f"{total_projets}")
        ]),
        html.Div([
            html.H3("Arrondissements"), html.P(f"{total_arrondissements}")
        ]),
        html.Div([
            html.H3("Moyenne par projet"), html.P(f"{moyenne_par_projet}")
        ])
    ]


# Nouveau callback combiné pour prévisions globales et par arrondissement
@app.callback(
    [Output('forecast-text', 'children'),
     Output('forecast-combined-graph', 'figure')],
    [Input('year-slider', 'value'),
     Input('forecast-arr-dropdown', 'value'),
     Input('forecast-view-mode', 'value')]
)
def update_forecast_combined(selected_year, selected_arrs, view_mode):
    messages = []
    fig = go.Figure()

    hist_x = np.array([int(x) for x in annees.index])
    hist_y = annees.values
    last_year = int(hist_x.max())

    if view_mode in ('global', 'both'):
        # Historique global
        fig.add_scatter(x=hist_x, y=hist_y, mode='lines+markers', name='Global (historique)', line=dict(color='black'))
        # Prévision globale
        future_years = np.arange(last_year + 1, selected_year + 1)
        if len(future_years) > 0:
            global_preds = model.predict(future_years.reshape(-1, 1)).astype(float)
            global_preds = np.clip(global_preds, 0, None)
            fig.add_scatter(x=future_years, y=global_preds, mode='lines', name='Global (prévision)', line=dict(color='green'))
            messages.append(f"Prévision globale {selected_year} : {int(global_preds[-1])} arbres")

    if view_mode in ('arr', 'both'):
        # Par arrondissement
        for arr in selected_arrs:
            dff = df[df['Arrondissement_mod'] == arr]
            grouped = dff.groupby('année')['nombre_arbres'].sum()
            if len(grouped) < 2:
                continue
            X_local = np.array([int(a) for a in grouped.index]).reshape(-1, 1)
            y_local = grouped.values
            local_model = LinearRegression().fit(X_local, y_local)

            # Historique arr
            fig.add_scatter(x=grouped.index, y=grouped.values, mode='lines+markers', name=f"{arr} (historique)")

            # Prévision arr
            X_future = np.array([[selected_year]])
            y_future = float(local_model.predict(X_future))
            y_future = max(y_future, 0.0)
            future_years_arr = np.arange(last_year + 1, selected_year + 1)
            preds_arr = local_model.predict(future_years_arr.reshape(-1, 1)).astype(float)
            preds_arr = np.clip(preds_arr, 0, None)
            fig.add_scatter(x=future_years_arr, y=preds_arr, mode='lines', name=f"{arr} (prévision)")
            messages.append(f"Arrondissement {arr} – {selected_year} : {int(y_future)} arbres")

    fig.add_vline(x=last_year + 0.5, line_dash='dash', line_color='gray')
    title = "Évolution réelle et prévisions"
    if view_mode == 'global':
        title += " (global)"
    elif view_mode == 'arr':
        title += " (arrondissements)"
    else:
        title += " (global + arrondissements)"
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title="Nombre d'arbres",
        template="plotly_white",
        title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return html.Ul([html.Li(m.replace("prédiction", "prévision")) for m in messages]), fig

@app.callback(
    Output('compare-arrondissements-graph', 'figure'),
    [Input('compare-arrondissements-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def compare_arrondissements(selected_arrondissements, selected_year):
    if not selected_arrondissements:
        return px.line(title="Sélectionnez au moins un arrondissement")

    fig = px.line(title=f"Comparaison entre arrondissements jusqu'en {selected_year}")

    for arr in selected_arrondissements:
        dff = df[df['Arrondissement_mod'] == arr]
        grouped = dff.groupby('année')['nombre_arbres'].sum()

        fig.add_scatter(
            x=grouped.index,
            y=grouped.values,
            mode='lines+markers',
            name=f"{arr} (réel)"
        )

        if len(grouped) >= 2:
            X_local = np.array([int(a) for a in grouped.index]).reshape(-1, 1)
            y_local = grouped.values
            local_model = LinearRegression().fit(X_local, y_local)

            X_future = np.array([[selected_year]])
            y_future = local_model.predict(X_future)[0]

            fig.add_scatter(
                x=[selected_year],
                y=[y_future],
                mode='markers',
                name=f"{arr} (prévision)",
                marker=dict(size=10, symbol='star')
            )

    fig.update_layout(
        xaxis_title="Année",
        yaxis_title="Nombre d'arbres",
        legend_title="Arrondissements",
        template="plotly_white",
        title_font=dict(family="Roboto, Arial, sans-serif", color="#2e7d32", size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


# 7️) Interactivité clic sur les graphiques d'arrondissement/tcam pour synchroniser le nouveau dropdown arrondissement-bar
# Synchroniser le dropdown d'arrondissement sur clic des graphiques d'arrondissement/tcam
@app.callback(
    Output('annee-bar-arrondissement-dropdown', 'value'),
    [
        Input('arrondissement-graph', 'clickData'),
        Input('tcam-graph', 'clickData'),
        State('annee-bar-arrondissement-dropdown', 'value')
    ],
    prevent_initial_call=True
)
def sync_arrondissement_dropdowns(arr_graph_click, tcam_graph_click, arr_dropdown_value):
    # Priorité au dernier graphique cliqué (si les deux sont fournis, ctx.triggered_id permet de savoir lequel)
    triggered = ctx.triggered_id
    clickData = None
    if triggered == 'arrondissement-graph':
        clickData = arr_graph_click
    elif triggered == 'tcam-graph':
        clickData = tcam_graph_click
    else:
        # fallback: priorité au premier non None
        clickData = arr_graph_click or tcam_graph_click

    if not clickData or 'points' not in clickData or not clickData['points']:
        raise PreventUpdate

    # Extraire l'arrondissement cliqué (x de la barre)
    clicked = clickData['points'][0].get('x', None)
    clicked_arr = clicked
    arr_dropdown_new = arr_dropdown_value[:] if arr_dropdown_value else []
    if clicked_arr not in arr_dropdown_new:
        arr_dropdown_new.append(clicked_arr)
    return arr_dropdown_new

if __name__ == "__main__":
    print("Dashboard démarré sur http://localhost:8050")
    import webbrowser
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False)
