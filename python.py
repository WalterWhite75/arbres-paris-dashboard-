# ==============================
# 🌳 Dashboard Arbres Plantés - Python
# ==============================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output
import plotly.express as px

# --- Ajout import & chargement GeoJSON pour choroplèthes
import json
import urllib.request

geojson_url = "https://opendata.paris.fr/explore/dataset/arrondissements/download/?format=geojson&timezone=Europe/Berlin&lang=fr"
with urllib.request.urlopen(geojson_url) as response:
    paris_geojson = json.load(response)

# ==============================
# 1️⃣ Données
# ==============================

file_path = os.path.expanduser('~/Desktop/arbres-plantes-par-projet.xlsx')
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

print("Colonnes :", df.columns.tolist())
print(df.head())

# ==============================
# 2️⃣ Analyse simple
# ==============================

annees = df.groupby('année')['nombre_arbres'].sum()
# plt.figure(figsize=(10, 5))
# annees.plot(kind='bar', color='green')
# plt.title("Nombre d'arbres plantés par année")
# plt.tight_layout()
# plt.show()

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

print(f"📊 RMSE Linéaire : {rmse_linear:.2f}")
print(f"📊 RMSE Polynomial (ordre 2) : {rmse_poly:.2f}")

next_year = int(max(annees.index)) + 1
prediction = model.predict(np.array([[next_year]]))
print(f"🌱 Prédiction pour {next_year} : {int(prediction[0])} arbres plantés")

# ==============================
# 3️⃣ Dashboard
# ==============================

app = Dash(__name__)

fig_annee = px.bar(
    x=annees.index,
    y=annees.values,
    labels={'x': 'Année', 'y': "Nombre d'arbres"},
    title="🌳 Nombre d'arbres plantés par année"
)

unique_years = sorted([int(a) for a in df['année'].dropna().unique()])
annee_options = [{'label': str(a), 'value': a} for a in unique_years]

# ==============================
# 4️⃣ Fonctions graphiques + KPI
# ==============================

def make_arrondissement_figure(selected_years, selected_arrondissements):
    dff = df[df['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement'].isin(selected_arrondissements)]
    arr_data = dff.groupby('Arrondissement')['nombre_arbres'].sum().sort_values(ascending=False)
    fig = px.bar(
        x=arr_data.index.astype(str),
        y=arr_data.values,
        labels={'x': 'Arrondissement', 'y': "Nombre d'arbres"},
        title=f"🏙️ Répartition par arrondissement ({', '.join(map(str, selected_years))})"
    )
    return fig

df_map = df.dropna(subset=['latitude', 'longitude'])

def make_map_figure(selected_years, selected_arrondissements):
    dff = df_map[df_map['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement'].isin(selected_arrondissements)]
    fig = px.scatter_mapbox(
        dff,
        lat='latitude',
        lon='longitude',
        hover_name='Nom du projet',
        hover_data={'latitude': False, 'longitude': False},
        color='Arrondissement',
        size='nombre_arbres',
        zoom=11,
        height=600
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        title=f"🗺️ Projets ({', '.join(map(str, selected_years))})",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

def compute_kpis(selected_years, selected_arrondissements):
    dff = df[df['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement'].isin(selected_arrondissements)]
    total_arbres = int(dff['nombre_arbres'].sum())
    total_projets = dff.shape[0]
    total_arrondissements = dff['Arrondissement'].nunique()
    moyenne_par_projet = round(dff['nombre_arbres'].mean(), 1) if total_projets > 0 else 0
    return total_arbres, total_projets, total_arrondissements, moyenne_par_projet

# ==============================
# ➕ Cartes choroplèthes
# ==============================
def make_choropleth_total(selected_years):
    dff = df[df['année'].isin(selected_years)]
    grouped = dff.groupby('Arrondissement')['nombre_arbres'].sum().reset_index()
    fig = px.choropleth_mapbox(
        grouped,
        geojson=paris_geojson,
        locations='Arrondissement',
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
        title="🌍 Carte choroplèthe - Total d’arbres plantés par arrondissement",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

def make_choropleth_tcam():
    fig = px.choropleth_mapbox(
        tcam_df,
        geojson=paris_geojson,
        locations='Arrondissement',
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
        title="📈 Carte choroplèthe - TCAM (%) par arrondissement",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

# ==============================
# ➕ Fonction utilitaire TCAM
# ==============================
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

# ==============================
# 5️⃣ Layout
# ==============================

#
# ==============================
# ➕ Création du DataFrame TCAM
# ==============================
tcam_list = []
for arr in sorted(df['Arrondissement'].unique()):
    grouped = df[df['Arrondissement'] == arr].groupby('année')['nombre_arbres'].sum()
    tcam = calculate_tcam(grouped)
    if tcam is not None:
        tcam_list.append({'Arrondissement': arr, 'TCAM': tcam * 100})
tcam_df = pd.DataFrame(tcam_list).sort_values('TCAM', ascending=False)

fig_tcam = px.bar(
    tcam_df,
    x='Arrondissement',
    y='TCAM',
    title='TCAM par arrondissement (%)',
    labels={'TCAM': 'TCAM (%)'},
)
fig_tcam.update_layout(
    xaxis={'tickmode': 'array', 'tickvals': tcam_df['Arrondissement'], 'ticktext': tcam_df['Arrondissement']},
    xaxis_tickangle=-45
)

app.layout = html.Div(
    [
        html.Div([
            html.Img(
                src="https://upload.wikimedia.org/wikipedia/fr/4/4e/Logo_Paris_%282019%29.svg",
                style={'height': '60px', 'marginRight': '20px'}
            ),
            html.H1(
                "Dashboard - Arbres Plantés 🌿",
                style={'flex': '1', 'textAlign': 'center', 'color': '#2e7d32', 'margin': '0'}
            )
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'marginBottom': '30px'
        }),

        html.Div(id='kpi-container', style={
            'display': 'flex',
            'justifyContent': 'space-around',
            'margin': '20px 0',
            'textAlign': 'center'
        }),

        html.Div([
            html.Button("📤 Exporter les données", id="export-button"),
            dcc.Download(id="download-dataframe-csv")
        ], style={'textAlign': 'center', 'margin': '10px 0'}),

        dcc.Tabs(
            value='vue-globale',
            style={"fontFamily": "Arial, sans-serif"},
            parent_style={"margin": "20px"},
            children=[
                dcc.Tab(
                    label='Vue Globale',
                    value='vue-globale',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#f0f0f0',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'marginRight': '5px'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#c8e6c9',
                        'fontWeight': 'bold',
                        'border': '1px solid #4caf50'
                    },
                    children=[
                        html.H2("Évolution annuelle", style={'color': '#2e7d32'}),
                        dcc.Graph(figure=fig_annee),
                        html.H2("Filtrer par arrondissement", style={'color': '#2e7d32'}),
                        dcc.Dropdown(
                            id='arrondissement-dropdown',
                            options=[{'label': str(a), 'value': a} for a in sorted(df['Arrondissement'].unique())],
                            value=[],
                            multi=True,
                            clearable=True
                        ),
                        html.H2("Filtrer par année", style={'color': '#2e7d32'}),
                        dcc.Dropdown(
                            id='year-dropdown',
                            options=annee_options,
                            value=[int(max(annees.index))],
                            multi=True,
                            clearable=False
                        ),
                        html.H2("Répartition par arrondissement", style={'color': '#2e7d32'}),
                        dcc.Graph(id='arrondissement-graph'),
                    ]
                ),
                dcc.Tab(
                    label='Cartes',
                    value='cartes',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#f0f0f0',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'marginRight': '5px'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#c8e6c9',
                        'fontWeight': 'bold',
                        'border': '1px solid #4caf50'
                    },
                    children=[
                        html.H2("Carte des projets", style={'color': '#2e7d32'}),
                        dcc.Graph(id='map-graph'),
                        html.H2("Choroplèthe - Total d’arbres", style={'color': '#2e7d32'}),
                        dcc.Graph(id='choropleth-total-graph'),
                        html.H2("Choroplèthe - TCAM (%)", style={'color': '#2e7d32'}),
                        dcc.Graph(id='choropleth-tcam-graph', figure=make_choropleth_tcam()),
                        dcc.Graph(id='tcam-graph', figure=fig_tcam),
                    ]
                ),
                dcc.Tab(
                    label='Prédictions',
                    value='predictions',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#f0f0f0',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'marginRight': '5px'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#c8e6c9',
                        'fontWeight': 'bold',
                        'border': '1px solid #4caf50'
                    },
                    children=[
                        html.H2("Prédiction", style={'color': '#2e7d32'}),
                        html.P(f"🌿 Nombre d'arbres prévu en {next_year} : {int(prediction[0])}"),
                        html.H2("Prédictions interactives", style={'color': '#2e7d32'}),
                        dcc.Slider(
                            id='year-slider',
                            min=int(min(annees.index)),
                            max=int(max(annees.index)) + 10,
                            step=1,
                            value=int(max(annees.index)) + 1,
                            marks={year: str(year) for year in range(int(min(annees.index)), int(max(annees.index)) + 11, 2)},
                        ),
                        html.Div(id='prediction-output', style={'textAlign': 'center', 'marginTop': '10px'}),
                        html.P(
                            f"Formule globale : ŷ = {model.coef_[0]:.2f} × Année + {model.intercept_:.2f}",
                            style={'textAlign': 'center', 'fontStyle': 'italic', 'color': 'gray'}
                        ),
                        dcc.Graph(id='prediction-graph'),
                        html.H3("Choisir un arrondissement pour la prédiction", style={'color': '#2e7d32'}),
                        dcc.Dropdown(
                            id='arrondissement-pred-dropdown',
                            options=[{'label': str(a), 'value': a} for a in sorted(df['Arrondissement'].unique())],
                            value=None,
                            placeholder="Sélectionner un arrondissement",
                            clearable=True
                        ),
                        html.Div(id='arrondissement-pred-output', style={'textAlign': 'center', 'marginTop': '10px'}),
                        dcc.Graph(id='arrondissement-pred-graph'),
                    ]
                ),
                dcc.Tab(
                    label='Comparaisons',
                    value='comparaisons',
                    style={
                        'padding': '10px',
                        'backgroundColor': '#f0f0f0',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'marginRight': '5px'
                    },
                    selected_style={
                        'padding': '10px',
                        'backgroundColor': '#c8e6c9',
                        'fontWeight': 'bold',
                        'border': '1px solid #4caf50'
                    },
                    children=[
                        html.H2("Comparaison entre arrondissements", style={'color': '#2e7d32'}),
                        dcc.Dropdown(
                            id='compare-arrondissements-dropdown',
                            options=[{'label': str(a), 'value': a} for a in sorted(df['Arrondissement'].unique())],
                            value=[17, 10],
                            multi=True,
                            clearable=True
                        ),
                        dcc.Graph(id='compare-arrondissements-graph'),
                    ]
                ),
            ]
        ),
    ],
    style={
        'backgroundColor': '#f9f9f9',
        'padding': '20px'
    }
)

# ==============================
# 6️⃣ Callbacks
# ==============================

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-button", "n_clicks"),
    Input("year-dropdown", "value"),
    Input("arrondissement-dropdown", "value"),
    prevent_initial_call=True
)
def export_filtered_data(n_clicks, selected_years, selected_arrondissements):
    if n_clicks is None:
        return None
    dff = df[df['année'].isin(selected_years)]
    if selected_arrondissements:
        dff = dff[dff['Arrondissement'].isin(selected_arrondissements)]
    return dcc.send_data_frame(dff.to_csv, "arbres_filtrés.csv", index=False)

@app.callback(
    Output('arrondissement-graph', 'figure'),
    Input('year-dropdown', 'value'),
    Input('arrondissement-dropdown', 'value')
)
def update_arrondissement_graph(selected_years, selected_arrondissements):
    return make_arrondissement_figure(selected_years, selected_arrondissements)

@app.callback(
    Output('map-graph', 'figure'),
    Input('year-dropdown', 'value'),
    Input('arrondissement-dropdown', 'value')
)
def update_map(selected_years, selected_arrondissements):
    return make_map_figure(selected_years, selected_arrondissements)

# Callback pour la carte choroplèthe totale
@app.callback(
    Output('choropleth-total-graph', 'figure'),
    Input('year-dropdown', 'value')
)
def update_choropleth_total(selected_years):
    return make_choropleth_total(selected_years)

@app.callback(
    Output('kpi-container', 'children'),
    Input('year-dropdown', 'value'),
    Input('arrondissement-dropdown', 'value')
)
def update_kpis(selected_years, selected_arrondissements):
    total_arbres, total_projets, total_arrondissements, moyenne_par_projet = compute_kpis(selected_years, selected_arrondissements)
    return [
        html.Div([html.H3("🌳 Total arbres", style={'color': '#2e7d32'}), html.P(f"{total_arbres}")]),
        html.Div([html.H3("📌 Projets", style={'color': '#2e7d32'}), html.P(f"{total_projets}")]),
        html.Div([html.H3("🏙️ Arrondissements", style={'color': '#2e7d32'}), html.P(f"{total_arrondissements}")]),
        html.Div([html.H3("🌿 Moyenne par projet", style={'color': '#2e7d32'}), html.P(f"{moyenne_par_projet}")])
    ]

@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-graph', 'figure')],
    Input('year-slider', 'value')
)
def update_prediction(selected_year):
    # Prédiction linéaire
    X_future = np.array([[selected_year]])
    y_future_linear = model.predict(X_future)[0]

    # Prédiction polynomiale
    X_future_poly = poly.transform(X_future)
    y_future_poly = poly_model.predict(X_future_poly)[0]

    # Texte affiché (avec composants Dash, pas de HTML brut)
    content = html.Div([
        html.Span(f"🌿 Année {selected_year} → Linéaire : {int(y_future_linear)} arbres | "),
        html.I("Polynomiale (ordre 2)"),
        html.Span(f" : {int(y_future_poly)} arbres"),
        html.Br(),
        html.Span(
            f"RMSE Linéaire = {rmse_linear:.2f} | RMSE Polynomiale = {rmse_poly:.2f}",
            style={'fontStyle': 'italic', 'color': 'gray'}
        )
    ])

    # Courbes
    years_range = np.arange(int(min(annees.index)), selected_year + 1)
    linear_preds = model.predict(years_range.reshape(-1, 1))
    poly_preds = poly_model.predict(poly.transform(years_range.reshape(-1, 1)))

    fig = px.line(
        x=years_range,
        y=linear_preds,
        labels={'x': 'Année', 'y': "Nombre d'arbres"},
        title="📈 Évolution réelle et prédite (linéaire vs polynomial)"
    )
    fig.add_scatter(x=years_range, y=poly_preds, mode='lines', name='Polynomial (ordre 2)', line=dict(color='blue'))
    fig.add_scatter(x=[selected_year], y=[y_future_linear], mode='markers', name='Prévision Linéaire', marker=dict(color='green', size=10))
    fig.add_scatter(x=[selected_year], y=[y_future_poly], mode='markers', name='Prévision Polynomial', marker=dict(color='red', size=10))

    return content, fig

@app.callback(
    [Output('arrondissement-pred-output', 'children'),
     Output('arrondissement-pred-graph', 'figure')],
    [Input('year-slider', 'value'),
     Input('arrondissement-pred-dropdown', 'value')]
)
def update_arrondissement_prediction(selected_year, selected_arrondissement):
    if selected_arrondissement is None:
        return "Sélectionnez un arrondissement pour voir la prédiction.", px.line()

    dff = df[df['Arrondissement'] == selected_arrondissement]
    grouped = dff.groupby('année')['nombre_arbres'].sum()

    if len(grouped) < 2:
        return f"Pas assez de données historiques pour l'arrondissement {selected_arrondissement}.", px.line()

    X_local = np.array([int(a) for a in grouped.index]).reshape(-1, 1)
    y_local = grouped.values
    local_model = LinearRegression().fit(X_local, y_local)
    a_local = local_model.coef_[0]
    b_local = local_model.intercept_

    X_future = np.array([[selected_year]])
    y_future = local_model.predict(X_future)[0]

    text = html.Span([
        f"🌿 Prédiction pour l'arrondissement {selected_arrondissement} en {selected_year} : {int(y_future)} arbres",
        html.Br(),
        html.Span(
            f"Formule locale : ŷ = {a_local:.2f} × Année + {b_local:.2f}",
            style={'fontStyle': 'italic', 'color': 'gray'}
        )
    ])

    future_years = list(grouped.index) + [selected_year]
    future_values = list(grouped.values) + [y_future]

    fig = px.line(
        x=future_years,
        y=future_values,
        labels={'x': 'Année', 'y': "Nombre d'arbres"},
        title=f"Évolution réelle et prédite - Arrondissement {selected_arrondissement}"
    )
    fig.add_scatter(x=[selected_year], y=[y_future], mode='markers', name='Prévision', marker=dict(color='red', size=10))

    return text, fig

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
        dff = df[df['Arrondissement'] == arr]
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
        template="plotly_white"
    )
    return fig

# ==============================
# 7️⃣ Lancement
# ==============================

if __name__ == "__main__":
    print("🚀 Dashboard → http://localhost:8050")
    import webbrowser
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False)