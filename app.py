import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd 
import numpy as np
import scipy.stats as stats
from dash.exceptions import PreventUpdate
import hashlib # Pour le hachage des mots de passe
from datetime import datetime 
from bs4 import BeautifulSoup # pour le web scraping(ça va nous permettre de récupérer les données des sites web des fintechs)
import requests
import os

# pour la partie machine learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# Fonction d'envoi d'email avec SMTP
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
 

# Définir le thème et les couleurs
THEME = dbc.themes.FLATLY
PRIMARY_COLOR = "#007bff"
SECONDARY_COLOR = "#6c757d"
SUCCESS_COLOR = "#28a745"
INFO_COLOR = "#17a2b8"
WARNING_COLOR = "#ffc107"
DANGER_COLOR = "#dc3545"

# Séquence de couleurs pour les graphiques
COLORS = px.colors.qualitative.Plotly

# Charger les données
data = pd.read_csv("data/BON_BON.csv", sep=";", header=0)
data.dropna(inplace=True)

# Initialiser l'application Dash
app = dash.Dash(__name__, 
                external_stylesheets=[THEME],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                suppress_callback_exceptions=True)

app.title = "Suivi des cotisations des retraités au GABON"

# ======================================================================
# SECTION AUTHENTIFICATION
# ======================================================================

# Fonction de hachage des mots de passe
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Utilisateurs autorisés (en production, utilisez une base de données)
USERS = {
    'admin': {
        'password_hash': hash_password('admin123'),
        'role': 'admin',
        'email':'depoiquetfrancklin@gmail.com'
    },
    'analyste': {
        'password_hash': hash_password('analyste123'),
        'role': 'analyste',
        'email':'massala-1959760@bem.sn'
    }
}

# Layout de connexion
login_layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src="https://upload.wikimedia.org/wikipedia/commons/0/04/Flag_of_Gabon.svg",
                    style={'height': '100px', 'margin-top': '1em','width': '100px', 'border-radius': '70%'}),
            html.H1("Bienvenue", className="text-center mb-4", style={'color': INFO_COLOR}),
            html.H2("Connexion au Dashboard", className="text-center mb-4", style={'color': INFO_COLOR}),
            html.P("Veuillez entrer vos identifiants pour accéder au tableau de bord.", 
                   className="text-center mb-4", style={'color': "Black"}),
            dbc.Input(id='login-username', type='text', placeholder="Nom d'utilisateur", 
                     className="mb-3"),
            dbc.Input(id='login-password', type='password', placeholder="Mot de passe", 
                     className="mb-3"),
            dbc.Button("Se connecter", id='login-button', color="primary", className="w-100 mb-3"),
            html.Div(id='login-alert')
    ], className="border rounded", style={
    'backgroundColor': 'rgba(255, 255, 255, 0.9)',
    'padding': '2rem',
    'boxShadow': '0 0 20px rgba(0, 0, 0, 0.2)',
    'borderRadius': '15px',
    'backdropFilter': 'blur(5px)'})
    ], className="row justify-content-center mt-5" ,style={'width': '35%'}),
      
], className=" bg-light", style={'height': '100vh','width': '100vw','display': 'flex', 'align-items': 'center', 'justify-content': 'center','backgroundImage': 'url("/assets/CNSS_Tempete.jpg")','backgroundSize': 'cover','backgroundRepeat': 'no-repeat','backgroundPosition': 'center'})

###################################
#Layout /otp pour saisir le code=========================
###################################
otp_layout = html.Div([
    html.H2("Vérification 2FA", className="text-center mt-5"),
    html.P("Veuillez entrer le code de vérification reçu par email.", className="text-center"),
    dbc.Input(id='otp-input', placeholder="Code reçu", type='text', className="mb-3"),
    dbc.Button("Vérifier", id='verify-otp', color="primary", className="mb-3"),
    html.Div(id='otp-alert')
], className="container", style={'maxWidth': '400px'})


# ======================================================================
# FONCTIONS UTILITAIRES ET VARIABLES GLOBALES
# ======================================================================

# Calcul des indicateurs pour le tableau de bord
total_cotisations = data['Montant_Paye'].sum()
nombre_entreprises = data['RAISON_SOCIALE'].nunique()
moyenne_cotisation = data['Montant_Paye'].mean()
nombre_periodes = data['PERIODE'].nunique()

# Variables globales pour stocker les modèles
global_cluster_model = None
global_classification_model = None
global_label_encoder = None
global_anomalies = None

# Fonction pour envoyer un email
def send_otp_email(receiver_email, code):
    sender_email = "depoiquetfrancklin@gmail.com"
    sender_password = "vuol czxk fsxh ntvs"  # Remplacez par votre mot de passe
    
    msg = MIMEText(f"Votre code de vérification est : {code}")
    msg['Subject'] = "Code de Vérification - Connexion Dashboard CNSS"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email envoyé avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email: {e}")


# Fonctions pour le machine learning
def prepare_clustering_data(df):
    features = df[['Masse_Salariale_Trimestrielle', 'Principal', 'Montant_Paye', 
                  'Penalite_Taxation', 'Penalite_Majoration']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def find_optimal_clusters(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k

def perform_clustering(df):
    cluster_data = prepare_clustering_data(df)
    optimal_k = find_optimal_clusters(cluster_data)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(cluster_data)
    df['Cluster'] = clusters
    
    # Détection d'anomalies
    clf = IsolationForest(random_state=42)
    df['anomalie'] = clf.fit_predict(df[['Masse_Salariale_Trimestrielle','Montant_Paye']])
    df['risque_retard'] = (df['Penalite_Majoration'] / df['Principal']).apply(
        lambda x: 'Élevé' if x > 0.2 else 'Modéré' if x > 0.1 else 'Faible'
    )
    
    return df, kmeans

def prepare_classification_data(df):
    le = LabelEncoder()
    df['MODE_DE_PAIEMENT_encoded'] = le.fit_transform(df['MODE_DE_PAIEMENT'])
    X = df[['Masse_Salariale_Trimestrielle', 'Principal', 'Montant_Paye', 
           'Penalite_Taxation', 'Penalite_Majoration', 'Cluster']]
    y = df['MODE_DE_PAIEMENT_encoded']
    return X, y, le

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X, y, cv=5)
        results[name] = {
            'model': model,
            'cv_accuracy': np.mean(scores),
            'cv_std': np.std(scores)
        }
    best_model_name = max(results, key=lambda x: results[x]['cv_accuracy'])
    best_model = results[best_model_name]['model']
    return best_model, results

def generate_html_report(content, filename):
    """Génère un rapport HTML à la place du PDF"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
        return False

def get_fintech_data():
    try:
        # Exemple simplifié - en pratique, utiliser des APIs ou web scraping
        fintechs = [
            {"nom": "MobilePay", "frais": "1%", "delai": "24h", "lien": "https://www.mobilepay.com"},
            {"nom": "QuickTransfer", "frais": "0.8%", "delai": "48h", "lien": "https://www.quicktransfer.com"},
            {"nom": "EcoBank", "frais": "0.5%", "delai": "72h", "lien":  "https://www.ecobank.com"}
        ]
        df = pd.DataFrame(fintechs)
        # Option: créer des liens cliquables
        df['lien'] = df.apply(lambda x: f'<a href="{x["lien"]}">Site web</a>', axis=1)

        return df
    except Execption as e:
        print(f"Erreur lors de la récupération des données FinTech: {e}")
        # Retourner un DataFrame vide en cas d'erreur
        return pd.DataFrame()

# ======================================================================
# LAYOUT PRINCIPAL AMELIORE
# ======================================================================

def create_main_layout():
    return html.Div([
        dcc.Interval(id='interval-component', interval=60*60*1000, n_intervals=0),  # Actualisation horaire
        dcc.Store(id='clustered-data'),
        
        # Bannière améliorée
        html.Div([
            html.Div([
                html.H2("Dispositif de Suivi des cotisations des retraités au GABON", 
                        style={'color': 'white'}),
                html.Div([
                    html.Span("Dernière mise à jour: ", style={'color': 'white'}),
                    html.Span(id='last-update', style={'color': 'white', 'font-weight': 'bold'})
                ]),
                dcc.Interval(id='update-interval', interval=300000)  # 5 min
            ], className="col-md-8"),
            html.Div([
                dbc.Button("Déconnexion", id='logout-button', color="danger", className="me-2"),
                dbc.Button("Exporter Rapport", id='generate-report', color="info", className="me-2"),
                html.Img(src="https://upload.wikimedia.org/wikipedia/commons/0/04/Flag_of_Gabon.svg", 
                        style={'height': '50px', 'float': 'right'})
            ], className="col-md-4"),
        ], className="row bg-primary p-3 mb-4"),
        
        # Nouvel onglet pour la vue opérationnelle
        dbc.Tabs([
            dbc.Tab([
                # Tableau de bord interactif
                html.Div([
                    html.H3("Tableau de Bord Opérationnel", className="text-primary mb-3"),
                    
                    # Alertes et indicateurs critiques
                    html.Div(id='alert-container'),
                    
                    # Cartes indicateurs améliorées
                    html.Div([
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Total des Cotisations", className="text-center"),
                                dbc.CardBody([
                                    html.H4(f"{total_cotisations:,.0f} FCFA", 
                                            className="card-title text-center text-success"),
                                    html.P(id='cotisation-trend', className="text-center")
                                ])
                            ], className="shadow-sm h-100")
                        ], className="col-md-3 mb-3"),
                        
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Entreprises à risque", className="text-center"),
                                dbc.CardBody([
                                    html.H4(id='risk-companies', className="card-title text-center text-danger"),
                                    html.P("Retards de paiement", className="text-center")
                                ])
                            ], className="shadow-sm h-100")
                        ], className="col-md-3 mb-3"),
                        
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Anomalies détectées", className="text-center"),
                                dbc.CardBody([
                                    html.H4(id='anomalies-count', className="card-title text-center text-warning"),
                                    html.P("Transactions suspectes", className="text-center")
                                ])
                            ], className="shadow-sm h-100")
                        ], className="col-md-3 mb-3"),
                        
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Solutions FinTech", className="text-center"),
                                dbc.CardBody([
                                    html.H4(id='fintech-count', className="card-title text-center text-info"),
                                    html.P("Alternatives bancaires", className="text-center")
                                ])
                            ], className="shadow-sm h-100")
                        ], className="col-md-3 mb-3"),
                    ], className="row mb-4"),
                    
                    # Graphiques temps réel
                    html.Div([
                        html.Div([
                            dcc.Graph(id='live-cotisations'),
                            html.Div([
                                dbc.Button("Actualiser", id='refresh-btn', color="primary", className="me-2"),
                                dbc.Button("Exporter Données", id='export-data', color="secondary")
                            ], className="mt-2 text-right")
                        ], className="col-md-12 mb-4"),
                        
                        # Nouvelle section: Analyse des modes de paiement
                        html.Div([
                            html.H4("Analyse des Modes de Paiement", className="mt-4"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='payment-mode-filter',
                                        options=[
                                            {'label': 'Tous les modes', 'value': 'all'},
                                            {'label': 'Virement', 'value': 'Virement'},
                                            {'label': 'Chèque', 'value': 'Chèque'},
                                            {'label': 'Espèces', 'value': 'Espèces'},
                                            {'label': 'Prélèvement', 'value': 'Prélèvement'}
                                        ],
                                        value='all',
                                        clearable=False,
                                        className="mb-3"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dcc.RangeSlider(
                                        id='amount-range-slider',
                                        min=0,
                                        max=data['Montant_Paye'].max(),
                                        step=1000000,
                                        value=[0, data['Montant_Paye'].max()],
                                        marks={i: f"{i/1000000:.0f}M" for i in range(0, int(data['Montant_Paye'].max())+1, 5000000)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=8)
                            ]),
                            dcc.Graph(id='payment-mode-chart')
                        ], className="mt-3 p-3 border rounded"),
                        
                        # Tableau des transactions
                        html.Div([
                            html.H4("Détail des Transactions", className="mt-4"),
                            dash.dash_table.DataTable(
                                id='transactions-table',
                                columns=[{"name": i, "id": i} for i in data[['RAISON_SOCIALE', 'PERIODE', 'MODE_DE_PAIEMENT', 'Montant_Paye']].columns],
                                data=data.to_dict('records'),
                                filter_action="native",
                                sort_action="native",
                                page_size=10,
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                }
                            )
                        ], className="mb-4")
                    ], className="row")
                ], className="container mb-5"),
            ], label="Vue Opérationnelle"),
            
            # ... (le reste de vos onglets existants reste inchangé)
            dbc.Tab([
                    dbc.Tab([
    html.Div([
        # Partie 1: Analyse Univariée
        html.Div([
            html.H3("Analyse Univariée", className="text-primary mb-3"),
            
            # Distribution des variables numériques
            html.Div([
                html.H4("Étude de Distribution", className="text-secondary mb-2"),
                
                # Sélecteurs
                html.Div([
                    html.Label("Sélectionner une variable:"),
                    dcc.Dropdown(
                        id='dist-variable',
                        options=[
                            {'label': 'Masse_Salariale_Trimestrielle', 'value':'Masse_Salariale_Trimestrielle'},
                            {'label': 'Principal', 'value': 'Principal'},
                            {'label': 'Montant_Paye', 'value': 'Montant_Paye'}
                        ],
                        value='Montant_Paye',
                        className="mb-2"
                    ),
                ], className="col-md-4"),
                
                # Graphique de distribution
                html.Div([
                    dcc.Graph(id='distribution-graph'),
                    html.Div([
                        html.H5("Commentaire", className="text-secondary"),
                        html.P(id='default-distribution-comment', children="La distribution montre la répartition des valeurs et leur densité de probabilité."),
                        dcc.Textarea(
                            id='comment-distribution-graph',
                            placeholder='Ajoutez vos notes ici...',
                            style={'width': '100%', 'height': 100},
                        )
                    ], className="p-3 border rounded mt-2")
                ], className="col-md-12 mb-4"),
            ], className="mb-4"),
            
            # Répartition des variables catégorielles
            html.Div([
                html.H4("Étude de Répartition", className="text-secondary mb-2"),
               
               # Sélecteurs
                html.Div([
                    html.Label("Sélectionner une variable:"),
                    dcc.Dropdown(
                        id='rep-variable',
                        options=[
                             {'label': 'LIBELLE', 'value': 'LIBELLE'},
                             {'label': 'MODE_DE_PAIEMENT', 'value': 'MODE_DE_PAIEMENT'},
                             {'label': 'PERIODE', 'value': 'PERIODE'},
                             {'label': 'RAISON_SOCIALE', 'value': 'RAISON_SOCIALE'}
                        ],
                        value='PERIODE',
                        className="mb-2"
                    ),
                ], className="col-md-4"),
                
                # Graphique de répartition
                html.Div([
                    dcc.Graph(id='repartition-graph'),
                    html.Div([
                        html.H5("Commentaire", className="text-secondary"),
                        html.P(id='default-repartition-comment', children="Ce diagramme circulaire montre la proportion des différentes catégories."),
                        dcc.Textarea(
                            id='comment-repartition-graph',
                            placeholder='Ajoutez vos notes ici...',
                            style={'width': '100%', 'height': 100},
                        )
                    ], className="p-3 border rounded mt-2")
                ], className="col-md-12 mb-4"),
            ], className="mb-4"),
        ], className="container mb-5"),
        
        # Partie 2: Analyse Bivariée
        html.Div([
            html.H3("Analyse Bivariée", className="text-primary mb-3"),
            
            # Diagrammes en barres
            html.Div([
                html.H4("Diagrammes en Barres", className="text-secondary mb-2"),
              
                # Sélecteurs
                html.Div([
                    html.Div([
                        html.Label("Grouper par:"),
                        dcc.Dropdown(
                            id='group-variable',
                            options=[
                             {'label': 'LIBELLE', 'value': 'LIBELLE'},
                             {'label': 'MODE_DE_PAIEMENT', 'value': 'MODE_DE_PAIEMENT'},
                             {'label': 'PERIODE', 'value': 'PERIODE'},
                             {'label': 'RAISON_SOCIALE', 'value': 'RAISON_SOCIALE'}
                            ],
                            value='RAISON_SOCIALE',
                            className="mb-2"
                        ),
                    ], className="col-md-4"),
                    
                    html.Div([
                        html.Label("Sélectionner une variable numérique:"),
                        dcc.Dropdown(
                            id='bivariate-variable',
                            options=[
                                {'label': 'Masse_Salariale_Trimestrielle', 'value': 'Masse_Salariale_Trimestrielle'},
                                {'label': 'Principal', 'value': 'Principal'},
                                {'label': 'Montant_Paye', 'value': 'Montant_Paye'},
                                {'label': 'Penalite_Taxation', 'value': 'Penalite_Taxation'},
                                {'label': 'Penalite_Majoration', 'value': 'Penalite_Majoration'}
                            ],
                            value='Montant_Paye',
                            className="mb-2"
                        ),
                    ], className="col-md-4"),
                ], className="row"),
                
                # Graphique en barres
                html.Div([
                    dcc.Graph(id='bar-graph'),
                    html.Div([
                        html.H5("Commentaire", className="text-secondary"),
                        html.P(id='default-bar-comment', children="Ce diagramme en barres montre la comparaison des valeurs médianes entre différentes catégories."),
                        dcc.Textarea(
                            id='comment-bar-graph',
                            placeholder='Ajoutez vos notes ici...',
                            style={'width': '100%', 'height': 100},
                        )
                    ], className="p-3 border rounded mt-2")
                ], className="mb-4"),
            ], className="mb-4"),
            
            # Nuage de points
            html.Div([
                html.H4("Nuage de Points", className="text-secondary mb-2"),
                # Sélecteurs
                html.Div([
                    html.Div([
                        html.Label("Variable X:"),
                        dcc.Dropdown(
                            id='x-variable',
                            options=[
                                {'label': 'Masse_Salariale_Trimestrielle', 'value': 'Masse_Salariale_Trimestrielle'},
                                {'label': 'Principal', 'value': 'Principal'},
                                {'label': 'Montant_Paye', 'value': 'Montant_Paye'},
                                {'label': 'Penalite_Taxation', 'value': 'Penalite_Taxation'},
                                {'label': 'Penalite_Majoration', 'value': 'Penalite_Majoration'}
                            ],
                            value='Principal',
                            className="mb-2"
                        ),
                    ], className="col-md-4"),
                    
                    html.Div([
                        html.Label("Variable Y:"),
                        dcc.Dropdown(
                            id='y-variable',
                            options=[
                                 {'label': 'Montant_Paye', 'value': 'Montant_Paye'},
                                 {'label': 'Penalite_Majoration', 'value': 'Penalite_Majoration'},
                                 {'label': 'Penalite_Taxation', 'value': 'Penalite_Taxation'},
                                 {'label': 'Masse_Salariale_Trimestrielle', 'value': 'Masse_Salariale_Trimestrielle'},
                                 {'label': 'Principal', 'value': 'Principal'}
                            ],
                            value='Montant_Paye',
                            className="mb-2"
                        ),
                    ], className="col-md-4"),
                    
                    html.Div([
                        html.Label("Couleur:"),
                        dcc.Dropdown(
                            id='color-variable',
                            options=[
                             {'label': 'LIBELLE', 'value': 'LIBELLE'},
                             {'label': 'MODE_DE_PAIEMENT', 'value': 'MODE_DE_PAIEMENT'},
                             {'label': 'PERIODE', 'value': 'PERIODE'},
                             {'label': 'RAISON_SOCIALE', 'value': 'RAISON_SOCIALE'}
                            ],
                            value='PERIODE',
                            className="mb-2"
                        ),
                    ], className="col-md-4"),
                ], className="row"),
                
                # Graphique de nuage de points
                html.Div([
                    dcc.Graph(id='scatter-graph'),
                    html.Div([
                        html.H5("Commentaire", className="text-secondary"),
                        html.P(id='default-scatter-comment', children="Ce nuage de points montre la relation entre deux variables."),
                        dcc.Textarea(
                            id='comment-scatter-graph',
                            placeholder='Ajoutez vos notes ici...',
                            style={'width': '100%', 'height': 100},
                        )
                    ], className="p-3 border rounded mt-2")
                ], className="mb-4"),
            ], className="mb-4"),
        ], className="container mb-5")
    ], className="container")
], label="Analyse Complète"),
                # ... (contenu existant de l'onglet Analyse Complète)
            ], label="Analyse Complète"),
            
            dbc.Tab([
                # ... (contenu existant de l'onglet Solutions FinTech)
            ], label="Solutions FinTech"),
            
            # Onglet Machine Learning
            # autres onglets...
            dbc.Tab(label="Machine Learning", children=[
                html.Div([
                    html.H3("Outils de Machine Learning pour l'analyse prédictive", className="text-primary mb-3"),

                    html.Div([
                        html.Label("Choisir un algorithme :"),
                        dcc.Dropdown(
                            id='ml-model-selector',
                            options=[
                                {'label': 'Régression Linéaire', 'value': 'linear_regression'},
                                {'label': 'Forêt Aléatoire (Random Forest)', 'value': 'random_forest'},
                                {'label': 'K-Means Clustering', 'value': 'kmeans'},
                                {'label': 'SVM', 'value': 'svm'}
                            ],
                            placeholder="Sélectionnez un modèle de Machine Learning"
                        ),
                    ], className="mb-3"),

                    html.Div([
                        html.Label("Variable cible (à prédire) :"),
                        dcc.Dropdown(
                            id='target-variable',
                            options=[{'label': col, 'value': col} for col in data.select_dtypes(include='number').columns],
                            placeholder="Choisir la variable cible"
                        )
                    ], className="mb-3"),

                    html.Div([
                        html.Label("Variables explicatives :"),
                        dcc.Dropdown(
                            id='feature-variables',
                            options=[{'label': col, 'value': col} for col in data.select_dtypes(include='number').columns],
                            multi=True,
                            placeholder="Choisir les variables explicatives"
                        )
                    ], className="mb-3"),

                    dbc.Button("Lancer l'entraînement", id='train-model-btn', color='primary'),

                    html.Div(id='ml-results', className="mt-4")
                ], className="container")
            
        ], className="mb-4")
    
    ], className="container-fluid bg-white"),
        ########################################
        ###### ONGLET REPORTING FINANCIER #######
        ########################################
        
        dbc.Tab(label="Reporting Financier", children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Filtrer par période :"),
                        dcc.Dropdown(
                            id='periode-filter',
                            options=[{'label': p, 'value': p} for p in sorted(data['PERIODE'].unique())],
                            multi=True
                        ),
                        html.Label("Filtrer par entreprise :"),
                        dcc.Dropdown(
                            id='entreprise-filter',
                            options=[{'label': r, 'value': r} for r in sorted(data['RAISON_SOCIALE'].unique())],
                            multi=True
                        )
                    ], width=4)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='mode-paiement-graph'), width=6),
                    dbc.Col(dcc.Graph(id='top-cotisants-graph'), width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='cotisations-trimestre-graph'), width=12)
                ]),
                html.H4("Entreprises à risque (écart-type/mean > 20%)"),
                html.Ul(id='alertes-entreprises'),
                html.Br(),
                dbc.Button("Télécharger le rapport", id="btn-download-report", color="info"),
                dcc.Download(id="download-report")
                 ], className="container"),
            ], className="p-4"), 
        ######################################
        # Fin de l'onglet Reporting Financier#
        ######################################
        
        # Pied de page
        html.Footer([
            html.P("Développé pour le projet de mémoire de BSID3 - BEM Dakar", 
                   className="text-center text-muted pt-3"),
            html.P(f"Dernière génération: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                   className="text-center text-muted small")
        ], className="container-fluid bg-light mt-5 p-2"),
        
        # Stockages
        dcc.Store(id='comment-store'),
        dcc.Download(id="download-report"),
        dcc.Download(id="download-data")
    ], className="container-fluid bg-white")

# ======================================================================
# NOUVEAUX CALLBACKS POUR LES MODES DE PAIEMENT - CORRIGÉ
# ======================================================================

@app.callback(
    Output('payment-mode-chart', 'figure'),
    [Input('payment-mode-filter', 'value'),
     Input('amount-range-slider', 'value')]
)
def update_payment_mode_chart(mode_filter, amount_range):
    # Vérification des entrées
    if None in [mode_filter, amount_range]:
        raise PreventUpdate
    
    # Copie des données pour éviter les SettingWithCopyWarning
    filtered_data = data.copy()
    
    # Application des filtres
    mask = (filtered_data['Montant_Paye'] >= amount_range[0]) & \
           (filtered_data['Montant_Paye'] <= amount_range[1])
    
    filtered_data = filtered_data[mask]
    
    if mode_filter != 'all':
        filtered_data = filtered_data[filtered_data['MODE_DE_PAIEMENT'] == mode_filter]
    
    # Création des visualisations
    mode_dist = filtered_data.groupby('MODE_DE_PAIEMENT')['Montant_Paye'].sum().reset_index()
    
    fig1 = px.pie(
        mode_dist,
        values='Montant_Paye',
        names='MODE_DE_PAIEMENT',
        title=f"Répartition par mode de paiement ({mode_filter if mode_filter != 'all' else 'Tous modes'})",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig2 = px.box(
        filtered_data,
        x='MODE_DE_PAIEMENT',
        y='Montant_Paye',
        color='MODE_DE_PAIEMENT',
        title="Distribution des montants par mode de paiement",
        log_y=True
    )
    
    # Combinaison des figures
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "box"}]],
        subplot_titles=("Répartition des montants", "Distribution des montants")
    )
    
    fig.add_trace(fig1.data[0], row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
        
    fig.update_layout(
        height=500,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

@app.callback(
    Output('transactions-table', 'data'),
    [Input('payment-mode-filter', 'value'),
     Input('amount-range-slider', 'value')]
)
def update_transactions_table(mode_filter, amount_range):
    # Vérification des entrées
    if None in [mode_filter, amount_range]:
        raise PreventUpdate
    
    # Copie des données pour éviter les SettingWithCopyWarning
    filtered_data = data.copy()
    
    # Application des filtres
    mask = (filtered_data['Montant_Paye'] >= amount_range[0]) & \
           (filtered_data['Montant_Paye'] <= amount_range[1])
    
    filtered_data = filtered_data[mask]
    
    if mode_filter != 'all':
        filtered_data = filtered_data[filtered_data['MODE_DE_PAIEMENT'] == mode_filter]
    
    return filtered_data[['RAISON_SOCIALE', 'PERIODE', 'MODE_DE_PAIEMENT', 'Montant_Paye']].to_dict('records')

# ======================================================================
# CALLBACKS AMELIORES
# ======================================================================
###############callback Machine Learning###############

@callback(
    Output('ml-results', 'children'),
    Input('train-model-btn', 'n_clicks'),
    State('ml-model-selector', 'value'),
    State('target-variable', 'value'),
    State('feature-variables', 'value')
)
def train_model(n_clicks, model_type, target, features):
    if not n_clicks or not model_type or not target or not features:
        return "Veuillez remplir tous les champs pour entraîner un modèle."

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    X = data[features].dropna()
    y = data.loc[X.index, target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVR()
    elif model_type == 'kmeans':
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        return html.Div([
            html.P(f"Cluster Centroids : {model.cluster_centers_}"),
            html.P(f"Inertie du modèle : {model.inertia_:.2f}")
        ])
    else:
        return "Modèle non reconnu."

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return html.Div([
        html.P(f"Score R² : {r2:.2f}"),
        html.P(f"RMSE : {rmse:.2f}")
    ])

@app.callback(
    [Output('last-update', 'children'),
     Output('cotisation-trend', 'children')],
    Input('update-interval', 'n_intervals')
)
def update_time_and_trend(n):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Calculer la tendance (simplifié)
    trend_data = data.groupby('PERIODE')['Montant_Paye'].sum().reset_index()
    if len(trend_data) > 1:
        last = trend_data.iloc[-1]['Montant_Paye']
        prev = trend_data.iloc[-2]['Montant_Paye']
        trend = "↑ Hausse" if last > prev else "↓ Baisse" if last < prev else "→ Stable"
        variation = f"{abs((last-prev)/prev)*100:.1f}%"
        trend_text = f"{trend} ({variation})"
    else:
        trend_text = "Données insuffisantes"
    
    return now, trend_text

@app.callback(
    Output('live-cotisations', 'figure'),
    [Input('refresh-btn', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_live_cotisations(n_clicks, n_intervals):
    # Préparer les données (exemple avec regroupement par période)
    df = data.groupby('PERIODE')['Montant_Paye'].sum().reset_index()
    
    fig = px.line(
        df,
        x='PERIODE',
        y='Montant_Paye',
        title='Évolution des Cotisations en Temps Réel',
        labels={'Montant_Paye': 'Montant (FCFA)', 'PERIODE': 'Période'},
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='Période',
        yaxis_title='Montant des Cotisations (FCFA)'
    )
    
    return fig

@app.callback(
    [Output('risk-companies', 'children'),
     Output('anomalies-count', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_risk_metrics(n):
    if global_cluster_model is not None:
        clustered_data = data.copy()
        clustered_data['Cluster'] = global_cluster_model.predict(prepare_clustering_data(clustered_data))
        risk_count = clustered_data[clustered_data['risque_retard'] == 'Élevé'].shape[0]
        anomaly_count = clustered_data[clustered_data['anomalie'] == -1].shape[0]
        return risk_count, anomaly_count
    return "-", "-"

@app.callback(
    [Output('fintech-count', 'children'),
     Output('fintech-table-container', 'children')],
    Input('refresh-fintech', 'n_clicks')
)
def update_fintech_info(n_clicks):
    fintech_data = get_fintech_data()
    if not fintech_data.empty:
        table = dbc.Table.from_dataframe(
            fintech_data,
            striped=True,
            bordered=True,
            hover=True
        )
        return len(fintech_data), table
    return "0", dbc.Alert("Aucune donnée FinTech disponible", color="warning")

@app.callback(
    Output("download-data", "data"),
    Input("export-data", "n_clicks"),
    prevent_initial_call=True
)
def export_data(n_clicks):
    if n_clicks:
        return dcc.send_data_frame(data.to_csv, "cotisations_export.csv")

@app.callback(
    Output("download-report", "data"),
    Input("generate-report", "n_clicks"),
    prevent_initial_call=True
)
def generate_report(n_clicks):
    if n_clicks:
        # Générer un rapport HTML simplifié
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport des Cotisations</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                h1 {{ color: #007bff; }}
                .stat {{ margin-bottom: 15px; }}
                .stat-label {{ font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Rapport des Cotisations</h1>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>Statistiques Clés</h2>
            <div class="stat">
                <span class="stat-label">Total des cotisations:</span> {total_cotisations:,.0f} FCFA
            </div>
            <div class="stat">
                <span class="stat-label">Nombre d'entreprises:</span> {nombre_entreprises}
            </div>
            <div class="stat">
                <span class="stat-label">Anomalies détectées:</span> {data[data['anomalie'] == -1].shape[0] if 'anomalie' in data.columns else "N/A"}
            </div>
            
            <h2>Top 10 des Entreprises</h2>
            {data.groupby('RAISON_SOCIALE')['Montant_Paye'].sum().nlargest(10).to_frame().to_html()}
        </body>
        </html>
        """
        
        # Sauvegarder temporairement et renvoyer
        filename = f"rapport_cotisations_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        if generate_html_report(report_content, filename):
            return dcc.send_file(filename)
        else:
            raise PreventUpdate

# ======================================================================
# GESTION DE L'APPLICATION
# ======================================================================

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='session', storage_type='session')
])

# Callbacks d'authentification
@app.callback(
    [Output('page-content', 'children'),
     Output('session', 'data')],
    Input('url', 'pathname'),
    State('session', 'data')
)
def display_page(pathname, session_data):
    if session_data and session_data.get('logged_in'):
        return create_main_layout(), session_data
    if not session_data:
        return login_layout, {}

    if session_data.get('otp_sent') and pathname == "/otp":
        return otp_layout, session_data
    
    if session_data.get('logged_in'):
        return create_main_layout(), session_data
    return login_layout, {}

@app.callback(
    [Output('session', 'data', allow_duplicate=True),
     Output('login-alert', 'children'),
     Output('url', 'pathname', allow_duplicate=True)],
    
    Input('login-button', 'n_clicks'),
    
    [State('login-username', 'value'),
     State('login-password', 'value'),
     State('session', 'data')],
    
    prevent_initial_call=True
)
def login(n_clicks, username, password, session_data):
    if n_clicks is None:
        raise PreventUpdate
    
    if not username or not password:
        return dbc.Alert("Veuillez saisir un nom d'utilisateur et un mot de passe", color="danger"), dash.no_update
    
    if username in USERS and USERS[username]['password_hash'] == hash_password(password):
        otp_code = str (random.randint(100000, 999999))
        send_otp_email(USERS[username]['email'], otp_code)
        
        ########################################################################
        #stocker temporairement les infos = OTP et l'utilisateur dans la session
        ########################################################################
        session_data = {
            'otp_sent': True,
            'username': username,
            'otp_code': otp_code,
            'role': USERS[username]['role']
        }
        return session_data,  dbc.Alert("Un code de vérification a été envoyé à votre adresse e-mail. Veuillez le saisir pour continuer.", color="info"),  "/otp"
    else:
        return session_data, dbc.Alert("Nom d'utilisateur ou mot de passe incorrect", color="danger"), dash.no_update

@app.callback(
    [Output('otp-alert', 'children'),
     Output('session', 'data', allow_duplicate=True),
     Output('url', 'pathname')],
    Input('verify-otp', 'n_clicks'),
    [State('otp-input', 'value'),
     State('session', 'data')],
    prevent_initial_call=True
)
def validate_otp(n_clicks, code_entered, session_data):
    if session_data and code_entered == session_data.get('otp_code'):
        session_data['logged_in'] = True
        session_data.pop('otp_code', None)
        session_data.pop('otp_sent', None)
        return "", session_data, "/"
    else:
        return dbc.Alert("Code invalide. Veuillez réessayer.", color="danger"), session_data, "/otp"


@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('session', 'data', allow_duplicate=True)],
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return "/login", {}


# ======================================================================
# CALLBACKS POUR LES GRAPHIQUES D'ANALYSE COMPLETE
# ======================================================================

# Callback pour le graphique de distribution
@app.callback(
    Output('distribution-graph', 'figure'),
    Input('dist-variable', 'value')
)
def update_distribution(variable):
    
    # Nettoyage supplémentaire des données
    clean_data = data[variable].dropna()
    
    # Conversion forcée en numérique
    clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
    
    
    # Créer l'histogramme
    fig = px.histogram(data, 
                     x=variable, 
                     nbins=20, 
                     histnorm='probability density',
                     color_discrete_sequence=[PRIMARY_COLOR])
    
    fig.update_traces(marker=dict(line=dict(width=1, color='white')))
    
    # Calculer la courbe de densité
    kde = stats.gaussian_kde(data[variable].dropna())
    x_range = np.linspace(min(data[variable]), max(data[variable]), 1000)
    y_kde = kde(x_range)
    
    # Ajouter la courbe de densité à l'histogramme
    fig.add_trace(go.Scatter(x=x_range, 
                           y=y_kde, 
                           mode='lines', 
                           name='Densité',
                           line=dict(color=DANGER_COLOR, width=2)))
    
    # Mise en page
    fig.update_layout(
        title=f"Distribution de {variable}",
        xaxis_title=f"{variable}",
        yaxis_title="Densité de probabilité",
        legend_title="Légende",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333')
    )
    
    return fig

# Callback pour le graphique de répartition
@app.callback(
    Output('repartition-graph', 'figure'),
    Input('rep-variable', 'value')
)
def update_repartition(variable):
    # Calculer les comptes
    counts = data[variable].value_counts()
    df_counts = pd.DataFrame({
        'categorie': counts.index,
        'count': counts.values
    })
    
    # Limiter à 10 catégories pour la lisibilité
    if len(df_counts) > 10:
        autres = pd.DataFrame({
            'categorie': ['Autres'],
            'count': [df_counts.iloc[10:]['count'].sum()]
        })
        df_counts = pd.concat([df_counts.iloc[:10], autres])
    
    # Créer le graphique circulaire
    fig = px.pie(df_counts, 
               values='count', 
               names='categorie',
               hole=0.5,
               color_discrete_sequence=COLORS,
               title=f"Répartition des {variable}")
    
    fig.update_traces(textposition='outside', 
                    textinfo='percent+label', 
                    hoverinfo='label+percent+value')
    
    # Mise en page
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333')
    )
    
    return fig

# Callback pour le graphique en barres
@app.callback(
    Output('bar-graph', 'figure'),
    [Input('group-variable', 'value'),
     Input('bivariate-variable', 'value')]
)
def update_bar_graph(group_var, value_var):
    # Agréger les données
    stat_tbl = data.groupby(group_var).agg({value_var: 'median'}).reset_index().sort_values(value_var, ascending=False)
    
    # Limiter à 10 catégories pour la lisibilité
    if len(stat_tbl) > 10:
        stat_tbl = stat_tbl.head(10)
    
    # Créer le graphique en barres
    fig = px.bar(stat_tbl, 
               x=group_var,
               y=value_var, 
               color=group_var,
               text_auto='.2s',
               color_discrete_sequence=COLORS,
               title=f"{value_var.capitalize()} Médian(e) par {group_var}")
    
    # Mise en page
    fig.update_layout(
        xaxis_title=f"{group_var}",
        yaxis_title=f"{value_var} médian(e)",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333')
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

# Callback pour le nuage de points
@app.callback(
    Output('scatter-graph', 'figure'),
    [Input('x-variable', 'value'),
     Input('y-variable', 'value'),
     Input('color-variable', 'value')]
)
def update_scatter_graph(x_var, y_var, color_var):
    # Création du nuage de points
    fig = px.scatter(data, 
                   x=x_var, 
                   y=y_var,
                   color=color_var,
                   size="Masse_Salariale_Trimestrielle" if x_var != "Masse_Salariale_Trimestrielle" and y_var != "Masse_Salariale_Trimestrielle" else None,
                   hover_name="RAISON_SOCIALE",
                   
                   log_x=True if x_var in ["Principal", "Montant_Paye"] else False,
                   log_y=True if y_var in ["Principal", "Montant_Paye"] else False,
                   size_max=30,
                   opacity=0.7,
                   color_discrete_sequence=COLORS,
                   title=f"Relation entre {x_var} et {y_var} selon {color_var}")
    
    # Mise en page
    fig.update_layout(
        xaxis_title=f"{x_var}",
        yaxis_title=f"{y_var}",
        legend_title=f"{color_var}",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333')
    )
    
    return fig

########################################
# Callbacks pour le REporting Financier#
########################################

@callback(
    Output('mode-paiement-graph', 'figure'),
    Input('periode-filter', 'value'),
    Input('entreprise-filter', 'value')
)
def update_mode_paiement(periodes, entreprises):
    df = data.copy()
    if periodes: df = df[df['PERIODE'].isin(periodes)]
    if entreprises: df = df[df['RAISON_SOCIALE'].isin(entreprises)]
    df_sum = df.groupby('MODE_DE_PAIEMENT')['Montant_Paye'].sum().reset_index()
    return px.bar(df_sum, x='MODE_DE_PAIEMENT', y='Montant_Paye', title="Montants par mode de paiement")

@callback(
    Output('cotisations-trimestre-graph', 'figure'),
    Input('periode-filter', 'value'),
    Input('entreprise-filter', 'value')
)
def update_cotisation_trimestre(periodes, entreprises):
    df = data.copy()
    if periodes: df = df[df['PERIODE'].isin(periodes)]
    if entreprises: df = df[df['RAISON_SOCIALE'].isin(entreprises)]
    df_sum = df.groupby('PERIODE')['Montant_Paye'].sum().reset_index()
    return px.line(df_sum, x='PERIODE', y='Montant_Paye', markers=True, title="Évolution des cotisations")

@callback(
    Output('top-cotisants-graph', 'figure'),
    Input('periode-filter', 'value')
)
def update_top_cotisants(periodes):
    df = data.copy()
    if periodes: df = df[df['PERIODE'].isin(periodes)]
    top = df.groupby('RAISON_SOCIALE')['Montant_Paye'].sum().nlargest(5).reset_index()
    return px.bar(top, x='RAISON_SOCIALE', y='Montant_Paye', title="Top 5 Cotisants")

@callback(
    Output('alertes-entreprises', 'children'),
    Input('periode-filter', 'value')
)
def update_alertes(periodes):
    df = data.copy()
    if periodes: df = df[df['PERIODE'].isin(periodes)]
    group = df.groupby('RAISON_SOCIALE')['Montant_Paye']
    ecarts = (group.std() / group.mean()).dropna()
    alerts = ecarts[ecarts > 0.2].index.tolist()
    return [html.Li(e) for e in alerts]

from dash import ctx




#========Fin reporting Financier=======#

####################################
# Callbacks pour le machine learning
####################################
@app.callback(
    [Output('clustering-graph', 'figure'),
     Output('cluster-info', 'children')],
    Input('run-clustering', 'n_clicks'),
    prevent_initial_call=True
)
def run_clustering(n_clicks):
    global global_cluster_model
    
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Exécuter le clustering
    clustered_data, kmeans = perform_clustering(data.copy())
    global_cluster_model = kmeans
    
    # Créer un graphique de visualisation des clusters
    fig = px.scatter(
        clustered_data,
        x='Principal',
        y='Montant_Paye',
        color='Cluster',
        hover_name='RAISON_SOCIALE',
        title='Visualisation des Clusters',
        color_continuous_scale=px.colors.qualitative.Plotly
    )
    
    # Statistiques des clusters
    cluster_stats = clustered_data.groupby('Cluster').agg({
        'Masse_Salariale_Trimestrielle': 'mean',
        'Principal': 'mean',
        'Montant_Paye': 'mean',
        'MODE_DE_PAIEMENT': lambda x: x.mode()[0]
    }).reset_index()
    
    cluster_table = dbc.Table.from_dataframe(
        cluster_stats.round(2),
        striped=True,
        bordered=True,
        hover=True
    )
    
    info = html.Div([
        html.H5(f"{len(cluster_stats)} clusters identifiés", className="mt-3"),
        html.P("Caractéristiques moyennes par cluster:"),
        cluster_table
    ])
    
    return fig, info

@app.callback(
    [Output('model-results', 'children'),
     Output('feature-importance-graph', 'figure')],
    Input('train-model', 'n_clicks'),
    prevent_initial_call=True
)
def train_classification_model(n_clicks):
    global global_classification_model, global_label_encoder, global_cluster_model
    
    if n_clicks is None or global_cluster_model is None:
        raise dash.exceptions.PreventUpdate
    
    # Préparer les données avec les clusters
    clustered_data = data.copy()
    clustered_data['Cluster'] = global_cluster_model.predict(prepare_clustering_data(clustered_data))
    
    # Préparer les données pour la classification
    X, y, le = prepare_classification_data(clustered_data)
    global_label_encoder = le
    
    # Entraîner les modèles
    best_model, results = train_models(X, y)
    global_classification_model = best_model
    
    # Créer un tableau de résultats
    results_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Précision moyenne': [res['cv_accuracy'] for res in results.values()],
        'Écart-type': [res['cv_std'] for res in results.values()]
    })
    
    results_table = dbc.Table.from_dataframe(
        results_df.round(4),
        striped=True,
        bordered=True,
        hover=True
    )
    
    # Visualisation de l'importance des caractéristiques (pour XGBoost)
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        features = X.columns
        fig = px.bar(
            x=features,
            y=importance,
            labels={'x': 'Caractéristique', 'y': 'Importance'},
            title='Importance des Caractéristiques'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="Le modèle sélectionné ne fournit pas d'importance des caractéristiques",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    return results_table, fig

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-masse', 'value'),
     State('input-principal', 'value'),
     State('input-montant', 'value'),
     State('input-taxation', 'value'),
     State('input-majoration', 'value'),
     State('input-cluster', 'value')],
    prevent_initial_call=True
)
def predict_payment_mode(n_clicks, masse, principal, montant, taxation, majoration, cluster):
    global global_classification_model, global_label_encoder
    
    if n_clicks is None or global_classification_model is None:
        raise dash.exceptions.PreventUpdate
    
    # Créer un dataframe avec les valeurs d'entrée
    input_data = pd.DataFrame([[masse, principal, montant, taxation, majoration, cluster]],
                             columns=['Masse_Salariale_Trimestrielle', 'Principal', 
                                     'Montant_Paye', 'Penalite_Taxation', 
                                     'Penalite_Majoration', 'Cluster'])
    
    # Faire la prédiction
    prediction = global_classification_model.predict(input_data)
    proba = global_classification_model.predict_proba(input_data)
    
    # Décoder la prédiction
    predicted_mode = global_label_encoder.inverse_transform(prediction)[0]
    proba_value = np.max(proba) * 100
    
    # Obtenir les probabilités pour tous les modes
    modes = global_label_encoder.classes_
    probas = {mode: f"{prob*100:.1f}%" for mode, prob in zip(modes, proba[0])}
    
    # Créer une liste d'éléments pour afficher les probabilités
    prob_items = [html.Li(f"{mode}: {proba}") for mode, proba in probas.items()]
    
    return html.Div([
        html.H5(f"Mode de paiement prédit: {predicted_mode} ({proba_value:.1f}% de confiance)"),
        html.P("Probabilités pour tous les modes:"),
        html.Ul(prob_items)
    ])

def logout(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return "/login", {}



if __name__ == '__main__':
    app.run(debug=True)