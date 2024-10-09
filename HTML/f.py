from flask import Flask, render_template_string
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
import io

# Create Flask server
server = Flask(__name__)

# Create Dash app
app = Dash(__name__, server=server, url_base_pathname='/dash/')

# Load Models
skin_cancer_model = load_model('models/skin_cancer_model.h5')
malaria_model = load_model('models/malaria_model.h5')

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Flask route for home page
@server.route('/')
def index():
    return render_template_string('''
    <h1 style="text-align: center; color: #4CAF50;">Disease Detection Dashboard</h1>
    <p style="text-align: center;">
        <a href="/dash/" style="font-size: 18px; color: #2196F3;">Go to Dashboard</a>
    </p>
    ''')

# Custom Styles
tab_style = {
    'padding': '20px',
    'fontWeight': 'bold',
    'fontSize': '18px',
    'backgroundColor': '#E0F7FA',
    'color': '#00796B',
    'border': 'none',
}

tab_selected_style = {
    'padding': '20px',
    'fontWeight': 'bold',
    'fontSize': '18px',
    'backgroundColor': '#00796B',
    'color': 'white',
    'borderRadius': '5px',
}

app.layout = html.Div([
    html.H1('Health Dashboard', style={
        'textAlign': 'center',
        'color': '#4CAF50',
        'marginBottom': '50px',
        'fontFamily': 'Arial, sans-serif'
    }),

    dcc.Tabs([
        dcc.Tab(label='Skin Cancer Detection', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                dcc.Upload(
                    id='upload-skin-cancer',
                    children=html.Div([
                        'Drag and Drop or ', html.A('Select a File')
                    ]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '20px', 'backgroundColor': '#F1F8E9'
                    },
                    multiple=False
                ),
                html.Div(id='output-skin-cancer', style={'textAlign': 'center', 'marginTop': '20px'})
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='Malaria Detection', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                dcc.Upload(
                    id='upload-malaria',
                    children=html.Div([
                        'Drag and Drop or ', html.A('Select a File')
                    ]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '20px', 'backgroundColor': '#F1F8E9'
                    },
                    multiple=False
                ),
                html.Div(id='output-malaria', style={'textAlign': 'center', 'marginTop': '20px'})
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='Health Articles', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.H2("Health Articles", style={'textAlign': 'center', 'color': '#009688'}),
                html.Ul([
                    html.Li("Article 1: Understanding Heart Health"),
                    html.Li("Article 2: Benefits of Regular Exercise"),
                    html.Li("Article 3: Nutrition and Well-being"),
                    html.Li("Article 4: Mental Health Awareness"),
                ], style={'textAlign': 'center', 'listStyleType': 'none', 'padding': 0, 'color': '#555'})
            ], style={'padding': '20px', 'backgroundColor': '#E8F5E9', 'borderRadius': '10px'})
        ]),

        dcc.Tab(label='Symptom Checker', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.H2("Symptom Checker", style={'textAlign': 'center', 'color': '#009688'}),
                html.Div([
                    dcc.Input(id='symptom-input', type='text', placeholder='Enter your symptoms',
                              style={'width': '80%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc', 'marginRight': '10px'}),
                    html.Button('Check Symptoms', id='symptom-button', n_clicks=0, style={
                        'backgroundColor': '#009688', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'
                    })
                ], style={'textAlign': 'center', 'marginTop': '20px'}),
                html.Div(id='symptom-result', style={'padding': '20px', 'textAlign': 'center', 'color': '#555'})
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='Health Dashboard', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Graph(id='health-graph'),
            dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
        ])
    ], style={'fontFamily': 'Arial, sans-serif'})
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})

@app.callback(Output('health-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    data = dict(
        x=['2023-01-01', '2023-01-02', '2023-01-03'],  # Placeholder dates
        y=[60, 70, 65],  # Placeholder heart rates
        mode='lines+markers'
    )
    return {
    'data': [data],
    'layout': {
        'title': 'Heart Rate Over Time',
        'xaxis': {'title': 'Date'},
        'yaxis': {'title': 'Heart Rate (bpm)'}
    }
}


@app.callback(Output('output-skin-cancer', 'children'),
              [Input('upload-skin-cancer', 'contents')])
def predict_skin_cancer(contents):
    if contents is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image)
        prediction = skin_cancer_model.predict(processed_image)
        result = 'Cancer Detected' if prediction[0][0] > 0.5 else 'No Cancer'
        return html.H5(result, style={'textAlign': 'center', 'color': 'green' if 'No' in result else 'red'})

@app.callback(Output('output-malaria', 'children'),
              [Input('upload-malaria', 'contents')])
def predict_malaria(contents):
    if contents is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image)
        prediction = malaria_model.predict(processed_image)
        result = 'Malaria Detected' if prediction[0][0] > 0.5 else 'No Malaria'
        return html.H5(result, style={'textAlign': 'center', 'color': 'green' if 'No' in result else 'red'})

@app.callback(Output('symptom-result', 'children'),
              [Input('symptom-button', 'n_clicks')],
              [State('symptom-input', 'value')])
def check_symptoms(n_clicks, symptoms):
    if n_clicks > 0:
        result = "Based on your symptoms, it's advised to consult a healthcare professional."
        return html.H5(result, style={'textAlign': 'center', 'color': 'blue'})

if __name__ == '__main__':
    server.run(debug=True)
