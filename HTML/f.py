from flask import Flask, request, jsonify, render_template_string
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import joblib
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
    <h1>Disease Detection</h1>
    <p>
        <a href="/dash/">Go to Dashboard</a>
    </p>
    ''')

# Dash Layout
app.layout = html.Div([
    html.H1('Disease Detection Dashboard'),
    dcc.Tabs([
        dcc.Tab(label='Skin Cancer Detection', children=[
            dcc.Upload(
                id='upload-skin-cancer',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-skin-cancer')
        ]),
        dcc.Tab(label='Malaria Detection', children=[
            dcc.Upload(
                id='upload-malaria',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-malaria')
        ]),
        dcc.Tab(label='COVID-19 Detection', children=[
            html.Div([
                dcc.Input(id='input-fever', type='number', placeholder='Fever', min=0, max=1, step=1),
                dcc.Input(id='input-cough', type='number', placeholder='Cough', min=0, max=1, step=1),
                dcc.Input(id='input-fatigue', type='number', placeholder='Fatigue', min=0, max=1, step=1),
                dcc.Input(id='input-difficulty-breathing', type='number', placeholder='Difficulty Breathing', min=0, max=1, step=1),
                html.Button('Submit', id='submit-covid', n_clicks=0),
                html.Div(id='output-covid')
            ])
        ])
    ])
])

# Callbacks for Dash interactions
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
        return html.Div([html.H5(result)])

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
        return html.Div([html.H5(result)])

if __name__ == '__main__':
    server.run(debug=True)
