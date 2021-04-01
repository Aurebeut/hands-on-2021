import base64
import yaml
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go


from PIL import Image
from dash.dependencies import Input, Output
from constants import CLASSES

with open('app.yaml') as yaml_data:
    
    params = yaml.safe_load(yaml_data)

IMAGE_WIDTH = params[2]['IMAGE_WIDTH']
IMAGE_HEIGHT = params[1]['IMAGE_HEIGHT']

# Load DNN model
classifier = tf.keras.models.load_model(params[0]['keras_path'])

def classify_image(image, model, image_box=None):
  """Classify image by model
  Parameters
  ----------
  content: image content
  model: tf/keras classifier
  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)

  return [np.argmax(model.predict(np.array(images_list)))] #np.argmax(model.predict(images_list), axis=-1)

def classify_image_probas(image, model, image_box=None):
  """Classify image by model
  Parameters
  ----------
  content: image content
  model: tf/keras classifier
  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)

  return model.predict(np.array(images_list)) #np.argmax(model.predict(images_list), axis=-1)


app = dash.Dash('Traffic Signs Recognition') #,external_stylesheets=dbc.themes.BOOTSTRAP)


pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}


# Define application layout

app.layout = html.Div([
    html.H1('Automatic traffic signs recognition',style = dict(textAlign='center',backgroundColor='#6633FF')),
    html.H3('This project aims at using machine learning to predict the type of traffic sign represented in a picture.',style = dict(backgroundColor='#a6aea9')),
    html.H4('The machine learning model used in this case is a Neural Network.',style = dict(backgroundColor='#a6aea9')),
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Drag and drop or ',
                    html.A('select a picture')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='mon-image'),
    html.Div(id='ma-zone-resultat'),
    html.A("Link to my Github", href='https://github.com/Aurebeut', target="_blank",style = dict(backgroundColor='#CCFF00')),
])


@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            probas = classify_image_probas(image, classifier)[0]
            probas, classes_list = (list(t) for t in zip(*sorted(zip(probas, CLASSES.values()), reverse=True)))
            return html.Div([
                html.Hr(),
                html.Img(src=contents),
                html.H3('Predicted class : {} with a probability of {}'.format(CLASSES[predicted_class],max(probas)),style = dict(textAlign='center',backgroundColor='#6633FF')),
                #html.H5('probas associées: {}'.format(probas)),
                #html.H5('classes associées: {}'.format(classes_list)),
                html.Hr(),
                html.Div([
                        dcc.Graph(id='horizon_proba', 
                                 figure={
                                     'data': [go.Bar(x=classes_list,
                                                     y=estimated_probabilities)],
                                     "layout": layout
                                 })
                    ]),
                html.Hr(),
                html.Div([
                            dcc.Graph(id='graph', 
                                     figure={
                                         'data': [go.Bar(x=classes_list,
                                                         y=probas)],
                                         "layout": {'title': 'Predicted probabilities of belonging to each class of traffic sign'}
                                     })
                        ]),
                    html.Hr(),
                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, classifier)[0]
                probas = classify_image_probas(image, classifier)[0]
                probas, classes_list = (list(t) for t in zip(*sorted(zip(probas, CLASSES.values()), reverse=True)))
                # Affichage de l'image
                return html.Div([
                    html.Hr(),
                    html.Img(src='data:image/png;base64,' + content_string),
                    html.H3('Predicted class : {} with a probability of {}'.format(CLASSES[predicted_class],max(probas)),style = dict(textAlign='center',backgroundColor='#6633FF')),
                    #html.H5('probas associées: {}'.format(probas)),
                    #html.H5('classes associées: {}'.format(classes_list)),
                    html.Hr(),
                    html.Div([
                            dcc.Graph(id='graph', 
                                     figure={
                                         'data': [go.Bar(x=classes_list,
                                                         y=probas)],
                                         "layout": {'title': 'Predicted probabilities of belonging to each class of traffic sign'}
                                     })
                        ]),
                    html.Hr(),
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
            

# Manage interactions with callbacks
@app.callback(
    Output(component_id='ma-zone-resultat', component_property='children'),
    [Input(component_id='mon-champ-texte', component_property='value')]
)
def update_output_div(input_value):
    return html.H3('Valeur saisie ici "{}"'.format(input_value))


# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)