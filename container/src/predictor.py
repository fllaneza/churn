
import os
import json
import pickle
import io
import sys
import signal
import traceback

import flask

import pandas as pd

from aux_functions import process_df

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Una clase para guardar el modelo. Simplemente carga el modelo y lo mantiene.
# Además, tiene una función que realiza la predicción con el modelo y los datos de entrada.
class ScoringService(object):
    model = None                

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, 'model.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)

# La aplicación de Flask para servir las predicciones
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convierte el JSON en un pandas DataFrame
    if flask.request.method == "POST":
        data = flask.request.get_json()
        data = pd.DataFrame.from_dict(data['data'])
        X_inference, _ = process_df(data, model_path, "inference")
    else:
        return flask.Response(response='Este predictor sólo soporta datos en formato JSON', status=415, mimetype='text/plain')

    print('Invocado con {} registros'.format(data.shape[0]))

    # Realiza la predicción
    predictions = ScoringService.predict(X_inference)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')