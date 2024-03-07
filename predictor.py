import pickle
from flask import Flask, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# Cargar modelos de predicción
with open('APSELogisticSimulator-main/data/eda/modelo_tiempo_entrega_paquetes.pkl', 'rb') as f:
    modelo_tiempo_viaje = pickle.load(f)
    
with open('APSELogisticSimulator-main/data/eda/modelo_tiempo_viaje_camiones.pkl', 'rb') as f:
    modelo_tiempo_entrega = pickle.load(f)

with open('APSELogisticSimulator-main/data/prediccionOnline/le.pkl', 'rb') as f:
    labelEncoder = pickle.load(f)

# Endpoint para predicción de ETA
@app.route('/predict_eta', methods=['POST'])
def predict_eta():
    
    # Obtener los datos JSON de la solicitud
    json_data = request.get_json()
    
    # Obtener los datos de tiempo de la solicitud
    time_data = json_data.get('time')
    
    # Convertir los datos de tiempo a una lista
    time_list = np.array(time_data).tolist()
    
    # Realizar la predicción
    prediction = modelo_tiempo_viaje.predict(np.array(time_list).reshape(-1, 1))
    
    # Convertir la predicción a una lista
    prediction_list = prediction.tolist()
    
    # Devolver la predicción como una respuesta JSON
    return jsonify({'prediction_eta': prediction_list[0]})



# Endpoint para cargar el punto de entrega.
# La entrada es un objeto JSON con los atributos truckId y time
@app.route('/predict_delivery', methods=['POST'])
def predict_delivery():
    # Obtener los datos JSON de la solicitud
    json_data = request.get_json()
    
    # Obtener el ID del camión de la solicitud
    truck_id = json_data.get('truck_id')
    
    # Convertir el ID del camión a una lista
    truck_id_list = np.array([truck_id]).tolist()
    
    # Codificar el ID del camión
    encoded_truck_id = labelEncoder.transform(truck_id_list)
    
    # Realizar la predicción
    prediction = modelo_tiempo_entrega.predict(encoded_truck_id.reshape(-1, 1))
    
    # Convertir la predicción a una lista
    prediction_list = prediction.tolist()
    
    # Devolver la predicción como una respuesta JSON
    return jsonify({'prediction_delivery': prediction_list[0]})



if __name__ == '__main__':
    app.run(debug=True, port=7777, host='0.0.0.0')
