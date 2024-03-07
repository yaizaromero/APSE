import requests
import json

# Datos para la predicción de ETA
data_eta = {"time": [10, 20, 30, 40]}

# Datos para la predicción de entrega
data_delivery = {"truck_id": "3321FBL"}

# URL del servicio
url = "http://localhost:7777"  # Reemplaza con la dirección del servidor donde se esté ejecutando tu servicio

# Enviar solicitud POST para la predicción de ETA
response_eta = requests.post(f"{url}/predict_eta", json=data_eta)
print("Respuesta de la predicción de ETA:", response_eta.json())

# Enviar solicitud POST para la predicción de entrega
response_delivery = requests.post(f"{url}/predict_delivery", json=data_delivery)
print("Respuesta de la predicción de entrega:", response_delivery.json())