from flask import Flask, request, render_template
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Función para obtener las coordenadas de una ciudad
def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="flight-price-predictor")
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

# Cargar el modelo preentrenado y scaler
BASE_DIR = "ruta/a/tu/directorio"
model_file = BASE_DIR + "/model.pkl"
scaler_file = BASE_DIR + "/scaler.pkl"

model = SARIMAX.load(model_file)  # Asegúrate de haber guardado el modelo después del entrenamiento
scaler = StandardScaler()  # Usa el mismo scaler que usaste en el entrenamiento

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    origin = request.form['origin']
    destination = request.form['destination']
    date_of_journey = request.form['date_of_journey']
    
    # Obtener coordenadas de origen y destino
    origin_coords = get_coordinates(origin)
    destination_coords = get_coordinates(destination)
    
    if origin_coords and destination_coords:
        flight_distance = geodesic(origin_coords, destination_coords).km
    else:
        return "Error: No se pudo encontrar las coordenadas."

    # Calcular otras variables necesarias, como el precio de combustible
    # Asumimos que ya tienes la función o variable que te da el precio de combustible para esa fecha.
    
    fuel_price = 60  # Esto es solo un ejemplo

    # Crear el DataFrame de entrada para la predicción
    input_data = pd.DataFrame([[fuel_price, 0, flight_distance]], columns=["Fuel_Price", "Is_Holiday", "Flight_Distance"])

    # Escalar los datos
    input_scaled = scaler.transform(input_data)

    # Realizar la predicción
    prediction = model.predict(exog=input_scaled)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
