import flask
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import holidays
from scipy.interpolate import interp1d
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('flight_price_model.pkl')

# Cargar datos de combustible (necesarios para la predicción)
df_fuel = pd.read_csv("data/fuel_prices_daily.csv")
df_fuel["Date"] = pd.to_datetime(df_fuel["Date"])
df_fuel = df_fuel.sort_values("Date")
df_fuel["Date_int"] = (df_fuel["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
interp_func = interp1d(df_fuel["Date_int"], df_fuel["Fuel_Price"], kind="linear", fill_value="extrapolate")

# Inicializar geolocator
geolocator = Nominatim(user_agent="flight-price-predictor")

# Definir las posibles ciudades de origen y destino (extraído de tu CSV)
POSSIBLE_SOURCES = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
POSSIBLE_DESTINATIONS = ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Hyderabad']

# Codificadores globales
le_source = LabelEncoder()
le_destination = LabelEncoder()
le_source.fit(POSSIBLE_SOURCES)  # Fit con las fuentes reales
le_destination.fit(POSSIBLE_DESTINATIONS) # Fit con los destinos reales

# Funciones auxiliares (adaptadas del preprocesamiento)

def get_coordinates(city_name):
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

def calculate_flight_distance(origin_coords, destination_coords):
    if origin_coords and destination_coords:
        return geodesic(origin_coords, destination_coords).km
    else:
        return None

def is_holiday(date):
    country = "IN"
    indian_holidays = holidays.country_holidays(country)
    return 1 if date in indian_holidays else 0

def days_to_holiday(date):
    country = "IN"
    indian_holidays = holidays.country_holidays(country)
    return min([abs((date.date() - h).days) for h in indian_holidays.keys()] + [365])

def prepare_features(source, destination, journey_date):
    """Prepara las características para la predicción."""

    # Obtener coordenadas
    origin_coords = get_coordinates(source)
    destination_coords = get_coordinates(destination)

    # Calcular distancia
    flight_distance = calculate_flight_distance(origin_coords, destination_coords)

    # Obtener precio del combustible
    fuel_price = interp_func((journey_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D'))

    # Verificar si es día festivo
    is_holiday_flag = is_holiday(journey_date)

    # Calcular días hasta el próximo día festivo
    days_until_holiday = days_to_holiday(journey_date)

    # Codificar origen y destino
    source_encoded = le_source.transform([source])[0]
    destination_encoded = le_destination.transform([destination])[0]

    # Crear diccionario de características
    features = {
        'Source_Encoded': source_encoded,
        'Destination_Encoded': destination_encoded,
        'Journey_Month': journey_date.month,
        'Journey_Week': int(journey_date.strftime("%W")), # Journey_Week
        'Journey_Weekday': journey_date.weekday(),
        'Flight_Distance': flight_distance,
        'Fuel_Price': fuel_price,
        'Is_Holiday': is_holiday_flag,
        'Days_To_Holiday': days_until_holiday,
    }
    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    if request.method == 'POST':
        source = request.form['source']
        destination = request.form['destination']
        date_str = request.form['date']
        journey_date = datetime.strptime(date_str, '%Y-%m-%d')

        for i in range(30):  # Predicciones para 30 días antes
            current_date = journey_date - timedelta(days=i)
            features = prepare_features(source, destination, current_date)

            # Convertir el diccionario de características en un DataFrame
            input_df = pd.DataFrame([features])

            # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
            feature_order = ['Source_Encoded', 'Destination_Encoded', 'Journey_Month', 'Journey_Week', 'Journey_Weekday', 'Flight_Distance', 'Fuel_Price', 'Is_Holiday', 'Days_To_Holiday']
            input_df = input_df[feature_order]

            # Manejar valores nulos (imputar con la media, por ejemplo)
            input_df = input_df.fillna(input_df.mean())

            prediction = model.predict(input_df)[0]
            predictions.append({'date': current_date.strftime('%Y-%m-%d'), 'price': round(prediction, 2)})

    return render_template('index.html', predictions=predictions,
                           possible_sources=POSSIBLE_SOURCES,
                           possible_destinations=POSSIBLE_DESTINATIONS)

if __name__ == '__main__':
    app.run(debug=True)