import flask
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
import re

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('flight_price_model.pkl')

# Definir las posibles ciudades de origen y destino (extraído de tu CSV)
POSSIBLE_SOURCES = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
POSSIBLE_DESTINATIONS = ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Hyderabad']

# Codificadores globales
le_source = LabelEncoder()
le_destination = LabelEncoder()
le_source.fit(POSSIBLE_SOURCES)  # Fit con las fuentes reales
le_destination.fit(POSSIBLE_DESTINATIONS) # Fit con los destinos reales

# Cargar datos de combustible (necesarios para la predicción)
df_fuel = pd.read_csv("data/fuel_prices_daily.csv")
df_fuel["Date"] = pd.to_datetime(df_fuel["Date"])
df_fuel = df_fuel.sort_values("Date")
df_fuel["Date_int"] = (df_fuel["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
interp_func = interp1d(df_fuel["Date_int"], df_fuel["Fuel_Price"], kind="linear", fill_value="extrapolate")

# Funciones auxiliares
def prepare_features(source, destination, journey_date):
    """Prepara las características para la predicción."""

    # Obtener precio del combustible
    fuel_price = interp_func((journey_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D'))

    # Codificar origen y destino
    source_encoded = le_source.transform([source])[0]
    destination_encoded = le_destination.transform([destination])[0]

    # Crear diccionario de características
    features = {
        'Source_Encoded': source_encoded,
        'Destination_Encoded': destination_encoded,
        'Journey_Month': journey_date.month,
        'Journey_Week': int(journey_date.strftime("%W")),
        'Journey_Weekday': journey_date.weekday(),
        'Flight_Duration_Minutes': 120, #Placeholder, ya que ahora todos los vuelos tienen la misma duración
        'Fuel_Price': float(fuel_price),
        'Is_Holiday': 0, # Placeholder, ya que no podemos predecir festivos
        'Days_To_Holiday': 10 #Placeholder, siempre es el mismo valor, da cero, no sirve este valor
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
            feature_order = ['Source_Encoded', 'Destination_Encoded', 'Journey_Month', 'Journey_Week', 'Journey_Weekday', 'Flight_Duration_Minutes', 'Fuel_Price', 'Is_Holiday', 'Days_To_Holiday']
            input_df = input_df[feature_order]
            # Convertir a tipo float
            input_df = input_df.astype(float) #Cassteo a Float
            if 'Price' not in input_df.columns:  # Verificar si la columna 'Price' ya existe
                input_df['Price'] = 0  # Agregar la columna 'Price' con valores iniciales de 0

            # Ahora, asegúrate de que las columnas estén en el mismo orden que durante el entrenamiento y convertir a numpy array
            feature_order = ['Source_Encoded', 'Destination_Encoded', 'Journey_Month', 'Journey_Week', 'Journey_Weekday', 'Flight_Duration_Minutes', 'Fuel_Price', 'Is_Holiday', 'Days_To_Holiday','Price']
            input_df = input_df[feature_order]
            prediction = model.predict(input_df.values)[0]

            predictions.append({'date': current_date.strftime('%Y-%m-%d'), 'price': round(prediction, 2)})

    return render_template('index.html', predictions=predictions,
                           possible_sources=POSSIBLE_SOURCES,
                           possible_destinations=POSSIBLE_DESTINATIONS)

if __name__ == '__main__':
    app.run(debug=True)