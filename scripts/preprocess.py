import pandas as pd
import os
import holidays
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re

# Cargar los archivos de datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flights_file = os.path.join(BASE_DIR, "data", "flights.csv")
flights_new_file = os.path.join(BASE_DIR, "data", "Data_Train.csv")  # Ruta al nuevo archivo
fuel_file = os.path.join(BASE_DIR, "data", "fuel_prices_daily.csv")
output_file = os.path.join(BASE_DIR, "data", "flights_cleaned.csv")

# Verificar si los archivos existen
if not os.path.exists(flights_file):
    raise FileNotFoundError(f"❌ El archivo '{flights_file}' no se encuentra.")
if not os.path.exists(flights_new_file):
    raise FileNotFoundError(f"❌ El archivo '{flights_new_file}' no se encuentra.")
if not os.path.exists(fuel_file):
    raise FileNotFoundError(f"❌ El archivo '{fuel_file}' no se encuentra.")

# Cargar datos de vuelos
df = pd.read_csv(flights_file)
df_new = pd.read_csv(flights_new_file)  # Cargar el nuevo archivo

# Imprimir las columnas de df y df_new para comparar y ver qué nombres coinciden
print("Columnas en flights.csv:", df.columns)
print("Columnas en Data_Train.csv:", df_new.columns)

# Renombrar las columnas de df_new para que coincidan con df
df_new.rename(columns={
    'Airline': 'Airline',
    'Date_of_Journey': 'Date_of_Journey',
    'Source': 'Source',
    'Destination': 'Destination',
    'Route': 'Route',
    'Dep_Time': 'Dep_Time',
    'Arrival_Time': 'Arrival_Time',
    'Duration': 'Duration',
    'Total_Stops': 'Total_Stops',
    'Additional_Info': 'Additional_Info',
    'Price': 'Price'
}, inplace=True)

# Concatenar los DataFrames después de renombrar las columnas
df = pd.concat([df, df_new], ignore_index=True)

# Convertir las fechas a formato datetime
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%Y-%m-%d", errors='coerce')
df["Date_of_Journey"] = df["Date_of_Journey"].fillna(pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y", errors='coerce'))
df = df.dropna(subset=['Date_of_Journey']) # Eliminar filas donde la conversión falló

# Extraer día, mes, semana y día de la semana
df["Journey_Day"] = df["Date_of_Journey"].dt.day
df["Journey_Month"] = df["Date_of_Journey"].dt.month
df["Journey_Week"] = df["Date_of_Journey"].dt.isocalendar().week
df["Journey_Weekday"] = df["Date_of_Journey"].dt.weekday

# Incorporar datos de días festivos (India)
country = "IN"
indian_holidays = holidays.country_holidays(country)
df["Is_Holiday"] = df["Date_of_Journey"].apply(lambda x: 1 if x in indian_holidays else 0)

# Calcular la distancia a los festivos más cercanos
df["Days_To_Holiday"] = df["Date_of_Journey"].apply(
    lambda x: min([abs((x.date() - h).days) for h in indian_holidays.keys()] + [365])
)

def convert_duration_to_minutes(duration):
    """Convert duration string to minutes."""
    if isinstance(duration, str):
        match = re.match(r'(\d+)h\s*(\d+)m', duration)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            return hours * 60 + minutes
        else:
            match = re.match(r'(\d+)h', duration)
            if match:
                hours = int(match.group(1))
                return hours * 60
            else:
                match = re.match(r'(\d+)m', duration)
                if match:
                    minutes = int(match.group(1))
                    return minutes
    return None

df["Flight_Duration_Minutes"] = df["Duration"].apply(convert_duration_to_minutes)
df = df.dropna(subset=['Flight_Duration_Minutes']) #Eliminamos los vuelos que tienen duración extraña o mal formateada

# Cargar precios de combustible y convertir fechas
df_fuel = pd.read_csv(fuel_file)

# Aplicar filtros para obtener datos relevantes
df_fuel = df_fuel[
    (df_fuel["product-name"] == "Carb Diesel") & (df_fuel["area-name"] == "LOS ANGELES")]

# Convertir la columna "Date" a formato datetime
df_fuel["Date"] = pd.to_datetime(df_fuel["Date"])
df_fuel = df_fuel.sort_values("Date")

# Calcular Date_int
df_fuel["Date_int"] = (df_fuel["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

# Interpolación del precio de combustible
interp_func = interp1d(df_fuel["Date_int"], df_fuel["Fuel_Price"], kind="linear", fill_value="extrapolate")
df["Fuel_Price"] = df["Date_of_Journey"].apply(
    lambda x: interp_func((x - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D'))
)

# Codificar las variables categóricas (Source y Destination)
le_source = LabelEncoder()
le_destination = LabelEncoder()
df['Source_Encoded'] = le_source.fit_transform(df['Source'])
df['Destination_Encoded'] = le_destination.fit_transform(df['Destination'])

# Guardar el dataset preprocesado
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Datos preprocesados y guardados en {output_file}")