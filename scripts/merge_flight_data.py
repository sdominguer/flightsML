import os
import pandas as pd

# Obtener rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flight_data_file = os.path.join(BASE_DIR, "data", "flights_cleaned.csv")
fuel_prices_file = os.path.join(BASE_DIR, "data", "fuel_prices.csv")
output_file = os.path.join(BASE_DIR, "data", "flights_with_exogenous.csv")

# Cargar datos de vuelos
df_flights = pd.read_csv(flight_data_file)
df_flights["Date_of_Journey"] = pd.to_datetime(df_flights["Date_of_Journey"])

# Cargar precios del combustible
df_fuel = pd.read_csv(fuel_prices_file)
df_fuel["Date"] = pd.to_datetime(df_fuel["Date"])

# Unir por fecha más cercana (Forward Fill)
df_flights = df_flights.merge(df_fuel, left_on="Date_of_Journey", right_on="Date", how="left")
df_flights["Fuel_Price"].fillna(method="ffill", inplace=True)

# Guardar dataset con variables exógenas
df_flights.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Datos combinados guardados en {output_file}")
