import pandas as pd
import os

# Obtener la ruta absoluta del directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construir rutas de los archivos
flight_data_file = os.path.join(BASE_DIR, "data", "flights_cleaned.csv")
exogenous_data_file = os.path.join(BASE_DIR, "data", "exogenous_data.csv")
output_file = os.path.join(BASE_DIR, "data", "flights_with_exogenous.csv")

# Cargar datos de vuelos
df_flights = pd.read_csv(flight_data_file)

# Generar datos exógenos ficticios (para demostración)
df_exogenous = pd.DataFrame({
    "Journey_Month": range(1, 13),
    "Fuel_Price_Index": [100, 110, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165],  # Ejemplo de variación mensual del precio del combustible
    "Holiday_Season": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]  # 1 si es temporada alta, 0 si no
})

# Fusionar con los datos de vuelos
df_final = df_flights.merge(df_exogenous, on="Journey_Month", how="left")

# Guardar archivo combinado
df_final.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Datos con variables exógenas guardados en {output_file}")
