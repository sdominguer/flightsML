import pandas as pd
import requests
import os

# Configurar API Key (Reemplázala con tu clave si es necesario)
API_KEY = "O7XfyrbpYO3qOreSWMcC5YTJBxQIybSzbcO1uMjy"
API_URL = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?frequency=daily&data[0]=value&start=2019-01-01&end=2019-12-31&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key={API_KEY}"


# Definir ruta de almacenamiento
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_file = os.path.join(BASE_DIR, "data", "fuel_prices_daily.csv")

# Llamada a la API
response = requests.get(API_URL)

if response.status_code == 200:
    data = response.json()
    
    # Extraer datos
    records = data.get("response", {}).get("data", [])

    if not records:
        print("❌ No se encontraron datos en la API.")
    else:
        # Convertir a DataFrame
        df_fuel = pd.DataFrame(records)

        # Renombrar columnas para claridad
        df_fuel = df_fuel.rename(columns={"period": "Date", "value": "Fuel_Price"})

        # Convertir fechas a datetime
        df_fuel["Date"] = pd.to_datetime(df_fuel["Date"])

        # Ordenar por fecha ascendente
        df_fuel = df_fuel.sort_values(by="Date")

        # Guardar en CSV
        df_fuel.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ Datos de gasolina descargados y guardados en {output_file}")
else:
    print(f"❌ Error al obtener datos: {response.status_code}")
