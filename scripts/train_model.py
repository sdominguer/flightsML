import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
import re
import holidays
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los archivos de datos
flights_file = "data/flights.csv"
flights_new_file = "data/Data_Train.csv"
fuel_file = "data/fuel_prices_daily.csv"
output_file = "data/flights_cleaned.csv"

# Cargar datos de vuelos
df = pd.read_csv(flights_file)
df_new = pd.read_csv(flights_new_file)

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
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y", errors='coerce')
df = df.dropna(subset=['Date_of_Journey'])

# Extraer día, mes, semana y día de la semana
df["Journey_Day"] = df["Date_of_Journey"].dt.day
df["Journey_Month"] = df["Date_of_Journey"].dt.month
df["Journey_Week"] = df["Date_of_Journey"].dt.isocalendar().week
df["Journey_Weekday"] = df["Date_of_Journey"].dt.weekday

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
df = df.dropna(subset=['Flight_Duration_Minutes'])

# Cargar precios de combustible y convertir fechas
df_fuel = pd.read_csv(fuel_file)
df_fuel = df_fuel.rename(columns={'Date': 'Fecha'})
df_fuel["Fecha"] = pd.to_datetime(df_fuel["Fecha"])
df_fuel = df_fuel.sort_values("Fecha")
df_fuel["Date_int"] = (df_fuel["Fecha"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

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

# Incorporar datos de días festivos (India)
country = "IN"
indian_holidays = holidays.country_holidays(country)
df["Is_Holiday"] = df["Date_of_Journey"].apply(lambda x: 1 if x in indian_holidays else 0)

# Calcular la distancia a los festivos más cercanos
df["Days_To_Holiday"] = df["Date_of_Journey"].apply(
    lambda x: min([abs((x.date() - h).days) for h in indian_holidays.keys()] + [365])
)

# Seleccionar las variables relevantes para el modelo
features = ['Source_Encoded', 'Destination_Encoded', 'Journey_Month', 'Journey_Week', 'Journey_Weekday', 'Flight_Duration_Minutes', 'Fuel_Price', 'Is_Holiday', 'Days_To_Holiday', 'Price']

X = df[features]

# Imprimir la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación entre Características')
plt.show()

y = df['Price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir hiperparámetros manualmente con regularización más fuerte
params = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_estimators': 700,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.3,
    'reg_alpha': 0.15,
    'reg_lambda': 0.15
}

# Crear el modelo XGBoost
xgb_model = xgb.XGBRegressor(**params)

# Entrenar el modelo
xgb_model.fit(X_train, y_train)

# Realizar predicciones sobre los datos de prueba
y_pred = xgb_model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Guardar el modelo entrenado
joblib.dump(xgb_model, 'flight_price_model.pkl')
print("✅ Modelo guardado como 'flight_price_model.pkl'")

np.save('le_source_classes.npy', le_source.classes_)
np.save('le_destination_classes.npy', le_destination.classes_)