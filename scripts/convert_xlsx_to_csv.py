import pandas as pd
import os

# Obtener la ruta absoluta del directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construir rutas correctas
input_file = os.path.join(BASE_DIR, "data", "Data_Train.xlsx")
output_file = os.path.join(BASE_DIR, "data", "Data_Train.csv")

# Verificar si el archivo existe
if not os.path.exists(input_file):
    raise FileNotFoundError(f"El archivo '{input_file}' no se encuentra. Verifica la ruta.")

# Cargar el archivo Excel
df = pd.read_excel(input_file)

# Guardarlo como CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Archivo convertido y guardado en {output_file}")
