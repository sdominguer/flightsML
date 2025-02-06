```markdown
# Flight Price Prediction App

This project is a Flask-based web application designed to predict flight prices. It takes a flight origin, destination, and date as input and provides price predictions for the 30 days leading up to the specified date.

## Overview

The goal of this project is to provide users with insights into the potential fluctuations in flight prices, helping them to identify the most economical time to purchase their tickets. The application uses a machine learning model trained on historical flight data, fuel prices, and other relevant variables to make these predictions.

## Features

*   **User-Friendly Interface:** A simple and intuitive web form for inputting flight details (origin, destination, and date).
*   **Price Prediction:** Predicts flight prices for the 30 days prior to the specified date.
*   **Data Visualization:** Presents the predicted prices in an organized list, making it easy to identify the lowest prices.
*   **Open Source:** A collaborative and open-source application.

## Technologies Used

*   **Flask:** A micro web framework for Python used to build the web application.
*   **Pandas:** A powerful data analysis and manipulation library.
*   **Scikit-learn:** A machine learning library for model training and evaluation.
*   **XGBoost:** A gradient boosting framework used for building the prediction model.
*   **Scipy:** Used for data interpolation.
*   **Datetime:** Used for data time features.
*   **NumPy:** A library for numerical computing in Python.

## Data Sources

The application relies on the following data sources:

*   `flights.csv`: Historical flight data, including origin, destination, date, and price.
*   `Data_Train.csv`: Additional flight data to increase dataset size.
*   `fuel_prices_daily.csv`: Daily fuel prices used as a feature in the prediction model.

## Setup and Installation

1.  **Clone the Repository:**

```bash
git clone [repository URL]
cd flight_price_predictor
```

2.  **Create a Virtual Environment (Recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

3.  **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4.  **Directory structure**
The project's folder structure should look like the following

```
flight_price_predictor/
├── app.py          # Aplicación Flask
├── flight_price_model.pkl  # Modelo entrenado
├── data/
│   └── fuel_prices_daily.csv
├── templates/
│   └── index.html  # Formulario HTML
└── requirements.txt # dependencias
```

5.  **Train the Model:**

Before running the application, you need to train the machine learning model. Execute the training script:

```bash
python train_model.py
```

This script performs data preprocessing, feature engineering, model training, and saves the trained model as `flight_price_model.pkl`.

6. **Create a new file called scaler.pkl** Run the previous step and the file should be created

7. **Create Label encoder files, `le_source_classes.npy` and `le_destination_classes.npy` , by making this code into your train model:**

```python
# Guarda las Clases del LabelEncoder
np.save('le_source_classes.npy', le_source.classes_)
np.save('le_destination_classes.npy', le_destination.classes_)
```

8.  **Run the Flask Application:**
   ```bash
   python app.py
   ```

This will start the Flask development server, and you can access the application in your web browser at `http://127.0.0.1:5000/`.

## Usage

1.  **Access the Web Application:** Open your web browser and go to `http://127.0.0.1:5000/`.
2.  **Enter Flight Details:**
    *   Select the origin city from the "Source" dropdown menu.
    *   Select the destination city from the "Destination" dropdown menu.
    *   Enter the desired date of travel in the "Date (YYYY-MM-DD)" field.
3.  **Predict Prices:** Click the "Predict Prices" button to submit the form.
4.  **View Predictions:** The application will display a list of predicted prices for the 30 days leading up to the specified date.

## Model Training and Data Preprocessing

The data preprocessing and model training are performed in the `train_model.py` script. Here is a summary of the steps involved:

1.  **Load Data:** Load flight data from `flights.csv` and `Data_Train.csv`. Load fuel price data from `fuel_prices_daily.csv`.
2.  **Data Cleaning:** Perform data cleaning steps, such as handling missing values and converting data types.
3.  **Feature Engineering:** Create new features, such as journey day, month, week, weekday, and fuel price.
4.  **Data Transformation:** Encode categorical variables using Label Encoding and apply log transformation to the target variable (Price) to reduce skewness.
5.  **Data Splitting:** Split the data into training and testing sets.
6.  **Model Training:** Train the XGBoost model using the training data and a predefined set of hyperparameters.
7.  **Model Evaluation:** Evaluate the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the testing data.
8.  **Model Persistence:** Save the trained model as a .pkl file for later use in the Flask application.

## Folder structure

The project's folder structure should look like the following

```
flight_price_predictor/
├── app.py          # Aplicación Flask
├── flight_price_model.pkl  # Modelo entrenado
├── data/
│   └── fuel_prices_daily.csv
├── templates/
│   └── index.html  # Formulario HTML
└── requirements.txt # dependencias
```

## Disclaimer

*   The accuracy of the predictions depends on the quality and representativeness of the data used to train the model.
*   The model may not be accurate for predictions far outside the range of dates or conditions present in the training data.

## Contributing

Contributions to this project are welcome! Feel free to submit bug reports, feature requests, or pull requests.

## License


