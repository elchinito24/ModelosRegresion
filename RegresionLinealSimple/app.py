from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

app = Flask(__name__)

# Cargar o entrenar el modelo
MODEL_PATH = 'modelo_salario.pkl'

def entrenar_modelo():
    """Entrena el modelo de regresión lineal con los datos de salario"""
    # Cargar datos - buscar CSV en el directorio padre
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'SalaryData.csv')
    df = pd.read_csv(csv_path)
    
    # Preparar datos
    X = df.iloc[:, :-1].values  # Años de experiencia
    y = df.iloc[:, 1].values    # Salario
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Guardar modelo
    joblib.dump(modelo, MODEL_PATH)
    
    return modelo

# Cargar o entrenar modelo al iniciar la aplicación
if os.path.exists(MODEL_PATH):
    modelo = joblib.load(MODEL_PATH)
else:
    modelo = entrenar_modelo()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    """Endpoint para realizar predicciones"""
    try:
        # Obtener años de experiencia del formulario
        anos_experiencia = float(request.form.get('anos_experiencia', 0))
        
        # Validar entrada
        if anos_experiencia < 0:
            return jsonify({'error': 'Los años de experiencia deben ser positivos'}), 400
        
        # Realizar predicción
        prediccion = modelo.predict([[anos_experiencia]])[0]
        
        # Obtener coeficientes del modelo
        coeficiente = modelo.coef_[0]
        intercepto = modelo.intercept_
        
        return jsonify({
            'anos_experiencia': anos_experiencia,
            'salario_predicho': round(prediccion, 2),
            'coeficiente': round(coeficiente, 2),
            'intercepto': round(intercepto, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/estadisticas')
def estadisticas():
    """Endpoint para obtener estadísticas del modelo"""
    try:
        # Cargar datos - buscar CSV en el directorio padre
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'SalaryData.csv')
        df = pd.read_csv(csv_path)
        
        # Preparar datos
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 1].values
        
        # Calcular R²
        score = modelo.score(X, y)
        
        return jsonify({
            'r2_score': round(score, 4),
            'coeficiente': round(modelo.coef_[0], 2),
            'intercepto': round(modelo.intercept_, 2),
            'num_muestras': len(df)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
