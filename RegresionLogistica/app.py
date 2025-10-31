from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Variables globales
model = None
scaler = None

def train_or_load_model():
    """Entrena o carga el modelo de predicción de compra"""
    global model, scaler
    
    model_path = 'purchase_model.pkl'
    scaler_path = 'purchase_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Modelo cargado")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
    
    # Intentar entrenar con UserData.csv
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'UserData.csv')
    
    if os.path.exists(csv_path):
        try:
            print(f"Cargando datos desde {csv_path}...")
            df = pd.read_csv(csv_path)
            
            X = df[['Age', 'EstimatedSalary']].values
            y = df['Purchased'].values
            
            print(f"Dataset: {X.shape[0]} registros")
            print(f"Clase 0 (No compra): {np.sum(y == 0)}")
            print(f"Clase 1 (Compra): {np.sum(y == 1)}")
            
            # Normalizar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Entrenar modelo
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y)
            
            # Guardar modelo
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            print(f"Modelo entrenado con precisión: {model.score(X_scaled, y):.2%}")
            print(f"Modelo guardado en {model_path}")
            return True
            
        except Exception as e:
            print(f"Error con UserData.csv: {e}")
    else:
        print(f"Archivo {csv_path} no encontrado")
    
    # Crear modelo sintético si no hay datos reales
    print("Generando modelo sintético...")
    rng = np.random.default_rng(42)
    n_samples = 1000
    
    # Features: Edad (18-60), Salario estimado (15000-150000)
    age = rng.uniform(18, 60, n_samples)
    salary = rng.uniform(15000, 150000, n_samples)
    
    X = np.column_stack([age, salary])
    
    # Target: mayor probabilidad de compra con mayor edad y salario
    z = -8.0 + 0.05 * age + 0.00005 * salary
    probs = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, probs)
    
    print(f"Datos sintéticos - No compra: {np.sum(y == 0)}, Compra: {np.sum(y == 1)}")
    
    # Normalizar y entrenar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Guardar modelo
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Modelo sintético entrenado con precisión: {model.score(X_scaled, y):.2%}")
    return True

@app.route('/')
def home():
    """Página principal"""
    return render_template('purchase.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    try:
        data = request.get_json()
        
        # Extraer datos
        age = float(data.get('age', 0))
        salary = float(data.get('salary', 0))
        
        # Validar datos
        if age < 18 or age > 100:
            return jsonify({'error': 'Edad debe estar entre 18 y 100'}), 400
        if salary < 0:
            return jsonify({'error': 'Salario debe ser positivo'}), 400
        
        # Preparar features
        features = np.array([[age, salary]])
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'message': '¡COMPRARÁ el producto!' if prediction == 1 else 'NO COMPRARÁ el producto'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en predicción: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("SISTEMA DE PREDICCIÓN DE COMPRA DE PRODUCTO")
    print("=" * 60)
    
    # Entrenar o cargar modelo
    if train_or_load_model():
        print("\n✓ Sistema listo")
        print("Servidor iniciando en http://127.0.0.1:5000")
        print("=" * 60)
        app.run(debug=True, port=5000)
    else:
        print("\n✗ Error: No se pudo inicializar el modelo")
        print("=" * 60)
