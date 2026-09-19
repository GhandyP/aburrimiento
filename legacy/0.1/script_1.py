
# Crear programa Python para análisis de aburrimiento grupal con ML y Deep Learning

python_program = '''
"""
Sistema de Análisis de Aburrimiento Grupal
Utilizando Machine Learning y Deep Learning con PyCM

Este programa analiza el nivel de aburrimiento en grupos basándose en:
- Estructuras socio-políticas y económicas
- Manifestaciones grupales
- Teorías de alienación y neoliberalismo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pycm import ConfusionMatrix
import json
import warnings
warnings.filterwarnings('ignore')


class AnalizadorAburrimiento:
    """
    Clase principal para analizar el aburrimiento en grupos
    utilizando técnicas de ML y Deep Learning
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.dl_model = None
        self.feature_names = []
        
    def preparar_datos(self, datos_json):
        """
        Prepara los datos de entrada para el análisis
        
        Parámetros:
        -----------
        datos_json : dict
            Diccionario con datos del grupo incluyendo:
            - estructuras_sistemicas: indicadores de dominación sociopolítica
            - manifestaciones_grupales: síntomas observables
            - dimensiones_medicion: métricas de estado y contexto
            - etiqueta: nivel de aburrimiento (bajo/medio/alto)
        """
        df = pd.DataFrame(datos_json)
        
        # Separar características y etiquetas
        X = df.drop('etiqueta', axis=1)
        y = df['etiqueta']
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded, X, y
    
    def entrenar_random_forest(self, X_train, y_train):
        """
        Entrena modelo Random Forest para clasificación de aburrimiento
        """
        print("\\n=== Entrenando Random Forest ===")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train, y_train)
        print("✓ Random Forest entrenado exitosamente")
        
        return self.rf_model
    
    def entrenar_deep_learning(self, X_train, y_train, num_clases):
        """
        Entrena red neuronal profunda para análisis de aburrimiento
        """
        print("\\n=== Entrenando Red Neuronal Profunda ===")
        
        # Arquitectura de la red
        self.dl_model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_clases, activation='softmax')
        ])
        
        # Compilar modelo
        self.dl_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar
        history = self.dl_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        print("✓ Red Neuronal entrenada exitosamente")
        return self.dl_model, history
    
    def evaluar_con_pycm(self, y_true, y_pred, nombre_modelo):
        """
        Evalúa el modelo usando PyCM para análisis detallado de la matriz de confusión
        """
        print(f"\\n=== Evaluación {nombre_modelo} con PyCM ===")
        
        # Crear matriz de confusión con PyCM
        cm = ConfusionMatrix(actual_vector=y_true.tolist(), 
                            predict_vector=y_pred.tolist())
        
        # Mostrar estadísticas principales
        print(f"\\nPrecisión General: {cm.Overall_ACC:.4f}")
        print(f"Kappa de Cohen: {cm.Kappa:.4f}")
        print(f"F1-Score Macro: {cm.F1_Macro:.4f}")
        
        # Estadísticas por clase
        print("\\nEstadísticas por Nivel de Aburrimiento:")
        clases_texto = self.label_encoder.inverse_transform(sorted(cm.classes))
        
        for i, clase in enumerate(sorted(cm.classes)):
            print(f"\\n  {clases_texto[i].upper()}:")
            print(f"    - Precisión: {cm.class_stat['PPV'][clase]:.4f}")
            print(f"    - Recall: {cm.class_stat['TPR'][clase]:.4f}")
            print(f"    - F1-Score: {cm.class_stat['F1'][clase]:.4f}")
        
        return cm
    
    def calcular_importancia_features(self):
        """
        Calcula la importancia de cada característica en la predicción
        """
        if self.rf_model is None:
            print("Error: Primero debe entrenar el modelo Random Forest")
            return None
        
        importancias = pd.DataFrame({
            'caracteristica': self.feature_names,
            'importancia': self.rf_model.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        return importancias
    
    def predecir_aburrimiento(self, nuevos_datos, usar_dl=False):
        """
        Predice el nivel de aburrimiento para nuevos datos
        
        Parámetros:
        -----------
        nuevos_datos : array-like
            Características del grupo a analizar
        usar_dl : bool
            Si True, usa red neuronal; si False, usa Random Forest
        """
        datos_scaled = self.scaler.transform(nuevos_datos)
        
        if usar_dl and self.dl_model is not None:
            predicciones = self.dl_model.predict(datos_scaled, verbose=0)
            clases_pred = np.argmax(predicciones, axis=1)
        elif self.rf_model is not None:
            clases_pred = self.rf_model.predict(datos_scaled)
        else:
            print("Error: No hay modelo entrenado")
            return None
        
        # Decodificar predicciones
        niveles = self.label_encoder.inverse_transform(clases_pred)
        
        return niveles, clases_pred
    
    def analizar_dinamica_grupo(self, datos_grupo):
        """
        Realiza un análisis completo de la dinámica de aburrimiento del grupo
        """
        print("\\n" + "="*60)
        print("ANÁLISIS DE ABURRIMIENTO GRUPAL")
        print("="*60)
        
        # Calcular métricas agregadas
        analisis = {
            'dominacion_sociopolitica': np.mean([
                datos_grupo['reflejo_sistemas_culturales'],
                datos_grupo['productividad_capitalista'],
                datos_grupo['alienacion_neoliberal']
            ]),
            'manifestacion_grupal': np.mean([
                datos_grupo['malestar_generalizado'],
                datos_grupo['carencia_sentido'],
                datos_grupo['restriccion_libertad']
            ]),
            'potencial_critico': np.mean([
                datos_grupo['frustracion_agencia'],
                datos_grupo['estrategias_bloqueadas'],
                datos_grupo['angustia_profunda']
            ])
        }
        
        print("\\nIndicadores Principales:")
        print(f"  - Dominación Sociopolítica: {analisis['dominacion_sociopolitica']:.2f}")
        print(f"  - Manifestación Grupal: {analisis['manifestacion_grupal']:.2f}")
        print(f"  - Potencial Crítico: {analisis['potencial_critico']:.2f}")
        
        return analisis


def generar_datos_ejemplo(n_muestras=500):
    """
    Genera datos de ejemplo para demostrar el funcionamiento del sistema
    """
    np.random.seed(42)
    
    datos = []
    etiquetas = ['bajo', 'medio', 'alto']
    
    for _ in range(n_muestras):
        # Seleccionar nivel de aburrimiento
        etiqueta = np.random.choice(etiquetas)
        
        # Generar características según el nivel
        if etiqueta == 'bajo':
            base = 0.3
            variacion = 0.2
        elif etiqueta == 'medio':
            base = 0.5
            variacion = 0.15
        else:  # alto
            base = 0.8
            variacion = 0.15
        
        muestra = {
            # Estructuras sistémicas (0-1)
            'reflejo_sistemas_culturales': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'productividad_capitalista': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'alienacion_neoliberal': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'racismo_sistemico': np.clip(base * 0.7 + np.random.normal(0, variacion), 0, 1),
            
            # Manifestaciones grupales (0-1)
            'malestar_generalizado': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'carencia_sentido': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'restriccion_libertad': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'frustracion_agencia': np.clip(base + np.random.normal(0, variacion), 0, 1),
            
            # Dimensiones medición (0-1)
            'desenganche': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'alta_excitacion': np.clip(base * 0.6 + np.random.normal(0, variacion), 0, 1),
            'inatencion': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'percepcion_tiempo_lenta': np.clip(base + np.random.normal(0, variacion), 0, 1),
            
            # Factores contextuales
            'estrategias_bloqueadas': np.clip(base + np.random.normal(0, variacion), 0, 1),
            'angustia_profunda': np.clip(base * 0.9 + np.random.normal(0, variacion), 0, 1),
            
            'etiqueta': etiqueta
        }
        
        datos.append(muestra)
    
    return datos


def main():
    """
    Función principal para ejecutar el análisis completo
    """
    print("\\n" + "="*60)
    print("SISTEMA DE ANÁLISIS DE ABURRIMIENTO GRUPAL")
    print("Machine Learning & Deep Learning con PyCM")
    print("="*60)
    
    # Generar datos de ejemplo
    print("\\n1. Generando datos de ejemplo...")
    datos = generar_datos_ejemplo(n_muestras=500)
    print(f"✓ Generados {len(datos)} casos de estudio")
    
    # Inicializar analizador
    analizador = AnalizadorAburrimiento()
    
    # Preparar datos
    print("\\n2. Preparando datos para análisis...")
    X_scaled, y_encoded, X_original, y_original = analizador.preparar_datos(datos)
    print(f"✓ Datos preparados: {X_scaled.shape[0]} muestras, {X_scaled.shape[1]} características")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Entrenar Random Forest
    print("\\n3. Entrenamiento de modelos...")
    analizador.entrenar_random_forest(X_train, y_train)
    
    # Entrenar Deep Learning
    num_clases = len(np.unique(y_encoded))
    analizador.entrenar_deep_learning(X_train, y_train, num_clases)
    
    # Evaluar Random Forest
    print("\\n4. Evaluación de modelos...")
    y_pred_rf = analizador.rf_model.predict(X_test)
    cm_rf = analizador.evaluar_con_pycm(y_test, y_pred_rf, "Random Forest")
    
    # Evaluar Deep Learning
    y_pred_dl_prob = analizador.dl_model.predict(X_test, verbose=0)
    y_pred_dl = np.argmax(y_pred_dl_prob, axis=1)
    cm_dl = analizador.evaluar_con_pycm(y_test, y_pred_dl, "Deep Learning")
    
    # Importancia de características
    print("\\n5. Análisis de importancia de características...")
    importancias = analizador.calcular_importancia_features()
    print("\\nTop 10 características más importantes:")
    print(importancias.head(10).to_string(index=False))
    
    # Ejemplo de predicción
    print("\\n6. Ejemplo de predicción para un nuevo grupo...")
    nuevo_grupo = pd.DataFrame([{
        'reflejo_sistemas_culturales': 0.8,
        'productividad_capitalista': 0.75,
        'alienacion_neoliberal': 0.85,
        'racismo_sistemico': 0.6,
        'malestar_generalizado': 0.9,
        'carencia_sentido': 0.85,
        'restriccion_libertad': 0.8,
        'frustracion_agencia': 0.9,
        'desenganche': 0.85,
        'alta_excitacion': 0.5,
        'inatencion': 0.8,
        'percepcion_tiempo_lenta': 0.9,
        'estrategias_bloqueadas': 0.85,
        'angustia_profunda': 0.8
    }])
    
    nivel_rf, _ = analizador.predecir_aburrimiento(nuevo_grupo, usar_dl=False)
    nivel_dl, _ = analizador.predecir_aburrimiento(nuevo_grupo, usar_dl=True)
    
    print(f"\\nPredicción Random Forest: {nivel_rf[0].upper()}")
    print(f"Predicción Deep Learning: {nivel_dl[0].upper()}")
    
    # Análisis de dinámica
    analizador.analizar_dinamica_grupo(nuevo_grupo.iloc[0].to_dict())
    
    print("\\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    
    return analizador, cm_rf, cm_dl


if __name__ == "__main__":
    analizador, cm_rf, cm_dl = main()
'''

# Guardar el programa Python
with open('analizador_aburrimiento_ml.py', 'w', encoding='utf-8') as f:
    f.write(python_program)

print("✓ Programa Python creado: analizador_aburrimiento_ml.py")
print(f"Tamaño del código: {len(python_program)} caracteres")
print("\nCaracterísticas del programa:")
print("- Random Forest para clasificación")
print("- Red Neuronal Profunda (Deep Learning)")
print("- Análisis con PyCM (matrices de confusión)")
print("- Generación de datos de ejemplo")
print("- Predicción de niveles de aburrimiento")
print("- Análisis de importancia de características")
