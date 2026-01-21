
# Crear un resumen visual de las características del sistema en formato tabla

import pandas as pd

# Tabla 1: Comparación de modelos
modelos_comparacion = pd.DataFrame({
    'Modelo': ['Random Forest', 'Red Neuronal Profunda', 'PyCM (Evaluación)'],
    'Tipo': ['Ensemble Learning', 'Deep Learning', 'Análisis de Métricas'],
    'Características Principales': [
        '100 árboles, balanced weights, max_depth=10',
        '4 capas (128→64→32→3), dropout, adam optimizer',
        'Matriz confusión, precision, recall, F1-score'
    ],
    'Accuracy Esperado': ['88-92%', '85-90%', 'N/A (herramienta de análisis)'],
    'Ventajas': [
        'Resistente a overfitting, interpretable',
        'Captura patrones complejos, alta capacidad',
        'Métricas detalladas por clase'
    ]
})

# Tabla 2: Estructura de indicadores
indicadores = pd.DataFrame({
    'Categoría': [
        'Estructuras Sistémicas', 'Estructuras Sistémicas', 'Estructuras Sistémicas', 'Estructuras Sistémicas',
        'Manifestaciones Grupales', 'Manifestaciones Grupales', 'Manifestaciones Grupales', 'Manifestaciones Grupales',
        'Dimensiones Medición', 'Dimensiones Medición', 'Dimensiones Medición', 'Dimensiones Medición', 
        'Dimensiones Medición', 'Dimensiones Medición'
    ],
    'Indicador': [
        'Reflejo sistemas culturales', 'Productividad capitalista', 'Alienación neoliberal', 'Racismo sistémico',
        'Malestar generalizado', 'Carencia de sentido', 'Restricción de libertad', 'Frustración de agencia',
        'Desenganche', 'Alta excitación', 'Inatención', 'Percepción tiempo lenta',
        'Estrategias bloqueadas', 'Angustia profunda'
    ],
    'Rango': ['0.0-1.0'] * 14,
    'Fundamento Teórico': [
        'Crítica artística al Fordismo', 'Lógica capitalista racional', 'Individualización neoliberal', 'Desigualdad racial sistémica',
        'Padecimiento colectivo crónico', 'Meaninglessness social', 'Opresión sistémica', 'Falta de agencia efectiva',
        'Boredom Proneness Scale', 'MSBS - Alta arousal', 'MSBS - Atención', 'MSBS - Percepción temporal',
        'Respuestas bloqueadas', 'Potencial revolucionario'
    ]
})

# Tabla 3: Niveles de clasificación
niveles_clasificacion = pd.DataFrame({
    'Nivel': ['BAJO', 'MEDIO', 'ALTO'],
    'Rango': ['0.0 - 0.4', '0.4 - 0.7', '0.7 - 1.0'],
    'Color': ['Verde', 'Naranja', 'Rojo'],
    'Interpretación': [
        'Grupo saludable con buen compromiso',
        'Señales de desenganche presentes',
        'Aburrimiento sistémico severo'
    ],
    'Acción Recomendada': [
        'Mantener condiciones actuales',
        'Intervenciones preventivas necesarias',
        'Intervención inmediata requerida'
    ]
})

# Tabla 4: Tecnologías utilizadas
tecnologias = pd.DataFrame({
    'Componente': ['Backend', 'Backend', 'Backend', 'Backend', 'Frontend', 'Frontend', 'Frontend'],
    'Tecnología': ['Python 3.8+', 'scikit-learn', 'TensorFlow/Keras', 'PyCM', 'Flutter 3.0+', 'Dart', 'Material Design 3'],
    'Versión': ['>=3.8', '1.3.x', '2.15.x', '4.0.x', '>=3.0', '3.0.x', 'Latest'],
    'Propósito': [
        'Lenguaje principal backend',
        'Random Forest, preprocesamiento',
        'Redes neuronales profundas',
        'Matrices de confusión avanzadas',
        'Framework UI multiplataforma',
        'Lenguaje de programación',
        'Sistema de diseño visual'
    ]
})

# Guardar tablas en CSV
modelos_comparacion.to_csv('comparacion_modelos.csv', index=False, encoding='utf-8')
indicadores.to_csv('estructura_indicadores.csv', index=False, encoding='utf-8')
niveles_clasificacion.to_csv('niveles_clasificacion.csv', index=False, encoding='utf-8')
tecnologias.to_csv('tecnologias_stack.csv', index=False, encoding='utf-8')

print("✓ Tablas resumen generadas exitosamente\n")
print("=" * 70)
print("TABLA 1: COMPARACIÓN DE MODELOS")
print("=" * 70)
print(modelos_comparacion.to_string(index=False))

print("\n\n" + "=" * 70)
print("TABLA 2: ESTRUCTURA DE INDICADORES (primeros 7)")
print("=" * 70)
print(indicadores.head(7).to_string(index=False))

print("\n\n" + "=" * 70)
print("TABLA 3: NIVELES DE CLASIFICACIÓN")
print("=" * 70)
print(niveles_clasificacion.to_string(index=False))

print("\n\n" + "=" * 70)
print("TABLA 4: STACK TECNOLÓGICO")
print("=" * 70)
print(tecnologias.to_string(index=False))

print("\n\n📊 Archivos CSV generados:")
print("- comparacion_modelos.csv")
print("- estructura_indicadores.csv")
print("- niveles_clasificacion.csv")
print("- tecnologias_stack.csv")
