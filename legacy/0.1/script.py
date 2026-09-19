
# Crear estructura de datos para el análisis de aburrimiento en grupos
# Basado en los elementos teóricos proporcionados

import json

# Definir las categorías y elementos estructurales del diagnóstico de aburrimiento grupal
estructura_diagnostico = {
    "estructuras_sistemicas": {
        "dominacion_sociopolitica_economica": [
            "reflejo_sistemas_culturales_economicos_politicos",
            "vida_racionalizada_productividad_capitalista",
            "aburrimiento_como_fallo_individual_neoliberal",
            "alienacion_comodificacion_neoliberalismo"
        ],
        "desigualdad_marginacion_racial": [
            "aburrimiento_racializado_fenomeno_sociopolitico",
            "racismo_sistemico_desigualdades_sociales",
            "condiciones_sistemicas_especificas_grupo",
            "perpetuacion_desigualdades_sociales"
        ],
        "imposicion_normas_apoliticas": [
            "marco_problema_individual_apolitico",
            "ocultacion_conexiones_sistemicas",
            "alineacion_demandas_neoliberales",
            "mecanismo_disciplinario_cotidiano"
        ],
        "dominio_tiempo_homogeneo": [
            "temporalidad_lineal_abstracta_capitalismo",
            "tiempo_reloj_produccion",
            "tiempo_ocio_trabajo_tediosos",
            "logica_capitalismo_racional"
        ]
    },
    
    "manifestaciones_grupales": {
        "padecimiento_generalizado": [
            "malestar_generalizado",
            "desolacion_espectacular",
            "aburrimiento_colectivo_cronico",
            "enraizado_estructuras_sociales"
        ],
        "carencia_sentido": [
            "percepcion_falta_significado",
            "falta_proposito_colectivo",
            "caracteristica_socialmente_inducida",
            "insatisfaccion_trato_identidad"
        ],
        "restriccion_falta_libertad": [
            "sensacion_restriccion_confinamiento",
            "opresion_sistemica_discriminacion",
            "frustracion_agencia",
            "incapacidad_cambiar_situacion"
        ],
        "respuestas_escape_compensacion": [
            "busqueda_estimulacion",
            "entretenimiento_masas",
            "reafirmacion_personalidad_excentricidades",
            "busqueda_desviado_peligroso"
        ],
        "potencial_negatividad_explosion": [
            "aburrimiento_disfuncional_cronico",
            "estrategias_escape_bloqueadas",
            "angustia_profunda",
            "explosiones_revolucionarias_nihilismo"
        ]
    },
    
    "dimensiones_medicion": {
        "estado_aburrimiento": [
            "desenganche",
            "alta_excitacion",
            "baja_excitacion",
            "inatencion",
            "percepcion_tiempo"
        ],
        "factores_contextuales": [
            "numero_replicas_grupo",
            "tiempo_transcurrido",
            "sentimiento_promedio_grupo",
            "auto_replicas_individuos"
        ],
        "metricas_agencia": [
            "capacidad_accion_efectiva",
            "autonomia_individual_colectiva",
            "realizacion_objetivos",
            "expresion_habilidades_deseos"
        ]
    }
}

# Guardar estructura
with open('estructura_aburrimiento.json', 'w', encoding='utf-8') as f:
    json.dump(estructura_diagnostico, f, ensure_ascii=False, indent=2)

print("✓ Estructura de diagnóstico de aburrimiento creada")
print(f"Total de categorías principales: {len(estructura_diagnostico)}")
print(f"- Estructuras sistémicas: {len(estructura_diagnostico['estructuras_sistemicas'])} subcategorías")
print(f"- Manifestaciones grupales: {len(estructura_diagnostico['manifestaciones_grupales'])} subcategorías")
print(f"- Dimensiones de medición: {len(estructura_diagnostico['dimensiones_medicion'])} subcategorías")
