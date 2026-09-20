// Generated from assets/schema.json. Do not hand-edit this file.
// Regenerate with: make gen-types
export const SCHEMA_VERSION = "1.0.0" as const;
export const VALUE_RANGE = {
  min: 0,
  max: 1,
} as const;

export interface Labels {
  es: string;
  en?: string;
}

export interface Group {
  id: string;
  order: number;
  labels: Labels;
}

export interface Indicator {
  id: string;
  order: number;
  labels: Labels;
  groupId: string;
  theoreticalBasis: string;
}

export interface Level {
  id: string;
  order: number;
  labels: Labels;
  color: string;
  interpretation: string;
  recommendedAction: string;
  range: { min: number; max: number };
}

export const INDICATOR_IDS = [
  "reflejo_sistemas_culturales",
  "productividad_capitalista",
  "alienacion_neoliberal",
  "racismo_sistemico",
  "malestar_generalizado",
  "carencia_de_sentido",
  "restriccion_de_libertad",
  "frustracion_de_agencia",
  "desenganche",
  "alta_excitacion",
  "inatencion",
  "percepcion_tiempo_lenta",
  "estrategias_bloqueadas",
  "angustia_profunda",
] as const;
export type IndicatorId = (typeof INDICATOR_IDS)[number];

export const GROUPS: readonly Group[] = [
  { id: "estructuras_sistemicas", order: 1, labels: {"es": "Estructuras Sistémicas", "en": "Systemic structures"} },
  { id: "manifestaciones_grupales", order: 2, labels: {"es": "Manifestaciones Grupales", "en": "Group signals"} },
  { id: "dimensiones_medicion", order: 3, labels: {"es": "Dimensiones Medición", "en": "Measurement dimensions"} },
] as const;

export const INDICATORS: readonly Indicator[] = [
  { id: "reflejo_sistemas_culturales", groupId: "estructuras_sistemicas", order: 1, labels: {"es": "Reflejo sistemas culturales", "en": "Cultural systems reflection"}, theoreticalBasis: "Crítica artística al Fordismo" },
  { id: "productividad_capitalista", groupId: "estructuras_sistemicas", order: 2, labels: {"es": "Productividad capitalista", "en": "Capitalist productivity"}, theoreticalBasis: "Lógica capitalista racional" },
  { id: "alienacion_neoliberal", groupId: "estructuras_sistemicas", order: 3, labels: {"es": "Alienación neoliberal", "en": "Neoliberal alienation"}, theoreticalBasis: "Individualización neoliberal" },
  { id: "racismo_sistemico", groupId: "estructuras_sistemicas", order: 4, labels: {"es": "Racismo sistémico", "en": "Systemic racism"}, theoreticalBasis: "Desigualdad racial sistémica" },
  { id: "malestar_generalizado", groupId: "manifestaciones_grupales", order: 5, labels: {"es": "Malestar generalizado", "en": "General discomfort"}, theoreticalBasis: "Padecimiento colectivo crónico" },
  { id: "carencia_de_sentido", groupId: "manifestaciones_grupales", order: 6, labels: {"es": "Carencia de sentido", "en": "Lack of meaning"}, theoreticalBasis: "Meaninglessness social" },
  { id: "restriccion_de_libertad", groupId: "manifestaciones_grupales", order: 7, labels: {"es": "Restricción de libertad", "en": "Restricted freedom"}, theoreticalBasis: "Opresión sistémica" },
  { id: "frustracion_de_agencia", groupId: "manifestaciones_grupales", order: 8, labels: {"es": "Frustración de agencia", "en": "Agency frustration"}, theoreticalBasis: "Falta de agencia efectiva" },
  { id: "desenganche", groupId: "dimensiones_medicion", order: 9, labels: {"es": "Desenganche", "en": "Disengagement"}, theoreticalBasis: "Boredom Proneness Scale" },
  { id: "alta_excitacion", groupId: "dimensiones_medicion", order: 10, labels: {"es": "Alta excitación", "en": "High arousal"}, theoreticalBasis: "MSBS - Alta arousal" },
  { id: "inatencion", groupId: "dimensiones_medicion", order: 11, labels: {"es": "Inatención", "en": "Inattention"}, theoreticalBasis: "MSBS - Atención" },
  { id: "percepcion_tiempo_lenta", groupId: "dimensiones_medicion", order: 12, labels: {"es": "Percepción tiempo lenta", "en": "Slow time perception"}, theoreticalBasis: "MSBS - Percepción temporal" },
  { id: "estrategias_bloqueadas", groupId: "dimensiones_medicion", order: 13, labels: {"es": "Estrategias bloqueadas", "en": "Blocked strategies"}, theoreticalBasis: "Respuestas bloqueadas" },
  { id: "angustia_profunda", groupId: "dimensiones_medicion", order: 14, labels: {"es": "Angustia profunda", "en": "Deep distress"}, theoreticalBasis: "Potencial revolucionario" },
] as const;

export const LEVELS: readonly Level[] = [
  { id: "bajo", order: 1, labels: {"es": "BAJO"}, color: "Verde", interpretation: "Grupo saludable con buen compromiso", recommendedAction: "Mantener condiciones actuales", range: { min: 0, max: 0.4 } },
  { id: "medio", order: 2, labels: {"es": "MEDIO"}, color: "Naranja", interpretation: "Señales de desenganche presentes", recommendedAction: "Intervenciones preventivas necesarias", range: { min: 0.4, max: 0.7 } },
  { id: "alto", order: 3, labels: {"es": "ALTO"}, color: "Rojo", interpretation: "Aburrimiento sistémico severo", recommendedAction: "Intervención inmediata requerida", range: { min: 0.7, max: 1 } },
] as const;
export type LevelId = (typeof LEVELS)[number]["id"];

export const REJECTED_FIELD_NAMES = {"carencia_sentido": "carencia_de_sentido", "restriccion_libertad": "restriccion_de_libertad", "frustracion_agencia": "frustracion_de_agencia"} as const;

export interface AnalyzeRequest {
  datos: Record<IndicatorId, number>;
}

export interface AnalyzeResponse {
  nivel: LevelId;
}
