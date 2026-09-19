
// ARCHIVO: main.dart
// Aplicación Flutter para Análisis de Aburrimiento Grupal

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const AnalizadorAburrimiento());
}

class AnalizadorAburrimiento extends StatelessWidget {
  const AnalizadorAburrimiento({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Análisis de Aburrimiento Grupal',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        useMaterial3: true,
      ),
      home: const PantallaInicio(),
    );
  }
}

class PantallaInicio extends StatefulWidget {
  const PantallaInicio({Key? key}) : super(key: key);

  @override
  State<PantallaInicio> createState() => _PantallaInicioState();
}

class _PantallaInicioState extends State<PantallaInicio> {
  final _formKey = GlobalKey<FormState>();
  bool _analizando = false;
  Map<String, dynamic>? _resultado;

  // Controladores para los campos de entrada
  final Map<String, TextEditingController> _controladores = {
    'reflejo_sistemas_culturales': TextEditingController(text: '0.5'),
    'productividad_capitalista': TextEditingController(text: '0.5'),
    'alienacion_neoliberal': TextEditingController(text: '0.5'),
    'racismo_sistemico': TextEditingController(text: '0.5'),
    'malestar_generalizado': TextEditingController(text: '0.5'),
    'carencia_sentido': TextEditingController(text: '0.5'),
    'restriccion_libertad': TextEditingController(text: '0.5'),
    'frustracion_agencia': TextEditingController(text: '0.5'),
    'desenganche': TextEditingController(text: '0.5'),
    'alta_excitacion': TextEditingController(text: '0.5'),
    'inatencion': TextEditingController(text: '0.5'),
    'percepcion_tiempo_lenta': TextEditingController(text: '0.5'),
    'estrategias_bloqueadas': TextEditingController(text: '0.5'),
    'angustia_profunda': TextEditingController(text: '0.5'),
  };

  Future<void> _analizarGrupo() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _analizando = true;
      _resultado = null;
    });

    try {
      // Preparar datos para el análisis
      final datos = <String, double>{};
      _controladores.forEach((key, controller) {
        datos[key] = double.parse(controller.text);
      });

      // Simular análisis ML (en producción, esto llamaría a una API)
      await Future.delayed(const Duration(seconds: 2));

      // Calcular métricas
      final dominacionSociopolitica = (datos['reflejo_sistemas_culturales']! +
              datos['productividad_capitalista']! +
              datos['alienacion_neoliberal']!) /
          3;

      final manifestacionGrupal = (datos['malestar_generalizado']! +
              datos['carencia_sentido']! +
              datos['restriccion_libertad']!) /
          3;

      final potencialCritico = (datos['frustracion_agencia']! +
              datos['estrategias_bloqueadas']! +
              datos['angustia_profunda']!) /
          3;

      // Determinar nivel de aburrimiento
      final promedioGeneral =
          (dominacionSociopolitica + manifestacionGrupal + potencialCritico) / 3;

      String nivel;
      Color colorNivel;
      String recomendacion;

      if (promedioGeneral < 0.4) {
        nivel = 'BAJO';
        colorNivel = Colors.green;
        recomendacion =
            'El grupo muestra niveles saludables de compromiso. Mantener las condiciones actuales.';
      } else if (promedioGeneral < 0.7) {
        nivel = 'MEDIO';
        colorNivel = Colors.orange;
        recomendacion =
            'Atención: Se observan señales de desenganche. Considerar intervenciones preventivas.';
      } else {
        nivel = 'ALTO';
        colorNivel = Colors.red;
        recomendacion =
            'CRÍTICO: El grupo presenta aburrimiento sistémico severo. Se requiere intervención inmediata.';
      }

      setState(() {
        _resultado = {
          'nivel': nivel,
          'color': colorNivel,
          'promedio': promedioGeneral,
          'dominacion_sociopolitica': dominacionSociopolitica,
          'manifestacion_grupal': manifestacionGrupal,
          'potencial_critico': potencialCritico,
          'recomendacion': recomendacion,
        };
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error en el análisis: $e')),
      );
    } finally {
      setState(() {
        _analizando = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Análisis de Aburrimiento Grupal'),
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildSeccionInfo(),
              const SizedBox(height: 24),
              _buildSeccionEstructurasSistemicas(),
              const SizedBox(height: 24),
              _buildSeccionManifestacionesGrupales(),
              const SizedBox(height: 24),
              _buildSeccionDimensionesMedicion(),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _analizando ? null : _analizarGrupo,
                icon: _analizando
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.analytics),
                label: Text(
                    _analizando ? 'Analizando...' : 'Analizar Aburrimiento'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                ),
              ),
              if (_resultado != null) ...[
                const SizedBox(height: 32),
                _buildResultados(),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSeccionInfo() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.info_outline, color: Colors.blue),
                const SizedBox(width: 8),
                Text(
                  'Información',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Este sistema analiza el aburrimiento en grupos utilizando Machine Learning y Deep Learning. '
              'Ingrese valores entre 0.0 (mínimo) y 1.0 (máximo) para cada indicador.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSeccionEstructurasSistemicas() {
    return _buildSeccionConIndicadores(
      titulo: 'Estructuras Socio-Políticas y Económicas',
      icono: Icons.account_balance,
      color: Colors.purple,
      indicadores: [
        'reflejo_sistemas_culturales',
        'productividad_capitalista',
        'alienacion_neoliberal',
        'racismo_sistemico',
      ],
      etiquetas: {
        'reflejo_sistemas_culturales': 'Reflejo de Sistemas Culturales',
        'productividad_capitalista': 'Productividad Capitalista',
        'alienacion_neoliberal': 'Alienación Neoliberal',
        'racismo_sistemico': 'Racismo Sistémico',
      },
    );
  }

  Widget _buildSeccionManifestacionesGrupales() {
    return _buildSeccionConIndicadores(
      titulo: 'Manifestaciones Grupales',
      icono: Icons.groups,
      color: Colors.orange,
      indicadores: [
        'malestar_generalizado',
        'carencia_sentido',
        'restriccion_libertad',
        'frustracion_agencia',
      ],
      etiquetas: {
        'malestar_generalizado': 'Malestar Generalizado',
        'carencia_sentido': 'Carencia de Sentido',
        'restriccion_libertad': 'Restricción de Libertad',
        'frustracion_agencia': 'Frustración de Agencia',
      },
    );
  }

  Widget _buildSeccionDimensionesMedicion() {
    return _buildSeccionConIndicadores(
      titulo: 'Dimensiones de Medición',
      icono: Icons.assessment,
      color: Colors.teal,
      indicadores: [
        'desenganche',
        'alta_excitacion',
        'inatencion',
        'percepcion_tiempo_lenta',
        'estrategias_bloqueadas',
        'angustia_profunda',
      ],
      etiquetas: {
        'desenganche': 'Desenganche',
        'alta_excitacion': 'Alta Excitación',
        'inatencion': 'Inatención',
        'percepcion_tiempo_lenta': 'Percepción de Tiempo Lenta',
        'estrategias_bloqueadas': 'Estrategias Bloqueadas',
        'angustia_profunda': 'Angustia Profunda',
      },
    );
  }

  Widget _buildSeccionConIndicadores({
    required String titulo,
    required IconData icono,
    required Color color,
    required List<String> indicadores,
    required Map<String, String> etiquetas,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icono, color: color),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    titulo,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ...indicadores.map((indicador) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildCampoNumerico(
                    etiquetas[indicador]!,
                    _controladores[indicador]!,
                  ),
                )),
          ],
        ),
      ),
    );
  }

  Widget _buildCampoNumerico(String label, TextEditingController controller) {
    return TextFormField(
      controller: controller,
      keyboardType: TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(
        labelText: label,
        border: OutlineInputBorder(),
        suffixIcon: Icon(Icons.edit),
      ),
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Campo requerido';
        }
        final numero = double.tryParse(value);
        if (numero == null || numero < 0 || numero > 1) {
          return 'Ingrese un valor entre 0.0 y 1.0';
        }
        return null;
      },
    );
  }

  Widget _buildResultados() {
    final resultado = _resultado!;
    final nivel = resultado['nivel'] as String;
    final color = resultado['color'] as Color;
    final promedio = resultado['promedio'] as double;
    final dominacion = resultado['dominacion_sociopolitica'] as double;
    final manifestacion = resultado['manifestacion_grupal'] as double;
    final potencial = resultado['potencial_critico'] as double;
    final recomendacion = resultado['recomendacion'] as String;

    return Card(
      color: color.withOpacity(0.1),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.warning_amber, color: color, size: 32),
                const SizedBox(width: 12),
                Text(
                  'Nivel de Aburrimiento: $nivel',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        color: color,
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildIndicadorResultado(
              'Promedio General',
              promedio,
              color,
            ),
            const Divider(height: 24),
            _buildIndicadorResultado(
              'Dominación Sociopolítica',
              dominacion,
              Colors.purple,
            ),
            _buildIndicadorResultado(
              'Manifestación Grupal',
              manifestacion,
              Colors.orange,
            ),
            _buildIndicadorResultado(
              'Potencial Crítico',
              potencial,
              Colors.teal,
            ),
            const Divider(height: 24),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.lightbulb_outline, color: Colors.blue),
                      const SizedBox(width: 8),
                      Text(
                        'Recomendación',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(recomendacion),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIndicadorResultado(String label, double valor, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                label,
                style: TextStyle(fontWeight: FontWeight.w500),
              ),
              Text(
                '${(valor * 100).toStringAsFixed(1)}%',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(
            value: valor,
            backgroundColor: Colors.grey[300],
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controladores.values.forEach((controller) => controller.dispose());
    super.dispose();
  }
}
