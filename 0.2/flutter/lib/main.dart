import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;

const String apiBaseUrl = 'http://127.0.0.1:8000';

void main() {
  runApp(const BoredomApp());
}

class BoredomApp extends StatelessWidget {
  const BoredomApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Boredom Analyzer',
      debugShowCheckedModeBanner: false,
      theme: _buildTheme(),
      home: const BoredomHome(),
    );
  }
}

ThemeData _buildTheme() {
  const seed = Color(0xFF115C4A);
  final colorScheme = ColorScheme.fromSeed(
    seedColor: seed,
    brightness: Brightness.light,
  );

  final baseText = GoogleFonts.spaceGroteskTextTheme();
  final bodyText = GoogleFonts.ibmPlexSansTextTheme();

  return ThemeData(
    colorScheme: colorScheme,
    useMaterial3: true,
    textTheme: baseText.copyWith(
      bodyLarge: bodyText.bodyLarge,
      bodyMedium: bodyText.bodyMedium,
      bodySmall: bodyText.bodySmall,
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: const Color(0xFFF5F1EA),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(14),
        borderSide: BorderSide.none,
      ),
      hintStyle: const TextStyle(color: Color(0xFF8A7E70)),
    ),
  );
}

class BoredomHome extends StatefulWidget {
  const BoredomHome({super.key});

  @override
  State<BoredomHome> createState() => _BoredomHomeState();
}

class _BoredomHomeState extends State<BoredomHome> {
  final _formKey = GlobalKey<FormState>();
  late final Map<String, TextEditingController> _controllers;
  String? _nivel;
  String? _error;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _controllers = {
      for (final field in indicatorFields)
        field.key: TextEditingController(text: '0.5')
    };
  }

  @override
  void dispose() {
    for (final controller in _controllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _loading = true;
      _nivel = null;
      _error = null;
    });

    try {
      final payload = <String, double>{};
      for (final entry in _controllers.entries) {
        payload[entry.key] = double.parse(entry.value.text);
      }

      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/analyze'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'datos': payload}),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode != 200) {
        throw Exception('API error: ${response.statusCode}');
      }

      final decoded = jsonDecode(response.body) as Map<String, dynamic>;
      setState(() {
        _nivel = (decoded['nivel'] as String).toUpperCase();
      });
    } catch (error) {
      setState(() {
        _error = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFFF2ECE2), Color(0xFFE3E9E1)],
        ),
      ),
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: SafeArea(
          child: LayoutBuilder(
            builder: (context, constraints) {
              final maxWidth = constraints.maxWidth;
              final formWidth = maxWidth > 900 ? 840.0 : maxWidth;

              return SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: Center(
                  child: ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: formWidth),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildHeader(context),
                        const SizedBox(height: 24),
                        _buildForm(context, maxWidth > 720),
                        const SizedBox(height: 24),
                        _buildFooter(context),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Boredom Analyzer',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: const Color(0xFF1B2B2A),
              ),
        ),
        const SizedBox(height: 12),
        Text(
          'Measure group boredom with a lightweight ML demo. Enter values between 0 and 1, then send them to the API.',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: const Color(0xFF47514F),
              ),
        ),
        const SizedBox(height: 8),
        Text(
          'API base: $apiBaseUrl',
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: const Color(0xFF7A6F61),
              ),
        ),
      ],
    );
  }

  Widget _buildForm(BuildContext context, bool isWide) {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          for (final group in indicatorGroups) ...[
            _buildGroupCard(context, group, isWide),
            const SizedBox(height: 20),
          ],
          _buildActionArea(context),
          const SizedBox(height: 20),
          _buildResultArea(context),
        ],
      ),
    );
  }

  Widget _buildGroupCard(
    BuildContext context,
    IndicatorGroup group,
    bool isWide,
  ) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.92),
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFFBAC2BA).withOpacity(0.4),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: group.accent.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(group.icon, color: group.accent),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  group.title,
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        color: const Color(0xFF1E2D2A),
                        fontWeight: FontWeight.w600,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 16,
            runSpacing: 14,
            children: [
              for (final field in group.fields)
                SizedBox(
                  width: isWide ? 250 : double.infinity,
                  child: _buildField(field),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildField(IndicatorField field) {
    return TextFormField(
      controller: _controllers[field.key],
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(
        labelText: field.label,
        hintText: '0.0 to 1.0',
      ),
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Required';
        }
        final number = double.tryParse(value);
        if (number == null || number < 0 || number > 1) {
          return 'Use 0 to 1';
        }
        return null;
      },
    );
  }

  Widget _buildActionArea(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: _loading ? null : _submit,
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              backgroundColor: const Color(0xFF115C4A),
              foregroundColor: Colors.white,
            ),
            child: _loading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : const Text('Run Analysis'),
          ),
        ),
      ],
    );
  }

  Widget _buildResultArea(BuildContext context) {
    final nivel = _nivel;
    final error = _error;
    final accent = _nivelColor(nivel);

    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 300),
      child: nivel == null && error == null
          ? const SizedBox.shrink()
          : Container(
              key: ValueKey(nivel ?? error),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.9),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: accent.withOpacity(0.3)),
              ),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: accent.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(
                      error == null ? Icons.insights : Icons.warning_amber,
                      color: accent,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      error ?? 'Predicted level: $nivel',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: const Color(0xFF25312E),
                          ),
                    ),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _buildFooter(BuildContext context) {
    return Text(
      'Tip: use 10.0.2.2 for Android emulator, or your machine IP on a phone.',
      style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: const Color(0xFF7A6F61),
          ),
    );
  }

  Color _nivelColor(String? nivel) {
    switch (nivel) {
      case 'ALTO':
        return const Color(0xFFC1442E);
      case 'MEDIO':
        return const Color(0xFFD08B24);
      case 'BAJO':
        return const Color(0xFF2E7D32);
      default:
        return const Color(0xFF145A57);
    }
  }
}

class IndicatorField {
  const IndicatorField(this.key, this.label);

  final String key;
  final String label;
}

class IndicatorGroup {
  const IndicatorGroup(this.title, this.icon, this.accent, this.fields);

  final String title;
  final IconData icon;
  final Color accent;
  final List<IndicatorField> fields;
}

const List<IndicatorGroup> indicatorGroups = [
  IndicatorGroup(
    'Systemic structures',
    Icons.account_balance,
    Color(0xFF2B5C6C),
    [
      IndicatorField('reflejo_sistemas_culturales', 'Cultural systems reflection'),
      IndicatorField('productividad_capitalista', 'Capitalist productivity'),
      IndicatorField('alienacion_neoliberal', 'Neoliberal alienation'),
      IndicatorField('racismo_sistemico', 'Systemic racism'),
    ],
  ),
  IndicatorGroup(
    'Group signals',
    Icons.groups,
    Color(0xFFC76E39),
    [
      IndicatorField('malestar_generalizado', 'General discomfort'),
      IndicatorField('carencia_sentido', 'Lack of meaning'),
      IndicatorField('restriccion_libertad', 'Restricted freedom'),
      IndicatorField('frustracion_agencia', 'Agency frustration'),
    ],
  ),
  IndicatorGroup(
    'Measurement dimensions',
    Icons.analytics,
    Color(0xFF2F7665),
    [
      IndicatorField('desenganche', 'Disengagement'),
      IndicatorField('alta_excitacion', 'High arousal'),
      IndicatorField('inatencion', 'Inattention'),
      IndicatorField('percepcion_tiempo_lenta', 'Slow time perception'),
      IndicatorField('estrategias_bloqueadas', 'Blocked strategies'),
      IndicatorField('angustia_profunda', 'Deep distress'),
    ],
  ),
];

const List<IndicatorField> indicatorFields = [
  for (final group in indicatorGroups) ...group.fields,
];
