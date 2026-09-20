# Evaluation numbers measure generator self-consistency, not boredom in the world

- Samples: `3000`; seed: `42`
- Stratified split sizes: `{'train': 1800, 'validation': 600, 'test': 600}`
- The labels are drawn from the same class-conditional distributions the model then learns, so accuracy measured here is a measure of self-consistency of the generator, not evidence about boredom in the world.
- There is no external validation anywhere in this repository.

## Random forest scaler comparison
### random_forest_scaled
- Test accuracy: `0.978333`
- Cohen's kappa: `0.967497`
- Macro F1: `0.978493`; weighted F1: `0.978333`
- CV accuracy: `[0.986111, 0.972222, 0.983333, 0.972222, 0.988889]`; mean `0.980556`, standard deviation `0.007027`
- Confusion matrix labels: `['bajo', 'medio', 'alto']`
- Per-class metrics:
  | Class | Precision | Recall | F1 | Support |
  |---|---:|---:|---:|---:|
  | bajo | 0.969849 | 0.965000 | 0.967419 | 200 |
  | medio | 0.965686 | 0.970443 | 0.968059 | 203 |
  | alto | 1.000000 | 1.000000 | 1.000000 | 197 |
- Confusion matrix: `[[193, 7, 0], [6, 197, 0], [0, 0, 197]]`
### random_forest_unscaled
- Test accuracy: `0.978333`
- Cohen's kappa: `0.967497`
- Macro F1: `0.978493`; weighted F1: `0.978333`
- CV accuracy: `[0.986111, 0.972222, 0.983333, 0.972222, 0.988889]`; mean `0.980556`, standard deviation `0.007027`
- Confusion matrix labels: `['bajo', 'medio', 'alto']`
- Per-class metrics:
  | Class | Precision | Recall | F1 | Support |
  |---|---:|---:|---:|---:|
  | bajo | 0.969849 | 0.965000 | 0.967419 | 200 |
  | medio | 0.965686 | 0.970443 | 0.968059 | 203 |
  | alto | 1.000000 | 1.000000 | 1.000000 | 197 |
- Confusion matrix: `[[193, 7, 0], [6, 197, 0], [0, 0, 197]]`
- Scaler metrics differ: **False**
- The tree is shown in scaled and unscaled forms because tree splits are expected to be invariant to feature scale; the comparison makes that assumption explicit. The linear baseline and dummy floor each have one model report.

## Model comparison
### logistic_regression
- Test accuracy: `0.980000`
- Cohen's kappa: `0.969997`
- Macro F1: `0.980146`; weighted F1: `0.979999`
- CV accuracy: `[0.988889, 0.975, 0.994444, 0.977778, 0.986111]`; mean `0.984444`, standard deviation `0.007158`
- Confusion matrix labels: `['bajo', 'medio', 'alto']`
- Per-class metrics:
  | Class | Precision | Recall | F1 | Support |
  |---|---:|---:|---:|---:|
  | bajo | 0.974747 | 0.965000 | 0.969849 | 200 |
  | medio | 0.965854 | 0.975369 | 0.970588 | 203 |
  | alto | 1.000000 | 1.000000 | 1.000000 | 197 |
- Confusion matrix: `[[193, 7, 0], [5, 198, 0], [0, 0, 197]]`
### dummy_most_frequent
- Test accuracy: `0.338333`
- Cohen's kappa: `0.000000`
- Macro F1: `0.168535`; weighted F1: `0.171063`
- CV accuracy: `[0.338889, 0.338889, 0.338889, 0.338889, 0.338889]`; mean `0.338889`, standard deviation `0.000000`
- Confusion matrix labels: `['bajo', 'medio', 'alto']`
- Per-class metrics:
  | Class | Precision | Recall | F1 | Support |
  |---|---:|---:|---:|---:|
  | bajo | 0.000000 | 0.000000 | 0.000000 | 200 |
  | medio | 0.338333 | 1.000000 | 0.505604 | 203 |
  | alto | 0.000000 | 0.000000 | 0.000000 | 197 |
- Confusion matrix: `[[0, 200, 0], [0, 203, 0], [0, 197, 0]]`

A small gap between the tree model and the linear baseline is evidence that the classification task is largely linearly separable in the generator, which says something about the generator and nothing about boredom. The dummy floor is reported with its measured accuracy above, not characterized as trivial without that measurement.

## Feature importances

- `frustracion_de_agencia`: `0.105616527376`
- `malestar_generalizado`: `0.095823659205`
- `inatencion`: `0.094270960446`
- `carencia_de_sentido`: `0.092815427806`
- `estrategias_bloqueadas`: `0.087194347527`
- `alienacion_neoliberal`: `0.078999440535`
- `percepcion_tiempo_lenta`: `0.073485534619`
- `reflejo_sistemas_culturales`: `0.072086472368`
- `desenganche`: `0.071206068193`
- `angustia_profunda`: `0.069771564034`
- `productividad_capitalista`: `0.067602447182`
- `restriccion_de_libertad`: `0.063303466655`
- `racismo_sistemico`: `0.014171029792`
- `alta_excitacion`: `0.013653054264`
