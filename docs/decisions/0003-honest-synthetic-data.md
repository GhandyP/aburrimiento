# 0003 — Keep synthetic data, measure it honestly, and open a path to real observations

## Context

The `0.1` prototype published accuracy figures — Random Forest 88–92 %, neural network 85–90 % — in
`comparacion_modelos.csv` and its documentation. None of them was measured against a held-out set. The
generator draws each label from a class-conditional distribution and the model then learns that same
distribution, so any accuracy on this data describes the self-consistency of the generator and nothing
about boredom in people.

## Decision

Keep the synthetic generator, make its assumptions explicit data rather than buried literals, and
measure honestly: a stratified train/validation/test split, stratified k-fold cross-validation, and the
metrics that were previously asserted — accuracy, Cohen's kappa, macro and weighted F1, per-class
precision/recall/F1/support, and the confusion matrix. Every published number comes from `make train`
and lands in `reports/`. The scope limitation is stated in the generator's docstring, in the report
header, in the README and here. Separately, `POST /samples` opens a capture path for real observations,
storing the 14 values, the predicted level, an optional observed level and a timestamp, and nothing
else: no identifiers, no names, no free text, no IP addresses, no device information.

## Consequences

- Measured on 3000 samples with seed 42: random forest 0.9783 accuracy, logistic regression 0.9800, and
  a most-frequent dummy floor at 0.3383. The linear baseline beats the 120-tree forest, so the task is
  largely linearly separable inside the generator.
- The two features carrying the most aggressive generator modifiers are the least used by the model —
  `racismo_sistemico` 0.0142 and `alta_excitacion` 0.0137 against `frustracion_de_agencia` 0.1056 —
  because a modifier applied to every class equally compresses a feature without separating classes.
- No claim of real-world validity appears anywhere in the repository, and the README says so explicitly.

## What would change this decision

Labelled real observations. The capture path exists so that collecting them does not require a redesign,
but nothing here should be read as evidence about people until they arrive.
