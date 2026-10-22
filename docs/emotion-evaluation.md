# Emotion evaluation

For multi-label emotions, DaMeme reports micro F1, macro F1, Hamming loss, and precision/recall/F1 for every emotion. For sentiment it reports accuracy, macro F1, and a three-class confusion matrix. Thresholds are selected only from the validation split and then applied unchanged to the held-out test split. JSON and Markdown reports are written with `emotion_cli evaluate --output-dir`.

The executed bounded research sample has 156 train, 26 validation, and 18 test records. It is still a functional research-sample evaluation, not a benchmark: several held-out label supports are as small as one. Do not make scientific performance claims from these results.
