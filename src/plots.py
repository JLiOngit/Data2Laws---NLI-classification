import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def histogram_df(df):
    label = (df.groupby('label').count().sort_values(by='message', ascending=False).reset_index())
    group = (df.groupby('group').count().sort_values(by='message', ascending=False).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    # dataset - Single Label
    axes[0, 0].bar(label['label'], label['message'])
    axes[0, 0].set_title("Training dataset - Single Label Count")
    axes[0, 0].set_xlabel("GitHub Single Label")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis='x', rotation=-45)
    # dataset - Group Label
    axes[0, 1].bar(group['group'], group['message'])
    axes[0, 1].set_title("Training dataset - Group Label Count")
    axes[0, 1].set_xlabel("GitHub Group Label")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].tick_params(axis='x', rotation=-45)
    plt.tight_layout()
    plt.show()


def plot_metrics(df, label):
    displayed_labels = df[label].unique()
    label_true = df[label]
    predicted = df['predicted_' + label]
    conf_matrix = confusion_matrix(label_true, predicted, labels=displayed_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=displayed_labels)
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_true, predicted, average='macro')
    print("Macro scores")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro: {recall_macro:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(label_true, predicted, labels=displayed_labels, average=None)
    print("\nClass Scores")
    for lbl, p, r, f, s in zip(displayed_labels, precision_cls, recall_cls, f1_cls, support_cls):
        print(f"Class: {lbl}")
        print(f"Precision: {p:.4f}")
        print(f"Recall: {r:.4f}")
        print(f"F1-score: {f:.4f}")
        print(f"Support: {s}")
        print('-' * 30)