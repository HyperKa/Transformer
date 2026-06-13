import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib

# Настройка для открытия окон в IDEA/Windows
matplotlib.use('TkAgg')

# 1. Загрузка данных
class_names = ['Safe', 'Leak', 'Exc.H', 'XXE', 'NPE', 'RCE', 'SQLi', 'Unsafe API']
preds = pd.read_csv('predictions.csv')

y_true = preds['true_label'].astype(int)
y_probs = preds.iloc[:, 1:].values
y_pred = np.argmax(y_probs, axis=1)
y_true_bin = label_binarize(y_true, classes=range(8))

# Печать отчета в консоль
print("\n" + "="*30)
print("ИТОГОВЫЙ ОТЧЕТ")
print("="*30)
print(classification_report(y_true, y_pred, target_names=class_names))

# --- ГРАФИК 1: ROC-AUC ---
plt.figure("Рисунок 1 - ROC-кривые", figsize=(10, 8))
for i in range(8):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Кривые ROC-AUC по классам')
plt.legend(loc="lower right", fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_auc_final.png', dpi=300)
print("Отображение ROC-AUC. ЗАКРОЙТЕ ОКНО, ЧТОБЫ УВИДЕТЬ ВТОРОЙ ГРАФИК...")
plt.show() # Остановка до закрытия окна

# --- ГРАФИК 2: PR-AUC (Precision-Recall) ---
plt.figure("Рисунок 2 - PR-кривые", figsize=(10, 8))
for i in range(8):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
    pr_auc = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (PR-AUC = {pr_auc:.2f})')

plt.xlabel('Recall (Полнота)')
plt.ylabel('Precision (Точность)')
plt.title('Кривые Precision-Recall (PR-AUC) по классам')
plt.legend(loc="lower left", fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_auc_final.png', dpi=300)
print("Отображение PR-AUC завершено.")
plt.show()