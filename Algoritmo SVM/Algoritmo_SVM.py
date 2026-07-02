import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def cargar_y_preprocesar(path: str) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(path)

    mapeo_buying = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
    mapeo_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
    mapeo_doors = {'2': 2, '3': 3, '4': 4, '5more': 5}
    mapeo_persons = {'2': 2, '4': 4, 'more': 5}
    mapeo_lug_boot = {'small': 0, 'med': 1, 'big': 2}
    mapeo_safety = {'low': 0, 'med': 1, 'high': 2}
    mapeo_class = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

    df['buying'] = df['buying'].map(mapeo_buying)
    df['maint'] = df['maint'].map(mapeo_maint)
    df['doors'] = df['doors'].map(mapeo_doors)
    df['persons'] = df['persons'].map(mapeo_persons)
    df['lug_boot'] = df['lug_boot'].map(mapeo_lug_boot)
    df['safety'] = df['safety'].map(mapeo_safety)
    df['class'] = df['class'].map(mapeo_class)

    X = df.drop('class', axis=1).to_numpy()
    y = df['class'].to_numpy()

    return X, y


def aplicar_smote(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    print('Distribución original de clases:', Counter(y))
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    print('Distribución después de SMOTE:', Counter(y_res))
    return X_res, y_res


def seleccionar_caracteristicas(X: np.ndarray, y: np.ndarray, k: int = 6) -> np.ndarray:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    np.set_printoptions(precision=3)
    print('Puntuaciones de características:', selector.scores_)
    return X_new


def entrenar_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    svm_clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
    svm_clf.fit(X_train, y_train)
    return svm_clf


def evaluar_modelo(modelo: SVC, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"F1-Score: {f1}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        plt.close()


def graficar_curvas_roc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    ovr_clf = OneVsRestClassifier(
        SVC(kernel='rbf', gamma='scale', probability=True))
    ovr_clf.fit(X_train, y_train)
    y_score = ovr_clf.predict_proba(X_test)
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print('AUC por clase:', roc_auc)

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f'ROC clase {i} (AUC = {roc_auc[i]:0.2f})'
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC Multiclase (One‑vs‑Rest)')
    plt.legend(loc="lower right")
    try:
        plt.show()
    except Exception:
        plt.close()


def evaluacion_cross_validation(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
    skf = StratifiedKFold(n_splits=n_splits)
    accuracies: list[float] = []
    all_sensitivities: list[float] = []
    all_specificities: list[float] = []

    fold = 1
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
        cm = confusion_matrix(y_val, y_pred)

        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            all_sensitivities.append(sens)
            all_specificities.append(spec)

        print(f'\nFold {fold}')
        print('-' * 50)
        print(f'Accuracy: {round(acc, 2)}')
        # Mostrar reporte de clasificación
        print(classification_report(y_val, y_pred))
        fold += 1

    print('-' * 50)
    print(f'Exactitud promedio: {np.mean(accuracies):.2f}')
    print(f'Sensibilidad promedio: {np.mean(all_sensitivities):.2f}')
    print(f'Especificidad promedio: {np.mean(all_specificities):.2f}')


def main():
    archivo = 'car_ready.csv'

    X, y = cargar_y_preprocesar(archivo)

    X_res, y_res = aplicar_smote(X, y)

    X_sel = seleccionar_caracteristicas(X_res, y_res, k=6)

    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_res, test_size=0.2, random_state=RANDOM_STATE
    )

    svm_model = entrenar_svm(X_train, y_train)

    evaluar_modelo(svm_model, X_test, y_test)

    graficar_curvas_roc(X_train, y_train, X_test, y_test)

    evaluacion_cross_validation(X_sel, y_res, n_splits=5)


if __name__ == '__main__':
    main()
