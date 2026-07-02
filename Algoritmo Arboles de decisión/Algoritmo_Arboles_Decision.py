import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


def cargar_y_preprocesar(path: str) -> tuple[np.ndarray, np.ndarray]:
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


def aplicar_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print('Distribución original de clases:', Counter(y))

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print('Distribución después de SMOTE:', Counter(y_res))
    return X_res, y_res


def entrenar_arbol_decision(
    X_train: np.ndarray,
    y_train: np.ndarray,
    criterion: str = 'gini',
    max_depth: int | None = 5,
) -> DecisionTreeClassifier:
    modelo = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=42,
    )

    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(
    modelo: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> None:
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)

    print('\nRESULTADOS DEL MODELO')
    print('-' * 50)
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'F1-Score: {f1:.4f}')

    print('\nReporte de clasificación:')
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Árbol de Decisión')
    plt.tight_layout()
    plt.show()


def graficar_arbol_decision(
    modelo: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    max_depth: int | None = 5,
) -> None:
    plt.figure(figsize=(20, 10))

    plot_tree(
        modelo,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=9,
    )

    plt.title('Árbol de Decisión')
    plt.tight_layout()
    plt.show()


def evaluacion_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    n_splits: int = 5,
) -> None:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    sensitivities = []
    specificities = []

    fold = 1

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        modelo = DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            random_state=42,
        )

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)

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

            sensitivities.append(sens)
            specificities.append(spec)

        print(f'\nFold {fold}')
        print('-' * 50)
        print(f'Accuracy: {acc:.4f}')
        print(classification_report(y_val, y_pred, target_names=class_names))

        fold += 1

    print('\nPROMEDIOS DE VALIDACIÓN CRUZADA')
    print('-' * 50)
    print(f'Exactitud promedio: {np.mean(accuracies):.4f}')
    print(f'Sensibilidad promedio: {np.mean(sensitivities):.4f}')
    print(f'Especificidad promedio: {np.mean(specificities):.4f}')


def main() -> None:
    archivo = 'car_ready.csv'

    feature_names = ['buying', 'maint', 'doors',
                     'persons', 'lug_boot', 'safety']
    class_names = ['unacc', 'acc', 'good', 'vgood']

    X, y = cargar_y_preprocesar(archivo)
    X_res, y_res = aplicar_smote(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res,
    )

    modelo_arbol = entrenar_arbol_decision(
        X_train,
        y_train,
        criterion='gini',
        max_depth=5,
    )

    evaluar_modelo(
        modelo_arbol,
        X_test,
        y_test,
        class_names,
    )

    graficar_arbol_decision(
        modelo_arbol,
        feature_names,
        class_names,
        max_depth=5,
    )

    evaluacion_cross_validation(
        X_res,
        y_res,
        class_names,
        n_splits=5,
    )


if __name__ == '__main__':
    main()
