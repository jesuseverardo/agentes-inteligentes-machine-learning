"""
Integrantes: Díaz Alvarado Jesús Everado
             Galván Quiroz Jesús David
Grupo: 8CM12
"""

from __future__ import annotations
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.exceptions import UndefinedMetricWarning
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import warnings
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")


RANDOM_STATE = 42


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def save_plot(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def load_and_preprocess(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo CSV en: {csv_path}")

    df = pd.read_csv(csv_path)

    print("Primeras filas del conjunto de datos original:")
    print(df.head())
    print()

    expected_columns = {
        "Student_ID",
        "study_hours",
        "attendance_percentage",
        "parental_education",
        "test_preparation",
        "extracurricular_participation",
        "past_grade",
        "performance_category",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en el CSV: {sorted(missing)}")

    df = df.drop(columns=["Student_ID"]).copy()

    performance_map = {"Low": 0, "Medium": 1, "High": 2}
    parental_map = {"HighSchool": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    test_prep_map = {"None": 0, "Completed": 1}
    extra_map = {"No": 0, "Yes": 1}

    cat_cols = [
        "performance_category",
        "parental_education",
        "test_preparation",
        "extracurricular_participation",
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    df["test_preparation"] = df["test_preparation"].replace(
        {"": "None", "nan": "None", "NaN": "None"}
    )

    df["performance_category"] = df["performance_category"].map(
        performance_map)
    df["parental_education"] = df["parental_education"].map(parental_map)
    df["test_preparation"] = df["test_preparation"].map(test_prep_map)
    df["extracurricular_participation"] = df["extracurricular_participation"].map(
        extra_map)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        bad_rows = df[df.isna().any(axis=1)]
        raise ValueError(
            "Se encontraron valores faltantes o categorias no reconocidas despues "
            "del preprocesamiento. Filas problematicas:\n"
            f"{bad_rows.head()}"
        )

    X_df = df.drop(columns=["performance_category"]).astype(float)
    y = df["performance_category"].astype(int).to_numpy()

    print("Distribucion original de clases:", Counter(y))
    print()
    return X_df, y


def build_pipeline(k: int, metric: str, n_features: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=0.0)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("select", SelectKBest(score_func=f_classif, k=n_features)),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric)),
        ]
    )


def summarize_feature_scores(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_features: int,
) -> tuple[pd.Series, list[str]]:
    variance = VarianceThreshold(threshold=0.0)
    X_nonconst = variance.fit_transform(X_train)
    kept_columns = X_train.columns[variance.get_support()].tolist()
    removed_columns = X_train.columns[~variance.get_support()].tolist()

    X_nonconst_df = pd.DataFrame(X_nonconst, columns=kept_columns)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_nonconst_df, y_train)
    X_bal_df = pd.DataFrame(X_bal, columns=kept_columns)

    effective_k = min(n_features, X_bal_df.shape[1])
    selector = SelectKBest(score_func=f_classif, k=effective_k)
    selector.fit(X_bal_df, y_bal)

    scores = pd.Series(selector.scores_, index=kept_columns).fillna(
        0.0).sort_values(ascending=False)
    selected = X_bal_df.columns[selector.get_support()].tolist()

    if removed_columns:
        print("Columnas constantes eliminadas:", removed_columns)

    print("Puntuaciones de las caracteristicas (calculadas solo con train balanceado):")
    print(scores.round(3))
    print("Caracteristicas seleccionadas:", selected)
    print()

    return scores, selected


def compute_multiclass_specificity_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    specificities = []
    fprs = []

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        specificities.append(specificity)
        fprs.append(fpr)

    return {
        "specificity_macro": float(np.mean(specificities)),
        "fpr_macro": float(np.mean(fprs)),
    }


def tune_knn_holdout(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_features: int,
    output_dir: Path,
) -> dict:
    max_k = min(35, len(X_train))
    ks = list(range(1, max_k + 1))
    metrics = ["euclidean", "manhattan"]

    results = []
    error_by_metric = {}

    for metric in metrics:
        train_errors = []
        test_errors = []

        for k in ks:
            model = build_pipeline(k=k, metric=metric, n_features=n_features)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_errors.append(1 - accuracy_score(y_train, train_pred))
            test_errors.append(1 - accuracy_score(y_test, test_pred))

            results.append(
                {
                    "k": k,
                    "metric": metric,
                    "train_error": train_errors[-1],
                    "test_error": test_errors[-1],
                    "test_accuracy": accuracy_score(y_test, test_pred),
                    "test_f1_weighted": f1_score(y_test, test_pred, average="weighted", zero_division=0),
                }
            )

        error_by_metric[metric] = (train_errors, test_errors)

        plt.figure(figsize=(9, 4))
        plt.plot(ks, train_errors, marker="o", label="Error entrenamiento")
        plt.plot(ks, test_errors, marker="s", label="Error prueba")
        plt.xlabel("Numero de vecinos (K)")
        plt.ylabel("Error")
        plt.title(f"Busqueda de K con distancia {metric}")
        plt.legend()
        save_plot(output_dir, f"knn_error_curve_{metric}.png")

    results_df = pd.DataFrame(results).sort_values(
        by=["test_accuracy", "test_f1_weighted"], ascending=False
    )
    best = results_df.iloc[0].to_dict()

    best_k = int(best["k"])
    best_metric = str(best["metric"])

    print("Top 10 combinaciones K + distancia en hold-out:")
    print(results_df.head(10).to_string(index=False))
    print()

    return {
        "best_k": best_k,
        "best_metric": best_metric,
        "search_results": results_df,
    }


def evaluate_holdout(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_features: int,
    output_dir: Path,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=== HOLD-OUT 80/20 ===")
    print("Distribucion train:", Counter(y_train))
    print("Distribucion test :", Counter(y_test))
    print()

    scores, selected_columns = summarize_feature_scores(
        X_train, y_train, n_features=n_features)

    tuned = tune_knn_holdout(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_features=n_features,
        output_dir=output_dir,
    )

    best_k = tuned["best_k"]
    best_metric = tuned["best_metric"]

    print(f"Mejor valor de K segun hold-out: {best_k}")
    print(f"Mejor metrica de distancia: {best_metric}")
    print()

    model = build_pipeline(k=best_k, metric=best_metric, n_features=n_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(
        y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(
        y_test, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(
        y_test, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(
        y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    extra = compute_multiclass_specificity_fpr(y_test, y_pred)

    print("Metricas hold-out:")
    print(f"Accuracy:             {acc:.3f}")
    print(f"Precision macro:      {precision_macro:.3f}")
    print(f"Precision ponderada:  {precision_weighted:.3f}")
    print(f"Recall macro:         {recall_macro:.3f}")
    print(f"Recall ponderado:     {recall_weighted:.3f}")
    print(f"F1 macro:             {f1_macro:.3f}")
    print(f"F1 ponderado:         {f1_weighted:.3f}")
    print(f"Especificidad macro:  {extra['specificity_macro']:.3f}")
    print(f"FPR macro:            {extra['fpr_macro']:.3f}")
    print()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.title("Matriz de confusion Hold-Out")
    cm_plot = save_plot(output_dir, "knn_confusion_matrix_holdout.png")

    print("Reporte de clasificacion (hold-out):")
    print(classification_report(y_test, y_pred, zero_division=0))

    classes = sorted(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=classes)

    ovr_model = OneVsRestClassifier(build_pipeline(
        k=best_k, metric=best_metric, n_features=n_features))
    ovr_model.fit(X_train, y_train)
    probas = ovr_model.predict_proba(X_test)

    plt.figure(figsize=(8, 6))
    roc_auc_dict = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probas[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[int(cls)] = roc_auc
        plt.plot(fpr, tpr, lw=2, label=f"Clase {cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC multiclase One-vs-Rest")
    plt.legend(loc="lower right")
    roc_plot = save_plot(output_dir, "knn_roc_multiclass.png")

    print("AUC por clase (One-vs-Rest):")
    for cls, roc_auc in roc_auc_dict.items():
        print(f"Clase {cls}: {roc_auc:.3f}")
    print()

    print("Graficos guardados en:")
    print(f" - {cm_plot}")
    print(f" - {roc_plot}")
    print(f" - {output_dir / 'knn_error_curve_euclidean.png'}")
    print(f" - {output_dir / 'knn_error_curve_manhattan.png'}")
    print()

    return {
        "best_k": best_k,
        "best_metric": best_metric,
        "scores": scores,
        "selected_columns": selected_columns,
        "metrics": {
            "accuracy": acc,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "specificity_macro": extra["specificity_macro"],
            "fpr_macro": extra["fpr_macro"],
        },
        "auc_by_class": roc_auc_dict,
    }


def evaluate_cross_validation(
    X_df: pd.DataFrame,
    y: np.ndarray,
    best_k: int,
    best_metric: str,
    n_features: int,
    n_splits: int = 5,
) -> None:
    print("=== VALIDACION CRUZADA ESTRATIFICADA ===")
    print(f"n_splits={n_splits}, K={best_k}, metric={best_metric}")
    print()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)

    accs = []
    precs = []
    recalls = []
    f1s = []
    specs = []
    fprs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), start=1):
        X_train = X_df.iloc[train_idx].copy()
        X_val = X_df.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_pipeline(
            k=best_k, metric=best_metric, n_features=n_features)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(
            y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        extra = compute_multiclass_specificity_fpr(y_val, y_pred)

        accs.append(acc)
        precs.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        specs.append(extra["specificity_macro"])
        fprs.append(extra["fpr_macro"])

        print(f"Fold {fold}")
        print(f"Accuracy:            {acc:.3f}")
        print(f"Precision ponderada: {prec:.3f}")
        print(f"Recall ponderado:    {rec:.3f}")
        print(f"F1 ponderado:        {f1:.3f}")
        print(f"Especificidad macro: {extra['specificity_macro']:.3f}")
        print(f"FPR macro:           {extra['fpr_macro']:.3f}")
        print("Matriz de confusion:")
        print(confusion_matrix(y_val, y_pred))
        print("Reporte de clasificacion:")
        print(classification_report(y_val, y_pred, zero_division=0))
        print("-" * 60)

    print("Promedios en validacion cruzada:")
    print(
        f"Accuracy media:            {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(
        f"Precision ponderada media: {np.mean(precs):.3f} +/- {np.std(precs):.3f}")
    print(
        f"Recall ponderado medio:    {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}")
    print(
        f"F1 ponderado medio:        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(
        f"Especificidad macro media: {np.mean(specs):.3f} +/- {np.std(specs):.3f}")
    print(
        f"FPR macro media:           {np.mean(fprs):.3f} +/- {np.std(fprs):.3f}")
    print()


def write_summary_report(
    output_dir: Path,
    holdout_result: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "resumen_resultados_knn.txt"

    metrics = holdout_result["metrics"]
    auc_by_class = holdout_result["auc_by_class"]

    lines = [
        "RESUMEN DEL MODELO KNN",
        "======================",
        "",
        "1. Tipo de problema",
        "Clasificacion supervisada multiclase.",
        "",
        "2. Tecnicas aplicadas",
        "- Hold-Out 80/20",
        "- K-Fold Cross-Validation estratificada",
        "- SMOTE para balanceo de clases",
        "- SelectKBest para seleccion de caracteristicas",
        "- KNN con busqueda de mejor K y mejor distancia",
        "",
        f"3. Mejor configuracion hold-out",
        f"- K = {holdout_result['best_k']}",
        f"- Distancia = {holdout_result['best_metric']}",
        "",
        "4. Caracteristicas seleccionadas",
        f"- {', '.join(holdout_result['selected_columns'])}",
        "",
        "5. Metricas hold-out",
        f"- Accuracy = {metrics['accuracy']:.3f}",
        f"- Precision macro = {metrics['precision_macro']:.3f}",
        f"- Precision ponderada = {metrics['precision_weighted']:.3f}",
        f"- Recall macro = {metrics['recall_macro']:.3f}",
        f"- Recall ponderado = {metrics['recall_weighted']:.3f}",
        f"- F1 macro = {metrics['f1_macro']:.3f}",
        f"- F1 ponderado = {metrics['f1_weighted']:.3f}",
        f"- Especificidad macro = {metrics['specificity_macro']:.3f}",
        f"- FPR macro = {metrics['fpr_macro']:.3f}",
        "",
        "6. AUC por clase",
    ]
    for cls, auc_value in auc_by_class.items():
        lines.append(f"- Clase {cls}: {auc_value:.3f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    _configure_console()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "student_performance_dataset.csv"
    output_dir = base_dir / "knn_outputs"

    X_df, y = load_and_preprocess(csv_file)

    n_features = min(6, X_df.shape[1])

    holdout_result = evaluate_holdout(
        X_df=X_df,
        y=y,
        n_features=n_features,
        output_dir=output_dir,
    )

    evaluate_cross_validation(
        X_df=X_df,
        y=y,
        best_k=holdout_result["best_k"],
        best_metric=holdout_result["best_metric"],
        n_features=n_features,
        n_splits=5,
    )

    report_path = write_summary_report(output_dir, holdout_result)
    print(f"Resumen guardado en: {report_path}")


if __name__ == "__main__":
    main()
