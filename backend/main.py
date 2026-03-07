from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import LinearSVC, LinearSVR


DEFAULT_FILE_PATH = r"C:\Users\Shahina\Downloads\loan_prediction_dataset.csv"
MAX_TRAIN_ROWS = 20000
MAX_CLASS_POINTS = 2000
MAX_REG_POINTS = 2000
MAX_REG_ERRORS = 5000
MAX_CORR_FEATURES = 24


class LoadRequest(BaseModel):
    file_path: str = DEFAULT_FILE_PATH
    high_cardinality_threshold: int = Field(default=50, ge=5, le=500)


class TrainRequest(BaseModel):
    mode: str = "both"  # one of: both, classification, regression
    classification_target: str | None = None
    regression_target: str | None = None
    classification_model: str = "random_forest"  # one of: random_forest, gradient_boosting, logistic_regression
    regression_model: str = "random_forest"  # one of: random_forest, gradient_boosting, linear_regression
    auto_select_best_models: bool = False


class PredictRequest(BaseModel):
    values: dict[str, Any]


class ModelStore:
    def __init__(self) -> None:
        self.raw_df: pd.DataFrame | None = None
        self.cleaned_df: pd.DataFrame | None = None
        self.dropped_columns: list[str] = []

        self.classification_target: str | None = None
        self.regression_target: str | None = None

        self.classifier_pipeline: Pipeline | None = None
        self.regressor_pipeline: Pipeline | None = None
        self.class_label_encoder: LabelEncoder | None = None
        self.class_feature_schema: list[dict[str, Any]] = []
        self.reg_feature_schema: list[dict[str, Any]] = []


store = ModelStore()

app = FastAPI(title="Customer Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file type. Use CSV or Excel.")


def _read_table_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    stream = BytesIO(content)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(stream)
    if suffix == ".csv":
        return pd.read_csv(stream)
    raise ValueError("Unsupported file type. Use CSV or Excel.")


def _clean_data(df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()
    data = data.drop_duplicates()

    datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64", "datetimetz"]).columns
    for col in datetime_cols:
        # Convert datetime to unix seconds for model compatibility.
        data[col] = pd.to_datetime(data[col], errors="coerce").astype("int64") // 10**9

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=["object", "category", "bool"]).columns

    # Normalize common placeholder tokens to missing values before imputation.
    missing_tokens = {"", "?", "na", "n/a", "null", "none", "nan"}
    for col in categorical_cols:
        normalized = data[col].astype(str).str.strip()
        data[col] = data[col].where(~normalized.str.lower().isin(missing_tokens), np.nan)

    if len(numeric_cols) > 0:
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    for col in categorical_cols:
        mode_values = data[col].mode(dropna=True)
        fallback = "NA" if mode_values.empty else str(mode_values.iloc[0])
        data[col] = data[col].fillna(fallback).astype(str)

    dropped_columns: list[str] = []
    for col in categorical_cols:
        if data[col].nunique(dropna=False) > threshold:
            dropped_columns.append(col)

    if dropped_columns:
        data = data.drop(columns=dropped_columns)

    return data, dropped_columns


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )


def _feature_schema(df_features: pd.DataFrame) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for col in df_features.columns:
        series = df_features[col]
        if pd.api.types.is_numeric_dtype(series):
            val = float(series.median()) if not series.empty else 0.0
            schema.append({"name": col, "type": "number", "default": val})
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            if not values:
                values = ["Unknown"]
            schema.append({"name": col, "type": "category", "options": values, "default": values[0]})
    return schema


def _top_importance_rows(pipeline: Pipeline, limit: int = 12) -> list[dict[str, Any]]:
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
    else:
        return []
    rows = pd.DataFrame({"feature": feature_names, "importance": importance})
    rows = rows.sort_values("importance", ascending=False).head(limit)
    return rows.to_dict(orient="records")


def _build_classifier_model(model_name: str) -> Any:
    name = model_name.strip().lower()
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "linear_svc":
        return LinearSVC(max_iter=5000)
    if name == "k_neighbors":
        return KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
    raise HTTPException(
        status_code=400,
        detail=(
            "Invalid classification model. Use random_forest, extra_trees, decision_tree, "
            "gradient_boosting, logistic_regression, linear_svc, or k_neighbors."
        ),
    )


def _build_regressor_model(model_name: str) -> Any:
    name = model_name.strip().lower()
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if name == "decision_tree":
        return DecisionTreeRegressor(random_state=42)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42)
    if name == "linear_regression":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "lasso":
        return Lasso(alpha=0.001, max_iter=5000)
    if name == "elastic_net":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
    if name == "linear_svr":
        return LinearSVR(random_state=42)
    if name == "k_neighbors":
        return KNeighborsRegressor(n_neighbors=9, n_jobs=-1)
    raise HTTPException(
        status_code=400,
        detail=(
            "Invalid regression model. Use random_forest, extra_trees, decision_tree, gradient_boosting, "
            "linear_regression, ridge, lasso, elastic_net, linear_svr, or k_neighbors."
        ),
    )


def _classification_model_choices() -> list[str]:
    return [
        "random_forest",
        "extra_trees",
        "decision_tree",
        "gradient_boosting",
        "logistic_regression",
        "linear_svc",
        "k_neighbors",
    ]


def _regression_model_choices() -> list[str]:
    return [
        "random_forest",
        "extra_trees",
        "decision_tree",
        "gradient_boosting",
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "linear_svr",
        "k_neighbors",
    ]


def _sample_for_training(df: pd.DataFrame, target: str, stratify: bool) -> pd.DataFrame:
    if len(df) <= MAX_TRAIN_ROWS:
        return df
    try:
        if stratify:
            return train_test_split(
                df,
                train_size=MAX_TRAIN_ROWS,
                random_state=42,
                stratify=df[target].astype(str),
            )[0]
    except Exception:
        pass
    return train_test_split(df, train_size=MAX_TRAIN_ROWS, random_state=42)[0]


def _run_classification_training(df: pd.DataFrame, target: str, model_name: str) -> dict[str, Any]:
    Xc = df.drop(columns=[target])
    yc_raw = df[target].astype(str)
    label_encoder = LabelEncoder()
    yc = label_encoder.fit_transform(yc_raw)

    try:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.2, random_state=42, stratify=yc
        )
    except ValueError:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.2, random_state=42
        )

    clf = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(Xc)),
            ("model", _build_classifier_model(model_name)),
        ]
    )
    clf.fit(Xc_train, yc_train)
    yc_pred = clf.predict(Xc_test)
    accuracy = float(accuracy_score(yc_test, yc_pred))
    class_labels = label_encoder.classes_

    precision, recall, f1, support = precision_recall_fscore_support(
        yc_test, yc_pred, labels=np.arange(len(class_labels)), zero_division=0
    )
    class_metrics = []
    for idx, label in enumerate(class_labels):
        class_metrics.append(
            {
                "label": str(label),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    cm = confusion_matrix(yc_test, yc_pred, labels=np.arange(len(class_labels)))
    confusion_pairs = []
    for i, actual_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            confusion_pairs.append(
                {
                    "actual": str(actual_label),
                    "predicted": str(pred_label),
                    "count": int(cm[i, j]),
                    "pair": f"{actual_label} -> {pred_label}",
                }
            )
    confusion_pairs = sorted(confusion_pairs, key=lambda row: row["count"], reverse=True)[:20]

    class_points = pd.DataFrame(
        {
            "sample": np.arange(len(yc_test)),
            "actual": label_encoder.inverse_transform(yc_test),
            "predicted": label_encoder.inverse_transform(yc_pred),
        }
    )
    if len(class_points) > MAX_CLASS_POINTS:
        class_points = class_points.sample(MAX_CLASS_POINTS, random_state=42).sort_values("sample")

    feature_schema = _feature_schema(Xc_train)
    top_features = _top_importance_rows(clf)
    mean_f1 = float(np.mean([row["f1"] for row in class_metrics])) if class_metrics else 0.0

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "mean_f1": mean_f1,
        "pipeline": clf,
        "label_encoder": label_encoder,
        "feature_schema": feature_schema,
        "payload": {
            "target": target,
            "model_name": model_name,
            "accuracy": accuracy,
            "points": class_points.to_dict(orient="records"),
            "class_metrics": class_metrics,
            "confusion_pairs": confusion_pairs,
            "top_features": top_features,
            "form_schema": feature_schema,
        },
    }


def _run_regression_training(df: pd.DataFrame, target: str, model_name: str) -> dict[str, Any]:
    Xr = df.drop(columns=[target])
    yr = df[target]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(Xr)),
            ("model", _build_regressor_model(model_name)),
        ]
    )
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)

    mae = float(mean_absolute_error(yr_test, yr_pred))
    r2 = float(r2_score(yr_test, yr_pred))

    reg_points = pd.DataFrame(
        {
            "sample": np.arange(len(yr_test)),
            "actual": yr_test.values,
            "predicted": yr_pred,
            "error": yr_test.values - yr_pred,
        }
    )

    feature_schema = _feature_schema(Xr_train)
    top_features = _top_importance_rows(reg)

    error_series = reg_points["error"]
    if len(error_series) > MAX_REG_ERRORS:
        error_series = error_series.sample(MAX_REG_ERRORS, random_state=42)

    return {
        "model_name": model_name,
        "mae": mae,
        "r2": r2,
        "pipeline": reg,
        "feature_schema": feature_schema,
        "payload": {
            "target": target,
            "model_name": model_name,
            "mae": mae,
            "r2": r2,
            "points": reg_points.head(MAX_REG_POINTS).to_dict(orient="records"),
            "errors": error_series.tolist(),
            "top_features": top_features,
            "form_schema": feature_schema,
        },
    }


def _ensure_dataset_loaded() -> pd.DataFrame:
    if store.cleaned_df is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded. Call /api/load first.")
    return store.cleaned_df


def _derive_target_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_targets: list[str] = []
    numeric_targets: list[str] = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_targets.append(col)
            if unique_count <= 20:
                categorical_targets.append(col)
        else:
            categorical_targets.append(col)
    return categorical_targets, numeric_targets


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/load")
def load_dataset(payload: LoadRequest) -> dict[str, Any]:
    try:
        raw_df = _read_table(Path(payload.file_path))
        cleaned_df, dropped = _clean_data(raw_df, payload.high_cardinality_threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.raw_df = raw_df
    store.cleaned_df = cleaned_df
    store.dropped_columns = dropped

    categorical_targets, numeric_targets = _derive_target_lists(cleaned_df)

    return {
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "dropped_columns": dropped,
        "all_columns": cleaned_df.columns.tolist(),
        "classification_targets": categorical_targets,
        "regression_targets": numeric_targets,
        "classification_models": _classification_model_choices(),
        "regression_models": _regression_model_choices(),
        "preview": cleaned_df.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


@app.post("/api/load/upload")
async def load_dataset_upload(
    file: UploadFile = File(...),
    high_cardinality_threshold: int = Form(50),
) -> dict[str, Any]:
    try:
        filename = file.filename or "uploaded.csv"
        content = await file.read()
        if not content:
            raise ValueError("Uploaded file is empty.")
        raw_df = _read_table_from_bytes(content, filename)
        cleaned_df, dropped = _clean_data(raw_df, int(high_cardinality_threshold))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.raw_df = raw_df
    store.cleaned_df = cleaned_df
    store.dropped_columns = dropped

    categorical_targets, numeric_targets = _derive_target_lists(cleaned_df)
    return {
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "dropped_columns": dropped,
        "all_columns": cleaned_df.columns.tolist(),
        "classification_targets": categorical_targets,
        "regression_targets": numeric_targets,
        "classification_models": _classification_model_choices(),
        "regression_models": _regression_model_choices(),
        "preview": cleaned_df.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


@app.post("/api/train")
def train_models(payload: TrainRequest) -> dict[str, Any]:
    df = _ensure_dataset_loaded()
    mode = (payload.mode or "both").strip().lower()
    if mode not in {"both", "classification", "regression"}:
        raise HTTPException(status_code=400, detail="Mode must be one of: both, classification, regression.")

    categorical_targets, numeric_targets = _derive_target_lists(df)
    classification_target = payload.classification_target
    regression_target = payload.regression_target

    if mode in {"both", "classification"} and not classification_target:
        classification_target = categorical_targets[0] if categorical_targets else None
    if mode in {"both", "regression"} and not regression_target:
        regression_target = numeric_targets[0] if numeric_targets else None

    if mode in {"both", "classification"} and not classification_target:
        raise HTTPException(status_code=400, detail="No valid classification target found in dataset.")
    if mode in {"both", "regression"} and not regression_target:
        raise HTTPException(status_code=400, detail="No valid regression target found in dataset.")

    if classification_target and classification_target not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid classification target.")
    if regression_target and regression_target not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid regression target.")
    if mode == "both" and classification_target == regression_target:
        raise HTTPException(status_code=400, detail="Targets must be different in both mode.")

    result: dict[str, Any] = {}

    # Classification (optional)
    if mode in {"both", "classification"} and classification_target is not None:
        class_models = _classification_model_choices() if payload.auto_select_best_models else [payload.classification_model]
        class_runs: list[dict[str, Any]] = []
        class_failures: list[dict[str, str]] = []
        class_df = _sample_for_training(df, classification_target, stratify=True)
        for model_name in class_models:
            try:
                class_runs.append(_run_classification_training(class_df, classification_target, model_name))
            except Exception as exc:
                class_failures.append({"model_name": model_name, "error": str(exc)})
        if not class_runs:
            raise HTTPException(
                status_code=400,
                detail=f"Classification training failed for all selected models: {class_failures}",
            )
        best_class = max(class_runs, key=lambda row: (row["accuracy"], row["mean_f1"]))

        store.classification_target = classification_target
        store.classifier_pipeline = best_class["pipeline"]
        store.class_label_encoder = best_class["label_encoder"]
        store.class_feature_schema = best_class["feature_schema"]

        result["classification"] = best_class["payload"]
        result["classification"]["model_comparison"] = [
            {"model_name": run["model_name"], "accuracy": run["accuracy"], "mean_f1": run["mean_f1"]}
            for run in sorted(class_runs, key=lambda row: (row["accuracy"], row["mean_f1"]), reverse=True)
        ]
        result["classification"]["model_failures"] = class_failures
    else:
        store.classification_target = None
        store.classifier_pipeline = None
        store.class_label_encoder = None
        store.class_feature_schema = []

    # Regression (optional)
    if mode in {"both", "regression"} and regression_target is not None:
        reg_models = _regression_model_choices() if payload.auto_select_best_models else [payload.regression_model]
        reg_runs: list[dict[str, Any]] = []
        reg_failures: list[dict[str, str]] = []
        reg_df = _sample_for_training(df, regression_target, stratify=False)
        for model_name in reg_models:
            try:
                reg_runs.append(_run_regression_training(reg_df, regression_target, model_name))
            except Exception as exc:
                reg_failures.append({"model_name": model_name, "error": str(exc)})
        if not reg_runs:
            raise HTTPException(
                status_code=400,
                detail=f"Regression training failed for all selected models: {reg_failures}",
            )
        best_reg = max(reg_runs, key=lambda row: (row["r2"], -row["mae"]))

        store.regression_target = regression_target
        store.regressor_pipeline = best_reg["pipeline"]
        store.reg_feature_schema = best_reg["feature_schema"]

        result["regression"] = best_reg["payload"]
        result["regression"]["model_comparison"] = [
            {"model_name": run["model_name"], "r2": run["r2"], "mae": run["mae"]}
            for run in sorted(reg_runs, key=lambda row: (row["r2"], -row["mae"]), reverse=True)
        ]
        result["regression"]["model_failures"] = reg_failures
    else:
        store.regression_target = None
        store.regressor_pipeline = None
        store.reg_feature_schema = []

    # Correlation for summary tab
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > MAX_CORR_FEATURES:
        variance = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
        numeric_cols = variance.head(MAX_CORR_FEATURES).index.tolist()
    correlation_matrix: list[dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(numeric_only=True)
        corr_long = corr_df.stack().reset_index()
        corr_long.columns = ["x", "y", "value"]
        correlation_matrix = corr_long.to_dict(orient="records")

    class_distribution: list[dict[str, Any]] = []
    if classification_target is not None and classification_target in df.columns:
        class_distribution = (
            df[classification_target]
            .astype(str)
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    result["summary"] = {
        "correlation": correlation_matrix,
        "class_distribution": class_distribution,
    }
    result["trained_mode"] = mode
    result["auto_selected_best_models"] = payload.auto_select_best_models
    result["available_tasks"] = {
        "classification": len(categorical_targets) > 0,
        "regression": len(numeric_targets) > 0,
    }
    return result


@app.post("/api/predict/classification")
def predict_classification(payload: PredictRequest) -> dict[str, Any]:
    if store.classifier_pipeline is None or store.class_label_encoder is None or store.classification_target is None:
        raise HTTPException(status_code=400, detail="Classification model not trained yet.")

    row = pd.DataFrame([payload.values])
    pred_encoded = store.classifier_pipeline.predict(row)[0]
    pred_label = store.class_label_encoder.inverse_transform([pred_encoded])[0]
    return {
        "target": store.classification_target,
        "prediction": str(pred_label),
    }


@app.post("/api/predict/regression")
def predict_regression(payload: PredictRequest) -> dict[str, Any]:
    if store.regressor_pipeline is None or store.regression_target is None:
        raise HTTPException(status_code=400, detail="Regression model not trained yet.")

    row = pd.DataFrame([payload.values])
    pred_value = float(store.regressor_pipeline.predict(row)[0])
    return {
        "target": store.regression_target,
        "prediction": pred_value,
    }
