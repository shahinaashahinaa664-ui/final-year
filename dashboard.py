import pathlib
import html

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")


def load_data(uploaded_file, file_path):
    if uploaded_file is not None:
        suffix = pathlib.Path(uploaded_file.name).suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)

    if file_path:
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        return pd.read_csv(path)

    raise ValueError("Please upload a file or provide a valid local path.")


def clean_data(df, high_card_threshold=50):
    data = df.copy()
    data = data.drop_duplicates()

    datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    for col in datetime_cols:
        data[col] = data[col].astype("int64") // 10**9

    num_cols = data.select_dtypes(include=[np.number]).columns
    cat_cols = data.select_dtypes(include=["object", "category"]).columns

    if len(num_cols) > 0:
        data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    for col in cat_cols:
        mode_values = data[col].mode(dropna=True)
        if len(mode_values) > 0:
            data[col] = data[col].fillna(mode_values.iloc[0])
        else:
            data[col] = data[col].fillna("Unknown")

    drop_cols = []
    for col in cat_cols:
        if data[col].nunique(dropna=False) > high_card_threshold:
            drop_cols.append(col)
    if drop_cols:
        data = data.drop(columns=drop_cols)

    return data, drop_cols


def build_preprocessor(X):
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


def train_classifier(df, target_col):
    X = df.drop(columns=[target_col])
    y_raw = df[target_col].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return {
        "pipeline": model,
        "label_encoder": label_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
    }


def train_regressor(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "pipeline": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "r2": r2,
    }


def get_feature_importance(trained_pipeline):
    preprocessor = trained_pipeline.named_steps["preprocess"]
    model = trained_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
    return imp_df


def build_prediction_form(title, form_key, X_train):
    st.subheader(title)
    with st.form(key=form_key):
        input_data = {}
        for col in X_train.columns:
            series = X_train[col]
            if pd.api.types.is_numeric_dtype(series):
                default_val = float(series.median()) if not series.empty else 0.0
                input_data[col] = st.number_input(
                    f"{col}",
                    value=default_val,
                    key=f"{form_key}_{col}",
                )
            else:
                options = (
                    series.dropna().astype(str).unique().tolist()
                    if not series.dropna().empty
                    else ["Unknown"]
                )
                options = sorted(options)
                input_data[col] = st.selectbox(
                    f"{col}",
                    options=options,
                    key=f"{form_key}_{col}",
                )
        submitted = st.form_submit_button("Predict")
    return submitted, input_data


def build_narrative_summary(
    df,
    class_target,
    reg_target,
    class_accuracy,
    reg_r2,
    reg_mae,
):
    narrative = []
    strengths = []
    weaknesses = []
    suggestions = []
    development_areas = []
    growth_actions = [
        "Higher premium revenue from focused tier-up of No Membership and Basic Membership.",
        "Lower churn by targeting low points_in_wallet customers with retention offers.",
        "Higher monthly revenue per user by lifting low avg_transaction_value segments.",
        "Better recurring revenue protection through weekly high-risk customer save lists.",
        "Higher campaign ROI via monthly KPI-led budget reallocation to best-performing segments.",
    ]

    target_distribution = df[class_target].astype(str).value_counts()
    top_label = None
    top_share = 0.0
    if not target_distribution.empty:
        top_label = target_distribution.index[0]
        top_share = target_distribution.iloc[0] / target_distribution.sum()

    narrative.append(
        f"The customer base is currently led by '{top_label}' ({top_share * 100:.1f}% of records), and the classification model is running at {class_accuracy * 100:.2f}% accuracy."
    )
    narrative.append(
        f"The regression model reports R2 of {reg_r2:.3f} with MAE of {reg_mae:.3f} for {reg_target}, indicating current forecast reliability."
    )

    if class_accuracy < 0.7:
        weaknesses.append(
            "Classification quality is below a strong operational threshold, so class-level errors should be reviewed before production decisions."
        )
        suggestions.append(
            f"Improve {class_target} prediction with feature refinement, class balancing, and hyperparameter tuning."
        )
    else:
        suggestions.append(
            f"Use the current {class_target} model for guided decision support while monitoring drift monthly."
        )
    strengths.append(
        "Segment targeting enables more efficient campaign spend and stronger ROI on retention and upgrade programs."
    )

    reg_std = float(df[reg_target].std()) if reg_target in df.columns else 0.0
    if reg_mae > reg_std:
        weaknesses.append(
            f"Average regression error (MAE {reg_mae:.3f}) is high relative to spread in {reg_target}, limiting precision."
        )
        suggestions.append(
            f"Add stronger predictors for {reg_target} and test alternative regressors to reduce MAE."
        )
    else:
        pass

    if reg_r2 >= 0:
        pass
    else:
        weaknesses.append(
            f"The regression fit is weak (R2 = {reg_r2:.3f}), so predictions may not reliably track changes in {reg_target}."
        )

    if top_label is not None:
        if top_share >= 0.45:
            weaknesses.append(
                f"The target '{class_target}' is concentrated in '{top_label}' ({top_share * 100:.1f}% of records), which can reduce minority-class prediction quality."
            )
            development_areas.append(
                f"Improve conversion from '{top_label}' into higher-value tiers using staged offers and personalized follow-ups."
            )
            strengths.append(
                f"Large concentration in '{top_label}' creates scale for conversion and upsell programs, improving premium revenue growth."
            )
        else:
            strengths.append(
                f"The '{class_target}' mix is relatively balanced, enabling consistent segment targeting and predictable campaign outcomes."
            )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2 and reg_target in numeric_cols:
        corr = (
            df[numeric_cols]
            .corr(numeric_only=True)[reg_target]
            .drop(labels=[reg_target], errors="ignore")
        )
        if not corr.empty:
            strongest_driver = corr.abs().idxmax()
            driver_value = corr[strongest_driver]
            narrative.append(
                f"The strongest measurable driver linked with {reg_target} is '{strongest_driver}' (correlation {driver_value:.2f})."
            )
            strengths.append(
                f"Clear driver '{strongest_driver}' enables targeted offers that protect retention and lift repeat purchases."
            )
            suggestions.append(
                f"Prioritize actions on '{strongest_driver}' (correlation {driver_value:.2f} with {reg_target}) in retention and upsell experiments."
            )
            development_areas.append(
                f"Build feature engineering around '{strongest_driver}' and monitor its monthly impact on {reg_target}."
            )

    if not weaknesses:
        weaknesses.append(
            "No critical modeling weakness is immediately visible from top-line metrics; continue monitoring for drift and segment-level variance."
        )
    if not development_areas:
        development_areas.append(
            "Set up monthly segment-wise monitoring and retraining triggers for cohorts with declining prediction confidence."
        )

    return {
        "narrative": " ".join(narrative[:3]),
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "suggestions": suggestions[:3],
        "development_areas": development_areas[:3],
        "growth_actions": growth_actions,
    }


st.markdown(
    """
    <style>
    .brand-header {
        padding: 10px 0 6px 0;
    }
    .brand-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin: 0;
        text-transform: uppercase;
        background: linear-gradient(90deg, rgba(173, 220, 255, 0.95), rgba(142, 255, 220, 0.95));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 10px 24px rgba(6, 24, 40, 0.55);
        animation: title-reveal 0.7s ease both, title-glow 2.8s ease-in-out 0.7s infinite alternate;
    }
    .brand-subtitle {
        font-size: 0.92rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 4px 0 0 0;
        background: linear-gradient(90deg, rgba(173, 220, 255, 0.95), rgba(142, 255, 220, 0.95));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 6px 14px rgba(6, 24, 40, 0.45);
        animation: subtitle-reveal 0.6s ease 0.2s both;
    }
    @keyframes title-glow {
        from { text-shadow: 0 8px 20px rgba(10, 28, 44, 0.5); }
        to { text-shadow: 0 12px 26px rgba(12, 46, 74, 0.8); }
    }
    @keyframes title-reveal {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes subtitle-reveal {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="brand-header">
        <div class="brand-title">Novasight</div>
        <div class="brand-subtitle">The Intelligence Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file", type=["csv", "xlsx", "xls"]
)
default_path = r"C:\Users\Shahina\Downloads\loan_prediction_dataset.csv"
file_path = st.sidebar.text_input("Or enter local file path", value=default_path)
high_card_threshold = st.sidebar.slider(
    "High-cardinality cutoff", min_value=10, max_value=200, value=50, step=5
)

try:
    raw_df = load_data(uploaded_file, file_path)
except Exception as exc:
    st.info("Load your dataset from upload or by local path to start the dashboard.")
    st.error(str(exc))
    st.stop()

clean_df, dropped_cols = clean_data(raw_df, high_card_threshold=high_card_threshold)

st.subheader("Dataset Overview")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Rows", f"{clean_df.shape[0]}")
col_b.metric("Columns", f"{clean_df.shape[1]}")
col_c.metric("Dropped High-Cardinality Columns", f"{len(dropped_cols)}")

if dropped_cols:
    st.warning("Dropped columns: " + ", ".join(dropped_cols))

with st.expander("Preview cleaned data"):
    st.dataframe(clean_df.head(30), use_container_width=True)

cat_like_targets = []
num_targets = []
for col in clean_df.columns:
    unique_count = clean_df[col].nunique(dropna=False)
    if pd.api.types.is_numeric_dtype(clean_df[col]):
        num_targets.append(col)
        if unique_count <= 20:
            cat_like_targets.append(col)
    else:
        cat_like_targets.append(col)

if not cat_like_targets:
    st.error("No suitable classification target found in this dataset.")
    st.stop()
if not num_targets:
    st.error("No numeric target found for regression in this dataset.")
    st.stop()

default_class_target = (
    "membership_category" if "membership_category" in clean_df.columns else cat_like_targets[0]
)
default_reg_target = (
    "churn_risk_score" if "churn_risk_score" in clean_df.columns else num_targets[0]
)

st.sidebar.header("Target Selection")
class_target = st.sidebar.selectbox(
    "Classification target",
    options=cat_like_targets,
    index=cat_like_targets.index(default_class_target),
)
reg_target = st.sidebar.selectbox(
    "Regression target",
    options=num_targets,
    index=num_targets.index(default_reg_target),
)

if class_target == reg_target:
    st.error("Classification and regression targets must be different.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "Summary Charts"])

with tab1:
    st.header(f"Classification: {class_target}")
    class_model = train_classifier(clean_df, class_target)
    st.metric("Accuracy", f"{class_model['accuracy'] * 100:.2f}%")

    class_chart_df = pd.DataFrame(
        {
            "Actual": class_model["label_encoder"].inverse_transform(class_model["y_test"]),
            "Predicted": class_model["label_encoder"].inverse_transform(class_model["y_pred"]),
        }
    )
    class_chart_df = class_chart_df.reset_index().rename(columns={"index": "Sample"})
    fig_cls = px.scatter(
        class_chart_df,
        x="Sample",
        y="Actual",
        color="Predicted",
        title="Actual vs Predicted Class (Test Samples)",
    )
    st.plotly_chart(fig_cls, use_container_width=True)

    imp_df_cls = get_feature_importance(class_model["pipeline"])
    fig_imp_cls = px.bar(
        imp_df_cls,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Features (Classification)",
    )
    fig_imp_cls.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp_cls, use_container_width=True)

    submitted, input_data = build_prediction_form(
        "Try a New Classification Prediction",
        "class_form",
        class_model["X_train"],
    )
    if submitted:
        input_df = pd.DataFrame([input_data])
        pred_encoded = class_model["pipeline"].predict(input_df)[0]
        pred_label = class_model["label_encoder"].inverse_transform([pred_encoded])[0]
        st.success(f"Predicted {class_target}: {pred_label}")

with tab2:
    st.header(f"Regression: {reg_target}")
    reg_model = train_regressor(clean_df, reg_target)
    col_r1, col_r2 = st.columns(2)
    col_r1.metric("MAE", f"{reg_model['mae']:.3f}")
    col_r2.metric("R2 Score", f"{reg_model['r2']:.3f}")

    reg_chart_df = pd.DataFrame(
        {
            "Actual": reg_model["y_test"].values,
            "Predicted": reg_model["y_pred"],
        }
    ).reset_index(drop=True)
    reg_chart_df["Error"] = reg_chart_df["Actual"] - reg_chart_df["Predicted"]
    reg_chart_df = reg_chart_df.reset_index().rename(columns={"index": "Sample"})

    fig_reg = px.line(
        reg_chart_df.head(120),
        x="Sample",
        y=["Actual", "Predicted"],
        title="Actual vs Predicted (First 120 Test Samples)",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    fig_err = px.histogram(
        reg_chart_df,
        x="Error",
        nbins=35,
        title="Prediction Error Distribution",
    )
    st.plotly_chart(fig_err, use_container_width=True)

    imp_df_reg = get_feature_importance(reg_model["pipeline"])
    fig_imp_reg = px.bar(
        imp_df_reg,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Features (Regression)",
    )
    fig_imp_reg.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp_reg, use_container_width=True)

    submitted_reg, input_data_reg = build_prediction_form(
        "Try a New Regression Prediction",
        "reg_form",
        reg_model["X_train"],
    )
    if submitted_reg:
        input_df_reg = pd.DataFrame([input_data_reg])
        pred_value = reg_model["pipeline"].predict(input_df_reg)[0]
        st.success(f"Predicted {reg_target}: {pred_value:.4f}")

with tab3:
    st.header("Simple Business Summary")
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = clean_df[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Numeric Correlation Heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation chart.")

    if class_target in clean_df.columns:
        class_dist = clean_df[class_target].astype(str).value_counts().reset_index()
        class_dist.columns = [class_target, "count"]
        fig_dist = px.bar(
            class_dist,
            x=class_target,
            y="count",
            title=f"Distribution of {class_target}",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Narrative Executive Summary")
    st.caption("Auto-generated interpretation of current model outputs for leadership decisions.")
    narrative = build_narrative_summary(
        clean_df,
        class_target,
        reg_target,
        class_model["accuracy"],
        reg_model["r2"],
        reg_model["mae"],
    )

    st.markdown(
        """
        <style>
        .exec-card {
            border: 1px solid #c9d4ea;
            border-radius: 14px;
            padding: 16px 18px;
            background: #f7faff;
            min-height: 200px;
        }
        .exec-card h4 {
            margin: 0 0 8px 0;
            color: #1f355c;
            font-size: 1.2rem;
        }
        .exec-card p {
            margin: 0;
            color: #344868;
            line-height: 1.55;
        }
        .exec-card ul {
            margin: 0;
            padding-left: 18px;
            color: #344868;
        }
        .exec-card li {
            margin-bottom: 6px;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    narrative_html = html.escape(narrative["narrative"])
    st.markdown(
        f"""
        <div class="exec-card" style="min-height: 120px;">
            <h4>Executive Narrative</h4>
            <p>{narrative_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    def _as_html_list(items):
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown(
            f"""
            <div class="exec-card">
                <h4>Strengths</h4>
                <ul>{_as_html_list(narrative["strengths"])}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class="exec-card">
                <h4>Suggestions</h4>
                <ul>{_as_html_list(narrative["suggestions"])}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_2:
        st.markdown(
            f"""
            <div class="exec-card">
                <h4>Weaknesses</h4>
                <ul>{_as_html_list(narrative["weaknesses"])}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class="exec-card">
                <h4>Areas for Development</h4>
                <ul>{_as_html_list(narrative["development_areas"])}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        f"""
        <div class="exec-card">
            <h4>Executive Growth Actions</h4>
            <ul>{_as_html_list(narrative["growth_actions"])}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
