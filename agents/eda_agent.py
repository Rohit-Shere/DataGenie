import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from model.llm import get_llm
from langchain_core.output_parsers import JsonOutputParser
from prompts.eda_insight_prompt import eda_insight_prompt


llm = get_llm()

parser = JsonOutputParser()

chain = eda_insight_prompt | llm | parser

#============================================================
# EDA INSIGHT GENERATOR
#============================================================

def generate_eda_insights(eda_metadata):
    """
    Generate EDA insights using LLM.
    """

    response = chain.invoke({

        "top_positive_correlations":
            eda_metadata[
                "top_positive_correlations"
            ],

        "top_negative_correlations":
            eda_metadata[
                "top_negative_correlations"
            ],

        "highly_skewed_features":
            eda_metadata[
                "highly_skewed_features"
            ],

        "important_features":
            eda_metadata[
                "important_features"
            ],

        "class_imbalance":
            eda_metadata[
                "class_imbalance"
            ],

        "high_outlier_features":
            eda_metadata[
                "high_outlier_features"
            ]
    })

    return response



# ============================================================
# 1. CORRELATION ANALYZER
# ============================================================

def compute_correlations(
    df,
    threshold=0.7
):
    """
    Compute correlation matrix and extract
    strong positive and negative correlations.
    """

    numerical_df = df.select_dtypes(
        include=np.number
    )

    corr_matrix = numerical_df.corr()

    strong_positive = []

    strong_negative = []

    columns = corr_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            col1 = columns[i]

            col2 = columns[j]

            corr_value = corr_matrix.iloc[i, j]

            if corr_value >= threshold:

                strong_positive.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(
                        float(corr_value),
                        3
                    )
                })

            elif corr_value <= -threshold:

                strong_negative.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(
                        float(corr_value),
                        3
                    )
                })

    return {
        "correlation_matrix":
            corr_matrix.to_dict(),

        "strong_positive_correlations":
            strong_positive,

        "strong_negative_correlations":
            strong_negative
    }


# ============================================================
# 2. DISTRIBUTION ANALYZER
# ============================================================

def analyze_distributions(df):
    """
    Analyze feature distributions using
    skewness and kurtosis.
    """

    numerical_df = df.select_dtypes(
        include=np.number
    )

    distribution_summary = {}

    for col in numerical_df.columns:

        skewness = numerical_df[col].skew()

        kurtosis = numerical_df[col].kurt()

        if abs(skewness) < 0.5:
            distribution_type = "approximately_normal"

        elif skewness > 0.5:
            distribution_type = "right_skewed"

        else:
            distribution_type = "left_skewed"

        distribution_summary[col] = {

            "skewness":
                round(float(skewness), 3),

            "kurtosis":
                round(float(kurtosis), 3),

            "distribution_type":
                distribution_type
        }

    return distribution_summary


# ============================================================
# 3. TARGET RELATIONSHIP ANALYZER
# ============================================================

def analyze_target_relationships(
    df,
    target_column,
    problem_type
):
    """
    Analyze relationship between features
    and target variable.
    """

    numerical_df = df.select_dtypes(
        include=np.number
    )

    if target_column not in numerical_df.columns:

        return {
            "error":
                "Target column not numerical."
        }

    X = numerical_df.drop(
        columns=[target_column],
        errors="ignore"
    )

    y = numerical_df[target_column]

    if X.empty:

        return {
            "error":
                "No numerical features available."
        }

    # ----------------------------------------
    # Classification
    # ----------------------------------------

    if problem_type.lower() == "classification":

        scores = mutual_info_classif(
            X,
            y,
            random_state=42
        )

    # ----------------------------------------
    # Regression
    # ----------------------------------------

    else:

        scores = mutual_info_regression(
            X,
            y,
            random_state=42
        )

    feature_importance = []

    for feature, score in zip(
        X.columns,
        scores
    ):

        feature_importance.append({

            "feature": feature,

            "importance_score":
                round(float(score), 4)
        })

    feature_importance = sorted(
        feature_importance,
        key=lambda x:
            x["importance_score"],
        reverse=True
    )

    return {
        "target_column":
            target_column,

        "feature_relationships":
            feature_importance
    }


# ============================================================
# 4. CLASS IMBALANCE DETECTOR
# ============================================================

def detect_class_imbalance(
    df,
    target_column
):
    """
    Detect imbalance in classification datasets.
    """

    if target_column not in df.columns:

        return {
            "error":
                "Target column not found."
        }

    class_counts = (
        df[target_column]
        .value_counts()
        .to_dict()
    )

    total_samples = len(df)

    class_percentages = {}

    for cls, count in class_counts.items():

        class_percentages[str(cls)] = round(
            (count / total_samples) * 100,
            2
        )

    max_class = max(class_counts.values())

    min_class = min(class_counts.values())

    imbalance_ratio = round(
        max_class / min_class,
        2
    )

    if imbalance_ratio > 1.5:

        imbalance_status = "imbalanced"

    else:

        imbalance_status = "balanced"

    return {

        "class_counts":
            class_counts,

        "class_percentages":
            class_percentages,

        "imbalance_ratio":
            imbalance_ratio,

        "imbalance_status":
            imbalance_status
    }


# ============================================================
# 5. OUTLIER SUMMARY GENERATOR
# ============================================================

def summarize_outliers(df):
    """
    Detect outliers using IQR method.
    """

    numerical_df = df.select_dtypes(
        include=np.number
    )

    outlier_summary = {}

    for col in numerical_df.columns:

        Q1 = numerical_df[col].quantile(0.25)

        Q3 = numerical_df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        outliers = numerical_df[
            (numerical_df[col] < lower_bound) |
            (numerical_df[col] > upper_bound)
        ]

        outlier_summary[col] = {

            "num_outliers":
                int(len(outliers)),

            "outlier_percentage":
                round(
                    (len(outliers) / len(df)) * 100,
                    2
                )
        }

    return outlier_summary


def prepare_eda_metadata(
    correlation_results,
    distribution_results,
    target_relationships,
    imbalance_info,
    outlier_summary
):

    # ========================================================
    # TOP CORRELATIONS
    # ========================================================

    top_positive = sorted(
        correlation_results[
            "strong_positive_correlations"
        ],
        key=lambda x: abs(x["correlation"]),
        reverse=True
    )[:5]

    top_negative = sorted(
        correlation_results[
            "strong_negative_correlations"
        ],
        key=lambda x: abs(x["correlation"]),
        reverse=True
    )[:5]

    # ========================================================
    # HIGHLY SKEWED FEATURES
    # ========================================================

    highly_skewed_features = []

    for feature, stats in (
        distribution_results.items()
    ):

        if abs(stats["skewness"]) > 1:

            highly_skewed_features.append({

                "feature": feature,

                "skewness":
                    stats["skewness"],

                "distribution_type":
                    stats["distribution_type"]
            })

    # ========================================================
    # IMPORTANT FEATURES
    # ========================================================

    important_features = (
        target_relationships.get(
            "feature_relationships",
            []
        )[:5]
    )

    # ========================================================
    # HIGH OUTLIER FEATURES
    # ========================================================

    high_outlier_features = []

    for feature, stats in (
        outlier_summary.items()
    ):

        if stats["outlier_percentage"] > 5:

            high_outlier_features.append({

                "feature": feature,

                "outlier_percentage":
                    stats["outlier_percentage"]
            })

    # ========================================================
    # FINAL METADATA
    # ========================================================

    metadata = {

        "top_positive_correlations":
            top_positive,

        "top_negative_correlations":
            top_negative,

        "highly_skewed_features":
            highly_skewed_features,

        "important_features":
            important_features,

        "class_imbalance":
            imbalance_info,

        "high_outlier_features":
            high_outlier_features
    }

    return metadata

# ============================================================
# EDA AGENT
# ============================================================

def eda_agent(state):

    print("🚀 Running EDA Agent...")

    # ========================================================
    # LOAD CLEANED DATAFRAME
    # ========================================================

    df = state["cleaned_dataframe"]

    target_column = state["target_column"]

    problem_type = state["problem_type"]

    # ========================================================
    # STEP 1 : STATISTICAL ENGINE
    # ========================================================

    correlation_results = compute_correlations(df)

    distribution_results = analyze_distributions(df)

    target_relationships = (
        analyze_target_relationships(
            df,
            target_column,
            problem_type
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION ONLY
    # --------------------------------------------------------

    if problem_type.lower() == "classification":

        imbalance_info = detect_class_imbalance(
            df,
            target_column
        )

    else:

        imbalance_info = {
            "message":
                "Not applicable for regression."
        }

    outlier_summary = summarize_outliers(df)

    # ========================================================
    # STEP 2 : METADATA COMPRESSION
    # ========================================================

    eda_metadata = prepare_eda_metadata(

        correlation_results,

        distribution_results,

        target_relationships,

        imbalance_info,

        outlier_summary
    )

    # ========================================================
    # STEP 3 : LLM INSIGHT GENERATION
    # ========================================================

    eda_insights = generate_eda_insights(
        eda_metadata
    )

    # ========================================================
    # STEP 4 : UPDATE STATE
    # ========================================================

    state["correlation_matrix"] = (
        correlation_results
    )

    state["feature_distributions"] = (
        distribution_results
    )

    state["target_relationships"] = (
        target_relationships
    )

    state["class_imbalance_info"] = (
        imbalance_info
    )

    state["outlier_summary"] = (
        outlier_summary
    )

    state["eda_metadata"] = (
        eda_metadata
    )

    state["eda_insights"] = (
        eda_insights
    )

    print("✅ EDA Agent Completed")

    return state