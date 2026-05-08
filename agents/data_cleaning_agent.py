import pandas as pd
import numpy as np

from model.llm import get_llm

from langchain_core.output_parsers import JsonOutputParser

from prompts.data_cleaning_recomendations import (
    cleaning_recommendation_prompt
)


# ============================================================
# LLM SETUP
# ============================================================

llm = get_llm()

parser = JsonOutputParser()

chain = cleaning_recommendation_prompt | llm | parser


# ============================================================
# CLEANING RECOMMENDATIONS
# ============================================================

def get_cleaning_recommendations(state):
    """
    Get cleaning recommendations from LLM.
    """

    response = chain.invoke({

        "basic_info": state["basic_info"],

        "column_types": state["column_types"],

        "statistical_summary":
            state["statistical_summary"],

        "missing_values":
            state["missing_values"],

        "duplicate_info":
            state["duplicate_info"],

        "target_column":
            state["target_column"],

        "dataset_context":
            state["dataset_context"]
    })

    return response


# ============================================================
# CLEANING LOG HELPER
# ============================================================

def update_cleaning_log(
    cleaning_log,
    column,
    operation,
    reason
):

    cleaning_log[column] = {

        "operation": operation,

        "reason": reason
    }

    return cleaning_log


# ============================================================
# IMPUTATION FUNCTIONS
# ============================================================

def median_imputation(df, column):

    median_value = df[column].median()

    df[column] = df[column].fillna(median_value)

    return df


def mode_imputation(df, column):

    mode_value = df[column].mode()[0]

    df[column] = df[column].fillna(mode_value)

    return df


def mean_imputation(df, column):

    mean_value = df[column].mean()

    df[column] = df[column].fillna(mean_value)

    return df


def knn_imputation(df, column):

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=5)

    df[[column]] = imputer.fit_transform(df[[column]])

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def impute_missing_values(
    df,
    recommendations,
    cleaning_log
):

    for recommendation in recommendations.get(
        "missing_value_recommendations",
        []
    ):

        column = recommendation["column"]

        strategy = recommendation["strategy"]

        reason = recommendation["reason"]

        if strategy == "median_imputation":

            df = median_imputation(df, column)

        elif strategy == "mode_imputation":

            df = mode_imputation(df, column)

        elif strategy == "mean_imputation":

            df = mean_imputation(df, column)

        elif strategy == "knn_imputation":

            df = knn_imputation(df, column)
        elif strategy == "drop":
            
            df = df.dropna(subset=[column])

        cleaning_log = update_cleaning_log(
            cleaning_log,
            column,
            strategy,
            reason
        )

    return df, cleaning_log


# ============================================================
# DUPLICATE HANDLING
# ============================================================

def remove_duplicates(df):

    return df.drop_duplicates()


def handle_duplicates(
    df,
    recommendations,
    cleaning_log
):

    duplicate_strategy = recommendations.get(
        "duplicate_handling",
        {}
    )

    strategy = duplicate_strategy.get(
        "strategy",
        None
    )

    reason = duplicate_strategy.get(
        "reason",
        ""
    )

    if strategy == "remove_duplicates":

        initial_rows = len(df)

        df = remove_duplicates(df)

        removed_rows = initial_rows - len(df)

        cleaning_log["duplicates"] = {

            "operation": strategy,

            "removed_rows": removed_rows,

            "reason": reason
        }

    return df, cleaning_log


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def label_encoding(df, column):

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    df[column] = le.fit_transform(
        df[column].astype(str)
    )

    return df, list(le.classes_)


def one_hot_encoding(df, column):

    df = pd.get_dummies(
        df,
        columns=[column],
        prefix=[column]
    )

    return df


def ordinal_encoding(
    df,
    column,
    order
):

    from sklearn.preprocessing import OrdinalEncoder

    oe = OrdinalEncoder(
        categories=[order]
    )

    df[column] = oe.fit_transform(
        df[[column]]
    )

    return df, order


# ============================================================
# HANDLE CATEGORICAL ENCODING
# ============================================================

def encode_categorical_columns(
    df,
    recommendations,
    cleaning_log
):

    encoding_info = {}

    for recommendation in recommendations.get(
        "categorical_encoding_recommendations",
        []
    ):

        column = recommendation["column"]

        strategy = recommendation["encoding_strategy"]

        reason = recommendation["reason"]

        if strategy == "label_encoding":

            df, classes = label_encoding(
                df,
                column
            )

            encoding_info[column] = {
                "classes": classes
            }

        elif strategy == "one_hot_encoding":

            df = one_hot_encoding(
                df,
                column
            )

        elif strategy == "ordinal_encoding":

            order = recommendation.get(
                "order",
                []
            )

            df, order = ordinal_encoding(
                df,
                column,
                order
            )

            encoding_info[column] = {
                "order": order
            }

        cleaning_log = update_cleaning_log(
            cleaning_log,
            column,
            strategy,
            reason
        )

    return df, cleaning_log, encoding_info


# ============================================================
# OUTLIER HANDLING
# ============================================================

def iqr_capping(df, column):

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    return df


# ============================================================
# HANDLE OUTLIERS
# ============================================================

def handle_outliers(
    df,
    recommendations,
    cleaning_log
):

    for recommendation in recommendations.get(
        "outlier_handling_recommendations",
        []
    ):

        column = recommendation["column"]

        strategy = recommendation["strategy"]

        reason = recommendation["reason"]

        if strategy == "iqr_capping":

            df = iqr_capping(df, column)

        elif strategy == "remove_outliers":

            Q1 = df[column].quantile(0.25)

            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR

            upper_bound = Q3 + 1.5 * IQR

            df = df[
                (df[column] >= lower_bound) &
                (df[column] <= upper_bound)
            ]

        elif strategy == "transformation":

            df[column] = np.log1p(
                df[column].clip(lower=0)
            )

        cleaning_log = update_cleaning_log(
            cleaning_log,
            column,
            strategy,
            reason
        )

    return df, cleaning_log


# ============================================================
# MAIN CLEANING AGENT
# ============================================================

def cleaning_agent(state):

    print("🚀 Running Cleaning Agent...")

    
    # COPY RAW DATAFRAME

    df = state["raw_dataframe"].copy()

    cleaning_log = {}

    # GET LLM RECOMMENDATIONS
    recommendations = get_cleaning_recommendations(
        state
    )

  
    # HANDLE MISSING VALUES
 
    df, cleaning_log = impute_missing_values(
        df,
        recommendations,
        cleaning_log
    )

   
    # HANDLE DUPLICATES
    df, cleaning_log = handle_duplicates(
        df,
        recommendations,
        cleaning_log
    )

    
    # HANDLE ENCODING
   
    df, cleaning_log, encoding_info = (
        encode_categorical_columns(
            df,
            recommendations,
            cleaning_log
        )
    )

   
    # HANDLE OUTLIERS
    df, cleaning_log = handle_outliers(
        df,
        recommendations,
        cleaning_log
    )

    # UPDATE STATE
    state["cleaned_dataframe"] = df

    state["cleaning_recommendations"] = (
        recommendations
    )

    state["cleaning_log"] = cleaning_log

    state["encoding_info"] = encoding_info

    print("✅ Cleaning Agent Completed")

    return state