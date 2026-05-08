# ============================================================
# Agent 1 : Dataset Understanding Agent
# ============================================================

# =========================
# Required Libraries
# =========================

import pandas as pd
import numpy as np
from typing import TypedDict, Dict, Any


# ============================================================
# SHARED STATE
# ============================================================

class AgentState(TypedDict, total=False):

    # DataFrames
    raw_dataframe: pd.DataFrame

    # Dataset Understanding Outputs
    basic_info: dict
    column_types: dict
    missing_values: dict
    duplicate_info: dict
    statistical_summary: dict
    target_column: str
    problem_type: str
    dataset_context: str


# ============================================================
# TOOL 1 : LOAD DATASET
# ============================================================

def load_dataset(file_path: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_path)

        print("✅ Dataset Loaded Successfully")

        return df

    except Exception as e:
        print(f"❌ Error Loading Dataset: {e}")
        raise


# ============================================================
# TOOL 2 : BASIC INFO
# ============================================================

def get_basic_info(df: pd.DataFrame) -> dict:

    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": list(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }

    return info


# ============================================================
# TOOL 3 : DETECT COLUMN TYPES
# ============================================================

def detect_column_types(df: pd.DataFrame) -> dict:

    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    boolean_cols = []
    text_cols = []

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)

        elif pd.api.types.is_bool_dtype(df[col]):
            boolean_cols.append(col)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)

        elif df[col].dtype == "object" or df[col].dtype == "str":

            unique_ratio = df[col].nunique() / len(df)

            if unique_ratio > 0.5:
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    return {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "boolean_columns": boolean_cols,
        "text_columns": text_cols
    }


# ============================================================
# TOOL 4 : MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df: pd.DataFrame) -> dict:

    missing_info = {}

    for col in df.columns:

        missing_count = df[col].isnull().sum()

        if missing_count > 0:

            missing_percentage = round(
                (missing_count / len(df)) * 100,
                2
            )

            missing_info[col] = {
                "missing_count": int(missing_count),
                "missing_percentage": missing_percentage
            }

    return missing_info


# ============================================================
# TOOL 5 : DUPLICATE ANALYSIS
# ============================================================

def detect_duplicates(df: pd.DataFrame) -> dict:

    duplicate_rows = df.duplicated().sum()

    return {
        "duplicate_rows": int(duplicate_rows)
    }


# ============================================================
# TOOL 6 : STATISTICAL SUMMARY
# ============================================================

def generate_statistical_summary(df: pd.DataFrame) -> dict:

    summary = {}

    numerical_df = df.select_dtypes(include=np.number)

    for col in numerical_df.columns:

        summary[col] = {
            "mean": round(float(numerical_df[col].mean()), 2),
            "median": round(float(numerical_df[col].median()), 2),
            "std": round(float(numerical_df[col].std()), 2),
            "min": round(float(numerical_df[col].min()), 2),
            "max": round(float(numerical_df[col].max()), 2),
            "skewness" : round(float(numerical_df[col].skew()), 2),
            "kurtosis" : round(float(numerical_df[col].kurtosis()),
            2)      
        }

    return summary


# ============================================================
# TOOL 7 : TARGET COLUMN DETECTION
# ============================================================

# def detect_target_column(Statistical_summary, Column_types) -> str:
    
    
#     llm = get_llm()
    
#     prompt = get_target_column_prompt(
#         column_types=Column_types,
#         column_summary=Statitical_summary
#     )
    
#     response = llm(prompt)
    
    

    # possible_targets = [
    #     "target",
    #     "label",
    #     "class",
    #     "output",
    #     "survived",
    #     "price",
    #     "salary"
    # ]

    # columns_lower = [col.lower() for col in df.columns]

    # for target in possible_targets:

    #     if target in columns_lower:

    #         index = columns_lower.index(target)

    #         return df.columns[index]

    # # fallback heuristic
    # for col in df.columns:

    #     unique_values = df[col].nunique()

    #     if unique_values < 10:
    #         return col

    # return "Not Detected"


# ============================================================
# TOOL 8 : PROBLEM TYPE INFERENCE
# ============================================================

def infer_problem_type(
    df: pd.DataFrame,
    target_column: str
) -> str:

    if target_column == "Not Detected":
        return "Unknown"

    unique_values = df[target_column].nunique()

    if unique_values <= 10:
        return "Classification"

    elif pd.api.types.is_numeric_dtype(df[target_column]):
        return "Regression"

    return "Unknown"


# ============================================================
# TOOL 9 : DATASET CONTEXT GENERATION
# ============================================================

def generate_dataset_context(
    basic_info: dict,
    target_column: str,
    problem_type: str
) -> str:

    context = f"""
    This dataset contains
    {basic_info['num_rows']} rows and
    {basic_info['num_columns']} columns.

    The detected target column is
    '{target_column}'.

    The inferred machine learning task is
    '{problem_type}'.
    """

    return context.strip()


# ============================================================
# FINAL REPORT CREATION
# ============================================================

def create_final_report(
    basic_info,
    column_types,
    missing_values,
    duplicate_info,
    statistical_summary,
    target_column,
    problem_type,
    dataset_context
):

    return {
        "basic_info": basic_info,
        "column_types": column_types,
        "missing_values": missing_values,
        "duplicate_info": duplicate_info,
        "statistical_summary": statistical_summary,
        "target_column": target_column,
        "problem_type": problem_type,
        "dataset_context": dataset_context
    }


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def dataset_understanding_agent(
    file_path: str,
    target_column: str
) -> AgentState:

    # ----------------------------
    # Load Dataset
    # ----------------------------

    df = load_dataset(file_path)

    # ----------------------------
    # Run Analysis
    # ----------------------------

    basic_info = get_basic_info(df)

    column_types = detect_column_types(df)

    missing_values = analyze_missing_values(df)

    duplicate_info = detect_duplicates(df)

    statistical_summary = generate_statistical_summary(df)

    # target_column = detect_target_column(df)

    problem_type = infer_problem_type(
        df,
        target_column
    )

    dataset_context = generate_dataset_context(
        basic_info,
        target_column,
        problem_type
    )

    # ----------------------------
    # Create Final Report
    # ----------------------------

    final_report = create_final_report(
        basic_info,
        column_types,
        missing_values,
        duplicate_info,
        statistical_summary,
        target_column,
        problem_type,
        dataset_context
    )

    # ----------------------------
    # Update Shared State
    # ----------------------------

    state: AgentState = {

        "raw_dataframe": df,

        "basic_info": basic_info,

        "column_types": column_types,

        "missing_values": missing_values,

        "duplicate_info": duplicate_info,

        "statistical_summary": statistical_summary,

        "target_column": target_column,

        "problem_type": problem_type,

        "dataset_context": dataset_context
    }

    print("\n✅ Dataset Understanding Agent Completed")

    return state


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    FILE_PATH = r"C:\Users\Rohit Shere\OneDrive\Desktop\Data Science Assistant\datasets\unique_tech_classroom_air_quality.csv"

    target_column = "air_quality_label"
    state = dataset_understanding_agent(FILE_PATH, target_column)

    print("\n==============================")
    print("FINAL STATE")
    print("==============================")

    for key, value in state.items():

        if key != "raw_dataframe":

            print(f"\n{key} :")
            print(value)