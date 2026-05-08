from langchain_core.prompts import PromptTemplate


cleaning_recommendation_prompt = PromptTemplate(
    template="""
You are an expert Data Preprocessing and Data Cleaning Assistant.

Analyze the dataset metadata carefully and generate ONLY JSON output containing recommended preprocessing strategies.

# DATASET METADATA

basic_info:
{basic_info}

column_types:
{column_types}

statistical_summary:
{statistical_summary}

missing_values:
{missing_values}

duplicate_info:
{duplicate_info}

target_column:
{target_column}

dataset_context:
{dataset_context}


# TASK

Analyze the dataset and generate cleaning recommendations ONLY for:

1. Missing Value Handling
2. Duplicate Removal
3. Categorical Encoding
4. Outlier Handling

Use dataset-aware reasoning while recommending strategies.

Examples:
- Use median imputation for skewed numerical columns
- Use mode imputation for categorical columns
- Recommend label encoding for ordinal categories
- Recommend one-hot encoding for low-cardinality nominal categories
- Recommend IQR capping for numerical outliers


# STRICT RULES

- Return ONLY valid JSON
- Do NOT return explanations outside JSON
- Do NOT include markdown
- Do NOT include comments
- Do NOT include additional text


# REQUIRED OUTPUT FORMAT

{{
    "missing_value_recommendations": [
        {{
            "column": "column_name",
            "strategy": "mean | median | mode | drop",
            "reason": "short reason"
        }}
    ],

    "duplicate_handling": {{
        "strategy": "remove_duplicates",
        "reason": "short reason"
    }},

    "categorical_encoding_recommendations": [
        {{
            "column": "column_name",
            "encoding_strategy": "label_encoding | one_hot_encoding",
            "reason": "short reason"
        }}
    ],

    "outlier_handling_recommendations": [
        {{
            "column": "column_name",
            "strategy": "iqr_capping | transformation | remove_outliers | none",
            "reason": "short reason"
        }}
    ]
}}
""",
    input_variables=[
        "basic_info",
        "column_types",
        "statistical_summary",
        "missing_values",
        "duplicate_info",
        "target_column",
        "dataset_context"
    ]
)