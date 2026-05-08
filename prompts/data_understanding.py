
def get_target_column_prompt(column_types , column_summary) -> str:
    return f"""

    Column Types:{column_types}

    Column Summary:{column_summary}
    
    Based on the dataset provided, identify the most likely target column for a machine learning task. 
    Consider factors such as:
    - Column names that suggest a target variable (e.g., "price", "label", "outcome")
    - Data types that are suitable for prediction (e.g., categorical for classification, numerical for regression)
    - The presence of missing values or unique values that might indicate a target variable
    - The distribution of values in each column

    ## Output Format:
    Return the name of the most likely target column.
    Don't give any other information or explanation, just the column name as a string.
    """