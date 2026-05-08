
def get_cleaning_recommendations_prompt(state) -> dict:
    return f"""
    basic_info: {state["basic_info"]}
    column_types: {state["column_types"]}
    statistical_summary: {state["statistical_summary"]}
    missing_values: {state["missing_values"]}
    duplicate_info: {state["duplicate_info"]}
    target_column: {state["target_column"]}
    context: {state["dataset_context"]}
    
    
    Based on the dataset provided, identify potential data quality issues and recommend appropriate data cleaning steps. 
    Consider factors such as:
    - Missing values: Identify columns with missing values and suggest strategies for handling them (e.g., imputation, removal).
    - Outliers: Detect columns with outliers and recommend methods for addressing them (e.g., transformation, capping).
    - Inconsistent data types: Highlight columns with inconsistent data types and propose solutions (e.g., type conversion).
    - Duplicate entries: Identify any duplicate records and suggest ways to handle them (e.g., removal, aggregation).
    - Irrelevant features: Point out any columns that may not be relevant for analysis and recommend whether to drop them.

    ## Output Format:
    Return a list of identified issues along with recommended cleaning steps for each issue.
    """