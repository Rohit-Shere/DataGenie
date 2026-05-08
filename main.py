# ============================================================
# SUPER STATE
# Shared State for Multi-Agent Autonomous
# Data Science Assistant
# ============================================================

from typing import TypedDict, Dict, Any, Optional
import pandas as pd


class SuperState(TypedDict, total=False):

    # ========================================================
    # DATAFRAMES
    # ========================================================

    # Original uploaded dataset
    raw_dataframe: pd.DataFrame

    # Cleaned dataset after preprocessing
    cleaned_dataframe: pd.DataFrame

    # Feature engineered dataset (future agent)
    feature_engineered_dataframe: pd.DataFrame


    # ========================================================
    # DATASET UNDERSTANDING AGENT
    # ========================================================

    # Basic dataset information
    basic_info: Dict[str, Any]

    # Column type classifications
    column_types: Dict[str, Any]

    # Missing value analysis
    missing_values: Dict[str, Any]

    # Duplicate analysis
    duplicate_info: Dict[str, Any]

    # Statistical summary
    statistical_summary: Dict[str, Any]

    # Detected target column
    target_column: str

    # Inferred ML task
    problem_type: str

    # Semantic dataset understanding
    dataset_context: str


    # ========================================================
    # CLEANING AGENT
    # ========================================================

    # Cleaning operation logs
    cleaning_log: Dict[str, Any]

    # LLM cleaning recommendations
    cleaning_recommendations: Dict[str, Any]

    # Outlier analysis
    outlier_info: Dict[str, Any]

    # Encoding details
    encoding_info: Dict[str, Any]


    # ========================================================
    # EDA AGENT (Future)
    # ========================================================

    # Correlation analysis
    correlation_matrix: Dict[str, Any]

    # EDA insights
    eda_insights: Dict[str, Any]

    # Visualization paths / metadata
    visualizations: Dict[str, Any]

    # Feature distributions
    feature_distributions: Dict[str, Any]


    # ========================================================
    # FEATURE ENGINEERING AGENT (Future)
    # ========================================================

    # Created features
    engineered_features: Dict[str, Any]

    # Feature selection results
    selected_features: Dict[str, Any]


    # ========================================================
    # ML AGENT (Future)
    # ========================================================

    # Recommended models
    model_recommendations: Dict[str, Any]

    # Training results
    training_results: Dict[str, Any]

    # Evaluation metrics
    evaluation_metrics: Dict[str, Any]

    # Best model information
    best_model_info: Dict[str, Any]


    # ========================================================
    # XAI AGENT (Future)
    # ========================================================

    # SHAP/LIME explanations
    feature_importance: Dict[str, Any]

    # Model explanation text
    model_explanations: Dict[str, Any]


    # ========================================================
    # REPORT AGENT (Future)
    # ========================================================

    # Final generated report
    final_report: Dict[str, Any]

    # Generated recommendations
    business_recommendations: Dict[str, Any]


    # ========================================================
    # SYSTEM / ORCHESTRATION
    # ========================================================

    # Current active agent
    current_agent: str

    # Completed agents
    completed_agents: list

    # Errors encountered
    errors: list

    # Execution metadata
    execution_metadata: Dict[str, Any]