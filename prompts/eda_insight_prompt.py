from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


eda_insight_prompt = PromptTemplate(
    template="""
        You are an expert Senior Data Scientist, ML Engineer,
        and Business Analyst.

        Your task is to analyze the provided EDA metadata
        and generate high-quality analytical insights.

        # EDA METADATA

        top_positive_correlations:
        {top_positive_correlations}

        top_negative_correlations:
        {top_negative_correlations}

        highly_skewed_features:
        {highly_skewed_features}

        important_features:
        {important_features}

        class_imbalance:
        {class_imbalance}

        high_outlier_features:
        {high_outlier_features}


        # TASK

        Analyze the dataset metadata carefully and generate:

        1. Key analytical insights
        2. Important predictive features
        3. Data quality observations
        4. Machine learning recommendations
        5. Risk factors and warnings
        6. Business insights


        # ANALYSIS GUIDELINES

        - Identify strong feature relationships
        - Detect possible multicollinearity
        - Identify skewed distributions
        - Analyze outlier severity
        - Analyze class imbalance impact
        - Suggest suitable ML model families
        - Highlight potential preprocessing needs
        - Mention possible overfitting risks
        - Mention possible data leakage risks if applicable


        # STRICT RULES

        - Return ONLY valid JSON
        - Do NOT include markdown
        - Do NOT include explanations outside JSON
        - Keep insights concise but meaningful
        - Avoid generic statements


        # REQUIRED OUTPUT FORMAT

        {{
            "key_insights": [
                {{
                    "insight": "string",
                    "importance": "high | medium | low"
                }}
            ],

            "important_predictive_features": [
                {{
                    "feature": "string",
                    "reason": "string"
                }}
            ],

            "data_quality_observations": [
                {{
                    "observation": "string",
                    "severity": "high | medium | low"
                }}
            ],

            "ml_recommendations": [
                {{
                    "recommendation": "string",
                    "reason": "string"
                }}
            ],

            "risk_factors": [
                {{
                    "risk": "string",
                    "severity": "high | medium | low"
                }}
            ],

            "business_insights": [
                {{
                    "insight": "string"
                }}
            ]
        }}

        """,

            input_variables=[

                "top_positive_correlations",

                "top_negative_correlations",

                "highly_skewed_features",

                "important_features",

                "class_imbalance",

                "high_outlier_features"
            ]
        )