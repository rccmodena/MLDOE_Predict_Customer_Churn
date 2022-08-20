"""
Module with constant values used in the churn_library.py module

Author: Rudi César Comitto Modena
Date: August, 2022
"""

from typing import Final

# Dataset settings
DATASET_PATH: Final[str] = "./data/bank_data.csv"

# EDA settings
RESPONSE_VARIABLE: Final[str] = "Churn"
RESPONSE_BASE_VARIABLE: Final[str] = "Attrition_Flag"
COSTUMER_CHURN_VALUE: Final[str] = "Attrited Customer"

EDA_FIGURE_WIDTH: Final[int] = 20
EDA_FIGURE_HEIGHT: Final[int] = 10

EDA_FIGURE_FOLDER: Final[str] = "./images/eda/"

EDA_CHURN_DISTRIB_FILENAME: Final[str] = EDA_FIGURE_FOLDER + "churn_distribution.png"

EDA_HEATMAP_FILENAME: Final[str] = EDA_FIGURE_FOLDER + "heatmap.png"

EDA_CUSTOMER_AGE_VARIABLE: Final[str] = "Customer_Age"
EDA_CUSTOMER_AGE_DISTRIB_FILENAME: Final[str] = EDA_FIGURE_FOLDER + "customer_age_distribution.png"

EDA_MARITAL_STATUS_VARIABLE: Final[str] = "Marital_Status"
EDA_MARITAL_STATUS_DISTR_FILENAME: Final[str] = EDA_FIGURE_FOLDER + "marital_status_distribution.png"

EDA_TOTAL_TRANSACTION_VARIABLE: Final[str] = "Total_Trans_Ct"
EDA_TOTAL_TRANSACTION_FILENAME: Final[str] = EDA_FIGURE_FOLDER + "total_transaction_distribution.png"

# Modeling Settings 
TEST_SIZE: Final[float] = 0.3
RANDOM_STATE: Final[int] = 42

LRC_SOLVER: Final[str] = 'lbfgs'
LRC_MAX_ITER: Final[int] = 3000

# Categorical Columns
CAT_COLUMNS: Final[list] = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

# Remove Columns for analysis
REMOVE_COLUMNS: Final[list] = [
    'Unnamed: 0',
    'CLIENTNUM',
    'Attrition_Flag',
]

PARAM_GRID: Final[dict] = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

CROSS_VALID: Final[int] = 5

# Results settings
MODELS_FOLDER: Final[str] = "./models/"

RFC_MODEL_FILENAME: Final[str] = MODELS_FOLDER + 'rfc_model.pkl'
LOGISTIC_MODEL_FILENAME: Final[str] = MODELS_FOLDER + 'logistic_model.pkl'

RESULTS_FIGURE_FOLDER: Final[str] = "./images/results/"

LOGISTIC_RESULTS_FILENAME: Final[str] = RESULTS_FIGURE_FOLDER + "logistic_results.png"
RFC_RESULTS_FILENAME: Final[str] = RESULTS_FIGURE_FOLDER + "rfc_results.png"

RESULTS_REPORTS_WIDTH: Final[int] = 5
RESULTS_REPORTS_HEIGHT: Final[int] = 5

RESULTS_FONT_SETUP: Final[dict] = {'fontsize': 10, 'fontproperties': 'monospace'}

RESULTS_ROC_FILENAME: Final[str] = RESULTS_FIGURE_FOLDER + "roc_curve_result.png"

RESULTS_ROC_WIDTH: Final[int] = 5
RESULTS_ROC_HEIGHT: Final[int] = 5

RESULTS_IMPORTANCE_FILENAME: Final[str] = RESULTS_FIGURE_FOLDER + "feature_importances.png"

RESULTS_IMPORTANCE_WIDTH: Final[int] = 20
RESULTS_IMPORTANCE_HEIGHT: Final[int] = 5
