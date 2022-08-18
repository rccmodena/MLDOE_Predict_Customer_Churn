"""
Module with constant values used in the churn_library.py module

Author: Rudi César Comitto Modena
Date: August, 2022
"""

from typing import Final

# EDA settings
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

RESULTS_FIGURE_FOLDER: Final[str] = "./images/results/"

MODELS_FOLDER: Final[str] = "./models/"

# Categorical Columns
CAT_COLUMNS: Final[list] = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

# Remove Columns
REMOVE_COLUMNS: Final[list] = [
    'Unnamed: 0',
    'CLIENTNUM',
    'Attrition_Flag',
]

# Modeling
TEST_SIZE: Final[float] = 0.3
RANDOM_STATE: Final[int] = 42

LRC_SOLVER: Final[str] = 'lbfgs'
LRC_MAX_ITER: Final[int] = 3000

PARAM_GRID: Final[dict] = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

CROSS_VALID: Final[int] = 5

RFC_MODEL_FILENAME: Final[str] = './models/rfc_model.pkl'
LOGISTIC_MODEL_FILENAME: Final[str] = './models/logistic_model.pkl'