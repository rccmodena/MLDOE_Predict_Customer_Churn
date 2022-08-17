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
EDA_FIGURA_HEIGHT: Final[int] = 10

EDA_CHURN_DISTRIB_FILENAME: Final[str] = "./images/eda/churn_distribution.png"

EDA_HEATMAP_FILENAME: Final[str] = "./images/eda/heatmap.png"

EDA_CUSTOMER_AGE_VARIABLE: Final[str] = "Customer_Age"
EDA_CUSTOMER_AGE_DISTRIB_FILENAME: Final[str] = "./images/eda/customer_age_distribution.png"

EDA_MARITAL_STATUS_VARIABLE: Final[str] = "Marital_Status"
EDA_MARITAL_STATUS_DISTR_FILENAME: Final[str] = "./images/eda/marital_status_distribution.png"

EDA_TOTAL_TRANSACTION_VARIABLE: Final[str] = "Total_Trans_Ct"
EDA_TOTAL_TRANSACTION_FILENAME: Final[str] = "./images/eda/total_transaction_distribution.png"

# # Categorical Columns
# cat_columns: Final[tuple] = (
#     'Gender',
#     'Education_Level',
#     'Marital_Status',
#     'Income_Category',
#     'Card_Category'                
# )

# # Categorical Columns
# quant_columns: Final[tuple] = (
#     'Customer_Age',
#     'Dependent_count', 
#     'Months_on_book',
#     'Total_Relationship_Count', 
#     'Months_Inactive_12_mon',
#     'Contacts_Count_12_mon', 
#     'Credit_Limit', 
#     'Total_Revolving_Bal',
#     'Avg_Open_To_Buy', 
#     'Total_Amt_Chng_Q4_Q1', 
#     'Total_Trans_Amt',
#     'Total_Trans_Ct', 
#     'Total_Ct_Chng_Q4_Q1', 
#     'Avg_Utilization_Ratio'
# )