# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This is the first project of the Machine Learning DevOps Engineer Nanodegree at Udacity. The main objetive is to create a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

This project is based on a Kaggle dataset ([click here for more details](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)) about identifying credit card customers that are most likely to churn.

## Files and data description

The project structure tree is shown below:

```
./MLDOE_Predict_Customer_Churn/
|
├── data/                               # Store the datasets
│   └── bank_data.csv                   # Dataset for this problem with 22 columns and 10,127 rows
|
├── images/                             # Store the images
│   ├── eda/                            # Store the images of the Exploratory Data Analysis
│   └── results/                        # Store the images of the modeling process
|
├── logs/                               # Store the logs
|
├── models/                             # Store the models generated
|
├── .gitignore                          # Specifies untracked files that Git should ignore
|
├── churn_library.py                    # Python module with the code refactored in functions
|
├── churn_notebook.ipynb                # Jupyter notebook containing the original code that will be refactored
|
├── churn_script_logging_and_tests.py   # Python module that runs the tests and generate the logs
|
├── constants.py                        # Python module with constant values used in the churn_library.py module
|
├── LICENSE                             # MIT License
|
├── README.md                           # Readme file of the project
|
└── requirements.txt                    # Store information about all the libraries used to develop the project
```

## Requirements

Python Libraries used for modeling process:

- scikit-learn (0.24.1)
- shap (0.40.0)
- joblib (1.0.1)
- pandas (1.2.4)
- numpy (1.20.1)
- matplotlib (3.3.4)
- seaborn (0.11.2)

Python Libraries used for code quality and tests:

- pylint (2.7.4)
- autopep8 (1.5.6)
- pytest (7.1.2)

To install all of the requirements:

```
$ pip install -r requirements.txt
```

## Running Files

To run all of the steps of the process

```
$ ipython churn_library.py
```

The plots stored:

```
./MLDOE_Predict_Customer_Churn/
|
├── images/
│   ├── eda/
│   |   ├── churn_distribution.png              # Churn Distribution
│   |   ├── customer_age_distribution.png       # Customer Age Distribution
│   |   ├── heatmap.png                         # Heatmap - Correlations
│   |   ├── marital_status_distribution.png     # Marital Status Distributions
│   |   └── total_transaction_distribution.png  # Total transactions Distributions
│   └── results/
|       ├── ...                                 # Store the images of the modeling process
```

- `churn_distribution.png`
  ![Churn Distribution](./images/eda/churn_distribution.png)
- `customer_age_distribution.png`
  ![Customer Age Distribution](./images/eda/customer_age_distribution.png)
- `heatmap.png`
  ![Heatmap - Correlations](./images/eda/heatmap.png)
- `marital_status_distribution.png`
  ![Marital Status Distributions](./images/eda/marital_status_distribution.png)
- `total_transaction_distribution.png`
  ![Total transactions Distributions](./images/eda/total_transaction_distribution.png)

The models stored:

```
./MLDOE_Predict_Customer_Churn/
|
├── models/
|   ├─ logistic_model.pkl               # Logistic Regression model
|   └─ rfc_model.pkl                    # Random Forest Classifier model
```

To test the churn_library run the followin command:

```
$ ipython churn_script_logging_and_tests.py
```

The output should be:

```
================================================= test session starts =================================================
platform win32 -- Python 3.8.10, pytest-7.1.2, pluggy-1.0.0 -- C:\Users\rudim\AppData\Local\Programs\Python\Python38\python.exe
cachedir: .pytest_cache
rootdir: D:\my_stuff\current\sandbox\cursos\udacity\machine_learning_devops_engineer\repos\MLDOE_Predict_Customer_Churn
collected 5 items

churn_script_logging_and_tests.py::TestClassChurnLibrary::test_import PASSED                                     [ 20%]
churn_script_logging_and_tests.py::TestClassChurnLibrary::test_eda PASSED                                        [ 40%]
churn_script_logging_and_tests.py::TestClassChurnLibrary::test_encoder_helper PASSED                             [ 60%]
churn_script_logging_and_tests.py::TestClassChurnLibrary::test_perform_feature_engineering PASSED                [ 80%]
churn_script_logging_and_tests.py::TestClassChurnLibrary::test_train_models PASSED                               [100%]

================================================== 5 passed in 0.06s ==================================================
```

The log file `./logs/churn_library.log` content should be like:

```
[2022-08-17 04:18:53] root - INFO - Testing import_data: SUCCESS
```

## License

The contents of this repository are covered under the [MIT License](LICENSE).
