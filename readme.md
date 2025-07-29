# Customer Churn Prediction

This project uses machine learning to predict customer churn for a fictional telecommunications company. By identifying customers who are likely to leave, the company can proactively offer them incentives to stay.

***
## Dataset
The project uses the "Telco Customer Churn" dataset, which contains customer account information, demographic data, and services they have signed up for.

***
## Project Workflow
1.  **Data Cleaning & EDA**: The data was cleaned by handling missing values and incorrect data types. Exploratory Data Analysis (EDA) was performed to understand the data distributions and correlations between features.
2.  **Feature Engineering**: Categorical features were converted into numerical format using `LabelEncoder`.
3.  **Handling Imbalanced Data**: The dataset was highly imbalanced. The **SMOTE** (Synthetic Minority Over-sampling Technique) was used on the training data to create a balanced dataset.
4.  **Model Training & Evaluation**: Three different classification models were trained and evaluated using 5-fold cross-validation:
    * Decision Tree
    * Random Forest
    * XGBoost
5.  **Model Saving**: The best-performing model and the data encoders were saved using `pickle` for future use.

***
## Installation

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/sanskar-502/Customer-Churn-Prediction.git](https://github.com/sanskar-502/Customer-Churn-Prediction.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd YourRepositoryName
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

***
## Usage
To use the trained model for predictions:

1.  Ensure you have the `customer_churn_model.pkl` and `encoders.pkl` files in your directory.
2.  Use the following Python script to make predictions on new data:
    ```python
    import pandas as pd
    import pickle

    # Load the model and encoders
    with open("customer_churn_model.pkl", "rb") as f_model:
        model = pickle.load(f_model)

    with open("encoders.pkl", "rb") as f_encoders:
        encoders = pickle.load(f_encoders)

    # Example new customer data
    new_data = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    # Convert to DataFrame and encode
    input_df = pd.DataFrame([new_data])
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    # Make prediction
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        print("Prediction: Customer will Churn")
    else:
        print("Prediction: Customer will not Churn")
    ```
***