**Project Title: Customer Churn Prediction**

Project Overview: This project involves building a machine learning model to predict customer churn for a telecommunications company. The process includes data loading, exploration, preprocessing, model training, evaluation, and building a predictive system.

Steps Involved:

Importing Dependencies:

Import necessary libraries for data manipulation (pandas, numpy), visualization (matplotlib, seaborn), machine learning (sklearn, imblearn, xgboost), and saving/loading models (pickle).
Data Loading and Understanding:

Load the customer churn dataset from a CSV file into a pandas DataFrame.
Examine the shape, head, info, and unique values of the DataFrame to understand the data structure and content.
Identify and handle missing values in the 'TotalCharges' column by replacing empty strings with '0.0' and converting the column to a float type.
Check the class distribution of the target variable ('Churn') to identify potential class imbalance.
Exploratory Data Analysis (EDA):

Analyze the distribution of numerical features ('tenure', 'MonthlyCharges', 'TotalCharges') using histograms and box plots. Calculate and visualize the mean and median on histograms.
Examine the correlation between numerical features using a heatmap.
Analyze the distribution of categorical features using count plots.
Data Preprocessing:

Label encode the target variable ('Churn') by mapping 'Yes' to 1 and 'No' to 0.
Label encode the categorical features using LabelEncoder and save the encoders to a pickle file for later use in the predictive system.
Split the data into training and testing sets using train_test_split.
Apply the Synthetic Minority Over-sampling Technique (SMOTE) to the training data to address class imbalance.
Model Training and Selection:

Initialize a dictionary of machine learning models: Decision Tree, Random Forest, and XGBoost.
Train each model on the SMOTE-oversampled training data using 5-fold cross-validation to evaluate their performance with default hyperparameters.
Select the best-performing model based on the mean cross-validation accuracy.
Fit the best model on the entire SMOTE-oversampled training data.
Save the trained model and the list of feature names to a pickle file.
Model Evaluation:

Evaluate the performance of the best model on the unseen test data.
Calculate and print evaluation metrics such as accuracy score, confusion matrix, and classification report.
Predictive System:

Load the saved model and encoders from the pickle files.
Create a function or code block to take new input data (representing a customer's details).
Preprocess the input data using the loaded encoders.
Use the loaded model to make predictions (churn or no churn) on the preprocessed input data.
Print the prediction and prediction probability.
How to Use:

Clone the repository to your local machine.
Ensure you have the necessary libraries installed (pandas, numpy, matplotlib, seaborn, sklearn, imblearn, xgboost, pickle).
Run the notebook cells sequentially to execute the data processing, model training, and evaluation steps.
Use the predictive system section to test the model with new customer data.