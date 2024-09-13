import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

#Load dataset
path = "C:\\Users\\SOL-62\\Desktop\\Financial Risk Analysis\\financial_risk_analysis_large.csv"
data = load_data(path)


data.head()

def clean_negative_values(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    mask = (data[numerical_columns] < 0).any(axis=1)
    data_cleaned = data[~mask]
    return data_cleaned
data = clean_negative_values(data)

# Apply the encoding function to your dataset
def encode_categorical_data(data):
    # Copy the original data to avoid modifying it directly
    data_encoded = data.copy()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # One-Hot Encoding for nominal categorical features
    data_encoded = pd.get_dummies(data_encoded, columns=categorical_columns, drop_first=True)
    return data_encoded
data_encoded = encode_categorical_data(data)


def perform_EDA(data):
    # Specify the directory where you want to save the figures
    save_dir = r"C:\Users\SOL-62\Desktop\Financial Risk Analysis"
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Basic summary of the data
    def basic_summary(data):
        print("Shape of the data:", data.shape)
        print("Data types:\n", data.dtypes)
        print("First few rows:\n", data.head())
        print("Summary statistics:\n", data.describe())
        print("Missing values:\n", data.isnull().sum())
        print("Unique value counts per column:\n", data.nunique())
    basic_summary(data)

    # Plot histograms for numerical features
    def plot_histograms(data):
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        numeric_data.hist(figsize=(26, 18), bins=30, edgecolor='black')
        plt.suptitle('Histograms of Numerical Features', y=1.02)
        plt.savefig(os.path.join(save_dir, 'histograms.png'))  # Save the figure
        plt.show()
    plot_histograms(data)

    # Plot boxplots for numerical features
    def plot_boxplots(data):
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(16, 12))
        sns.boxplot(data=numeric_data, orient="h", palette="Set2")
        plt.title('Boxplots of Numerical Features')
        plt.savefig(os.path.join(save_dir, 'boxplots.png'))  # Save the figure
        plt.show()
    plot_boxplots(data)

    # Full correlation matrix plot
    def plot_full_correlation_matrix(data):
        encoded_data = pd.get_dummies(data, drop_first=True)
        corr_matrix = encoded_data.corr()
        plt.figure(figsize=(18, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title('Full Correlation Matrix')
        plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))  # Save the figure
        plt.show()
    plot_full_correlation_matrix(data)

    # Correlation with 'LoanApproved' feature
    def plot_correlation_with_loan_approved(data):
        # Encode categorical features for correlation
        encoded_data = pd.get_dummies(data, drop_first=True)
        
        # Calculate correlations with the 'LoanApproved' column
        loan_approved_corr = encoded_data.corr()['LoanApproved'].sort_values(ascending=False)

        # Plot the correlation of each feature with 'LoanApproved'
        plt.figure(figsize=(10, 6))
        sns.barplot(x=loan_approved_corr.index, y=loan_approved_corr.values, palette="coolwarm")
        plt.xticks(rotation=90)
        plt.title('Correlation of Features with LoanApproved')
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Features')
        plt.savefig(os.path.join(save_dir, 'correlation_with_loan_approved.png'))  # Save the figure
        plt.show()
    plot_correlation_with_loan_approved(data)

perform_EDA(data_encoded)


# Feature Engineering
def create_feature_engineering(data):
    # feature engineering
    data['TotalDebt'] = data['MortgageBalance'] + data['AutoLoanBalance'] + data['PersonalLoanBalance'] + data['StudentLoanBalance']
    data['SavingsToDebtRatio'] = np.where(data['TotalDebt'] == 0, 0, data['SavingsAccountBalance'] / data['TotalDebt'])
    data['MonthlyIncomeToDebtRatio'] = np.where(data['MonthlyDebtPayments'] == 0, 0, data['AnnualIncome'] / (12 * data['MonthlyDebtPayments']))
    return data
create_feature_engineering(data)

# Function to clean the data
def clean_data(data):
    data = data.dropna()
    data = data.drop_duplicates()
    return data

# Function to define numerical and categorical features
def get_feature_lists():
    numerical_features = ['CreditScore', 'AnnualIncome', 'LoanAmount', 'LoanDuration', 'Age', 
                          'DebtToIncomeRatio', 'CreditCardUtilizationRate',
                          'NumberOfOpenCreditLines', 'LengthOfCreditHistory', 'TotalAssets', 
                          'NetWorth', 'MonthlySavings', 'TotalDebt']
    
    categorical_features = ['EmploymentStatus', 'EducationLevel', 
                            'HomeOwnershipStatus', 'LoanPurpose']
    
    return numerical_features, categorical_features

# Function to define preprocessing steps
def define_preprocessing(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

# Function to separate features and target variable
def separate_features_target(data, numerical_features, categorical_features):
    X = data[numerical_features + categorical_features]  # Features
    y = data['LoanApproved']  # Target variable
    return X, y

# Function to preprocess features
def preprocess_features(preprocessor, X):
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

# Function to convert target variable
def convert_target(y):
    y = y.astype(np.float32)
    return y

# Function to split the data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Main function to orchestrate the data preparation process
def prepare_data(data):
    # Clean data
    data = clean_data(data)
    
    # Get feature lists
    numerical_features, categorical_features = get_feature_lists()
    
    # Define preprocessing steps
    preprocessor = define_preprocessing(numerical_features, categorical_features)
    
    # Separate features and target variable
    X, y = separate_features_target(data, numerical_features, categorical_features)
    
    # Apply preprocessing
    X_preprocessed = preprocess_features(preprocessor, X)
    
    # Convert target variable
    y = convert_target(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X_preprocessed, y)
    
    return X_train, X_test, y_train, y_test


# Function to build the model
def build_model(input_shape):
    model = Sequential()
    
    # Input layer and first hidden layer
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Second hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Third hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Fourth hidden layer
    model.add(Dense(16, activation='relu'))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Function to compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main function to bring everything together
def main(X_train, X_test, y_train, y_test):
    # Build, compile, and train the model
    model = build_model(X_train.shape[1])
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train)
    
    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test)

# Example usage with data
# Assuming `X_preprocessed`, `y`, `X_train`, `X_test`, `y_train`, `y_test` are already prepared as shown in the initial code
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Call the main function
main(X_train, X_test, y_train, y_test)