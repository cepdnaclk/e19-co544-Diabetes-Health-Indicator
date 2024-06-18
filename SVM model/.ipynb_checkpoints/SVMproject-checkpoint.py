import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Select relevant columns
    features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits'
    ]
    X = df[features]
    y = df['Diabetes_012']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Standardize the feature variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, label_encoder, scaler

def train_svm_model(X_train, X_test, y_train, y_test):
    # Initialize and train the SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict_new_data(model, new_data, scaler, label_encoder):
    # Standardize the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    
    # Convert predictions to original labels
    predictions_labels = label_encoder.inverse_transform(predictions)
    
    return predictions_labels

def get_user_input():
    HighBP = float(input("Enter HighBP (0 = no high BP, 1 = high BP): "))
    HighChol = float(input("Enter HighChol (0 = no high cholesterol, 1 = high cholesterol): "))
    CholCheck = float(input("Enter CholCheck (0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years): "))
    BMI = float(input("Enter BMI: "))
    Smoker = float(input("Enter Smoker (0 = no, 1 = yes): "))
    Stroke = float(input("Enter Stroke (0 = no, 1 = yes): "))
    HeartDiseaseorAttack = float(input("Enter HeartDiseaseorAttack (0 = no, 1 = yes): "))
    PhysActivity = float(input("Enter PhysActivity (0 = no, 1 = yes): "))
    Fruits = float(input("Enter Fruits (0 = no, 1 = yes): "))
    
    # Create a new data array
    new_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits]])
    return new_data

def map_prediction_to_stage(prediction):
    stages = {0: 'no diabetes', 1: 'prediabetes', 2: 'diabetes'}
    return stages[prediction]

# Main execution
def main():
    # Load and preprocess the data
    X, y, label_encoder, scaler = load_and_preprocess_data('diabetes_012_health_indicators_BRFSS2015.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the SVM model
    model = train_svm_model(X_train, X_test, y_train, y_test)
    
    # Get user input
    new_data = get_user_input()
    
    # Predict diabetes stage for new user input
    predictions = predict_new_data(model, new_data, scaler, label_encoder)
    prediction_stage = map_prediction_to_stage(predictions[0])
    
    print("\nPrediction for the new data:")
    print(prediction_stage)

if __name__ == "__main__":
    main()
