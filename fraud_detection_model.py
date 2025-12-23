import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score,
    auc, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """
    Credit Card Fraud Detection Model using Machine Learning
    Provides fraud detection with high accuracy using multiple algorithms
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the fraud detection model
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, filepath):
        """
        Load credit card fraud detection dataset
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"\nClass distribution:\n{df['Class'].value_counts()}")
        return df
    
    def preprocess_data(self, df, test_size=0.2, use_smote=True):
        """
        Preprocess the data for model training
        
        Args:
            df (pd.DataFrame): Input dataset
            test_size (float): Proportion of test set
            use_smote (bool): Whether to use SMOTE for handling imbalance
        """
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("Applying SMOTE for handling imbalanced data...")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"Training data shape after SMOTE: {self.X_train.shape}")
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("\nTraining models...")
        
        # Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        print("   Logistic Regression trained!")
        
        # Random Forest
        print("2. Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print("   Random Forest trained!")
        
        # XGBoost
        print("3. Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, random_state=self.random_state, use_label_encoder=False)
        xgb.fit(self.X_train, self.y_train, verbose=0)
        self.models['XGBoost'] = xgb
        print("   XGBoost trained!")
        
        # Support Vector Machine
        print("4. Training SVM...")
        svm = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        svm.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm
        print("   SVM trained!")
    
    def evaluate_models(self):
        """
        Evaluate all trained models
        
        Returns:
            dict: Model evaluation metrics
        """
        print("\nEvaluating models...")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}...")
            print(f"{'='*50}")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(self.X_test)
            
            # Calculate metrics
            accuracy = (y_pred == self.y_test).mean()
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            # Classification report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        return results
    
    def select_best_model(self, results, metric='ROC-AUC'):
        """
        Select the best model based on a metric
        
        Args:
            results (dict): Model evaluation results
            metric (str): Metric to use for selection (default: ROC-AUC)
        """
        print(f"\nSelecting best model based on {metric}...")
        best_score = -1
        
        for model_name, metrics in results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"Best Model: {self.best_model_name} with {metric} = {best_score:.4f}")
    
    def predict_fraud(self, transaction_data):
        """
        Predict if a transaction is fraudulent
        
        Args:
            transaction_data (np.ndarray or list): Transaction features
            
        Returns:
            dict: Prediction result with probability
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Ensure proper shape
        if isinstance(transaction_data, list):
            transaction_data = np.array(transaction_data).reshape(1, -1)
        
        # Scale the data
        transaction_scaled = self.scaler.transform(transaction_data)
        
        # Make prediction
        prediction = self.best_model.predict(transaction_scaled)[0]
        probability = self.best_model.predict_proba(transaction_scaled)[0][1] if hasattr(self.best_model, 'predict_proba') else abs(self.best_model.decision_function(transaction_scaled)[0])
        
        result = {
            'prediction': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability),
            'is_fraud': bool(prediction)
        }
        
        return result
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_saved_model(filepath):
        """
        Load a previously saved model
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            dict: Loaded model data
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model_data


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION MODEL")
    print("="*60)
    
    # Initialize model
    fraud_detector = FraudDetectionModel()
    
    # Note: Replace 'creditcard.csv' with actual dataset path
    # Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
    try:
        # Load data
        df = fraud_detector.load_data('creditcard.csv')
        
        # Preprocess
        fraud_detector.preprocess_data(df, test_size=0.2, use_smote=True)
        
        # Train models
        fraud_detector.train_models()
        
        # Evaluate
        results = fraud_detector.evaluate_models()
        
        # Select best model
        fraud_detector.select_best_model(results, metric='ROC-AUC')
        
        # Save the best model
        fraud_detector.save_model('fraud_detection_model.pkl')
        
        # Example prediction
        print("\n" + "="*60)
        print("MAKING A TEST PREDICTION")
        print("="*60)
        
        # Create a sample transaction (should have 30 features)
        sample_transaction = np.zeros(30)  # Placeholder
        prediction_result = fraud_detector.predict_fraud(sample_transaction)
        
        print(f"\nPrediction Result:")
        print(f"Status: {prediction_result['prediction']}")
        print(f"Fraud Probability: {prediction_result['fraud_probability']:.4f}")
        print(f"Is Fraud: {prediction_result['is_fraud']}")
        
    except FileNotFoundError:
        print("\nNote: Dataset file 'creditcard.csv' not found.")
        print("Please download the dataset from Kaggle and place it in the project directory.")
        print("Dataset URL: https://www.kaggle.com/mlg-ulb/creditcardfraud")
