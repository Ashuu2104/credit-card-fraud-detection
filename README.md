# Credit Card Fraud Detection

## Overview
This machine learning project detects fraudulent credit card transactions using supervised learning classification algorithms. The project demonstrates the complete ML pipeline from data exploration to model evaluation and deployment.

## Features
- Data preprocessing and exploratory data analysis (EDA)
- Handling imbalanced datasets using SMOTE and class weights
- Multiple classification algorithms (Logistic Regression, Random Forest, XGBoost, SVM)
- Feature engineering and selection techniques
- Model evaluation with precision, recall, F1-score, and ROC-AUC metrics
- Comparison of model performance
- Hyperparameter tuning using GridSearchCV

## Dataset
- **Source**: Credit Card Fraud Detection dataset
- **Records**: 284,807 transactions
- **Features**: 30 (28 PCA-transformed features + Time + Amount)
- **Target**: Binary classification (Fraud vs Legitimate)
- **Imbalance Ratio**: 0.172% fraudulent transactions

## Technologies Used
- Python 3.8+
- Pandas, NumPy for data manipulation
- Scikit-learn for ML algorithms and preprocessing
- XGBoost for gradient boosting
- Matplotlib, Seaborn for visualization
- Jupyter Notebook for development

## Project Structure
```
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── models/
│   └── trained_model.pkl
├── requirements.txt
└── README.md
```

## Installation
```bash
git clone https://github.com/Ashuu2104/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## Usage
```bash
# Run the Jupyter notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Data_Preprocessing.ipynb
jupyter notebook notebooks/03_Model_Training.ipynb
jupyter notebook notebooks/04_Evaluation.ipynb
```

## Results
- Best Model: XGBoost with ROC-AUC score of 0.98+
- Precision: >95% for fraud detection
- Recall: >80% for fraud detection
- F1-Score: >0.87

## Key Learnings
- Handling imbalanced datasets is crucial in fraud detection
- Ensemble methods outperform individual classifiers
- Feature scaling and engineering significantly impact model performance
- Cross-validation prevents overfitting

## Future Improvements
- Implement deep learning models (Neural Networks)
- Deploy as a REST API using Flask
- Add real-time fraud detection capability
- Integrate with cloud platforms (AWS/GCP)

## Contributing
Feel free to fork this project and submit pull requests with improvements.

## License
MIT License - See LICENSE file for details

## Contact
For questions or suggestions, please open an issue or contact via GitHub.
