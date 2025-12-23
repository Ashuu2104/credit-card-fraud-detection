from flask import Flask, request, jsonify
from flask_cors import CORS
from fraud_detection_model import FraudDetectionModel
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
try:
    model_data = FraudDetectionModel.load_saved_model('fraud_detection_model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    model_name = model_data['model_name']
except Exception as e:
    print(f"Warning: Could not load pre-trained model: {e}")
    model = None
    scaler = None
    model_name = None

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Credit Card Fraud Detection API',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/predict',
            'health': '/health',
            'model_info': '/model_info'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    return jsonify({
        'model_name': model_name,
        'input_features': 30,
        'output': 'fraud_probability (0-1)'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a transaction is fraudulent
    
    Expected JSON format:
    {
        "features": [f1, f2, ..., f30]  # List of 30 features
    }
    
    Returns:
    {
        "prediction": "FRAUDULENT" or "LEGITIMATE",
        "fraud_probability": float (0-1),
        "is_fraud": boolean
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing required field: features',
                'expected_format': {
                    'features': 'array of 30 numbers'
                }
            }), 400
        
        features = data['features']
        
        # Validate features
        if not isinstance(features, list):
            return jsonify({'error': 'Features must be a list'}), 400
        
        if len(features) != 30:
            return jsonify({
                'error': f'Expected 30 features, got {len(features)}'
            }), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_scaled)[0][1]
        else:
            probability = abs(model.decision_function(features_scaled)[0])
            probability = 1 / (1 + np.exp(-probability))  # Sigmoid
        
        result = {
            'prediction': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability),
            'is_fraud': bool(prediction),
            'confidence': float(max(model.predict_proba(features_scaled)[0])) if hasattr(model, 'predict_proba') else float(probability)
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict fraud for multiple transactions
    
    Expected JSON format:
    {
        "transactions": [
            {"features": [f1, f2, ..., f30]},
            {"features": [f1, f2, ..., f30]},
            ...
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Missing required field: transactions'}), 400
        
        transactions = data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        results = []
        
        for i, transaction in enumerate(transactions):
            try:
                if 'features' not in transaction:
                    results.append({
                        'index': i,
                        'error': 'Missing features'
                    })
                    continue
                
                features = transaction['features']
                
                if len(features) != 30:
                    results.append({
                        'index': i,
                        'error': f'Expected 30 features, got {len(features)}'
                    })
                    continue
                
                # Prepare and predict
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                prediction = model.predict(features_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(features_scaled)[0][1]
                else:
                    probability = abs(model.decision_function(features_scaled)[0])
                    probability = 1 / (1 + np.exp(-probability))
                
                results.append({
                    'index': i,
                    'prediction': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
                    'fraud_probability': float(probability),
                    'is_fraud': bool(prediction)
                })
            
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(transactions),
            'processed': sum(1 for r in results if 'error' not in r),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
