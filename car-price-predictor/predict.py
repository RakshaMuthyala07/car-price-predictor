"""
Standalone Prediction Script
Usage: python predict.py
Author: Raksha Muthyala
"""

import pandas as pd
import pickle

def load_model():
    """Load the trained model and encoder"""
    try:
        with open('car_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, encoder, feature_columns
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'train_model.py' first.")
        exit(1)

def predict_price(company, year, kms_driven, fuel_type):
    """
    Predict car price based on input features
    
    Parameters:
    -----------
    company : str
        Car manufacturer (e.g., 'Maruti', 'Hyundai', 'Ford')
    year : int
        Manufacturing year (e.g., 2018)
    kms_driven : int
        Kilometers driven (e.g., 30000)
    fuel_type : str
        Fuel type ('Petrol', 'Diesel', 'CNG')
    
    Returns:
    --------
    float : Predicted price in INR
    """
    model, encoder, feature_columns = load_model()
    
    # Prepare input
    input_data = pd.DataFrame({
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })
    
    # Encode categorical features
    encoded_features = encoder.transform(input_data[['company', 'fuel_type']])
    encoded_df = pd.DataFrame(encoded_features, 
                              columns=encoder.get_feature_names_out(['company', 'fuel_type']))
    
    # Combine with numeric features
    X_input = pd.concat([input_data[['year', 'kms_driven']].reset_index(drop=True), 
                        encoded_df.reset_index(drop=True)], axis=1)
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in X_input.columns:
            X_input[col] = 0
    
    X_input = X_input[feature_columns]
    
    # Predict
    prediction = model.predict(X_input)[0]
    return prediction

if __name__ == "__main__":
    print("="*60)
    print("CAR PRICE PREDICTOR - STANDALONE SCRIPT")
    print("="*60)
    
    # Example predictions
    print("\n📋 Example Predictions:\n")
    
    examples = [
        {"company": "Maruti", "year": 2018, "kms_driven": 30000, "fuel_type": "Petrol"},
        {"company": "Hyundai", "year": 2016, "kms_driven": 45000, "fuel_type": "Diesel"},
        {"company": "Ford", "year": 2015, "kms_driven": 60000, "fuel_type": "Diesel"},
    ]
    
    for i, example in enumerate(examples, 1):
        price = predict_price(**example)
        print(f"{i}. {example['company']} {example['year']} ({example['fuel_type']}, {example['kms_driven']:,} km)")
        print(f"   💰 Predicted Price: ₹{price:,.2f}\n")
    
    # Interactive prediction
    print("="*60)
    print("🔮 Make Your Own Prediction")
    print("="*60)
    
    try:
        company = input("\n🏢 Enter Company (e.g., Maruti, Hyundai, Ford): ").strip()
        year = int(input("📅 Enter Year (e.g., 2018): "))
        kms_driven = int(input("🛣️ Enter Kilometers Driven (e.g., 30000): "))
        fuel_type = input("⛽ Enter Fuel Type (Petrol/Diesel/CNG): ").strip()
        
        price = predict_price(company, year, kms_driven, fuel_type)
        
        print("\n" + "="*60)
        print("✅ PREDICTION RESULT")
        print("="*60)
        print(f"\n💰 Predicted Price: ₹{price:,.2f}")
        print(f"📊 Price Range: ₹{price*0.9:,.2f} - ₹{price*1.1:,.2f}")
        print("\n" + "="*60)
        
    except ValueError as e:
        print(f"\n❌ Error: Invalid input. Please enter correct values.")
    except KeyboardInterrupt:
        print("\n\n👋 Prediction cancelled.")
