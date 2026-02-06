"""
Car Price Predictor - Model Training Script
Author: Raksha Muthyala
Email: rakshamuthyala@gmail.com
GitHub: RakshaMuthyala07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("CAR PRICE PREDICTOR - MACHINE LEARNING PROJECT")
print("Author: Raksha Muthyala")
print("="*60)

# 1. LOAD DATA
print("\n1. Loading Dataset...")
df = pd.read_csv('quikr_car.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData Info:")
print(df.info())

# 2. DATA PREPROCESSING
print("\n" + "="*60)
print("2. DATA PREPROCESSING")
print("="*60)

# Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Create a backup
df_original = df.copy()

# Clean year column - remove non-numeric values
print("\nCleaning 'year' column...")
df['year'] = pd.to_numeric(df['year'], errors='coerce')
print(f"Non-numeric years converted to NaN: {df['year'].isnull().sum()}")

# Clean Price column
print("\nCleaning 'Price' column...")
df['Price'] = df['Price'].astype(str).str.replace(',', '')
# Remove 'Ask For Price' entries
df = df[df['Price'] != 'Ask For Price']
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
print(f"Invalid prices removed")

# Clean kms_driven column
print("\nCleaning 'kms_driven' column...")
df['kms_driven'] = df['kms_driven'].astype(str).str.replace(',', '').str.replace(' kms', '').str.replace('kms', '')
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

# Drop rows with NaN values
print(f"\nRows before removing NaN: {len(df)}")
df = df.dropna()
print(f"Rows after removing NaN: {len(df)}")

# Remove outliers using IQR method
print("\nRemoving outliers...")
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'Price')
df = remove_outliers(df, 'kms_driven')
df = remove_outliers(df, 'year')

print(f"Rows after outlier removal: {len(df)}")

# Reset index
df = df.reset_index(drop=True)

# Clean company and fuel_type
df['company'] = df['company'].str.strip()
df['fuel_type'] = df['fuel_type'].str.strip()

print("\nData cleaning completed!")
print(f"Final dataset shape: {df.shape}")
print("\nCleaned data summary:")
print(df.describe())

# 3. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*60)
print("3. EXPLORATORY DATA ANALYSIS")
print("="*60)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# Price Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Price'], bins=50, edgecolor='black', color='skyblue')
plt.xlabel('Price (₹)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Car Prices', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['Price'])
plt.ylabel('Price (₹)', fontsize=12)
plt.title('Price Boxplot', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/price_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Price distribution plot saved")

# Year vs Price
plt.figure(figsize=(12, 6))
plt.scatter(df['year'], df['Price'], alpha=0.5, color='coral')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Price (₹)', fontsize=12)
plt.title('Car Year vs Price', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/year_vs_price.png', dpi=300, bbox_inches='tight')
print("✓ Year vs Price plot saved")

# Fuel Type Analysis
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
fuel_counts = df['fuel_type'].value_counts()
plt.bar(fuel_counts.index, fuel_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution by Fuel Type', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
fuel_price = df.groupby('fuel_type')['Price'].mean().sort_values(ascending=False)
plt.bar(fuel_price.index, fuel_price.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Average Price (₹)', fontsize=12)
plt.title('Average Price by Fuel Type', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/fuel_type_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Fuel type analysis plot saved")

# Company Analysis (Top 10)
plt.figure(figsize=(14, 6))
top_companies = df['company'].value_counts().head(10)
plt.subplot(1, 2, 1)
plt.bar(range(len(top_companies)), top_companies.values, color='steelblue')
plt.xlabel('Company', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Top 10 Car Companies', fontsize=14, fontweight='bold')
plt.xticks(range(len(top_companies)), top_companies.index, rotation=45, ha='right')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
company_price = df.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
plt.bar(range(len(company_price)), company_price.values, color='coral')
plt.xlabel('Company', fontsize=12)
plt.ylabel('Average Price (₹)', fontsize=12)
plt.title('Top 10 Companies by Average Price', fontsize=14, fontweight='bold')
plt.xticks(range(len(company_price)), company_price.index, rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/company_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Company analysis plot saved")

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_data = df[['year', 'Price', 'kms_driven']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved")

# 4. FEATURE ENGINEERING
print("\n" + "="*60)
print("4. FEATURE ENGINEERING")
print("="*60)

# Prepare features
X = df[['company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# One-Hot Encoding for categorical variables
print("\nApplying One-Hot Encoding...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[['company', 'fuel_type']])
encoded_df = pd.DataFrame(encoded_features, 
                          columns=encoder.get_feature_names_out(['company', 'fuel_type']))

# Combine with numeric features
X_final = pd.concat([X[['year', 'kms_driven']].reset_index(drop=True), 
                     encoded_df.reset_index(drop=True)], axis=1)

print(f"Final feature shape after encoding: {X_final.shape}")
print(f"Feature columns: {X_final.columns.tolist()[:10]}... (showing first 10)")

# 5. TRAIN-TEST SPLIT
print("\n" + "="*60)
print("5. SPLITTING DATA")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# 6. MODEL BUILDING & EVALUATION
print("\n" + "="*60)
print("6. MODEL TRAINING & EVALUATION")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
}

results = []

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Training {name}...")
    print(f"{'='*40}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluation metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"Mean Squared Error: ₹{mse:,.2f}")
    print(f"Root Mean Squared Error: ₹{rmse:,.2f}")
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    })

# Create results comparison
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# R² Score comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
ax1.bar(x_pos - 0.2, results_df['Train R²'], 0.4, label='Train R²', color='skyblue')
ax1.bar(x_pos + 0.2, results_df['Test R²'], 0.4, label='Test R²', color='coral')
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.grid(alpha=0.3)

# MAE comparison
ax2 = axes[0, 1]
ax2.bar(results_df['Model'], results_df['MAE'], color='steelblue')
ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('MAE (₹)', fontsize=12)
ax2.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(alpha=0.3)

# RMSE comparison
ax3 = axes[1, 0]
ax3.bar(results_df['Model'], results_df['RMSE'], color='lightcoral')
ax3.set_xlabel('Model', fontsize=12)
ax3.set_ylabel('RMSE (₹)', fontsize=12)
ax3.set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=15)
ax3.grid(alpha=0.3)

# Best model prediction vs actual
best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

ax4 = axes[1, 1]
ax4.scatter(y_test, y_pred_best, alpha=0.5, color='purple')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Price (₹)', fontsize=12)
ax4.set_ylabel('Predicted Price (₹)', fontsize=12)
ax4.set_title(f'{best_model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison plot saved")

# 7. SAVE THE BEST MODEL
print("\n" + "="*60)
print("7. SAVING MODELS")
print("="*60)

# Save the best model (Random Forest typically performs best)
best_model = models['Random Forest']
best_model.fit(X_final, y)  # Retrain on full dataset

with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Best model saved as 'car_price_model.pkl'")

# Save encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("✓ Encoder saved as 'encoder.pkl'")

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X_final.columns.tolist(), f)
print("✓ Feature columns saved")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Test R² Score: {results_df.loc[results_df['Test R²'].idxmax(), 'Test R²']:.4f}")
print(f"\nAll models and artifacts saved successfully!")
print("\nVisualization saved in 'visualizations/' folder")
