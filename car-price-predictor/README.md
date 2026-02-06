# 🚗 Car Price Predictor using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project that predicts used car prices based on features like company, manufacturing year, kilometers driven, and fuel type. Includes an interactive **Streamlit dashboard** for real-time predictions.

![Car Price Predictor](https://img.icons8.com/fluency/96/000000/car.png)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Author](#author)
- [License](#license)

## 🎯 Overview

This project implements a **Random Forest Regressor** to predict the selling price of used cars with **89% accuracy (R² score)**. The model is trained on real-world data from Quikr car listings and deployed through an interactive web dashboard.

### Problem Statement
Predicting accurate prices for used cars is challenging due to various factors like depreciation, market demand, and vehicle condition. This project provides data-driven price estimates to help buyers and sellers make informed decisions.

## ✨ Features

- 🎯 **Accurate Price Predictions** - Uses Random Forest with R² score of 0.89
- 📊 **Interactive Dashboard** - Beautiful Streamlit web interface
- 📈 **Data Visualizations** - Comprehensive EDA with charts and graphs
- 🔧 **Multiple ML Models** - Compares Linear Regression, Decision Tree, and Random Forest
- 💾 **Model Persistence** - Saved models for quick predictions
- 📱 **Responsive Design** - Works on desktop and mobile browsers
- 🎨 **Professional UI** - Clean, modern interface with custom styling

## 🖼️ Demo

### Main Dashboard
The interactive dashboard allows users to:
- Select car company, year, kilometers driven, and fuel type
- Get instant price predictions
- View similar car prices for comparison
- Explore data insights and model performance

### Sample Prediction
```
Input:
- Company: Maruti
- Year: 2018
- Kilometers: 30,000 km
- Fuel Type: Petrol

Output:
- Predicted Price: ₹3,45,000
- Price Range: ₹3,10,500 - ₹3,79,500
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/RakshaMuthyala07/car-price-predictor.git
cd car-price-predictor
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Train the Model
First, train the machine learning model on the dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Perform exploratory data analysis
- Train multiple ML models
- Save the best model and encoders
- Generate visualization plots

**Output Files:**
- `car_price_model.pkl` - Trained Random Forest model
- `encoder.pkl` - OneHotEncoder for categorical features
- `feature_columns.pkl` - Feature column names
- `visualizations/` - Folder with all plots

### 2. Run the Dashboard
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Make Predictions
1. Select your car specifications from the dropdown menus
2. Click "Predict Price" button
3. View the predicted price and insights

## 📁 Project Structure

```
car-price-predictor/
│
├── app.py                      # Streamlit dashboard application
├── train_model.py              # Model training script
├── quikr_car.csv               # Dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore file
│
├── models/                     # Saved models (generated)
│   ├── car_price_model.pkl
│   ├── encoder.pkl
│   └── feature_columns.pkl
│
├── visualizations/             # EDA plots (generated)
│   ├── price_distribution.png
│   ├── year_vs_price.png
│   ├── fuel_type_analysis.png
│   ├── company_analysis.png
│   ├── correlation_heatmap.png
│   └── model_comparison.png
│
└── notebooks/                  # Jupyter notebooks (optional)
    └── EDA.ipynb
```

## 📊 Dataset

**Source:** Quikr Car Listings

### Features:
- `name` - Car model name
- `company` - Manufacturer (Maruti, Hyundai, Ford, etc.)
- `year` - Manufacturing year
- `Price` - Selling price in INR (target variable)
- `kms_driven` - Total kilometers driven
- `fuel_type` - Petrol, Diesel, or CNG

### Dataset Statistics:
- **Total Records:** 800+ cars
- **Companies:** 25+ brands
- **Year Range:** 2000-2024
- **Price Range:** ₹50,000 - ₹30,00,000

### Data Preprocessing:
- Removed missing values and "Ask For Price" entries
- Converted price and kilometers to numeric format
- Handled outliers using IQR method
- Applied OneHotEncoding for categorical features

## 📈 Model Performance

### Models Compared:
1. **Linear Regression** (Baseline)
2. **Decision Tree Regressor**
3. **Random Forest Regressor** ⭐ (Best Model)

### Random Forest Results:
| Metric | Value |
|--------|-------|
| **R² Score** | 0.89 |
| **Mean Absolute Error (MAE)** | ₹58,000 |
| **Root Mean Squared Error (RMSE)** | ₹76,000 |
| **Training R²** | 0.94 |

### Key Insights:
- ✅ Model explains **89% of price variance**
- ✅ Average prediction error: **₹58,000**
- ✅ No significant overfitting (Train R²: 0.94, Test R²: 0.89)
- ✅ Random Forest outperforms other models

## 🛠️ Technologies Used

### Core Technologies:
- **Python 3.8+** - Programming language
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning models

### Visualization:
- **matplotlib** - Static plots
- **seaborn** - Statistical visualizations

### Dashboard:
- **Streamlit** - Interactive web application

### Development:
- **Git** - Version control
- **GitHub** - Code hosting

## 👤 Author

**Raksha Muthyala**

- 📧 Email: rakshamuthyala@gmail.com
- 🐙 GitHub: [@RakshaMuthyala07](https://github.com/RakshaMuthyala07)
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/rakshamuthyala)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: Quikr Car Listings
- scikit-learn documentation and community
- Streamlit for the amazing dashboard framework
- Open-source ML community

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/RakshaMuthyala07/car-price-predictor/issues).

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Made with ❤️ by Raksha Muthyala**

