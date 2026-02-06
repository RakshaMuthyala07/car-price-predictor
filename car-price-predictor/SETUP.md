# Setup Instructions

## Quick Start Guide

Follow these steps to set up and run the Car Price Predictor project on your local machine.

### Prerequisites
- Python 3.8 or higher installed on your system
- Git installed (optional, for cloning)
- Basic understanding of command line/terminal

---

## Installation Steps

### 1. Download the Project

**Option A: Using Git**
```bash
git clone https://github.com/RakshaMuthyala07/car-price-predictor.git
cd car-price-predictor
```

**Option B: Download ZIP**
- Click the green "Code" button on GitHub
- Select "Download ZIP"
- Extract the ZIP file
- Navigate to the extracted folder in terminal/command prompt

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (machine learning)
- streamlit (dashboard)
- matplotlib & seaborn (visualizations)

### 4. Train the Model
```bash
python train_model.py
```

**Expected Output:**
- Console logs showing data preprocessing steps
- Model training progress
- Performance metrics
- Generated files: `car_price_model.pkl`, `encoder.pkl`, `feature_columns.pkl`
- Visualizations saved in `visualizations/` folder

**Time:** ~30-60 seconds depending on your system

### 5. Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, copy the URL from terminal and paste it in your browser.

---

## Using the Dashboard

### Making a Prediction:
1. **Navigate to "Price Prediction" tab**
2. **Select car details:**
   - Company (e.g., Maruti, Hyundai, Ford)
   - Manufacturing Year (slider)
   - Kilometers Driven (number input)
   - Fuel Type (Petrol/Diesel/CNG)
3. **Click "Predict Price" button**
4. **View results:**
   - Predicted price
   - Price range
   - Comparison with similar cars

### Exploring Data:
- **Data Insights tab:** View visualizations and statistics
- **Model Performance tab:** See model evaluation metrics
- **About tab:** Learn about the project

---

## Troubleshooting

### Issue: "Module not found" error
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Streamlit port already in use
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### Issue: Model files not found
**Solution:** Make sure you ran `train_model.py` first
```bash
python train_model.py
```

### Issue: Dataset not found
**Solution:** Ensure `quikr_car.csv` is in the project root directory

---

## File Structure After Setup

```
car-price-predictor/
│
├── app.py                      ✅ Dashboard app
├── train_model.py              ✅ Training script
├── quikr_car.csv               ✅ Dataset
├── requirements.txt            ✅ Dependencies
├── README.md                   ✅ Documentation
│
├── car_price_model.pkl         ⭐ Generated after training
├── encoder.pkl                 ⭐ Generated after training
├── feature_columns.pkl         ⭐ Generated after training
│
└── visualizations/             ⭐ Generated after training
    ├── price_distribution.png
    ├── year_vs_price.png
    ├── fuel_type_analysis.png
    ├── company_analysis.png
    ├── correlation_heatmap.png
    └── model_comparison.png
```

---

## Next Steps

### For Resume/Portfolio:
1. ✅ Train the model
2. ✅ Take screenshots of the dashboard
3. ✅ Push to your GitHub repository
4. ✅ Add project link to your resume
5. ✅ Prepare to explain the project in interviews

### For Further Development:
- Add more features (transmission type, owner count)
- Implement hyperparameter tuning
- Create API endpoints using Flask/FastAPI
- Deploy on Streamlit Cloud or Heroku
- Add user authentication

---

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Search for similar issues on Stack Overflow
4. Contact: rakshamuthyala@gmail.com

---

**Happy Coding! 🚀**
