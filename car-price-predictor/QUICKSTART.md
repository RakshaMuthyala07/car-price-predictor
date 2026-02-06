# 🚀 QUICK START GUIDE

## Installation & Running (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Model (First Time Only)
```bash
python train_model.py
```
⏱️ Takes ~30-60 seconds

### Step 3: Launch Dashboard
```bash
streamlit run app.py
```
🌐 Opens at: http://localhost:8501

---

## File Structure

```
car-price-predictor/
├── 🎯 Main Files
│   ├── train_model.py          # Train ML model
│   ├── app.py                  # Streamlit dashboard
│   ├── predict.py              # Standalone predictions
│   └── quikr_car.csv           # Dataset
│
├── 📚 Documentation
│   ├── README.md               # Main documentation
│   ├── SETUP.md                # Detailed setup guide
│   ├── DOCUMENTATION.md        # Resume & interview guide
│   └── PROJECT_SUMMARY.md      # Complete project overview
│
├── ⚙️ Configuration
│   ├── requirements.txt        # Python packages
│   ├── .gitignore             # Git ignore rules
│   └── LICENSE                # MIT License
│
└── 🤖 Generated (after training)
    ├── car_price_model.pkl     # Trained model
    ├── encoder.pkl             # Categorical encoder
    ├── feature_columns.pkl     # Feature names
    └── visualizations/         # EDA plots (6 images)
```

---

## Commands Cheat Sheet

| Task | Command |
|------|---------|
| Install packages | `pip install -r requirements.txt` |
| Train model | `python train_model.py` |
| Run dashboard | `streamlit run app.py` |
| Quick prediction | `python predict.py` |
| Check version | `python --version` |

---

## Dashboard Features

### 1️⃣ Price Prediction
- Select car specs → Get instant price
- View price range (±10%)
- Compare with similar cars

### 2️⃣ Data Insights
- Price distribution charts
- Year vs Price analysis
- Fuel type breakdown
- Top car companies

### 3️⃣ Model Performance
- R² Score comparison
- Error metrics (MAE, RMSE)
- Model evaluation charts

### 4️⃣ About
- Project overview
- Technical details
- Author info

---

## For Resume

**Short Description:**
```
Car Price Predictor - ML model predicting used car prices 
with Random Forest (Python, scikit-learn, Streamlit)
```

**GitHub Link:**
```
https://github.com/RakshaMuthyala07/car-price-predictor
```

---

## Troubleshooting

**Issue:** Module not found
```bash
pip install --upgrade -r requirements.txt
```

**Issue:** Port in use
```bash
streamlit run app.py --server.port 8502
```

**Issue:** Model not found
```bash
python train_model.py
```

---

## Next Steps

1. ✅ Push to GitHub
2. ✅ Add to resume
3. ✅ Prepare demo
4. ✅ Deploy to Streamlit Cloud (optional)

---

**Author:** Raksha Muthyala  
**Email:** rakshamuthyala@gmail.com  
**GitHub:** @RakshaMuthyala07
