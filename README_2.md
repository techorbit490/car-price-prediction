# 🚗 CarValueAI — Used Car Price Prediction System

> **AI-powered used car valuation** using Machine Learning + Flask + MySQL  
> Ek hi command mein poora full-stack app run ho jaata hai!

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [File Structure](#-file-structure)
- [Installation & Setup](#-installation--setup)
- [API Endpoints](#-api-endpoints)
- [ML Models](#-ml-models)
- [History PIN](#-history-pin)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

**CarValueAI** ek intelligent used car price estimator hai jo:
- Indian car market ke **15 brands** aur **80+ models** ko support karta hai
- Multiple ML algorithms ko train karke **best model automatically select** karta hai
- Predicted price ke saath **confidence range (±8%)** bhi deta hai
- Saari predictions **MySQL database** mein save hoti hain
- Interactive **EDA (Exploratory Data Analysis)** charts dikhata hai

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML Price Prediction | Brand, model, year, km, fuel type etc. se price predict karta hai |
| 📊 EDA Dashboard | Brand-wise avg price, fuel distribution, year trend charts |
| 🏆 Model Comparison | GradientBoosting, RandomForest, ExtraTrees, Ridge ka comparison |
| 📈 Feature Importance | Konsa factor price pe sabse zyada affect karta hai |
| 🗂️ Prediction History | Saari past predictions PIN-protected history mein |
| 🔁 Auto Dataset Generation | 3000 synthetic Indian car records auto-generate hote hain |
| 🛠️ Auto Package Install | Missing Python packages khud install ho jaate hain |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Database:** MySQL (`mysql-connector-python`)
- **ML:** scikit-learn (GradientBoosting, RandomForest, ExtraTrees, Ridge)
- **Data:** pandas, numpy
- **Frontend:** HTML + CSS + JS (single file — `index.html`)

---

## 📁 File Structure

```
carvalueai_final/
├── server.py          # Main file — Flask server + MySQL setup + ML training (sab kuch!)
├── index.html         # Frontend UI (single page app)
├── requirements.txt   # Python dependencies
└── README.txt         # Quick setup guide (original)
```

> **Note:** `ml_models/` folder runtime pe automatically create hota hai jisme trained models save hote hain.

---

## 🚀 Installation & Setup

### Step 1 — Python Packages Install Karo

```bash
pip install -r requirements.txt
```

Ya manually:
```bash
pip install flask mysql-connector-python scikit-learn pandas numpy
```

> 💡 `server.py` khud bhi missing packages detect karke install kar leta hai startup par.

---

### Step 2 — MySQL Password Set Karo

`server.py` file mein sabse upar `DB_CONFIG` section mein apna MySQL password daalo:

```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "your_password",   # ← YAHAN APNA PASSWORD DAALO
    ...
}
```

---

### Step 3 — Server Start Karo

```bash
python server.py
```

Startup sequence kuch aisa dikhega:

```
=======================================================
  CarValueAI — Full Stack Server Starting...
=======================================================

[1/4] Setting up MySQL database...
[DB] ✅ MySQL connected!

[2/4] Checking dataset...
[DATA] Generating 3000 car records...

[3/4] Loading/Training ML models...
[ML] Training new models (takes 1-2 minutes)...

[4/4] Starting Flask server...
✅ CarValueAI is RUNNING!
🌐 Open browser: http://localhost:5000
```

---

### Step 4 — Browser Mein Kholo

```
http://localhost:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/brands` | Saare available brands |
| `GET` | `/api/models/<brand>` | Kisi brand ke saare models |
| `POST` | `/api/predict` | Car ki predicted price |
| `GET` | `/api/eda/summary` | Dataset ka overview |
| `GET` | `/api/eda/brand_avg_price` | Brand-wise average price |
| `GET` | `/api/eda/fuel_distribution` | Fuel type distribution |
| `GET` | `/api/eda/year_trend` | Year-wise price trend |
| `GET` | `/api/eda/transmission_price` | Transmission-wise price |
| `GET` | `/api/eda/price_histogram` | Price distribution histogram |
| `GET` | `/api/feature_importance` | ML feature importance scores |
| `GET` | `/api/model_metrics` | Saare ML models ke performance metrics |
| `GET` | `/api/history` | Past predictions list |
| `DELETE` | `/api/history/clear` | Saari history delete karo |
| `DELETE` | `/api/history/<id>` | Ek specific prediction delete karo |
| `GET` | `/api/health` | Server aur DB status check |

### Predict API — Example Request

```json
POST /api/predict
{
  "brand": "Maruti",
  "model_name": "Swift",
  "year": 2020,
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "owner": "First Owner",
  "seller_type": "Individual",
  "km_driven": 35000,
  "engine_cc": 1197,
  "max_power_hp": 82,
  "mileage_kmpl": 21.5,
  "seats": 5
}
```

### Predict API — Example Response

```json
{
  "predicted_price": 620000,
  "confidence_low": 570400,
  "confidence_high": 669600,
  "model_used": "GradientBoosting",
  "r2_score": 0.94,
  "car_age": 4,
  "session_id": "uuid-here"
}
```

---

## 🤖 ML Models

Startup par **4 models train** hote hain aur best ek automatically select hota hai:

| Model | Type |
|---|---|
| GradientBoostingRegressor | Ensemble (boosting) |
| RandomForestRegressor | Ensemble (bagging) |
| ExtraTreesRegressor | Ensemble (extremely randomized) |
| Ridge | Linear regression |

**Features used for prediction:**
- Brand, Model, Year, Fuel Type, Transmission
- Owner type, Seller type
- KM Driven, Engine CC, Max Power (HP), Mileage (KMPL), Seats
- Derived: `Car_Age`, `Age_Km_Ratio`, `Power_Per_CC`

---

## 🔐 History PIN

Prediction history dekho PIN se protected hai:

```
PIN: 2580
```

---

## ❗ Troubleshooting

**MySQL connect nahi ho raha?**
- `server.py` mein `DB_CONFIG` ka `password` field sahi set karo
- MySQL service chalu hai ya nahi check karo: `sudo service mysql start`
- App MySQL ke bina bhi run kar sakta hai — sirf DB features kaam nahi karenge

**Models train hone mein zyada time lag raha hai?**
- Pehli baar ~1-2 minutes lagenge
- Dobara run karne par models `ml_models/` folder se load ho jaate hain (fast)

**Port already in use error?**
```bash
# Koi aur process port 5000 use kar raha hai
lsof -i :5000
kill -9 <PID>
```

**Packages import error?**
```bash
pip install --upgrade flask mysql-connector-python scikit-learn pandas numpy
```

---

## 📝 Notes

- Dataset **synthetic** hai (auto-generated) — real market prices se approximate hai
- Supported brands: Maruti, Hyundai, Honda, Toyota, Ford, Volkswagen, BMW, Mercedes, Audi, Tata, Kia, MG, Mahindra, Renault, Nissan
- App `http://0.0.0.0:5000` par bind hota hai — local network pe bhi accessible hai

---

*Made with ❤️ — CarValueAI*
