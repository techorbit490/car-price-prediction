"""
╔══════════════════════════════════════════════════════════════╗
║           CarValueAI — server.py                            ║
║   ONE FILE: Flask Backend + MySQL Setup + ML Training       ║
╠══════════════════════════════════════════════════════════════╣
║  Steps:                                                      ║
║  1. pip install flask mysql-connector-python scikit-learn   ║
║           pandas numpy                                       ║
║  2. Edit DB_CONFIG below (set your MySQL password)          ║
║  3. python server.py                                         ║
║  4. Open browser: http://localhost:5000                      ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
#  STEP 0 — Auto install missing packages
# ─────────────────────────────────────────────────────────────
import subprocess, sys

REQUIRED = ["flask", "mysql-connector-python", "scikit-learn", "pandas", "numpy"]

def install_missing():
    for pkg in REQUIRED:
        try:
            __import__(pkg.replace("-", "_").split("-")[0])
        except ImportError:
            print(f"[INSTALL] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

install_missing()

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import os, json, pickle, random, uuid, traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file

import numpy as np
import pandas as pd
import mysql.connector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
#  ✅ CONFIG — Change your MySQL password here
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":      "localhost",
    "port":      3306,
    "user":      "root",
    "password":  "python server.py",   # ← CHANGE THIS
    "charset":   "utf8mb4",
    "autocommit": True,
}
DB_NAME   = "car_price_db"
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  DATASET GENERATOR
# ─────────────────────────────────────────────────────────────
BRANDS = {
    "Maruti":     ["Swift","Baleno","Dzire","Wagon R","Alto","Vitara Brezza","Ertiga"],
    "Hyundai":    ["i20","Creta","Venue","Verna","Grand i10","Tucson","i10"],
    "Honda":      ["City","Amaze","Jazz","WR-V","CR-V","Civic"],
    "Toyota":     ["Innova","Fortuner","Corolla","Camry","Glanza","Urban Cruiser"],
    "Ford":       ["EcoSport","Endeavour","Aspire","Figo","Freestyle"],
    "Volkswagen": ["Polo","Vento","Tiguan","T-Roc","Ameo"],
    "BMW":        ["3 Series","5 Series","X1","X3","X5","7 Series"],
    "Mercedes":   ["C-Class","E-Class","GLA","GLC","S-Class","A-Class"],
    "Audi":       ["A4","A6","Q3","Q5","Q7","A3"],
    "Tata":       ["Nexon","Harrier","Safari","Altroz","Punch","Tiago"],
    "Kia":        ["Seltos","Sonet","Carnival","EV6"],
    "MG":         ["Hector","ZS EV","Gloster","Astor"],
    "Mahindra":   ["Scorpio","XUV500","Thar","XUV300","Bolero","XUV700"],
    "Renault":    ["Duster","Kwid","Triber","Kiger"],
    "Nissan":     ["Magnite","Kicks","Terrano","Datsun GO"],
}
BASE_P = {
    "Maruti":400000,"Hyundai":500000,"Honda":600000,"Toyota":700000,"Ford":550000,
    "Volkswagen":650000,"BMW":3000000,"Mercedes":3500000,"Audi":3200000,"Tata":450000,
    "Kia":700000,"MG":800000,"Mahindra":600000,"Renault":400000,"Nissan":450000,
}
ENG_R = {
    "Maruti":(800,1500),"Hyundai":(1000,2000),"Honda":(1200,2000),"Toyota":(1500,2700),
    "Ford":(1000,2000),"Volkswagen":(1000,2000),"BMW":(1500,4400),"Mercedes":(1500,4700),
    "Audi":(1400,4000),"Tata":(1000,2000),"Kia":(1200,2200),"MG":(1500,2000),
    "Mahindra":(1500,2500),"Renault":(800,1500),"Nissan":(1000,1600),
}
FUELS   = ["Petrol","Diesel","CNG","Electric","Hybrid"]
TRANS   = ["Manual","Automatic","CVT","DCT"]
OWNERS  = ["First Owner","Second Owner","Third Owner","Fourth & Above Owner"]
SELLERS = ["Individual","Dealer","Trustmark Dealer"]

def generate_dataset(n=3000):
    random.seed(42); np.random.seed(42)
    rows = []
    for _ in range(n):
        b   = random.choice(list(BRANDS))
        m   = random.choice(BRANDS[b])
        yr  = random.randint(2005, 2023)
        age = 2024 - yr
        fu  = random.choice(FUELS)
        tr  = random.choice(TRANS)
        ow  = random.choice(OWNERS)
        sl  = random.choice(SELLERS)
        km  = random.randint(1000, 300000)
        eng = random.randint(*ENG_R[b])
        hp  = round(eng * random.uniform(0.06, 0.12), 1)
        se  = random.choice([2,4,5,6,7,8])
        ml  = round(random.uniform(8, 35), 1)
        dep = max(0.2, 1 - age * 0.07)
        kmf = max(0.3, 1 - km / 500000)
        ff  = {"Petrol":1,"Diesel":1.1,"CNG":0.9,"Electric":1.3,"Hybrid":1.2}[fu]
        tf  = {"Manual":1,"Automatic":1.15,"CVT":1.12,"DCT":1.18}[tr]
        of  = {"First Owner":1,"Second Owner":0.85,"Third Owner":0.7,"Fourth & Above Owner":0.55}[ow]
        ef  = 1 + (eng - 1000) / 10000
        pr  = max(50000, round(BASE_P[b] * dep * kmf * ff * tf * of * ef * random.uniform(0.85, 1.15)))
        rows.append([b,m,yr,fu,tr,ow,sl,km,eng,hp,ml,se,pr])
    cols = ["Brand","Model","Year","Fuel_Type","Transmission","Owner","Seller_Type",
            "KM_Driven","Engine_CC","Max_Power_HP","Mileage_KMPL","Seats","Selling_Price"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["Mileage_KMPL","Engine_CC","Max_Power_HP","Seats"]:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, c] = np.nan
    return df

# ─────────────────────────────────────────────────────────────
#  ML TRAINING
# ─────────────────────────────────────────────────────────────
CAT_COLS  = ["Brand","Model","Fuel_Type","Transmission","Owner","Seller_Type"]
NUM_COLS  = ["Year","KM_Driven","Engine_CC","Max_Power_HP","Mileage_KMPL","Seats",
             "Car_Age","Age_Km_Ratio","Power_Per_CC"]
FEAT_COLS = CAT_COLS + NUM_COLS

ml_model = None; ml_encoders = {}; ml_imputer = None; ml_meta = {}

def train_models(df):
    global ml_model, ml_encoders, ml_imputer, ml_meta
    print("\n[ML] Starting training...")
    df = df.drop_duplicates()
    df = df[(df.Selling_Price > 0) & (df.KM_Driven > 0)]
    q1, q3 = df.Selling_Price.quantile(0.01), df.Selling_Price.quantile(0.99)
    df = df[(df.Selling_Price >= q1) & (df.Selling_Price <= q3)]
    for c in ["Mileage_KMPL","Engine_CC","Max_Power_HP","Seats"]:
        df[c] = df[c].fillna(df[c].median())
    df["Car_Age"]      = 2024 - df["Year"]
    df["Age_Km_Ratio"] = df["Car_Age"] / (df["KM_Driven"] / 10000 + 1)
    df["Power_Per_CC"] = df["Max_Power_HP"] / df["Engine_CC"]

    encoders = {}
    for c in CAT_COLS:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    imp = SimpleImputer(strategy="median")
    df[NUM_COLS] = imp.fit_transform(df[NUM_COLS])

    X = df[FEAT_COLS]; y = df["Selling_Price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    models_to_train = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
    }
    results = {}; best_name = None; best_r2 = -np.inf; best_mdl = None
    for name, mdl in models_to_train.items():
        print(f"   Training {name}...")
        mdl.fit(Xtr, ytr); yp = mdl.predict(Xte)
        mae  = mean_absolute_error(yte, yp)
        rmse = mean_squared_error(yte, yp) ** 0.5
        r2   = r2_score(yte, yp)
        mape = np.mean(np.abs((yte - yp) / yte)) * 100
        results[name] = {"MAE":round(mae,2),"RMSE":round(rmse,2),"R2":round(r2,4),"MAPE":round(mape,2)}
        print(f"   {name}: R²={r2:.4f}  MAE=₹{mae:,.0f}")
        if r2 > best_r2: best_r2, best_name, best_mdl = r2, name, mdl

    fi = {}
    if hasattr(best_mdl, "feature_importances_"):
        fi = dict(sorted(zip(FEAT_COLS, best_mdl.feature_importances_), key=lambda x: -x[1]))
        fi = {k: round(float(v), 6) for k, v in fi.items()}

    print(f"\n[ML] Best: {best_name}  R²={best_r2:.4f}")

    with open(os.path.join(MODEL_DIR,"model.pkl"),"wb")    as f: pickle.dump(best_mdl, f)
    with open(os.path.join(MODEL_DIR,"encoders.pkl"),"wb") as f: pickle.dump(encoders, f)
    with open(os.path.join(MODEL_DIR,"imputer.pkl"),"wb")  as f: pickle.dump(imp, f)

    meta = {
        "best_model_name": best_name, "best_r2": round(best_r2,4),
        "feature_cols": FEAT_COLS, "cat_cols": CAT_COLS, "num_cols": NUM_COLS,
        "model_results": results, "feature_importance": fi,
        "train_size": len(Xtr), "test_size": len(Xte), "total_samples": len(df)
    }
    with open(os.path.join(MODEL_DIR,"meta.json"),"w") as f: json.dump(meta, f, indent=2)

    ml_model = best_mdl; ml_encoders = encoders; ml_imputer = imp; ml_meta = meta
    print("[ML] All artifacts saved!\n")
    return meta

def load_models():
    global ml_model, ml_encoders, ml_imputer, ml_meta
    mp = os.path.join(MODEL_DIR,"model.pkl")
    ep = os.path.join(MODEL_DIR,"encoders.pkl")
    ip = os.path.join(MODEL_DIR,"imputer.pkl")
    jp = os.path.join(MODEL_DIR,"meta.json")
    if not all(os.path.exists(p) for p in [mp,ep,ip,jp]):
        return False
    with open(mp,"rb") as f: ml_model    = pickle.load(f)
    with open(ep,"rb") as f: ml_encoders = pickle.load(f)
    with open(ip,"rb") as f: ml_imputer  = pickle.load(f)
    with open(jp)      as f: ml_meta     = json.load(f)
    print(f"[ML] Loaded: {ml_meta.get('best_model_name')}  R²={ml_meta.get('best_r2')}")
    return True

# ─────────────────────────────────────────────────────────────
#  MYSQL SETUP
# ─────────────────────────────────────────────────────────────
def get_conn(with_db=True):
    cfg = {**DB_CONFIG}
    if with_db: cfg["database"] = DB_NAME
    return mysql.connector.connect(**cfg)

def db_exec(sql, params=None, fetch=False):
    conn = get_conn(); cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, params or ())
        if fetch: return cur.fetchall()
        conn.commit(); return cur.lastrowid
    finally:
        cur.close(); conn.close()

def setup_database():
    print("[DB] Setting up MySQL...")
    # Create database
    conn = get_conn(with_db=False); cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit(); cur.close(); conn.close()

    # Create tables
    db_exec("""CREATE TABLE IF NOT EXISTS cars (
        id INT AUTO_INCREMENT PRIMARY KEY,
        brand VARCHAR(60), model VARCHAR(80), year SMALLINT,
        fuel_type VARCHAR(20), transmission VARCHAR(20),
        owner VARCHAR(40), seller_type VARCHAR(30),
        km_driven INT, engine_cc FLOAT, max_power_hp FLOAT,
        mileage_kmpl FLOAT, seats TINYINT, selling_price BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_brand(brand), INDEX idx_year(year)
    ) ENGINE=InnoDB""")

    db_exec("""CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(64),
        brand VARCHAR(60), model VARCHAR(80), year SMALLINT,
        fuel_type VARCHAR(20), transmission VARCHAR(20),
        owner VARCHAR(40), seller_type VARCHAR(30),
        km_driven INT, engine_cc FLOAT, max_power_hp FLOAT,
        mileage_kmpl FLOAT, seats TINYINT,
        predicted_price BIGINT, confidence_low BIGINT, confidence_high BIGINT,
        model_used VARCHAR(60), r2_score FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_created(created_at)
    ) ENGINE=InnoDB""")

    db_exec("""CREATE TABLE IF NOT EXISTS brands_models (
        id INT AUTO_INCREMENT PRIMARY KEY,
        brand VARCHAR(60) NOT NULL, model VARCHAR(80) NOT NULL,
        UNIQUE KEY uq(brand,model)
    ) ENGINE=InnoDB""")

    print("[DB] Tables ready!")

def seed_brands_models():
    for brand, models in BRANDS.items():
        for m in models:
            try:
                db_exec("INSERT IGNORE INTO brands_models(brand,model) VALUES(%s,%s)", (brand,m))
            except: pass
    print("[DB] Brand/model data seeded!")

def seed_cars(df):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cars"); cnt = cur.fetchone()[0]
    if cnt > 0: print(f"[DB] Cars table has {cnt} records, skipping seed."); cur.close(); conn.close(); return
    sql = """INSERT INTO cars(brand,model,year,fuel_type,transmission,owner,seller_type,
             km_driven,engine_cc,max_power_hp,mileage_kmpl,seats,selling_price)
             VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r.get("Brand","")),str(r.get("Model","")),int(r.get("Year",2020)),
            str(r.get("Fuel_Type","Petrol")),str(r.get("Transmission","Manual")),
            str(r.get("Owner","First Owner")),str(r.get("Seller_Type","Individual")),
            int(r.get("KM_Driven",0)),float(r.get("Engine_CC",0) or 0),
            float(r.get("Max_Power_HP",0) or 0),float(r.get("Mileage_KMPL",0) or 0),
            int(r.get("Seats",5) or 5),int(r.get("Selling_Price",0))
        ))
    cur.executemany(sql, rows); conn.commit()
    print(f"[DB] Seeded {len(rows)} cars into MySQL!")
    cur.close(); conn.close()

def save_metrics_to_db(meta):
    try:
        db_exec("""CREATE TABLE IF NOT EXISTS model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY, model_name VARCHAR(60),
            r2_score FLOAT, mae FLOAT, rmse FLOAT, mape FLOAT,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB""")
        db_exec("DELETE FROM model_metrics")
        for name, m in meta.get("model_results",{}).items():
            db_exec("INSERT INTO model_metrics(model_name,r2_score,mae,rmse,mape) VALUES(%s,%s,%s,%s,%s)",
                    (name, m["R2"], m["MAE"], m["RMSE"], m["MAPE"]))
        print("[DB] Model metrics saved!")
    except Exception as e:
        print(f"[DB] Could not save metrics: {e}")

# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return response

@app.route("/api/<path:p>", methods=["OPTIONS"])
def options(p): return jsonify({}), 200

# ── Serve frontend ──
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ── Health ──
@app.route("/api/health")
def health():
    db_ok = False
    try: conn=get_conn(); conn.close(); db_ok=True
    except: pass
    return jsonify({
        "status":"ok","db":"connected" if db_ok else "error",
        "ml_model": ml_meta.get("best_model_name","not loaded"),
        "r2": ml_meta.get("best_r2",0),
        "timestamp": datetime.now().isoformat()
    })

# ── Brands / Models ──
@app.route("/api/brands")
def brands():
    rows = db_exec("SELECT DISTINCT brand FROM brands_models ORDER BY brand", fetch=True)
    return jsonify([r["brand"] for r in rows])

@app.route("/api/models/<brand>")
def models_for_brand(brand):
    rows = db_exec("SELECT model FROM brands_models WHERE brand=%s ORDER BY model",(brand,),fetch=True)
    return jsonify([r["model"] for r in rows])

# ── PREDICT ──
@app.route("/api/predict", methods=["POST"])
def predict():
    if not ml_model:
        return jsonify({"error":"ML model not loaded"}), 503
    d = request.get_json()
    required = ["brand","model_name","year","fuel_type","transmission","owner",
                "seller_type","km_driven","engine_cc","max_power_hp","mileage_kmpl","seats"]
    for f in required:
        if f not in d: return jsonify({"error":f"Missing: {f}"}), 400
    try:
        inp = {
            "Brand":d["brand"],"Model":d["model_name"],"Year":int(d["year"]),
            "Fuel_Type":d["fuel_type"],"Transmission":d["transmission"],
            "Owner":d["owner"],"Seller_Type":d["seller_type"],
            "KM_Driven":int(d["km_driven"]),"Engine_CC":float(d["engine_cc"]),
            "Max_Power_HP":float(d["max_power_hp"]),"Mileage_KMPL":float(d["mileage_kmpl"]),
            "Seats":int(d["seats"]),
        }
        df = pd.DataFrame([inp])
        df["Car_Age"]      = 2024 - df["Year"]
        df["Age_Km_Ratio"] = df["Car_Age"] / (df["KM_Driven"] / 10000 + 1)
        df["Power_Per_CC"] = df["Max_Power_HP"] / df["Engine_CC"]
        for col in CAT_COLS:
            le = ml_encoders.get(col)
            if le:
                val = str(df[col].iloc[0])
                df[col] = int(le.transform([val])[0]) if val in le.classes_ else -1
        df[NUM_COLS] = ml_imputer.transform(df[NUM_COLS])
        X     = df[[c for c in FEAT_COLS if c in df.columns]]
        price = float(ml_model.predict(X)[0])
        price = round(max(50000, price))
        cl, ch = round(price*0.92), round(price*1.08)
        sid = str(uuid.uuid4())
        try:
            db_exec("""INSERT INTO predictions
                (session_id,brand,model,year,fuel_type,transmission,owner,seller_type,
                 km_driven,engine_cc,max_power_hp,mileage_kmpl,seats,
                 predicted_price,confidence_low,confidence_high,model_used,r2_score)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (sid,inp["Brand"],inp["Model"],inp["Year"],inp["Fuel_Type"],inp["Transmission"],
                 inp["Owner"],inp["Seller_Type"],inp["KM_Driven"],inp["Engine_CC"],
                 inp["Max_Power_HP"],inp["Mileage_KMPL"],inp["Seats"],
                 price,cl,ch,ml_meta.get("best_model_name",""),ml_meta.get("best_r2",0)))
        except Exception as e: print(f"[DB] Save error: {e}")
        return jsonify({
            "predicted_price":price,"confidence_low":cl,"confidence_high":ch,
            "model_used":ml_meta.get("best_model_name",""),"r2_score":ml_meta.get("best_r2",0),
            "car_age":int(2024-int(d["year"])),"session_id":sid
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 500

# ── EDA APIs ──
@app.route("/api/eda/summary")
def eda_summary():
    r = db_exec("""SELECT COUNT(*) total_cars, AVG(selling_price) avg_price,
        MIN(selling_price) min_price, MAX(selling_price) max_price,
        AVG(km_driven) avg_km, AVG(2024-year) avg_age,
        COUNT(DISTINCT brand) brands_count,
        COUNT(DISTINCT CONCAT(brand,model)) models_count FROM cars""", fetch=True)
    row = r[0] if r else {}
    return jsonify({k:(float(v) if v else 0) for k,v in row.items()})

@app.route("/api/eda/brand_avg_price")
def eda_brand():
    r = db_exec("""SELECT brand, ROUND(AVG(selling_price)) avg_price, COUNT(*) cnt
        FROM cars GROUP BY brand ORDER BY avg_price DESC LIMIT 15""", fetch=True)
    return jsonify([dict(x) for x in r])

@app.route("/api/eda/fuel_distribution")
def eda_fuel():
    r = db_exec("SELECT fuel_type, COUNT(*) cnt FROM cars GROUP BY fuel_type ORDER BY cnt DESC", fetch=True)
    return jsonify([dict(x) for x in r])

@app.route("/api/eda/year_trend")
def eda_year():
    r = db_exec("SELECT year, ROUND(AVG(selling_price)) avg_price, COUNT(*) cnt FROM cars GROUP BY year ORDER BY year", fetch=True)
    return jsonify([dict(x) for x in r])

@app.route("/api/eda/transmission_price")
def eda_trans():
    r = db_exec("SELECT transmission, ROUND(AVG(selling_price)) avg_price, COUNT(*) cnt FROM cars GROUP BY transmission", fetch=True)
    return jsonify([dict(x) for x in r])

@app.route("/api/eda/price_histogram")
def eda_hist():
    r = db_exec("SELECT FLOOR(selling_price/200000)*200000 bucket, COUNT(*) cnt FROM cars GROUP BY bucket ORDER BY bucket", fetch=True)
    return jsonify([dict(x) for x in r])

# ── ML Metrics ──
@app.route("/api/feature_importance")
def feat_imp():
    fi = ml_meta.get("feature_importance", {})
    return jsonify([{"feature":k,"importance":v} for k,v in fi.items()])

@app.route("/api/model_metrics")
def model_metrics():
    res = ml_meta.get("model_results", {})
    return jsonify(sorted([{"model":k,**v} for k,v in res.items()], key=lambda x:-x["R2"]))

# ── History ──
@app.route("/api/history")
def history():
    limit = int(request.args.get("limit", 100))
    r = db_exec("""SELECT id, brand, model, year, fuel_type, transmission,
        owner, seller_type, km_driven, engine_cc, max_power_hp, mileage_kmpl,
        seats, predicted_price, confidence_low, confidence_high, model_used, r2_score,
        DATE_FORMAT(created_at,'%%d %%b %%Y %%H:%%i') created_at
        FROM predictions ORDER BY id DESC LIMIT %s""", (limit,), fetch=True)
    return jsonify([dict(x) for x in r])

@app.route("/api/history/clear", methods=["DELETE"])
def clear_history():
    db_exec("DELETE FROM predictions")
    return jsonify({"status":"cleared"})

@app.route("/api/history/<int:pid>", methods=["DELETE"])
def del_pred(pid):
    db_exec("DELETE FROM predictions WHERE id=%s", (pid,))
    return jsonify({"deleted":pid})

# ─────────────────────────────────────────────────────────────
#  MAIN — STARTUP SEQUENCE
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CarValueAI — Full Stack Server Starting...")
    print("="*55)

    # Step 1: MySQL Setup
    db_ok = False
    try:
        print("\n[1/4] Setting up MySQL database...")
        setup_database()
        seed_brands_models()
        db_ok = True
        print("[DB] ✅ MySQL connected!")
    except Exception as e:
        print(f"\n[DB] ❌ MySQL Error: {e}")
        print("[DB] Fix: Set your MySQL password in DB_CONFIG at top of file")
        print("[DB] Server will start but database features won't work\n")

    # Step 2: Generate Dataset
    print("\n[2/4] Checking dataset...")
    df_data = None
    if db_ok:
        try:
            cnt = db_exec("SELECT COUNT(*) AS c FROM cars", fetch=True)[0]["c"]
            if cnt < 100:
                print("[DATA] Generating 3000 car records...")
                df_data = generate_dataset(3000)
                seed_cars(df_data)
            else:
                print(f"[DATA] ✅ Dataset exists ({cnt} records)")
        except Exception as e:
            print(f"[DATA] Error: {e}")

    # Step 3: Train ML Models
    print("\n[3/4] Loading/Training ML models...")
    if not load_models():
        print("[ML] Training new models (takes 1-2 minutes)...")
        if df_data is None:
            df_data = generate_dataset(3000)
        meta = train_models(df_data)
        if db_ok:
            try: save_metrics_to_db(meta)
            except: pass
    else:
        print(f"[ML] ✅ Model loaded: {ml_meta.get('best_model_name')}  R²={ml_meta.get('best_r2')}")

    # Step 4: Start Server
    print("\n[4/4] Starting Flask server...")
    print("\n" + "="*55)
    print("  ✅ CarValueAI is RUNNING!")
    print("="*55)
    print(f"\n  🌐 Open browser: http://localhost:5000")
    print(f"  🗄️  Database:     {'✅ MySQL Connected' if db_ok else '❌ MySQL Error (check password)'}")
    print(f"  🤖 ML Model:     {ml_meta.get('best_model_name','Not loaded')}  R²={ml_meta.get('best_r2',0)}")
    print(f"\n  Press Ctrl+C to stop\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
