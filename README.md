# Québec Wildfire Risk Prediction (Abitibi Region)

An end-to-end Machine Learning pipeline that predicts daily wildfire ignition risk across Québec by combining official provincial fire data (**SOPFEU**) with historical weather APIs (**Open-Meteo ERA5**). 

Instead of relying solely on single-day weather metrics, this project engineers **14-day rolling drought memory features** to achieve **96%+ classification accuracy**.

---

##Features & Architecture

* **Automated Data Pipeline:** Batches spatial coordinates into grid locations and extracts multi-year weather timelines via the Open-Meteo API.
* **Feature Engineering:** Computes 14-day cumulative rainfall, 14-day mean maximum temperatures, and consecutive days since rain (>1mm).
* **Model Benchmarking:** Evaluates both **Random Forest Classifier** and **XGBoost Classifier** against a balanced 9,000-instance dataset.

---

## Dataset & Feature Importance

The final dataset consists of **9,000 instances** (5,000 historical fires + 4,000 safe pseudo-absence days) spanning from 2018 to 2024.

Dataset is publicly available on [Kaggle]https://www.kaggle.com/datasets/tashasteve/quebec-wildfire-prediction-dataset.

### Key Predictors
1. **Latitude / Longitude:** Geographical climate zone context.
2. **Minimum Relative Humidity (`MIN_HUMIDITY`):** Primary driver for fuel dryness.
3. **Maximum Daily Temperature (`MAX_TEMP`):** Thermal intensity factor.
4. **14-Day Average Temperature (`14_DAY_AVG_TEMP`):** Forest canopy dry-out indicator.
5. **Days Since Rain (`DAYS_SINCE_RAIN`):** Fuel ignition susceptibility.

---
# This project is built strictly for educational and portfolio demonstration purposes. It is not an official early-warning system. Always refer to official SOPFEU guidelines and alerts for official fire safety decisions.


##Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/stevesenju/quebec-wildfire-prediction
cd quebec-wildfire-prediction
pip install -r requirements.txt



