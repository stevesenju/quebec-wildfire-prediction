import pandas as pd
import numpy as np
import requests
import joblib
from datetime import date

print("1. Initializing Live Fire Risk Script...")

#Coordinates for Val-d'Or / Abitibi region
LATITUDE = 48.1
LONGITUDE = -77.78
today = date.today()

#Fetch today's weather + the last 14 days
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_min",
    "past_days": 14,
    "forecast_days": 1, # Just gets today
    "timezone": "America/Toronto"
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    daily = response.json().get("daily", {})
    
    temps = daily.get("temperature_2m_max", [])
    precips = daily.get("precipitation_sum", [])
    winds = daily.get("wind_speed_10m_max", [])
    hums = daily.get("relative_humidity_2m_min", [])
    
    #memory features
    curr_temp = temps[-1]
    curr_precip = precips[-1]
    curr_wind = winds[-1]
    curr_hum = hums[-1]
    
    past_precips = precips[:-1]
    past_temps = temps[:-1]
    
    rain_14d = sum(p for p in past_precips if pd.notnull(p))
    avg_temp_14d = np.nanmean(past_temps)
    
    days_since = 14
    for i, p in enumerate(reversed(past_precips)):
        if p > 1.0:
            days_since = i + 1
            break

    #Format the Random Forest expects
    live_features = pd.DataFrame([{
        'LATITUDE': LATITUDE,
        'LONGITUDE': LONGITUDE,
        'MONTH': today.month,
        'MAX_TEMP': curr_temp,
        'PRECIPITATION': curr_precip,
        'WIND_SPEED': curr_wind,
        'MIN_HUMIDITY': curr_hum,
        '14_DAY_RAIN': rain_14d,
        '14_DAY_AVG_TEMP': avg_temp_14d,
        'DAYS_SINCE_RAIN': days_since
    }])

    #Load model and predict
    print("2. Loading Model...")
    model = joblib.load('model\quebec_fire_predictor.pkl')
    
    prediction = model.predict(live_features)[0]
    confidence = model.predict_proba(live_features)[0][1] * 100
    
   #Output Results
    print("\n" + "="*40)
    print("LIVE FIRE RISK ASSESSMENT")
    print("="*40)
    print(f"Location: Abitibi (Lat: {LATITUDE}, Lon: {LONGITUDE})")
    print(f"Date:     {today}")
    print("-" * 40)
    print(f"Current Temp:      {curr_temp}°C")
    print(f"Days Since Rain:   {days_since} days")
    print(f"14-Day Rain Total: {rain_14d:.1f} mm")
    print("-" * 40)
    
    if prediction == 1:
        print(f"FIRE IGNITION RISK")
        print(f"Confidence: {confidence:.1f}%")
    else:
        print(f"STATUS: SAFE")
        print(f"Probability of fire: {confidence:.1f}%")
    print("="*40 + "\n")

except FileNotFoundError:
    print("\n Model not found.")
except Exception as e:
    print(f"\n Failed to fetch weather data or run prediction. Details: {e}")