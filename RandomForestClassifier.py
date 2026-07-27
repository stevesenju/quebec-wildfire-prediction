import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data\wildfire_ml_engineered_data.csv")
df.head(5)
df.isna().sum()
features = df[[
    'LATITUDE', 'LONGITUDE', 'MONTH', 'MAX_TEMP', 
    'PRECIPITATION', 'WIND_SPEED', 'MIN_HUMIDITY', 
    '14_DAY_RAIN', '14_DAY_AVG_TEMP', 'DAYS_SINCE_RAIN'
]]

model = RandomForestClassifier(n_estimators=1000, random_state=35)

X = features
y = df['FIRE_RISK']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=35)

model.fit(X_train, y_train)
prediction = model.predict(X_test)
score = accuracy_score(y_test,prediction)
score
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)



#Package Model
joblib.dump(model, 'data\quebec_fire_predictor.pkl')
print("Model saved as quebec_fire_predictor.pkl")