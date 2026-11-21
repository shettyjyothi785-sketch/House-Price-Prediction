
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score



data = {
    "location": [
        "Whitefield", "Whitefield", "Whitefield",
        "Electronic City", "Electronic City", "Electronic City",
        "Marathahalli", "Marathahalli", "Marathahalli",
        "Indiranagar", "Indiranagar", "Indiranagar",
        "BTM Layout", "BTM Layout", "BTM Layout",
        "Hebbal", "Hebbal", "Hebbal",
        "Jayanagar", "Jayanagar", "Jayanagar",
        "Sarjapur", "Sarjapur", "Sarjapur",
        "Whitefield", "Electronic City",
        "Marathahalli", "Indiranagar",
        "BTM Layout", "Hebbal"
    ],

    "total_sqft": [
        1000, 1200, 1500,
        900, 1100, 1400,
        1000, 1300, 1600,
        1200, 1500, 1800,
        950, 1150, 1450,
        1100, 1400, 1700,
        1000, 1300, 1600,
        1000, 1250, 1550,
        1800, 1600, 1800,
        2000, 1600, 1900
    ],

    "bhk": [
        2, 2, 3, 2, 2, 3,
        2, 3, 3, 2, 3, 3,
        2, 2, 3, 2, 3, 3,
        2, 3, 3, 2, 2, 3,
        3, 3, 3, 3, 3, 3
    ],

    "bath": [
        2, 2, 3, 2, 2, 3,
        2, 2, 3, 2, 3, 3,
        2, 2, 3, 2, 3, 3,
        2, 2, 3, 2, 2, 3,
        3, 3, 3, 3, 3, 3
    ],

    
    "price": [
        45, 55, 75,
        35, 42, 60,
        48, 62, 80,
        90, 115, 145,
        40, 50, 68,
        65, 85, 110,
        75, 95, 125,
        42, 52, 72,
        95, 70, 100,
        160, 85, 120
    ]
}

df = pd.DataFrame(data)

print("Dataset created successfully!")
print(df.head())



df = pd.get_dummies(
    df,
    columns=["location"],
    dtype=int
)



X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

 

predictions = model.predict(X_test)

print("\nModel Evaluation")

print(
    "MAE:",
    mean_absolute_error(y_test, predictions)
)

print(
    "R2 Score:",
    r2_score(y_test, predictions)
)


joblib.dump(model, "house_price_model.pkl")

joblib.dump(
    X.columns.tolist(),
    "house_columns.pkl"
)

print("\nModel trained successfully!")
print("Model files saved!")
