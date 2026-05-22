import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("HousingData.csv")

data = data.dropna()

X = data.drop("MEDV", axis=1)
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("===== Boston House Price Prediction =====")

CRIM = float(input("Enter crime rate: "))
ZN = float(input("Enter residential land zone value: "))
INDUS = float(input("Enter industrial area proportion: "))
CHAS = int(input("Near Charles River? (0 or 1): "))
NOX = float(input("Enter nitric oxide concentration: "))
RM = float(input("Enter average number of rooms: "))
AGE = float(input("Enter house age: "))
DIS = float(input("Enter distance to employment centers: "))
RAD = int(input("Enter highway accessibility index: "))
TAX = float(input("Enter property tax rate: "))
PTRATIO = float(input("Enter pupil-teacher ratio: "))
B = float(input("Enter proportion of black population value: "))
LSTAT = float(input("Enter lower status population percentage: "))

user_data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM,
                           AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
columns=X.columns)

prediction = model.predict(user_data)

print("\nPredicted House Price:", prediction[0])
