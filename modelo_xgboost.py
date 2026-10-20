import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('dataset_2024.csv')
df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True)


df_grouped = (
    df.groupby(["FECHA", "TRATAMIENTO"])
      .mean(numeric_only=True)
      .reset_index()
      .sort_values(["TRATAMIENTO", "FECHA"])
)



for lag in range(1, 5):
    df_grouped[f"Rendimiento_lag{lag}"] = (
        df_grouped.groupby("TRATAMIENTO")["RENDIMIENTO"].shift(lag)
    )

df_grouped["Rendimiento_roll2"] = (
    df_grouped.groupby("TRATAMIENTO")["RENDIMIENTO"]
    .rolling(2).mean().reset_index(0, drop=True)
)

df_grouped["DAY"] = df_grouped["FECHA"].dt.day
df_grouped["MONTH"] = df_grouped["FECHA"].dt.month


df_grouped["RENDIMIENTO_log"] = df_grouped["RENDIMIENTO"] * 31250
df_grouped["Rendimiento_future"] = df_grouped["RENDIMIENTO_log"]


df_grouped = df_grouped.dropna()
df_model = pd.get_dummies(df_grouped, columns=["TRATAMIENTO"])


X = df_model.drop(
    columns=[
        "FECHA", "Rendimiento_future",
        "RENDIMIENTO", "RENDIMIENTO_log"
    ]
)
y = df_model["Rendimiento_future"]


split_date = df_model["FECHA"].quantile(0.7)
train_mask = df_model["FECHA"] <= split_date
test_mask = df_model["FECHA"] > split_date

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]


def evaluar(y_real, predicciones, nombre):
    rmse = np.sqrt(mean_squared_error(y_real, predicciones))
    mae = mean_absolute_error(y_real, predicciones)
    r2 = r2_score(y_real, predicciones)

    print(f"\n--- {nombre} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")


from xgboost import XGBRegressor

modelo = XGBRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42
)

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
evaluar(y_test, predicciones, "XGBoost - Rendimiento")
