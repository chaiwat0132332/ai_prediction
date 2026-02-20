import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from train.forecast import DirectLinearForecast


# ==========================
# CONFIG
# ==========================

DATA_PATH = "data/raw/Ucที่ทำกับน้ำร้อนกับRfของคราบ.xlsx"   # 🔁 เปลี่ยนเป็น path ของคุณ
TARGET_COL = "U 0rpm"

LAG = 700
HORIZON = 200


# ==========================
# LOAD DATA
# ==========================

print("Loading data...")
df = pd.read_excel(DATA_PATH)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())

series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

print("Series length:", len(series))


# ==========================
# TRAIN FORECAST MODEL
# ==========================

print("\nTraining Direct Multi-step Linear Forecast...")

model = DirectLinearForecast(lag=LAG, horizon=HORIZON)

model.fit(series)

print("Training complete")
print("Lag:", LAG)
print("Horizon:", HORIZON)


# ==========================
# FORECAST
# ==========================

print("\nRunning forecast...")

future = model.forecast(series)

print("Forecast shape:", future.shape)
print("First 5 forecast values:", future[:5])


# ==========================
# PLOT
# ==========================

plt.figure(figsize=(12, 6))

# actual data
plt.plot(series.values, label="Actual")

# forecast continuation
forecast_index = np.arange(len(series), len(series) + HORIZON)
plt.plot(forecast_index, future, '--', label="Future Forecast")

plt.title("Direct Multi-step Linear Forecast Debug")
plt.legend()
plt.grid(True)

plt.show()