import numpy as np
import os
import pandas as pd

np.random.seed(42)

TRAIN_LEN = 5000
TEST_LEN = 3000
INPUT_DIM = 1

SAVE_DIR = "data"
TRAIN_DIR = os.path.join(SAVE_DIR, "train")
TEST_DIR = os.path.join(SAVE_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# ==========================================================
# ДОПОЛНИТЕЛЬНЫЕ ГЕНЕРАТОРЫ СИГНАЛОВ
# ==========================================================

def generate_sine(length):
    t = np.arange(length)
    return np.sin(0.03 * t)

def generate_trend_noise(length):
    t = np.arange(length)
    return 0.002 * t + 0.2 * np.random.randn(length)

def generate_seasonal(length):
    t = np.arange(length)
    return 0.5*np.sin(0.01*t) + 0.5*np.sin(0.07*t)

def generate_piecewise(length):
    signal = np.zeros(length)
    for i in range(0, length, 1000):
        level = np.random.uniform(-1, 1)
        signal[i:i+1000] = level + 0.05*np.random.randn(min(1000, length-i))
    return signal

def generate_ar_process(length):
    signal = np.zeros(length)
    for i in range(1, length):
        signal[i] = 0.8 * signal[i-1] + 0.1*np.random.randn()
    return signal

def generate_small_scale(length):
    t = np.arange(length)
    signal = 0.5 + 0.4*np.sin(0.02*t)
    signal += 0.05*np.random.randn(length)
    return np.clip(signal, 0, 1)

def generate_small_noise(length):
    signal = 0.5 + 0.1*np.random.randn(length)
    return np.clip(signal, 0, 1)

def generate_drift(length):
    t = np.arange(length)
    return 0.0008*t + 0.5*np.sin(0.03*t)

def generate_complex(length):
    t = np.arange(length)
    return (
        0.4*np.sin(0.02*t) +
        0.3*np.sin(0.11*t) +
        0.001*t +
        0.1*np.random.randn(length)
    )

# ==========================================================
# НОВЫЕ ТИПЫ АНОМАЛИЙ
# ==========================================================

def inject_point_anomalies(series, n_points=20, scale=3.0):
    series = series.copy()
    intervals = []
    for _ in range(n_points):
        idx = np.random.randint(100, len(series)-100)
        series[idx] += scale*np.random.randn()
        intervals.append([int(idx), int(idx+1)])
    return series, intervals

def inject_drift(series):
    series = series.copy()
    start = np.random.randint(500, len(series)-500)
    end = start + np.random.randint(200, 500)
    series[start:end] += np.linspace(0, 3, end-start)
    return series, [[int(start), int(end)]]

def inject_flat(series):
    series = series.copy()
    start = np.random.randint(500, len(series)-500)
    end = start + np.random.randint(100, 300)
    series[start:end] = np.mean(series[start-50:start])
    return series, [[int(start), int(end)]]

# ==========================================================
# СОЗДАНИЕ 9 ДОПОЛНИТЕЛЬНЫХ КАНАЛОВ
# ==========================================================

# -----------------------------
# ДОБАВЛЕНИЕ АНОМАЛИЙ + СОХРАНЕНИЕ ИНТЕРВАЛОВ
# -----------------------------
def inject_anomalies(series, n_anomalies=3):
    series = series.copy()
    length = len(series)
    anomaly_intervals = []
    anomaly_classes = []

    for _ in range(n_anomalies):
        start = np.random.randint(200, length - 200)
        duration = np.random.randint(50, 150)
        end = min(start + duration, length)

        anomaly_type = np.random.choice([
            "contextual",
            "amplitude",
            "frequency",
            "flat",
            "noise"
        ])

        if anomaly_type == "contextual":
            series[start:end] += np.random.uniform(1.5, 2.5)

        elif anomaly_type == "amplitude":
            series[start:end] *= np.random.uniform(2.0, 3.0)

        elif anomaly_type == "frequency":
            t_local = np.arange(end - start)
            series[start:end] = (
                0.7 * np.sin(0.15 * t_local) +
                0.3 * np.sin(0.25 * t_local)
            )

        elif anomaly_type == "flat":
            series[start:end] = np.mean(series[start-50:start])

        elif anomaly_type == "noise":
            series[start:end] += 0.8 * np.random.randn(end - start)

        anomaly_intervals.append([int(start), int(end)])
        anomaly_classes.append("contextual")

    return series, anomaly_intervals, anomaly_classes



generators = [
    generate_sine,
    generate_trend_noise,
    generate_seasonal,
    generate_piecewise,
    generate_ar_process,
    generate_small_scale,
    generate_small_noise,
    generate_drift,
    generate_complex
]

# ==========================================================
# CHAN_1 (оригинальный нормальный + сложные аномалии)
# ==========================================================

def generate_normal_series(length):
    t = np.arange(length)
    signal = (
        0.7 * np.sin(0.02 * t) +
        0.3 * np.sin(0.05 * t) +
        0.001 * t
    )
    noise = 0.05 * np.random.randn(length)
    return signal + noise


# TRAIN
train_series = generate_normal_series(TRAIN_LEN).reshape(-1, 1)
np.save(os.path.join(TRAIN_DIR, "chan_1.npy"), train_series)

# TEST
test_series = generate_normal_series(TEST_LEN)
test_series, anomaly_intervals, anomaly_classes = inject_anomalies(
    test_series, n_anomalies=4
)

test_series = test_series.reshape(-1, 1)
np.save(os.path.join(TEST_DIR, "chan_1.npy"), test_series)

# создаём labels_df сразу
labels_df = pd.DataFrame({
    "chan_id": ["chan_1"],
    "spacecraft": ["SIM"],
    "anomaly_sequences": [str(anomaly_intervals)],
    "class": [str(anomaly_classes)]
})

for idx, generator in enumerate(generators, start=2):
    chan_name = f"chan_{idx}"

    # TRAIN
    train_series = generator(TRAIN_LEN).reshape(-1, 1)
    np.save(os.path.join(TRAIN_DIR, f"{chan_name}.npy"), train_series)

    # TEST
    test_series = generator(TEST_LEN)

    # разные типы аномалий
    if idx in [3, 8]:
        test_series, intervals = inject_point_anomalies(test_series)
    elif idx in [9]:
        test_series, intervals = inject_drift(test_series)
    elif idx in [5]:
        test_series, intervals = inject_flat(test_series)
    else:
        test_series, intervals, _ = inject_anomalies(test_series, n_anomalies=3)

    test_series = test_series.reshape(-1, 1)
    np.save(os.path.join(TEST_DIR, f"{chan_name}.npy"), test_series)

    # добавляем в labels
    labels_df = pd.concat([
        labels_df,
        pd.DataFrame({
            "chan_id": [chan_name],
            "spacecraft": ["SIM"],
            "anomaly_sequences": [str(intervals)],
            "class": ["synthetic"]
        })
    ])

# перезаписываем labels.csv
labels_df.to_csv("labels.csv", index=False)

print("Создано 10 каналов с различной структурой.")
# -----------------------------
# НОРМАЛЬНЫЙ СИГНАЛ
# -----------------------------
def generate_normal_series(length):
    t = np.arange(length)

    signal = (
        0.7 * np.sin(0.02 * t) +
        0.3 * np.sin(0.05 * t) +
        0.001 * t
    )

    noise = 0.05 * np.random.randn(length)

    return signal + noise

print("Данные и labels.csv успешно созданы.")