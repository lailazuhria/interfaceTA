import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("asthma_disease_data_ref.csv")

df = load_data()

# --- Sidebar ---
st.title("PERBANDINGAN PERFORMA ALGORITMA RANDOM FOREST DAN SUPPORT VECTOR MACHINE ")

col1, col2, col3 = st.columns(3)
with col1:
    train_percent = st.selectbox("Pilih persentase Training data:", [60, 70, 75, 80, 90])
with col2:
    test_percent = st.selectbox("Pilih persentase Testing data:", [5, 10, 20, 30])
with col3:
    forecast_percent = st.selectbox("Pilih persentase Forecasting data:", [5, 10])

total = train_percent + test_percent + forecast_percent

if total != 100:
    st.error(f"Total pembagian dataset harus 100%. Sekarang totalnya adalah {total}%.")
else:
    if st.button("Proses"):
        # --- Split Dataset ---
        X = df.drop("Age", axis=1)
        y = df["Age"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_percent/100), random_state=42)
        test_size_rel = forecast_percent / (test_percent + forecast_percent)
        X_test, X_forecast, y_test, y_forecast = train_test_split(X_temp, y_temp, test_size=test_size_rel, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_forecast_scaled = scaler.transform(X_forecast)

        # --- Random Forest ---
        rf_model = RandomForestClassifier()
        start = time.time()
        rf_model.fit(X_train, y_train)
        rf_train_time = time.time() - start

        y_test_pred_rf = rf_model.predict(X_test)
        rf_test_time = time.time() - start
        rf_test_acc = accuracy_score(y_test, y_test_pred_rf)
        rf_test_mae = mean_absolute_error(y_test, y_test_pred_rf)

        start = time.time()
        y_forecast_pred_rf = rf_model.predict(X_forecast)
        rf_forecast_time = time.time() - start
        rf_forecast_mae = mean_absolute_error(y_forecast, y_forecast_pred_rf)

        # --- SVM ---
        svm_model = SVC()
        start = time.time()
        svm_model.fit(X_train_scaled, y_train)
        svm_train_time = time.time() - start

        y_test_pred_svm = svm_model.predict(X_test_scaled)
        svm_test_time = time.time() - start
        svm_test_acc = accuracy_score(y_test, y_test_pred_svm)
        svm_test_mae = mean_absolute_error(y_test, y_test_pred_svm)

        start = time.time()
        y_forecast_pred_svm = svm_model.predict(X_forecast_scaled)
        svm_forecast_time = time.time() - start
        svm_forecast_mae = mean_absolute_error(y_forecast, y_forecast_pred_svm)

        # --- Display Metrics ---
        st.subheader("📊 Hasil Evaluasi")
        st.write("### Random Forest")
        st.markdown(f"- Training Time: `{rf_train_time:.4f} s`")
        st.markdown(f"- Testing Time: `{rf_test_time:.4f} s`, Accuracy: `{rf_test_acc:.4f}`, MAE: `{rf_test_mae:.4f}`")
        st.markdown(f"- Forecasting Time: `{rf_forecast_time:.4f} s`, MAE: `{rf_forecast_mae:.4f}`")

        st.write("### Support Vector Machine")
        st.markdown(f"- Training Time: `{svm_train_time:.4f} s`")
        st.markdown(f"- Testing Time: `{svm_test_time:.4f} s`, Accuracy: `{svm_test_acc:.4f}`, MAE: `{svm_test_mae:.4f}`")
        st.markdown(f"- Forecasting Time: `{svm_forecast_time:.4f} s`, MAE: `{svm_forecast_mae:.4f}`")

        # --- Show Forecasting Dataset ---
        st.subheader("📋 Tabel Data Forecasting")
        forecast_df = X_forecast.copy()
        forecast_df["RF_Prediction"] = y_forecast_pred_rf
        forecast_df["SVM_Prediction"] = y_forecast_pred_svm
        forecast_df["Actual"] = y_forecast.values
        st.dataframe(forecast_df)

        # --- Visualization ---
        st.subheader("📉 Visualisasi Hasil Prediksi")
        tab1, tab2, tab3 = st.tabs(["Training", "Testing", "Forecasting"])

        # --- Tab 1: Training ---
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(y_train.values, label="Actual", marker='o')
            ax1.plot(rf_model.predict(X_train), label="RF Prediction", linestyle='--', marker='x')
            ax1.plot(svm_model.predict(X_train_scaled), label="SVM Prediction", linestyle='--', marker='s')
            ax1.set_title("Training Data Prediction")
            ax1.set_xlabel("Sample")
            ax1.set_ylabel("Age")
            ax1.grid(True)
            ax1.legend()
            st.pyplot(fig1)

        # --- Tab 2: Testing ---
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(y_test.values, label="Actual", marker='o')
            ax2.plot(y_test_pred_rf, label="RF Prediction", linestyle='--', marker='x')
            ax2.plot(y_test_pred_svm, label="SVM Prediction", linestyle='--', marker='s')
            ax2.set_title("Testing Data Prediction")
            ax2.set_xlabel("Sample")
            ax2.set_ylabel("Age")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

        # --- Tab 3: Forecasting (sudah ada) ---
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(y_forecast.values, label="Actual", marker='o')
            ax3.plot(y_forecast_pred_rf, label="RF Prediction", linestyle='--', marker='x')
            ax3.plot(y_forecast_pred_svm, label="SVM Prediction", linestyle='--', marker='s')
            ax3.set_title("Forecasting Prediction")
            ax3.set_xlabel("Sample")
            ax3.set_ylabel("Age")
            ax3.grid(True)
            ax3.legend()
            st.pyplot(fig3)
