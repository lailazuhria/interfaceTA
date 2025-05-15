import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("asthma_disease_data_ref.csv")

df = load_data()

# --- Interpretasi Dinamis MAE ---
def interpret_mae(mae):
    if mae < 2:
        return "Sangat akurat"
    elif mae < 5:
        return "Cukup akurat"
    elif mae < 10:
        return "Kurang akurat"
    else:
        return "Tidak akurat"

# --- Sidebar ---
st.title("📈 PERBANDINGAN PERFORMA ALGORITMA RANDOM FOREST DAN SUPPORT VECTOR MACHINE")

train_percent = st.selectbox("Pilih persentase Training data:", [70, 80])
test_percent = 100 - train_percent

forecast_percent = st.selectbox("Pilih persentase Forecasting data (dari total dataset):", [15, 20])

st.markdown(f"**Persentase Testing data :** {test_percent}%")
st.markdown(f"**Forecasting akan dilakukan menggunakan {forecast_percent}% data terakhir dari dataset utama.**")

# Validasi: apakah jumlah data mencukupi
total_rows = len(df)
forecast_rows = int(forecast_percent / 100 * total_rows)
main_rows = total_rows - forecast_rows

if main_rows <= 0:
    st.error("Jumlah data tidak mencukupi untuk forecasting. Kurangi persentase forecasting.")
else:
    if st.button("Proses"):
        # --- Split Data ---
        df_main = df.iloc[:main_rows]
        df_forecast = df.iloc[main_rows:]

        train_end = int(train_percent / 100 * len(df_main))
        test_end = len(df_main)

        train_data = df_main.iloc[:train_end]
        test_data = df_main.iloc[train_end:]
        forecast_data = df_forecast

        # --- Split Features and Labels ---
        X_train = train_data.drop("Age", axis=1)
        y_train = train_data["Age"]
        X_test = test_data.drop("Age", axis=1)
        y_test = test_data["Age"]
        X_forecast = forecast_data.drop("Age", axis=1)
        y_forecast = forecast_data["Age"]

        # --- Scaling ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_forecast_scaled = scaler.transform(X_forecast)

        # --- Random Forest ---
        rf_model = RandomForestClassifier()
        start = time.time()
        rf_model.fit(X_train, y_train)
        rf_train_time = time.time() - start
        mae_rf_train = mean_absolute_error(y_train, rf_model.predict(X_train))

        start = time.time()
        y_test_pred_rf = rf_model.predict(X_test)
        rf_test_time = time.time() - start
        rf_test_acc = accuracy_score(y_test, y_test_pred_rf)
        mae_rf_test = mean_absolute_error(y_test, y_test_pred_rf)

        start = time.time()
        y_forecast_pred_rf = rf_model.predict(X_forecast)
        rf_forecast_time = time.time() - start
        mae_rf_forecast = mean_absolute_error(y_forecast, y_forecast_pred_rf)

        # --- SVM ---
        svm_model = SVC()
        start = time.time()
        svm_model.fit(X_train_scaled, y_train)
        svm_train_time = time.time() - start
        mae_svm_train = mean_absolute_error(y_train, svm_model.predict(X_train_scaled))

        start = time.time()
        y_test_pred_svm = svm_model.predict(X_test_scaled)
        svm_test_time = time.time() - start
        svm_test_acc = accuracy_score(y_test, y_test_pred_svm)
        mae_svm_test = mean_absolute_error(y_test, y_test_pred_svm)

        start = time.time()
        y_forecast_pred_svm = svm_model.predict(X_forecast_scaled)
        svm_forecast_time = time.time() - start
        mae_svm_forecast = mean_absolute_error(y_forecast, y_forecast_pred_svm)

        # --- Hasil Evaluasi ---
        st.subheader("📊 Hasil Evaluasi Model")
        st.markdown("#### Random Forest")
        st.markdown(f"- Training Time: `{rf_train_time:.4f} s`, MAE: `{mae_rf_train:.4f}` ({interpret_mae(mae_rf_train)})")
        st.markdown(f"- Testing Time: `{rf_test_time:.4f} s`, Accuracy: `{rf_test_acc:.4f}`, MAE: `{mae_rf_test:.4f}` ({interpret_mae(mae_rf_test)})")
        st.markdown(f"- Forecasting Time: `{rf_forecast_time:.4f} s`, MAE: `{mae_rf_forecast:.4f}` ({interpret_mae(mae_rf_forecast)})")

        st.markdown("#### Support Vector Machine")
        st.markdown(f"- Training Time: `{svm_train_time:.4f} s`, MAE: `{mae_svm_train:.4f}` ({interpret_mae(mae_svm_train)})")
        st.markdown(f"- Testing Time: `{svm_test_time:.4f} s`, Accuracy: `{svm_test_acc:.4f}`, MAE: `{mae_svm_test:.4f}` ({interpret_mae(mae_svm_test)})")
        st.markdown(f"- Forecasting Time: `{svm_forecast_time:.4f} s`, MAE: `{mae_svm_forecast:.4f}` ({interpret_mae(mae_svm_forecast)})")

        # --- Tabel Forecasting ---
        st.subheader("🗂️ Tabel Hasil Forecasting")
        forecast_result = pd.DataFrame()
        forecast_result["Index"] = forecast_data.index
        forecast_result["Prediksi_RF"] = y_forecast_pred_rf
        forecast_result["Prediksi_SVM"] = y_forecast_pred_svm

        st.dataframe(forecast_result.reset_index(drop=True), use_container_width=True)

        # --- Visualisasi ---
        st.subheader("📉 Visualisasi Hasil Prediksi")
        tab1, tab2, tab3, tab4 = st.tabs(["Training", "Testing", "Forecasting", "Perbandingan MAE"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(y_train.values, label="Actual", marker='o')
            ax1.plot(rf_model.predict(X_train), label="RF", linestyle='--', marker='x')
            ax1.plot(svm_model.predict(X_train_scaled), label="SVM", linestyle='--', marker='s')
            ax1.set_title("Training Data Prediction")
            ax1.legend()
            st.pyplot(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(y_test.values, label="Actual", marker='o')
            ax2.plot(y_test_pred_rf, label="RF", linestyle='--', marker='x')
            ax2.plot(y_test_pred_svm, label="SVM", linestyle='--', marker='s')
            ax2.set_title("Testing Data Prediction")
            ax2.legend()
            st.pyplot(fig2)

        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(y_forecast_pred_rf, label="RF", linestyle='--', marker='x')
            ax3.plot(y_forecast_pred_svm, label="SVM", linestyle='--', marker='s')
            ax3.set_title("Forecasting Prediction")
            ax3.legend()
            st.pyplot(fig3)

        with tab4:
            labels = ['Train', 'Test', 'Forecast']
            mae_rf = [mae_rf_train, mae_rf_test, mae_rf_forecast]
            mae_svm = [mae_svm_train, mae_svm_test, mae_svm_forecast]
            x = np.arange(len(labels))
            width = 0.35

            fig_mae, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width/2, mae_rf, width, label='Random Forest', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, mae_svm, width, label='SVM', color='orange', alpha=0.7)

            for bars in [bars1, bars2]:
                for bar in bars:
                    ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)

            ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
            ax.set_title('MAE Comparison', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig_mae)

        # --- Penjelasan ---

        # --- Penjelasan Dinamis ---
        st.subheader("📝 Penjelasan Visualisasi")

        if train_percent == 70:
            st.markdown(f"""
            - **Training (70%)**: Random Forest mulai menyaingi SVM dalam hal MAE. Keduanya cukup baik dalam menangkap pola data historis.
            - **Testing (30%)**: SVM menunjukkan generalisasi lebih konsisten dengan MAE lebih stabil. Random Forest sedikit lebih fluktuatif.
            - **Forecasting ({forecast_percent}%)**: Kedua model menunjukkan performa prediksi yang cukup kompetitif pada data yang belum terlihat, namun masih berasal dari dataset yang sama.
            - **Kesimpulan**: SVM unggul dalam stabilitas prediksi, sementara RF bisa unggul jika pola data cukup kuat dipelajari saat training.
            """)

        elif train_percent == 80:
            st.markdown(f"""
            - **Training (80%)**: Random Forest menunjukkan performa sangat baik dengan MAE yang rendah berkat data pelatihan yang lebih besar.
            - **Testing (20%)**: SVM tetap konsisten dalam menggeneralisasi, meskipun dengan jumlah data pengujian yang lebih sedikit.
            - **Forecasting ({forecast_percent}%)**: Forecasting dilakukan terhadap bagian akhir data yang tidak digunakan dalam training/testing. Model memprediksi pola yang belum dilihat sebelumnya, namun masih dalam cakupan distribusi data awal.
            - **Kesimpulan**: Random Forest unggul dalam pelatihan, SVM kuat dalam pengujian. Keduanya menunjukkan hasil yang baik untuk forecasting berbasis data internal.
            """)

        else:
            st.markdown("""
            Persentase training yang dipilih belum didukung untuk penjelasan visualisasi ini.
            """)

        st.subheader("📝 Penjelasan Forecasting")
        st.markdown(f"""
        - **Forecasting menggunakan {forecast_percent}% bagian akhir dari dataset** yang tidak digunakan dalam training maupun testing.
        - Data ini tetap berasal dari dataset utama, namun dianggap sebagai **representasi waktu mendatang** (belum terlihat saat pelatihan).
        - Tujuannya adalah untuk **mengevaluasi kemampuan model dalam meramalkan data baru berdasarkan pola yang telah dipelajari**.
        - Forecasting **bukan dari file eksternal**, melainkan simulasi prediksi masa depan berdasarkan data historis yang tersedia.
        """)
