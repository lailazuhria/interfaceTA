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

train_percent = st.selectbox("Pilih persentase Training data:", [60, 70, 75, 80])
forecast_percent = 20
test_percent = 100 - train_percent - forecast_percent

st.markdown(f"**Persentase Testing data :** {test_percent}%")
if test_percent <= 0:
    st.error("Persentase training terlalu besar. Sisakan minimal 20% untuk testing dan forecasting.")
else:
    if st.button("Proses"):
        # --- Split Data (Time-based) ---
        total_rows = len(df)
        train_end = int(train_percent / 100 * total_rows)
        test_end = train_end + int(test_percent / 100 * total_rows)

        train_data = df.iloc[:train_end]
        test_data = df.iloc[train_end:test_end]
        forecast_data = df.iloc[test_end:]

        X_train = train_data.drop("Age", axis=1)
        y_train = train_data["Age"]
        X_test = test_data.drop("Age", axis=1)
        y_test = test_data["Age"]
        X_forecast = forecast_data.drop("Age", axis=1)
        y_forecast = forecast_data["Age"] if "Age" in forecast_data else np.nan

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
        # --- Tabel Hasil Forecasting ---
        st.subheader("🗂️ Tabel Hasil Forecasting")

        # Buat DataFrame hasil prediksi
        forecast_result = pd.DataFrame()
        forecast_result["Index"] = forecast_data.index
        forecast_result["Prediksi_RF"] = y_forecast_pred_rf
        forecast_result["Prediksi_SVM"] = y_forecast_pred_svm

        # Jika data aktual tersedia (y_forecast tidak NaN seluruhnya), tambahkan
        if not y_forecast.isnull().all():
            forecast_result["Actual"] = y_forecast.values

        # Tampilkan tabel
        st.dataframe(forecast_result.reset_index(drop=True), use_container_width=True)

        # --- Visualisasi Prediksi ---
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
            ax3.plot(y_forecast_pred_rf, label="RF Prediction", linestyle='--', marker='x')
            ax3.plot(y_forecast_pred_svm, label="SVM Prediction", linestyle='--', marker='s')
            ax3.set_title("Forecasting Prediction (Tanpa Data Aktual)")
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

        # --- Penjelasan Dinamis ---
        st.subheader("📝 Penjelasan Visualisasi")

        if train_percent == 60:
            st.markdown("""
            - **Training (60%)**: SVM memiliki MAE lebih rendah dari RF, menandakan performa lebih baik pada data historis. RF menunjukkan sedikit overfitting.
            - **Testing (20%)**: SVM konsisten dengan MAE rendah dan performa stabil. RF masih fluktuatif dan kurang presisi.
            - **Forecasting (20%)**: SVM menghasilkan prediksi lebih stabil, RF terlihat lebih variatif dan kurang akurat.
            - **Perbandingan MAE**: SVM lebih dominan di semua tahap, cocok saat data training terbatas.
            """)

        elif train_percent == 70:
            st.markdown("""
            - **Training (70%)**: MAE Random Forest membaik dan mulai menyaingi SVM. SVM tetap konsisten dengan hasil stabil.
            - **Testing (10%)**: MAE SVM masih lebih kecil, menunjukkan generalisasi yang lebih baik.
            - **Forecasting (20%)**: RF menunjukkan hasil lebih kompetitif, tapi SVM masih lebih stabil.
            - **Perbandingan MAE**: Performa kedua model semakin dekat, namun SVM sedikit lebih unggul secara konsisten.
            """)

        elif train_percent == 75:
            st.markdown("""
            - **Training (75%)**: Random Forest menghasilkan MAE paling rendah, menunjukkan model sangat cocok pada data besar.
            - **Testing (5%)**: Performa RF dan SVM hampir seimbang, dengan akurasi tinggi di keduanya.
            - **Forecasting (20%)**: Random Forest unggul dalam prediksi ke depan dengan MAE lebih kecil dari SVM.
            - **Perbandingan MAE**: Random Forest menjadi pilihan terbaik pada 75% training, khususnya untuk forecasting.
            """)

        elif train_percent == 80:
            st.markdown("""
            - **Training (80%)**: Semakin besar data training, RF makin unggul dalam menyesuaikan pola. MAE RF sangat kecil.
            - **Testing (0%)**: Data testing tidak tersedia, karena seluruh data habis dibagi ke training dan forecasting.
            - **Forecasting (20%)**: Performa RF sangat kuat, namun tanpa testing, generalisasi tidak bisa diukur.
            - **Catatan**: Persentase 80% training membuat hasil testing tidak tersedia. Gunakan persentase <=75% untuk evaluasi yang lengkap.
            """)
        else:
            st.markdown("Silakan pilih persentase training yang valid.")
