# ==========================================================
# 🤖 ระบบพยากรณ์ AI (เวอร์ชันใช้งานจริง)
# คุณสมบัติ:
# - สอนโมเดล / พยากรณ์อนาคต
# - ตรวจสอบข้อมูลก่อนพยากรณ์
# - วิเคราะห์ความน่าเชื่อถือและให้คำแนะนำ
# - ส่งออกข้อมูลเป็น Excel และ CSV
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import io


# ==========================================================
# เชื่อมต่อระบบหลังบ้าน (BACKEND)
# ==========================================================

try:
    from src.train.pipeline import run_training
    from src.utils.model_io import save_model, load_model, list_models
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในระบบหลังบ้าน: {e}")
    st.stop()


# ==========================================================
# การตั้งค่าหน้าจอ
# ==========================================================

st.set_page_config(
    layout="wide",
    page_title="ระบบพยากรณ์ AI อัจฉริยะ",
    page_icon="🤖"
)

st.title("🤖 ระบบพยากรณ์ AI (Smart Forecast)")
st.caption("เครื่องมือวิเคราะห์และทำนายข้อมูลด้วยเทคโนโลยี LSTM และ Linear Regression")


# ==========================================================
# ฟังก์ชันคำนวณความสัมพันธ์ของข้อมูล
# ==========================================================

def autocorr(x, lag=1):
    if len(x) <= lag:
        return 0
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


# ==========================================================
# ฟังก์ชันทำความสะอาดข้อมูลด้วยมือ
# ==========================================================

def manual_clean_data(df, col, z_threshold, window_size):
    df_out = df.copy()
    z = np.abs(stats.zscore(df_out[col]))
    outliers = z > z_threshold

    if outliers.sum() > 0:
        median = df_out[col].median()
        df_out.loc[outliers, col] = median

    if window_size > 1:
        df_out[col] = (
            df_out[col]
            .rolling(window_size, center=True)
            .mean()
            .bfill()
            .ffill()
        )
    return df_out, int(outliers.sum())


# ==========================================================
# ฟังก์ชันโหลดข้อมูล
# ==========================================================

# ==========================================================
# ฟังก์ชันโหลดข้อมูล (เวอร์ชันอัปเกรด: ตรวจสอบ NaN และพรีวิว)
# ==========================================================

def load_data():
    file = st.file_uploader(
        "📁 อัปโหลดไฟล์ CSV หรือ Excel",
        type=["csv", "xlsx"]
    )
    if file is None:
        st.stop()

    # โหลดไฟล์ต้นฉบับเพื่อตรวจสอบเบื้องต้น
    df_raw = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    
    # --- ส่วนการพรีวิวและแจ้งเตือนสถิติข้อมูล ---
    st.subheader("📊 ตรวจสอบความสมบูรณ์ของไฟล์")
    
    # 1. แสดง Metrics หลัก
    m1, m2, m3 = st.columns(3)
    total_rows = len(df_raw)
    total_nan = df_raw.isnull().sum().sum()
    
    m1.metric("จำนวนแถวทั้งหมด", f"{total_rows:,}")
    m2.metric("ค่าที่หายไป (NaN) ทั้งไฟล์", f"{total_nan:,}", delta=f"{total_nan}" if total_nan > 0 else None, delta_color="inverse")
    m3.metric("จำนวนคอลัมน์", f"{len(df_raw.columns)}")

    # 2. พรีวิวข้อมูล 10 แถวแรก
    with st.expander("👀 ดูตัวอย่างข้อมูล 10 แถวแรก", expanded=True):
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        # แสดงรายการ NaN รายคอลัมน์ (ถ้ามี)
        nan_info = df_raw.isnull().sum()
        if nan_info.sum() > 0:
            st.warning("⚠️ ตรวจพบค่าว่างในคอลัมน์:")
            # กรองเฉพาะคอลัมน์ที่มี NaN
            st.write(nan_info[nan_info > 0])
        else:
            st.success("✅ ข้อมูลสมบูรณ์ (ไม่พบค่าว่าง)")

    st.divider()

    # 3. เลือกคอลัมน์เป้าหมาย
    target = st.selectbox(" เลือกคอลัมน์ที่ต้องการทำนาย (Target)", df_raw.columns)

    # จัดการข้อมูล: แปลงเป็นตัวเลข และลบ NaN เฉพาะส่วนที่จำเป็น
    series = pd.to_numeric(df_raw[target], errors="coerce")
    nan_in_target = series.isna().sum()
    
    if nan_in_target > 0:
        st.info(f"💡 ระบบพบค่าที่ไม่ใช่ตัวเลขหรือค่าว่างในคอลัมน์ '{target}' จำนวน {nan_in_target} แถว (จะถูกตัดออกเพื่อใช้ในการประมวลผล)")

    series = series.dropna().reset_index(drop=True)
    df_out = pd.DataFrame({
        "value": series,
        "time": np.arange(len(series))
    })

    return df_out, "time", "value"


# ==========================================================
# แถบเมนูข้าง (SIDEBAR)
# ==========================================================

with st.sidebar:
    st.header(" เมนูหลัก")
    mode = st.radio("โหมดการทำงาน", [" สอนโมเดล (Train)", " พยากรณ์ (Forecast)"])
    st.divider()
    st.caption("เวอร์ชัน: Production 2.0")


# ==========================================================
# โหมดสอนโมเดล (TRAIN MODE)
# ==========================================================

if mode == " สอนโมเดล (Train)":
    st.header(" สอนโมเดล (Train Model)")

    df, time_col, target_col = load_data()

    col1, col2 = st.columns(2)
    with col1:
        z = st.slider("ระดับการกำจัดค่าผิดปกติ (Outlier Z-score)", 1.0, 5.0, 3.0, help="ค่าน้อยจะกำจัดค่าที่กระโดดออกจากกลุ่มมาก")
    with col2:
        smooth = st.slider("ความเนียนของเส้น (Smooth window)", 1, 21, 3, help="เพิ่มความสมูทเพื่อลดสัญญาณรบกวน")

    df_clean, out_count = manual_clean_data(df, target_col, z, smooth)
    st.info(f"🔍 ตรวจพบค่าผิดปกติ: {out_count} จุด (ถูกแทนที่ด้วยค่ากลางแล้ว)")

    # กราฟเปรียบเทียบ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[target_col], name="ข้อมูลเดิม", line=dict(color='silver')))
    fig.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean[target_col], name="ข้อมูลที่คลีนแล้ว", line=dict(color='#1f77b4')))
    fig.update_layout(template="plotly_white", title="การเปรียบเทียบข้อมูลก่อนและหลังการเตรียมการ")
    st.plotly_chart(fig, use_container_width=True)

    # วิเคราะห์ความสัมพันธ์
    ac = autocorr(df_clean[target_col].values)
    st.metric("ความสัมพันธ์รายเวลา (Autocorrelation)", f"{ac:.3f}")
    if ac < 0.3:
        st.warning("⚠️ ข้อมูลมีความสัมพันธ์รายเวลาต่ำ โมเดลอาจพยากรณ์ได้ไม่แม่นยำนัก")

    # ตั้งค่าการสอน
    st.subheader("⚙️ ตั้งค่าโมเดล")
    model_type = st.selectbox("เลือกประเภทโมเดล", ["linear", "lstm"], format_func=lambda x: "Linear Regression" if x=="linear" else "LSTM (Deep Learning)")
    lag = st.number_input("จำนวนข้อมูลย้อนหลังที่ใช้ทาย (Lag)", 5, len(df)-1, min(60, len(df)//10))

    if model_type == "lstm":
        c1, c2, c3 = st.columns(3)
        epochs = c1.number_input("รอบการสอน (Epochs)", 10, 500, 100)
        hidden = c2.number_input("ขนาดความจำ (Hidden size)", 16, 512, 128)
        dropout = c3.slider("Dropout", 0.0, 0.5, 0.2)
    else:
        epochs = hidden = dropout = None

    model_name = st.text_input("ชื่อโมเดล", f"model_{datetime.now().strftime('%H%M%S')}")

    if st.button("🚀 เริ่มสอนโมเดล"):
        with st.spinner("🧠 AI กำลังเรียนรู้ข้อมูล..."):
            artifact = run_training(df_clean, target_col, model_type, lag, hidden_size=hidden, dropout=dropout, epochs=epochs)
            save_model(artifact, model_name)

            r2 = r2_score(artifact["test_true"], artifact["test_pred"])
            mse = mean_squared_error(artifact["test_true"], artifact["test_pred"])

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("ความแม่นยำ (R²)", f"{r2:.4f}")
            col_m2.metric("ความคลาดเคลื่อน (MSE)", f"{mse:.4f}")

            # --- ระบบวิเคราะห์และให้คำแนะนำ ---
            st.subheader("💡 การวินิจฉัยและคำแนะนำจาก AI")
            if r2 < 0.3:
                st.error("❌ **ผลลัพธ์: ต่ำมาก**")
                st.markdown("""
                **คำแนะนำเพื่อปรับปรุง:**
                1. **เพิ่มค่า Lag:** ลองเพิ่มจำนวนข้อมูลย้อนหลังเพื่อให้โมเดลเห็นรูปแบบที่กว้างขึ้น
                2. **ตรวจสอบข้อมูล:** ข้อมูลอาจมีความเป็นสุ่ม (Random) มากเกินไป หรือไม่มีรูปแบบที่ชัดเจน
                3. **เปลี่ยนประเภทโมเดล:** หากใช้ Linear ลองเปลี่ยนเป็น LSTM หรือเพิ่มค่า Epochs ใน LSTM
                """)
            elif r2 < 0.6:
                st.warning("⚠️ **ผลลัพธ์: ปานกลาง**")
                st.markdown("""
                **คำแนะนำเพื่อปรับปรุง:**
                1. **ปรับความสมูท:** ลองเพิ่ม/ลดค่า Smooth window ในขั้นตอนเตรียมข้อมูล
                2. **เพิ่มความจำ:** ลองเพิ่มค่า Hidden Size ใน LSTM เพื่อให้โมเดลจำรายละเอียดได้มากขึ้น
                """)
            else:
                st.success("✅ **ผลลัพธ์: ดีมาก**")
                st.write("โมเดลเรียนรู้รูปแบบข้อมูลได้ดีเยี่ยม พร้อมใช้งานพยากรณ์แล้ว!")


# ==========================================================
# โหมดพยากรณ์ (FORECAST MODE)
# ==========================================================

elif mode == " พยากรณ์ (Forecast)":
    st.header(" การพยากรณ์ (Forecast)")

    models = list_models()
    if not models:
        st.warning("โปรดสอนโมเดลก่อนเริ่มต้น")
        st.stop()

    model_sel = st.selectbox("เลือกโมเดลที่จะใช้", models)
    horizon = st.slider("จำนวนก้าวที่ต้องการพยากรณ์ล่วงหน้า", 1, 500, 24)
    smooth_forecast = st.slider("ความเนียนของเส้นพยากรณ์", 1, 20, 1)

    df, time_col, target_col = load_data()
    df_clean, _ = manual_clean_data(df, target_col, 3.0, 3)

    artifact = load_model(model_sel)
    model = artifact["model"]
    lag = artifact["config"]["lag"]
    series = df_clean[target_col].values

    # --- ตรวจสอบข้อมูลก่อนทำนาย ---
    st.subheader("🔍 ตรวจสอบหน้าต่างข้อมูลล่าสุด (Forecast Window)")
    last_window = series[-lag:]
    preview_index = np.arange(len(series)-lag, len(series))
    
    st.metric("ความสัมพันธ์ข้อมูลชุดล่าสุด (Autocorr)", f"{autocorr(series):.3f}")

    fig_pre = go.Figure()
    fig_pre.add_trace(go.Scatter(x=df[time_col].values, y=df[target_col].values, name="ข้อมูลประวัติ", line=dict(color="silver")))
    fig_pre.add_trace(go.Scatter(x=preview_index, y=last_window, name="หน้าต่างข้อมูลที่ใช้ทำนาย", line=dict(color="red", width=3)))
    fig_pre.update_layout(template="plotly_white", title="ข้อมูลล่าสุดที่ AI จะนำไปใช้ประมวลผล")
    st.plotly_chart(fig_pre, use_container_width=True)

    if st.button(" เริ่มพยากรณ์อนาคต"):
        with st.spinner(" AI กำลังคำนวณอนาคต..."):
            # พยากรณ์
            if hasattr(model, "forecast"):
                future = model.forecast(last_window, steps=horizon)
            else:
                history = list(series.copy())
                future = []
                for _ in range(horizon):
                    x = np.array(history[-lag:]).reshape(1, -1)
                    pred = model.predict(x)[0]
                    future.append(pred)
                    history.append(pred)
                future = np.array(future)

            # การทำให้เส้นเนียน
            future_smooth = pd.Series(future).rolling(smooth_forecast).mean().bfill().values

            # แสดงผล Metrics
            test_true = artifact.get("test_true", [])
            test_pred = artifact.get("test_pred", [])
            if len(test_true) > 0:
                r2 = r2_score(test_true, test_pred)
                st.metric("ความน่าเชื่อถือของโมเดลนี้ (R²)", f"{r2:.4f}")

            # กราฟผลลัพธ์
            future_x = np.arange(len(series), len(series)+horizon)
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=df[time_col].values, y=df[target_col].values, name="ประวัติ"))
            fig_res.add_trace(go.Scatter(x=future_x, y=future, name="พยากรณ์ (ดิบ)", line=dict(dash="dot", color="orange")))
            fig_res.add_trace(go.Scatter(x=future_x, y=future_smooth, name="พยากรณ์ (เนียน)", line=dict(color="red", width=3)))
            fig_res.update_layout(template="plotly_white", title="ผลลัพธ์การพยากรณ์")
            st.plotly_chart(fig_res, use_container_width=True)

            # --- ส่วนส่งออกข้อมูล ---
            st.subheader("📥 ดาวน์โหลดผลลัพธ์")
            result_df = pd.DataFrame({
                "ลำดับเวลา": future_x,
                "พยากรณ์_ดิบ": future,
                "พยากรณ์_เนียน": future_smooth
            })
            st.dataframe(result_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 ดาวน์โหลด CSV", result_df.to_csv(index=False).encode("utf-8-sig"), "forecast.csv", "text/csv")
            with c2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Forecast')
                st.download_button("📥 ดาวน์โหลด Excel", output.getvalue(), "forecast.xlsx", "application/vnd.ms-excel")