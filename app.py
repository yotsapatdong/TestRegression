import streamlit as st
import pandas as pd
import pickle
import sklearn


# --- Step 1: โหลดโมเดลที่ฝึกไว้ ---
with open("model-reg-67130701917.pkl", "rb") as f:
    model = pickle.load(f)

# --- ส่วนหัวของเว็บแอป ---
st.title("📈 Sales Prediction App")
st.write("ทำนายยอดขายจากงบโฆษณาใน YouTube, TikTok และ Instagram")

# --- Step 2: รับค่าจากผู้ใช้ ---
youtube = st.number_input("งบโฆษณาใน YouTube", min_value=0.0, value=50.0)
tiktok = st.number_input("งบโฆษณาใน TikTok", min_value=0.0, value=50.0)
instagram = st.number_input("งบโฆษณาใน Instagram", min_value=0.0, value=50.0)

# --- Step 3: เมื่อกดปุ่มให้ทำนาย ---
if st.button("ทำนายยอดขาย"):
    # สร้าง DataFrame ใหม่จากค่าที่ผู้ใช้กรอก
    new_data = pd.DataFrame({
        "youtube": [youtube],
        "tiktok": [tiktok],
        "instagram": [instagram]
    })

    # ทำการทำนาย
    prediction = model.predict(new_data)
    st.success(f"ยอดขายที่คาดการณ์ได้คือ: {prediction[0]:.2f}")
