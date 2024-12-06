import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# โหลดโมเดลที่ฝึกมาแล้ว
model = tf.keras.models.load_model('math_score_predictor.h5')

# โหลดและเตรียมข้อมูลต้นแบบสำหรับ LabelEncoder และ Scaler
data = pd.read_csv("StudentsPerFormance.csv")
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

scaler = StandardScaler()
X = data.drop(columns=['math score'])
scaler.fit(X)

# ฟังก์ชันตรวจสอบและเพิ่มค่าที่ไม่เคยเห็นมาก่อนใน LabelEncoder
def safe_transform(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

# แอป Streamlit
st.title("ตัวทำนายคะแนนคณิตศาสตร์")

# ฟอร์มกรอกข้อมูลจากผู้ใช้
gender = st.selectbox("เพศ", list(encoders['gender'].classes_))
race = st.selectbox("เชื้อชาติ/กลุ่มชาติพันธุ์", list(encoders['race/ethnicity'].classes_))
parental_education = st.selectbox("ระดับการศึกษาของผู้ปกครอง", list(encoders['parental level of education'].classes_))
lunch = st.selectbox("ประเภทมื้ออาหาร", list(encoders['lunch'].classes_))
test_preparation = st.selectbox("หลักสูตรเตรียมสอบ", list(encoders['test preparation course'].classes_))
reading_score = st.slider("คะแนนการอ่าน", 0, 100, 50)
writing_score = st.slider("คะแนนการเขียน", 0, 100, 50)

# การประมวลผลข้อมูลจากผู้ใช้
user_data = np.array([
    safe_transform(encoders['gender'], gender),
    safe_transform(encoders['race/ethnicity'], race),
    safe_transform(encoders['parental level of education'], parental_education),
    safe_transform(encoders['lunch'], lunch),
    safe_transform(encoders['test preparation course'], test_preparation),
    reading_score,
    writing_score
]).reshape(1, -1)

user_data = scaler.transform(user_data)

# ทำนายคะแนน
prediction = model.predict(user_data)
st.write(f"คะแนนคณิตศาสตร์ที่ทำนายได้: {prediction[0][0]:.2f}")
