import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from deepface import DeepFace
import numpy as np
from PIL import Image

st.title(" Emotion-Aware Smart Mirror")
st.write(" Capture your photo to detect your emotion.")
st.subheader(" Created By Akshay Mudavath ")


img_file = st.camera_input("Take a picture")

if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption='Your captured image', use_column_width=True)

    result = DeepFace.analyze(img_path=np.array(img), actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']

    st.subheader(f"🧠 Detected Emotion: `{emotion.upper()}`")


    tips = {
        "happy": "Keep smiling! You're glowing today 😊",
        "sad": "It's okay to feel down. Take a walk or talk to someone 💙",
        "angry": "Take a deep breath and count to 10 🔥",
        "surprise": "Something unexpected? Stay flexible! ✨",
        "neutral": "Stay calm and balanced today 🤍",
        "fear": "Try to ground yourself. You’re safe 🧘",
        "disgust": "Shift focus. Watch something relaxing 🍃"
    }

    st.success(tips.get(emotion, "You're doing great! 💪"))

    # Save to CSV
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'emotion': emotion
    }
    df = pd.DataFrame([log])
    df.to_csv("emotion_log.csv", mode='a', header=not bool(open("emotion_log.csv").read().strip()), index=False)
st.markdown("---")
if st.button("📊 Show Emotion History"):
    data = pd.read_csv("emotion_log.csv")
    st.dataframe(data.tail(10))
    emotion_counts = data['emotion'].value_counts()
    plt.figure(figsize=(6,4))
    emotion_counts.plot(kind='bar', color='skyblue')
    plt.title("Your Emotion History")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.grid(True)
    st.pyplot(plt)
