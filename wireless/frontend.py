import streamlit as st
import requests
import time

st.title("🚗 Smart Car Status Monitor")
placeholder = st.empty()

while True:
    try:
        res = requests.get("http://localhost:5000/get")
        if res.ok:
            data = res.json()
            if "message" in data:
                placeholder.markdown(f"### 🛑 {data['message']}")
            else:
                placeholder.markdown("### ✅ Vehicle Running")
        else:
            placeholder.markdown(f"### ❌ Error: Server returned {res.status_code}")
    except Exception as e:
        placeholder.markdown(f"### ❌ Exception: {e}")
    time.sleep(1)
