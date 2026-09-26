import streamlit as st
import joblib


model = joblib.load("models/bec_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


st.title("BEC Email Detector")

st.write(
    "Enter an email message to check whether it may be a "
    "Business Email Compromise attack."
)

email_text = st.text_area(
    "Email content",
    height=250,
    placeholder="Paste email here..."
)


if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Enter an email first.")

    else:
        vector = vectorizer.transform([email_text])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100

        st.subheader("Result")

        if str(prediction).lower() in [
            "bec",
            "phishing",
            "malicious",
            "1"
        ]:
            st.error("Possible BEC attack !!!")
        else:
            st.success("Legitimate email")

        st.write(f"Prediction: **{prediction}**")
        st.write(f"Confidence: **{confidence:.2f}%**")