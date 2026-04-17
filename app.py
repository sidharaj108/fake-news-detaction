import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load model locally from the GitHub repo files
@st.cache_resource(show_spinner="Loading model... Please wait (first time may be slow)")
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.success("✅ Model loaded successfully!")

def preprocess_text(text):
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_fake_news(news_text):
    if not news_text.strip():
        return "❌ Please enter news text!", 0.0
    
    clean_text = preprocess_text(news_text)
    vectorized = vectorizer.transform([clean_text])
    
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    confidence = max(prob) * 100
    
    result = "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE / MISINFORMATION"
    return result, round(confidence, 2)

# UI
st.title("📰 Fake News / Misinformation Detector")
st.markdown("**Trained on ISOT Fake News Dataset (~45,000 articles)**")

news_input = st.text_area("Paste news article, headline or statement here:", height=250)

if st.button("🔍 Predict", type="primary"):
    with st.spinner("Analyzing the news..."):
        result, confidence = predict_fake_news(news_input)
        
        if "REAL" in result:
            st.success(result)
        else:
            st.error(result)
        
        st.metric("Confidence Score", f"{confidence}%")
        st.progress(confidence / 100)

st.caption("Built with Streamlit • TF-IDF + Logistic Regression")
