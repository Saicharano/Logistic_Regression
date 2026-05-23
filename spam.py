import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv",encoding="latin-1")
    data = data.drop(
        labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
        axis=1
    )
    data.columns = ["category","text"]
    data["category"] = data["category"].map({
        "ham":0,
        "spam":1
    })
    
    x = vectorizer.fit_transform(data["text"])
    y= data["category"]
    # x_vectorized = vectorizer.fit_transform(x);
    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
    model = LogisticRegression(max_iter=1000,
                               class_weight="balanced")
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    #st.write(df["Prediction"].value_counts())
    return model,accuracy,vectorizer
model,accuracy,vectorizer= train_model()
#print(accuracy)
st.title("Spam Email Detector")
st.write(f"Model Accuracy: {accuracy:.2f}")
email = st.text_area("Enter Your Email: ")
email = re.sub(r'http\S+', '', email)
email = re.sub(r'[^\w\s.,!?@\-]', '', email)
email = " ".join(email.split())
if st.button("Predict"):
    email_vector = vectorizer.transform([email])
    # matched_words = []

    # for word in email_words:
    #     if word in email_vector.columns:
    #         email_vector.at[0, word] = 1
    #         matched_words.append(word)

    # st.write("Matched Words:", matched_words)
    # st.write("Total Matched:", len(matched_words))
    prediction= model.predict(email_vector)
    probability = model.predict_proba(email_vector)[0][1]
    st.write(probability)
    if(probability>0.35):
        st.error("Spam Email")
    else:
        st.success("Not Spam Email")

