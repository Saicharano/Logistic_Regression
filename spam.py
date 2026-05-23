import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import re
import streamlit as st
data = pd.read_csv("emails.csv")
df = pd.DataFrame(data)
x = df.drop(["Prediction","Email No."],axis=1)
y = df["Prediction"]
features = x.columns
vectorizer = CountVectorizer()
# x_vectorized = vectorizer.fit_transform(x);
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.2,
                                                 random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
#print(accuracy)
st.title("Spam Email Detector")
st.write(f"Model Accuracy: {accuracy:.2f}")
email = st.text_area("Enter Your Email: ")
email = re.sub(r'http\S+', '', email)
email = re.sub(r'[^\w\s.,!?@\-]', '', email)
email = " ".join(email.split())
if st.button("Predict"):
    email_words = email.lower().split()
    email_vector = pd.DataFrame([[0]*len(features)], columns=features)
    # for word in email_words:
    #     if word in email_vector.columns:
    #         email_vector.at[0, word] = 1
    prediction= model.predict(email_vector)
    probability = model.predict_proba(email_vector)
    print(probability[0][1])
    if(prediction[0]==1):
        st.error("Spam Email")
    else:
        st.success("Not Spam Email")

