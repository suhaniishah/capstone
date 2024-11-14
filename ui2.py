import streamlit as st
import bcrypt
import json
import os
import re
import pandas as pd
import time
import torch
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests

# Define paths for user data
USER_DATA_FILE = "user_data.json"
USERS_FOLDER = "users"

# Initialize session state
if "username" not in st.session_state:
    st.session_state["username"] = None
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Set up the Google AI API key in the environment
os.environ["GOOGLE_API_KEY"] = "paste_your_api_key_here"

# Initialize user data JSON if it doesn't exist
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump({}, f)

# Load and save user data functions
def load_user_data():
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_user_data(user_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(user_data, f)

# Helper functions for validation
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def is_valid_phone(phone):
    return len(phone) == 10 and phone.isdigit()

# Sign-up function
def sign_up(username, password, email, phone):
    user_data = load_user_data()
    if username in user_data:
        st.error("Username already exists. Please choose a different username.")
        return False
    if not is_valid_email(email):
        st.error("Invalid email format. Please enter a valid email.")
        return False
    if not is_valid_phone(phone):
        st.error("Invalid phone number. It should be 10 digits.")
        return False
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_data[username] = {
        "password": hashed_password,
        "email": email,
        "phone": phone
    }
    user_folder = os.path.join(USERS_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    user_data[username]["log_file"] = os.path.join(user_folder, "logs.csv")
    save_user_data(user_data)
    # Success message that should appear after successful registration
    print("test")
    st.success("User registered successfully! You can now log in.")
    print("test")
    return True

# Authentication wrapper
def authenticate():
    st.sidebar.title("User Authentication")
    auth_mode = st.sidebar.selectbox("Select Mode", ["Sign In", "Sign Up"])
    
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    email = st.sidebar.text_input("Email") if auth_mode == "Sign Up" else None
    phone = st.sidebar.text_input("Phone") if auth_mode == "Sign Up" else None
    
    if auth_mode == "Sign Up":
        if st.sidebar.button("Register"):
            if username and password and email and phone:
                if sign_up(username, password, email, phone):
                    st.sidebar.success("Registration successful! You can now log in.")
            else:
                st.sidebar.error("All fields are required for registration.")
                
    elif auth_mode == "Sign In":
        if st.sidebar.button("Login", on_click=sign_in, args=(username, password)):
            pass


# Login functions
def login_callback(username):
    st.session_state["username"] = username
    st.session_state["logged_in"] = True

def sign_in(username, password):
    user_data = load_user_data()
    if username not in user_data:
        st.error("Username does not exist. Please sign up first.")
        return False
    hashed_password = user_data[username]["password"]
    if bcrypt.checkpw(password.encode(), hashed_password.encode()):
        login_callback(username)
        print("test")
        st.success("Login successful!")
        return True
    else:
        st.error("Incorrect password.")
        return False

# Authentication wrapper
def authenticate():
    st.sidebar.title("User Authentication")
    auth_mode = st.sidebar.selectbox("Select Mode", ["Sign In", "Sign Up"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    email = st.sidebar.text_input("Email") if auth_mode == "Sign Up" else None
    phone = st.sidebar.text_input("Phone") if auth_mode == "Sign Up" else None
    if auth_mode == "Sign Up":
        if st.sidebar.button("Register"):
            if username and password and email and phone:
                sign_up(username, password, email, phone)
            else:
                st.error("All fields are required for registration.")
    elif auth_mode == "Sign In":
        if st.sidebar.button("Login", on_click=sign_in, args=(username, password)):
            pass

# Logout function
def logout():
    st.session_state["username"] = None
    st.session_state["logged_in"] = False
    st.experimental_set_query_params(rerun="true")  # Force a page refresh

# Dashboard to display quick stats and insights for all users
def show_dashboard():
    st.title("TranscribeX")
    st.markdown("A Multilingual Sentiment and Text Transformation NLP Application")
    
    if os.path.exists(USERS_FOLDER):
        user_files = []
        for user_dir in os.listdir(USERS_FOLDER):
            log_path = os.path.join(USERS_FOLDER, user_dir, "logs.csv")
            if os.path.exists(log_path):
                user_files.append(pd.read_csv(log_path, names=["Task Type", "Input Text", "Output"]))
        
        if user_files:
            log_df = pd.concat(user_files, ignore_index=True)
            st.subheader("Overview of Recent Activity for All Users")
            
            # Display individual task counts
            task_counts = log_df["Task Type"].value_counts()
            st.write(f"**Sentiment Analysis Tasks:** {task_counts.get('Sentiment Analysis', 0)}")
            st.write(f"**Rephrasing Tasks:** {task_counts.get('Rephrasing Text', 0)}")
            st.write(f"**Translation Tasks:** {task_counts.get('Translation', 0)}")
            
            # Sentiment Analysis Distribution
            sentiment_logs = log_df[log_df["Task Type"] == "Sentiment Analysis"]
            if not sentiment_logs.empty:
                st.subheader("Sentiment Distribution")
                sentiment_counts = sentiment_logs["Output"].value_counts()
                
                # Ensure all sentiment types are included, even with zero counts
                all_sentiments = ['Positive', 'Neutral', 'Negative']
                sentiment_counts = sentiment_counts.reindex(all_sentiments, fill_value=0)

                sentiment_chart = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                                         labels={'x': 'Sentiment', 'y': 'Count'},
                                         title="Sentiment Analysis Results")
                st.plotly_chart(sentiment_chart)

            # Word Cloud Generation
            if not log_df.empty:
                st.subheader("Word Cloud from Recent Inputs")
                all_text = " ".join(log_df["Input Text"].dropna().tolist())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

                # Display Word Cloud with Matplotlib
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
        else:
            st.write("No logs found.")
    st.subheader("AI Fun Fact")
    st.write("Did you know? AI can analyze and understand human emotions through sentiment analysis!")

# Log tasks for specific user
def log_task(task_type, input_text, output):
    if st.session_state["username"]:
        log_file = os.path.join(USERS_FOLDER, st.session_state["username"], "logs.csv")
        log_data = pd.DataFrame([[task_type, input_text, output]], columns=["Task Type", "Input Text", "Output"])
        if os.path.exists(log_file):
            log_data.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_data.to_csv(log_file, mode='w', header=True, index=False)
    else:
        st.error("No user is currently logged in.")

# Display recent tasks for the logged-in user
def display_recent_tasks(task_type):
    if st.session_state["username"]:
        log_file = os.path.join(USERS_FOLDER, st.session_state["username"], "logs.csv")
        try:
            log_df = pd.read_csv(log_file, names=["Task Type", "Input Text", "Output"])
            task_logs = log_df[log_df["Task Type"] == task_type]
            recent_tasks = task_logs.tail(5)

            if not recent_tasks.empty:
                st.write(f"### Recently Completed {task_type} Tasks")
                for index, row in recent_tasks.iterrows():
                    st.markdown(f"**Text**: {row['Input Text']}")
                    st.markdown(f"**Output**: {row['Output']}")
                    st.write("---")
            else:
                st.write(f"No recent {task_type.lower()} tasks found.")
        except FileNotFoundError:
            st.write("Your recently analyzed comments will appear here.")
    else:
        st.warning("No user is logged in. Please sign in to view recent tasks.")

# AI functions
def analyze_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    confidence_score = probabilities[0][predicted_class_id].item()
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_labels[predicted_class_id], confidence_score

def translate_text(input_language, output_language, text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
         ("human", "{input}")]
    )
    chain = prompt | llm
    ai_msg = chain.invoke({"input_language": input_language, "output_language": output_language, "input": text})
    return ai_msg.content

def rephrase_text(input_text, expected_sentiment):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Rephrase text to match {expected_sentiment}."),
         ("human", "{input_text}")]
    )
    chain = prompt | llm
    ai_msg = chain.invoke({"expected_sentiment": expected_sentiment, "input_text": input_text})
    return ai_msg.content

# Main logic
if not st.session_state["logged_in"]:
    show_dashboard()
    authenticate()
else:
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout()

    # Feature tabs
    st.sidebar.title("Choose an option")
    tabs = ["Sentiment Analysis", "Text Rephrasing", "Text Translation"]
    tab = st.sidebar.radio("Options", tabs)

    if tab == "Sentiment Analysis":
        st.title("Sentiment Analysis")
        text_input = st.text_area("Enter your text here:")
        if st.button("Analyze Sentiment"):
            model_directory = "/Users/suhanishah/Desktop/try2/Final Training"
            model, tokenizer = BertForSequenceClassification.from_pretrained(model_directory), BertTokenizer.from_pretrained("bert-base-uncased")
            with st.spinner('Analyzing sentiment... Please wait.'):
                sentiment, confidence = analyze_sentiment(text_input, model, tokenizer)
            st.success('Analysis complete!')
            st.markdown(f"Sentiment: {sentiment}")
            st.markdown(f"Confidence: {confidence:.2f}")
            log_task("Sentiment Analysis", text_input, sentiment)
        display_recent_tasks("Sentiment Analysis")

    elif tab == "Text Rephrasing":
        st.title("Text Rephrasing")
        last_text = ""
        log_file = os.path.join(USERS_FOLDER, st.session_state["username"], "logs.csv")
        try:
            log_df = pd.read_csv(log_file, names=["Task Type", "Input Text", "Output"])
            if "Sentiment Analysis" in log_df["Task Type"].values:
                last_text = log_df[log_df["Task Type"] == "Sentiment Analysis"]["Input Text"].iloc[-1]
        except FileNotFoundError:
            pass
        text_input = st.text_area("Enter your text here:", value=last_text)
        expected_sentiment = st.selectbox("Expected Sentiment", ["Positive", "Neutral", "Negative"])
        if st.button("Rephrase Text"):
            with st.spinner(f'Rephrasing to {expected_sentiment}...'):
                rephrased_text = rephrase_text(text_input, expected_sentiment)
            st.success('Rephrasing complete!')
            st.markdown(f"Rephrased Text: {rephrased_text}")
            log_task("Rephrasing Text", text_input, rephrased_text)
        display_recent_tasks("Rephrasing Text")

    elif tab == "Text Translation":
        st.title("Text Translation")
        input_language = st.text_input("Input Language", value="English")
        output_language = st.text_input("Output Language", value="German")
        text_to_translate = st.text_area("Enter text to translate:")
        if st.button("Translate"):
            with st.spinner('Translating... Please wait.'):
                translated_text = translate_text(input_language, output_language, text_to_translate)
            st.success('Translation complete!')
            st.markdown(f"Translated Text: {translated_text}")
            log_task("Translation", text_to_translate, translated_text)
        display_recent_tasks("Translation")

