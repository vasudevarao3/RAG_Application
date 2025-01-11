import streamlit as st
import os
import json     # login = {"email": "password"}, history = {"email": [[session1], [session2],...]}
from rag_pinecone import dash_page

# File to store user credentials
CREDENTIALS_FILE = "credentials.json"
HISTORY_FILE = "history.json"


# Save a new user to the credentials file
def save_user(email, password):
    with open(CREDENTIALS_FILE, "r") as f:
        credentials = json.load(f)
    
    if email in credentials:
        return False, "Email already exists."

    credentials[email] = password
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f)

    return True, "Successfully signed up!"

def save_query(email, query):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        
    print(type(history), history[email])
    latest_session = history[email][-1]
    latest_session.append(query)
    history[email][-1] = latest_session

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

    return True, "Successfully history Updated"


# Authenticate user login
def authenticate_user(email, password):
    with open(CREDENTIALS_FILE, "r") as f:
        credentials = json.load(f)

    if email in credentials and credentials[email] == password:
        return True, "Login successful!"
    return False, "Invalid email or password."


#Signup page
def signup_page():
    st.title("Sign Up")
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    entered_email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    
    if st.button("Sign Up"):
        if len(entered_email.split('@')) == 2:
            email = entered_email
            if email and password:
                success, message = save_user(email, password)
                print("email",email, success, message)
                print(history)
                if email not in history:
                    history[email] = []
                    with open(HISTORY_FILE, "w") as f:
                        json.dump(history, f)
                    print("successfully dumped")

                if success:
                    st.success(message) 
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields.")
        else:
            st.error("Please Enter Correct Mail")


#Login Page
def login_page():
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email and password:
            success, message = authenticate_user(email, password)
            if success:
                st.session_state["authenticated"] = True
                st.session_state["user"] = email
            else:
                st.error(message)
            return email, password
        else:
            st.error("Please fill in all fields.")


def dashboard_page():
    # st.title("Dashboard")
    # with open(HISTORY_FILE, "r") as f:      #To add next session
    #     history = json.load(f)

    email = st.session_state['user']
    # st.write(f"Welcome, {email}!")
    # st.success("You have successfully logged in.")

    # user_input = st.text_input("Enter something:")
    # submit = st.button("submit")
    # if submit:
    #     st.write(f"You entered: {user_input}")
    #     save_query(email, user_input)

    # if st.button("Logout"):
    #     st.session_state["authenticated"] = False
    #     st.session_state["user"] = None
    #     history[email].append([])
    #     with open(HISTORY_FILE, "w") as f:
    #         json.dump(history, f)
    dash_page(email)