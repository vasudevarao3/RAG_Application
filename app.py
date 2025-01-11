
import streamlit as st
from login_signup import login_page, signup_page, dashboard_page
import json, os


# File to store user credentials
CREDENTIALS_FILE = "credentials.json"
HISTORY_FILE = "history.json"

# Ensure the credentials file exists
def init_credentials_file():
    if not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump({}, f)

#Ensuring the history file exists
def init_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump({}, f)


def main():
    st.sidebar.title("Navigation")
    init_credentials_file()
    init_history_file()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user"] = None

    if st.session_state["authenticated"]:
        dashboard_page()
    else:
        page = st.sidebar.radio("Go to", ["Login", "Sign Up"])

        if page == "Login":
            login_page()
        elif page == "Sign Up":
            signup_page()

if __name__ == "__main__":

    try:
        main()
        
    except Exception as e:
        print(e)



