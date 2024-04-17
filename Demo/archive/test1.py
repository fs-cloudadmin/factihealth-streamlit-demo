import streamlit as st
import pickle
import streamlit_authenticator as stauth

# Load the hashed passwords from the .pkl file
file_path = "C:/Users/krishika.R/gitcodescpace/factihealth-streamlit-demo/Demo/hashed_pwd.pkl"
with open(file_path, 'rb') as file:
    loaded_credentials = pickle.load(file)

# Usernames
usernames = list(loaded_credentials.keys())

# Initialize the authenticator
authenticator = stauth.Authenticate(names=usernames,usernames=usernames,passwords=loaded_credentials)

# Display the login form and handle the authentication
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.write(f"Welcome {name}!")
    # Place the main content of your app here
elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
