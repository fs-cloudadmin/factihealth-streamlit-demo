import pickle
import streamlit_authenticator as stauth

# Usernames and their plain text passwords (for hashing)
usernames = ["kr", "sd", "ak", "admin"]
passwords = ["krish21", "sourdas", "abhik", "admin"]

# Hash the passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# Combine usernames and their corresponding hashed passwords into a dictionary
credentials = {username: hashed_passwords[i] for i, username in enumerate(usernames)}


file_path = "hashed_pwd.pkl"

# Serialize and save the dictionary to a file
with open(file_path, 'wb') as file:
    pickle.dump(credentials, file)



#import pickle
#from pathlib import Path
#
#import streamlit_authenticator as stauth
#
#names = ["Krishika R","Sourodeep Das","Abhishek K","admin"]
#usernames = ["kr","sd","ak","admin"]
#passwords = ["krish21","sourdas","abhik","admin"]                                                                                               #["abc123","def456","ghi789"]
#
#hashed_passwords = stauth.Hasher(passwords).generate()
#
#file_path = "hashed_pwd.pkl"  # Updated to the intended file path
#
#with open(file_path, 'wb') as file:  # Using file_path variable
#    pickle.dump(hashed_passwords, file)