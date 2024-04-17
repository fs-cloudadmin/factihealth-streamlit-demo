import streamlit as st
from datetime import date, timedelta

st.set_page_config(layout="wide")

# Title for the form
st.title('Date Selection Form')

# Custom CSS to inject into the Streamlit page to style the submit button
st.markdown("""
    <style>
    /* Center text */
    .centered-text {
        text-align: center;
    }
    .stButton>button {
        height: 2em;     /* Increase button size */
        width: 5em;      /* Increase button width */
        font-size: 1em; /* Increase arrow size */
        color: green;    /* Arrow color green */
    }
    </style>""",
    unsafe_allow_html=True
)

# Create a form and three columns within the form
with st.form(key='date_form',border=False):
    col1, col2, col3 = st.columns([1, 1, 1])

    # Default dates
    default_start_date = date(2024, 1, 1)
    default_end_date = date(2024, 1, 15)

    with col1:
        # Use the first column for the date range input
        st.markdown('<p class="centered-text">Select Date Range:</p>', unsafe_allow_html=True)
        # Date range input without a label
        start_date, end_date = st.date_input("", (default_start_date, default_end_date))


    with col2:
        st.markdown('<p class="centered-text">Select N days:</p>', unsafe_allow_html=True)
        # Use the second column for the 'N' days input
        n_days = st.number_input("", min_value=1, value=1)

    with col3:
        # Add an empty line above the button
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # Use the third column for the form submit button with a green arrow
        submit_button = st.form_submit_button(label='OK')

# Logic for processing the form input outside the form
if submit_button:
    st.write("Selected Date Range: ", start_date, "to", end_date)
    st.write("Selected 'N' days: ", n_days)

    # Example logic to check if N days is within the selected date range
    if start_date and end_date:
        if start_date + timedelta(days=n_days-1) <= end_date:
            st.success(f"The selected 'N' days are within the range.")
        else:
            st.error(f"The selected 'N' days exceed the selected date range.")
