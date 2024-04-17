import pandas as pd
import numpy as np
import streamlit as st
#import streamlit_authenticator as stauth
from PIL import Image
import plotly.graph_objects as go
import json
import pickle
import datetime as dt
from datetime import date,datetime, timedelta
from langchain_community.chat_models import ChatOpenAI
from helpers import data_connections as connect
from helpers import pre_processing as pp
import boto3
from io import BytesIO
from io import StringIO
import zipfile
import io
import boto3

# Set wide layout
st.set_page_config(layout="wide")

# Authentication setup
#names = ["Krishika R", "Sourodeep Das", "Abhishek K","admin"]
#usernames = ["kr", "sd", "ak","admin"]
#
#with open("hashed_pwd.pkl", 'rb') as file:
#    hashed_passwords = pickle.load(file)
#
#authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "Factihealth_login","")
bucket_name = 'factihealth'

# Provide AWS credentials as strings without trailing commas
aws_access_key_id = 'AKIATTAFLHZHBIMCNXZX'
aws_secret_access_key = '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'
aws_region = 'ap-south-1'
bucket_name = 'factihealth'

@st.cache(allow_output_mutation=True)
def load_model_from_zip_in_s3(bucket_name, object_key, model_file_name, aws_access_key_id, aws_secret_access_key, aws_region):
    # Initialize S3 client with credentials
    s3 = boto3.client('s3', 
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region)
    
    # Fetch the .zip object from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    zip_content = response['Body'].read()
    
    # Read the .zip content into a BytesIO buffer
    zip_buffer = io.BytesIO(zip_content)
    
    # Use zipfile to extract the model file
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        # Extract only the model file you need
        with zip_ref.open(model_file_name, 'r') as model_file:
            model = pickle.load(model_file)
    return model


#if 'admission_model' not in st.session_state or st.session_state['admission_model'] is None:
#    with st.spinner('Loading admission model... This may take a few minutes.'):
#        start_time = datetime.now()  # Start timing
#        st.write('admission_model:', start_time)
#        object_key = 'models/demo/admission_model.zip'
#        model_file_name = 'admission_model.pkl'  # The name of the .pkl file inside the .zip
#        st.session_state['admission_model'] = load_model_from_zip_in_s3(bucket_name, object_key, model_file_name, aws_access_key_id, aws_secret_access_key, aws_region)
#        end_time = datetime.now()  # End timing
#        load_duration = end_time - start_time
#    st.write(f"Model load time: {load_duration}")
#
#if 'discharge_model' not in st.session_state:
#    start_time = datetime.now()  # Start timing
#    st.write('discharge_model:', start_time)
#    object_key = 'models/demo/discharge_model.zip'
#    model_file_name = 'discharge_model.pkl'  # The name of the .pkl file inside the .zip
#    st.session_state['discharge_model'] = load_model_from_zip_in_s3(bucket_name, object_key, model_file_name, aws_access_key_id, aws_secret_access_key, aws_region)
#    end_time = datetime.now()  # End timing
#    load_duration = end_time - start_time
#    st.write(f"Model load time: {load_duration}")
#
#if 'transfer_model' not in st.session_state:
#    start_time = datetime.now()  # Start timing
#    st.write('transfer_model:', start_time)
#    object_key = 'models/demo/transfer_model.zip'
#    model_file_name = 'transfer_model.pkl'  # The name of the .pkl file inside the .zip
#    st.session_state['transfer_model'] = load_model_from_zip_in_s3(bucket_name, object_key, model_file_name, aws_access_key_id, aws_secret_access_key, aws_region)
#    end_time = datetime.now()  # End timing
#    load_duration = end_time - start_time
#    st.write(f"Model load time: {load_duration}")
#

# Main guard
if __name__ == "__main__":
    #name, authenticator_status, username = authenticator.login("Factihealth Login", "main")

#    if authenticator_status: # User has successfully authenticated
#        
    image = Image.open('background/Factihealth.jpg')
    aspect_ratio = image.height / image.width         # Calculate the new height to maintain the aspect ratio
    new_width = 500
    new_height = int(new_width * aspect_ratio)
    image = image.resize((new_width, new_height)) # Resize the image
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(image, width=new_width)
        box_style = f"background-color: #FFA500; padding: 2px; border-radius: 5px;"
#        st.write(f"&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Logged in as: {name}&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Username: {username}")
    st.header("", divider='rainbow')
    
    with col3:
        # Center the "Logout" button horizontally using the column layout
        st.write('<div style="text-align: center;">', unsafe_allow_html=True)
#        authenticator.logout("Logout", "main")
        st.write('</div>', unsafe_allow_html=True)
    st.empty()

        # Initialize session state for tracking the last clicked button
    if 'last_clicked' not in st.session_state:
        st.session_state['last_clicked'] = None

    # Function to update the last clicked button
    def update_last_clicked(option):
        st.session_state['last_clicked'] = option

    # Section for buttons - only appears after login
    cols = ['Admissions', 'Emergency Dept.', 'Surgical Procedure', 'Maternity Care', 'Specialized Clinics', 'Outpatient', 'Dialysis Treatment', 'Rehab Service', 'Home Health Care', 'Dialysis Treatment']
    layout_cols = st.columns(len(cols))

    # Loop through each column to add a button
    for i, col_label in enumerate(cols):
        with layout_cols[i]:
            button_text = str(col_label)  # Convert label to string

            if st.button(button_text, key=f'button_{i}'):
                update_last_clicked(button_text)

    st.header("", divider='rainbow')
    st.empty()
    
    with st.container():
        # Check if a new button was clicked and clear previous content
        if st.session_state['last_clicked']:
            # Clear previous content by using st.empty() or similar methods if necessary

            if st.session_state['last_clicked'] == "Admissions":
                st.header("Admissions")
                # Title for the form
                # Create a form and three columns within the form
                default_start_date = date(2024, 3, 1)
                default_end_date = date(2024, 3, 4)

                # Initialize session state for start_date and end_date if not already set
                if 'start_date' not in st.session_state:
                    st.session_state['start_date'] = default_start_date
                if 'end_date' not in st.session_state:
                    st.session_state['end_date'] = default_end_date
                with st.form(key='date_form',border=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col2:
                        # Use the first column for the date range input
                        start_date, end_date = st.date_input(
                            "Select Date Range:", 
                            (st.session_state['start_date'], st.session_state['end_date'])
                        )

                        # Check if selected start date is before the default start date
                        if start_date < default_start_date:
                            st.error(f"Please select a start date from {default_start_date} onwards.")
                            # Reset to default dates and update session state
                            st.session_state['start_date'] = default_start_date
                            st.session_state['end_date'] = default_end_date
                            start_date, end_date = default_start_date, default_end_date
                    
                    with col3:
                        # Use the second column for the 'N' days input
                        n_days = st.number_input("Choose the historical data duration, in days", min_value=0, value=1)

                    with col4:
                        # Add an empty line above the button
                        st.write("")
                        st.write("")
                        # Use the third column for the form submit button with a green arrow
                        submit_button = st.form_submit_button(label='RUN')
                    old_end_date = default_end_date
                    if end_date > default_start_date:
                        old_start_date = old_end_date - timedelta(days=n_days)
                    else:
                        old_start_date = default_start_date
                    
#                    old_start_date = old_end_date - timedelta(days=n_days)
                    st.write(old_start_date,old_end_date)
                col1,col2,col3 = st.columns(3)
                # Logic for processing the form input outside the form
                with col2:
                    if submit_button:
                        st.markdown(f"##### Below are the forecast results FROM: {start_date} TO: {end_date}")

                bar_color = 'green'
                line_color = 'blue'
                red_line_color = 'red'
                col1, col2, col3 = st.columns(3)
#                
                with col1:
                    # S3 Bucket details
                    bucket_name = 'factihealth'
                    object_key = 'data/demo/admissions_forecast_data.csv'

                    # Initialize an S3 client
                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

                    # Get the object from the bucket
                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)

                    # Read the CSV data into a pandas DataFrame
                    admissions_df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    admissions_df['Date'] = pd.to_datetime(admissions_df['Date'])
                    
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    admissions_filtered_df = admissions_df[admissions_df['Date'].isin(date_range)]
                    admissions_forecast_data = pd.DataFrame({
                        'Date': admissions_filtered_df['Date'].dt.date,
                        'Forecasted Admissions Count': admissions_filtered_df['Forecasted Admissions Count'].astype(int)            
                                    })
                    total_admissions = admissions_forecast_data['Forecasted Admissions Count'].sum()
                    st.write(f"Total Admissions Forecast: {total_admissions}")

                    old_date_range = connect.data_connection_admissions(old_start_date,old_end_date)
                    old_date_range['admittime'] = old_date_range['admittime']#.dt.date # Create a new column for just the date part
                    old_admissions_count = old_date_range.groupby(old_date_range['admittime'])['subject_id'].count().reset_index()                        
                    old_admissions_count.columns = ['Date', 'Admissions_Patient_Count'] # Rename the columns for clarity

                    with st.expander("Admissions Forecast Details"):
                                                # Your chart creation code here
                                                fig1 = go.Figure()
                                                fig1.add_trace(go.Bar(x=old_admissions_count['Date'], y=old_admissions_count['Admissions_Patient_Count'],
                                                                        name='Admissions Raw Data', marker_color=bar_color))
                                                fig1.add_trace(go.Scatter(x=old_admissions_count['Date'], y=old_admissions_count['Admissions_Patient_Count'],
                                                                            mode='lines+markers', name='Admissions Forecast', line=dict(color=line_color)))
                                                fig1.add_trace(go.Scatter(x=admissions_forecast_data['Date'], y=admissions_forecast_data['Forecasted Admissions Count'],
                                                                            mode='lines+markers', name='Admission Forecast', line=dict(color=red_line_color)))
                                                fig1.update_layout(title_text='Admissions Data and Forecast',
                                                                    xaxis_title='Date',
                                                                    yaxis_title='Admissions Patient Count',
                                                                    height=300,
                                                                    width=450,
                                                                    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                                                    xaxis=dict(type='category', tickangle=-45),
                                                                    title=dict(x=0.3, y=0.9)
                                                                    )
                                                # Display the chart inside the expander
                                                st.plotly_chart(fig1)
                                                
                with col2:
                    # S3 Bucket details
                    bucket_name = 'factihealth'
                    object_key = 'data/demo/discharge_forecast_data.csv'
                    # Initialize an S3 client
                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                    # Get the object from the bucket
                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                    # Read the CSV data into a pandas DataFrame
                    discharge_df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    discharge_df['Date'] = pd.to_datetime(discharge_df['Date'])
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    discharge_filtered_df = discharge_df[discharge_df['Date'].isin(date_range)]
                    discharge_forecast_data = pd.DataFrame({
                        'Date': discharge_filtered_df['Date'].dt.date,
                        'Forecasted Discharge Count': discharge_filtered_df['Forecasted Discharge Count'].astype(int)            
                                    })
                    total_discharges = discharge_forecast_data['Forecasted Discharge Count'].sum()
                    st.write(f"Total Discharge Forecast: {total_discharges}")
                   
                   
                    old_date_range['dischtime'] = pd.to_datetime(old_date_range['dischtime'])
                    old_date_range['dischtime'] = old_date_range['dischtime'].dt.date
                    old_discharge_count = old_date_range.groupby(old_date_range['dischtime'])['subject_id'].count().reset_index()

                    # Rename the columns for clarity
                    old_admissions_count.columns = ['Date', 'Discharge_Patient_Count']

                    with st.expander("Discharge Forecast Details"):
                        # Your chart creation code here
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(x=old_admissions_count['Date'], y=old_admissions_count['Discharge_Patient_Count'],
                                                name='Discharge Raw Data', marker_color=bar_color))
                        fig2.add_trace(go.Scatter(x=old_admissions_count['Date'], y=old_admissions_count['Discharge_Patient_Count'],
                                                    mode='lines+markers', name='Discharge Forecast', line=dict(color=line_color)))
                        fig2.add_trace(go.Scatter(x=discharge_forecast_data['Date'], y=discharge_forecast_data['Forecasted Discharge Count'],
                                                    mode='lines+markers', name='Discharge Forecast', line=dict(color=red_line_color)))
                        fig2.update_layout(title_text='Discharge Data and Forecast',
                                            xaxis_title='Date',
                                            yaxis_title='Discharge Patient Count',
                                            height=300,
                                            width=450,
                                            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                            xaxis=dict(type='category', tickangle=-45),
                                            title=dict(x=0.3, y=0.9)
                                            )
                        # Display the chart inside the expander
                        st.plotly_chart(fig2)

                with col3:
                    # S3 Bucket details
                    bucket_name = 'factihealth'
                    object_key = 'data/demo/transfer_forecast_data.csv'
                    # Initialize an S3 client
                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                    # Get the object from the bucket
                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                    # Read the CSV data into a pandas DataFrame
                    transfer_df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    transfer_df['Date'] = pd.to_datetime(transfer_df['Date'])
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

                    transfer_filtered_df = transfer_df[transfer_df['Date'].isin(date_range)]
                    transfer_forecast_data = pd.DataFrame({
                        'Date': transfer_filtered_df['Date'].dt.date,
                        'Forecasted Transfer Count': transfer_filtered_df['Forecasted Transfer Count'].astype(int)            
                                    })
                    total_transfers = transfer_forecast_data['Forecasted Transfer Count'].sum()
                    st.write(f"Total Transfer Forecast: {total_transfers}")

                    old_date_range['transtime'] = pd.to_datetime(old_date_range['transtime'])
                    old_date_range['transtime'] = old_date_range['transtime'].dt.date
                    
                    with st.expander("Transfer Forecast Details"):
                        old_transfer_count = old_date_range.groupby(old_date_range['transtime'])['subject_id'].count().reset_index()
                    old_transfer_count.columns = ['Date', 'Transfer_Patient_Count']   # Rename the columns for clarity

                    with st.expander("Transfer Forecast Details"):
                        # Your chart creation code here
                        fig3 = go.Figure()
                        fig3.add_trace(go.Bar(x=old_transfer_count['Date'], y=old_transfer_count['Transfer_Patient_Count'],
                                                name='Transfer Raw Data', marker_color=bar_color))
                        fig3.add_trace(go.Scatter(x=old_transfer_count['Date'], y=old_transfer_count['Transfer_Patient_Count'],
                                                    mode='lines+markers', name='Transfer Forecast', line=dict(color=line_color)))
                        fig3.add_trace(go.Scatter(x=transfer_forecast_data['Date'], y=transfer_forecast_data['Forecasted Transfer Count'],
                                                    mode='lines+markers', name='Transfer Forecast', line=dict(color=red_line_color)))
                        fig3.update_layout(title_text='Transfer Data and Forecast',
                                            xaxis_title='Date',
                                            yaxis_title='Transfer Patient Count',
                                            height=300,
                                            width=450,
                                            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                            xaxis=dict(type='category', tickangle=-45),
                                            title=dict(x=0.3, y=0.9)
                                            )
                        # Display the chart inside the expander
                        st.plotly_chart(fig3)
                st.write("")
                st.write("")
                
                if st.button('Generate Insights'):
                
                    st.subheader("Insights:")
                    st.write("")
                    st.write("")
                    # Insights display code...
                    ad_counts_data = pd.merge(admissions_forecast_data,discharge_forecast_data,on='Date',  how = 'inner')
                    adt_counts_data = pd.merge(ad_counts_data,transfer_forecast_data, on='Date', how='inner')
                    json_text = pp.pre_process_adt_insights(adt_counts_data)
                    json_text = json_text.split('```', 1)[1]
                    json_text = json_text.split('```', 1)[0]
                    # Streamlit app
                    # Creating a box with background color
                    box_style = f"background-color: #FFA500; padding: 2px; border-radius: 5px;"
                    st.markdown(f'<div style="{box_style}">', unsafe_allow_html=True)
                    # Extract JSON content
                    start_index = json_text.find('[')  # Find the start of JSON content
                    end_index = json_text.rfind(']') + 1  # Find the end of JSON content
                    json_content = json_text[start_index:end_index]
                    # Parse JSON text
                    parsed_data = json.loads(json_content)
                    
                    st.write(parsed_data)
                    # Iterate through the list of dictionaries
                    for d in parsed_data:
                        # Check if 'implications' key exists in the dictionary
                        if 'implications' in d:
                            # Rename 'implications' to 'implication'
                            d['implication'] = d.pop('implications')
                    # Print the modified list of dictionaries
                    # Divide data into three equal parts
                    chunk_size = len(parsed_data) // 3
                    columns_data = [parsed_data[i:i + chunk_size] for i in range(0, len(parsed_data), chunk_size)]
                    # Calculate maximum lines occupied by detail in any row
                    max_lines = max(max(len(row['implication'].split('\n')) for row in col) for col in columns_data)
                    # Create a container for three columns within the box
                    container_style = f"background-color: #gyhuij; padding: 20px; border-radius: 5px;"
                    #st.write('completed till here')
                    with st.container():
                        st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True)
                        # Display insights in three columns
                        col1, col2, col3 = st.columns(3)
                        for idx, (c1, c2, c3) in enumerate(zip(columns_data[0], columns_data[1], columns_data[2])):
                            with col1:
                                st.write(f"{idx + 1}. **{c1['observation']}**")
                                st.write(f"   - *Detail:* {c1['implication']}")
                                st.write("---")
                            with col2:
                                st.write(f"{idx + 1 + chunk_size}. **{c2['observation']}**")
                                st.write(f"   - *Detail:* {c2['implication']}")
                                st.write("---")
                            with col3:
                                st.write(f"{idx + 1 + 2 * chunk_size}. **{c3['observation']}**")
                                st.write(f"   - *Detail:* {c3['implication']}")
                                st.write("---")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Close the div for insights box
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Adjustments to old_date_range DataFrame (if needed)
                old_date_range.drop(columns=['hadm_id'], inplace=True, errors='ignore') 
                old_date_range.fillna(' ',inplace=True)
                old_date_range.rename(columns={
                    'subject_id': 'Patient ID',
                    'admittime': 'Admission Date',
                    'dischtime': 'Discharge Date',
                    'transtime': 'Transfer Date',
                    'admission_type': 'Admission Type',
                    'admission_location': 'Admission Location',
                    'discharge_location': 'Discharge Location',
                    'marital_status': 'Marital Status',
                    'race': 'Ethnicity'
                }, inplace=True)
                
                @st.cache_resource
                def filter_data(data, search_term):
                    filtered_df = data[
                        data['Patient ID'].astype(str).str.contains(search_term, case=False) |
                        data['Admission Date'].astype(str).str.contains(search_term, case=False) |
                        data['Discharge Date'].astype(str).str.contains(search_term, case=False) |
                        data['Transfer Date'].astype(str).str.contains(search_term, case=False) |
                        data['Admission Type'].astype(str).str.contains(search_term, case=False) |
                        data['Admission Location'].astype(str).str.contains(search_term, case=False) |
                        data['Discharge Location'].astype(str).str.contains(search_term, case=False) |
                        data['Marital Status'].astype(str).str.contains(search_term, case=False) |
                        data['Ethnicity'].astype(str).str.contains(search_term, case=False)
                    ]
                    return filtered_df

                # Load the data
                data = old_date_range
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader(f"List of Patients admitted in the last {n_days} day(s).")
                with col3:
                    search_term = st.text_input("Enter a value to search:")
                
                st.subheader("", divider='rainbow')
                st.write(f"Historical data from {old_start_date} to {old_end_date}")
                #Filter the data
                filtered_df = filter_data(data, search_term)
                #st.write("Filtered DataFrame:", filtered_df)
                                                        
                ## Filter the DataFrame based on the search term
                filtered_df = old_date_range[
                    old_date_range['Patient ID'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Admission Date'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Discharge Date'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Transfer Date'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Admission Type'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Admission Location'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Discharge Location'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Marital Status'].astype(str).str.contains(search_term, case=False) |
                    old_date_range['Ethnicity'].astype(str).str.contains(search_term, case=False)
                ]
                
                # Table styling
                css_styles = """
                <style>
                    table {
                        width: 100%;
                        margin: 1px 0 0 0; /* Add some space around the table */
                        border-spacing: 0; /* Remove spacing between cells */
                        border-collapse: collapse; /* Collapse borders */
                    }
                    th, td {
                        text-align: center;
                        padding: 8px; /* Add padding for readability */
                        border: thin solid #ddd; /* Add subtle borders to cells */
                    }
                    th {
                        font-weight: bold; /* Ensure header cells are bold */
                        background-color: #8DBAC1; /* Professional color for headers */
                        color: #FFFFFF; /* White color for header text */
                        text-align: center;
                        font-size: 14px;
                    }
                    tr:nth-child(even) {
                        background-color: #f2f2f2; /* Zebra stripes for rows */
                    }
                    tr:hover {
                        background-color: #ddd; /* Highlight row on hover */
                    }
                </style>
                """
                filtered_df = filtered_df[['Patient ID','Admission Date','Admission Type','Admission Location','Marital Status','Ethnicity']]
                # Convert DataFrame to HTML without index
                html_table = filtered_df.to_html(index=False, escape=False)
                
                # Concatenate the style and HTML table
                styled_html = f"{css_styles}\n{html_table}"
                
                # Render the styled table in Streamlit
                st.markdown(styled_html, unsafe_allow_html=True)
            
            elif st.session_state['last_clicked'] == "Emergency Dept.":
                st.header("Emergency Department")
                st.empty()
                
                # Default dates
                start_date = date(2023, 12, 1)
                end_date = date(2024, 12, 31)
            
                with st.form(key='date_form',border=False):
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        start_date, end_date = st.date_input("Select Date Range:", value=(start_date, end_date))
                    with col3:
                        st.write("")
                        st.write("")
                        submit_button = st.form_submit_button(label='RUN')
            # Logic for processing the form input outside the form
                if submit_button:
                    st.markdown(f"##### Below are the results FROM: {start_date} TO: {end_date}")
                                                                                                                                        
                    if 'icu_patients_ecg_prediction' not in st.session_state:
                        object_key = 'models/demo/icu_patients_ecg_prediction.pkl'  # Updated object key for the .pkl file
                        # Load the updated model from S3
                        st.session_state['icu_patients_ecg_prediction'] = connect.load_pkl_from_s3(bucket_name, object_key, aws_access_key_id, aws_secret_access_key, aws_region)
                    #    st.write('icu_patients_ecg_prediction completed')
        
                    ## Update ICU Patients Mortality Prediction Model
                    if 'icu_patients_mortality_prediction' not in st.session_state:
                        object_key = 'models/demo/icu_patients_mortality_prediction.pkl'  # Updated object key for the .pkl file
                        # Load the updated model from S3
                        st.session_state['icu_patients_mortality_prediction'] = connect.load_pkl_from_s3(bucket_name, object_key, aws_access_key_id, aws_secret_access_key, aws_region)
                        #st.write('icu_patients_mortality_prediction completed')
        
                    ## Update Gout Pipeline Model
                    if 'gout_pipeline' not in st.session_state:
                        object_key = 'models/demo/gout_pipeline.pkl'  # Updated object key for the .pkl file
                        # Load the updated model from S3
                        st.session_state['gout_pipeline'] = connect.load_pkl_from_s3(bucket_name, object_key, aws_access_key_id, aws_secret_access_key, aws_region)
                        #st.write('gout_pipeline completed')
        
                    #st.write(start_date,end_date)

                    start_date = start_date.strftime('%Y-%m-%d')
                    end_date = end_date.strftime('%Y-%m-%d')
                    
                    features = connect.mortality_features()
                    master  = connect.data_connection_ed_mortality(start_date,end_date)
        
                    features_data = pp.pre_process_mortality_feature_list(features)
                    mortal_data = pp.pre_process_mortality(master)
        
                    df_merge = mortal_data.copy()
                    df_merge['index'] = range(0, len(df_merge))
                    df_merge = df_merge[['index','subject_id','age','height_cm','ethnicity','edadmit_time']]
        #
                    final_data = pp.model_input_mortality(mortal_data,features_data)
                    final_data['height_cm'] = pd.to_numeric(final_data['height_cm'], errors='coerce')  # 'coerce' will convert non-numeric values to NaN
        #
                    mortality_model = st.session_state['icu_patients_mortality_prediction']
                    prediction_probabilities = mortality_model.predict_proba(final_data)
        #
        #
                    probabilities_df = pd.DataFrame(prediction_probabilities).reset_index()
                    mortality_pred = pd.merge(df_merge,probabilities_df,on='index',how ='inner')
                    mortality_pred.drop(columns = 'index',inplace = True)
                    mortality_pred.rename(columns={0:'Alive',1:'Dead'},inplace = True)
        #
                    # Exclude the index column from the DataFrame
                    mortality_pred = mortality_pred.copy()
                    mortality_pred.index = [''] * len(mortality_pred)
        #
                    mortality_pred['height_cm'] = mortality_pred['height_cm'].astype(float)
                    mortality_pred['Alive'] = mortality_pred['Alive'].astype(float)
                    mortality_pred['Dead']  = mortality_pred['Dead'].astype(float)
        #
                    mortality_pred['height_cm'] = mortality_pred['height_cm'].apply(lambda x: f'{x:.1f}')
                    mortality_pred['Alive'] = mortality_pred['Alive'].apply(lambda x: f'{x:.1%}')
                    #mortality_pred['Dead'] = mortality_pred['Dead'].apply(lambda x: f'{x:.1%}')
        #
                    # Display the formatted DataFrame using st.table()
                    mortality_pred.rename(columns={'height_cm':'Height','subject_id':'Patient ID',
                                                                        'ethnicity':'Ethnicity','edadmit_time':'ED: Admission Date',
                                                                        'Alive':'Probabilty of being Alive'
                                                                        #'Dead':'Probabilty of being Dead'
                                                                        },inplace=True)
        #
                    mortality_pred = mortality_pred[['ED: Admission Date', 'Patient ID', 'Probabilty of being Alive']]
                    features = connect.ecg_features()
                    master  = connect.data_connection_ed_ecg(start_date,end_date)
        #
                    features_data = pp.pre_process_ecg_feature_list(features)
                    ecg_data = pp.pre_process_ecg(master)
        #
                    df_merge = ecg_data.copy()
                    df_merge['index'] = range(0, len(df_merge))
                    ecg_df_merge = df_merge[['index','subject_id','edadmit_time','ecg_time']]
                    final_data = pp.model_input_ecg(ecg_data,features_data)
        #
                    ecg_model = st.session_state['icu_patients_ecg_prediction']
                    ecg_output=pd.DataFrame(ecg_model.predict(final_data)).reset_index()
        #
                    ecg_output.rename(columns={0:'ECG Results'},inplace = True)
                    class_mapping = {0:'Normal ECG', 1: 'Abnormal ECG' }
                    ecg_output['ECG Results'].replace(class_mapping, inplace=True)
        #
                    ecg_pred = pd.merge(ecg_df_merge,ecg_output,on='index',how ='inner')
                    ecg_pred.drop(columns = 'index',inplace = True)
                    ecg_pred.rename(columns={0:'Alive',1:'Dead'},inplace = True)
        #
                    # Display the formatted DataFrame using st.table()
                    ecg_pred.rename(columns={'subject_id':'Patient ID',
                                                                'edadmit_time':'ED: Admission Date',
                                                                'ecg_time':'ECG Performed On'
                                                                }
                                                                ,inplace=True)
        
                    display_df = pd.merge(mortality_pred,ecg_pred, on=['Patient ID','ED: Admission Date'],how='right')
        
                    display_df2 = display_df.copy()
        
        
                    master  = connect.data_connection_ed_gout(start_date,end_date)
        
                    gout_model = st.session_state['gout_pipeline']
        
                    gout_data = pp.model_input_gout(master, gout_model)
        
                    gout_model = gout_model['model']
        
                    # Predict using the loaded model
                    prediction = pd.DataFrame(gout_model.predict(gout_data))
                    prediction.replace({0: 'N', 2: 'Y'},inplace=True)
                    gout_prediction = prediction.reset_index()
        
                    display_df2 = display_df2.reset_index()
                    combined_df = pd.merge(display_df2,gout_prediction, on='index', how= 'left')
                    combined_df.drop(columns='index',inplace=True)
                    combined_df.rename(columns={0: 'Gout Prediction'}, inplace=True)
                    # Exclude the index column from the DataFrame
                    combined_df.index = [''] * len(display_df)
                    #combined_df.reset_index(drop=True, inplace=True)
                    combined_df = combined_df[['Patient ID','ED: Admission Date','Probabilty of being Alive','ECG Performed On','ECG Results','Gout Prediction']]
                    #combined_df.fillna('Patient vital sign readings are inaccurate', inplace=True)
                    combined_df.dropna(inplace=True)
                    css_styles = """
                    <style>
                                            table {
                                                width: 100%;
                                                margin: 1px 0 0 0; /* Add some space around the table */
                                                border-spacing: 0; /* Remove spacing between cells */
                                                border-collapse: collapse; /* Collapse borders */
                                            }
                                            th, td {
                                                text-align: center;
                                                padding: 8px; /* Add padding for readability */
                                                border: thin solid #ddd; /* Add subtle borders to cells */
                                            }
                                            th {
                                                font-weight: bold; /* Ensure header cells are bold */
                                                background-color: #8DBAC1; /* Professional color for headers */
                                                color: #FFFFFF; /* White color for header text */
                                                text-align: center;
                                                font-size: 14px;
                                            }
                                            tr:nth-child(even) {
                                                background-color: #f2f2f2; /* Zebra stripes for rows */
                                            }
                                            tr:hover {
                                                background-color: #ddd; /* Highlight row on hover */
                                            }
                    </style>
                    """
                    # Convert DataFrame to HTML without index
                    html_table = combined_df.to_html(index=False, escape=False)
        
                    # Concatenate the style and HTML table
                    styled_html = f"{css_styles}\n{html_table}"
        
                    # Render the styled table in Streamlit
                    st.markdown(styled_html, unsafe_allow_html=True)
                    
##    elif authenticator_status == False:
##        st.error("Username/Password is incorrect!")
##
##    elif authenticator_status == None:
##        st.warning("Please enter your username and password!")