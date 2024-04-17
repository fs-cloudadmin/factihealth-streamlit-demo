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

st.set_page_config(layout="wide")

bucket_name = 'factihealth'

# Provide AWS credentials as strings without trailing commas
aws_access_key_id = 'AKIATTAFLHZHBIMCNXZX'
aws_secret_access_key = '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'
aws_region = 'ap-south-1'
bucket_name = 'factihealth'


# Main guard
if __name__ == "__main__":
    
    image = Image.open('background/Factihealth.jpg')
    aspect_ratio = image.height / image.width         # Calculate the new height to maintain the aspect ratio
    new_width = 500
    new_height = int(new_width * aspect_ratio)
    image = image.resize((new_width, new_height)) # Resize the image
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(image, width=new_width)
        box_style = f"background-color: #FFA500; padding: 2px; border-radius: 5px;"
    st.header("", divider='rainbow')
    
    with col3:
        # Center the "Logout" button horizontally using the column layout
        
        st.write('<div style="text-align: center;">', unsafe_allow_html=True)
        st.write("Today's Date: 04/03/2024")
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
            
                default_start_date = date(2024, 3, 4)
                default_end_date = date(2024, 3, 11)
                
                if 'form_submitted' not in st.session_state:
                    st.session_state.form_submitted = False
                
                with st.form(key='date_form',border=False):
                    col1, col2, col3, col4 = st.columns(4)

                    with col2:
                        # Use the first column for the date range input
                        start_date, end_date = st.date_input("Select Date Range:", (default_start_date, default_end_date))
                
                    with col3:
                        # Use the second column for the 'N' days input
                        n_days = st.number_input("Choose the historical data duration, in days", min_value=0, value=7)
                        
                    with col4:
                        # Add an empty line above the button
                        st.write("")
                        st.write("")
                        # Use the third column for the form submit button with a green arrow
                        submit_button = st.form_submit_button(label='RUN')
                    
                    if submit_button or st.session_state.form_submitted:
                        try:
                            st.session_state.form_submitted = True 
                            if start_date < default_start_date:
                                # Reset the start date to the default start date and inform the user
                                start_date = default_start_date
                                st.error(f"The start date has been reset to {default_start_date.strftime('%B %d, %Y')}. Start date cannot be before the default start date.")

                            old_end_date = date(2024, 3, 3)
                            old_start_date = old_end_date - timedelta(days=n_days)

                            st.markdown(f"##### Below are the forecast results FROM: {start_date} TO: {end_date}")

                            if st.session_state.form_submitted:

                            # S3 Bucket details
                                bucket_name = 'factihealth'
                                # Initialize an S3 client
                                s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

                                object_key = 'data/demo/Admissions_Data.csv'
                                # Get the object from the bucket
                                obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                                adt_historic_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                                adt_historic_data.rename(columns={'subject_id': 'Patient ID', 
                                                                    'admittime':  'Admission Date',
                                                                    'dischtime':  'Discharge Date',
                                                                    'transtime': 'Transfer Date'
                                                                    },inplace=True)
                                adt_historic_data_df = adt_historic_data[['Patient ID','Admission Date','Discharge Date','Transfer Date']]

                                patient_count_per_admission_date = adt_historic_data_df.groupby('Admission Date')['Patient ID'].count().reset_index(name='Patient Count')
                                patient_count_per_discharge_date = adt_historic_data_df.groupby('Discharge Date')['Patient ID'].count().reset_index(name='Patient Count')
                                patient_count_per_transfer_date = adt_historic_data_df.groupby('Transfer Date')['Patient ID'].count().reset_index(name='Patient Count')

                                patient_count_per_admission_date['Admission Date'] = pd.to_datetime(patient_count_per_admission_date['Admission Date'])
                                patient_count_per_discharge_date['Discharge Date'] = pd.to_datetime(patient_count_per_discharge_date['Discharge Date'])
                                patient_count_per_transfer_date['Transfer Date'] = pd.to_datetime(patient_count_per_transfer_date['Transfer Date'])

                                bar_color = 'green'
                                line_color = 'blue'
                                red_line_color = 'red'

                                col1,col2,col3 = st.columns(3)
                                # Logic for processing the form input outside the form
                                with col1:
                                    object_key = 'data/demo/admissions_forecast_data.csv'
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

                                    # Assuming old_start_date and old_end_date are defined
                                    date_range_hist = pd.date_range(start=old_start_date, end=old_end_date, freq='D')

                                    # Convert columns to datetime
                                    adt_historic_data_df['Admission Date'] = pd.to_datetime(adt_historic_data_df['Admission Date'])
                                    adt_historic_data_df['Discharge Date'] = pd.to_datetime(adt_historic_data_df['Discharge Date'])
                                    adt_historic_data_df['Transfer Date'] = pd.to_datetime(adt_historic_data_df['Transfer Date'])

                                    # Filter based on Admission Date within the historical date range
                                    hist_admissions_filtered_df = patient_count_per_admission_date[patient_count_per_admission_date['Admission Date'].isin(date_range_hist)]

                                    # Format the filtered DataFrame
                                    hist_admissions_filtered_df = pd.DataFrame({
                                        'Date': hist_admissions_filtered_df['Admission Date'].dt.date,
                                        'Historical Admissions Count': hist_admissions_filtered_df['Patient Count'].astype(int)
                                                                })
                                    hist_admissions_filtered_df = hist_admissions_filtered_df.sort_values(by='Date', ascending=True)

                                    with st.expander("Admissions Forecast Details"):
                                    # Your chart creation code here
                                        fig1 = go.Figure()
                                        fig1.add_trace(go.Bar(x=hist_admissions_filtered_df['Date'], y=hist_admissions_filtered_df['Historical Admissions Count'],
                                                                name='Admissions Actuals', marker_color=bar_color))
                                        fig1.add_trace(go.Scatter(x=hist_admissions_filtered_df['Date'], y=hist_admissions_filtered_df['Historical Admissions Count'],
                                                                    mode='lines+markers', name='Admissions Actuals', line=dict(color=line_color)))
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
                                    object_key = 'data/demo/discharge_forecast_data.csv'
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

                                    # Assuming old_start_date and old_end_date are defined
                                    date_range_hist = pd.date_range(start=old_start_date, end=old_end_date, freq='D')

                                    # Convert columns to datetime
                                    adt_historic_data_df['Admission Date'] = pd.to_datetime(adt_historic_data_df['Admission Date'])
                                    adt_historic_data_df['Discharge Date'] = pd.to_datetime(adt_historic_data_df['Discharge Date'])
                                    adt_historic_data_df['Transfer Date'] = pd.to_datetime(adt_historic_data_df['Transfer Date'])

                                    # Filter based on Discharge Date within the historical date range
                                    hist_discharge_filtered_df = patient_count_per_discharge_date[patient_count_per_discharge_date['Discharge Date'].isin(date_range_hist)]

                                    # Format the filtered DataFrame
                                    hist_discharge_filtered_df = pd.DataFrame({
                                    'Date': hist_discharge_filtered_df['Discharge Date'].dt.date,
                                    'Historical Discharge Count': hist_discharge_filtered_df['Patient Count'].astype(int)
                                                            })
                                    hist_discharge_filtered_df = hist_discharge_filtered_df.sort_values(by='Date', ascending=True)

                                    with st.expander("Discharge Forecast Details"):
                                    # Your chart creation code here
                                        fig1 = go.Figure()
                                        fig1.add_trace(go.Bar(x=hist_discharge_filtered_df['Date'], y=hist_discharge_filtered_df['Historical Discharge Count'],
                                                                name='Discharge Actuals', marker_color=bar_color))
                                        fig1.add_trace(go.Scatter(x=hist_discharge_filtered_df['Date'], y=hist_discharge_filtered_df['Historical Discharge Count'],
                                                                    mode='lines+markers', name='Discharge Actuals', line=dict(color=line_color)))
                                        fig1.add_trace(go.Scatter(x=discharge_forecast_data['Date'], y=discharge_forecast_data['Forecasted Discharge Count'],
                                                                    mode='lines+markers', name='Discharge Forecast', line=dict(color=red_line_color)))
                                        fig1.update_layout(title_text='Discharge Data and Forecast',
                                                            xaxis_title='Date',
                                                            yaxis_title='Discharge Patient Count',
                                                            height=300,
                                                            width=450,
                                                            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                                            xaxis=dict(type='category', tickangle=-45),
                                                            title=dict(x=0.3, y=0.9)
                                                            )
                                        # Display the chart inside the expander
                                        st.plotly_chart(fig1)

                                with col3:
                                    object_key = 'data/demo/transfer_forecast_data.csv'
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
                                    total_transfer = transfer_forecast_data['Forecasted Transfer Count'].sum()
                                    st.write(f"Total Transfer Forecast: {total_transfer}")

                                    # Assuming old_start_date and old_end_date are defined
                                    date_range_hist = pd.date_range(start=old_start_date, end=old_end_date, freq='D')

                                    # Convert columns to datetime
                                    adt_historic_data_df['Admission Date'] = pd.to_datetime(adt_historic_data_df['Admission Date'])
                                    adt_historic_data_df['Discharge Date'] = pd.to_datetime(adt_historic_data_df['Discharge Date'])
                                    adt_historic_data_df['Transfer Date'] = pd.to_datetime(adt_historic_data_df['Transfer Date'])

                                    # Filter based on Admission Date within the historical date range
                                    hist_transfer_filtered_df = patient_count_per_transfer_date[patient_count_per_transfer_date['Transfer Date'].isin(date_range_hist)]

                                    # Format the filtered DataFrame
                                    hist_transfer_filtered_df = pd.DataFrame({
                                    'Date': hist_transfer_filtered_df['Transfer Date'].dt.date,
                                    'Historical Transfer Count': hist_transfer_filtered_df['Patient Count'].astype(int)
                                                            })
                                    hist_transfer_filtered_df = hist_transfer_filtered_df.sort_values(by='Date', ascending=True)

                                    with st.expander("Transfer Forecast Details"):
                                    # Your chart creation code here
                                        fig1 = go.Figure()
                                        fig1.add_trace(go.Bar(x=hist_transfer_filtered_df['Date'], y=hist_transfer_filtered_df['Historical Transfer Count'],
                                                                name='Transfer Actuals', marker_color=bar_color))
                                        fig1.add_trace(go.Scatter(x=hist_transfer_filtered_df['Date'], y=hist_transfer_filtered_df['Historical Transfer Count'],
                                                                    mode='lines+markers', name='Transfer Actuals', line=dict(color=line_color)))
                                        fig1.add_trace(go.Scatter(x=transfer_forecast_data['Date'], y=transfer_forecast_data['Forecasted Transfer Count'],
                                                                    mode='lines+markers', name='Transfer Forecast', line=dict(color=red_line_color)))
                                        fig1.update_layout(title_text='Transfer Data and Forecast',
                                                            xaxis_title='Date',
                                                            yaxis_title='Transfer Patient Count',
                                                            height=300,
                                                            width=450,
                                                            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                                            xaxis=dict(type='category', tickangle=-45),
                                                            title=dict(x=0.3, y=0.9)
                                                            )
                                        # Display the chart inside the expander
                                        st.plotly_chart(fig1)


                                #-----------------------------------------------------------------------------------------------------------------
                                #------------------------------------ ADT Insights ---------------------------------------------------------------
                                #-----------------------------------------------------------------------------------------------------------------

                                st.header('Insights on ADT forecasts:',divider='rainbow')
                                with st.expander("Insights"):
                                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                                    object_key = 'data/demo/insights.csv'
                                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                                    adt_insights = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

                                    adt_insights = adt_insights[['Date','Analysis','Action Items']]
                                    adt_insights['Date'] = pd.to_datetime(adt_insights['Date'])

                                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                                    adt_insights_filetered = adt_insights[adt_insights['Date'].isin(date_range)]

                                    adt_insights_filetered['Date'] = adt_insights_filetered['Date'].dt.date


                                    def chunker(seq, size):
                                        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
                                    # Iterate over the DataFrame in chunks of 3 rows

                                    for chunk in chunker(adt_insights_filetered, 3):
                                        cols = st.columns(3)  # Create 3 columns for each day
                                        for i, (index, row) in enumerate(chunk.iterrows()):
                                            with cols[i]:
                                                st.markdown(f"**{row['Date']}**")
                                                st.markdown(f"**Analysis:**\n{row['Analysis']}")

                                                # Split Action Items by ';' and format as bullet points
                                                action_items = row['Action Items'].split(';')
                                                action_items_md = "\n".join([f"- {item.strip()}" for item in action_items if item])
                                                st.markdown(f"**Recommendations:**\n{action_items_md}")
                                                st.write("---") 
                        except Exception as e:
                            st.error(f"{e}")
                #---------------------------------------------------------------------------------------------------------------------                    
                #-------------------------------------- ADT Details ------------------------------------------------------------------
                #---------------------------------------------------------------------------------------------------------------------
                
                st.subheader( "")
                st.subheader( "")
                st.subheader( "", divider='rainbow')
                col1, col2 , col3 = st.columns(3)
                with col1:
                    default_start_date_1 = date(2024, 3, 3)
                    default_end_date_1 = date(2024, 3, 3)
                    st.subheader(f"Admission Details")
                
                if 'los_button_clicked' not in st.session_state:
                    st.session_state.los_button_clicked = False
                
                historic_start_date_1 = default_start_date_1
                historic_end_date_1 = default_end_date_1
                with col3:
                    selected_dates = st.date_input("Select Date Range:", (default_start_date_1, default_end_date_1),key='date_range_1')

                    if len(selected_dates) == 2:
                        historic_start_date_1, historic_end_date_1 = selected_dates
                    else: 
                        st.write('Please select both dates')
                try:
                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                    object_key = 'data/demo/Admissions_Data.csv'
                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                    adt_historic_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    adt_historic_data.rename(columns={'subject_id' : 'Patient ID',
                                                      'subject_name':'Patient Name',
                                                        'hadm_id'  : 'Admission ID',
                                                        'admittime': 'Registration Date',
                                                        'dischtime': 'Discharge Date',
                                                        'transtime': 'Transfer Date',
                                                        'admission_type': 'Admission Type',
                                                        'admission_location': 'Admission Location',
                                                        'discharge_location': 'Discharge Location',
                                                        'marital_status': 'Marital Status',
                                                        'race': 'Race',
                                                        'age': 'Age',
                                                        'gender': 'Gender',
                                                        'predicted_los': 'Predicted LOS (in days)'
                                                        },inplace=True)
                    adt_historic_data.fillna(' ',inplace=True)
                    adt_historic_data['Predicted LOS (in days)'] = adt_historic_data['Predicted LOS (in days)'].replace(' ', np.nan).fillna(0).astype(int)
                    adt_historic_data_1 = adt_historic_data[['Registration Date','Patient Name','Admission Type','Admission Location','Age','Gender','Marital Status','Race','Predicted LOS (in days)']]

                    adt_historic_data_1['Registration Date'] = pd.to_datetime(adt_historic_data_1['Registration Date'])
                    adt_historic_data_1 = adt_historic_data_1.sort_values(by='Registration Date',ascending= True)
                                                                        
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
                        background-color: #000000; /* Professional color for headers */
                        color: #FFFFFF; /* White color for header text */
                        text-align: center;
                        font-size: 18px;
                    }                                                            
                    tr:nth-child(even) {
                        background-color: #f2f2f2; /* Zebra stripes for rows */
                    }
                    tr:hover {
                        background-color: #ddd; /* Highlight row on hover */
                    }
                    </style>
                    """
                    adt_historic_data_1_los = adt_historic_data_1.copy()
                    adt_historic_data_1_los[['Predicted LOS (in days)']]=''
                    #st.write(adt_historic_data_1_los)
                    date_range_1 = pd.date_range(start=historic_start_date_1, end=historic_end_date_1, freq='D')
                    adt_historic_data_1_los = adt_historic_data_1_los[adt_historic_data_1_los['Registration Date'].isin(date_range_1)]
                    html_table = adt_historic_data_1_los.to_html(index=False, escape=False)
                                                            
                    if st.button('Click Here to Predict LOS') or st.session_state.los_button_clicked:
                        st.session_state.los_button_clicked = True
                        adt_historic_data_1 = adt_historic_data_1[adt_historic_data_1['Registration Date'].isin(date_range_1)]
                        html_table = adt_historic_data_1.to_html(index=False, escape=False)

                    # Concatenate the style and HTML table
                    styled_html = f"{css_styles}\n{html_table}"

                    # Render the styled table in Streamlit
                    st.markdown(styled_html, unsafe_allow_html=True)
                    st.subheader( "", divider='rainbow')
                except Exception as e:
                    st.error(f"Failed to fetch or process data: {e}")

                st.subheader("")
                #col1, col2 , col3 = st.columns(3)
                #with col1:
                #    default_start_date_2 = date(2024, 3, 3)
                #    default_end_date_2 = date(2024, 3, 3)
                st.subheader(f"Discharge Details")
                #with col3:
                #    historic_start_date_2, historic_end_date_2 = st.date_input("Select Date Range:", (default_start_date_2, default_end_date_2),key='date_range_2')
                
                if 'readmission_button_clicked' not in st.session_state:
                    st.session_state.readmission_button_clicked = False
                
                try:
                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                    object_key = 'data/demo/Readmission_static.csv'
                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                    readmission_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    readmission_data.rename(columns={'subject_id': 'Patient ID',
                                                     'subject_name':'Patient name',
                                        'hadm_id'     :'Admission ID',
                                        'discharge_location': 'Discharge Location',
                                        'dischtime': 'Discharge Date',
                                        '30_days' : 'Readmission Within 30 days',
                                        '60_days' : 'Readmission Within 60 days',
                                        '365_days': 'Readmission Within 365 days'},inplace=True)
                    readmission_data['Discharge Date'] = pd.to_datetime(readmission_data['Discharge Date'])
                    readmission_data = readmission_data[['Patient name','Discharge Date','Discharge Location','Readmission Within 30 days','Readmission Within 60 days','Readmission Within 365 days']]
                    readmission_data=readmission_data.sort_values(by='Discharge Date', ascending=True)
                    
                    readmission_data_preds = readmission_data.copy()
                    
                    readmission_data[['Readmission Within 30 days','Readmission Within 60 days','Readmission Within 365 days']]=''
                    readmission_data_1 = readmission_data[readmission_data['Discharge Date'].isin(date_range_1)]
                    # Convert DataFrame to HTML without index
                    html_table = readmission_data_1.to_html(index=False, escape=False)
                    
                    if st.button('Click Here to Predict Readmission') or st.session_state.readmission_button_clicked:
                        st.session_state.readmission_button_clicked = True
                        readmission_data_preds = readmission_data_preds[readmission_data_preds['Discharge Date'].isin(date_range_1)]
                        html_table = readmission_data_preds.to_html(index=False, escape=False)
                except Exception as e:
                    st.error(f"Failed to fetch or process data: {e}")
                
                
                # Concatenate the style and HTML table
                styled_html = f"{css_styles}\n{html_table}"
        
                # Render the styled table in Streamlit
                st.markdown(styled_html, unsafe_allow_html=True)
                st.subheader( "", divider='rainbow')

                st.subheader( "")
                st.subheader( "")
                #col1, col2 , col3 = st.columns(3)
                #with col1:
                #    default_start_date_3 = date(2024, 3, 3)
                #    default_end_date_3 = date(2024, 3, 3)
                st.subheader(f"Transfer Details")
                #with col3:
                #    historic_start_date_3, historic_end_date_3 = st.date_input("Select Date Range:", (default_start_date_3, default_end_date_3),key='date_range_3')


                s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                object_key = 'data/demo/Admissions_Data.csv'
                obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                adt_historic_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                adt_historic_data.rename(columns={'subject_id' : 'Patient ID',
                                                  'subject_name':'Patient Name',
                                                  'hadm_id'     :'Admission ID',
                                                    'admittime': 'Admission Date',
                                                    'dischtime': 'Discharge Date',
                                                    'transtime': 'Transfer Date',
                                                    'admission_type': 'Admission Type',
                                                    'admission_location': 'Transfer Location',
                                                    'discharge_location': 'Discharge Location',
                                                    'marital_status': 'Marital Status',
                                                    'race': 'Race',
                                                    'age': 'Age',
                                                    'gender': 'Gender'
                                                    },inplace=True)
                adt_historic_data.dropna(inplace=True)
                adt_historic_data_3 = adt_historic_data[['Patient Name','Transfer Date','Transfer Location']]

                adt_historic_data_3['Transfer Date'] = pd.to_datetime(adt_historic_data_3['Transfer Date'])
                adt_historic_data_3 = adt_historic_data_3.sort_values(by='Transfer Date',ascending= True)

                adt_historic_data_3 = adt_historic_data_3[adt_historic_data_3['Transfer Date'].isin(date_range_1)]
                
                # Convert DataFrame to HTML without index
                html_table = adt_historic_data_3.to_html(index=False, escape=False)

                # Concatenate the style and HTML table
                styled_html = f"{css_styles}\n{html_table}"

                # Render the styled table in Streamlit
                st.markdown(styled_html, unsafe_allow_html=True)
                st.subheader( "", divider='rainbow')
            
            #--------------------------------------------------------------------------------------------------------------
            #-----------------------------------------EMERGENCY DEPARTMENT TAB --------------------------------------------
            #--------------------------------------------------------------------------------------------------------------
            
            elif st.session_state['last_clicked'] == "Emergency Dept.":
                st.header("Emergency Department")
            
                default_start_date = date(2024, 3, 4)
                default_end_date = date(2024, 3, 18)

                if 'form_submitted1' not in st.session_state:
                    st.session_state.form_submitted1 = False
                
                with st.form(key='date_form',border=False):
                    col1, col2, col3, col4 = st.columns(4)

                    with col2:
                        # Use the first column for the date range input
                        start_date, end_date = st.date_input("Select Date Range:", (default_start_date, default_end_date))

                    with col3:
                        # Use the second column for the 'N' days input
                        n_days = st.number_input("Choose the historical data duration, in days", min_value=0, value=15)
                    with col4:
                        # Add an empty line above the button
                        st.write("")
                        st.write("")
                        # Use the third column for the form submit button with a green arrow
                        submit_button = st.form_submit_button(label='RUN')

                    if submit_button or st.session_state.form_submitted1:
                        try:
                            st.session_state.form_submitted1 = True 
                            
                            if start_date < default_start_date:
                                # Reset the start date to the default start date and inform the user
                                start_date = default_start_date
                                st.error(f"The start date has been reset to {default_start_date.strftime('%B %d, %Y')}. Start date cannot be before the default start date.")

                            old_end_date = date(2024, 3, 3)
                            old_start_date = old_end_date - timedelta(days=n_days)

                            st.markdown(f"##### Below are the forecast results FROM: {start_date} TO: {end_date}")


                            # S3 Bucket details
                            bucket_name = 'factihealth'
                            # Initialize an S3 client
                            s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

                            object_key = 'data/demo/mortal_ecg_dataset.csv'
                            # Get the object from the bucket
                            obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                            ed_historic_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

                            patient_count_per_ed_admission_date = ed_historic_data.groupby('ED: Admission Date')['Patient ID'].count().reset_index(name='Patient Count')
                            patient_count_per_ed_admission_date['ED: Admission Date'] = pd.to_datetime(patient_count_per_ed_admission_date['ED: Admission Date'])                                                

                            bar_color = 'green'
                            line_color = 'blue'
                            red_line_color = 'red'


                            # Logic for processing the form input outside the form
                            object_key = 'data/demo/ed_admissions_forecast_Data.csv'
                            obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                            # Read the CSV data into a pandas DataFrame
                            ed_admissions_df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                            ed_admissions_df['Date'] = pd.to_datetime(ed_admissions_df['Date'])

                            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

                            ed_admissions_filtered_df = ed_admissions_df[ed_admissions_df['Date'].isin(date_range)]
                            ed_admissions_forecast_data = pd.DataFrame({
                                'Date': ed_admissions_filtered_df['Date'].dt.date,
                                'Forecasted ED Admissions Count': ed_admissions_filtered_df['Forecasted ED Admissions Count'].astype(int)            
                                            })
                            total_ed_admissions = ed_admissions_forecast_data['Forecasted ED Admissions Count'].sum()
                            st.write(f"Total ED Admissions Forecast: {total_ed_admissions}")

                            # Assuming old_start_date and old_end_date are defined
                            date_range_hist = pd.date_range(start=old_start_date, end=old_end_date, freq='D')


                            # Filter based on Admission Date within the historical date range
                            hist_ed_admissions_filtered_df = patient_count_per_ed_admission_date[patient_count_per_ed_admission_date['ED: Admission Date'].isin(date_range_hist)]

                            # Format the filtered DataFrame
                            hist_ed_admissions_filtered_df = pd.DataFrame({
                                'Date': hist_ed_admissions_filtered_df['ED: Admission Date'].dt.date,
                                'Historical ED Admissions Count': hist_ed_admissions_filtered_df['Patient Count'].astype(int)
                                                        })
                            hist_ed_admissions_filtered_df = hist_ed_admissions_filtered_df.sort_values(by='Date', ascending=True)

                            #with st.expander('ED Admission Forecast'):
                                # Your chart creation code here
                            fig5 = go.Figure()
                            fig5.add_trace(go.Bar(x=hist_ed_admissions_filtered_df['Date'], y=hist_ed_admissions_filtered_df['Historical ED Admissions Count'],
                                                    name='ED Admissions Raw Data', marker_color=bar_color))
                            fig5.add_trace(go.Scatter(x=hist_ed_admissions_filtered_df['Date'], y=hist_ed_admissions_filtered_df['Historical ED Admissions Count'],
                                                        mode='lines+markers', name='ED Admissions Forecast', line=dict(color=line_color)))
                            fig5.add_trace(go.Scatter(x=ed_admissions_forecast_data['Date'], y=ed_admissions_forecast_data['Forecasted ED Admissions Count'],
                                                        mode='lines+markers', name='ED Admission Forecast', line=dict(color=red_line_color)))
                            fig5.update_layout(title_text='ED Admissions Data and Forecast',
                                                xaxis_title='Date',
                                                yaxis_title='ED Admissions Patient Count',
                                                height=450,
                                                width=1500,
                                                legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                                                xaxis=dict(type='category', tickangle=-45),
                                                title=dict(x=0.3, y=0.9)
                                                )

                            st.plotly_chart(fig5,)#use_container_width=True)
                        except Exception as e:
                            st.error(f"{e}")

                st.subheader("",divider='rainbow')
                st.subheader("Gout Prediction")
                user_input = st.text_input("Enter please chief complaint")
                
                
                if user_input:
                    object_key = 'models/demo/gout_prediction_xgb.pkl'  # Updated object key for the .pkl file
                    # Load the updated model from S3
                    tfidf_vectorizer, label_encoder, gout_model  = connect.load_pkl_from_s3(bucket_name, object_key, aws_access_key_id, aws_secret_access_key, aws_region)
                    gout_data = pp.preProcessGout(user_input,tfidf_vectorizer, label_encoder, gout_model)
                    
                    if gout_data == 'Y':
                        st.error("Our assessment indicates a potential risk for gout. Recommend scheduling a consultation with a healthcare provider for a comprehensive evaluation and to discuss the next steps.")
                    elif gout_data == 'N':
                        st.success("Our assessment does not indicate signs of gout.")
                    
                   
                st.subheader( "")
                st.subheader( "")
                st.subheader( "",divider='rainbow')
                col1, col2 , col3 = st.columns(3)
                with col1:
                    ed_default_start_date_1 = date(2024, 3, 3)
                    ed_default_end_date_1 = date(2024, 3, 3)
                    st.subheader(f"ED Admission Details")
                # Initialize variables with default values
                ed_historic_start_date_1 = ed_default_start_date_1
                ed_historic_end_date_1 = ed_default_end_date_1
                
                if 'ed_button_clicked' not in st.session_state:
                    st.session_state.ed_button_clicked = False
                
                with col3:
                    selected_dates = st.date_input("Select Date Range:", (ed_default_start_date_1, ed_default_end_date_1),key='date_range_4')
                    if len(selected_dates)==2:
                        ed_historic_start_date_1, ed_historic_end_date_1 = selected_dates
                    else:
                        st.write('Please select both dates')
                try:        
                    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
                    object_key = 'data/demo/mortal_ecg_dataset.csv'
                    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                    ed_historic_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

                    ed_historic_data['ED: Admission Date'] = pd.to_datetime(ed_historic_data['ED: Admission Date'])
                    ed_historic_data = ed_historic_data.sort_values(by='ED: Admission Date',ascending= True)
                    ed_historic_data.rename(columns={'ED: Admission Date':'Registration Date'},inplace=True)
                    ed_historic_data = ed_historic_data[['Registration Date','Patient Name','ECG Performed On','ECG Results','Probabilty of being Alive','Patient Severity Index','Auto Diagnosis Prediction']]
                    
                    ed_historic_data_filtered = ed_historic_data.copy()
                    ed_historic_data[['Probabilty of being Alive','ECG Results','Patient Severity Index','Auto Diagnosis Prediction']]=''
                    date_range_4 = pd.date_range(start=ed_historic_start_date_1, end=ed_historic_end_date_1, freq='D')
                    ed_historic_data = ed_historic_data[ed_historic_data['Registration Date'].isin(date_range_4)]
                    html_table = ed_historic_data.to_html(index=False, escape=False)
                    
                    if st.button('Click Here for ED Predictions') or st.session_state.ed_button_clicked:
                        st.session_state.ed_button_clicked = True
                        ed_historic_data_filtered = ed_historic_data_filtered[ed_historic_data_filtered['Registration Date'].isin(date_range_4)]
                        html_table = ed_historic_data_filtered.to_html(index=False, escape=False)
                        
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
                        background-color: #000000; /* Professional color for headers */
                        color: #FFFFFF; /* White color for header text */
                        text-align: center;
                        font-size: 18px;
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
                    

                    # Concatenate the style and HTML table
                    styled_html = f"{css_styles}\n{html_table}"

                    # Render the styled table in Streamlit
                    st.markdown(styled_html, unsafe_allow_html=True)
                    st.subheader( "", divider='rainbow')
                except Exception as e:
                    st.error(f"Failed to process data: {e}")
        
            
            elif st.session_state['last_clicked'] == "Surgical Procedure":
                st.header("Surgical Procedure")
            elif st.session_state['last_clicked'] == "Maternity Care":
                st.header("Maternity Care")
            elif st.session_state['last_clicked'] == "Specialized Clinics":
                st.header("Specialized Clinics")
            elif st.session_state['last_clicked'] == "Outpatient":
                st.header("Outpatient")
            elif st.session_state['last_clicked'] == "Dialysis Treatment":
                st.header("Dialysis Treatment")                                                
            elif st.session_state['last_clicked'] == "Rehab Service":
                st.header("Rehab Service")                
            elif st.session_state['last_clicked'] == "Home Health Care":
                st.header("Home Health Care")
            elif st.session_state['last_clicked'] == "Dialysis Treatment":
                st.header("Dialysis Treatment")