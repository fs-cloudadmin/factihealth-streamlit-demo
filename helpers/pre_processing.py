from sklearn.base import BaseEstimator, TransformerMixin
import openai
import pandas as pd
import pickle
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

def pre_process_adt_insights(data):
    
    predicted_data_string = "\n".join(f"On {row['Date']} the predicted count of admissions is {row['Forecasted Admissions Count']}, predicted count of discharges is {row['Forecasted Discharge Count']} and predicted count of transfers is {row['Forecasted Transfer Count']}."
                            for index, row in data.iterrows())

    persona = """
            Imagine you are a Hospital Administrator, adept in the dynamic and challenging environment of healthcare management. Your primary role
            revolves around scrutinizing a predictive dashboard, which provides a comprehensive view of the data provided forecasted patient admissions, discharges, and transfers. Equipped with strong analytical skills, you adeptly interpret these data trends
            to make critical operational decisions, ensuring optimal resource allocation and staffing efficiency. Your days are marked by coordinating
            with various departments, managing staffing schedules, and adjusting resources in response to fluctuating patient flows. With a background
            in healthcare administration and a keen understanding of healthcare systems and policies, you are well-versed in contingency planning,
            ready to adapt to unforeseen circumstances. Your leadership is defined by quick decision-making, effective communication, and a
            detail-oriented approach, ensuring the hospital operates smoothly and continues to provide high-quality patient care.
            """
    task = """
        You are tasked with extracting insights from the data received from the hospital's predictive dashboard. This critical role 
        requires you to analyze forecasted data for patient admissions, discharges, and transfers. Your aim is to identify key trends, patterns, and outliers in this data that
        could significantly influence hospital operations. Provide through insights which involves not only a deep understanding of the data but 
        also a foresighted approach to foresee potential challenges and opportunities it indicates. Your insights should focus on optimizing 
        resource allocation, effectively managing staffing, and enhancing overall operational efficiency. Importantly, each insight must be 
        accompanied by a justification, explaining why it is necessary and how it will positively impact the hospital's ability to provide 
        high-quality patient care while adapting to the dynamic healthcare environment. Need data in json format with atleast 6 different points as date range, observations, implications and actions.
        """
    context = f"""
            Am a hospital administrator. I have patients count for Admissions, Discharges and Transfers. 
            The {data} is the forecasted data for Admissions, Discharges and Transfers of my hospital.
            Based on the {data} I need to generate Insights with specific graphs or data points. 
            This data will be consumed for hospital staffing, bed utilization and other hospital administration purposes.
            """
    template = (
            """{persona}
            {context}
            {task}"""
        )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = """Here is the data from the dashboard
            {predicted_data_string}
            """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    OPENAI_API_KEY = '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' #'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' #'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'
    chat = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
    #with open("adt_chat_model.pkl",'rb') as model_file:
     #   chat=pickle.load(model_file)
    chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
    )
    # get a chat completion from the formatted messages
    response = chat(
        chat_prompt.format_prompt(
            persona=persona, task=task, predicted_data_string=data,
            context = context
        ).to_messages()
    )
    output = response.content

    return output

def pre_process_mortality_feature_list(data):
        # dropping columns which are not required
    data.drop(['ph_dipstick','ethnicity','admission_type','mixed_venous_o2_sat','first_careunit','age_bucket','icu_los','edadmittime','eddischargetime'],axis = 1, inplace = True)

    # Cleaning the data by adding limits to remove unrealistic data due to human/techincal errors

    condition_1 = data['non_invasive_blood_pressure_diastolic'] > 200
    data.loc[condition_1, 'non_invasive_blood_pressure_diastolic'] = data.loc[condition_1, 'non_invasive_blood_pressure_mean']
    data = data[(data['heart_rate_mean'] >= 0) & (data['heart_rate_mean'] <= 500)]
    data = data[(data['respiratory_rate'] >= 0) & (data['respiratory_rate'] <= 100)]
    data = data[(data['temperature_fahrenheit'] >= 0) & (data['temperature_fahrenheit'] <= 250)]
    data = data[(data['admission_weight_lbs'] >= 0) & (data['admission_weight_lbs'] <= 1000)]
    data = data[(data['o2_saturation_pulseoxymetry2'] >= 0) & (data['o2_saturation_pulseoxymetry2'] <= 100)]
    data = data[(data['height_cm'] >= 0) & (data['height_cm'] <= 300)]
    data.drop(['non_invasive_blood_pressure_mean'],axis = 1, inplace = True)

    df_encoded = pd.get_dummies(data, columns=['admission_type_combined', 'first_careunit_combined','ethnicity_combined','gender'])

    mean_value = df_encoded['non_invasive_blood_pressure_diastolic'].mean()

    # Replace NaNs in the column with the calculated mean value
    df_encoded['non_invasive_blood_pressure_diastolic'].fillna(mean_value, inplace=True)
    df_encoded['gcs_eye_opening'].fillna(0, inplace=True)
    df_encoded['gcs_verbal_response'].fillna(0, inplace=True)
    df_encoded['gcs_motor_response'].fillna(0, inplace=True)

    df_encoded.drop(columns=['subject_id','hadm_id','hospital_expire_flag'],inplace = True)
    df_sorted = df_encoded.sort_index(axis=1)
    return df_sorted

def pre_process_mortality(data):    
    # dropping columns which are not required
    data.drop(['ph_dipstick','hospital_expire_flag','admission_type','mixed_venous_o2_sat','first_careunit','age_bucket','icu_los','edadmittime','eddischargetime'],axis = 1, inplace = True)

    # Cleaning the data by adding limits to remove unrealistic data due to human/techincal errors

    condition_1 = data['non_invasive_blood_pressure_diastolic'] > 200
    data.loc[condition_1, 'non_invasive_blood_pressure_diastolic'] = data.loc[condition_1, 'non_invasive_blood_pressure_mean']
    data = data[(data['heart_rate_mean'] >= 0) & (data['heart_rate_mean'] <= 500)]
    data = data[(data['respiratory_rate'] >= 0) & (data['respiratory_rate'] <= 100)]
    data = data[(data['temperature_fahrenheit'] >= 0) & (data['temperature_fahrenheit'] <= 250)]
    data = data[(data['admission_weight_lbs'] >= 0) & (data['admission_weight_lbs'] <= 1000)]
    data = data[(data['o2_saturation_pulseoxymetry2'] >= 0) & (data['o2_saturation_pulseoxymetry2'] <= 100)]
    data = data[(data['height_cm'] >= 0) & (data['height_cm'] <= 300)]
    data.drop(['non_invasive_blood_pressure_mean'],axis = 1, inplace = True)
    
    df_encoded = pd.get_dummies(data, columns=['admission_type_combined', 'first_careunit_combined','ethnicity_combined','gender'])
    
    mean_value = df_encoded['non_invasive_blood_pressure_diastolic'].mean()

    # Replace NaNs in the column with the calculated mean value
    df_encoded['non_invasive_blood_pressure_diastolic'].fillna(mean_value, inplace=True)
    df_encoded['gcs_eye_opening'].fillna(0, inplace=True)
    df_encoded['gcs_verbal_response'].fillna(0, inplace=True)
    df_encoded['gcs_motor_response'].fillna(0, inplace=True)
    
    #df_encoded.drop(columns=['subject_id','hadm_id'],inplace = True)
    df_sorted = df_encoded.sort_index(axis=1)
    return df_sorted

def model_input_mortality(data_1,data_2):
    
    data = data_1
    features_data = data_2
    
    missing_columns = set(features_data.columns)-set(data.columns)

    if missing_columns:
        # Add missing columns and fill them with zeros
        for column in missing_columns:
            data[column] = 0  # Fill with zeros
            
    data.drop(columns=['subject_id','hadm_id','ethnicity','edadmit_time'],inplace = True)
    data=data.sort_index(axis=1)    
    
    return data

def pre_process_ecg_feature_list(data):
    df_encoded = pd.get_dummies(data, columns=['bandwidth','filtering','gender'])
    df_sorted = df_encoded.sort_index(axis=1)
    return df_sorted

def pre_process_ecg(data):
    df_encoded = pd.get_dummies(data, columns=['bandwidth','filtering','gender'])
    df_sorted = df_encoded.sort_index(axis=1)
    return df_sorted

def model_input_ecg(data_1,data_2):
    
    data = data_1
    features_data = data_2
    
    missing_columns = set(features_data.columns)-set(data.columns)

    if missing_columns:
        # Add missing columns and fill them with zeros
        for column in missing_columns:
            data[column] = 0  # Fill with zeros
            
    data.drop(columns=['subject_id','ecg_time','edadmit_time','dod','target_variable'],inplace = True)
    data=data.sort_index(axis=1)
    
    return data


def preProcessGout(user_input, tfidf_vectorizer, label_encoder, gout_model):
    
    # Preprocess the new data using the TF-IDF vectorizer
    new_data_vectorized = tfidf_vectorizer.transform([user_input])
    new_data_predictions = gout_model.predict(new_data_vectorized)
    decoded_predictions = label_encoder.inverse_transform(new_data_predictions)
    return decoded_predictions


def model_input_gout(data, gout_model):
    # data.reset_index(inplace=True)
    # data['chiefcomplaint'] = data['chiefcomplaint'].fillna('NA')
    # Extract the components
    tfidf_transformer = gout_model['tfidf_transformer']
    tfidf_vector = tfidf_transformer.transform(data)
    return tfidf_vector