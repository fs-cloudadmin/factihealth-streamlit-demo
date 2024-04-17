import streamlit as st
import boto3
import io
import zipfile
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_model_from_zip_in_s3(bucket_name, object_key, model_file_name, aws_access_key_id, aws_secret_access_key, aws_region):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    zip_content = response['Body'].read()
    zip_buffer = io.BytesIO(zip_content)
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        with zip_ref.open(model_file_name, 'r') as model_file:
            model = pickle.load(model_file)
    return model

def load_model(model_name, object_key, model_file_name, aws_credentials):
    try:
        start_time = datetime.now()
        bucket_name, aws_access_key_id, aws_secret_access_key, aws_region = aws_credentials
        model = load_model_from_zip_in_s3(bucket_name, object_key, model_file_name, aws_access_key_id, aws_secret_access_key, aws_region)
        end_time = datetime.now()
        load_duration = end_time - start_time
        return model_name, model, load_duration
    except Exception as e:
        return model_name, None, e

def main():
    st.title('Model Loading Example')
    aws_access_key_id = 'AKIATTAFLHZHBIMCNXZX'
    aws_secret_access_key = 'M5fwSAj9TSfG4IkyUoyqoJtuFCXDBhCFT0AOedG0'
    aws_region = 'ap-south-1'
    bucket_name = 'factihealth'
    aws_credentials = (bucket_name, aws_access_key_id, aws_secret_access_key, aws_region)

    models = {
        'admission_model': ('models/demo/admission_model.zip', 'admission_model.pkl'),
        'discharge_model': ('models/demo/discharge_model.zip', 'discharge_model.pkl'),
        'transfer_model': ('models/demo/transfer_model.zip', 'transfer_model.pkl'),
        'icu_patients_mortality_prediction': ('models/demo/icu_patients_mortality_prediction.zip', 'icu_patients_mortality_prediction.pkl'),
        'icu_patients_ecg_prediction': ('models/demo/icu_patients_ecg_prediction.zip', 'icu_patients_ecg_prediction.pkl'),
        'gout_pipeline': ('models/demo/gout_pipeline.zip', 'gout_pipeline.pkl')
    }

    st.write("Loading models...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for model_name, (object_key, model_file_name) in models.items():
            future = executor.submit(load_model, model_name, object_key, model_file_name, aws_credentials)
            futures.append(future)

        for future in futures:
            model_name, model, load_duration = future.result()
            if model is not None:
                st.write(f"{model_name} loaded in {load_duration}.")
            else:
                st.error(f"Failed to load {model_name}: {load_duration}")

if __name__ == "__main__":
    st.header("hi")
    main()
