import os
#import load_dotenv
import pandas as pd
import boto3
import pickle
import redshift_connector

def data_connection_admissions(old_start_date, old_end_date):
    # Load path to environment variables
    #dotenv_path = 'helpers/database_config.env'
    #load_dotenv(dotenv_path)

    # Database connection parameters
    db_name = 'factihealth'
    db_user = 'fh_user'
    db_password = 'Facti@874'
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'
    db_port = 5439

    # Load path for admissions master data SQL query
    sql_file_path = 'queries/admissions_query.sql'
    with open(sql_file_path, 'r') as file:
        sql_query = file.read()

    # Replace placeholders in the SQL query
    sql_query = sql_query.replace('{old_start_date}', old_start_date.strftime('%Y-%m-%d'))
    sql_query = sql_query.replace('{old_end_date}', old_end_date.strftime('%Y-%m-%d'))

    try:
        # Use redshift_connector to establish a connection to the Redshift database
        with redshift_connector.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        ) as conn:
            # Execute the SQL query and fetch the data into a DataFrame
            admissions_master_data = pd.read_sql(sql_query, conn)
            return admissions_master_data
    except Exception as e:
        print(f"Database connection or query failed due to {e}")


def mortality_features():
    # Load path to environment variables
    #dotenv_path = 'helpers/database_config.env'
    #load_dotenv(dotenv_path)

    # Database connection parameters
    db_name = 'factihealth'
    db_user = 'fh_user'
    db_password = 'Facti@874'
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'
    db_port = 5439
    
    # Load path for admissions master data sql query
    sql_file_path = r'queries/ed_query.sql'
    with open(sql_file_path, 'r') as sql_file:
        sql_queries = sql_file.read().split(';')
    try:
        with redshift_connector.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
            ) as conn:
            # Execute the SQL query and fetch the data into a DataFrame
            df = pd.read_sql(sql_queries[0], conn)
        return df
        print("Query 1 executed successfully.")
    except Exception as e:
        print(f"Error executing Query 1: {str(e)}")

def data_connection_ed_mortality(start_date, end_date):
    # Load environment variables from .env file
    #dotenv_path = 'helpers/database_config.env'
    #load_dotenv(dotenv_path)

    # Database connection parameters
    db_name = 'factihealth'
    db_user = 'fh_user'
    db_password = 'Facti@874'
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'
    db_port = 5439

    # Load path for SQL query file
    sql_file_path = r'queries/ed_query.sql'

    with open(sql_file_path, 'r') as sql_file:
        sql_queries = sql_file.read().split(';')
    # Replace placeholders with formatted dates
#    sql_query = sql_queries[1].replace('{start_date}', start_date.strftime('%Y-%m-%d'))
#    sql_query = sql_query.replace('{end_date}', end_date.strftime('%Y-%m-%d'))
    sql_query = sql_queries[1].replace('{start_date}', start_date)
    sql_query = sql_query.replace('{end_date}', end_date)
    try:
        with redshift_connector.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
            ) as conn:
            # Execute the SQL query and fetch the data into a DataFrame
            df = pd.read_sql(sql_query, conn)
        return df
        print("Query 1 executed successfully.")
    except Exception as e:
        print(f"Error executing Query 1: {str(e)}")


def ecg_features():
    # Load environment variables from .env file
    #dotenv_path = 'helpers/database_config.env'
    #load_dotenv(dotenv_path)

    # Database connection parameters
    db_name = 'factihealth'
    db_user = 'fh_user'
    db_password = 'Facti@874'
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'
    db_port = 5439

    # Load path for SQL query file
    sql_file_path = r'queries/ed_query.sql'

    with open(sql_file_path, 'r') as sql_file:
        sql_queries = sql_file.read().split(';')
    
    try:
        with redshift_connector.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
            ) as conn:
            # Execute the SQL query and fetch the data into a DataFrame
            df = pd.read_sql(sql_queries[2], conn)
        return df
        print("Query 1 executed successfully.")
    except Exception as e:
        print(f"Error executing Query 1: {str(e)}")

def data_connection_ed_ecg(start_date, end_date):
    # Load environment variables from .env file
    #dotenv_path = 'helpers/database_config.env'
    #load_dotenv(dotenv_path)

    # Database connection parameters
    db_name = 'factihealth'
    db_user = 'fh_user'
    db_password = 'Facti@874'
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'
    db_port = 5439

    # Load path for SQL query file
    sql_file_path = r'queries/ed_query.sql'

    with open(sql_file_path, 'r') as sql_file:
        sql_queries = sql_file.read().split(';')

    # Replace placeholders with formatted dates
#    sql_query = sql_queries[3].replace('{start_date}', start_date.strftime('%Y-%m-%d'))
#    sql_query = sql_query.replace('{end_date}', end_date.strftime('%Y-%m-%d'))
    sql_query = sql_queries[3].replace('{start_date}', start_date)
    sql_query = sql_query.replace('{end_date}', end_date)
    
    try:
        with redshift_connector.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
            ) as conn:
            # Execute the SQL query and fetch the data into a DataFrame
            df = pd.read_sql(sql_query, conn)
        return df
        print("Query 1 executed successfully.")
    except Exception as e:
        print(f"Error executing Query 1: {str(e)}")

def data_connection_ed_gout(start_date, end_date):
    # Load environm
    #dotenv_path = 'helpers/database_config.env'
    #load_dotenv(dotenv_path)

    # Database connection parameters
    db_name = 'factihealth'
    db_user = 'fh_user'
    db_password = 'Facti@874'
    db_host = 'redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com'
    db_port = 5439

    # Load path for SQL query file
    sql_file_path = r'queries/ed_query.sql'

    with open(sql_file_path, 'r') as sql_file:
        sql_queries = sql_file.read().split(';')
    result_dataframes = []
    # Replace placeholders with formatted dates
#    sql_query = sql_queries[4].replace('{start_date}', start_date.strftime('%Y-%m-%d'))
#    sql_query = sql_query.replace('{end_date}', end_date.strftime('%Y-%m-%d'))
    sql_query = sql_queries[4].replace('{start_date}', start_date)
    sql_query = sql_query.replace('{end_date}', end_date)
    

    try:
        with redshift_connector.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
            ) as conn:
            # Execute the SQL query and fetch the data into a DataFrame
            df = pd.read_sql(sql_query, conn)
        return df
        print("Query 1 executed successfully.")
    except Exception as e:
        print(f"Error executing Query 1: {str(e)}")
        

def load_pkl_from_s3(bucket_name, object_key, aws_access_key_id, aws_secret_access_key, aws_region=None):
    # Create a session using explicit credentials
    session = boto3.Session(
        aws_access_key_id='AKIATTAFLHZHBIMCNXZX',
        aws_secret_access_key='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',
        region_name='ap-south-1'
    )
    # Initialize an S3 client from the session
    s3 = session.client('s3')

    try:
        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        
        # Read the object's content into bytes
        file_content = response['Body'].read()
        
        # Deserialize the bytes back into a Python object using pickle
        data = pickle.loads(file_content)
        
        return data
    except Exception as e:
        print(f"Error loading .pkl file from S3: {e}")
        return None