import streamlit as st
import pandas as pd
import numpy as np
import glob
from collections import Counter
from music21 import converter, note, chord, key, tempo
from io import BytesIO
import tempfile
import os
import joblib
import shutil
from ast import literal_eval

# Create a temporary directory
TEMP_DIR = tempfile.mkdtemp()
import sys
sys.path.append('../')
from src.utils import extract_features_from_mdi  #, preprocess_inference_data
# scaler = joblib.load('feature_scaler.pkl')
# features_list = joblib.load('../models/features_list.csv')
# model_pipeline = joblib.load('../models/best_model_pipeline_0.868_score.joblib')
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def preprocess_inference_data(df, features_list, scaler):
    """
    Preprocesses the inference data by applying the same transformations as the training data.

    Parameters:
    - df: pandas DataFrame containing the inference data.
    - features_list: list of features to include in the model.
    - scaler: Scaler object used to scale the training data.

    Returns:
    - numpy array containing the preprocessed inference data.
    """    
    transformed_df = pd.DataFrame()
    for i in range(df.shape[0]):
        individual_df = pd.DataFrame(df.iloc[i]).T
        individual_df.drop(['most_frequent_key_signature','most_common_chord','most_common_instrument','measure_count'], axis=1, inplace=True)
        individual_df.dropna(inplace=True)
        individual_df.reset_index(drop=True, inplace=True)

        # Safely parse strings using literal_eval
        individual_df['note_density'] = individual_df['note_density'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) else x
        )
        # individual_df['note_density'] = individual_df['note_density'].apply(lambda x: eval(x))


        individual_df['total_duration'] = individual_df['total_duration'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) else x
        )
        # individual_df['total_duration'] = individual_df['total_duration'].apply(lambda x: eval(x))
        

        individual_df['most_frequent_time_signature'] = individual_df['most_frequent_time_signature'].apply(lambda x: eval(x))
        # Apply one-hot encoding to the categorical features
        key_name_dummies_unknown = pd.get_dummies(individual_df['key_name'], dtype=int)
        key_mode_dummies_unknown = pd.get_dummies(individual_df['key_mode'], dtype=int)
        individual_df = pd.concat([individual_df, key_name_dummies_unknown, key_mode_dummies_unknown], axis=1)
        individual_df.drop(['key_name', 'key_mode'], axis=1, inplace=True)
        # drop the 'composer' column and get the features
        individual_df = individual_df.drop('composer', axis=1)
        # key names that needs to be present in the inference data
        feature_list_key_names = features_list[21:33]
        new_keys = list(set(feature_list_key_names) - set(list(individual_df.columns)))
        # handle the case where the key names are not present in the inference data
        for key in new_keys:
            individual_df[key] = 0
        feature_list_key_mode = features_list[-1]
        # key mode that needs to be present in the inference data
        new_mode = list(set([feature_list_key_mode]) - set(list(individual_df.columns)))
        # handle the case where the key mode is not present in the inference data
        for mode in new_mode:
            individual_df[mode] = 0
        individual_df = individual_df[features_list]
        transformed_df = pd.concat([transformed_df, individual_df], ignore_index=True)
        # Scale the features using the scaler object
        inference_data = scaler.transform(transformed_df)
    return inference_data


# Build paths to the model, scaler, and feature list
model_dir = os.path.join(current_dir, "../model")
scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
features_list_path = os.path.join(model_dir, "features_list.csv")
model_pipeline_path = os.path.join(model_dir, "best_model_pipeline_0.868_score.joblib")

# Load files
scaler = joblib.load(scaler_path)
features_list = pd.read_csv(features_list_path)
features_list = features_list['features'].tolist()
model_pipeline = joblib.load(model_pipeline_path)


# set page configuration
st.set_page_config(
    page_title="Classifier",
    page_icon = ':bar_chart:',
    layout="wide",
    initial_sidebar_state="expanded",
)
st.image('images/deloitte_logo.png', width=100)
st.title('Classification Model Powered by SFL Scientific (A Deloitte Business)')
st.sidebar.image('images/sfl_logo.png', use_column_width=True)

menu_1= ['About', 'Demo']
choice_1 = st.sidebar.selectbox('Select Option', menu_1)

if choice_1 == 'About':
    st.write('''
    This is a classification model that predicts the likelihood of a customer to purchase a product. 
    The model is powered by SFL Scientific, a Deloitte Business. 
    ''')
elif choice_1 == 'Demo':
    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Upload your file(s)", type=['mid'], accept_multiple_files=True)
    if uploaded_files:
        st.write("Uploaded files:")
        for uploaded_file in uploaded_files:
            file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            # Save file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.write(f"- {uploaded_file.name}")

        if st.button("Delete Files"):
            # Delete all temporary files
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)  # Recreate an empty temporary directory
            st.success("All uploaded files have been deleted.")

        if st.button("Process Files"):
            # Example of processing files
            for file_name in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file_name)
                st.write(f"Processing: {file_path}")
            features = extract_features_from_mdi(os.path.join(TEMP_DIR, ''), composer='Unknown')
            features_html = """
            <span style="color:gold; font-size:20px; font-weight:bold;">Features</span>
            """
            st.markdown(features_html, unsafe_allow_html=True)
            # add the corresponding file name to the features DataFrame
            st.write(features)
            st.success("Files processed.")
            inference_data = preprocess_inference_data(features, features_list, scaler)
            classification = model_pipeline.predict(inference_data)
            # Get the maximum probability of each class for each sample in the inference data set
            max_probs = model_pipeline.predict_proba(inference_data).max(axis=1)

            threshold = 0.5 # a set threshold for classification
            class_dict = {'Bach': 0, 'Beethoven': 1, 'Brahms': 2, 'Schubert': 3}
            # Classify the samples based on the threshold
            labels = []
            for i in range(len(classification)):
                if max_probs[i] > threshold:
                    # Get the class label from the dictionary
                    labels.append(list(class_dict.keys())[list(class_dict.values()).index(classification[i])])
                else:
                    labels.append('Unknown')
            # Create a DataFrame to display the results
            results = pd.DataFrame({'Composer': labels, 'Probability': max_probs})
            st.write(results)
                
    