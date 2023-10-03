# -*- coding: utf-8 -*-

#App structure
#This is the first page, others found in ./pages
#To run locally: streamlit run Train.py

import streamlit as st
st.set_page_config(
    page_title="xAnomaly",
    page_icon= None,
    layout="wide"
)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


# #Hardcoded Paths- you need to change this to your path!
# MAIN_FOLDER = os.getcwd()
# if MAIN_FOLDER == '/home/sm/ml/yggio-ai-deployment-examples-dev/app':
#     PROJECT_FOLDER = MAIN_FOLDER
# else:
#     PROJECT_FOLDER = MAIN_FOLDER + '/ml/yggio-ai-deployment-examples-dev/app'
# MODELPATH = PROJECT_FOLDER + '/models'

# import sys
# sys.path.append(PROJECT_FOLDER) #For accessing backend when developing
# import backend.data_extraction as data_extraction
# import backend.train_models as train_models
# import backend.send_email as send_email




#For deployment
PROJECT_FOLDER = os.path.dirname(__file__)
print("Root dir:", PROJECT_FOLDER)
MODELPATH =  PROJECT_FOLDER + '/models'
import backend.data_extraction as data_extraction
import backend.train_models as train_models
import backend.send_email as send_email



#UI
st.title('xAnomaly')



#BACKEND & LOGIC
#Load data needed from Yggio to see population of sensors

#Hardcoded authentication
SERVER, USERNAME, PASSWORD = data_extraction.get_credentials(PROJECT_FOLDER + "/config.yaml")

#Hardcoded model support
MEASUREMENTS = ['value_data']
train_models.NUM_COL_NAMES = MEASUREMENTS

# E-mail sender settings
EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SERVER, EMAIL_PORT = send_email.get_sender_info(PROJECT_FOLDER + "/config.yaml")



#Continue load sensor data
data_extraction.authorize_and_create_session()
pop = data_extraction.load_sensor_metadata()
names = list(pop.name)
_ids = list(pop._id)



#UI

#Session state and callbacks
if 'training_initalized' not in st.session_state:
    st.session_state.training_initalized = False

def train_button_clicked():
    st.session_state.training_initalized = True
    
#Outputs    
st.write('Select sensors to add anomaly detection on:')
selected_names = st.multiselect(
    'Select sensors to add anomaly detection models to:',
    names,
    ['Daphnet Historic S03R01E0 4 '], 
    label_visibility="collapsed")
st.write('It will take a while to train anomaly detection models, get an e-mail when they are ready:')
mail_adress = st.text_input('E-mail:', 'anna@andersson.se', label_visibility="collapsed")
if st.session_state.training_initalized & (mail_adress == 'anna@andersson.se'):
    warning_placeholder = st.empty()
    warning_placeholder.warning('No e-mail entered: Come back later to check if the models are ready in the Inspect- tab', icon="⚠️")
st.button('Train', on_click=train_button_clicked, type="primary")



#BACKEND & LOGIC MINGLED WITH UI
once = True #Think I need to set this not to rerun training on page auto reload (eg. user must realod page)
if st.session_state.training_initalized & once:
    
    selected_ids = [_ids[names.index(selected_name)] for selected_name in selected_names]
    pop = pop[pop['_id'].isin(selected_ids)]
    
    with st.spinner('Wait for training to complete...'):
        print("Start loading data of: ", selected_ids)
        data_extraction.extract_training_data(pop, MEASUREMENTS, PROJECT_FOLDER)
        print("Finised loading data of: ", selected_ids)
        train_models.set_subpaths(PROJECT_FOLDER)
        train_models.train(PROJECT_FOLDER)
        print("Training done of: ", selected_ids)
    st.success('Training done, inspect your models in the Inspect- tab!')

    if mail_adress != 'anna@andersson.se':
        send_email.send_training_done_email(PROJECT_FOLDER, mail_adress)
    else:
        warning_placeholder.empty()
    once = False
    st.session_state.training_initalized = False
    

