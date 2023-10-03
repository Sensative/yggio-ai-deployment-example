#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts data before training
"""
import streamlit as st
import requests
import pandas as pd
import os
import numpy as np
import time
import yaml

@st.cache_resource
def get_credentials(path):
    
    global SERVER
    global USERNAME
    global PASSWORD
    
    #Load config and its parameters- you need to change this to your account!
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    SERVER = config["yggio_account"]["server"]
    USERNAME = config["yggio_account"]["username"]
    PASSWORD = config["yggio_account"]["password"]
    
    return SERVER, USERNAME, PASSWORD

@st.cache_resource
def authorize_and_create_session():
    
    global my_session
    
    response = requests.post(SERVER +'api/auth/local', json={"username": USERNAME,"password": PASSWORD})
    authorization = response.json()
    token = authorization['token']
    my_headers = {'Authorization' : 'Bearer ' + token + ''}
    my_session = requests.Session()
    my_session.headers.update(my_headers)

@st.cache_data
def load_sensor_metadata():
    #Load relevant nodes for the user to be able to select
    response = my_session.get(SERVER +'api/iotnodes')
    if response.status_code == 200:
        print('Metadata loaded')
    elif response.status_code == 401:
        print("Updated headers!")
    else:
        print("No response: ", response)
    jsonResponse = response.json()
    df = pd.json_normalize(jsonResponse)
    return df 


def extract_training_data(pop, measurements, path):
    
    ids = list(pop._id)

    pop['starttime'] = np.nan
    pop['endtime'] = np.nan
    
    start = time.time()
    for i in ids:
        
        df_collect=pd.DataFrame()
        
        for m in measurements:
     
            #Loop back in time to collect all datapoints
            call = SERVER +'api/iotnodes/'+i+'/stats?measurement='+m+''
            response = my_session.get(call)
            jsonResponse = response.json()
            df = pd.json_normalize(jsonResponse)
            startTimeLastCall = pd.to_datetime(min(df.time)) 
            startTimeLastCallUnix = int(startTimeLastCall.timestamp()*1000)
                
                
            maxIter = 100
            maxCounter = 0
            while (startTimeLastCallUnix > 0) and (maxCounter < maxIter):
                call = SERVER +'api/iotnodes/'+i+'/stats?measurement='+m+'&end=' + str(startTimeLastCallUnix)
                response = my_session.get(call)
                jsonResponse = response.json()
                if jsonResponse:        # Catches if return is succesfull but an empty list
                    dfLastCall = pd.json_normalize(jsonResponse)
                    startTimeLastCall = pd.to_datetime(min(dfLastCall.time)) 
                    startTimeLastCallUnix = int(startTimeLastCall.timestamp()*1000)
                    df = pd.concat([df, dfLastCall])                  
                else:
                    break
                maxCounter = maxCounter+1
            
            # Clean    
            df.drop_duplicates(inplace=True)
            df.time = pd.to_datetime(df.time)
            df.set_index('time', inplace=True)
            df.drop(columns = ['mean'], inplace=True)
            df.rename(columns={"value": m}, inplace=True)
    
            #Append column to other measurments of sensor
            df_collect = pd.concat([df_collect, df], axis=1)
    
        df_collect.to_csv(path + '/data/' +i+'.csv')
        
        #Append start and endtime to metadata
        pop['starttime'] = np.where(pop['_id'] == i, str(df_collect.index.min()), pop['starttime'])
        pop['endtime'] = np.where(pop['_id'] == i, str(df_collect.index.max()), pop['endtime'])
        pop.to_csv(path + '/pop.csv')
        
    end = time.time()
    print("Downloading data took (minutes)", (end - start)/60)