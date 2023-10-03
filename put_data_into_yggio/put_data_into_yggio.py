#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import requests
import json
import time
import yaml    

#Hardcoded Paths- you need to change this to your path!
MAIN_FOLDER = os.getcwd()
if MAIN_FOLDER== '/home/sm':
    PROJECT_FOLDER = MAIN_FOLDER + '/ml/yggio-ai-deployment-examples-dev/put_data_into_yggio'
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)


#Load config and its parameters- you need to change this to your account!
with open(PROJECT_FOLDER + "/../config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
SERVER = config["yggio_account"]["server"]
USERNAME = config["yggio_account"]["username"]
PASSWORD = config["yggio_account"]["password"]



#######
# Load and prepare data
#######


#Load local data
df_with_missing = pd.read_csv(PROJECT_FOLDER + '/data/S03R01E0.test.csv@4.out', header=None)
print("Len with missing: ", len(df_with_missing))
df = df_with_missing.dropna()
print("Len removed missing: ", len(df))
df=df.to_numpy()
data = df[:,0].astype(float)
label = df[:,1].astype(int)

#Here we emulate that some data is training and validation (historic data) and some test (future data)
test_frac = 0.5
index_to_cut = len(data)- int(len(data)*test_frac)
historic_data = data[0:index_to_cut]
historic_label = label[0:index_to_cut]
future_data = data[index_to_cut:len(data)]
future_label = label[index_to_cut:len(data)]

#We also need timestamps to view the data nicly in Yggio:s charts
timestamps_string = pd.bdate_range(end=(pd.Timestamp.now()-pd.Timedelta(days=1)), periods=len(data), freq="T")
timestamps = (timestamps_string - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
historic_timestamps = timestamps[0:index_to_cut]
future_timestamps = timestamps[index_to_cut:len(data)]


#######
# Send into yggio
#######
SECRETS = ['XXJFEOJFAO', 'RAEHIREOERJFDK']
NAMES = ['Daphnet Historic S03R01E0 4 ', 'Daphnet Future S03R01E0 4']

#Authorize and create session
response = requests.post(SERVER +'api/auth/local', json={"username": USERNAME,"password": PASSWORD})
authorization = response.json()
token = authorization['token']
my_headers = {'Authorization' : 'Bearer ' + token + ''}
my_session = requests.Session()
my_session.headers.update(my_headers)


#Create an iotnode to post data to
SECRET = SECRETS[0]
NAME = NAMES[0]

response = requests.post(
    SERVER + 'api/iotnodes',
    data = {'name': NAME,
            'secret': SECRET},
    headers=my_headers
)
jsonResponse = response.json()
print("Your new nodes id is:", jsonResponse['_id'])

#Post data to the node
#Go and do somehing more interesting- this takes time! 2000 timestamps takes roughtly 10 minutes
headers = {
  'Content-Type': 'application/json'
}

starttime = time.time()
for i in range(len(historic_data)):
    payload = json.dumps({
      "secret": SECRET,
      "timestamp": int(historic_timestamps[i]),
      "data": int(historic_data[i]),
      "label": int(historic_label[i])
    })
    response = requests.request("POST", SERVER + "http-push/generic", headers=headers, data=payload)
    if i % 100 == 0:
        print('On index', i, 'of total', len(historic_data), 'which has taken', (time.time()-starttime)/60, 'minutes')

   
#Redo for Future data
SECRET = SECRETS[1]
NAME = NAMES[1]

#New node
response = requests.post(
    SERVER + 'api/iotnodes',
    data = {'name': NAME,
            'secret': SECRET},
    headers=my_headers
)
jsonResponse = response.json()
print("Your new nodes id is:", jsonResponse['_id'])

#Post
headers = {
  'Content-Type': 'application/json'
}
starttime = time.time()
for i in range(len(future_data)):
    payload = json.dumps({
      "secret": SECRET,
      "timestamp": int(future_timestamps[i]),
      "data": int(future_data[i]),
      "label": int(future_label[i])
    })
    response = requests.request("POST", SERVER + "http-push/generic", headers=headers, data=payload)
    if i % 100 == 0:
        print('On index', i, 'of total', len(future_data), 'which has taken', (time.time()-starttime)/60, 'minutes')
