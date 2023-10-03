import streamlit as st
import yaml

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

@st.cache_resource
def get_sender_info(path):
    
    global EMAIL_ADDRESS
    global EMAIL_PASSWORD
    global EMAIL_SERVER
    global EMAIL_PORT
    
    #Load config and its parameters- you need to change this to your account!
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    EMAIL_ADDRESS = config["sender_email"]["email_address"]
    EMAIL_PASSWORD = config["sender_email"]["password"]
    EMAIL_SERVER = config["sender_email"]["smtp_server"]
    EMAIL_PORT = config["sender_email"]["smtp_port"]

    
    return EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SERVER, EMAIL_PORT


def send_training_done_email(path, receiver_mail_adress):
    # E-mail functionality
    
    # Create the email message
    msg = MIMEMultipart()
    msg['Subject'] = 'xAnomaly ready for inspection!'
    msg['From'] = EMAIL_ADDRESS
    msg.preamble = 'xAnomaly ready for inspection!'
    
    
    #Attach Image
    fp = open(path + '/assets/logo.jpg', 'rb') #Read image
    logoImage = MIMEImage(fp.read())
    fp.close()
    logoImage.add_header('Content-ID', '<image1>')
    logoImage.add_header('Content-Disposition', 'inline', filename='image1')
    msg.attach(logoImage)
    
    
    # Attach HTML body
    msg.attach(MIMEText(
        '''
        <html>
            <body style="background-color: #FFFFFF;">
                <img style="max-width:3em; max-height:3em; display: block; margin-left: auto; margin-right: auto;" src="cid:image1" alt="Yggio logo">
                <h2 style="text-align: center;">Hi!</h2>
                <h1 style="text-align: center;">Your anomaly detection models are ready, go and inspect what they do <a href="https://xanomaly.dev-playground.sensative.net/Inspect">here!</a></h1>
                <h2 style="text-align: center;">Best,</h2>
                <h2 style="text-align: center;">Data Science Team at Sensative</h2>
    
            </body>
        </html>'
        ''',
        'html', 'utf-8'
        )
    )
    
    
    # Send mail
    server = smtplib.SMTP_SSL(EMAIL_SERVER, EMAIL_PORT)
    server.ehlo()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.sendmail(EMAIL_ADDRESS, receiver_mail_adress, msg.as_string())
    server.quit()
