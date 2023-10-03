# -*- coding: utf-8 -*-
"""
Ui- need to be hand curated
"""

import streamlit as st
st.set_page_config(
    page_title="xAnomaly",
    page_icon= None,
    layout="wide"
)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import time
import os
from bokeh.layouts import column, gridplot
from bokeh.models import ColorBar, HoverTool, BoxEditTool, DatetimeTickFormatter, ColumnDataSource, Range1d, LinearAxis, RangeTool, Ray, CDSView, GroupFilter, LegendItem, Legend
from bokeh.plotting import figure, show, curdoc
from bokeh.palettes import Viridis6, Inferno11
from bokeh.palettes import RdYlGn
from bokeh.transform import linear_cmap
from datetime import datetime, timedelta

#App structure
#Selector: Which id
#Compare anomalies to "normal day"


#Hardcoded Paths- you need to change this to your path!

#For deployment
CURRENT_FOLDER = os.path.dirname(__file__)
PROJECT_FOLDER = CURRENT_FOLDER + '/..'
MODELPATH =  PROJECT_FOLDER + '/models'


#Data - you need to change this by hand and include the relevant pop-file from other folder
NUM_COL_NAMES = ['value_data']
NR_OF_SERIES = len(NUM_COL_NAMES)
pop = pd.read_csv(PROJECT_FOLDER+ '/pop.csv')
_ids = list(pop._id)
names = list(pop.name)


#Load data selected data

# streamlit run ui.py
st.title('xAnomaly')

selected_name = st.selectbox(
    'Select sensor to inspect:',
    tuple(names))
selected_id = _ids[names.index(selected_name)]
df = pd.read_csv(MODELPATH+ '/' + selected_id +'_df_for_visualization.csv')
df = df.astype({'time': 'datetime64[ns, Europe/Stockholm]'})

# #Make tables for comparisons
#First find the most normal day according to lowest error
df['date'] = df['time'].dt.date
means = df.groupby(['date']).mean()
if len(means[means['window_mae'] == min(means['window_mae'])]) == 1:
    date_min_mae = means[means['window_mae'] == min(means['window_mae'])].index[0]
else:
    date_min_mae = means[means['window_mae'] == min(means['window_mae'])].index[1]
date_min_mae_start = str(pd.to_datetime(date_min_mae, format='%Y/%m/%d'))
date_min_mae_stop = str(pd.to_datetime(date_min_mae, format='%Y/%m/%d')+pd.Timedelta(days=1))

cond = ((df['time'] > date_min_mae_start) & (df['time'] < date_min_mae_stop))
normal_day_1=df[cond]

#Now let the user decide a date to compare to
#Decide data range
first_date = df['date'].min()
last_date = df['date'].max()

st.subheader('What does the model do?')
st.write('Compares any date to the most normal day accoridng to the model (which was ' + date_min_mae_start[0:10]+ '). The datapoints for the normal day is colored gray. The datapoints for your selected day are colored according to how normal the model think they are, where green means normal and red means strange.' )
selected_date = st.date_input("Select date:", last_date, first_date, last_date)
selected_date_start = str(pd.to_datetime(selected_date, format='%Y/%m/%d'))
selected_date_stop = str(pd.to_datetime(selected_date, format='%Y/%m/%d')+pd.Timedelta(days=1))

cond = ((df['time'] > selected_date_start) & (df['time'] < selected_date_stop))
selected_day=df[cond]
normal_day_1.iloc[:,normal_day_1.columns.get_loc('time')] = normal_day_1['time'].apply(lambda x: x.replace(year=selected_date.year, month=selected_date.month, day= selected_date.day))
normal_day_1['normal'] = 'normal'
selected_day['normal'] = 'not'
selected_day = selected_day.append(normal_day_1)


#Make bokeh plot
selected_day=selected_day.fillna('')
source_normal = ColumnDataSource(selected_day)
cmap_normal = linear_cmap('window_mae', RdYlGn[8], 0, 1)
view_selected_day = CDSView(source=source_normal, filters=[GroupFilter(column_name='normal', group='not')])
view_normal_day = CDSView(source=source_normal, filters=[GroupFilter(column_name='normal', group='normal')])

p1_normal = figure(x_axis_type="datetime", 
            x_range=(pd.Timestamp(selected_date_start, tz='Europe/Stockholm'), pd.Timestamp(selected_date_stop, tz='Europe/Stockholm')),
            background_fill_color="#efefef",
            height=100,
            y_range=(df[NUM_COL_NAMES[0]].min(), df[NUM_COL_NAMES[0]].max()),
            y_axis_label=NUM_COL_NAMES[0]
            )
p1_normal.xaxis[0].formatter = DatetimeTickFormatter(days="%A %d/%m", hours="%H")
p1_normal.xaxis[0].ticker.desired_num_ticks = 7
p1_normal.circle('time', NUM_COL_NAMES[0], source=source_normal, view=view_normal_day, color= 'darkgray', size=10)
p1_normal.circle('time', NUM_COL_NAMES[0], source=source_normal, view=view_selected_day, color= cmap_normal, size=10)

# Colorbar
color_bar_normal = ColorBar(color_mapper=cmap_normal['transform'], height=10, orientation='horizontal')
p1_normal.add_layout(color_bar_normal, 'above')


st.bokeh_chart(column(p1_normal, sizing_mode="scale_width"), use_container_width=False)
