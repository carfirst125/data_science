##############################################################
# REVENUE PREDICTION MODEL
# File name: vtidss_ml_rev_pre_dnn_rl_0.06.3.py
# Method: DNN
# Date: 23/10/2019
# Author: Nhan Thanh Ngo
# Company: VTI-DSS
# Description: 
#    Predict revenue of sales data
# Status: SUCCESS
#         Accuracy: 90.5%
# Specification:
#    Previous version: 006.1
#    AUTO Check Loss --> Find WINDOW SIZE
#    AUTO Choose LR
#     

import sys, getopt

debug=0
filename = 'vtidss_ml_rev_pre_dnn_dev_0.06.3.py'

def main(argv):  
  try:
    opts, args = getopt.getopt(argv,"hd")
  except getopt.GetoptError:
    print("Run command: python {} -h\nfor more information".format(filename))
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print("Run training model: python {}\n\tOptions: -d\tdebug\n".format(filename))
	  
    elif opt == '-d':
      debug=1
      print("[INFO] DEBUG MODE: Active")
	

if __name__ == "__main__":
  main(sys.argv[1:])


#-----------------------------------------------------
#create DEBUG directory
import os
cwd = os.getcwd()
print (cwd)
debug_path = "./0_06_3_debug"

if os.path.exists(debug_path):
    print ("\'{}\' is already EXISTED!\n".format(debug_path))
else:    
    os.mkdir(debug_path)
    print ("\'{}\' is CREATED!\n".format(debug_path))
	
#-----------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
print(tf.__version__)

print ("From VTI: Hello WORLD")
###################################################
# QUERY DATA FROM BIGQUERY
#
#

print ('#############################')
print ('# QUERY DATA FROM BIGQUERY')
print ('#############################')


from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("../../vti_sandbox.json")
project_id = 'vti-sandbox'
client = bigquery.Client(credentials= credentials,project=project_id)

sql = """
    SELECT sales_date, sum(net_amount) as total_sales
    FROM `hlc.fact_all_sale`
    GROUP BY sales_date
    ORDER BY sales_date"""

# Run a Standard SQL query using the environment's default project
df = client.query(sql).to_dataframe()

# Run a Standard SQL query with the project set explicitly
df = client.query(sql, project=project_id).to_dataframe()

print ("[INFO] input data for prediction as below\n{}".format(df))
print ("[INFO] length of input data before process: {}".format(len(df)))

###################################################
# GENERAL FUNCTION
#
#

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def aeb(a,b):
    if abs(a-b)<1e-10:
        return True
    else:
        return False
		
###################################################
# CREATE TRAIN AND VALID DATA
#
#

#IMPORTANT: Need to remove all NaN from dataframe
df.dropna(inplace=True)
print ("[INFO] length of input data after drop NaN: {}".format(len(df)))

#change data to array
series_data = np.asarray(df['total_sales'])

split_time = 500

time = np.arange(len(df), dtype="float32")
time1=time
series_data1=series_data
time_train = time[:split_time]
x_train = series_data[:split_time]
time_valid = time[split_time:]
x_valid = series_data[split_time:]


# HYPER-PARAMETERS
window_size = 30
batch_size = 32
shuffle_buffer_size = 500

chart = plt.figure(figsize=(20,6),dpi=200,linewidth=0.1)
input_chart = plt.subplot(1,1,1)
input_chart.plot(time,series_data,marker=".")	
chart.savefig(debug_path+"/006-3_DNN_inputdata.png")

#plot_series(time,series_data)
#plt.savefig('./DNN_inputdata.png')

###################################################
# WINDOW DATA DEFINE
# AS INPUT DATASET FOR MODEL
#
#

def windowed_dataset(series_data, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series_data)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

###################################################
# STRUCTURE MODEL AND FIT MODEL
#
#

print ('########################################')
print ('# FIND BEST LEARNING RATE FOR ML MODEL')
print ('########################################')


print("HYPER PARAMETERS FOR RUNNING:\nwindow_size={}\nbatch_size={}\nshuffle_buffer_size={}".format(window_size, batch_size, shuffle_buffer_size))
#print (x_train[:2])
# MODEL DEFINE, COMPILE AND BUILD
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

#print(type(dataset))

'''
for window in dataset:
    for val in window:       
      print(val.numpy(), end = " ")
    print()
'''

#Layer 1: 10 neurons, act: Relu
#Layer 2: 10 neurons, act: Relu
#Layer 3:  1 neurons

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"),          
    tf.keras.layers.Dense(1)
])

# set schedule for Learning rate for each Epoch run time
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-20 * 10**(epoch / 20))
#early_stop = EarlyStopping(monitor='loss', patience=2)

optimizer = tf.keras.optimizers.SGD(lr=1e-20, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)#,metrics=['accuracy'])

num_epochs = 200
#save model name is "history" with learning rate feeded by lr_schedule
history = model.fit(dataset, epochs=num_epochs, callbacks=[lr_schedule], verbose=0)

#model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
#model.fit(dataset,epochs=10,verbose=0)
print ("[INFO] Complete training model with different learning rates...\n")

###################################################
# GET OPTIMIZED LEARNING RATE
#
#
print ('#################################')
print ('# PLOT LEARNING RATES and LOSS')
print ('#################################')

#lrs = 1e-20 * (10 ** (np.arange(num_epochs) / 20))
#print (history.history.keys())

#print (len(history.history["lr"]))
#print (len(history.history["loss"]))

#print(history.history["lr"],'\n',history.history["loss"])
#print(min(history.history["loss"]))

#-----------------------------
lrs = 1e-20 * (10 ** (np.arange(num_epochs) / 20))
epoch_loss = history.history["loss"]

df_loss_lrs = pd.DataFrame()
df_loss_lrs['lrs'] = np.array(lrs)
df_loss_lrs['epoch_loss'] = np.array(epoch_loss)

print("[DEBUG] Epoch Loss of Different LRs:{} ".format(df_loss_lrs))
print("[INFO] min Epoch Loss of different Lrs: {}".format(df_loss_lrs['epoch_loss'].min()))

min_loss = df_loss_lrs['epoch_loss'].min()


min_loss_detect = []
for i in range(len(df_loss_lrs)):
  min_loss_detect.append(aeb(df_loss_lrs['epoch_loss'][i],min_loss))

#print("[DEBUG] Min of Loss by LRs TRUE FALSE: {}".format(min_loss_detect))

#BEST WINDOW SIZE HERE
lr_best = np.array(df_loss_lrs[np.array(min_loss_detect)]['lrs']).mean()

print ("[INFO] Best LR: {}".format(lr_best))

#-----------------------------
# semilogx will plot with x axis is format of 10^x
# we can using lrs or history.history['lr'] is OKAY. They are the same.
plt.semilogx(lrs, history.history["loss"])
#plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-20, 1e-1, 0, 1e+14])
plt.savefig(debug_path+"/006-3_DNN_loss_by_lrs.png")

print ("[INFO] Loss of model by learning rates, refer at ./DNN_loss_by_lrs.png")

###################################################
# TRAIN MODEL WITH BEST LRS = 1e-17
# LOOP WITH DIFF WINDOW SIZE TO FIND BEST WINDOW SIZE
#

print ('########################################################')
print ('# USE BEST LR and LOOP TO FIND BEST WINDOW SIZE')
print ('########################################################')

#---------------------------------------------------------------
def check_window_size (window_size):

#window_size = 30
  dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.SGD(lr=lr_best, momentum=0.9)
  model.compile(loss="mse", optimizer=optimizer)
  history = model.fit(dataset, epochs=200, verbose=0)

  print ("[INFO] WINDOW_SIZE={} \n\tTraining model completed...".format(window_size))
  
  #ml_loss = history.history['loss']
  
  forecast = []
  for time in range(len(series_data) - window_size):
    forecast.append(model.predict(series_data[time:time + window_size][np.newaxis]))

  forecast = forecast[split_time-window_size:]
  results = np.array(forecast)[:, 0, 0]

  print ("\tForecast completed...")

  final_mae = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
  print ("\tFinal MAE = {}".format(final_mae))
  
  return final_mae
#---------------------------------------------------------------

mae_ws = []
wd_sz = []
for wd_size in range(15,35,3):
  wd_sz.append(wd_size)
  mae_ws.append(check_window_size(wd_size))

print ("window size = {}, MAE = {}".format(wd_sz,mae_ws))

df_mae_ws = pd.DataFrame()
df_mae_ws['mae_ws'] = np.array(mae_ws)
df_mae_ws['wd_sz'] = np.array(wd_sz)

print("[DEBUG] MAE of Different Window Size:{} ".format(df_mae_ws))
print("[INFO] min MAE of different Window Size: {}".format(df_mae_ws['mae_ws'].min()))

min_mae = df_mae_ws['mae_ws'].min()


min_detect = []
for i in range(len(df_mae_ws)):
  min_detect.append(aeb(df_mae_ws['mae_ws'][i],min_mae))

print("[DEBUG] Min of MAE by Window: ".format(min_detect))

#BEST WINDOW SIZE HERE
window_size = np.array(df_mae_ws[np.array(min_detect)]['wd_sz'])[0]

print ("[INFO] Best window size: {}".format(window_size))

chart = plt.figure(figsize=(20,6),dpi=200,linewidth=0.1)
mae_ws_chart = plt.subplot(1,1,1)
mae_ws_chart.plot(wd_sz,mae_ws,marker=".")	
chart.savefig(debug_path+"/006-3_DNN_mae_by_window_size.png")

print ("[INFO] Finding best Window Size completed!")
print ("[INFO] WINDOW SIZE for Lowest Loss is: {}".format(window_size))
print ("[INFO] Refer chart Loss by Window Size at ./006-3_DNN_mae_by_window_size.png")

###################################################
# TRAIN MODEL WITH BEST LRS and WINDOW SIZE
#
#
print ('#########################################################')
print ('# BUILD FINAL MODEL AGAIN WITH BEST LR and WINDOW SIZE')
print ('#########################################################')

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(lr=lr_best, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=200, verbose=0)

print ('############################')
print ('# PLOT LOSS')
print ('############################')
#print (len(loss)) #equal number of Epoch

# list all data in history
print(history.history.keys())

loss = history.history['loss']
epochs = range(len(loss))
#plt.plot(epochs, loss, 'b', label='Training Loss')
#plt.plot(epochs, loss, 'b', label='Training Loss')
#plt.savefig('./DNN_loss_of_best_lrs.png')

chart = plt.figure(figsize=(20,6),dpi=200,linewidth=0.1)
loss_chart = plt.subplot(1,1,1)
loss_chart.plot(epochs,loss,marker=".")	
chart.savefig(debug_path+"/006-3_DNN_loss_of_best_lrs.png")

# Plot all but the first 10
loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
#print(plot_loss)
#plt.plot(epochs, plot_loss, 'b', label='Training Loss')

chart = plt.figure(figsize=(20,6),dpi=200,linewidth=0.1)
loss_chart = plt.subplot(1,1,1)
loss_chart.plot(epochs,plot_loss,marker=".")	
chart.savefig(debug_path+"/006-3_DNN_loss_of_best_lrs_from_epoch10.png")

#plt.savefig('./DNN_loss_of_best_lrs_from_epoch10.png')

print ("[INFO] Plot completed...\nrefer at ./DNN_loss_of_best_lrs.png and ./DNN_loss_of_best_lrs_from_epoch10.png")

###################################################
# FORECAST RESULT
#
#

print ('##################################')
print ('# RUN FORECASTING FOR VALID DATA')
print ('##################################')

forecast = []
for time in range(len(series_data) - window_size):
  forecast.append(model.predict(series_data[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


#plt.figure(figsize=(10, 6))

chart = plt.figure(figsize=(20,6),dpi=200,linewidth=0.1)
predict_chart = plt.subplot(1,1,1)
predict_chart.plot(time_valid,x_valid,marker=".")	
predict_chart.plot(time_valid,results,marker=".")
chart.savefig(debug_path+"/006-3_DNN_Predict.png")

#plot_series(time_valid, x_valid)
#plot_series(time_valid, results)
#plt.savefig('./DNN_Predict.png')
print ("[INFO] Forecast completed...")

final_mae = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
print ("Final MAE = {}".format(final_mae))

percent_mae = final_mae/df['total_sales'].mean()
print ("Percentage of Mean Absolute Error = {0:.2%}".format(percent_mae))

print ("#************************************")
print ("# COMPLETE BUILD AND CHECK MODEL")
print ("#************************************")

###################################################
# CREATE FOLDER FOR STORING MODEL
#
#

print ("#************************************")
print ("# NOW STORE MODEL")
print ("#************************************")

#Run look-alike Bash command

cwd = os.getcwd()
print (cwd)
model_path = "./0_006_3_model"
#os.listdir("model_006_2")

#os.mkdir("/content/drive/My Drive/VTI/ML_model")
if os.path.exists(model_path):
    print ("\'{}\' is already EXISTED!\n".format(model_path))
else:    
    os.mkdir(model_path)
    print ("\'{}\' is CREATED!\n".format(model_path))
	
###################################################
# SAVE MODEL AT PARTICULAR PATH
#
#

tf.keras.models.save_model(
    model,
    model_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

###################################################
# LOAD MODEL FROM PARTICULAR PATH
#
#

print ('##################################')
print ('# LOAD SAVED MODEL AND CHECK')
print ('##################################')

ld_model=tf.keras.models.load_model(
    model_path,
    custom_objects=None,
    compile=True
)

###################################################
# TEST USING LOADING MODEL
#
#

split_time1 = 50
time_valid1 = time1[split_time1:]
x_valid1 = series_data1[split_time1:]

forecast1 = []
for time in range(len(series_data) - window_size):
  forecast1.append(ld_model.predict(series_data1[time:time + window_size][np.newaxis]))

forecast1 = forecast1[split_time1-window_size:]
results1 = np.array(forecast1)[:, 0, 0]

print ("[INFO] LOAD and CHECK MODEL completed...")
#plt.figure(figsize=(10, 6))
print ('###########################################')
print ('# FINAL REPORT - BETA RELEASE VERSION 1.0')
print ('###########################################')

chart1 = plt.figure(figsize=(20,6),dpi=200,linewidth=0.1)
predict_chart1 = plt.subplot(1,1,1)
predict_chart1.plot(time_valid1,x_valid1,marker=".")	
predict_chart1.plot(time_valid1,results1,marker=".")
chart1.savefig(debug_path+"/006-3_DNN_Predict_checkloadmodel.png")

print ("INFORMATION:")
print ("PREDICT MODEL: Sequential DNN")
print ("LIBRARY: Keras, Tensorflow")
print("HYPER PARAMETERS FOR RUNNING:\nwindow_size={}\nbatch_size={}\nshuffle_buffer_size={}\nlearing_rate={}".format(window_size, batch_size, shuffle_buffer_size,lr_best))

print ("\nRESULT:")
print ("Mean Absolute Error (MAE) = {}".format(final_mae))
print ("Percentage of MAE = {0:.2%}\n".format(percent_mae))
print ("Graph for Predict Result: refer at ./DNN_Predict.png")
print ("SAVED MODEL at {}".format(model_path))

print ('##################################')
print ('# ENDING ! CONGRATULATION')
print ('##################################')