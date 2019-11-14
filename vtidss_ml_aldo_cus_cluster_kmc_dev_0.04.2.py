##############################################################
# CLUSTER CUSTOMER by PRODUCT
# File name: vtidss_ml_aldo_cus_cluster_kmc_dev_0.04.2.py
# Previous: 0.04.1
# Method: K-Mean Clustering
# Date: 14/11/2019
# Author: Nhan Thanh Ngo
# Company: VTI-DSS
# Description: 
#    Cluster based on ALDO data based on products
# Status: COMPLETED
# Specification:
#    Data: Real data, input data in sandbox must follow the format for running
#    Getoption: k cluster (default k=4)
#    Mode: Full Rfm group (Group 1, 2, 3, 4)
#    Output: all in one csv file    
#    Complete Cluster by K cluster + Special Cluster of Refund customer
#    
#    Three type of Labels: Dom, 2-Dom, Mix
#    Auto choosing group and run.
#    Add Getoption
# Command: python vtidss_ml_aldo_cus_cluster_kmc_dev_0.04.2.py -h
#


import re
# <-- CHANGE THIS FOR NEW VERSION -->

version = '0_04_2' 
versionx=re.sub("_",".",version)
filename = "vtidss_ml_aldo_cus_cluster_kmc_dev_"+versionx+".py"


####################################################
# GET OPTIONS
import sys, getopt


#default
#group = 'Group 4'
kcluster = 4 
#groupx = ''
debug = 0 
outdir = "."
input = "Aldo.rfm_segment_by_behavior"
pivot_off = 0
cus_outdir=0

from optparse import OptionParser

usage = "usage: %prog [options] arg1 arg2\n\nExample: python %prog -k 10 -i Aldo.table -o ./folder_path --nopivot"
parser = OptionParser(usage=usage)
#parser.add_option("-h", "--help",
#                  action="store_true", dest="verbose", default=True,
#                  help="print help information of the script such as how to run and arguments")
parser.add_option("-i", "--input",
                  default="`Aldo.rfm_segment_by_behavior`",
                  metavar="SANDBOX", help="Sandbox dataname"
                                         "[default: %default]")						
parser.add_option("-k", "--kcluster",
                  action="store_false", dest="verbose",
                  help="number cluster that you want to generate by K-Mean Method [Default: k=4]")
				  
parser.add_option("-o", "--outdir",
                  default="./"+version+"_ouput",
                  metavar="OUTDIR", help="write output to OUTDIR"
                                         "[default: %default]")				 
parser.add_option("-d", "--debug",
                  default="OFF",
                  help="Debug mode "
                       "[default: %default]")
parser.add_option("-n", "--nopivot",
                  default="pivot ON",
                  help="pivot OFF"
                       "[default: %default]")

try:
  opts, args = getopt.getopt(sys.argv[1:], 'hk:o:d:i:n', ['help','kcluster=','outdir=', 'debug','input=','nopivot'])
    
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-k', '--kcluster'):
    kcluster = int(arg)
  if opt in ('-d', '--debug'):
    debug = 1
  if opt in ('-o', '--outdir'):
    outdir = str(arg) 
    cus_outdir = 1
  if opt in ('-i', '--input'):
    input = str(arg)
  if opt in ('-n', '--nopivot'):
    pivot_off = 1
  if opt in ('-h', '--help'):
    parser.print_help()
    #print ("Run command: python {} -k kcluster -g rfm_Group -d debug\n ".format(filename))
    sys.exit(2)
	  
if kcluster == None:
  sys.exit(3)


print ("From VTI: Hello WORLD")
print ("This is {}\n".format(filename))

print("##################################")
print("RUN INFORMATION")
print("K cluster: {}".format(kcluster))  
print("DEBUG MODE: {}".format(debug))
print("OUT DIR: {}/".format(outdir))
print("Input: {}".format(input))
print("Pivot OFF: {}".format(pivot_off))
print("##################################")

#system.exit()
#------------------------------------------------------
# Rule of label
# Gt than 55 and 1st/2nd >2.0 -->

#-----------------------------------------------------
#create DEBUG directory
import os
import glob
import shutil

cwd = os.getcwd()
print ("at {}\nBEGIN...".format(cwd))


if cus_outdir:
  if not os.path.exists(outdir):    
    os.mkdir(outdir)
	
# create debug folder	
debug_path = outdir+"/"+version+"_debug"

if os.path.exists(debug_path):
  print ("\'{}\' is already EXISTED! --> REMOVE OLD DIR...".format(debug_path))
  #shutil.rmtree(debug_path)
else:
  os.mkdir(debug_path)
  print ("\'{}\' is CREATED!".format(debug_path))

# create output folder
output_path = outdir+"/"+version+"_output"

if os.path.exists(output_path):
  print ("\'{}\' is already EXISTED! --> REMOVE OLD DIR...".format(output_path))
  #shutil.rmtree(output_path)
else:
  os.mkdir(output_path)
  print ("\'{}\' is CREATED!".format(output_path))
	
#-----------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
print(tf.__version__)

#import sklearn
# import k-means from clustering stage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

#!conda install python-graphviz --yes
import graphviz
from sklearn.tree import export_graphviz
import itertools


#df_group_cluster_info = pd.DataFrame()
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

sql = "SELECT * FROM "+input

# Run a Standard SQL query with the project set explicitly
df_bq = client.query(sql, project=project_id).to_dataframe()

#short data for test
#df1=df_bq[df_bq['rfm_group']=='Group 1'][:1000]
#df2=df_bq[df_bq['rfm_group']=='Group 2'][:1000]
#df3=df_bq[df_bq['rfm_group']=='Group 3'][:1000]
#df4=df_bq[df_bq['rfm_group']=='Group 4'][:1000]

#group_df = pd.concat([df1,df2,df3,df4])

#df_bq = group_df
print ("[INFO] input data for prediction as below\n{}".format(df_bq[:5]))
print ("[INFO] length of input data before process: {}".format(len(df_bq)))

###################################################
# PIVOT TABLE PROCESS
#
#
if pivot_off==0:

  df_pivot = pd.pivot_table(df_bq, values='quantity_item', index=['rfm_group', 'cus_id'], columns=['style'], aggfunc=np.sum, fill_value=0)
  df_pivot.reset_index(inplace=True)

  print (df_pivot.index.values)
  print (df_pivot.columns.values)
  print (len(df_pivot))

  while len(df_bq['cus_id'].unique()) == len(df_pivot):
    try:
      print ("[PIVOT PASS] Length of original query data in unique equals length of pivot data")
      break
    except ValueError:
      print ("ERROR: Length of df_bq.unique does not equal length of df_pivot. Please check...")

else:
  df_pivot = df_bq.copy()

###################################################
# GENERAL FUNCTION
#
#

#++++++++++++++++++++++++++++
#FUNCTION : PLOT BARH CHART
#
def plot_pie (df, xlabel, ylabel, title, color='grey', fontsize=12, file_name='./'): # df_name is df with 2 columns, 1st cols is name, 2nd col is number

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)

  print ("Total items customer purchased: {}".format(df['count'].sum()))
  df_plot = df[df['count']!=0][-4:] 
  df_plot.loc['others'] = df.iloc[:-4,0].sum()

  print (df_plot)
  label_list = np.array(df_plot.index.values)
  colors_array = cm.rainbow(np.linspace(0, 0.8, len(df_plot.index.values)))
  rainbow = [colors.rgb2hex(i) for i in colors_array]

  #print(label_list)
  #print(colors_array)
  #print(rainbow)


  ax=df_plot['count'].plot(kind='pie',
          figsize=(9, 6),
          autopct='%1.1f%%',
          startangle=90,
          shadow=True,
          labels=label_list,         # turn off labels on pie chart
          pctdistance=.5,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
          colors=rainbow  # add custom colors
          #explode=explode_list # 'explode' lowest 3 continents
          )

  ax.set_ylabel(ylabel,fontsize=fontsize)
  ax.set_xlabel(xlabel,fontsize=fontsize)
  ax.set_title(title,fontsize=fontsize)
  
  #save chart
  #chart_name = debug_path+"/"+str(kcluster)+"_"+groupx+"_"+title+"_"+ylabel+"_"+xlabel+".png"
  
  chart.savefig(file_name)
  #chart.close()
  
#++++++++++++++++++++++++++++
#FUNCTION : PLOT BARH CHART
#

def plot_barh(df, xlabel, ylabel, title, color='grey', fontsize=12,file_name='./'):

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)
  
  ax=df.plot(kind='barh', 
             figsize=(12,6),
             fontsize=fontsize,
             color=color,
             rot=0
            )

  ax.set_ylabel(ylabel,fontsize=fontsize)
  ax.set_xlabel(xlabel,fontsize=fontsize)
  ax.set_title(title,fontsize=fontsize)

  idx_adj=-0.3
  for col in df.columns.values:
    #print (col)
    #plt.legend(col,loc=4)

    idx_adj+=0.3 #[-.4,-.1,.2]
    for index, value in enumerate(df[col]): 
        #print (index,value)
        #label = format(int(value), ',') # format int with commas (dau phay hang ngan)
        #print (label)
        #place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
        plt.annotate(value, xy=(value+0.3, index+idx_adj), color='black', fontsize=fontsize)   
  
  #save chart
  chart_name = debug_path+"/"+str(kcluster)+"_"+groupx+"_"+title+"_"+ylabel+"_"+xlabel+".png"
  chart.savefig(chart_name)
  chart.close()



#--------------------------------------
# Function: sort_values of dataframe, and get the top values of them
# return the name of top venues
def return_most_favourite(row, num_top_favourite):
  row_categories = row.iloc[1:]    
  row_categories_sorted = row_categories.sort_values(ascending=False)
    
  return [row_categories_sorted.index.values[0:num_top_favourite], row_categories_sorted.values[0:num_top_favourite]]



def create_group_label_product_quan_table(df,group,cluster):
  #IMPORTANT NOTE: df in order from low to high
  
  cluster_type = ''
  cluster_labeling = ''
  total_percentage = 0
  
  df_ret = df.copy()
  total = df_ret['count'].sum(axis=0)
  print ("total = ",total)
  df_ret['percentage']=df_ret['count']/total
  
  df_ret['rfm_group']= [group]*len(df_ret)
  df_ret['Cluster Labels']= [cluster] * len(df_ret)
  df_ret.reset_index(level=0, inplace=True)
  df_ret.rename(columns={'index':'category'},inplace=True)
  
  #-----labeling------
  #DOM 1
  #rule for label: dominant: 1st > 0.55 and gt 2nd more than 2 
  #dom = 1
  if (df_ret['percentage'][len(df_ret)-1]>0.55) & (df_ret['percentage'][len(df_ret)-1]/df_ret['percentage'][len(df_ret)-2]>2):
    cluster_type = 'dominance'
    cluster_labeling = cluster_type[:3]+'.'+df_ret['category'][len(df_ret)-1]
	
  #DOM 2
  elif ((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2])>0.55) & ((df_ret['percentage'][len(df_ret)-1]+df_ret['percentage'][len(df_ret)-2])/df_ret['percentage'][len(df_ret)-3]>2):
    cluster_type = '2-dominances'
    cluster_labeling = cluster_type[:5]+'.'+df_ret['category'][len(df_ret)-1]+'.'+df_ret['category'][len(df_ret)-2]
    #dom = 2
  else:
    cluster_type = 'mixture'
    cluster_labeling=cluster_type[:3]
    
    for i in range(len(df_ret)):
      #print(df_ret['percentage'][len(df_ret)-i-1])

      total_percentage+=df_ret['percentage'][len(df_ret)-i-1]
      cluster_labeling+='.'+df_ret['category'][len(df_ret)-i-1]
      print(total_percentage)
      print(cluster_labeling)
      if total_percentage>=0.75:
        break
	
    
  #for i in range(len(df_ret)):
    #print(df_ret['percentage'][len(df_ret)-i])
    #break
  
  print ("\n[INFO] [{}, cluster {}] Type: {}, Labeling: {}".format(group,cluster,cluster_type,cluster_labeling))
  print ("[INFO] Labeling completed...\n")
  
  df_ret['cluster_type']= [cluster_type] * len(df_ret)
  df_ret['cluster_labeling']= [cluster_labeling] * len(df_ret)
  #----end labeling-------
  #system.exit(4)
  print(df_ret)
  return df_ret  


#++++++++++++++++++++++++++++
# Group cluster 
def df_group_cluster(df,group,kcluster):

  #this DataFrame using for create summary table for each cluster by product quantity
  df_group_cluster_info = pd.DataFrame()
  
  ###################################################
  # DATAFRAME INFO AND PREPROCESSING
  #
  #  
  
  groupx=re.sub(" ","_",group)
  
  df.reset_index(inplace=True)
  df.drop(columns='index',inplace=True)
  
  print("***************************************************")
  print("[INFO] Information of Customer {} will be clustered:".format(group))
  print("Shape: {}".format(df.shape))
  print("Columns: {}".format(df.columns.values))
  print("df=\n{}".format(df[:10]))
  print("***************************************************")

  # check if there are any row with all zero value
  col = df.columns.values
  df_data = df[col[2:]]
  rs = df_data[df_data.eq(0).all(1)]

  if len(rs) == 0:
    print ("[INFO] Number line of all features get value of zero is {}".format(len(rs)))
    df_rest = df.copy()
    df_drop = pd.DataFrame(columns=df.columns.values)
    print ("SUCCESSFUL!!!")
  else:
    print ("[INFO] Number line of all features get value of zero is {}".format(len(rs)))
    print ("[INFO]PROCESS to DROP All Zero value line")
    
    print("***len df: {}\nlen df_data {}\n".format(len(df),len(df_data)))
    df_data['sum']=df_data.sum(axis=1)
    drop_id = df[df_data['sum']==0].index.values
    
    print ("[INFO] drop_id list is: \n{}".format(drop_id))
	#df_drop = df[df.index==drop_id]
    print ("[INFO]df.index[drop_id] = \n{}".format(df.index[drop_id]))
    df_drop = df.iloc[drop_id,:]
	
    print ("[INFO] Length of drop_id: \n{}".format(len(drop_id)))
    print ("[INFO] Length of df_drop: \n{}".format(len(df_drop)))
    print ("[INFO] df_drop.head(): \n{}".format(df_drop[:5]))
    df_drop['sum']=df_data[1:].sum(axis=1)
    print ("df_drop sum \n{}".format(df_drop['sum']))
    print ("NICE...\n\n\n\n\n")
	
    df_rest = df.drop(drop_id)
    #df_rest = df.iloc[lambda x: x not in drop_id,:]
    print ("[INFO] Length of df_rest: \n{}".format(len(df_rest)))
    print ("[INFO] df_rest: \n{}".format(df_rest[:5]))
    #print("***len df: {}\nlen df_data {}\n".format(len(df_rest),len(df_data)))
	
    #df.drop(df.index[drop_id], inplace=True)
    #df.drop(drop_id, inplace=True)
    df_rest.reset_index(inplace=True)
    df_rest.drop(['index'],axis=1, inplace=True)
	
  print ("[INFO] Total row number of dataframe {}".format(len(df)))


  ###################################################
  # FIGURE OUT MOST FAVOURITE ITEMS OF EACH CUSTOMER
  #    Arange by number of items folowing order.
  #    Choose the most favourite item to DataFrame
  # OUTPUT: df_cus_favourite
  #
  
  df_x = df_rest.copy()
  
  df_x.drop(['rfm_group'],axis=1, inplace=True)
  
  print("AAA:\n",df_x.columns.values)
  print(df_x[:5])
  #--------------------------------------
  # Call Function
  # Arrange customer product purchase by most Favourite
  num_top_favourite = 7

  indicators = ['st', 'nd', 'rd']

  # create columns according to number of top favourite
  columns = ['cus_id']

  for ind in np.arange(num_top_favourite):
    try:
      columns.append('{}{} Favourite'.format(ind+1, indicators[ind]))
    except:
      columns.append('{}th Favourite'.format(ind+1))

  # create a new dataframe
  df_cus_favourite = pd.DataFrame(columns=columns)
  df_cus_favourite['cus_id'] = df_x['cus_id']

  df_cus_favourite_quan = pd.DataFrame(columns=columns)
  df_cus_favourite_quan['cus_id'] = df_x['cus_id']
  #print(type(df_x.iloc[1, :]))
  #exit()

  print ("[INFO] Clustering for {} is processing...\nit could take few minutes to complete......".format(group))
  for ind in np.arange(df_x.shape[0]):
    #print(df_x.iloc[ind, :])
    df_cus_favourite.iloc[ind, 1:] = return_most_favourite(df_x.iloc[ind, :], num_top_favourite)[0]
    df_cus_favourite_quan.iloc[ind, 1:] = return_most_favourite(df_x.iloc[ind, :], num_top_favourite)[1] # return_most_favourite_quan(df_x.iloc[ind, :], num_top_favourite)
    #exit()

  list_fav_colname = df_cus_favourite.columns.values[1:]
  print ("[INFO] list_fav_colname = \n{}".format(list_fav_colname))
  print ("[INFO] df_cus_favourite_quan = \n{}".format(df_cus_favourite_quan[:5]))
  print ("[INFO] df_cus_favourite = \n{}".format(df_cus_favourite[:5]))

  ###################################################
  # DATA PROCESSING: SCALING 
  #

  # list_features: all input feature for cluster
  list_features = df_rest.columns.values[2:]

  df_features = df_rest[list_features]

  df_features_scale = pd.DataFrame(columns=df_features.columns.values)
  for i, product in zip(range(len(list_features)),list_features):
    print (i,product)
    df_features_scale[product]=df_features[product]/df_features.sum(axis=1)

  print ("Length of df_feature_scale {}".format(len(df_features_scale)))
  
  
  '''
  ###################################################
  # RUN DIFFERENT K TO FIND SUITABLE K value
  # Elbow and Silhouette method is applied
  #

  # CHOOSING K: Run Clustering Evaluation, consider SSE and Silhouette score.
  sse = {} #sse: sum of squared error
  silhouette_kclus = []

  for k in range(2, 25):
    # Kmeans Model
    kmeans = KMeans(n_clusters = k, max_iter=1000).fit(df_features_scale)
    
    # add cluster result following k to top venues df
    #cluster_x="cluster_"+str(k)    
    #df_hcmc_top_venues[cluster_x] = kmeans.labels_   
    
    #df_clustering["clusters_"+k] = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    print("For n_clusters={}: SSE is {}".format(k, sse[k]))
    
    if (k>1):
      # For Silhouette score
      label = kmeans.labels_
      sil_coeff = silhouette_score(df_features_scale, label, metric='euclidean')  
      silhouette_kclus.append(sil_coeff)
      print("For n_clusters={}: The Silhouette Coefficient is {}".format(k, sil_coeff))

   #df_hcmc_top_venues.head()

  #--------------------------------------
  #plot for SSE 
  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)
  plt.plot(list(sse.keys()), list(sse.values()), marker='^')
  plt.plot(4,list(sse.values())[2], marker='o', color='w', markersize=12, markeredgewidth=4, markeredgecolor='r')
  plt.xlabel("Number of cluster (K)")
  plt.ylabel("Sum of Squared Errors (SSE)")
  plt.title("K-Means Cluster Evaluation by Elbow Method")
  chart.savefig(debug_path+"/Elbow_KMean_SSE_by_k.png")
  # by the Elbow method as in below figure, the value of 3 is chosen

  #--------------------------------------
  #plot for Silhouette score
  index_of_max = silhouette_kclus.index(max(silhouette_kclus[1:]))

  chart = plt.figure(figsize=(20,10),dpi=200,linewidth=0.1)
  plt.plot(range(2,len(silhouette_kclus)+2),silhouette_kclus, marker='^', color='b')
  plt.plot(index_of_max+2,silhouette_kclus[index_of_max], marker='o', color='w', markersize=12, markeredgewidth=4, markeredgecolor='r')

  plt.xlabel("Number of cluster (K)")
  plt.ylabel("Silhouette Score")
  plt.title("K-Means Cluster Evaluation by Silhouette Analysis")
  chart.savefig(debug_path+"/Silhoutte_by_k.png")
  
  system.exit(1)
  
  '''
  ###################################################
  # THROUGH Elbow method and Silhouette value, choose suitable k value
  # Here chooses k=4
  # 

  # run Kmean again with k = 4
  # set number of clusters
  kclusters = kcluster

  # run k-means clustering
  kmeans = KMeans(n_clusters = kclusters, max_iter=1000, random_state=0).fit(df_features_scale)

  # check cluster labels generated for each row in the dataframe
  #kmeans.labels_[0:10]

  #-----------------------------------
  # add clustering labels
  df_cus_favourite2 = df_cus_favourite.copy()
  df_cus_favourite2["Cluster Labels"] = kmeans.labels_


  ###################################################
  # JOIN CLUSTER LABEL TO INPUT TABLE 
  # 

  # join
  df_merge2 = pd.merge(df_cus_favourite2,df_x, how='left', on=['cus_id'])
  
  #return df_merge2
  # export csv file
  outfile = output_path+"/"+version+"_aldo_cus_cluster_by_product_"+str(groupx)+"_"+str(kclusters)+".csv"
  print ("\n[DONE] Result of Cluster is at {}".format(outfile))

  print ('\n##################################')
  print ('# KMEAN CLUSTER REPORT')
  print ('##################################')

  #print lenght of each cluster
  for i in range(kclusters):
    print ("[INFO] length of cluster {} is {}".format(i,len(df_cus_favourite2[df_cus_favourite2['Cluster Labels']==i])))

  #summarize the quantity of product sold by each cluster and pie chart plot
  for i in range(kclusters):
    df_cluster_counti = pd.DataFrame(columns=['count'],data=df_merge2[df_merge2['Cluster Labels']==i][list_features].sum(axis=0).sort_values(ascending=True))
    
    print(df_cluster_counti[:5])
    df_temp = create_group_label_product_quan_table(df_cluster_counti,group,i)
    #print(df_temp)
    #print(df_temp.columns.values)
    #print("*********************************\n\n\n")
	
    df_group_cluster_info = pd.concat([df_group_cluster_info,df_temp])
    #df_group_cluster_info = df_group_cluster_infox.copy()
    #plot_barh(df_cluster_counti,'Quantity','Category',"Most Favourite Categories of Cluster"+str(i))
    
    xlabel = 'Categories'
    ylabel = "Cluster "+str(i)
    title = 'Percentage most purchase category'
    fname = debug_path+"/"+str(kcluster)+"_"+groupx+"_"+title+"_"+ylabel+"_"+xlabel+".png"
    print("[INFO] file name is {}".format(fname))
	
    plot_pie(df_cluster_counti,'Categories',"Cluster "+str(i),'Percentage most purchase category',file_name=fname)
    
  #df_group_cluster = df_group_cluster_info.copy()
  df_group_cluster_label = df_group_cluster_info[['rfm_group','Cluster Labels','cluster_type','cluster_labeling']]
  df_group_cluster_label.drop_duplicates(subset=['rfm_group','Cluster Labels','cluster_type','cluster_labeling'], keep='first',inplace=True)
  print (df_group_cluster_label)
	
   
  #print("fdasfdsafdsfsda\n\n\n")
  # add two labeling columns to df_merge2
  df_merge2['cluster_type']=np.zeros(len(df_merge2))
  df_merge2['cluster_labeling']=np.zeros(len(df_merge2))

  for j in range(len(df_merge2)):
    df_merge2.loc[j,'cluster_type']=df_group_cluster_label[df_group_cluster_label['Cluster Labels']==df_merge2['Cluster Labels'][j]]['cluster_type'][0]
    df_merge2.loc[j,'cluster_labeling']=df_group_cluster_label[df_group_cluster_label['Cluster Labels']==df_merge2['Cluster Labels'][j]]['cluster_labeling'][0]
  #	df_ret['cluster_type']= [cluster_type] * len(df_ret)
  #df_ret['cluster_labeling']= [cluster_labeling] * len(df_ret)
  #df_group_cluster_info[]
  
  #return df_merge2
  # export csv file
  outfile = output_path+"/"+version+"_aldo_cus_cluster_by_product_"+str(groupx)+"_"+str(kclusters)+".csv"
  print ("\n[DONE] Result of Cluster is at {}".format(outfile))
  
  df_merge3=df_merge2.copy()  
  
  df_merge3['rfm_group']=df_rest['rfm_group']
  colum = list(df_merge3.columns[-3:]) + list(df_merge3.columns[:-3])
  #cols = df_merge3.columns.tolist()
  #cols = [cols[-1]]+cols[:-1]
  df_merge3=df_merge3[colum]
  
  #process drop cluster

  drop_column = df_merge3.columns.values
  df_drop_cluster = pd.DataFrame(columns=drop_column)
  df_drop_cluster['rfm_group']=df_drop['rfm_group']
  df_drop_cluster['Cluster Labels']=(np.zeros(len(df_drop))+1)*kcluster
  df_drop_cluster['cus_id']=df_drop['cus_id']
  df_drop_cluster[list_features]=df_drop[list_features]
  df_drop_cluster[list_fav_colname]=np.zeros((len(df_drop),len(list_fav_colname)))
  df_drop_cluster['cluster_type']=['refund']*len(df_drop)
  df_drop_cluster['cluster_labeling']=['refund']*len(df_drop)
  
  print(df_drop_cluster[:3])
  print(len(df_drop_cluster))

  df_clustered = pd.concat([df_merge3,df_drop_cluster])
  
  df_clustered.to_csv(outfile)
  # DONE: This file is sent to DW.
  final_column1 = df_clustered.columns.values
  final_column2 = df_group_cluster_info.columns.values
  print ('##################################')
  print ('# COMPLETED: {} k={}'.format(groupx,kcluster))
  print ('##################################')
  print (df_group_cluster_info.shape)
  print (df_group_cluster_info)
  
  return df_clustered,df_group_cluster_info
### END
############################################################

# main

output_df = pd.DataFrame()
output_gc_df = pd.DataFrame()

for groupi in list(df_pivot['rfm_group'].unique()):
  print('Cluster for {}',groupi)
  df_groupi = df_pivot[df_pivot['rfm_group']==groupi]
  df_groupi_cluster,df_groupi_cluster_info = df_group_cluster(df_groupi,groupi,kcluster)
  
  output_df = pd.concat([output_df, df_groupi_cluster])
  output_gc_df = pd.concat([output_gc_df,df_groupi_cluster_info])
#---------------------------------

output_short_df = output_df.iloc[:,0:12]

outfinal = output_path+"/"+version+"_aldo_cus_cluster_by_product.csv"
output_df.to_csv(outfinal)

outfinal = output_path+"/"+version+"_aldo_cus_cluster_by_product_short.csv"
output_short_df.to_csv(outfinal)

outfinal = output_path+"/"+version+"_aldo_group_cluster_info.csv"
output_gc_df.to_csv(outfinal)
#----------------------------------

cluster_summary_df = output_short_df[['rfm_group','Cluster Labels','cluster_type','cluster_labeling']]
cluster_summary_df.drop_duplicates(subset=['rfm_group','Cluster Labels','cluster_type','cluster_labeling'], keep='first',inplace=True)


#compute number customers for each cluster
cluster_summary_df['total_cus']=np.zeros(len(cluster_summary_df))

for i in list(cluster_summary_df['rfm_group'].unique()):
  for j in list(cluster_summary_df['Cluster Labels'].unique()):
    total_cus = output_short_df[(output_short_df['rfm_group']==i) & (output_short_df['Cluster Labels']==j)]['cus_id'].count()	  
    cluster_summary_df.loc[(cluster_summary_df['rfm_group']==i) & (cluster_summary_df['Cluster Labels']==j),'total_cus']=total_cus
	
outfinal = output_path+"/"+version+"_aldo_group_cluster_summary.csv"
cluster_summary_df.to_csv(outfinal)

print ('[INFO] Cluster Summary:\n{}'.format(cluster_summary_df))


print ('##################################')
print ('# ENDING ! CONGRATULATION')
print ('##################################')