##############################################################
# COMMIT FILE FOR LOCAL TO BIGQUERY
# File name: vtidss_bq_commit.py
# Previous: 
# Method: 
# Date: 20/11/2019
# Author: Nhan Thanh Ngo
# Company: VTI-DSS
# Description: 
#    
# Status: COMPLETED
# Specification:
#     Commit Specific File, need:      --source [file_path]  , --destination [BQ-TableName]
#     Commit all CSV in folder, need:  --source [folder_path], --destination [BQ-Parent_branch]
#
# Command: python vtidss_bq_commit.py -h
#          python vtidss_bq_commit.py -s file_path -d bq_table_path -p 200
#
###################################################
# IMPORT LIBRARY
# 

# GENERAL LIB     
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import glob
import re
import traceback #for exception
import os

# GET OPTIONS
import sys, getopt #for getting option for cmd
from optparse import OptionParser # for help note



###############################################################################
###############################################################################
# INITIAL DECLARATION
# 

# <-- CHANGE THIS FOR NEW VERSION -->
version = '0_00_2' 
versionx=re.sub("_",".",version)
filename = "vtidss_bq_commit_"+versionx+".py"
dblog = filename+'\n'

#default variables
source = '' 
dest = '' 
sweep = False
verify = False
pause = float(0.1)

###############################################################################
# GET OPTIONS
# 
# 

usage = "usage: %prog [options] arg1 arg2\n\nExample: 1) COMMIT 1 FILE: \n\tpython %prog -s ./filename.csv -d Aldo.tb_name [--pause 500] [--verify]\n\t 2) COMMIT ALL FILES IN FOLDER: \n\tpython %prog -s ./folder/ -d Aldo --sweep [--pause 100] [--verify]"

parser = OptionParser(usage=usage)

#parser.add_option("-h", "--help",
#                  action="store_true", dest="verbose", default=True,
#                  help="print help information of the script such as how to run and arguments")

parser.add_option("-s", "--source",
                  default="./",
                  metavar="SOURCE", help="Folder and File path. To commit a particular file, -s will be filepath. To commit all file in folder, -s will be path to folder store files."
                                        "[default: %default]")						
				 
parser.add_option("-d", "--destination",
                  default="./",
                  metavar="BIGQUERY", help="if -w disable, -d should include path + table name. If -w enable, -d should by path to where tables will be generated, the table name is auto-gen."
                                           "[default: %default]")		
										   
parser.add_option("-w", "--sweep",
                  default="OFF", metavar="SWEEP",
                  help="Sweep All Folder, name of table in Sandbox will be auto-gen based on source file name. In this case, source will be filepath."
                       "[default: %default]")
					   
parser.add_option("-v", "--verify",
                  default="OFFLINE CHECKING", metavar="VERIFY",
                  help="[ONLINE CHECKING] Verify commit operation by check directly data in BigQuery"
                       "[default: %default]")

parser.add_option("-p", "--pause",
                  default="0.1s", metavar="PAUSE",
                  help="pause time for each INSERT COMMAND, input value in mili-second (eg. 100, means 100ms)"
                       "[default: %default]")
				  
try:
  opts, args = getopt.getopt(sys.argv[1:], 'hs:d:p:v:w', ['help','source=','destination=','pause=','verify','sweep'])
    
except getopt.GetoptError as err:
  print ("ERROR: Getoption gets error... please check!\n {}",err)
  sys.exit(1)

for opt, arg in opts:
  if opt in ('-s', '--source'):
    source = str(arg)
  if opt in ('-d', '--destination'):
    dest = str(arg)
  if opt in ('-v', '--verify'):
    verify = True
  if opt in ('-w', '--sweep'):
    sweep = True
  if opt in ('-p', '--pause'):
    pause = int(arg)/1000
  if opt in ('-h', '--help'):
    parser.print_help()    
    sys.exit(2)

  
# Get option completed and show info
print ("From VTI: Hello WORLD")
print ("This is {}\n".format(filename))

print("##################################")
print("COMMIT INFORMATION")
print("SWEEP FOLDER: {}".format(sweep))  
print("SOURCE: {}".format(source))
print("DESTINATION: {}/".format(dest))
print("VERIFY: {}".format(verify))
print("PAUSE: {}".format(pause))
print("##################################")

########################################################
# Function: Print Debug
#

def print_debug (strcont):
  print(strcont)  
  with open(debug_path+"/debug.log", "a") as logfile:
    logfile.write("\n"+strcont)

###################################################
# Function: Upload data to sandbox
#      In sandbox: create table, then insert data to table
#      Input: filename: (.csv file)
#             sandbox_tbname: table in sandbox
#      Output: return True/False (successful or not)
#

def commit_data_to_sandbox (filename,sandbox_tbname):
  
  print ("[Commit data to sandbox] filename = {}".format(filename))
  print ("[Commit data to sandbox] tbname = {}".format(sandbox_tbname))
   
  def_status = False
  tb_bq_dtype = []
  tb_assign = []
  
  sql_create_cmd = "CREATE TABLE IF NOT EXISTS " + sandbox_tbname + " ("
  
  sql_insert_cmd1 = "\"INSERT INTO " + sandbox_tbname + " (" 
  sql_insert_cmd2 = " VALUES ("
  sql_insert_feedinfo = "%(" 

  
  df = pd.read_csv(filename)
  df.drop(df.columns[0], axis = 1, inplace=True)
  dftypes = df.dtypes
  for i in range(len(dftypes)):
   
    if dftypes[i] == 'int64':
      tb_bq_dtype.append('INT64')
      tb_assign.append("%d")
    elif dftypes[i] == 'float64':
      tb_bq_dtype.append('FLOAT64')
      tb_assign.append("%d")
    elif dftypes[i] == 'object':
      tb_bq_dtype.append('STRING')
      tb_assign.append("'%s'")
    
    #create command in string
    sql_create_cmd = sql_create_cmd + df.columns.values[i].replace(" ", "_") + " " + tb_bq_dtype[i] + (" )" if i == (len(dftypes)-1) else ",")

    #insert command in string
    #INSERT INTO Aldo.ml_aldo_cus_cluster_kmc_rs (rfm_group,cluster_number,cluster_type,cluster_labeling,total_cus)
    sql_insert_cmd1 = sql_insert_cmd1 + df.columns.values[i].replace(" ", "_") + (")" if i == (len(dftypes)-1) else ",")
	
    #VALUES ('%s',%d,'%s','%s',%d)	
    sql_insert_cmd2 = sql_insert_cmd2 + tb_assign[i] + (")\"" if i == (len(dftypes)-1) else ",")
	
    #%(df['rfm_group'][i],df['Cluster Labels'][i],df['cluster_type'][i],df['cluster_labeling'][i],df['total_cus'][i])
    sql_insert_feedinfo = sql_insert_feedinfo + "df['" + df.columns.values[i] + "'][i]" + (")" if i == (len(dftypes)-1) else ",")
	
  print(sql_insert_cmd1)
  print(sql_insert_cmd2)
  print(sql_insert_feedinfo)
  
  #DROP ---------
  sql_drop = ('DROP TABLE IF EXISTS '+sandbox_tbname)
  print_debug ("[INFO] SQL sql_drop: \n {}\n".format(sql_drop))
    
  sandbox_tb_drop = client.query(sql_drop, project=project_id)
  rows = sandbox_tb_drop.result() #waiting complete

  print_debug ("[INFO] Complete all part of SQL command\n Commit ready...\n")
  
  #CREATE -------
  sql_create = (sql_create_cmd)
  print_debug ("[INFO] SQL sql_create_cmd:\n {}\n".format(sql_create))   
  
  
  sandbox_tb_create = client.query(sql_create, project=project_id)
  wait = sandbox_tb_create.result() #waiting complete
    
  #INSERT TABLE CONTENT
  num_insert_row = 0
  for i in range(len(df)):
    
    #INSERT -------
    sql_insert = eval(sql_insert_cmd1+sql_insert_cmd2+" "+sql_insert_feedinfo)
    print_debug("[{}] ".format(i) + "{}".format(sql_insert) + "\n")
   
	
    sandbox_tb_insert = client.query(sql_insert, project=project_id)
    wait = sandbox_tb_insert.result() #waiting complete    
    time.sleep(pause)
    num_insert_row+=1
  
  #Verify after completing insert data
  if verify:
    print_debug ("ONLINE CHECKING\n")
    #sql_select = "SELECT count(*) as table_length FROM "+sandbox_tbname
    sql_select = "SELECT * FROM "+sandbox_tbname
    print_debug ("[INFO] SQL sql_select:\n {}\n".format(sql_select))  
  
    # Run a Standard SQL query with the project set explicitly
    df_select = client.query(sql_select, project=project_id).to_dataframe() 
    print_debug ("[INFO] df_select: \n{}\n".format(df_select))
 
    df_unmatch = pd.concat([df, df_select]).drop_duplicates(keep=False)
	
    if len(df_unmatch) == 0:     
    #if len(df) == df_select['table_length'][0]:
      print_debug ("[PASS] length of df_unmatch {}".format(len(df_unmatch)))
      print_debug ("[PASS] ONLINE CHECK: COMMIT SUCCESSFUL!".format(filename))
      return True
    else:
      print_debug ("[FAIL] ONLINE CHECK: COMMIT MISMATCH!".format(filename))
      print_debug ("[FAIL] mismatch content \n{}".format(len(df_unmatch)))
      return False
  else:
    print_debug ("OFFLINE CHECKING\n")
    if num_insert_row!=len(df):
      print_debug("[PASS] num_insert_row is {} differing with expected {}\n".format(num_insert_row, len(df)))
      return False
    else:
      print_debug("[FAIL] num_insert_row {} equalizing with expected {}\n".format(num_insert_row, len(df)))
      return True

###################################################
# OUTPUT FOLDER AND DEBUG
#
#

# create debug folder	
debug_path = "./"+filename.split('.py')[0].replace(".","_")+"_debug"

if os.path.exists(debug_path):
  print ("\'{}\' is already EXISTED! --> REMOVE OLD DIR...".format(debug_path))
  #shutil.rmtree(debug_path)
else:
  os.mkdir(debug_path)
  print ("\'{}\' is CREATED!".format(debug_path))

if os.path.isfile(debug_path+"/debug.log"):
  os.remove(debug_path+"/debug.log")

###################################################
# QUERY DATA FROM BIGQUERY
#
#

from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("../../vti_sandbox.json")
project_id = 'vti-sandbox'
client = bigquery.Client(credentials= credentials,project=project_id)

'''
sql_drop = ('DROP TABLE IF EXISTS Aldo.abc')
print ("[INFO] SQL sql_drop: \n {}\n".format(sql_drop))
  
sandbox_tb_drop = client.query(sql_drop, project=project_id)
rows = sandbox_tb_drop.result() #waiting complete
'''
response = False

try: 
  if not sweep:  
    print ('NO SWEEP')    
	
    print_debug("####################################################################\n")
    print_debug("# COMMIT {} TO {}\n".format(source,dest))
    print_debug("####################################################################\n")
  
    response = commit_data_to_sandbox(source,dest)

    if response:
      print_debug("[INFO] COMMIT {} TO {} SUCCESSED ...".format(source,dest))	
    else:
      print_debug('ERROR: FAIL COMMIT {} TO {}...'.format(source,dest))
      system.exit(5)  
  else:	 
    print ('SWEEP')
    ls_dir = os.listdir(source)
    csv_flist = []
    bq_nlist = []
	
    for f in ls_dir:
      fm = re.match("^([\-\.\w]+).csv$",f)
      if fm:
        csv_flist.append(fm.group())     
        bq_nlist.append(dest+'.'+(fm.group().split('.csv')[0]).replace('-','_').replace('.','_'))
    
    print ("[INFO] Source List: \n{}".format(csv_flist))
    print ("[INFO] Destination List: \n{}".format(bq_nlist))
  
    #Run commit for each files in folder
    for i in range(len(csv_flist)):
	
      file_path = source + '/' + csv_flist[i]
      print_debug("####################################################################")
      print_debug("# COMMIT {} TO {}".format(file_path,bq_nlist[i]))
      print_debug("####################################################################")
	  
      response = commit_data_to_sandbox(file_path,bq_nlist[i])    
  
      if response:
        print_debug("[INFO] COMMIT {} TO {} SUCCESSED ...".format(file_path,bq_nlist[i]))
	
      else:
        print_debug('ERROR: FAIL COMMIT {} TO {}...'.format(file_path,bq_nlist[i]))
	
except Exception:
    traceback.print_exc()

print("COMPLETED!!!")

