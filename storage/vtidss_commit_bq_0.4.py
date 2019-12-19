##############################################################
# COMMIT FILE FOR LOCAL TO BIGQUERY
# File name: vtidss_commit_bq_0.4.py
# Method: 
# Date: 17/12/2019
# Author: Nhan Thanh Ngo
# Company: VTI-DSS
# Description: 
#   
# Status: DEV
# Specification:
#     Commit Specific File, need:      --source [file_path]  , --destination [BQ-TableName]
#     Commit all CSV in folder, need:  --source [folder_path], --destination [BQ-Parent_branch]
#
# Command: python vtidss_bq_commit.py -h
#          python vtidss_bq_commit.py -s file_path -d bq_table_path -p 200
#  
# Function:
#   [0.2] Two main modes:
#         Commit File:   Commit particular csv file to BigQuery Table name (eg. Aldo.tablename)
#                        python vtidss_commit_bq_0.2.py -s ./filename.csv -d Aldo.tb_name [--pause 500] [--verify]
#         Commit Folder: Commit all csv file in a folder to BigQuery branch (eg. Aldo or Hlc)(--sweep is must)
#                        python vtidss_commit_bq_0.2.py -s ./folder/ -d Aldo --sweep [--pause 100] [--verify]
#                        Auto sweep file csv in folder
#                        Auto create table name
#         Auto create SQL command including: DROP, CREATE, INSERT
#         Debug log in filename_debug/debug.log
#
#   [0.3] Auto detect INSERT corruption and re-INSERT Until complete
#   [0.4] INSERT multi-rows in 01 command
#
#   Run Command: python vtidss_commit_bq_0.4.py -s ./0_04_4_output/0_04_4_aldo_group_cluster_info.csv -d Aldo.test --pause 50 --verify --numrow 1000
#

###################################################
# IMPORT LIBRARY
# 

# GENERAL LIB     
import numpy as np
import pandas as pd
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
version = '0_00_4' 
versionx=re.sub("_",".",version)
filename = "vtidss_bq_commit_"+versionx+".py"
dblog = filename+'\n'

#default variables
source = '' 
dest = '' 
sweep = False
verify = False
pause = float(0.1)
NUM_ROWS = 5

###############################################################################
# GET OPTIONS
# 
# 

usage = "usage: %prog [options] arg1 arg2\n\nExample: 1) COMMIT 1 FILE: \n\tpython %prog -s ./filename.csv -d Aldo.tb_name [--pause 500] [--verify] [--numrow 10]\n\t 2) COMMIT ALL FILES IN FOLDER: \n\tpython %prog -s ./folder -d Aldo --sweep [--pause 100] [--verify] [--numrow 10]"

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
					   
parser.add_option("-n", "--numrow",
                  default="5", metavar="NUM_ROWS",
                  help="number of rows will be upload in each INSERT command"
                       "[default: %default]")


try:
  opts, args = getopt.getopt(sys.argv[1:], 'hs:d:p:v:w:n', ['help','source=','destination=','pause=','verify','sweep','numrow='])
    
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
  if opt in ('-n', '--numrow'):
    NUM_ROWS = int(arg)
  if opt in ('-h', '--help'):
    parser.print_help()    
    sys.exit(2)

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

debug_log = source.split('/')[-1].split('.')[0]+'_debug.log'

if os.path.isfile(debug_path+"/"+debug_log):
  os.remove(debug_path+"/"+debug_log)
  
unmatch_csv = source.split('/')[-1].split('.')[0]+'_unmatch.csv'

if os.path.isfile(debug_path+"/"+unmatch_csv):
  os.remove(debug_path+"/"+unmatch_csv)
########################################################
# Function: Print Debug
#

def print_debug (strcont):
  print(strcont)  
  with open(debug_path+"/"+debug_log, "a") as logfile:
    logfile.write("\n"+strcont)

########################################################
# Function: Print Debug
#
  
# Get option completed and show info
print ("From VTI: Hello WORLD")
print ("This is {}\n".format(filename))

print_debug("##################################")
print_debug("COMMIT INFORMATION")
print_debug("SWEEP FOLDER: {}".format(sweep))  
print_debug("SOURCE: {}".format(source))
print_debug("DESTINATION: {}".format(dest))
print_debug("VERIFY: {}".format(verify))
print_debug("PAUSE: {}".format(pause))
print_debug("NUM_ROWS: {}".format(NUM_ROWS))
print_debug("##################################")


###################################################
# Function: Upload data to sandbox
#      In sandbox: create table, then insert data to table
#      Input: filename: (.csv file)
#             sandbox_tbname: table in sandbox
#      Output: return True/False (successful or not)

#"VALUES ('%s',%d,'%s','%s',%d),('%s',%d,'%s','%s',%d);"%(df['rfm_group'][i],df['Cluster Labels'][i],df['cluster_type'][i],df['cluster_labeling'][i],df['total_cus'][i])
def insert_cmd_val (df,NUM_ROWS,tb_assign):

  dftypes = df.dtypes  
  sql_insert_cmd2 = " VALUES "
  sql_insert_value = ""
  sql_insert_feedinfo = "%("
  for k in range(NUM_ROWS):
    sql_insert_value_row = "("
    sql_insert_feedinfo_row = ""
    for i in range(len(dftypes)):     
        
      #VALUES ('%s',%d,'%s','%s',%d)    
      sql_insert_value_row = sql_insert_value_row + tb_assign[i] + (")" if i == (len(dftypes)-1) else ",")
    
      #%(df['rfm_group'][i],df['Cluster Labels'][i],df['cluster_type'][i],df['cluster_labeling'][i],df['total_cus'][i])
      sql_insert_feedinfo_row = sql_insert_feedinfo_row + "df['" + df.columns.values[i] + "'][i+"+str(k)+"]" + ("" if i == (len(dftypes)-1) else ",")
    
    #final string
    sql_insert_value = sql_insert_value + sql_insert_value_row + (";\"" if k == (NUM_ROWS-1) else ",")
    sql_insert_feedinfo = sql_insert_feedinfo + sql_insert_feedinfo_row + (")" if k == (NUM_ROWS-1) else ",")
  
  sql_insert_cmd2 = sql_insert_cmd2 + sql_insert_value + sql_insert_feedinfo 
  
  return sql_insert_cmd2
  
#################################################################

def commit_data_to_sandbox (filename,sandbox_tbname,NUM_ROWS,PAUSE):
  
  print ("[Commit data to sandbox] filename = {}".format(filename))
  print ("[Commit data to sandbox] tbname = {}".format(sandbox_tbname))
    
  def_status = False
  tb_bq_dtype = []
  tb_assign = []
  
  sql_create_cmd = "CREATE TABLE IF NOT EXISTS " + sandbox_tbname + " ("
  
  sql_insert_cmd1 = "\"INSERT INTO " + sandbox_tbname + " (" 
  sql_insert_cmd2 = " VALUES "
  sql_insert_feedinfo = "%(" 


  df = pd.read_csv(filename)
  df.drop(df.columns[0], axis = 1, inplace=True)
  #df.reset_index(drop=True)
  dftypes = df.dtypes  
  
  remain = len(df)
  
  for i, col in zip(range(len(dftypes)),df.columns.values):
    if dftypes[i] == 'float64':
      df[col] = round(df[col],6)
  
  for i in range(len(dftypes)):
    
    if dftypes[i] == 'int64':
      tb_bq_dtype.append('INT64')
      tb_assign.append("%d")
    elif dftypes[i] == 'float64':
      tb_bq_dtype.append('FLOAT64')	  
      tb_assign.append("%f")
    elif dftypes[i] == 'object':
      tb_bq_dtype.append('STRING')
      tb_assign.append("'%s'")
     
    #create command in string
    sql_create_cmd = sql_create_cmd + df.columns.values[i].replace(" ", "_") + " " + tb_bq_dtype[i] + (" )" if i == (len(dftypes)-1) else ",")

    #insert command in string
    #INSERT INTO Aldo.ml_aldo_cus_cluster_kmc_rs (rfm_group,cluster_number,cluster_type,cluster_labeling,total_cus)
    sql_insert_cmd1 = sql_insert_cmd1 + df.columns.values[i].replace(" ", "_") + (")" if i == (len(dftypes)-1) else ",")
      
  
  sql_insert_cmd2_normal = insert_cmd_val (df,NUM_ROWS,tb_assign)

  print(sql_insert_cmd2_normal)

  #print(sql_insert_feedinfo)
  

  #DROP ---------
  sql_drop = ('DROP TABLE IF EXISTS '+sandbox_tbname)
  print_debug ("[INFO] SQL sql_drop: \n {}\n".format(sql_drop))  
  
  sandbox_tb_drop = client.query(sql_drop, project=project_id)
  rows = sandbox_tb_drop.result() #waiting complete
  time.sleep(5*PAUSE)
  
  print_debug ("[INFO] Complete all part of SQL command\n Commit ready...\n")
  
  #CREATE -------
  sql_create = (sql_create_cmd)
  print_debug ("[INFO] SQL sql_create_cmd:\n {}\n".format(sql_create))   
  
  
  sandbox_tb_create = client.query(sql_create, project=project_id)
  wait = sandbox_tb_create.result() #waiting complete
  time.sleep(5*PAUSE)
  
  #INSERT TABLE CONTENT
  num_insert_row = 0
  commit_not_done = True
  last_numrow = 0
  count_stuck = 0
  while commit_not_done:
    try: 
      for i in range(num_insert_row,len(df),NUM_ROWS):
        #INSERT -------
        sql_insert = ''
        if (i+NUM_ROWS-1) < len(df):
          sql_insert = eval( sql_insert_cmd1 + sql_insert_cmd2_normal)
          		  
        else:
          sql_insert_cmd2_last = insert_cmd_val (df,len(df)-i,tb_assign)
          sql_insert = eval( sql_insert_cmd1 + sql_insert_cmd2_last)
          last_numrow = len(df)-i
          #print_debug("[3]last_numrow = {}".format(last_numrow))

        #debug
        print_debug("[{}] ".format(i) + "{}".format(sql_insert) + " [{}]".format(i+NUM_ROWS-1) + "\n")        
        num_insert_row += NUM_ROWS     
        sandbox_tb_insert = client.query(sql_insert, project=project_id)
        wait = sandbox_tb_insert.result() #waiting complete  
        time.sleep(PAUSE)
        #print_debug("num_insert_row = {}".format(num_insert_row))

      commit_not_done = False
      print_debug ("INSERT TABLE COMPLETED!")
    except:
      count_stuck+=1
      if count_stuck==10:
        count_stuck=0
        PAUSE=PAUSE*2
      print_debug("count_stuck={}, PAUSE={}".format(count_stuck,PAUSE))
      num_insert_row = num_insert_row - NUM_ROWS      
      print_debug("[WARNING] CORRUPTION OCCURS WHEN INSERTING")
      print_debug("Waiting {} second before re-inserting...".format(count_stuck*10*PAUSE))
      time.sleep(count_stuck*10*PAUSE)
	
    print_debug("Coming back and try to insert again from corrupted line...\n")  
	
  #update num_insert_row
  if last_numrow!=0:
    num_insert_row = num_insert_row - NUM_ROWS + last_numrow
	
  print_debug("[Final]num_insert_row = {}".format(num_insert_row))
  
  #Verify after completing insert data
  if verify:
    print_debug ("ONLINE CHECKING\n")
    #sql_select = "SELECT count(*) as table_length FROM "+sandbox_tbname
    sql_select = "SELECT * FROM "+sandbox_tbname
    print_debug ("[INFO] SQL sql_select:\n {}\n".format(sql_select))  
  
    # Run a Standard SQL query with the project set explicitly
    df_select = client.query(sql_select, project=project_id).to_dataframe()
    time.sleep(PAUSE)
    print_debug ("[INFO] df_select: \n{}\n".format(df_select))
    df.columns = df_select.columns.values
	
    df_unmatch = pd.concat([df, df_select]).drop_duplicates(keep=False)
    
    if len(df_unmatch) == 0:     
    #if len(df) == df_select['table_length'][0]:
      print_debug ("[PASS] length of df_unmatch {}".format(len(df_unmatch)))
      print_debug ("[PASS] ONLINE CHECK: COMMIT SUCCESSFUL!".format(filename))
      return True
    else:
      df_unmatch.to_csv(debug_path+"/"+unmatch_csv)
      print_debug ("[FAIL] ONLINE CHECK: COMMIT MISMATCH!".format(filename))
      print_debug ("[FAIL] mismatch content \n{}".format(len(df_unmatch)))
      print_debug ("[MISMATCH] df_unmatch = {}".format(df_unmatch))
	  
      return False
  else:
    print_debug ("OFFLINE CHECKING\n")
    if num_insert_row!=len(df):
      print_debug("[FAIL] num_insert_row is {} differing with expected {}\n".format(num_insert_row, len(df)))
      return False
    else:
      print_debug("[PASS] num_insert_row {} equalizing with expected {}\n".format(num_insert_row, len(df)))
      return True



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
  
    response = commit_data_to_sandbox(source,dest,NUM_ROWS,pause)

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
        csv_flist.append(fm.group()) #get value in group, remove .csv    
        bq_nlist.append(dest+'.'+(fm.group().split('.csv')[0]).replace('-','_').replace('.','_'))
    
    print ("[INFO] Source List: \n{}".format(csv_flist))
    print ("[INFO] Destination List: \n{}".format(bq_nlist))
  
    #Run commit for each files in folder
    for i in range(len(csv_flist)):
    
      file_path = source + '/' + csv_flist[i]
      print_debug("####################################################################")
      print_debug("# COMMIT {} TO {}".format(file_path,bq_nlist[i]))
      print_debug("####################################################################")
      
      response = commit_data_to_sandbox(file_path,bq_nlist[i],NUM_ROWS,pause)    
  
      if response:
        print_debug("[INFO] COMMIT {} TO {} SUCCESSED ...".format(file_path,bq_nlist[i]))
    
      else:
        print_debug('ERROR: FAIL COMMIT {} TO {}...'.format(file_path,bq_nlist[i]))
    
except Exception:
    traceback.print_exc()

print("[DONE] vtidss_commit_bq_0.4.py commited {} SUCESSFULLY!!!".format(source))

