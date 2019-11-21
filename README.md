# data_science
using for sharing data science code


**Project 1: Customer Clustering by Product (K-Means Clustering)**

0.04.1

Run command: 

  >python ml_cus_cluster_kmc_rl_0.04.1.py -k N
  
  with N is number of Clusters
  
0.04.3
  
Command Options:

   >python ml_cus_cluster_kmc_rl_0.04.3.py -h

     Allow user in INDICATING NAME OF TABLE in BigQuery to query data     
     Allow user in INDICATING OUT-DIRECTORY (output will include file and debug part)     
     Allow user in TURNING OFF QUERY data from BigQuery if NO NEED UPDATE DATA (previous queried data is stored offline already)     
     Allow user in COMMITING data to Bigquery
     
    Getoption: -k --kcluster   num of cluster (default k=4)
               -q --query      query for query data from bigquery (default: no query)
               -i --input      name of data table in sandbox
               -o --outdir     path to output result 
               -l --limit      LIMIT in Query command (eg. -l 100 means LIMIT 100)
               -n --nopivot    disable pivot table (support another format of data)
               -c --commit     commit output to sandbox
               -h --help       run for help

    Example: python vtidss_ml_aldo_cus_cluster_kmc_rl_0.04.3.py -k 10 -i Aldo.tablex [-o outdir_path] [-limit 1000] [--querry] [--commit]

Code Flow:
 
    Query data from BigQuery    
    Data preprocessing (clean data, data wrangling)    
    Run K-mean clustering    
    Analyse customer insight    
    Labeling for each customer    
       - Three type of Labels: Dom, 2-Dom, Mix        
       - Current rule: gt 55% and twice gt the next       
    Save data to file    
    commit to sandbox
 
**Project 2: Revenue Prediction (Deep Neural Networks)**

0.06.3

Run command: 

  >python ml_rev_pre_dnn_rl_0.06.3.py -h

Code Flow:
   
    Query data from BigQuery
    Data processing
    Train and Valid data spliting
    Process input data for Window dataset
    Model Declaration
    Train Model and check Loss with diff Learning Rates (LRs)
    Train Model again with best LR
    Prediction with valid data
    Get Accuracy of the model
   
   
**mylib: Commit data from local to BigQuery**

0.2

Run command: 

  >python commit_bq_0.2.py -h
  
Command Guide:

    Command: python bq_commit.py -h
             python bq_commit.py -s file_path -d bq_table_path -p 200
  
    Function:
     Two main modes:
         Commit File:   Commit particular csv file to BigQuery Table name (eg. Aldo.tablename)
                        python vtidss_commit_bq_0.2.py -s ./filename.csv -d Aldo.tb_name [--pause 500] [--verify]
         Commit Folder: Commit all csv file in a folder to BigQuery branch (eg. Aldo or Hlc)(--sweep is must)
                        python vtidss_commit_bq_0.2.py -s ./folder/ -d Aldo --sweep [--pause 100] [--verify]
                        Auto sweep file csv in folder
                        Auto create table name
         2 modes of verification: OFFLINE VERIFY: check number of INSERT cmd line ran with total line of csv file
                                  ONLINE VERIFY:  query commited data in BigQuery and check content with offline csv
         Allow change pause time between INSERT command to pass collapse
         Auto create SQL command including: DROP, CREATE, INSERT
         Debug log in filename_debug/debug.log
