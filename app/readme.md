### 8/19/2019
update run.py with below points:
1. add two plots for data visualtions
2. correct the typos

Note:
if we change the code following the suggestions as below:
  1) line 61: 
  You have to go a directory back to read from the data directory. You can do that using two dots .. like this sqlite:///../data/DisasterResponse.db.
  2) line 66:
  The model directory is two directory back as well. So the correct path is ../model/classifier.pkl
  
  we will have below errors:
  1) line 61: 
    File "app/run.py", line 62, in <module>
    df = pd.read_sql_table('data/DisasterResponse.db', engine)
    ...
    sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) 
    unable to open database file (Background on this error at: http://sqlalche.me/e/e3q8)
    
   2) line 66:
     File "app/run.py", line 66, in <module>
     model = pickle.load(open('../models/classifier.pkl', 'rb'))
     FileNotFoundError: [Errno 2] No such file or directory: '../models/classifier.pkl'
     
   Thus, the current version of run.py does not include above two changes.
    
    
