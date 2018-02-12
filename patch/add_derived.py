import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
while cur_path.split('/')[-1] != 'bb_preds':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(-1, os.path.join(cur_path, 'model_conf'))
sys.path.insert(-1, os.path.join(cur_path, 'db_utils'))
sys.path.insert(-1, os.path.join(cur_path, 'model_tuning'))
derived_folder = os.path.join(cur_path, 'derived_data')
import pandas as pd
import update_dbs
import numpy as np
cnx = update_dbs.mysql_client()

for name in ['offensive_preds', 'defensive_preds']:
    data = pd.read_csv('%s.csv'%(name))
    cursor = cnx.cursor() 
    insertlist = []
    continuance = 0
    for entry in np.array(data):
        insert = list(entry)
        idx = insert[0]
        date = '"'+idx[:10]+'"'
        tname = '"'+idx[10:].replace('_', ' ')+'"'
        insert = insert[1:]
        sql_insert = []
        sql_insert.append(tname)
        sql_insert.append(date)
        for each in insert:
            sql_insert.append(str(each))
        sql_insert = '('+', '.join(sql_insert)+')'
        insertlist.append(sql_insert)
        continuance += 1
        if continuance == 500:
            insertlist = ', '.join(insertlist)
            oddslist = ['INSERT INTO %s VALUES '%(name), insertlist, ';']
            initialoddsinsert = ' '.join(oddslist)  
            add_odds = initialoddsinsert  
            cursor.execute('SET foreign_key_checks = 0;')
            try:
                cursor.execute(add_odds)
                cnx.commit()
                print(entry)
            except:
                pass
            cursor.execute('SET foreign_key_checks = 1;')
            insertlist = []
            continuance = 0
            
            
    insertlist = ', '.join(insertlist)
    oddslist = ['INSERT INTO %s VALUES '%(name), insertlist, ';']
    initialoddsinsert = ' '.join(oddslist)  
    add_odds = initialoddsinsert  
    cursor.execute('SET foreign_key_checks = 0;')
    try:
        cursor.execute(add_odds)
        cnx.commit()
        print(entry)
    except:
        pass
    cursor.execute('SET foreign_key_checks = 1;')
    insertlist = []
    continuance = 0