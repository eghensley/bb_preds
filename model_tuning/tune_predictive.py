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

output_folder = os.path.join(cur_path, 'model_results')
features_folder = os.path.join(cur_path, 'feature_dumps')
input_folder = os.path.join(cur_path, 'derived_data')

import lasso_tuning
import ridge_tuning
import lgb_tuning
import linsvm_tuning
import pull_data
import update_dbs
import pandas as pd
import numpy as np
import saved_models
import random

train_index = pull_data.pull_train_index(update_dbs.mysql_client())
#cnx = update_dbs.mysql_client()
random.seed(86)
random.shuffle(train_index)
derived_data = {}

x_vals = 'points'
y_val = '+pts'
x_data_stable = pull_data.score(update_dbs.mysql_client())
x_cols = list(x_data_stable)
x_cols.remove(y_val)
y_data = x_data_stable[y_val] 

#x_data_stable = x_data_stable[x_cols]
#x_data = None
#training_partitions = [train_index[i:i + int(len(train_index)/50)] for i in range(0, len(train_index), int(len(train_index)/50))]
#model = saved_models.stored_models[x_vals][y_val]['model']
#features = saved_models.stored_models[x_vals][y_val]['features']
#print('Loading %s Values'%(x_vals+'_'+y_val))
#derived_data[x_vals+'_'+y_val] = {}
#for part in training_partitions:
#    prediction = model.fit(x_data_stable[features].loc[set(x_data_stable.index) - set(x_data_stable.index[x_data_stable[features].index.isin(part)])], np.ravel(y_data.loc[set(y_data.index) - set(y_data.index[y_data.index.isin(part)])])).predict(x_data_stable[features].loc[x_data_stable[features].index.isin(part)])
#    for pred, idx in zip(prediction, x_data_stable[features].loc[x_data_stable[features].index.isin(part)].index):
#        derived_data[x_vals+'_'+y_val][idx] = pred
#print('...Loaded %s Values'%(x_vals+'_'+y_val))




x_data = x_data_stable[x_cols]
result = lasso_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
print("Best %s %s score: %s" % (x_vals, y_val, result))
 
x_data = x_data_stable   
result = ridge_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
print("Best %s %s score: %s" % (x_vals, y_val, result))               

x_data = x_data_stable   
result = linsvm_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
print("Best %s %s score: %s" % (x_vals, y_val, result)) 
        
x_data = x_data_stable   
result = lgb_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
print("Best %s %s score: %s" % (x_vals, y_val, result))  
