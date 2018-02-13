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
derived_folder = os.path.join(cur_path, 'derived_data')

import pull_data
import update_dbs
import pandas as pd
import numpy as np
import log_tuning
import lgclass_tuning
import linsvc_tuning
import knn_tuning
import feature_lists


for each in [('combined', 'pts')]:
        x_vals, y_val = each
        data = pd.read_csv(os.path.join(derived_folder, 'points.csv'))
        data = data.set_index('Unnamed: 0')
        x_feats = list(data)
        x_feats.remove(y_val)
        y_data = np.ravel(data[[y_val]])
        x_data_stable = data[x_feats]
        
        x_data = x_data_stable   
        result = log_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
        print("Best %s %s score: %s" % (x_vals, y_val, result))  
        
        
        x_data = x_data_stable   
        result = knn_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
        print("Best %s %s score: %s" % (x_vals, y_val, result))
 
        x_data = x_data_stable   
        result = linsvc_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
        print("Best %s %s score: %s" % (x_vals, y_val, result))               

        x_data = x_data_stable   
        result = lgclass_tuning.execute(y_val, x_vals, X_data = x_data, Y_data = y_data)
        print("Best %s %s score: %s" % (x_vals, y_val, result)) 


        
