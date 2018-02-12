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

import random
import pull_data
import update_dbs
import saved_models
import pandas as pd
import numpy as np

for x_vals, y_val in [('line', 'result'), ('ou', 'result')]: 
        x_data_stable = pd.read_csv(os.path.join(derived_folder, '%s_data.csv' % (x_vals)))
        x_data_stable = x_data_stable.set_index('Unnamed: 0')
        y_data = x_data_stable[[y_val]]
        x_cols = list(x_data_stable)
        x_cols.remove(y_val)
        x_data_stable = x_data_stable[x_cols]   
        train_index = pull_data.pull_train_index(update_dbs.mysql_client())
        
        random.seed(86)
        random.shuffle(train_index)
        derived_data = {}
        
        training_partitions = [train_index[i:i + int(len(train_index)/25)] for i in range(0, len(train_index), int(len(train_index)/50))]
        for model_name, model_details in saved_models.stored_models[y_val][x_vals].items():
            print('Loading %s Values'%(model_name))
            derived_data[model_name] = {}
            for part in training_partitions:
                prediction = model_details['model'].fit(x_data_stable[model_details['features']].loc[set(x_data_stable.index) - set(x_data_stable.index[x_data_stable[model_details['features']].index.isin(part)])], np.ravel(y_data.loc[set(y_data.index) - set(y_data.index[y_data.index.isin(part)])])).predict(x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)])
                for pred, idx in zip(prediction, x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)].index):
                    derived_data[model_name][idx] = pred
            print('...Loaded %s Values'%(model_name))

        print('Loading %s Values'%(y_val))            
        save_data = pd.DataFrame.from_dict(derived_data).join(y_data)
        save_data.to_csv(os.path.join(derived_folder, '%s-%s_derived.csv'%(x_vals, y_val)))
        print('...Loaded %s Values'%(y_val))
        
    
    
    