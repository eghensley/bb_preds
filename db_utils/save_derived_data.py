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

import pull_data
import update_dbs
import pandas as pd
import numpy as np
import feature_lists
import saved_models
import random

def hfa_patch(x, cnx):
    print('Running HFA Patch')
    keep_stats = []
    patch_stats = []
    for stat in list(x):
        try:
            stat.split('_HAspread_')[1]
            patch_stats.append(stat)
        except IndexError:
            keep_stats.append(stat)
    
    patch_data = x[patch_stats]
    keep_data = x[keep_stats]
    cursor = cnx.cursor()
    query = 'Select oddsdate, favorite, underdog, homeaway from oddsdata;'
    cursor.execute(query)
    patch = pd.DataFrame(cursor.fetchall(), columns = ['date', 't1', 't2', 'location'])
    cursor.close()
    
    loc_adj = {}
    for d,t1, t2,l in np.array(patch):
        if l == 0:
            loc_adj[str(d)+t1.replace(' ', '_')] = 1
            loc_adj[str(d)+t2.replace(' ', '_')] = -1     
        else:
            loc_adj[str(d)+t1.replace(' ', '_')] = -1
            loc_adj[str(d)+t2.replace(' ', '_')] = 1
    patch = None 
    
    patch_data = patch_data.join(pd.DataFrame.from_dict(list(loc_adj.items())).set_index(0), how = 'left')
    away_data = patch_data[patch_data[1]==-1]
    away_data *= -1
    home_data = patch_data[patch_data[1]==1]
    patch_data = home_data.append(away_data)
    del patch_data[1]
    x = patch_data.join(keep_data)
    print('...Completed HFA Patch')
    return x

for y_val in ['pts_scored', 'pts_allowed']:  
    for x_vals in ['defensive_stats', 'offensive_stats', 'full-team', 'all', 'possessions', 'target']:
        if x_vals in ['defensive_stats', 'offensive_stats'] and y_val == 'pts_allowed':
            continue
        if x_vals in ['full-team', 'defensive_stats'] and y_val == 'pts_scored':
            continue
        if x_vals == 'possessions':
            y_data = pull_data.pull_possessions(y_val, update_dbs.mysql_client())
            x_data = pull_data.pull_model_features(y_val, x_vals, update_dbs.mongodb_client)
            x_data = hfa_patch(x_data, update_dbs.mysql_client())             
            train_index = pull_data.pull_train_index(update_dbs.mysql_client())
            x_data = x_data.loc[x_data.index.isin(train_index)]
            y_data = x_data.join(y_data, how = 'inner')['possessions']
            x_data = x_data.join(y_data, how = 'inner')[list(x_data)]
            
        elif x_vals in ['target', 'defensive_stats', 'offensive_stats', 'full-team', 'all']:
            y_data = pull_data.pull_ppp(y_val, update_dbs.mysql_client())
            
            
        if x_vals == 'full-team':
            def_data = pull_data.pull_model_features(y_val, 'defensive_stats', update_dbs.mongodb_client)
            def_data = hfa_patch(def_data, update_dbs.mysql_client())            
            off_data = pull_data.pull_model_features(y_val, 'offensive_stats', update_dbs.mongodb_client)
            off_feats = [i for i in list(off_data) if i not in list(def_data)]
            off_data = off_data[off_feats]
            off_data = hfa_patch(off_data, update_dbs.mysql_client())
            x_data = def_data.join(off_data, how = 'inner')       
            train_index = pull_data.pull_train_index(update_dbs.mysql_client())
            x_data = x_data.loc[x_data.index.isin(train_index)]
            y_data = x_data.join(y_data, how = 'inner')['ppp']
            x_data = x_data.join(y_data, how = 'inner')[list(x_data)]
            off_data = None
            def_data = None

        elif x_vals == 'all':
            def_data = pull_data.pull_model_features(y_val, 'defensive_stats', update_dbs.mongodb_client)
            def_data = hfa_patch(def_data, update_dbs.mysql_client())            
            off_data = pull_data.pull_model_features(y_val, 'offensive_stats', update_dbs.mongodb_client)
            off_feats = [i for i in list(off_data) if i not in list(def_data)]
            off_data = off_data[off_feats]
            off_data = hfa_patch(off_data, update_dbs.mysql_client())
            poss_data = pull_data.pull_model_features(y_val, 'possessions', update_dbs.mongodb_client)
            poss_data = hfa_patch(poss_data, update_dbs.mysql_client())  
            tar_data = pull_data.pull_model_features(y_val, 'target', update_dbs.mongodb_client)
            tar_data = hfa_patch(tar_data, update_dbs.mysql_client())  
            x_data = def_data.join(off_data, how = 'inner')   
            x_data = x_data.join(poss_data, how = 'inner')
            x_data = x_data.join(tar_data, how = 'inner')
            train_index = pull_data.pull_train_index(update_dbs.mysql_client())
            x_data = x_data.loc[x_data.index.isin(train_index)]
            y_data = x_data.join(y_data, how = 'inner')['ppp']
            x_data = x_data.join(y_data, how = 'inner')[list(x_data)]
            def_data = None
            off_data = None
            poss_data = None
            tar_data = None
           
        elif x_vals in ['target', 'defensive_stats', 'offensive_stats']:
            x_data = pull_data.pull_model_features(y_val, x_vals, update_dbs.mongodb_client)
            x_data = hfa_patch(x_data, update_dbs.mysql_client())             
            train_index = pull_data.pull_train_index(update_dbs.mysql_client())
            x_data = x_data.loc[x_data.index.isin(train_index)]
            y_data = x_data.join(y_data, how = 'inner')['ppp']
            x_data = x_data.join(y_data, how = 'inner')[list(x_data)]     
        
        x_data_stable = x_data[feature_lists.optimized_feats[y_val][x_vals]]
        x_data = None
        
        random.seed(86)
        random.shuffle(train_index)
        derived_data = {}
        
        training_partitions = [train_index[i:i + int(len(train_index)/25)] for i in range(0, len(train_index), int(len(train_index)/25))]
        for model_name, model_details in saved_models.stored_models[y_val][x_vals].items():
            print('Loading %s Values'%(model_name))
            derived_data[model_name] = {}
            for part in training_partitions:
                prediction = model_details['model'].fit(x_data_stable[model_details['features']].loc[set(x_data_stable.index) - set(x_data_stable.index[x_data_stable[model_details['features']].index.isin(part)])], y_data.loc[set(y_data.index) - set(y_data.index[y_data.index.isin(part)])]).predict(x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)])
                for pred, idx in zip(prediction, x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)].index):
                    derived_data[model_name][idx] = pred
            print('...Loaded %s Values'%(model_name))

        print('Loading %s Values'%(y_val))            
        save_data = pd.DataFrame.from_dict(derived_data).join(y_data)
        save_data.to_csv(os.path.join(derived_folder, '%s-%s_derived.csv'%(x_vals, y_val)))
        print('...Loaded %s Values'%(y_val))
        
    
    
    