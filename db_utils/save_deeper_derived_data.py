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
import saved_models
import random
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import LocallyLinearEmbedding, Isomap
#cnx = update_dbs.mysql_client()
#od = 'offensive'

train_index = pull_data.pull_train_index(update_dbs.mysql_client())

random.seed(86)
random.shuffle(train_index)

for x_vals in ['line', 'ou']:
    derived_data = {}
    y_val = 'result'
    x_data_stable = pd.read_csv(os.path.join(derived_folder, '%s_data.csv'%(x_vals)))
    x_data_stable = x_data_stable.set_index('Unnamed: 0')
    y_data = x_data_stable[[y_val]]
    x_cols = list(x_data_stable)
    x_cols.remove(y_val)
    x_data_stable = x_data_stable[x_cols]
    training_partitions = [train_index[i:i + int(len(train_index)/50)] for i in range(0, len(train_index), int(len(train_index)/50))]
    for model_name, model_details in saved_models.stored_models[y_val][x_vals].items():
        print('Loading %s Values'%(model_name))
        derived_data[model_name] = {}
        for part in training_partitions:
            prediction = model_details['model'].fit(x_data_stable[model_details['features']].loc[set(x_data_stable.index) - set(x_data_stable.index[x_data_stable[model_details['features']].index.isin(part)])], np.ravel(y_data.loc[set(y_data.index) - set(y_data.index[y_data.index.isin(part)])])).predict(x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)])
            for pred, idx in zip(prediction, x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)].index):
                derived_data[model_name][idx] = pred
        print('...Loaded %s Values'%(model_name))

    for model_name, model in [('Isomap', Isomap(n_components = 1)), ('LLE', LocallyLinearEmbedding(n_components = 1, random_state = 1108)), ('PCA', PCA(n_components = 1, random_state = 1108)), ("TSVD", TruncatedSVD(n_components = 1, random_state = 1108))]:
        try:
            print('Loading %s Values'%(model_name))
            derived_data[model_name] = {}
            for part in training_partitions:
                prediction = model.fit(x_data_stable.loc[set(x_data_stable.index) - set(x_data_stable.index.isin(part))]).fit_transform(x_data_stable.loc[x_data_stable.index.isin(part)])
                for pred, idx in zip(prediction, x_data_stable[model_details['features']].loc[x_data_stable[model_details['features']].index.isin(part)].index):
                    derived_data[model_name][idx] = pred[0]
            print('...Loaded %s Values'%(model_name))
        except:
            derived_data[model_name] = None
            pass

    derived_data = pd.DataFrame.from_dict(derived_data)
    derived_data.to_csv(os.path.join(derived_folder, '%s.csv'%(x_vals)))

    break



        
for x_vals in ['offense', 'defense']:  
    for y_val in ['pace', 'ppp']:
        if y_val == 'ppp':
            data = pull_data.ppp(update_dbs.mysql_client(), x_vals)
            y_data = data[[y_val]]
            x_feats = list(data)
            x_feats.remove(y_val)
            x_data = data[x_feats]
        elif y_val == 'pace':
            data = pull_data.pace(update_dbs.mysql_client(), x_vals)
            y_data = data[['possessions']]
            x_feats = list(data)
            x_feats.remove('possessions')
            x_data = data[x_feats]            
            
        x_data_stable = x_data
        x_data = None
        training_partitions = [train_index[i:i + int(len(train_index)/50)] for i in range(0, len(train_index), int(len(train_index)/50))]
        model = saved_models.stored_models[x_vals][y_val]['model']
        features = saved_models.stored_models[x_vals][y_val]['features']
        print('Loading %s Values'%(x_vals+'_'+y_val))
        derived_data[x_vals+'_'+y_val] = {}
        for part in training_partitions:
            prediction = model.fit(x_data_stable[features].loc[set(x_data_stable.index) - set(x_data_stable.index[x_data_stable[features].index.isin(part)])], np.ravel(y_data.loc[set(y_data.index) - set(y_data.index[y_data.index.isin(part)])])).predict(x_data_stable[features].loc[x_data_stable[features].index.isin(part)])
            for pred, idx in zip(prediction, x_data_stable[features].loc[x_data_stable[features].index.isin(part)].index):
                derived_data[x_vals+'_'+y_val][idx] = pred
        print('...Loaded %s Values'%(x_vals+'_'+y_val))


derived_data_1 = pd.DataFrame.from_dict(derived_data)
#points = pull_data.pull_pts('offensive', update_dbs.mysql_client())
#allowed = pull_data.pull_pts('defensive', update_dbs.mysql_client())
derived_data = derived_data.join(points)
    
derived_data.to_csv(os.path.join(derived_folder, 'points.csv'))