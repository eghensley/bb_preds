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
model_storage = os.path.join(cur_path, 'saved_models')

import pull_data
import update_dbs
import saved_models
from sklearn.externals import joblib
import numpy as np

train_index = pull_data.pull_train_index(update_dbs.mysql_client())

for x_vals in ['offense', 'defense']:  
    for y_val in ['pace', 'ppp']:
        if y_val == 'ppp':
            data = pull_data.ppp(update_dbs.mysql_client(), x_vals)
            y_data = data[[y_val]]
            x_feats = list(data)
            x_feats.remove(y_val)
            x_data = data[x_feats]
            data = x_data.join(y_data, how = 'inner')
            data = data.loc[data.index.isin(train_index)]
            x_data = data[x_feats]                       
            y_data = data[[y_val]]
        elif y_val == 'pace':
            data = pull_data.pace(update_dbs.mysql_client(), x_vals)
            y_data = data[['possessions']]
            x_feats = list(data)
            x_feats.remove('possessions')
            x_data = data[x_feats]                       
            data = x_data.join(y_data, how = 'inner')
            data = data.loc[data.index.isin(train_index)]
            x_data = data[x_feats]                       
            y_data = data[['possessions']]
        
        if not os.path.isfile(os.path.join(model_storage, '%s_%s_regression.pkl' % (x_vals,y_val))):
            print('Loading %s_%s'%(x_vals, y_val))
            
            model = saved_models.stored_models[x_vals][y_val]['model']
            scale = saved_models.stored_models[x_vals][y_val]['scale']
            
            scale.fit(x_data[saved_models.stored_models[x_vals][y_val]['features']])
            joblib.dump(scale,os.path.join(model_storage, '%s_%s_regression_scaler.pkl' % (y_val, x_vals)))             
            model.fit(scale.transform(x_data[saved_models.stored_models[x_vals][y_val]['features']]), np.ravel(y_data))
            joblib.dump(model,os.path.join(model_storage, '%s_%s_regression_model.pkl' % (y_val, x_vals))) 

            print('Stored %s_%s'%(x_vals, y_val))
            
x_vals = 'predictive'
y_val = '+pts'
x_data_stable = pull_data.score(update_dbs.mysql_client())
x_data_stable = x_data_stable.loc[x_data_stable.index.isin(train_index)]
x_cols = list(x_data_stable)
x_cols.remove(y_val)
y_data = x_data_stable[y_val] 
x_data = x_data_stable[x_cols]                       
if not os.path.isfile(os.path.join(model_storage, '%s_%s_regression.pkl' % (x_vals,y_val))):
            print('Loading %s_%s'%(x_vals, y_val))
    
            model = saved_models.stored_models[x_vals][y_val]['model']
            scale = saved_models.stored_models[x_vals][y_val]['scale']
            
            scale.fit(x_data[saved_models.stored_models[x_vals][y_val]['features']])
            joblib.dump(scale,os.path.join(model_storage, '%s_%s_regression_scaler.pkl' % (y_val, x_vals)))             
            model.fit(scale.transform(x_data[saved_models.stored_models[x_vals][y_val]['features']]), np.ravel(y_data))
            joblib.dump(model,os.path.join(model_storage, '%s_%s_regression_model.pkl' % (y_val, x_vals))) 

            print('Stored %s_%s'%(x_vals, y_val))