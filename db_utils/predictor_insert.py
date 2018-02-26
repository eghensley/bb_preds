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
import add_derived
import pandas as pd

def update():
    train_index = pull_data.update_idx(update_dbs.mysql_client(), 'predictions')  
    update_df = pd.DataFrame()
    update_df['idx'] = train_index
    update_df = update_df.set_index('idx')
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
    
            if os.path.isfile(os.path.join(model_storage, '%s_%s_regression.pkl' % (x_vals,y_val))):
                print('Loading %s_%s'%(x_vals, y_val))
                model = joblib.load(os.path.join(model_storage, '%s_%s_regression.pkl' % (x_vals,y_val))) 
                preds = model.predict(saved_models.stored_models[x_vals][y_val]['scale'].fit_transform(x_data[list(set(saved_models.stored_models[x_vals][y_val]['features']))]))
                indy_pred = pd.DataFrame()
                indy_pred[x_vals+'_'+y_val] = preds
                indy_pred['idx'] = list(x_data.index)
                indy_pred = indy_pred.set_index('idx')
                update_df = update_df.join(indy_pred, how = 'inner')
                print('Loaded %s_%s'%(x_vals, y_val))
                           
    x_vals = 'predictive'
    y_val = '+pts'
    x_data_stable = pull_data.score(update_dbs.mysql_client())
    x_data_stable = x_data_stable.loc[x_data_stable.index.isin(train_index)]
    x_cols = list(x_data_stable)
    x_cols.remove(y_val)
    y_data = x_data_stable[y_val] 
    x_data = x_data_stable[x_cols]                       
    if os.path.isfile(os.path.join(model_storage, '%s_%s_regression.pkl' % (x_vals,y_val))):
        print('Loading %s_%s'%(x_vals, y_val))
        model = joblib.load(os.path.join(model_storage, '%s_%s_regression.pkl' % (x_vals,y_val))) 
        preds = model.predict(saved_models.stored_models[x_vals][y_val]['scale'].fit_transform(x_data[list(set(saved_models.stored_models[x_vals][y_val]['features']))]))
        indy_pred = pd.DataFrame()
        indy_pred['pred_points'] = preds
        indy_pred['idx'] = list(x_data.index)
        indy_pred = indy_pred.set_index('idx')
        update_df = update_df.join(indy_pred, how = 'inner')
        print('Loaded %s_%s'%(x_vals, y_val))
        
    update_df = update_df[['defense_pace', 'defense_ppp', 'offense_pace', 'offense_ppp', 'pred_points']]
    add_derived.update('predictions', update_df)        