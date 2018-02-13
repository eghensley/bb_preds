import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
    try:
        import keras
    except ImportError:
        for loc in ['/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload', '/home/eric/ncaa_bb/lib/python3.6/site-packages']:
            sys.path.insert(-1, loc)
        sys.path.insert(-1, '/home/eric/stats_bb')
while cur_path.split('/')[-1] != 'bb_preds':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(-1, os.path.join(cur_path, 'model_conf'))
sys.path.insert(-1, os.path.join(cur_path, 'db_utils'))
sys.path.insert(-1, os.path.join(cur_path, 'model_tuning'))

output_folder = os.path.join(cur_path, 'model_results')
sys.path.insert(-1, output_folder)
f = open('keras_model_tuning.txt', 'w')
f.write('Starting Keras Analysis...')
f.close()

features_folder = os.path.join(cur_path, 'feature_dumps')
input_folder = os.path.join(cur_path, 'derived_data')
import pull_data
import update_dbs
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import pandas as pd

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
    print('Completed HFA Patch')
    return x

def raw_data():
    def_data = pull_data.pull_model_features('pts_scored', 'defensive_stats', update_dbs.mongodb_client)
    def_data = hfa_patch(def_data, update_dbs.mysql_client())            
    off_data = pull_data.pull_model_features('pts_scored', 'offensive_stats', update_dbs.mongodb_client)
    off_feats = [i for i in list(off_data) if i not in list(def_data)]
    off_data = off_data[off_feats]
    off_data = hfa_patch(off_data, update_dbs.mysql_client())
    poss_data = pull_data.pull_model_features('pts_scored', 'possessions', update_dbs.mongodb_client)
    poss_data = hfa_patch(poss_data, update_dbs.mysql_client())  
    tar_data = pull_data.pull_model_features('pts_scored', 'target', update_dbs.mongodb_client)
    tar_data = hfa_patch(tar_data, update_dbs.mysql_client())  
    x_data = def_data.join(off_data, how = 'inner')   
    x_data = x_data.join(poss_data, how = 'inner')
    x_data = x_data.join(tar_data, how = 'inner')
    train_index = pull_data.pull_train_index(update_dbs.mysql_client())
    x_data = x_data.loc[x_data.index.isin(train_index)]
    y_data = pull_data.pull_pts('offensive', update_dbs.mysql_client())
    y_data = x_data.join(y_data, how = 'inner')['pts']
    team_data = x_data.join(y_data, how = 'inner')[list(x_data)]
    def_data = None
    off_data = None
    poss_data = None
    tar_data = None
       
    def_data = pull_data.pull_model_features('pts_allowed', 'defensive_stats', update_dbs.mongodb_client)
    def_data = hfa_patch(def_data, update_dbs.mysql_client())            
    off_data = pull_data.pull_model_features('pts_allowed', 'offensive_stats', update_dbs.mongodb_client)
    off_feats = [i for i in list(off_data) if i not in list(def_data)]
    off_data = off_data[off_feats]
    off_data = hfa_patch(off_data, update_dbs.mysql_client())
    poss_data = pull_data.pull_model_features('pts_allowed', 'possessions', update_dbs.mongodb_client)
    poss_data = hfa_patch(poss_data, update_dbs.mysql_client())  
    tar_data = pull_data.pull_model_features('pts_allowed', 'target', update_dbs.mongodb_client)
    tar_data = hfa_patch(tar_data, update_dbs.mysql_client())  
    x_data = def_data.join(off_data, how = 'inner')   
    x_data = x_data.join(poss_data, how = 'inner')
    x_data = x_data.join(tar_data, how = 'inner')
    train_index = pull_data.pull_train_index(update_dbs.mysql_client())
    opponent_data = x_data.loc[x_data.index.isin(train_index)]
    def_data = None
    off_data = None
    poss_data = None
    tar_data = None
     
    cnx = update_dbs.mysql_client()           
    cursor = cnx.cursor()
    query = 'SELECT * from gamedata;'
    cursor.execute(query)
    switch = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'opponent', 'location'])
    idx_switch = {}
    for t,d,o,l in np.array(switch):
        idx_switch[str(d)+t.replace(' ', '_')] = str(d)+o.replace(' ', '_')
    idx = []
    for idxx in opponent_data.index:
        idx.append(idx_switch[idxx])
    opponent_data['idx'] = idx
    opponent_data = opponent_data.set_index('idx')
    opponent_data *= -1
    opponent_data = opponent_data.rename(columns = {i:'-'+i for i in list(opponent_data)})
    data = opponent_data.join(team_data) 
    data = data.join(y_data, how = 'inner')
    y_data = data[['pts']]
    x_feats = list(data)
    x_feats.remove('pts')
    data = data[x_feats]
    return data, y_data
x_data, y_data = raw_data()    
seed = 7
np.random.seed(seed)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(268, input_dim=268, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def test_scaler(x, y):
    print('Searching for best scaler...')
    scores = []
    for scale in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
        pipe = Pipeline([('scale',scale), ('clf',KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=64, verbose=1))])
        score = cross_val_score(pipe, x, y, scoring = 'explained_variance' ,cv = KFold(n_splits = 10, random_state = 46))
        scores.append(np.mean(score))
    f = open('keras_model_tuning.txt', 'w')
    f.write('Baseline: %s.  ' % (max(scores)))
    f.close()
    if scores.index(max(scores)) == 0:
        print('Using Standard Scaler')
        return StandardScaler()
    elif scores.index(max(scores)) == 1:
        print('Using Min Max Scaler')
        return MinMaxScaler()
    elif scores.index(max(scores)) == 2:
        print('Using Robust Scaler')
        return RobustScaler()
    
results = {}
kfold = KFold(n_splits=10, random_state=seed)
scaler = test_scaler(x_data, y_data) #RobustScaler
scaler = RobustScaler()
f = open('keras_model_tuning.txt', 'a')
f.write('Scaler: %s  ' % (scaler))
f.close()
#estimators = []
#estimators.append(('standardize', scaler))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=25, batch_size=64, verbose=1)))
#pipeline = Pipeline(estimators)
#baseline_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = kfold)
#print("Results: %.2f (%.2f) Explained Variance" % (baseline_results.mean(), baseline_results.std()))
#results['baseline'] = baseline_results
#f = open('keras_model_tuning.txt', 'a')
#f.write('Baseline: %s, %s.  ' % (baseline_results.mean(), baseline_results.std()))
#f.close()
for width in np.linspace(.25, 2, 6):
    for depth in range(0,4):
        def nn_model():
        	# create model
            model = Sequential()
            model.add(Dense(int(268*width), input_dim=268, kernel_initializer='normal', activation='relu'))
            for lay in range(depth):
                model.add(Dropout(.9))
                model.add(Dense(int((float(268*width)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
        	# Compile model
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model  
        estimators = []
        estimators.append(('standardize', scaler))
        estimators.append(('mlp', KerasRegressor(build_fn=nn_model, epochs=75, batch_size=64, verbose=1)))
        pipeline = Pipeline(estimators)
        test_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = kfold)
        print("Results: %.2f (%.2f) Explained Variance" % (test_results.mean(), test_results.std()))
        results['width-%s_depth-%s_model' % (width, depth)] = test_results
        f = open('keras_model_tuning.txt', 'a')
        f.write('Width-%s_depth-%s_model: %s, %s.  ' % (width, depth, test_results.mean(), test_results.std()))
        f.close()



