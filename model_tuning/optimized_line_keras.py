import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
    try:
        import pandas as pd
    except ImportError:
        for loc in ['/usr/lib/python3.5','/usr/lib/python3.5/plat-x86_64-linux-gnu','/usr/lib/python3.5/lib-dynload','/usr/local/lib/python3.5/dist-packages','/usr/lib/python3/dist-packages']:
            sys.path.insert(-1, loc)
    try:
        import keras
    except ImportError:
        for loc in ['/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload', '/home/eric/ncaa_bb/lib/python3.6/site-packages']:
            sys.path.insert(-1, loc)
        sys.path.insert(-1, '/home/eric/stats_bb')
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
f = open('keras_model_tuning_line.txt', 'a')
f.write('Starting Keras Analysis... \n')
f.close()

features_folder = os.path.join(cur_path, 'feature_dumps')
input_folder = os.path.join(cur_path, 'derived_data')
import pull_data
import update_dbs
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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
    opponent_data = x_data.join(tar_data, how = 'inner')
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
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.replace('NULL', np.nan)
    data = data.dropna(how = 'any')
    return data
data = raw_data()
x_feats = ['20_game_avg_30_g_HAweight_allow_fta-per-fga',
'-75_g_HAspread_allow_defensive-efficiency',
'-expected_poss_pg_allowed',
'50_game_avg_50_g_HAweight_for_defensive-rebounding-pct',
'75_g_HAspread_allow_defensive-efficiency',
'-20_game_avg_50_g_Tweight_allow_fta-per-fga',
'50_g_HAspread_allow_assist--per--turnover-ratio',
'50_game_avg_30_g_Tweight_allow_offensive-efficiency',
'-50_game_avg_15_g_HAweight_allow_blocks-per-game',
'pregame_offensive-rebounding-pct_for',
'10_game_avg_30_g_Tweight_for_assists-per-game',
'20_game_avg_30_g_Tweight_allow_assist--per--turnover-ratio',
'-30_game_avg_10_g_HAweight_allow_possessions-per-game',
'-50_game_avg_50_g_Tweight_for_assist--per--turnover-ratio',
'100_g_HAspread_allow_block-pct',
'75_g_HAspread_for_defensive-efficiency',
'1_game_avg_10_g_HAweight_for_points-per-game',
'-50_game_avg_30_g_Tweight_allow_block-pct',
'25_g_HAspread_for_possessions-per-game',
'-5_game_avg_10_g_Tweight_allow_possessions-per-game',
'100_g_HAspread_for_defensive-efficiency',
'-10_game_avg_50_g_Tweight_for_assists-per-game',
'-20_game_avg_15_g_Tweight_allow_extra-chances-per-game',
'pregame_ppp_for',
'-expected_effective-field-goal-pct_allowed',
'-5_game_avg_50_g_HAweight_allow_possessions-per-game',
'-10_g_HAspread_allow_points-per-game`/`possessions-per-game',
'-50_game_avg_15_g_Tweight_allow_blocks-per-game',
'-50_game_avg_50_g_HAweight_for_offensive-rebounding-pct',
'-20_game_avg_50_g_Tweight_for_block-pct', 'pca_line', 'tsvd_line', 'lasso_line', 'lightgbm_line', 'ridge_line']
y_data = pull_data.line_wl(update_dbs.mysql_client())
data = data.join(y_data, how = 'inner')
line_preds = pull_data.line_preds(update_dbs.mysql_client())
data = data.join(line_preds, how = 'inner')
data = data.reset_index()
x_data = data[x_feats]
y_data = data[['line']]
seed = 7
np.random.seed(seed)
data = None
line_preds = None
#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(35, input_dim=35, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
#
#def test_scaler(x, y):
#    print('Searching for best scaler...')
#    scores = []
#    for scale in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
#        pipe = Pipeline([('scale',scale), ('clf',KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=64, verbose=1))])
#        score = cross_val_score(pipe, x, y,cv = StratifiedKFold(n_splits = 3, random_state = 46))
#        scores.append(np.mean(score))
#        f = open('keras_model_tuning_line.txt', 'a')
#        f.write('%s: %s.  ' % (scale, max(scores)))
#        f.close()
#    if scores.index(max(scores)) == 0:
#        print('Using Standard Scaler')
#        return StandardScaler()
#    elif scores.index(max(scores)) == 1:
#        print('Using Min Max Scaler')
#        return MinMaxScaler()
#    elif scores.index(max(scores)) == 2:
#        print('Using Robust Scaler')
#        return RobustScaler()
#    
#scaler = test_scaler(x_data, y_data) #RobustScaler
#f = open('keras_model_tuning_line.txt', 'a')
#f.write('Scaler: %s  \n' % (scaler))
#f.close()
scaler = StandardScaler()
#width, depth = 500, 64
#def nn_model():
#	# create model
#    model = Sequential()
#    model.add(Dense(int(35*2), input_dim=35, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#    return model 
#model = nn_model()
#cv_acc = []
#cv_logloss = []
#for test_idx, train_idx in StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data):
##            pipe = Pipeline([('scale',scaler), ('clf',KerasRegressor(build_fn=nn_model, epochs=10, batch_size=64, verbose=1))])
##            score = cross_val_score(pipe, x_data, y_data,cv = StratifiedKFold(n_splits = 3, random_state = 46))
#
#    model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=1000, batch_size=64, verbose=1)
#    cv_results = model.evaluate(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx]))
#    cv_acc.append(cv_results[1])
#    cv_logloss.append(cv_results[0])
#print("Results: logloss %.2f, Accuracy %.2f " % (np.mean(cv_logloss), np.mean(cv_acc)))
#f = open('keras_model_tuning_line.txt', 'a')
#f.write('Width-%s_depth-%s_model: \n logloss %.4f, Accuracy %.4f \n' % (width, depth, np.mean(cv_logloss), np.mean(cv_acc)))
#f.close()

#
#scaler = StandardScaler()
#for width in [2,2.5]:
#    for depth in [1,2,3]:
##        depth = 0
#        def nn_model():
#        	# create model
#            model = Sequential()
#            model.add(Dense(int(35*width), input_dim=35, kernel_initializer='normal', activation='relu'))
#            for lay in range(depth):
#                model.add(Dropout(.1))
#                model.add(Dense(int((float(35*width)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
#            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#        	# Compile model
#            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#            return model 
#        print(nn_model().summary())
#        estimators = []
#        estimators.append(('standardize', scaler))
#        model = nn_model()
#        cv_acc = []
#        cv_logloss = []
#        for test_idx, train_idx in StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data):
##            pipe = Pipeline([('scale',scaler), ('clf',KerasRegressor(build_fn=nn_model, epochs=10, batch_size=64, verbose=1))])
##            score = cross_val_score(pipe, x_data, y_data,cv = StratifiedKFold(n_splits = 3, random_state = 46))
#
#            model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=100, batch_size=64, verbose=1)
#            cv_results = model.evaluate(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx]))
#            cv_acc.append(cv_results[1])
#            cv_logloss.append(cv_results[0])
#        print("Results: logloss %.2f, Accuracy %.2f " % (np.mean(cv_logloss), np.mean(cv_acc)))
#        f = open('keras_model_tuning_line.txt', 'a')
#        f.write('Width-%s_depth-%s_model: \n logloss %.4f, Accuracy %.4f \n' % (width, depth, np.mean(cv_logloss), np.mean(cv_acc)))
#        f.close()


scaler = StandardScaler()
for width in [.4]:
#    for depth in [1,2,3]:
    depth = 1
    def nn_model():
    	# create model
        model = Sequential()
        model.add(Dense(int(35*2.5), input_dim=35, kernel_initializer='normal', activation='relu'))
        for lay in range(depth):
            model.add(Dropout(width))
            model.add(Dense(int((float(35*2.5)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
            model.add(Dropout(width/2))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    	# Compile model
        model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
        return model 
    print(nn_model().summary())
    estimators = []
    estimators.append(('standardize', scaler))
    model = nn_model()
    cv_acc = []
    cv_logloss = []
    for test_idx, train_idx in StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data):
#            pipe = Pipeline([('scale',scaler), ('clf',KerasRegressor(build_fn=nn_model, epochs=10, batch_size=64, verbose=1))])
#            score = cross_val_score(pipe, x_data, y_data,cv = StratifiedKFold(n_splits = 3, random_state = 46))

        model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=100, batch_size=64, verbose=1)
        cv_results = model.evaluate(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx]))
        cv_acc.append(cv_results[1])
        cv_logloss.append(cv_results[0])
    print("Results: logloss %.2f, Accuracy %.2f " % (np.mean(cv_logloss), np.mean(cv_acc)))
    f = open('keras_model_tuning_line.txt', 'a')
    f.write('Width-%s_depth-%s_model: \n logloss %.4f, Accuracy %.4f \n' % (width, depth, np.mean(cv_logloss), np.mean(cv_acc)))
    f.close()