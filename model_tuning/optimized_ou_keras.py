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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt

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
x_feats = ['expected_effective-field-goal-pct_for',
'10_game_avg_30_g_Tweight_for_true-shooting-percentage',
'-75_g_HAspread_allow_defensive-efficiency',
'100_g_HAspread_allow_block-pct',
'10_game_avg_10_g_Tweight_for_possessions-per-game',
'-expected_effective-field-goal-pct_allowed',
'75_g_HAspread_for_floor-percentage',
'-10_game_avg_15_g_HAweight_for_defensive-efficiency',
'-30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game',
'30_game_avg_5_g_Tweight_for_possessions-per-game',
'25_g_HAspread_for_points-per-game',
'-30_game_avg_50_g_HAweight_allow_points-per-game',
'-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game',
'-1_game_avg_10_g_Tweight_allow_possessions-per-game',
'-100_g_HAspread_for_points-per-game',
'-50_game_avg_50_g_HAweight_for_assists-per-game',
'25_g_HAspread_for_possessions-per-game',
'10_game_avg_50_g_HAweight_for_blocks-per-game',
'-50_game_avg_50_g_HAweight_allow_ftm-per-100-possessions',
'20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game',
'-20_game_avg_50_g_Tweight_for_floor-percentage',
'20_game_avg_10_g_HAweight_for_possessions-per-game',
'-20_game_avg_50_g_Tweight_allow_points-per-game',
'-100_g_HAspread_allow_assist--per--turnover-ratio',
'-10_game_avg_10_g_HAweight_allow_points-per-game',
'75_g_HAspread_allow_percent-of-points-from-3-pointers',
'-15_g_HAspread_allow_block-pct',
'-20_game_avg_25_g_Tweight_allow_possessions-per-game',
'-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game',
'-20_game_avg_50_g_HAweight_allow_defensive-efficiency',
'50_game_avg_50_g_HAweight_for_assists-per-game',
'-30_game_avg_25_g_Tweight_allow_points-per-game',
'-25_g_HAspread_allow_possessions-per-game', 'pca_ou', 'tsvd_ou', 'lasso_ou', 'lightgbm_ou', 'ridge_ou']
y_data = pull_data.ou_wl(update_dbs.mysql_client())
data = data.join(y_data, how = 'inner')
line_preds = pull_data.ou_preds(update_dbs.mysql_client())
data = data.join(line_preds, how = 'inner')
data = data.reset_index()
x_data = data[x_feats]
y_data = data[['ou']]
seed = 86
np.random.seed(seed)
import random
random.seed(86)
data = None
line_preds = None
#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
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
#    f = open('keras_model_tuning.txt', 'w')
#    f.write('Baseline: %s.  ' % (max(scores)))
#    f.close()
#    if scores.index(max(scores)) == 0:
#        print('Using Standard Scaler')
#        return StandardScaler()
#    elif scores.index(max(scores)) == 1:
#        print('Using Min Max Scaler')
#        return MinMaxScaler()
#    elif scores.index(max(scores)) == 2:
#        print('Using Robust Scaler')
#        return RobustScaler()
    
#scaler = test_scaler(x_data, y_data) #RobustScaler
#f = open('keras_model_tuning_ou.txt', 'a')
#f.write('Scaler: %s  \n' % (scaler))
#f.close()

scaler = StandardScaler()
def nn_model():
    model = Sequential()
    model.add(Dense(76, input_dim=38, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(.45))
    model.add(Dense(50, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(.45))
    model.add(Dense(25, kernel_initializer='normal', activation='elu'))
    model.add(Dropout(.15))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model 

for width in [7,6,5,4,3]:
        num_epochs = 1000
        model = nn_model()
        for test_idx, train_idx in StratifiedShuffleSplit(n_splits=1, test_size=0.90, random_state=86).split(x_data, y_data):
            acc_results = []
            logloss_results = []
            history = model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=num_epochs, batch_size=2**width, verbose=1, validation_data=(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx])), shuffle = True)
            plt.plot(history.history['acc'], linestyle = '-.')
            plt.plot(history.history['val_acc'], linestyle = ':')
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test', 'validation'], loc='upper left')
            plt.show()
            print('accuracy graph ^')
            plt.plot(history.history['loss'], linestyle = '-.')
            plt.plot(history.history['val_loss'], linestyle = ':')            
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test', 'validation'], loc='upper left')
            plt.show()
            print('log loss graph ^')

            print("Results: best logloss %.4f @ epoch %s, best accuracy %.4f @ epoch %s" % (min(history.history['val_loss']), list(history.history['val_loss']).index(min(history.history['val_loss'])), max(history.history['val_acc']), list(history.history['val_acc']).index(max(history.history['val_acc']))))
            f = open('keras_model_tuning_ou.txt', 'a')
            f.write('BatchSize-%s_epochs-%s_model: \n best logloss %.4f @ epoch %s, best accuracy %.4f @ epoch %s\n' % (2**width, num_epochs, min(history.history['val_loss']), list(history.history['val_loss']).index(min(history.history['val_loss'])), max(history.history['val_acc']), list(history.history['val_acc']).index(max(history.history['val_acc']))))
            f.close()
            