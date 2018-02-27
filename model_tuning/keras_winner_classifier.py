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
f = open('keras_model_tuning.txt', 'a')
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
x_cols = ['expected_ppp_for',
'expected_effective-field-goal-pct_for',
'50_g_HAspread_allow_defensive-efficiency',
'expected_offensive-rebounding-pct_for',
'-20_game_avg_50_g_Tweight_for_floor-percentage',
'100_g_HAspread_for_personal-fouls-per-game',
'-100_g_HAspread_allow_assist--per--turnover-ratio',
'expected_turnovers-per-possession_for',
'30_g_HAspread_allow_free-throw-rate',
'-100_g_HAspread_for_defensive-efficiency',
'pregame_turnovers-per-possession_for',
'-50_g_HAspread_for_personal-fouls-per-game',
'75_g_HAspread_for_defensive-efficiency',
'100_g_HAspread_for_defensive-efficiency',
'75_g_HAspread_allow_points-per-game',
'-75_g_HAspread_allow_floor-percentage',
'30_g_HAspread_for_floor-percentage',
'expected_ftm-per-100-possessions_for',
'-75_g_HAspread_allow_defensive-efficiency',
'-50_g_HAspread_allow_points-per-game`/`possessions-per-game',
'-50_game_avg_30_g_Tweight_allow_fta-per-fga',
'-50_g_HAspread_for_assist--per--turnover-ratio',
'-10_g_HAspread_allow_ftm-per-100-possessions']
y_data = pull_data.pull_wl(update_dbs.mysql_client())
all_data = data.join(y_data, how = 'inner')
all_data = all_data.reset_index()
all_data = all_data.replace([np.inf, -np.inf], np.nan)
all_data = all_data.replace('NULL', np.nan)
all_data = all_data.dropna(how = 'any')
y_data = all_data[['outcome']]
x_data = all_data[x_cols]


seed = 86
np.random.seed(seed)
import random
random.seed(86)


#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(24, input_dim=24, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
#
#def test_scaler(x, y):
#    print('Searching for best scaler...')
#    scores = []
#    model = baseline_model()
#    for scale in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
#        cv_acc, cv_logloss = [], []
#        for test_idx, train_idx in StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data):
#            model.fit(scale.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=100, batch_size=32, verbose=1)
#            cv_results = model.evaluate(scale.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx]))
#            cv_acc.append(cv_results[1])
#            cv_logloss.append(cv_results[0])        
#        f = open('keras_model_tuning.txt', 'a')
#        f.write('%s: accuracy: %s, logloss: %s  \n' % (scale, np.mean(cv_acc), np.mean(cv_logloss)))
#        f.close()
#        scores.append(np.mean([np.mean(cv_acc), -1*np.mean(cv_logloss)]))
#    f = open('keras_model_tuning.txt', 'a')
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
#    
#results = {}
#kfold = StratifiedKFold(n_splits=10, random_state=86)
#scaler = test_scaler(x_data, y_data) #RobustScaler
scaler = RobustScaler()
#f = open('keras_classifier_tuning.txt', 'a')
#f.write('Scaler: %s  ' % (scaler))
#f.close()
#estimators = []
#estimators.append(('standardize', scaler))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=25, batch_size=64, verbose=1)))
#pipeline = Pipeline(estimators)
#baseline_results = cross_val_score(pipeline, x_data, y_data,cv = kfold)
#print("Results: %.2f (%.2f) accuracy " % (baseline_results.mean(), baseline_results.std()))
#results['baseline'] = baseline_results
#f = open('keras_classifier_tuning.txt', 'a')
#f.write('Baseline: %s, %s.  ' % (baseline_results.mean(), baseline_results.std()))
#f.close()

#depth, width = 3, 3
#def nn_model():
#        	# create model
#            model = Sequential()
#            model.add(Dense(int(24*3), input_dim=24, kernel_initializer='normal', activation='relu'))
##            model.add(Dense(depth, kernel_initializer='normal', activation='relu'))
#            for lay in range(depth):
#                model.add(Dropout(.1))
#                model.add(Dense(int((float(24*width)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
#            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#        	# Compile model
#            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#            return model  
#
#model = nn_model()
#cv_acc = []
#cv_logloss = []
#for test_idx, train_idx in StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data):
#    model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=1000, batch_size=32, verbose=1)
#    cv_results = model.evaluate(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx]))
#    cv_acc.append(cv_results[1])
#    cv_logloss.append(cv_results[0])
#history = model.fit(scaler.fit_transform(x_data), np.ravel(y_data), epochs=50, batch_size=32, verbose=1, validation_split=0.1, shuffle = True)
#
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
  
for width in [.05, .1, .15]:
    for depth in [0]:
#for width in [.001, .0005, .0001]: #[7,6,5,4,3]:
        def nn_model():
            model = Sequential()
            model.add(Dense(48, input_dim=23, kernel_initializer='normal', activation='elu'))
            model.add(Dropout(.15))
            model.add(Dense(32, kernel_initializer='normal', activation='elu'))
            model.add(Dropout(.15))
            model.add(Dense(16, kernel_initializer='normal', activation='elu'))
            model.add(Dropout(.05))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.0005, momentum=width, decay=depth), metrics=['accuracy'])
#            model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=width, rho=0.9, decay=0.0), metrics=['accuracy'])
            return model
        num_epochs = 150
        model = nn_model()
        for test_idx, train_idx in StratifiedShuffleSplit(n_splits=1, test_size=0.90, random_state=86).split(x_data, y_data):
            acc_results = []
            logloss_results = []
            history = model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=num_epochs, batch_size=2**4, verbose=1, validation_data=(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx])), shuffle = True)
#            random.shuffle(test_idx)
#            for val_idx in np.array_split(test_idx,3):
#                cv_results = model.evaluate(scaler.fit_transform(x_data.loc[val_idx]), np.ravel(y_data.loc[val_idx]))
#                acc_results.append(cv_results[1])
#                logloss_results.append(cv_results[0])

            plt.plot(history.history['acc'], linestyle = '-.')
            plt.plot(history.history['val_acc'], linestyle = ':')
#            plt.axhline(y=np.mean(acc_results), color='r', linestyle='-') 
#            for score in acc_results:
#                plt.axhline(y=score, color='r', linestyle=':') 
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test', 'validation'], loc='upper left')
            plt.show()
            print('accuracy graph ^')
            # summarize history for loss
            plt.plot(history.history['loss'], linestyle = '-.')
            plt.plot(history.history['val_loss'], linestyle = ':')
#            plt.axhline(y=cv_results[0], color='r', linestyle='-')
#            for score in logloss_results:
#                plt.axhline(y=score, color='r', linestyle=':')             
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test', 'validation'], loc='upper left')
            plt.show()
            print('log loss graph ^')

            print("Results: best logloss %.4f @ epoch %s, best accuracy %.4f @ epoch %s" % (min(history.history['val_loss']), list(history.history['val_loss']).index(min(history.history['val_loss'])), max(history.history['val_acc']), list(history.history['val_acc']).index(max(history.history['val_acc']))))
            f = open('keras_model_tuning.txt', 'a')
            f.write('BatchSize-%s_epochs-%s_model: \n best logloss %.4f @ epoch %s, best accuracy %.4f @ epoch %s\n' % (width, depth, min(history.history['val_loss']), list(history.history['val_loss']).index(min(history.history['val_loss'])), max(history.history['val_acc']), list(history.history['val_acc']).index(max(history.history['val_acc']))))
            f.close()
