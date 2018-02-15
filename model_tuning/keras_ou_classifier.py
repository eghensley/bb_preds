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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
import pandas as pd
import random

train_index = pull_data.pull_train_index(update_dbs.mysql_client())
x_vals = 'points'
y_val = 'line'
x_data_stable = pull_data.score(update_dbs.mysql_client())
line_preds = pull_data.ou_preds(update_dbs.mysql_client())
x_data_stable = x_data_stable.join(line_preds, how = 'inner')
x_cols = list(x_data_stable)
x_cols.remove('+pts')
x_cols.remove('+possessions')
x_cols.remove('-possessions')
y_data = pull_data.ou_wl(update_dbs.mysql_client())
all_data = x_data_stable.join(y_data, how = 'inner')
all_data = all_data.reset_index()
y_data = all_data[['ou']]
x_data = all_data[x_cols]

#random.shuffle(train_index)
#one,two = StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data)
#holdout_index = all_data.index[:int(len(all_data.index)/20)]

#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(29, input_dim=29, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model

#def test_scaler(x, y):
#    print('Searching for best scaler...')
#    scores = []
#    for scale in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
#        pipe = Pipeline([('scale',scale), ('clf',KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=32, verbose=1))])
#        score = cross_val_score(pipe, x, y,cv = kfold)
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
    
results = {}
kfold = StratifiedKFold(n_splits=10, random_state=86)
#scaler = test_scaler(x_data, y_data) #RobustScaler
scaler = MinMaxScaler()
#f = open('keras_classifier_tuning.txt', 'a')
#f.write('Scaler: %s  ' % (scaler))
#f.close()
#estimators = []
#estimators.append(('standardize', scaler))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=25, batch_size=32, verbose=1)))
#pipeline = Pipeline(estimators)
#baseline_results = cross_val_score(pipeline, x_data, y_data, scoring = 'accuracy' ,cv = kfold)
#print("Results: %.2f (%.2f) accuracy " % (baseline_results.mean(), baseline_results.std()))
#results['baseline'] = baseline_results
#f = open('keras_ou_classifier_tuning.txt', 'a')
#f.write('Baseline: %s, %s.  ' % (baseline_results.mean(), baseline_results.std()))
#f.close()
for width in np.linspace(.5, 1.5, 5):
    for depth in range(1,4):
        def nn_model():
        	# create model
            model = Sequential()
            model.add(Dense(int(29*width), input_dim=29, kernel_initializer='normal', activation='relu'))
            for lay in range(depth):
                model.add(Dropout(.9))
                model.add(Dense(int((float(29*width)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        	# Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model  
        estimators = []
        estimators.append(('standardize', scaler))
        model = nn_model()
        cv_acc = []
        cv_logloss = []
        for test_idx, train_idx in StratifiedShuffleSplit(n_splits=3, test_size=0.9, random_state=86).split(x_data, y_data):
#            np.mean(y_data.loc[train_idx])
#            np.mean(y_data.loc[test_idx])
#        KerasClassifier(build_fn=nn_model, epochs=5, batch_size=32, verbose=1)
            model.fit(scaler.fit_transform(x_data.loc[train_idx]), np.ravel(y_data.loc[train_idx]), epochs=200, batch_size=64, verbose=1)
            cv_results = model.evaluate(scaler.fit_transform(x_data.loc[test_idx]), np.ravel(y_data.loc[test_idx]))
            cv_acc.append(cv_results[1])
            cv_logloss.append(cv_results[0])
        print("Results: logloss %.2f, Accuracy %.2f " % (np.mean(cv_logloss), np.mean(cv_acc)))
        f = open('keras_ou_classifier_tuning.txt', 'a')
        f.write('Width-%s_depth-%s_model: \n logloss %.4f, Accuracy %.4f \n' % (width, depth, np.mean(cv_logloss), np.mean(cv_acc)))
        f.close()