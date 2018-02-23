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
#    try:
#        import pandas as pd
#    except ImportError:
#        for loc in ['/usr/lib/python3.5','/usr/lib/python3.5/plat-x86_64-linux-gnu','/usr/lib/python3.5/lib-dynload','/usr/local/lib/python3.5/dist-packages','/usr/lib/python3/dist-packages']:
#            sys.path.insert(-1, loc)
while cur_path.split('/')[-1] != 'bb_preds':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(-1, os.path.join(cur_path, 'model_conf'))
sys.path.insert(-1, os.path.join(cur_path, 'db_utils'))
sys.path.insert(-1, os.path.join(cur_path, 'model_tuning'))
f = open('keras_model_tuning_line.txt', 'w')
f.write('Starting Keras Analysis...')
f.close()
features_folder = os.path.join(cur_path, 'feature_dumps')
import pull_data
import update_dbs
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

x_vals = 'predictive'
y_val = 'result'
x_data = pull_data.line(update_dbs.mysql_client())
x_cols = list(x_data)
x_cols.remove(y_val)
y_data = x_data[y_val] 
x_data = x_data[x_cols]

y_data = np.ravel(y_data)

seed = 7
np.random.seed(seed)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(29, input_dim=29, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def test_scaler(x, y):
    print('Searching for best scaler...')
    scores = []
    for scale in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
        pipe = Pipeline([('scale',scale), ('clf',KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=64, verbose=1))])
        score = cross_val_score(pipe, x, y, scoring = 'explained_variance' ,cv = KFold(n_splits = 10, random_state = 46))
        scores.append(np.mean(score))
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
scaler = test_scaler(x_data, y_data)
estimators = []
estimators.append(('standardize', scaler))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=64, verbose=1)))
pipeline = Pipeline(estimators)
baseline_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = kfold)
print("Results: %.2f (%.2f) Explained Variance" % (baseline_results.mean(), baseline_results.std()))
results['baseline'] = baseline_results
f = open('keras_model_tuning_line.txt', 'a')
f.write('Baseline: %s, %s.  ' % (baseline_results.mean(), baseline_results.std()))
f.close()
for width in np.linspace(1, 2, 4):
    for depth in range(5):
        def nn_model():
        	# create model
            model = Sequential()
            model.add(Dense(int(29*width), input_dim=29, kernel_initializer='normal', activation='relu'))
            for lay in range(depth):
                model.add(Dropout(.9))
                model.add(Dense(int((float(29*width)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))
        	# Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam')
            return model  
        estimators = []
        estimators.append(('standardize', scaler))
        estimators.append(('mlp', KerasClassifier(build_fn=nn_model, epochs=100, batch_size=32, verbose=1)))
        pipeline = Pipeline(estimators)
        test_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = kfold)
        print("Results: %.2f (%.2f) Explained Variance" % (baseline_results.mean(), baseline_results.std()))
        results['width-%s_depth-%s_model' % (width, depth)] = test_results
        f = open('keras_model_tuning_line.txt', 'a')
        f.write('Width-%s_depth-%s_model: %s, %s.  ' % (width, depth, baseline_results.mean(), baseline_results.std()))
        f.close()



