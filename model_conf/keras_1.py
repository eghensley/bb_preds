import os, sys
os.getcwd()
ide = False
try:
    cur_path = os.path.abspath(__file__)
except NameError:
    ide = True

if ide:
#    import scipy
    for loc in ['/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload', '/home/eric/ncaa_bb/lib/python3.6/site-packages']:
        sys.path.insert(-1, loc)
    sys.path.insert(-1, '/home/eric/stats_bb')

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import update_dbs
import pull_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(52, input_dim=52, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model




od, sa, mongodb_client, mysql_client = 'offensive',  'pts_scored', update_dbs.mongodb_client, update_dbs.mysql_client()
y_data = pull_data.pull_ppp(od, mysql_client)
x_data = pull_data.pull_model_features(od, sa, mongodb_client)
train_index = pull_data.pull_train_index(mysql_client)

x_data = x_data.loc[x_data.index.isin(train_index)]
y_data = x_data.join(y_data, how = 'inner')['ppp']
x_data = x_data.join(y_data, how = 'inner')[list(x_data)]
x_feats = list(x_data)
results = {}
kfold = KFold(n_splits=10, random_state=seed)

estimators = []
estimators.append(('standardize', MinMaxScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=32, verbose=1)))
pipeline = Pipeline(estimators)
baseline_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = kfold)
print("Results: %.2f (%.2f) Explained Variance" % (baseline_results.mean(), baseline_results.std()))
results['baseline'] = baseline_results
f = open('model_tuning.txt', 'a')
f.write('Baseline: %s, %s.  ' % (baseline_results.mean(), baseline_results.std()))
f.close()
for width in np.linspace(1, 3, 6):
    for depth in range(7):
        def nn_model():
        	# create model
            model = Sequential()
            model.add(Dense(int(52*width), input_dim=52, kernel_initializer='normal', activation='relu'))
            for lay in range(depth):
                model.add(Dropout(.9))
                model.add(Dense(int((float(52*width)/(depth+1))*(depth-lay)), kernel_initializer='normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='normal'))
        	# Compile model
            model.compile(loss='mean_squared_error', optimizer='adam')
            return model  
        estimators = []
        estimators.append(('standardize', MinMaxScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=nn_model, epochs=100, batch_size=32, verbose=1)))
        pipeline = Pipeline(estimators)
        test_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = kfold)
        print("Results: %.2f (%.2f) Explained Variance" % (baseline_results.mean(), baseline_results.std()))
        results['width-%s_depth-%s_model' % (width, depth)] = test_results
        f = open('model_tuning.txt', 'a')
        f.write('Width-%s_depth-%s_model: %s, %s.  ' % (width, depth, baseline_results.mean(), baseline_results.std()))
        f.close()
        




def random_model():
	# create model
    model = Sequential()
    model.add(Dense(100, input_dim=52, kernel_initializer='normal', activation='relu'))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(52, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.8))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def random_model():
	# create model
    model = Sequential()
    model.add(Dense(100, input_dim=52, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.8))
    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
    model.add(Dense(52, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.8))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=64, verbose=1)



#scaler = test_scaler(x_data, y_data, estimator)



estimators = []
estimators.append(('standardize', MinMaxScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=random_model, epochs=100, batch_size=16, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
random_results = cross_val_score(pipeline, x_data, y_data, scoring = 'explained_variance' ,cv = KFold(n_splits = 10, random_state = 46))
print("Results: %.2f (%.2f) MSE" % (random_results.mean(), random_results.std()))




kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_data, y_data, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))






















# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=86)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

for j in N_FEATURES_OPTIONS:
    x = data[x_feat[:j]]
    allaccuracy = []
    allroc = []
    alllogloss = []
    allf1 = []
    allnewlogloss = []
    score = None
    acc = []
    log_loss = []
    trainx, trainy, testx, testy, trainj, testj, model, pred = None, None, None, None, None, None, None, None
    for i in range(1,20):
        trainx, testx, trainy, testy, trainj, testj = train_test_split(x, y, juice, train_size = .9, test_size = .1, random_state = i, stratify = y)
        trainx = StandardScaler().fit_transform(trainx)
        testx = StandardScaler().fit_transform(testx)
#        trainx = PCA(random_state = 1108, n_components=j,whiten=True,svd_solver='full').fit_transform(trainx)
#        testx = PCA(random_state = 1108, n_components=j,whiten=True,svd_solver='full').fit_transform(testx)
        trainj = np.array(trainj)
        testj = np.array(testj)
        y_juice = []
        for every in range(0, len(np.array(testj))):
            ju = [testj[every]/200, -testj[every]/200]
            y_juice.append(ju)
        model = Sequential()
        model.add(Dense(j, input_shape=(j,), activation = 'relu'))
        model.add(Dropout(.8))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(trainx, trainy, batch_size = 32, epochs = 100, verbose = 2)
        score = model.evaluate(testx, testy, batch_size = 32, verbose = 2)
        acc.append(score[1])
        log_loss.append(score[0])
    allaccuracy.append(np.mean(acc))
    alllogloss.append(np.mean(log_loss))

   
