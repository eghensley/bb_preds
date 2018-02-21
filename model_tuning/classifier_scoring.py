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

output_folder = os.path.join(cur_path, 'model_results')
features_folder = os.path.join(cur_path, 'feature_dumps')
derived_folder = os.path.join(cur_path, 'derived_data')

import numpy as np
import pull_data
import update_dbs
import random
import saved_models
from sklearn.model_selection import cross_val_score, StratifiedKFold
train_index = pull_data.pull_train_index(update_dbs.mysql_client())
#cnx = update_dbs.mysql_client()
random.seed(86)
random.shuffle(train_index)
derived_data = {}

x_vals = 'points'
y_val = '+pts'
x_data_stable = pull_data.score(update_dbs.mysql_client())
x_cols = list(x_data_stable)
x_cols.remove('+pts')
x_cols.remove('+possessions')
x_cols.remove('-possessions')
y_data_stable = pull_data.pull_wl(update_dbs.mysql_client())
y_data = x_data_stable.join(y_data_stable, how = 'inner')['outcome']
x_data = x_data_stable.join(y_data_stable, how = 'inner')[x_cols]

random.seed(86)
random.shuffle(train_index)
derived_data = {}

for model_name, model_details in saved_models.stored_models['winner'][y_val].items():
    print('Scoring %s'%(model_name))
    f = open(os.path.join(output_folder, 'winner_classifier_pts+.txt'), 'a')
    f.write('model: %s,' % (model_name))
    f.close()
    scores = cross_val_score(model_details['model'], x_data[model_details['features']], y_data, cv = StratifiedKFold(n_splits = 20, random_state = 1108), n_jobs = -1, scoring = 'accuracy', verbose = 1)
    print('...Scored %s Values'%(model_name))
    f = open(os.path.join(output_folder, 'winner_classifier_pts+.txt'), 'a')
    f.write('accuracy: %s \n' % (np.mean(scores)))
    f.close()
