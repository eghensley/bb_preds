import os, sys

try:
    import lightgbm as lgb
except ImportError:
    sys.path.insert(-1, "/home/eric/LightGBM/python-package")
    import lightgbm as lgb
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier


stored_models = {
        'offense':{
            'pace': {
                'features': ['lasso_possessions', 'lightgbm_possessions', 'linsvm_possessions'],
                'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.019012104600893226))]),
            },    
            'ppp': {
                'features': ['linsvm_all', 'ridge_all', 'lightgbm_target', 'ridge_target', 'lightgbm_team', 'linsvm_team', 'rest', 'lasso_target', 'linsvm_target'],
                'model': Pipeline([('scale',RobustScaler()), ('clf',LinearSVR(random_state = 1108, C = 0.215329117725, epsilon=0))]),
            },                 
        },
        'defense':{
            'pace': {
                'features': ['lasso_possessions', 'lightgbm_possessions', 'ridge_possessions'],
                'model': Pipeline([('scale',MinMaxScaler()), ('clf',LinearSVR(random_state = 1108, C = 0.6652011887133216, epsilon=0))]),
            },  
            'ppp': {
                'features': ['lightgbm_all', 'ridge_all', 'lasso_target', 'lightgbm_target', 'lasso_team', 'lightgbm_team', 'linsvm_team', 'ridge_team', 'rest'],
                'model': Pipeline([('scale',StandardScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 0.00101115979472))]),
            },  
        },
        'points':{  # +pts regression
                    '+pts': {
                        'features': ['-lightgbm_all', '+lasso_possessions', '+linsvm_target', '+ridge_target', '+lightgbm_target', '+linsvm_team', '+lightgbm_team', '+ridge_all', '+linsvm_all', '+lightgbm_possessions', '-ridge_possessions', '-lasso_possessions', '-lightgbm_target', '-lasso_target', '-ridge_team', '-linsvm_team', '-lightgbm_team', '-lasso_team', '-ridge_all', '-lightgbm_possessions', '+linsvm_possessions', '-rest', '+lasso_target', '+rest'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',LinearSVR(random_state = 1108, C = 11.80012956536623, epsilon=0))]),
                        },    
        },
        'winner':{
                '+pts':{
                    'lightgbc': {
                        'features': ['+rest', '-ridge_possessions', '+linsvm_all', '-possessions', '+linsvm_team', '+lightgbm_target', '-lasso_target', '+lasso_target', '+linsvm_possessions', '-lightgbm_all', '-ridge_team', '-lasso_team', '+possessions', '-lasso_possessions', '+lasso_possessions', '-lightgbm_target', '+lightgbm_team', '+ridge_target', '-linsvm_team', '+linsvm_target', '+ridge_all', '+pts', '-lightgbm_team', '-ridge_all', '-rest', '+lightgbm_possessions'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMClassifier(random_state = 1108, n_estimators = 2000, colsample_bytree = 0.711915990666168, min_child_samples = 146, num_leaves = 22, subsample = 0.449024728208082, max_bin = 1143, learning_rate = 0.005))]),
                        },
                    'log': {
                        'features': ['-lightgbm_all', '+pts', '+linsvm_target', '+ridge_target', '+lightgbm_target', '+linsvm_team', '+lightgbm_team', '+ridge_all', '+linsvm_all', '-linsvm_team', '-ridge_all', '-lasso_team', '-lightgbm_team', '-ridge_team', '-lightgbm_target', '-lasso_target', '-lasso_possessions', '-lightgbm_possessions', '-ridge_possessions', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-rest', '+rest', '-possessions', '+possessions'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', LogisticRegression(random_state = 1108, C = 975.204044353, solver = "liblinear"))]),
                        },
                    'linsvc': {
                        'features': ['-lightgbm_all', '+pts', '+linsvm_target', '+ridge_target', '+lightgbm_target', '+linsvm_team', '+lightgbm_team', '+ridge_all', '+linsvm_all', '-linsvm_team', '-ridge_all', '-lasso_team', '-lightgbm_team', '-ridge_team', '-lightgbm_target', '-lasso_target', '-lasso_possessions', '-lightgbm_possessions', '-ridge_possessions', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-rest', '+rest', '-possessions', '+possessions'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',LinearSVC(random_state = 1108, C = 0.261256486245))]),
                        }, 
                    'rbfsvc': {
                        'features': ['-lightgbm_all', '+pts', '+linsvm_target', '+ridge_target', '+lightgbm_target', '+linsvm_team', '+lightgbm_team', '+ridge_all', '+linsvm_all', '-linsvm_team', '-ridge_all', '-lasso_team', '-lightgbm_team', '-ridge_team', '-lightgbm_target', '-lasso_target', '-lasso_possessions', '-lightgbm_possessions', '-ridge_possessions', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-rest', '+rest', '-possessions'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',SVC(random_state = 1108, C = 6.24660175945, gamma = 0.0884396941322, solver = 'rbf'))]),
                        }, 
                    'polysvc': {
                        'features': ['-lightgbm_all', '+pts', '+linsvm_target', '+ridge_target', '+lightgbm_target', '+linsvm_team', '+lightgbm_team', '+ridge_all', '+linsvm_all', '-linsvm_team', '-ridge_all', '-lasso_team', '-lightgbm_team', '-ridge_team', '-lightgbm_target', '-lasso_target', '-lasso_possessions', '-lightgbm_possessions', '-ridge_possessions', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-rest', '+rest', '-possessions'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',SVC(random_state = 1108, C = 9.39273679717, gamma = 0.289168829661, solver = 'poly', degree = 2))]),
                        }, 
                    },
                'raw':{
                    'knn': {
                        'features': ['expected_ppp_for', '-50_g_HAspread_allow_points-per-game`/`possessions-per-game', '-75_g_HAspread_allow_defensive-efficiency', 'expected_ftm-per-100-possessions_for', '30_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_floor-percentage', '75_g_HAspread_allow_points-per-game', '100_g_HAspread_for_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '-50_g_HAspread_for_assist--per--turnover-ratio', '-50_g_HAspread_for_personal-fouls-per-game', '-100_g_HAspread_for_defensive-efficiency', '30_g_HAspread_allow_free-throw-rate', '-100_g_HAspread_allow_assist--per--turnover-ratio', '100_g_HAspread_for_personal-fouls-per-game', 'expected_offensive-rebounding-pct_for', '50_g_HAspread_allow_defensive-efficiency', 'expected_effective-field-goal-pct_for', '-10_g_HAspread_allow_ftm-per-100-possessions', 'expected_turnovers-per-possession_for', 'pregame_turnovers-per-possession_for', '-50_game_avg_30_g_Tweight_allow_fta-per-fga'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', KNeighborsClassifier(n_neighbors = 166, leaf_size = 14))]),
                        },                            
                    'log': {
                        'features': ['expected_ppp_for', '-50_g_HAspread_allow_points-per-game`/`possessions-per-game', '-75_g_HAspread_allow_defensive-efficiency', 'expected_ftm-per-100-possessions_for', '30_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_floor-percentage', '75_g_HAspread_allow_points-per-game', '100_g_HAspread_for_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '-50_g_HAspread_for_assist--per--turnover-ratio', '-50_g_HAspread_for_personal-fouls-per-game', '-100_g_HAspread_for_defensive-efficiency', '30_g_HAspread_allow_free-throw-rate', '-100_g_HAspread_allow_assist--per--turnover-ratio', '100_g_HAspread_for_personal-fouls-per-game', 'expected_offensive-rebounding-pct_for', '50_g_HAspread_allow_defensive-efficiency', 'expected_effective-field-goal-pct_for', '-10_g_HAspread_allow_ftm-per-100-possessions', 'expected_turnovers-per-possession_for', 'pregame_turnovers-per-possession_for'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', LogisticRegression(random_state = 1108, C = 1.007409512406823, solver = "lbfgs"))]),
                        }, 
                },
        },
        'ou':{
                '+pts':{      
                    'log': {
                        'features': ['+linsvm_team', '+lightgbm_team', '-lasso_team', '-ridge_team', '-linsvm_team', '-ridge_all', '-lightgbm_team', '-lightgbm_target', '-lightgbm_all', '-lasso_target', '+lightgbm_target', '+linsvm_target', '+ridge_target', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-ridge_possessions', '-lasso_possessions', '-lightgbm_possessions', '+ridge_all', '+linsvm_all', '+rest'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', LogisticRegression(random_state = 1108, C = 15.56187413425767, solver = "newton-cg"))]),
                        },                                              
                },
                'raw':{
                    'lightgbc': {
                        'features': ['10_game_avg_10_g_Tweight_for_possessions-per-game', 'ridge_ou', '-1_game_avg_10_g_Tweight_allow_possessions-per-game', '-expected_effective-field-goal-pct_allowed', 'lasso_ou', '-30_game_avg_25_g_Tweight_allow_points-per-game', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '25_g_HAspread_for_possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '-30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', 'expected_effective-field-goal-pct_for', 'lightgbm_ou', 'tsvd_ou', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-50_game_avg_50_g_HAweight_for_assists-per-game', '10_game_avg_50_g_HAweight_for_blocks-per-game', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '75_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_defensive-efficiency', 'pca_ou', '-50_game_avg_50_g_HAweight_allow_ftm-per-100-possessions', '-25_g_HAspread_allow_possessions-per-game', '100_g_HAspread_allow_block-pct', '-30_game_avg_50_g_HAweight_allow_points-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-15_g_HAspread_allow_block-pct', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-100_g_HAspread_for_points-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '25_g_HAspread_for_points-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, colsample_bytree = 0.785582982952984, min_child_samples = 198, num_leaves = 11, subsample = 0.633073504349269, max_bin = 1359, learning_rate = 0.02))]),
                        },
                    'knn': {
                        'features': ['expected_effective-field-goal-pct_for', '-expected_effective-field-goal-pct_allowed', '20_game_avg_10_g_HAweight_for_possessions-per-game', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-20_game_avg_25_g_Tweight_allow_possessions-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '10_game_avg_10_g_Tweight_for_possessions-per-game', '25_g_HAspread_for_points-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio', '-75_g_HAspread_allow_defensive-efficiency', '-30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '75_g_HAspread_for_floor-percentage', 'tsvd_ou', '-50_game_avg_50_g_HAweight_allow_ftm-per-100-possessions', '-15_g_HAspread_allow_block-pct', '-50_game_avg_50_g_HAweight_for_assists-per-game', '-100_g_HAspread_for_points-per-game', '-30_game_avg_25_g_Tweight_allow_points-per-game', '25_g_HAspread_for_possessions-per-game', 'pca_ou', '-25_g_HAspread_allow_possessions-per-game', 'lightgbm_ou', '10_game_avg_50_g_HAweight_for_blocks-per-game', '-30_game_avg_50_g_HAweight_allow_points-per-game', '100_g_HAspread_allow_block-pct', 'ridge_ou', 'lasso_ou'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', KNeighborsClassifier(n_neighbors = 88, leaf_size = 30))]),
                        },                            
                    'log': {
                        'features': ['expected_effective-field-goal-pct_for', '-expected_effective-field-goal-pct_allowed', '20_game_avg_10_g_HAweight_for_possessions-per-game', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-20_game_avg_25_g_Tweight_allow_possessions-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '10_game_avg_10_g_Tweight_for_possessions-per-game', '25_g_HAspread_for_points-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', LogisticRegression(random_state = 1108, C = 0.04795626933077879, solver = "liblinear"))]),
                        }, 
                    'linsvc': {
                        'features': ['expected_effective-field-goal-pct_for', '-expected_effective-field-goal-pct_allowed', '20_game_avg_10_g_HAweight_for_possessions-per-game', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-20_game_avg_25_g_Tweight_allow_possessions-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '10_game_avg_10_g_Tweight_for_possessions-per-game', '25_g_HAspread_for_points-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio', '-75_g_HAspread_allow_defensive-efficiency', '-30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '75_g_HAspread_for_floor-percentage', 'tsvd_ou', '-50_game_avg_50_g_HAweight_allow_ftm-per-100-possessions', '-15_g_HAspread_allow_block-pct', '-50_game_avg_50_g_HAweight_for_assists-per-game', '-100_g_HAspread_for_points-per-game', '-30_game_avg_25_g_Tweight_allow_points-per-game', '25_g_HAspread_for_possessions-per-game', 'pca_ou', '-25_g_HAspread_allow_possessions-per-game', 'lightgbm_ou', '10_game_avg_50_g_HAweight_for_blocks-per-game', '-30_game_avg_50_g_HAweight_allow_points-per-game', '100_g_HAspread_allow_block-pct', 'ridge_ou', 'lasso_ou'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',LinearSVC(random_state = 1108, C = 1.42797971225))]),
                        }, 
                },
        },
        'line':{
                '+pts':{
                    'knn': {
                        'features': ['lasso_line'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', KNeighborsClassifier(n_neighbors = 198, leaf_size = 100))]),
                        },       
                    'log': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line', '+linsvm_target', '+ridge_target', '+lasso_target', '-ridge_team', '-lasso_team', '-lasso_target'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf', LogisticRegression(random_state = 1108, C = 13.48854758964598, solver = "liblinear"))]),
                        },                                              
            },
                'raw':{
                    'lightgbc': {
                        'features': ['lasso_line', '-50_game_avg_15_g_HAweight_allow_blocks-per-game', 'pregame_ppp_for', 'tsvd_line', '-10_game_avg_50_g_Tweight_for_assists-per-game', '-20_game_avg_15_g_Tweight_allow_extra-chances-per-game', '50_game_avg_50_g_HAweight_for_defensive-rebounding-pct', '100_g_HAspread_for_defensive-efficiency', '20_game_avg_30_g_HAweight_allow_fta-per-fga', 'pca_line', '-20_game_avg_50_g_Tweight_allow_fta-per-fga', 'pregame_offensive-rebounding-pct_for', '-75_g_HAspread_allow_defensive-efficiency', '-50_game_avg_30_g_Tweight_allow_block-pct', '-20_game_avg_50_g_Tweight_for_block-pct', '-expected_poss_pg_allowed', 'lightgbm_line', '20_game_avg_30_g_Tweight_allow_assist--per--turnover-ratio', '-50_game_avg_15_g_Tweight_allow_blocks-per-game', '10_game_avg_30_g_Tweight_for_assists-per-game', 'ridge_line', '75_g_HAspread_for_defensive-efficiency', '-50_game_avg_50_g_Tweight_for_assist--per--turnover-ratio', '75_g_HAspread_allow_defensive-efficiency', '-5_game_avg_50_g_HAweight_allow_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_allow_assist--per--turnover-ratio', '-30_game_avg_10_g_HAweight_allow_possessions-per-game', '-50_game_avg_50_g_HAweight_for_offensive-rebounding-pct', '-5_game_avg_10_g_Tweight_allow_possessions-per-game', '100_g_HAspread_allow_block-pct', '-expected_effective-field-goal-pct_allowed', '1_game_avg_10_g_HAweight_for_points-per-game', '50_game_avg_30_g_Tweight_allow_offensive-efficiency'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', lgb.LGBMClassifier(random_state = 1108, n_estimators = 150, colsample_bytree = 0.609201056258738, min_child_samples = 177, num_leaves = 49, subsample = 0.814351700300212, max_bin = 1958, learning_rate = 0.005))]),
                        },
                    'knn': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line', '-75_g_HAspread_allow_defensive-efficiency', '75_g_HAspread_allow_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_block-pct', '100_g_HAspread_for_defensive-efficiency', '50_g_HAspread_allow_assist--per--turnover-ratio'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', KNeighborsClassifier(n_neighbors = 12, leaf_size = 3))]),
                        },                            
                    'log': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf', LogisticRegression(random_state = 1108, C = 0.012407087605742084, solver = "liblinear"))]),
                        }, 
                    'linsvc': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line', '-75_g_HAspread_allow_defensive-efficiency', '75_g_HAspread_allow_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_block-pct', '100_g_HAspread_for_defensive-efficiency', '50_g_HAspread_allow_assist--per--turnover-ratio', '-expected_effective-field-goal-pct_allowed', 'pregame_offensive-rebounding-pct_for', '-10_g_HAspread_allow_points-per-game`/`possessions-per-game', '25_g_HAspread_for_possessions-per-game', '-5_game_avg_10_g_Tweight_allow_possessions-per-game', '-5_game_avg_50_g_HAweight_allow_possessions-per-game', '20_game_avg_30_g_Tweight_allow_assist--per--turnover-ratio', '1_game_avg_10_g_HAweight_for_points-per-game', '10_game_avg_30_g_Tweight_for_assists-per-game', '-expected_poss_pg_allowed', '-50_game_avg_15_g_Tweight_allow_blocks-per-game', 'pca_line', '-50_game_avg_30_g_Tweight_allow_block-pct', '20_game_avg_30_g_HAweight_allow_fta-per-fga', '-50_game_avg_15_g_HAweight_allow_blocks-per-game', '-20_game_avg_50_g_Tweight_for_block-pct', '-20_game_avg_15_g_Tweight_allow_extra-chances-per-game', '-30_game_avg_10_g_HAweight_allow_possessions-per-game', '-50_game_avg_50_g_HAweight_for_offensive-rebounding-pct', 'tsvd_line', 'pregame_ppp_for', '-50_game_avg_50_g_Tweight_for_assist--per--turnover-ratio', '50_game_avg_30_g_Tweight_allow_offensive-efficiency', '-10_game_avg_50_g_Tweight_for_assists-per-game', '-20_game_avg_50_g_Tweight_allow_fta-per-fga'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',LinearSVC(random_state = 1108, C = 0.00100000009439))]),
                        }, 
                },
        },
        'result':{
                'line':{
                    'ridge': {
                        'features': ['ha', '10_game_avg', 'streak'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'sag', alpha = 584.39992591))]),
                        },
                    'lasso': {
                        'features': ['ha', '10_game_avg', 'streak'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.0011089952827, max_iter = 2000))]),
                        },
                    'lightgbm': {
                        'features': ['10_game_avg', 'streak', 'ha', '50_game_avg'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 1000, colsample_bytree = 0.980189782695348, min_child_samples = 189, num_leaves = 14, subsample = 0.791883150188403, max_bin = 1988, learning_rate = 0.00140625))]),
                        }
                },
                'ou': {
                    'ridge': {
                        'features': ['10_game_avg', '15_game_avg', '50_game_avg', '5_game_avg', 'streak', '30_game_avg'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 737.480281596))]),
                        },
                    'lasso': {
                        'features': ['10_game_avg', '15_game_avg', '50_game_avg', '5_game_avg', 'streak', '30_game_avg'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.00171909758817))]),
                        },
                    'lightgbm': {
                        'features': ['3_game_avg', '15_game_avg', '10_game_avg', '5_game_avg', 'streak', '50_game_avg', '30_game_avg'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 300, colsample_bytree = 0.952264164385702, min_child_samples = 12, num_leaves = 36, subsample = 0.64633747168907, max_bin = 1138, learning_rate = 0.015))]),                
                }
            },
        },
        'pts_allowed': {
                'all': {
                    'ridge': {
                        'features': ['expected_ppp_allowed', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', 'pregame_pts_pg_allowed', 'expected_offensive-rebounding-pct_allowed', '50_g_HAspread_for_assist--per--turnover-ratio', '75_g_HAspread_allow_floor-percentage', 'pregame_ppp_allowed', '100_g_HAspread_for_defensive-efficiency', 'expected_turnovers-per-possession_allowed', 'expected_pts_pg_allowed', 'expected_effective-field-goal-pct_allowed', 'pregame_turnovers-per-possession_allowed', 'expected_poss_pg_allowed', 'pregame_ftm-per-100-possessions_allowed', '20_game_avg_50_g_Tweight_for_defensive-efficiency', '30_game_avg_10_g_HAweight_allow_possessions-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_ftm-per-100-possessions', '50_game_avg_30_g_Tweight_allow_fta-per-fga', '50_game_avg_30_g_HAweight_allow_defensive-rebounds-per-game', '25_g_HAspread_allow_possessions-per-game', '10_game_avg_50_g_Tweight_for_assists-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '10_game_avg_50_g_Tweight_allow_offensive-rebounding-pct', '10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '10_game_avg_10_g_HAweight_allow_points-per-game'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'sparse_cg', alpha = 0.89393381144))]),
                        },
                    'lightgbm': {
                        'features': ['pregame_ppp_allowed', '50_game_avg_50_g_HAweight_for_assists-per-game', 'expected_effective-field-goal-pct_allowed', '100_g_HAspread_for_defensive-efficiency', '30_game_avg_25_g_Tweight_allow_points-per-game', '75_g_HAspread_allow_floor-percentage', 'pregame_ftm-per-100-possessions_allowed', 'expected_offensive-rebounding-pct_allowed', 'expected_poss_pg_allowed', 'pregame_pts_pg_allowed', 'expected_ppp_allowed', '10_game_avg_50_g_Tweight_for_assists-per-game', 'expected_pts_pg_allowed', '10_game_avg_10_g_HAweight_allow_points-per-game', '50_game_avg_30_g_Tweight_allow_fta-per-fga', '25_g_HAspread_allow_possessions-per-game', '20_game_avg_50_g_Tweight_for_defensive-efficiency', '20_game_avg_30_g_HAweight_for_ftm-per-100-possessions', '50_g_HAspread_for_assist--per--turnover-ratio', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', 'pregame_turnovers-per-possession_allowed', '10_game_avg_50_g_Tweight_allow_offensive-rebounding-pct', 'expected_turnovers-per-possession_allowed', '30_game_avg_10_g_HAweight_allow_possessions-per-game', '50_game_avg_30_g_HAweight_allow_defensive-rebounds-per-game'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 400, colsample_bytree = 0.796359005305649, min_child_samples = 198, num_leaves = 13, subsample = 0.65344498465166, max_bin = 1953, learning_rate = 0.03))]),
                        }
                    },
#                    
                'target':{
                    'lightgbm': {
                        'features': ['20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_25_g_Tweight_allow_points-per-game', '50_g_HAspread_allow_points-per-game`/`possessions-per-game', 'expected_ppp_allowed', '10_game_avg_5_g_Tweight_allow_points-per-game`/`possessions-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '25_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_game_avg_25_g_HAweight_allow_points-per-game', '1_game_avg_25_g_Tweight_allow_points-per-game`/`possessions-per-game', 'pregame_ppp_allowed', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', 'expected_pts_pg_allowed'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 900, colsample_bytree = 0.956416580127077, min_child_samples = 4, num_leaves = 61, subsample = 0.631441132669909, max_bin = 1974, learning_rate = 0.00375))]),
                        },                         
#                    'ridge': {
#                        'features': ['expected_ppp_allowed', '50_g_HAspread_allow_points-per-game`/`possessions-per-game', 'expected_pts_pg_allowed', 'pregame_ppp_allowed', '25_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '20_game_avg_5_g_HAweight_allow_points-per-game', '20_game_avg_25_g_Tweight_allow_points-per-game', '20_game_avg_50_g_Tweight_allow_points-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_allow_points-per-game`/`possessions-per-game', '10_game_avg_25_g_HAweight_allow_points-per-game', '1_game_avg_25_g_Tweight_allow_points-per-game`/`possessions-per-game'],
#                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 7.004254054653961))]),
#                        }, 
                    'lasso': {
                        'features': ['expected_ppp_allowed', '50_g_HAspread_allow_points-per-game`/`possessions-per-game', 'expected_pts_pg_allowed', 'pregame_ppp_allowed', '25_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '20_game_avg_5_g_HAweight_allow_points-per-game', '20_game_avg_25_g_Tweight_allow_points-per-game', '20_game_avg_50_g_Tweight_allow_points-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_allow_points-per-game`/`possessions-per-game', '10_game_avg_25_g_HAweight_allow_points-per-game', '1_game_avg_25_g_Tweight_allow_points-per-game`/`possessions-per-game'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.001, max_iter = 2000))]),
                        },
                    },
#                 
                'possessions': {
                    'lightgbm': {
                        'features': ['5_game_avg_25_g_Tweight_allow_possessions-per-game', 'expected_poss_pg_allowed', '20_game_avg_50_g_Tweight_allow_possessions-per-game', '5_game_avg_10_g_Tweight_allow_possessions-per-game', '25_g_HAspread_allow_possessions-per-game', '1_game_avg_50_g_HAweight_allow_possessions-per-game', 'pregame_poss_pg_allowed', '30_game_avg_5_g_Tweight_allow_possessions-per-game', '1_game_avg_10_g_Tweight_allow_possessions-per-game', '30_game_avg_25_g_HAweight_allow_possessions-per-game'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 4050, colsample_bytree = 0.947978555103353, min_child_samples = 4, num_leaves = 20, subsample = 0.417125690980936, max_bin = 1114, learning_rate = 0.00125))]),
                        },
                    'ridge': {
                        'features': ['expected_poss_pg_allowed', 'pregame_poss_pg_allowed', '5_game_avg_25_g_Tweight_allow_possessions-per-game', '5_game_avg_10_g_Tweight_allow_possessions-per-game', '25_g_HAspread_allow_possessions-per-game', '1_game_avg_50_g_HAweight_allow_possessions-per-game', '1_game_avg_10_g_Tweight_allow_possessions-per-game', '20_game_avg_50_g_Tweight_allow_possessions-per-game', '30_game_avg_25_g_HAweight_allow_possessions-per-game'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 4.095337845324041))]),
                        }, 
                    'lasso': {
                        'features': ['expected_poss_pg_allowed', 'pregame_poss_pg_allowed', '5_game_avg_25_g_Tweight_allow_possessions-per-game', '5_game_avg_10_g_Tweight_allow_possessions-per-game', '25_g_HAspread_allow_possessions-per-game', '1_game_avg_50_g_HAweight_allow_possessions-per-game', '1_game_avg_10_g_Tweight_allow_possessions-per-game', '20_game_avg_50_g_Tweight_allow_possessions-per-game', '30_game_avg_25_g_HAweight_allow_possessions-per-game', '30_game_avg_5_g_Tweight_allow_possessions-per-game'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.001))]),
                        },
                    },
#                        
                'full-team':{
                    'linsvm': {
                        'features': ['expected_effective-field-goal-pct_allowed', '75_g_HAspread_allow_floor-percentage', 'expected_turnovers-per-possession_allowed', 'expected_effective-field-goal-pct_allowed', 'expected_offensive-rebounding-pct_allowed', '75_g_HAspread_allow_defensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_assist--per--turnover-ratio', '50_g_HAspread_for_assist--per--turnover-ratio', '30_g_HAspread_for_offensive-efficiency'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',LinearSVR(random_state = 1108, C = 2.45976772084, epsilon=0))]),
                        }, 
                    'ridge': {
                        'features': ['expected_effective-field-goal-pct_allowed', '75_g_HAspread_allow_floor-percentage', 'expected_turnovers-per-possession_allowed', 'expected_effective-field-goal-pct_allowed', 'expected_offensive-rebounding-pct_allowed', '75_g_HAspread_allow_defensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_assist--per--turnover-ratio', '50_g_HAspread_for_assist--per--turnover-ratio', '30_g_HAspread_for_offensive-efficiency', '100_g_HAspread_for_points-per-game', 'pregame_turnovers-per-possession_allowed', 'pregame_offensive-rebounding-pct_allowed'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 6.07138889187))]),
                        },  
                    'lasso': {
                        'features': ['expected_effective-field-goal-pct_allowed', '75_g_HAspread_allow_floor-percentage', 'expected_turnovers-per-possession_allowed', 'expected_effective-field-goal-pct_allowed', 'expected_offensive-rebounding-pct_allowed', '75_g_HAspread_allow_defensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_assist--per--turnover-ratio', '50_g_HAspread_for_assist--per--turnover-ratio', '30_g_HAspread_for_offensive-efficiency', '100_g_HAspread_for_points-per-game'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.00125925622966, max_iter = 2000))]),
                        },
                    'lightgbm': {
                        'features': ['75_g_HAspread_allow_floor-percentage', 'expected_effective-field-goal-pct_allowed', '50_game_avg_15_g_Tweight_allow_blocks-per-game', '30_g_HAspread_for_offensive-efficiency', 'expected_effective-field-goal-pct_allowed', '100_g_HAspread_allow_assist--per--turnover-ratio', '20_game_avg_30_g_Tweight_for_defensive-rebounding-pct', 'pregame_offensive-rebounding-pct_allowed', '100_g_HAspread_for_points-per-game', '100_g_HAspread_for_defensive-efficiency', '50_g_HAspread_for_assist--per--turnover-ratio', '75_g_HAspread_allow_defensive-efficiency', 'expected_offensive-rebounding-pct_allowed', 'pregame_turnovers-per-possession_allowed', 'expected_turnovers-per-possession_allowed'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 1400, colsample_bytree = 0.99323664013058, min_child_samples = 61, num_leaves = 46, subsample = 0.464276892923476, max_bin = 1557, learning_rate = 0.0028125))]),
                        }                            
                    }
                },
        'pts_scored': {
                'all':{
                    'linsvm': {
                        'features': ['expected_ppp_for', '25_g_HAspread_for_points-per-game', '30_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_block-pct', '30_g_HAspread_allow_floor-percentage', '100_g_HAspread_allow_assist--per--turnover-ratio', 'pregame_pts_pg_for', '75_g_HAspread_for_floor-percentage', '50_g_HAspread_for_points-per-game', '100_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', 'pregame_ppp_for', '75_g_HAspread_for_offensive-efficiency', 'expected_pts_pg_for', 'expected_ppp_for', '10_g_HAspread_allow_personal-fouls-per-possession', 'expected_ftm-per-100-possessions_for', 'expected_effective-field-goal-pct_for', '75_g_HAspread_allow_percent-of-points-from-3-pointers', 'expected_poss_pg_for', '50_game_avg_50_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_HAweight_allow_two-point-rate', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '1_game_avg_50_g_HAweight_for_possessions-per-game', 'expected_turnovers-per-possession_for'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',LinearSVR(random_state = 1108, C = 1.19681838901, epsilon=0))]),
                        },
                    'ridge': {
                        'features': ['expected_ppp_for', '25_g_HAspread_for_points-per-game', '30_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_block-pct', '30_g_HAspread_allow_floor-percentage', '100_g_HAspread_allow_assist--per--turnover-ratio', 'pregame_pts_pg_for', '75_g_HAspread_for_floor-percentage', '50_g_HAspread_for_points-per-game', '100_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', 'pregame_ppp_for', '75_g_HAspread_for_offensive-efficiency', 'expected_pts_pg_for', 'expected_ppp_for', '10_g_HAspread_allow_personal-fouls-per-possession', 'expected_ftm-per-100-possessions_for', 'expected_effective-field-goal-pct_for', '75_g_HAspread_allow_percent-of-points-from-3-pointers', 'expected_poss_pg_for', '50_game_avg_50_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_HAweight_allow_two-point-rate', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '1_game_avg_50_g_HAweight_for_possessions-per-game', 'expected_turnovers-per-possession_for'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 0.00115411795019))]),
                        },                                              
                    },
#                
                'possessions':{
                    'linsvm': {
                        'features': ['expected_poss_pg_for', 'pregame_poss_pg_for', '10_game_avg_10_g_Tweight_for_possessions-per-game', '10_game_avg_10_g_HAweight_for_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '1_game_avg_5_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '30_game_avg_10_g_Tweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',LinearSVR(random_state = 1108, C = 2.7208908100107254, epsilon=0))]),
                        },
                    'lightgbm': {
                        'features': ['10_game_avg_10_g_Tweight_for_possessions-per-game', 'expected_poss_pg_for', '1_game_avg_5_g_Tweight_for_possessions-per-game', 'pregame_poss_pg_for', '10_game_avg_10_g_HAweight_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 2800, colsample_bytree = 0.961267241448647, min_child_samples = 9, num_leaves = 18, subsample = 0.596425797228693, max_bin = 1844, learning_rate = 0.0025))]),
                        },
#                    'ridge': {
#                        'features': ['expected_poss_pg_for', 'pregame_poss_pg_for', '10_game_avg_10_g_Tweight_for_possessions-per-game', '10_game_avg_10_g_HAweight_for_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '1_game_avg_5_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '30_game_avg_10_g_Tweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game'],
#                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'lsqr', alpha = 1.985020145021695))]),
#                        },
                    'lasso': {
                        'features': ['expected_poss_pg_for', 'pregame_poss_pg_for', '10_game_avg_10_g_Tweight_for_possessions-per-game', '10_game_avg_10_g_HAweight_for_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '1_game_avg_5_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '30_game_avg_10_g_Tweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.0015540696172751227, max_iter = 2000))]),
                        },
                    },
#                        
                'target':{
                    'linsvm': {
                        'features': ['expected_ppp_for', 'pregame_ppp_for', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '50_g_HAspread_for_points-per-game', 'pregame_pts_pg_for', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_for_points-per-game`/`possessions-per-game', '5_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '1_game_avg_10_g_HAweight_for_points-per-game'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',LinearSVR(random_state = 1108, C = 0.12071774068337349, epsilon=0))]),
                        },
                    'lightgbm': {
                        'features': ['1_game_avg_10_g_HAweight_for_points-per-game', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'pregame_ppp_for', 'expected_ppp_for', '50_g_HAspread_for_points-per-game', '10_game_avg_5_g_Tweight_for_points-per-game`/`possessions-per-game', '5_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', 'pregame_pts_pg_for'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 500, colsample_bytree = 0.780673078959247, min_child_samples = 5, num_leaves = 17, subsample = 0.607386007072246, max_bin = 1307, learning_rate = 0.0075))]),
                        },
                    'ridge': {
                        'features': ['expected_ppp_for', 'pregame_ppp_for', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '50_g_HAspread_for_points-per-game', 'pregame_pts_pg_for', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_for_points-per-game`/`possessions-per-game', '5_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'sparse_cg', alpha = 0.5619646264922706))]),
                        },
                    'lasso': {
                        'features': ['expected_ppp_for', 'pregame_ppp_for', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '50_g_HAspread_for_points-per-game', 'pregame_pts_pg_for', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game'],
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.13526474650418807, max_iter = 2000))]),
                        },
                    },
#                        
                'offensive_stats': {                       
                    'linsvm': {
                        'features': ['expected_effective-field-goal-pct_for', '75_g_HAspread_for_floor-percentage', 'expected_turnovers-per-possession_for', 'expected_effective-field-goal-pct_for', 'pregame_effective-field-goal-pct_for', '100_g_HAspread_for_floor-percentage', '100_g_HAspread_for_offensive-efficiency', '30_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', '100_g_HAspread_for_defensive-efficiency', 'expected_ftm-per-100-possessions_for', 'pregame_ftm-per-100-possessions_for', 'pregame_offensive-rebounding-pct_for'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',LinearSVR(random_state = 1108, C = 0.0869486130678, epsilon=0))]),
                        },                          
#                    'ridge': {
#                        'features': ['expected_effective-field-goal-pct_for', '75_g_HAspread_for_floor-percentage', 'expected_turnovers-per-possession_for', 'expected_effective-field-goal-pct_for', 'pregame_effective-field-goal-pct_for', '100_g_HAspread_for_floor-percentage', '100_g_HAspread_for_offensive-efficiency', '30_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', '100_g_HAspread_for_defensive-efficiency', 'expected_ftm-per-100-possessions_for', 'pregame_ftm-per-100-possessions_for', 'pregame_offensive-rebounding-pct_for'],
#                        'model': Pipeline([('scale',MinMaxScaler()), ('clf',Ridge(random_state = 1108, solver = 'sparse_cg', alpha = 2.15118131602))]),
#                        },  
                    'lightgbm': {
                        'features': ['75_g_HAspread_for_floor-percentage', '30_g_HAspread_for_floor-percentage', 'pregame_offensive-rebounding-pct_for', 'expected_offensive-rebounding-pct_for', 'expected_effective-field-goal-pct_for', 'expected_ftm-per-100-possessions_for', '100_g_HAspread_for_offensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_for_floor-percentage', 'pregame_effective-field-goal-pct_for', 'pregame_ftm-per-100-possessions_for', 'expected_effective-field-goal-pct_for', 'expected_turnovers-per-possession_for'],
                        'model': Pipeline([('scale',MinMaxScaler()), ('clf', lgb.LGBMRegressor(random_state = 1108, n_estimators = 500, colsample_bytree = 0.808603278021336, min_child_samples = 176, num_leaves = 24, subsample = 0.678375514083654, max_bin = 1032, learning_rate = 0.01))]),
                        }
                    }
                }
            }