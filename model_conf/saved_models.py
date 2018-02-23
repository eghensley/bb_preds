import sys

try:
    import lightgbm as lgb
except ImportError:
    sys.path.insert(-1, "/home/eric/LightGBM/python-package")
    import lightgbm as lgb
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.svm import LinearSVR, SVC
from sklearn.neighbors import KNeighborsClassifier


stored_models = {
        'offense':{
            'pace': {
                'features': ['lasso_possessions', 'lightgbm_possessions', 'linsvm_possessions'],
                'model': Lasso(random_state = 1108, alpha = 0.019012104600893226),
                'scale':StandardScaler()
            },    
            'ppp': {
                'features': ['linsvm_all', 'ridge_all', 'lightgbm_target', 'ridge_target', 'lightgbm_team', 'linsvm_team', 'rest', 'lasso_target', 'linsvm_target'],
                'model': LinearSVR(random_state = 1108, C = 0.215329117725, epsilon=0),
                'scale':RobustScaler()
            },                 
        },
        'defense':{
            'pace': {
                'features': ['lasso_possessions', 'lightgbm_possessions', 'ridge_possessions'],
                'model': LinearSVR(random_state = 1108, C = 0.6652011887133216, epsilon=0),
                'scale':MinMaxScaler()
            },  
            'ppp': {
                'features': ['lightgbm_all', 'ridge_all', 'lasso_target', 'lightgbm_target', 'lasso_team', 'lightgbm_team', 'linsvm_team', 'ridge_team', 'rest'],
                'model': Ridge(random_state = 1108, solver = 'lsqr', alpha = 0.00101115979472),
                'scale':StandardScaler()
            },  
        },
        'predictive':{
                    '+pts': {
                        'features': ['-lightgbm_all', '+lasso_possessions', '+linsvm_target', '+ridge_target', '+lightgbm_target', '+linsvm_team', '+lightgbm_team', '+ridge_all', '+linsvm_all', '+lightgbm_possessions', '-ridge_possessions', '-lasso_possessions', '-lightgbm_target', '-lasso_target', '-ridge_team', '-linsvm_team', '-lightgbm_team', '-lasso_team', '-ridge_all', '-lightgbm_possessions', '+linsvm_possessions', '-rest', '+lasso_target', '+rest'],
                        'model': LinearSVR(random_state = 1108, C =  0.0395279798349, epsilon=0),
                        'scale':StandardScaler(),
                        },    
        },
        'winner':{
                '+pts':{
                    'log': {
                        'features': ['-lightgbm_all', '+lightgbm_team', '+ridge_all', '+linsvm_all', '+ridge_target', '+linsvm_team', '-lightgbm_target', '+linsvm_target', '-ridge_team', '-linsvm_team', '-lightgbm_team', '-lasso_team', '-ridge_all', '-lasso_target', '+lightgbm_target', '-lasso_possessions', '-lightgbm_possessions', '-ridge_possessions', '+lasso_target'],
                        'model': LogisticRegression(random_state = 1108, C = 1000, solver = "liblinear"),
                        'scale':StandardScaler(),
                        'acc_weight': 0.0584245954,
                        'logloss_weight': 0.1132769581,
                        },
                    },
                'raw':{
                    'knn': {
                        'features': ['expected_ppp_for', '-50_g_HAspread_allow_points-per-game`/`possessions-per-game', '-75_g_HAspread_allow_defensive-efficiency', 'expected_ftm-per-100-possessions_for', '30_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_floor-percentage', '75_g_HAspread_allow_points-per-game', '100_g_HAspread_for_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '-50_g_HAspread_for_assist--per--turnover-ratio', '-50_g_HAspread_for_personal-fouls-per-game', '-100_g_HAspread_for_defensive-efficiency', '30_g_HAspread_allow_free-throw-rate', '-100_g_HAspread_allow_assist--per--turnover-ratio', '100_g_HAspread_for_personal-fouls-per-game', 'expected_offensive-rebounding-pct_for', '50_g_HAspread_allow_defensive-efficiency', 'expected_effective-field-goal-pct_for', '-10_g_HAspread_allow_ftm-per-100-possessions', 'expected_turnovers-per-possession_for', 'pregame_turnovers-per-possession_for', '-50_game_avg_30_g_Tweight_allow_fta-per-fga'],
                        'model': KNeighborsClassifier(n_neighbors = 166, leaf_size = 14),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.094108854,
                        'logloss_weight':0.1591595226,                        
                        },                            
                    'log': {
                        'features': ['expected_ppp_for', '-50_g_HAspread_allow_points-per-game`/`possessions-per-game', '-75_g_HAspread_allow_defensive-efficiency', 'expected_ftm-per-100-possessions_for', '30_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_floor-percentage', '75_g_HAspread_allow_points-per-game', '100_g_HAspread_for_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '-50_g_HAspread_for_assist--per--turnover-ratio', '-50_g_HAspread_for_personal-fouls-per-game', '-100_g_HAspread_for_defensive-efficiency', '30_g_HAspread_allow_free-throw-rate', '-100_g_HAspread_allow_assist--per--turnover-ratio', '100_g_HAspread_for_personal-fouls-per-game', 'expected_offensive-rebounding-pct_for', '50_g_HAspread_allow_defensive-efficiency', 'expected_effective-field-goal-pct_for', '-10_g_HAspread_allow_ftm-per-100-possessions', 'expected_turnovers-per-possession_for', 'pregame_turnovers-per-possession_for'],
                        'model': LogisticRegression(random_state = 1108, C = 1.007409512406823, solver = "lbfgs"),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.2800316853,
                        'logloss_weight':0.2399684964,                        
                        }, 
                    'linsvc': {
                        'features': ['expected_ppp_for', '-50_g_HAspread_allow_points-per-game`/`possessions-per-game', '-75_g_HAspread_allow_defensive-efficiency', 'expected_ftm-per-100-possessions_for', '30_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_floor-percentage', '75_g_HAspread_allow_points-per-game', '100_g_HAspread_for_defensive-efficiency', '75_g_HAspread_for_defensive-efficiency', '-50_g_HAspread_for_assist--per--turnover-ratio', '-50_g_HAspread_for_personal-fouls-per-game', '-100_g_HAspread_for_defensive-efficiency', '30_g_HAspread_allow_free-throw-rate', '-100_g_HAspread_allow_assist--per--turnover-ratio', '100_g_HAspread_for_personal-fouls-per-game', 'expected_offensive-rebounding-pct_for', '50_g_HAspread_allow_defensive-efficiency', 'expected_effective-field-goal-pct_for', '-10_g_HAspread_allow_ftm-per-100-possessions', 'expected_turnovers-per-possession_for', 'pregame_turnovers-per-possession_for', '-50_game_avg_30_g_Tweight_allow_fta-per-fga', '-20_game_avg_50_g_Tweight_for_floor-percentage'],
                        'model': SVC(random_state = 1108, C = 0.0207173498763, kernel = 'linear', probability = True),
                        'scale':StandardScaler(),
                        'acc_weight': 0.2778671236,
                        'logloss_weight':0.2364422815,                        
                        }, 
                    'lightgbc': {
                        'features': ['-100_g_HAspread_allow_assist--per--turnover-ratio', '-10_g_HAspread_allow_ftm-per-100-possessions', '50_g_HAspread_allow_defensive-efficiency', 'expected_ppp_for', '30_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', 'pregame_turnovers-per-possession_for', '-100_g_HAspread_for_defensive-efficiency', '-50_game_avg_30_g_Tweight_allow_fta-per-fga', '-50_g_HAspread_allow_points-per-game`/`possessions-per-game', '100_g_HAspread_for_defensive-efficiency', 'expected_turnovers-per-possession_for', 'expected_effective-field-goal-pct_for', '-20_game_avg_50_g_Tweight_for_floor-percentage', 'expected_ftm-per-100-possessions_for', '-50_g_HAspread_for_assist--per--turnover-ratio', '100_g_HAspread_for_personal-fouls-per-game', '75_g_HAspread_allow_points-per-game', '-75_g_HAspread_allow_floor-percentage', '75_g_HAspread_for_defensive-efficiency', '-50_g_HAspread_for_personal-fouls-per-game', '-75_g_HAspread_allow_defensive-efficiency', '30_g_HAspread_allow_free-throw-rate'],
                        'model': lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, colsample_bytree = 0.642120736080607, min_child_samples = 116, num_leaves = 12, subsample = 0.897114330960264, max_bin = 1021, learning_rate = 0.08),
                        'scale':StandardScaler(),
                        'acc_weight': 0.2895677417,
                        'logloss_weight':0.2511527414,                        
                        },
                },
        },
        'ou':{
                '+pts':{      
                    'log': {
                        'features': ['+linsvm_team', '+lightgbm_team', '-lasso_team', '-ridge_team', '-linsvm_team', '-ridge_all', '-lightgbm_team', '-lightgbm_target', '-lightgbm_all', '-lasso_target', '+lightgbm_target', '+linsvm_target', '+ridge_target', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-ridge_possessions', '-lasso_possessions', '-lightgbm_possessions', '+ridge_all', '+linsvm_all', '+rest'],
                        'model':  LogisticRegression(random_state = 1108, C = 15.56187413425767, solver = "newton-cg"),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.1529158222,
                        'logloss_weight':0.1693534377, 
                        }, 
                    'linsvc': {
                        'features': ['+linsvm_team', '+lightgbm_team', '-lasso_team', '-ridge_team', '-linsvm_team', '-ridge_all', '-lightgbm_team', '-lightgbm_target', '-lightgbm_all', '-lasso_target', '+lightgbm_target', '+linsvm_target', '+ridge_target', '+lasso_target', '+lightgbm_possessions', '+linsvm_possessions', '+lasso_possessions', '-ridge_possessions', '-lasso_possessions', '-lightgbm_possessions', '+ridge_all', '+linsvm_all', '+rest', '-rest', 'tsvd_ou', 'pca_ou', 'lightgbm_ou', 'ridge_ou'],
                        'model': SVC(random_state = 1108, C =  0.15817437173, kernel = 'linear', probability = True),
                        'scale':MinMaxScaler(),
                        'acc_weight':0.1018977904,
                        'logloss_weight':0.1562141864, 
                        },    
                    'lightgbc': {
                        'features': ['lasso_ou', '+linsvm_possessions', '+lightgbm_team', '-lasso_team', 'pca_ou', 'lightgbm_ou', '-ridge_possessions', 'tsvd_ou', '+ridge_target', '+linsvm_all', '-lasso_target', '+linsvm_team', '-lightgbm_all', '-lasso_possessions', '+lasso_possessions', '+ridge_all', 'ridge_ou', '-ridge_team', '-lightgbm_target', '-lightgbm_team', '+lightgbm_target', '-rest', '+rest', '-linsvm_team', '+linsvm_target', '+lasso_target', '+lightgbm_possessions', '-ridge_all', '-lightgbm_possessions'],
                        'model': lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, colsample_bytree = 0.925424645171526, min_child_samples = 63, num_leaves = 159, subsample = 0.417196512684593, max_bin = 1011, learning_rate = 0.01),
                        'scale':StandardScaler(),
                        'acc_weight': 0.0955044275,
                        'logloss_weight':0.1493841177, 
                        },                                          
                },
                'raw':{
                    'lightgbc': {
                        'features': ['10_game_avg_10_g_Tweight_for_possessions-per-game', 'ridge_ou', '-1_game_avg_10_g_Tweight_allow_possessions-per-game', '-expected_effective-field-goal-pct_allowed', 'lasso_ou', '-30_game_avg_25_g_Tweight_allow_points-per-game', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '25_g_HAspread_for_possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '-30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', 'expected_effective-field-goal-pct_for', 'lightgbm_ou', 'tsvd_ou', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-50_game_avg_50_g_HAweight_for_assists-per-game', '10_game_avg_50_g_HAweight_for_blocks-per-game', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '75_g_HAspread_for_floor-percentage', '-75_g_HAspread_allow_defensive-efficiency', 'pca_ou', '-50_game_avg_50_g_HAweight_allow_ftm-per-100-possessions', '-25_g_HAspread_allow_possessions-per-game', '100_g_HAspread_allow_block-pct', '-30_game_avg_50_g_HAweight_allow_points-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-15_g_HAspread_allow_block-pct', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-100_g_HAspread_for_points-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '25_g_HAspread_for_points-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game'],
                        'model': lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, colsample_bytree = 0.785582982952984, min_child_samples = 198, num_leaves = 11, subsample = 0.633073504349269, max_bin = 1359, learning_rate = 0.02),
                        'scale':StandardScaler(),
                        'acc_weight': 0.1817906222,
                        'logloss_weight':0.162580079, 
                        },
                    'knn': {
                        'features': ['expected_effective-field-goal-pct_for', '-expected_effective-field-goal-pct_allowed', '20_game_avg_10_g_HAweight_for_possessions-per-game', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-20_game_avg_25_g_Tweight_allow_possessions-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '10_game_avg_10_g_Tweight_for_possessions-per-game', '25_g_HAspread_for_points-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio', '-75_g_HAspread_allow_defensive-efficiency', '-30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '75_g_HAspread_for_floor-percentage', 'tsvd_ou', '-50_game_avg_50_g_HAweight_allow_ftm-per-100-possessions', '-15_g_HAspread_allow_block-pct', '-50_game_avg_50_g_HAweight_for_assists-per-game', '-100_g_HAspread_for_points-per-game', '-30_game_avg_25_g_Tweight_allow_points-per-game', '25_g_HAspread_for_possessions-per-game', 'pca_ou', '-25_g_HAspread_allow_possessions-per-game', 'lightgbm_ou', '10_game_avg_50_g_HAweight_for_blocks-per-game', '-30_game_avg_50_g_HAweight_allow_points-per-game', '100_g_HAspread_allow_block-pct', 'ridge_ou', 'lasso_ou'],
                        'model': KNeighborsClassifier(n_neighbors = 88, leaf_size = 30),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.2339456689,
                        'logloss_weight':0.1929314672, 
                        },                            
                    'log': {
                        'features': ['expected_effective-field-goal-pct_for', '-expected_effective-field-goal-pct_allowed', '20_game_avg_10_g_HAweight_for_possessions-per-game', '-20_game_avg_50_g_Tweight_for_floor-percentage', '-10_game_avg_10_g_Tweight_allow_points-per-game`/`possessions-per-game', '-20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_defensive-rebounds-per-game', '10_game_avg_30_g_Tweight_for_true-shooting-percentage', '-20_game_avg_25_g_Tweight_allow_possessions-per-game', '-20_game_avg_50_g_HAweight_allow_defensive-efficiency', '-10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '-10_game_avg_15_g_HAweight_for_defensive-efficiency', '75_g_HAspread_allow_percent-of-points-from-3-pointers', '10_game_avg_10_g_Tweight_for_possessions-per-game', '25_g_HAspread_for_points-per-game', '-10_game_avg_10_g_HAweight_allow_points-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '-100_g_HAspread_allow_assist--per--turnover-ratio'],
                        'model':LogisticRegression(random_state = 1108, C = 0.04795626933077879, solver = "liblinear"),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.2339456689,
                        'logloss_weight':0.1695367121, 
                        }, 
                },
        },
        'line':{
                '+pts':{
                    'knn': {
                        'features': ['lasso_line'],
                        'model': KNeighborsClassifier(n_neighbors = 198, leaf_size = 100),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.0957541993,
                        'logloss_weight':0.1242102222,                        
                        },       
                    'log': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line', '+linsvm_target', '+ridge_target', '+lasso_target', '-ridge_team', '-lasso_team', '-lasso_target'],
                        'model': LogisticRegression(random_state = 1108, C = 13.48854758964598, solver = "liblinear"),
                        'scale':RobustScaler(),
                        'acc_weight': 0.1769992154,
                        'logloss_weight':0.1717807627, 
                        },                                              
                    'linsvc': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line', '+linsvm_target', '+ridge_target', '+lasso_target', '-ridge_team', '-lasso_team'],
                        'model': SVC(random_state = 1108, C =  4.87778133978, kernel = 'linear', probability = True),
                        'scale':StandardScaler(),
                        'acc_weight': 0.1358476918,
                        'logloss_weight':0.169374042,  
                        }, 
                    'lightgbc': {
                        'features': ['lasso_line', '+linsvm_possessions', 'tsvd_line', 'pca_line', '-lasso_possessions', '+lasso_possessions', 'lightgbm_line', '+ridge_target', '-lasso_target', '-ridge_possessions', '+linsvm_all', '-lasso_team', '+linsvm_team'],
                        'model': lgb.LGBMClassifier(random_state = 1108, n_estimators = 225, colsample_bytree =0.985094185628027, min_child_samples = 13, num_leaves = 15, subsample = 0.721806428713343, max_bin = 1110, learning_rate = 0.03),
                        'scale':RobustScaler(),
                        'acc_weight': 0.2126075061,
                        'logloss_weight':0.1811016839, 
                        },                                             
            },
                'raw':{
                    'lightgbc': {
                        'features': ['lasso_line', '-50_game_avg_15_g_HAweight_allow_blocks-per-game', 'pregame_ppp_for', 'tsvd_line', '-10_game_avg_50_g_Tweight_for_assists-per-game', '-20_game_avg_15_g_Tweight_allow_extra-chances-per-game', '50_game_avg_50_g_HAweight_for_defensive-rebounding-pct', '100_g_HAspread_for_defensive-efficiency', '20_game_avg_30_g_HAweight_allow_fta-per-fga', 'pca_line', '-20_game_avg_50_g_Tweight_allow_fta-per-fga', 'pregame_offensive-rebounding-pct_for', '-75_g_HAspread_allow_defensive-efficiency', '-50_game_avg_30_g_Tweight_allow_block-pct', '-20_game_avg_50_g_Tweight_for_block-pct', '-expected_poss_pg_allowed', 'lightgbm_line', '20_game_avg_30_g_Tweight_allow_assist--per--turnover-ratio', '-50_game_avg_15_g_Tweight_allow_blocks-per-game', '10_game_avg_30_g_Tweight_for_assists-per-game', 'ridge_line', '75_g_HAspread_for_defensive-efficiency', '-50_game_avg_50_g_Tweight_for_assist--per--turnover-ratio', '75_g_HAspread_allow_defensive-efficiency', '-5_game_avg_50_g_HAweight_allow_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_allow_assist--per--turnover-ratio', '-30_game_avg_10_g_HAweight_allow_possessions-per-game', '-50_game_avg_50_g_HAweight_for_offensive-rebounding-pct', '-5_game_avg_10_g_Tweight_allow_possessions-per-game', '100_g_HAspread_allow_block-pct', '-expected_effective-field-goal-pct_allowed', '1_game_avg_10_g_HAweight_for_points-per-game', '50_game_avg_30_g_Tweight_allow_offensive-efficiency'],
                        'model': lgb.LGBMClassifier(random_state = 1108, n_estimators = 150, colsample_bytree = 0.609201056258738, min_child_samples = 177, num_leaves = 49, subsample = 0.814351700300212, max_bin = 1958, learning_rate = 0.005),
                        'scale':MinMaxScaler(),
                        'acc_weight': 0.2461102595,
                        'logloss_weight':0.1818867281,                         
                        },                          
                    'log': {
                        'features': ['lasso_line', 'ridge_line', 'lightgbm_line'],
                        'model': LogisticRegression(random_state = 1108, C = 0.012407087605742084, solver = "liblinear"),
                        'scale':RobustScaler(),
                        'acc_weight': 0.1326811279,
                        'logloss_weight':0.171646561,                         
                        }, 
                },
        },
        'result':{
                'line':{
                    'ridge': {
                        'features': ['ha', '10_game_avg', 'streak'],
                        'model': Ridge(random_state = 1108, solver = 'sag', alpha = 584.39992591),
                        'scale':MinMaxScaler()
                        },
                    'lasso': {
                        'features': ['ha', '10_game_avg', 'streak'],
                        'model': Lasso(random_state = 1108, alpha = 0.0011089952827, max_iter = 2000),
                        'scale':StandardScaler()
                        },
                    'lightgbm': {
                        'features': ['10_game_avg', 'streak', 'ha', '50_game_avg'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 1000, colsample_bytree = 0.980189782695348, min_child_samples = 189, num_leaves = 14, subsample = 0.791883150188403, max_bin = 1988, learning_rate = 0.00140625),
                        'scale':StandardScaler()
                        }
                },
                'ou': {
                    'ridge': {
                        'features': ['10_game_avg', '15_game_avg', '50_game_avg', '5_game_avg', 'streak', '30_game_avg'],
                        'model': Ridge(random_state = 1108, solver = 'lsqr', alpha = 737.480281596),
                        'scale':MinMaxScaler()
                        },
                    'lasso': {
                        'features': ['10_game_avg', '15_game_avg', '50_game_avg', '5_game_avg', 'streak', '30_game_avg'],
                        'model': Lasso(random_state = 1108, alpha = 0.00171909758817),
                        'scale':StandardScaler()
                        },
                    'lightgbm': {
                        'features': ['3_game_avg', '15_game_avg', '10_game_avg', '5_game_avg', 'streak', '50_game_avg', '30_game_avg'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 300, colsample_bytree = 0.952264164385702, min_child_samples = 12, num_leaves = 36, subsample = 0.64633747168907, max_bin = 1138, learning_rate = 0.015),                
                        'scale':StandardScaler()
                }
            },
        },
        'pts_allowed': {
                'all': {
                    'ridge': {
                        'features': ['expected_ppp_allowed', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', 'pregame_pts_pg_allowed', 'expected_offensive-rebounding-pct_allowed', '50_g_HAspread_for_assist--per--turnover-ratio', '75_g_HAspread_allow_floor-percentage', 'pregame_ppp_allowed', '100_g_HAspread_for_defensive-efficiency', 'expected_turnovers-per-possession_allowed', 'expected_pts_pg_allowed', 'expected_effective-field-goal-pct_allowed', 'pregame_turnovers-per-possession_allowed', 'expected_poss_pg_allowed', 'pregame_ftm-per-100-possessions_allowed', '20_game_avg_50_g_Tweight_for_defensive-efficiency', '30_game_avg_10_g_HAweight_allow_possessions-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '20_game_avg_30_g_HAweight_for_ftm-per-100-possessions', '50_game_avg_30_g_Tweight_allow_fta-per-fga', '50_game_avg_30_g_HAweight_allow_defensive-rebounds-per-game', '25_g_HAspread_allow_possessions-per-game', '10_game_avg_50_g_Tweight_for_assists-per-game', '50_game_avg_50_g_HAweight_for_assists-per-game', '10_game_avg_50_g_Tweight_allow_offensive-rebounding-pct', '10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', '10_game_avg_10_g_HAweight_allow_points-per-game'],
                        'model': Ridge(random_state = 1108, solver = 'sparse_cg', alpha = 0.89393381144),
                        'scale':MinMaxScaler()
                        },
                    'lightgbm': {
                        'features': ['pregame_ppp_allowed', '50_game_avg_50_g_HAweight_for_assists-per-game', 'expected_effective-field-goal-pct_allowed', '100_g_HAspread_for_defensive-efficiency', '30_game_avg_25_g_Tweight_allow_points-per-game', '75_g_HAspread_allow_floor-percentage', 'pregame_ftm-per-100-possessions_allowed', 'expected_offensive-rebounding-pct_allowed', 'expected_poss_pg_allowed', 'pregame_pts_pg_allowed', 'expected_ppp_allowed', '10_game_avg_50_g_Tweight_for_assists-per-game', 'expected_pts_pg_allowed', '10_game_avg_10_g_HAweight_allow_points-per-game', '50_game_avg_30_g_Tweight_allow_fta-per-fga', '25_g_HAspread_allow_possessions-per-game', '20_game_avg_50_g_Tweight_for_defensive-efficiency', '20_game_avg_30_g_HAweight_for_ftm-per-100-possessions', '50_g_HAspread_for_assist--per--turnover-ratio', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_game_avg_15_g_HAweight_allow_defensive-rebounds-per-game', 'pregame_turnovers-per-possession_allowed', '10_game_avg_50_g_Tweight_allow_offensive-rebounding-pct', 'expected_turnovers-per-possession_allowed', '30_game_avg_10_g_HAweight_allow_possessions-per-game', '50_game_avg_30_g_HAweight_allow_defensive-rebounds-per-game'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 400, colsample_bytree = 0.796359005305649, min_child_samples = 198, num_leaves = 13, subsample = 0.65344498465166, max_bin = 1953, learning_rate = 0.03),
                        'scale':StandardScaler()
                        }
                    },
                'target':{
                    'lightgbm': {
                        'features': ['20_game_avg_50_g_Tweight_allow_points-per-game', '20_game_avg_25_g_Tweight_allow_points-per-game', '50_g_HAspread_allow_points-per-game`/`possessions-per-game', 'expected_ppp_allowed', '10_game_avg_5_g_Tweight_allow_points-per-game`/`possessions-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '25_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_game_avg_25_g_HAweight_allow_points-per-game', '1_game_avg_25_g_Tweight_allow_points-per-game`/`possessions-per-game', 'pregame_ppp_allowed', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', 'expected_pts_pg_allowed'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 900, colsample_bytree = 0.956416580127077, min_child_samples = 4, num_leaves = 61, subsample = 0.631441132669909, max_bin = 1974, learning_rate = 0.00375),
                        'scale':MinMaxScaler()
                        },                         
                    'lasso': {
                        'features': ['expected_ppp_allowed', '50_g_HAspread_allow_points-per-game`/`possessions-per-game', 'expected_pts_pg_allowed', 'pregame_ppp_allowed', '25_g_HAspread_allow_points-per-game`/`possessions-per-game', '10_g_HAspread_allow_points-per-game`/`possessions-per-game', '20_game_avg_5_g_HAweight_allow_points-per-game', '20_game_avg_25_g_Tweight_allow_points-per-game', '20_game_avg_50_g_Tweight_allow_points-per-game', '30_game_avg_25_g_Tweight_allow_points-per-game', '30_game_avg_50_g_Tweight_allow_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_allow_points-per-game`/`possessions-per-game', '10_game_avg_25_g_HAweight_allow_points-per-game', '1_game_avg_25_g_Tweight_allow_points-per-game`/`possessions-per-game'],
                        'model': Lasso(random_state = 1108, alpha = 0.001, max_iter = 2000),
                        'scale':StandardScaler()
                        },
                    },
                'possessions': {
                    'lightgbm': {
                        'features': ['5_game_avg_25_g_Tweight_allow_possessions-per-game', 'expected_poss_pg_allowed', '20_game_avg_50_g_Tweight_allow_possessions-per-game', '5_game_avg_10_g_Tweight_allow_possessions-per-game', '25_g_HAspread_allow_possessions-per-game', '1_game_avg_50_g_HAweight_allow_possessions-per-game', 'pregame_poss_pg_allowed', '30_game_avg_5_g_Tweight_allow_possessions-per-game', '1_game_avg_10_g_Tweight_allow_possessions-per-game', '30_game_avg_25_g_HAweight_allow_possessions-per-game'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 4050, colsample_bytree = 0.947978555103353, min_child_samples = 4, num_leaves = 20, subsample = 0.417125690980936, max_bin = 1114, learning_rate = 0.00125),
                        'scale':StandardScaler()
                        },
                    'ridge': {
                        'features': ['expected_poss_pg_allowed', 'pregame_poss_pg_allowed', '5_game_avg_25_g_Tweight_allow_possessions-per-game', '5_game_avg_10_g_Tweight_allow_possessions-per-game', '25_g_HAspread_allow_possessions-per-game', '1_game_avg_50_g_HAweight_allow_possessions-per-game', '1_game_avg_10_g_Tweight_allow_possessions-per-game', '20_game_avg_50_g_Tweight_allow_possessions-per-game', '30_game_avg_25_g_HAweight_allow_possessions-per-game'],
                        'model': Ridge(random_state = 1108, solver = 'lsqr', alpha = 4.095337845324041),
                        'scale':MinMaxScaler()
                        }, 
                    'lasso': {
                        'features': ['expected_poss_pg_allowed', 'pregame_poss_pg_allowed', '5_game_avg_25_g_Tweight_allow_possessions-per-game', '5_game_avg_10_g_Tweight_allow_possessions-per-game', '25_g_HAspread_allow_possessions-per-game', '1_game_avg_50_g_HAweight_allow_possessions-per-game', '1_game_avg_10_g_Tweight_allow_possessions-per-game', '20_game_avg_50_g_Tweight_allow_possessions-per-game', '30_game_avg_25_g_HAweight_allow_possessions-per-game', '30_game_avg_5_g_Tweight_allow_possessions-per-game'],
                        'model': Lasso(random_state = 1108, alpha = 0.001),
                        'scale':RobustScaler()
                        },
                    },
                'full-team':{
                    'linsvm': {
                        'features': ['expected_effective-field-goal-pct_allowed', '75_g_HAspread_allow_floor-percentage', 'expected_turnovers-per-possession_allowed', 'expected_effective-field-goal-pct_allowed', 'expected_offensive-rebounding-pct_allowed', '75_g_HAspread_allow_defensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_assist--per--turnover-ratio', '50_g_HAspread_for_assist--per--turnover-ratio', '30_g_HAspread_for_offensive-efficiency'],
                        'model': LinearSVR(random_state = 1108, C = 2.45976772084, epsilon=0),
                        'scale':MinMaxScaler()
                        }, 
                    'ridge': {
                        'features': ['expected_effective-field-goal-pct_allowed', '75_g_HAspread_allow_floor-percentage', 'expected_turnovers-per-possession_allowed', 'expected_effective-field-goal-pct_allowed', 'expected_offensive-rebounding-pct_allowed', '75_g_HAspread_allow_defensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_assist--per--turnover-ratio', '50_g_HAspread_for_assist--per--turnover-ratio', '30_g_HAspread_for_offensive-efficiency', '100_g_HAspread_for_points-per-game', 'pregame_turnovers-per-possession_allowed', 'pregame_offensive-rebounding-pct_allowed'],
                        'model': Ridge(random_state = 1108, solver = 'lsqr', alpha = 6.07138889187),
                        'scale':MinMaxScaler()
                        },  
                    'lasso': {
                        'features': ['expected_effective-field-goal-pct_allowed', '75_g_HAspread_allow_floor-percentage', 'expected_turnovers-per-possession_allowed', 'expected_effective-field-goal-pct_allowed', 'expected_offensive-rebounding-pct_allowed', '75_g_HAspread_allow_defensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_assist--per--turnover-ratio', '50_g_HAspread_for_assist--per--turnover-ratio', '30_g_HAspread_for_offensive-efficiency', '100_g_HAspread_for_points-per-game'],
                        'model': Lasso(random_state = 1108, alpha = 0.00125925622966, max_iter = 2000),
                        'scale':RobustScaler()
                        },
                    'lightgbm': {
                        'features': ['75_g_HAspread_allow_floor-percentage', 'expected_effective-field-goal-pct_allowed', '50_game_avg_15_g_Tweight_allow_blocks-per-game', '30_g_HAspread_for_offensive-efficiency', 'expected_effective-field-goal-pct_allowed', '100_g_HAspread_allow_assist--per--turnover-ratio', '20_game_avg_30_g_Tweight_for_defensive-rebounding-pct', 'pregame_offensive-rebounding-pct_allowed', '100_g_HAspread_for_points-per-game', '100_g_HAspread_for_defensive-efficiency', '50_g_HAspread_for_assist--per--turnover-ratio', '75_g_HAspread_allow_defensive-efficiency', 'expected_offensive-rebounding-pct_allowed', 'pregame_turnovers-per-possession_allowed', 'expected_turnovers-per-possession_allowed'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 1400, colsample_bytree = 0.99323664013058, min_child_samples = 61, num_leaves = 46, subsample = 0.464276892923476, max_bin = 1557, learning_rate = 0.0028125),
                        'scale':StandardScaler()
                        }                            
                    }
                },
        'pts_scored': {
                'all':{
                    'linsvm': {
                        'features': ['expected_ppp_for', '25_g_HAspread_for_points-per-game', '30_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_block-pct', '30_g_HAspread_allow_floor-percentage', '100_g_HAspread_allow_assist--per--turnover-ratio', 'pregame_pts_pg_for', '75_g_HAspread_for_floor-percentage', '50_g_HAspread_for_points-per-game', '100_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', 'pregame_ppp_for', '75_g_HAspread_for_offensive-efficiency', 'expected_pts_pg_for', 'expected_ppp_for', '10_g_HAspread_allow_personal-fouls-per-possession', 'expected_ftm-per-100-possessions_for', 'expected_effective-field-goal-pct_for', '75_g_HAspread_allow_percent-of-points-from-3-pointers', 'expected_poss_pg_for', '50_game_avg_50_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_HAweight_allow_two-point-rate', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '1_game_avg_50_g_HAweight_for_possessions-per-game', 'expected_turnovers-per-possession_for'],
                        'model': LinearSVR(random_state = 1108, C = 1.19681838901, epsilon=0),
                        'scale':MinMaxScaler()
                        },
                    'ridge': {
                        'features': ['expected_ppp_for', '25_g_HAspread_for_points-per-game', '30_g_HAspread_for_defensive-efficiency', '100_g_HAspread_allow_block-pct', '30_g_HAspread_allow_floor-percentage', '100_g_HAspread_allow_assist--per--turnover-ratio', 'pregame_pts_pg_for', '75_g_HAspread_for_floor-percentage', '50_g_HAspread_for_points-per-game', '100_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', 'pregame_ppp_for', '75_g_HAspread_for_offensive-efficiency', 'expected_pts_pg_for', 'expected_ppp_for', '10_g_HAspread_allow_personal-fouls-per-possession', 'expected_ftm-per-100-possessions_for', 'expected_effective-field-goal-pct_for', '75_g_HAspread_allow_percent-of-points-from-3-pointers', 'expected_poss_pg_for', '50_game_avg_50_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_Tweight_for_personal-fouls-per-possession', '50_game_avg_30_g_HAweight_allow_two-point-rate', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '1_game_avg_50_g_HAweight_for_possessions-per-game', 'expected_turnovers-per-possession_for'],
                        'model': Ridge(random_state = 1108, solver = 'lsqr', alpha = 0.00115411795019),
                        'scale':StandardScaler()
                        },                                              
                    },
                'possessions':{
                    'linsvm': {
                        'features': ['expected_poss_pg_for', 'pregame_poss_pg_for', '10_game_avg_10_g_Tweight_for_possessions-per-game', '10_game_avg_10_g_HAweight_for_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '1_game_avg_5_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '30_game_avg_10_g_Tweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game'],
                        'model': LinearSVR(random_state = 1108, C = 2.7208908100107254, epsilon=0),
                        'scale':MinMaxScaler()
                        },
                    'lightgbm': {
                        'features': ['10_game_avg_10_g_Tweight_for_possessions-per-game', 'expected_poss_pg_for', '1_game_avg_5_g_Tweight_for_possessions-per-game', 'pregame_poss_pg_for', '10_game_avg_10_g_HAweight_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 2800, colsample_bytree = 0.961267241448647, min_child_samples = 9, num_leaves = 18, subsample = 0.596425797228693, max_bin = 1844, learning_rate = 0.0025),
                        'scale':MinMaxScaler()
                        },
                    'lasso': {
                        'features': ['expected_poss_pg_for', 'pregame_poss_pg_for', '10_game_avg_10_g_Tweight_for_possessions-per-game', '10_game_avg_10_g_HAweight_for_possessions-per-game', '25_g_HAspread_for_possessions-per-game', '50_g_HAspread_for_possessions-per-game', '1_game_avg_50_g_Tweight_for_possessions-per-game', '1_game_avg_5_g_Tweight_for_possessions-per-game', '20_game_avg_10_g_HAweight_for_possessions-per-game', '30_game_avg_25_g_Tweight_for_possessions-per-game', '30_game_avg_10_g_Tweight_for_possessions-per-game', '30_game_avg_5_g_Tweight_for_possessions-per-game', '30_game_avg_50_g_HAweight_for_possessions-per-game'],
                        'model': Lasso(random_state = 1108, alpha = 0.0015540696172751227, max_iter = 2000),
                        'scale':StandardScaler()
                        },
                    },
                'target':{
                    'linsvm': {
                        'features': ['expected_ppp_for', 'pregame_ppp_for', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '50_g_HAspread_for_points-per-game', 'pregame_pts_pg_for', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_for_points-per-game`/`possessions-per-game', '5_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '1_game_avg_10_g_HAweight_for_points-per-game'],
                        'model': LinearSVR(random_state = 1108, C = 0.12071774068337349, epsilon=0),
                        'scale':MinMaxScaler()
                        },
                    'lightgbm': {
                        'features': ['1_game_avg_10_g_HAweight_for_points-per-game', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'pregame_ppp_for', 'expected_ppp_for', '50_g_HAspread_for_points-per-game', '10_game_avg_5_g_Tweight_for_points-per-game`/`possessions-per-game', '5_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', 'pregame_pts_pg_for'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 500, colsample_bytree = 0.780673078959247, min_child_samples = 5, num_leaves = 17, subsample = 0.607386007072246, max_bin = 1307, learning_rate = 0.0075),
                        'scale':MinMaxScaler()
                        },
                    'ridge': {
                        'features': ['expected_ppp_for', 'pregame_ppp_for', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '50_g_HAspread_for_points-per-game', 'pregame_pts_pg_for', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_HAweight_for_points-per-game`/`possessions-per-game', '10_game_avg_5_g_Tweight_for_points-per-game`/`possessions-per-game', '5_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game'],
                        'model': Ridge(random_state = 1108, solver = 'sparse_cg', alpha = 0.5619646264922706),
                        'scale':MinMaxScaler()
                        },
                    'lasso': {
                        'features': ['expected_ppp_for', 'pregame_ppp_for', '50_g_HAspread_for_points-per-game`/`possessions-per-game', 'expected_pts_pg_for', '50_g_HAspread_for_points-per-game', 'pregame_pts_pg_for', '25_g_HAspread_for_points-per-game`/`possessions-per-game', '20_game_avg_50_g_HAweight_for_points-per-game`/`possessions-per-game'],
                        'model': Lasso(random_state = 1108, alpha = 0.13526474650418807, max_iter = 2000),
                        'scale':StandardScaler()
                        },
                    },
                'offensive_stats': {                       
                    'linsvm': {
                        'features': ['expected_effective-field-goal-pct_for', '75_g_HAspread_for_floor-percentage', 'expected_turnovers-per-possession_for', 'expected_effective-field-goal-pct_for', 'pregame_effective-field-goal-pct_for', '100_g_HAspread_for_floor-percentage', '100_g_HAspread_for_offensive-efficiency', '30_g_HAspread_for_floor-percentage', 'expected_offensive-rebounding-pct_for', '100_g_HAspread_for_defensive-efficiency', 'expected_ftm-per-100-possessions_for', 'pregame_ftm-per-100-possessions_for', 'pregame_offensive-rebounding-pct_for'],
                        'model': LinearSVR(random_state = 1108, C = 0.0869486130678, epsilon=0),
                        'scale':RobustScaler()
                        },                          
                    'lightgbm': {
                        'features': ['75_g_HAspread_for_floor-percentage', '30_g_HAspread_for_floor-percentage', 'pregame_offensive-rebounding-pct_for', 'expected_offensive-rebounding-pct_for', 'expected_effective-field-goal-pct_for', 'expected_ftm-per-100-possessions_for', '100_g_HAspread_for_offensive-efficiency', '100_g_HAspread_for_defensive-efficiency', '100_g_HAspread_for_floor-percentage', 'pregame_effective-field-goal-pct_for', 'pregame_ftm-per-100-possessions_for', 'expected_effective-field-goal-pct_for', 'expected_turnovers-per-possession_for'],
                        'model': lgb.LGBMRegressor(random_state = 1108, n_estimators = 500, colsample_bytree = 0.808603278021336, min_child_samples = 176, num_leaves = 24, subsample = 0.678375514083654, max_bin = 1032, learning_rate = 0.01),
                        'scale': MinMaxScaler(),
                        }
                    }
                }
            }