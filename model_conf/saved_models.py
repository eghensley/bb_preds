import os, sys

try:
    import lightgbm as lgb
except ImportError:
    sys.path.insert(-1, "/home/eric/LightGBM/python-package")
    import lightgbm as lgb
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import LinearSVR


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
        'points':{  # RIDGE
                    '+pts': {
                        'features': ['-lightgbm_team', '+lasso_possessions', '+ridge_team', '+linsvm_team', '+lightgbm_team', '+lasso_team', '+ridge_all', '+lightgbm_all', '+lightgbm_target', '+lasso_target', '-ridge_target', '-lightgbm_target', '-lasso_target', '-linsvm_possessions', '-lightgbm_possessions', '-lasso_possessions', '-ridge_all', '-linsvm_all', '-linsvm_team', '-linsvm_target', '+ridge_possessions', '-rest', '+lightgbm_possessions', '+rest'],
                        'model': Pipeline([('scale',RobustScaler()), ('clf',Ridge(random_state = 1108, solver = 'saga', alpha = 0.5904378324937618))]),
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
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.0011089952827))]),
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
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.001))]),
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
                        'model': Pipeline([('scale',RobustScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.00125925622966))]),
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
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.0015540696172751227))]),
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
                        'model': Pipeline([('scale',StandardScaler()), ('clf',Lasso(random_state = 1108, alpha = 0.13526474650418807))]),
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