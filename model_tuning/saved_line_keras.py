#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:25:22 2018

@author: eric.hensleyibm.com
"""

scaler = StandardScaler()
def nn_model():
    model = Sequential()
    model.add(Dense(87, input_dim=36, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(44, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=.05, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
    return model 
num_epochs = 30
['20_game_avg_30_g_HAweight_allow_fta-per-fga',
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
'-20_game_avg_50_g_Tweight_for_block-pct', 'pca_line', 'tsvd_line', 'lasso_line', 'lightgbm_line', 'ridge_line', 'vegas_line']
