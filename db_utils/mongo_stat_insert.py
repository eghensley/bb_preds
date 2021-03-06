def update(od, sa, client):
    import numpy as np
    import feature_lists
    import bb_odds
       
    db = client['ncaa_bb']

    if od == 'offensive_stats':
        for_against = 'for'
    elif od == 'defensive_stats':
        for_against = 'allow'
    elif od in ['possessions', 'target']:
        if sa == 'pts_scored':
            for_against = 'for'
        elif sa == 'pts_allowed':
            for_against = 'allow'
              
    print('---- Beginning %s %s Rolling Avg ----' % (od, sa))
    
    stat_list = []
    for stt in feature_lists.stats[sa][od]:
        if len(stt.split('_game_ha_spread_')) == 1:
            stat_list.append(stt.replace('_game_ha_weighted_', '_g_HAweight_%s_' % (for_against)).replace('_game_team_weighted_', '_g_Tweight_%s_' % (for_against)))
            
            
    for teamname in bb_odds.teamnames:
        if db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_')}, sort=[('_id', -1)]) is None:
            if db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find({'_team':teamname.replace(' ','_'), '_game': 2}).count() == 0:
                continue
            else:
                start_sum = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': 2})
                start_data = {'_id': start_sum['_id'], '_date' : start_sum['_date'], '_team' : start_sum['_team'], '_game': 2, 'stats' : {}}
                start_weight = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': 1})['stats']        
                for start_key, start_val in start_weight.items():
                    for ref_stat in stat_list:
                        if start_key == ref_stat.split('_game_avg_')[1]:
                            start_data['stats']['%s_game_avg_%s' % (ref_stat.split('_game_avg_')[0], start_key)] = start_val
                        ref_stat = None
                    start_key, start_val = None, None
                db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].insert_one(start_data)
                start_sum = None
                start_data = None
                
            if db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find({'_team':teamname.replace(' ','_'), '_game': 3}).count() == 0:
                continue
            else:
                start_sum = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': 3})
                start_data = {'_id': start_sum['_id'], '_date' : start_sum['_date'], '_team' : start_sum['_team'], '_game': 3, 'stats' : {}}
                start_weight = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': 2})['stats']        
                start_weight_1 = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': 1})['stats']        
                for start_key, start_val in start_weight.items():
                    for ref_stat in stat_list:
                        if start_key == ref_stat.split('_game_avg_')[1]:
                            if start_weight_1[start_key] != 'NULL' and start_val != 'NULL':
                                start_data['stats']['%s_game_avg_%s' % (ref_stat.split('_game_avg_')[0], start_key)] = np.mean([start_weight_1[start_key], start_val])
                            elif start_weight_1[start_key] != 'NULL':
                                start_data['stats']['%s_game_avg_%s' % (ref_stat.split('_game_avg_')[0], start_key)] = start_weight_1[start_key]
                            else:
                                start_data['stats']['%s_game_avg_%s' % (ref_stat.split('_game_avg_')[0], start_key)] = start_val
                        ref_stat = None
                    start_key, start_val = None, None
                db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].insert_one(start_data)
                start_sum = None
                start_data = None
        
        latest = db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_')}, sort=[('_game', -1)])['_game']
        limit = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_')}, sort=[('_game', -1)])['_game']        
        
        prev_weighted = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': latest - 1})['stats']
        roll_weight_prev1 = db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': latest})['stats']        
        roll_weight_prev2 = db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': latest-1})['stats']        
        
        stat_dict = {}
        for sst in stat_list:
            if sst in prev_weighted.keys() and sst in roll_weight_prev1.keys() and sst in roll_weight_prev2.keys():
                stat_dict[sst] = {'prev_weighted':prev_weighted[sst.split('_game_avg_')[1]], 'roll_weight_prev1':roll_weight_prev1[sst], 'roll_weight_prev2':roll_weight_prev2[sst]}
            elif sst in prev_weighted.keys() and sst in roll_weight_prev1.keys():            
                stat_dict[sst] = {'prev_weighted':prev_weighted[sst.split('_game_avg_')[1]], 'roll_weight_prev1':roll_weight_prev1[sst], 'roll_weight_prev2': "NULL"}
            elif sst in prev_weighted.keys() and sst in roll_weight_prev2.keys():
                stat_dict[sst] = {'prev_weighted':prev_weighted[sst.split('_game_avg_')[1]], 'roll_weight_prev1':0, 'roll_weight_prev2':roll_weight_prev2[sst]}
            elif sst in roll_weight_prev1.keys() and sst in roll_weight_prev2.keys():
                stat_dict[sst] = {'prev_weighted':0, 'roll_weight_prev1':roll_weight_prev1[sst], 'roll_weight_prev2':roll_weight_prev2[sst]}
            
            elif sst in prev_weighted.keys():            
                stat_dict[sst] = {'prev_weighted':prev_weighted[sst.split('_game_avg_')[1]], 'roll_weight_prev1':0, 'roll_weight_prev2': "NULL"}
            elif sst in roll_weight_prev2.keys():
                stat_dict[sst] = {'prev_weighted':0, 'roll_weight_prev1':0, 'roll_weight_prev2':roll_weight_prev2[sst]}
            elif sst in roll_weight_prev1.keys():
                stat_dict[sst] = {'prev_weighted':0, 'roll_weight_prev1':roll_weight_prev1[sst], 'roll_weight_prev2': "NULL"}
            
            else:
                stat_dict[sst] = {'prev_weighted':0, 'roll_weight_prev1':0, 'roll_weight_prev2': 0}
            
            if stat_dict[sst]['prev_weighted'] == "NULL":
                stat_dict[sst]['prev_weighted'] = 0
            if stat_dict[sst]['roll_weight_prev1'] == "NULL":
                stat_dict[sst]['roll_weight_prev1'] = 0 
            if stat_dict[sst]['roll_weight_prev2'] == "NULL":
                stat_dict[sst]['roll_weight_prev2'] = stat_dict[sst]['roll_weight_prev1']            
        
        
        prev_weighted, roll_weight_prev1, roll_weight_prev2  = None, None, None
        all_team_data = []

        update_mongo = False
        while latest < limit:
            update_mongo = True
            roll_sum = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': latest + 1})
            roll_data = {'_id': roll_sum['_id'], '_date' : roll_sum['_date'], '_team' : roll_sum['_team'], '_game': latest + 1, 'stats' : {}}
            roll_sum = None
                
            new_weighted = db['weighted_%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].find_one({'_team':teamname.replace(' ','_'), '_game': latest})['stats']
            for use_stat in stat_dict.keys():
                roll_len = int(use_stat.split('_game_avg_')[0])
                if use_stat.split('_game_avg_')[1] in new_weighted.keys():
                    new_weight = new_weighted[use_stat.split('_game_avg_')[1]]
                    if new_weight == 'NULL':
                        new_weight = 0
                else:
                    new_weight = 0
                
                if roll_len >= latest - 1:
                    prev_sum_1 = stat_dict[use_stat]['roll_weight_prev1']*(latest-1)
                    prev_sum_2 = stat_dict[use_stat]['roll_weight_prev2']*(latest-2)
                    prev_weight = stat_dict[use_stat]['prev_weighted']
                    new_avg = (prev_sum_1 - (prev_sum_2 - (prev_sum_1 - prev_weight)) + new_weight)/ latest
                else:
                    prev_sum_1 = stat_dict[use_stat]['roll_weight_prev1']*roll_len
                    prev_sum_2 = stat_dict[use_stat]['roll_weight_prev2']*roll_len
                    prev_weight = stat_dict[use_stat]['prev_weighted']
                    new_avg = (prev_sum_1 - (prev_sum_2 - (prev_sum_1 - prev_weight)) + new_weight)/ latest                    
                roll_data['stats'][use_stat] = new_avg
                stat_dict[use_stat]['prev_weighted'] = new_weight
                stat_dict[use_stat]['roll_weight_prev2'] = stat_dict[use_stat]['roll_weight_prev1']
                stat_dict[use_stat]['roll_weight_prev1'] = new_avg
                roll_len, new_weight, prev_weight, prev_sum_1, prev_sum_2, new_avg = None, None, None, None, None, None
            all_team_data.append(roll_data)
            latest = roll_data['_game']
        if update_mongo:
            db['%s_%s'% (sa.replace('_','-'), od.replace('_','-'))].insert_many(all_team_data)
            print('-- + %s'%(teamname))
        
