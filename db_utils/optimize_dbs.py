import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
while cur_path.split('/')[-1] != 'stats_bb':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(-1, os.path.join(cur_path, 'model_conf'))

import update_dbs
import feature_lists

def step_1(sa, od, client):
#    sa, od, client = 'pts_scored', 'possessions', update_dbs.mongodb_client
    if od == 'possessions':
        cursor_sa = ''
        if sa == 'pts_scored':
            cursor_od = 'possessions_for'
        elif sa == 'pts_allowed':
            cursor_od = 'possessions_allowed'
    elif od == 'target':
        cursor_od = 'targets'
        cursor_sa = sa.replace('_','-')+'_'

    if sa == 'pts_scored':
        for_against = 'for'
    elif sa == 'pts_allowed':
        for_against = 'allow'    
        
    db = client['ncaa_bb']           
    all_rolled_data = []  

    features = [i.split('_game_avg_')[1].replace('_game','_g').replace('_team_','_T').replace('_ha_','_HA').replace('weighted_','weight_') for i in feature_lists.stats[sa][od] if i.find('_game_avg') != -1]
    features = list(set(features))
    print('---- Beginning %s %s Weighted Optimization ----' % (od, sa))
    cursor= db['weighted_%s%s'% (cursor_sa, cursor_od)].find()
    for entry in cursor: 
        if len(entry['stats']) > 0:
            rolled_data = {'_id': entry['_id'], '_game': entry['_game'], '_date' : entry['_date'], '_team' : entry['_team'], 'stats' : {}}   
            for stat in features:
                if stat in entry['stats']:
                    rolled_data['stats'][stat.replace('weight_', 'weight_%s_' % (for_against))] = entry['stats'][stat]
            all_rolled_data.append(rolled_data)
    db['weighted_%s_%s'% (sa.replace('_','-'), od)].insert_many(all_rolled_data)
    cursor.close()
    all_rolled_data = None
    print('---- ...Finshed %s %s Weighted Optimization ----' % (od, sa))
    
    all_rolled_data = []      
    features = [i.replace('_game_ha_spread','_g_HAspread') for i in feature_lists.stats[sa][od] if i.find('_game_ha_spread_') != -1]    
    features = list(set(features))
    print('---- Beginning %s %s HFA Optimization ----' % (od, sa))
    cursor= db['hfa-spread_%s%s' % (cursor_sa, cursor_od)].find()
    for entry in cursor: 
        if len(entry['stats']) > 0:
            rolled_data = {'_id': entry['_id'], '_date' : entry['_date'], '_team' : entry['_team'], 'stats' : {}}   
            for stat in features:
                if stat in entry['stats']:
                    rolled_data['stats'][stat.replace('weight_', 'weight_%s_' % (for_against))] = entry['stats'][stat]
            all_rolled_data.append(rolled_data)
    db['hfa-spread_%s_%s'% (sa.replace('_','-'), od)].insert_many(all_rolled_data)
    cursor.close()
    print('---- ...Finshed %s %s HFA Optimization ----' % (od, sa))    
    
def step_2(sa, od, client):
#    sa, od, client = 'pts_scored', 'possessions', update_dbs.mongodb_client
    if od == 'possessions':
        if sa == 'pts_scored':
            cursor_od = 'possessions_for'
        elif sa == 'pts_allowed':
            cursor_od = 'possessions_allowed'
    elif od == 'target':
        cursor_od = sa.replace('_','-')+'_targets'
        
    if sa == 'pts_scored':
        for_against = 'for'
    elif sa == 'pts_allowed':
        for_against = 'allow'
        
    db = client['ncaa_bb']           
        
    all_rolled_data = []      
    features = [i.replace('_game_ha_weighted', '_g_HAweight').replace('_game_team_weighted','_g_Tweight') for i in feature_lists.stats[sa][od] if i.find('_game_avg') != -1]   
    features = list(set(features))
    print('---- Beginning %s %s Rolled Optimization ----' % (od, sa))
    cursor= db[cursor_od].find()
    for entry in cursor: 
        if len(entry['stats']) > 0:
            rolled_data = {'_id': entry['_id'], '_date' : entry['_date'], '_team' : entry['_team'], 'stats' : {}}   
            for stat in features:
                if stat in entry['stats']:
                    rolled_data['stats'][stat.replace('weight_', 'weight_%s_' % (for_against))] = entry['stats'][stat]
            all_rolled_data.append(rolled_data)

    db['%s_%s'% (sa.replace('_','-'), od)].insert_many(all_rolled_data)
    cursor.close()
    print('---- ...Finished %s %s Rolled Optimization ----' % (od, sa))

if __name__ == '__main__':
    for od in ['possessions', 'target']:
        for sa in ['pts_scored', 'pts_allowed']:
            step_1(sa, od, update_dbs.mongodb_client)
            step_2(sa, od, update_dbs.mongodb_client)
