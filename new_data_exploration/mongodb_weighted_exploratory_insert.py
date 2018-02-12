from weighted_stats import ha_rolling_avg_weighted, team_rolling_avg_weighted, pull_index
import numpy as np
import pandas as pd
from datetime import datetime
import bb_odds

def latest_stat(cnx):
    cursor = cnx.cursor()
    cursor.execute('select max(statdate) from basestats')
    x = cursor.fetchall()
    cursor.close()
    return x
    
def aggregate_weighted_ha(index, stats, date_from):
    teamdict = {}
    for team in bb_odds.teamnames:
        teamdict[team.replace(' ','_')] = 0
    weighted_ha_scored = {}
    weighted_ha_allowed = {}
    for each in index.index:
        if datetime.strptime(each[:10], '%Y-%m-%d').date() > datetime.strptime(date_from, '%Y-%m-%d').date():
            weighted_ha_scored[each] = {'_id': int(index.loc[each]['id']), '_game': teamdict[each[10:]], '_date' : each[:10], '_team' : each[10:], 'stats' : {}}
            weighted_ha_allowed[each] = {'_id': int(index.loc[each]['id']), '_game': teamdict[each[10:]], '_date' : each[:10], '_team' : each[10:], 'stats' : {}}
            teamdict[each[10:]] += 1
    for stat in stats:
        for num in [5, 10, 25, 50]:
            ha_for, ha_against = ha_rolling_avg_weighted(stat, num, date_from)
            for key in ha_for.keys():
                weighted_ha_scored[key]['stats']['%s_g_HAweight_%s' % (num, stat)] = ha_for[key]['%s_g_HAweight_for_%s' % (num, stat)]
            for key in ha_against.keys():
                weighted_ha_allowed[key]['stats']['%s_g_HAweight_%s' % (num, stat)] = ha_against[key]['%s_g_HAweight_allow_%s' % (num, stat)]
            print('finished compiling %s game home field weighted %s' %(num, stat))
    return weighted_ha_scored, weighted_ha_allowed

def aggregate_weighted_team(weighted_team_scored, weighted_team_allowed, stats, date_from):
    for stat in stats:
        for num in [5, 10, 25, 50]:
            team_for, team_against = team_rolling_avg_weighted(stat, num, date_from)
            for key in team_for.keys():
                weighted_team_scored[key]['stats']['%s_g_Tweight_%s' % (num, stat)] = team_for[key]['%s_g_Tweight_for_%s' % (num, stat)]
            for key in team_against.keys():
                weighted_team_allowed[key]['stats']['%s_g_Tweight_%s' % (num, stat)] = team_against[key]['%s_g_Tweight_allow_%s' % (num, stat)]
            print('finished compiling %s game team weighted %s' %(num, stat))
    return weighted_team_scored, weighted_team_allowed

def aggregate_hfa_spread(index, stats, date_from):
    weighted_hfaspread_scored = {}
    weighted_hfaspread_allowed = {}
    for each in index.index:
        if datetime.strptime(each[:10], '%Y-%m-%d').date() > datetime.strptime(date_from, '%Y-%m-%d').date():
            weighted_hfaspread_scored[each] = {'_id': int(index.loc[each]['id']), '_date' : each[:10], '_team' : each[10:], 'stats' : {}}
            weighted_hfaspread_allowed[each] = {'_id': int(index.loc[each]['id']), '_date' : each[:10], '_team' : each[10:], 'stats' : {}}
    for stat in stats:
        for num in [5, 10, 25, 50]:
            hfa_for, hfa_against = ha_rolling_avg_weighted(stat, num, date_from)
            for key in hfa_for.keys():
                weighted_hfaspread_scored[key]['stats']['%s_g_HAspread_%s' % (num, stat)] = hfa_for[key]['%s_g_HAspread_for_%s' % (num, stat)]
            for key in hfa_against.keys():
                weighted_hfaspread_allowed[key]['stats']['%s_g_HAspread_%s' % (num, stat)] = hfa_against[key]['%s_g_HAspread_allow_%s' % (num, stat)]
            print('finished compiling %s game home field weighted %s' %(num, stat))
    return weighted_hfaspread_scored, weighted_hfaspread_allowed
    
def insert(client, mysql_client, exploratory_name, statlist):
    index_dict = pd.DataFrame()
    index_list = []
    progress_list = []
    progress = 0
    for teamname,allowdate in np.array(pull_index()):
        index_list.append(str(allowdate)+teamname.replace(' ', '_'))
        progress_list.append(progress)
        progress += 1
    index_dict['idx'] = index_list
    index_dict['id'] = progress_list
    index_dict['id'].astype(int)
    index_dict = index_dict.set_index('idx')
    print('Set Index')
    teamname = None
    progress = None
    allowdate = None   
    db = client['ncaa_bb']
    try:
        latest_weighted = db['weighted_pts-scored_%s' % (exploratory_name)].find_one(sort=[('_id', -1)])['_date']
    except TypeError:
        latest_weighted = "2009-01-01"
    try:
        latest_spread = db['hfa-spread_pts-scored_%s' % (exploratory_name)].find_one(sort=[('_id', -1)])['_date']
    except TypeError:
        latest_spread = "2009-01-01"    
        
    if datetime.strptime(latest_weighted, '%Y-%m-%d').date() < latest_stat(mysql_client)[0][0] or datetime.strptime(latest_spread, '%Y-%m-%d').date() < latest_stat(mysql_client)[0][0]:  
        for_data, against_data = aggregate_weighted_ha(index_dict, statlist, latest_weighted)
        for_data, against_data = aggregate_weighted_team(for_data, against_data, statlist, latest_weighted)

        for_data = [for_data[key] for key in for_data.keys()]
        against_data = [against_data[key] for key in against_data.keys()]
        db['weighted_pts-scored_%s' % (exploratory_name)].insert_many(for_data)
        print('------------- completed uploading weighted points scored ---------------')
        for_data = None
        db['weighted_pts-allowed_%s' % (exploratory_name)].insert_many(against_data)
        print('------------- completed uploading weighted points allowed ---------------')
        against_data = None        
        
        for_data, against_data = aggregate_hfa_spread(index_dict, statlist, latest_weighted)
        for_data = [for_data[key] for key in for_data.keys()]
        against_data = [against_data[key] for key in against_data.keys()]


        db['hfa-spread_pts-scored_%s' % (exploratory_name)].insert_many(for_data)
        print('------------- completed uploading weighted points scored ---------------')
        for_data = None
        db['hfa-spread_pts-allowed_%s' % (exploratory_name)].insert_many(against_data)
        print('------------- completed uploading weighted points allowed ---------------')
        against_data = None

    else:
        print('Already Up To Date')