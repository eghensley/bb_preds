from weighted_stats import ha_rolling_avg_weighted, team_rolling_avg_weighted, pull_index
import feature_lists
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
    
def aggregate_weighted_ha(index, fa, stats, date_from):
    weighted_ha_data = {}
    for each in index.index:
        if datetime.strptime(each[:10], '%Y-%m-%d').date() > datetime.strptime(date_from, '%Y-%m-%d').date():
            weighted_ha_data[each] = {'_id': int(index.loc[each]['id']), '_game': int(index.loc[each]['game']), '_date' : each[:10], '_team' : each[10:], 'stats' : {}}
    for each in stats:
        num, stat = each
        ha_for, ha_against = ha_rolling_avg_weighted(stat[0], int(num[0]), date_from)
        if fa == 'for':
            ha_against = None
            use_stat = ha_for
            ha_for = None
        elif fa == 'allow':
            ha_for = None
            use_stat = ha_against
            ha_against = None
        for key in use_stat.keys():
            weighted_ha_data[key]['stats']['%s_g_HAweight_%s_%s' % (num[0], fa, stat[0])] = use_stat[key]['%s_g_HAweight_%s_%s' % (num[0], fa, stat[0])]
        use_stat = None
        print('finished compiling %s game home field weighted %s' %(num[0], stat[0]))
    return weighted_ha_data

def aggregate_weighted_team(fa, weighted_team_data, stats, date_from):
    for each in stats:
        num, stat = each
        team_for, team_against = team_rolling_avg_weighted(stat[0], int(num[0]), date_from)
        if fa == 'for':
            team_against = None
            use_stat = team_for
            team_for = None
        elif fa == 'allow':
            team_for = None
            use_stat = team_against
            team_against = None
        for key in use_stat.keys():
            weighted_team_data[key]['stats']['%s_g_Tweight_%s_%s' % (num[0], fa, stat[0])] = use_stat[key]['%s_g_Tweight_%s_%s' % (num[0], fa, stat[0])]
        use_stat = None
        print('finished compiling %s game team weighted %s' %(num[0], stat[0]))
    return weighted_team_data

def aggregate_hfa_spread(index, fa, stats, date_from):
    ha_spread_data = {}
    for each in index.index:
        if datetime.strptime(each[:10], '%Y-%m-%d').date() > datetime.strptime(date_from, '%Y-%m-%d').date():
            ha_spread_data[each] = {'_id': int(index.loc[each]['id']), '_game': int(index.loc[each]['game']), '_date' : each[:10], '_team' : each[10:], 'stats' : {}}
    for each in stats:
        num, stat = each
        team_for, team_against = ha_rolling_avg_weighted(stat, int(num), date_from)
        if fa == 'for':
            team_against = None
            use_stat = team_for
            team_for = None
        elif fa == 'allow':
            team_for = None
            use_stat = team_against
            team_against = None
        for key in use_stat.keys():
            ha_spread_data[key]['stats']['%s_g_HAspread_%s_%s' % (num, fa, stat)] = use_stat[key]['%s_g_HAspread_%s_%s' % (num, fa, stat)]
        use_stat = None
        print('finished compiling %s home field (dis)advantage for %s' %(num, stat))
    return ha_spread_data
    
def insert(od, sa, client, mysql_client):
    if od == 'offensive_stats':
        for_against = 'for'
    elif od == 'defensive_stats':
        for_against = 'allow'
    elif od in ['possessions', 'target']:
        if sa == 'pts_scored':
            for_against = 'for'
        elif sa == 'pts_allowed':
            for_against = 'allow'
    
    index_dict = pd.DataFrame()
    index_list = []
    tmgm_list = []
    progress_list = []
    progress = 0
    
    team_game_dict = {}
    for tmnm in bb_odds.teamnames:
        team_game_dict[tmnm.replace(' ', '_')] = 0
        
    for teamname,allowdate in np.array(pull_index()):
        index_list.append(str(allowdate)+teamname.replace(' ', '_'))
        progress_list.append(progress)
        tmgm_list.append(team_game_dict[teamname.replace(' ', '_')])
        progress += 1
        team_game_dict[teamname.replace(' ', '_')] += 1
    index_dict['idx'] = index_list
    index_dict['id'] = progress_list
    index_dict['id'].astype(int)
    index_dict['game'] = tmgm_list
    index_dict['game'].astype(int)
    index_dict = index_dict.set_index('idx')
    print('Set Index')
    teamname = None
    progress = None
    allowdate = None   
    db = client['ncaa_bb']
    try:
        latest_weighted = db['weighted_%s_%s'% (sa.replace('_', '-'), od.replace('_', '-'))].find_one(sort=[('_id', -1)])['_date']
    except TypeError:
        print('***Empty weighted %s %s db found. You can bootstrap the historical data from GitHub.' % (sa, od.replace('_', '-')))
        latest_weighted = "2009-01-01"
    try:
        latest_spread = db['hfa-spread_%s_%s'% (sa.replace('_', '-'), od.replace('_', '-'))].find_one(sort=[('_id', -1)])['_date']
    except TypeError:
        print('***Empty homefield advantage %s %s db found.  You can bootstrap the historical data from GitHub.' % (sa, od.replace('_', '-')))
        latest_spread = "2009-01-01"        
    if datetime.strptime(latest_weighted, '%Y-%m-%d').date() < latest_stat(mysql_client)[0][0] or datetime.strptime(latest_spread, '%Y-%m-%d').date() < latest_stat(mysql_client)[0][0]:
        statlist = feature_lists.stats[sa][od]
        weighted_stats = []
        hfaspread_stats = []
        for stat in statlist:
            try:
                if stat.split('_game_avg_')[1] not in weighted_stats:
                    weighted_stats.append(stat.split('_game_avg_')[1])
            except IndexError:
                n, s = stat.split('_game_ha_spread_')
                if stat not in hfaspread_stats:
                    hfaspread_stats.append((n, s))
        stat = None
        weighted_ha_stats = []
        weighted_team_stats = []
        for stat in weighted_stats:
            try:
                n, s = stat.split('_game_ha_weighted_')
                if (n.split('_'), s.split('_')) not in weighted_ha_stats:
                    weighted_ha_stats.append((n.split('_'), s.split('_')))
            except ValueError:
                n, s = stat.split('_game_team_weighted_')
                if (n.split('_'), s.split('_')) not in weighted_team_stats:
                    weighted_team_stats.append((n.split('_'), s.split('_')))
        n = None
        s = None
        stat = None
        
        if datetime.strptime(latest_weighted, '%Y-%m-%d').date() < latest_stat(mysql_client)[0][0]:
            data = aggregate_weighted_ha(index_dict, for_against, weighted_ha_stats, latest_weighted)
            data = aggregate_weighted_team(for_against, data, weighted_team_stats, latest_weighted)
            data = [data[key] for key in data.keys()]
            for each in data:
                type(each['_id'])
                break
            db['weighted_%s_%s'% (sa.replace('_', '-'), od.replace('_', '-'))].insert_many(data)
            print('------------- completed uploading weighted %s %s-stats ---------------' %(sa, od))
            data = None
        if datetime.strptime(latest_spread, '%Y-%m-%d').date() < latest_stat(mysql_client)[0][0]:
            data = aggregate_hfa_spread(index_dict, for_against, hfaspread_stats, latest_spread)
            data = [data[key] for key in data.keys()]
            db['hfa-spread_%s_%s'% (sa.replace('_', '-'), od.replace('_', '-'))].insert_many(data)
            print('------------- completed uploading hfa spread %s %s-stats ---------------' %(sa, od))
            data = None
    else:
        print('Already Up To Date')
