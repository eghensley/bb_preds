def insert(sa, client, exploratory_name, statlist):
    import numpy as np
    from weighted_stats import pull_index
    import pandas as pd
    
    if sa == 'pts_scored':
        for_against = 'for'
    elif sa == 'pts_allowed':
        for_against = 'allowed'
        
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
    
    db = client['ncaa_bb']
           
    all_rolled_data = []
    try:
        latest = db['%s_%s'% (exploratory_name, for_against)].find_one(sort=[('_id', -1)])['_id']        
    except TypeError:
        latest = -1    

    total = db['weighted_%s_%s'% (exploratory_name, for_against)].count()
    checkpoints = np.linspace(0, total, 51)
    checkpoints = [str(int(i)) for i in checkpoints]
    advance = 0   
    
    print('---- Beginning %s %s Rolling Avg ----' % (exploratory_name, for_against))

    cursor= db['weighted_%s_%s'% (exploratory_name, for_against)].find({'_id' : {'$gt' : latest}}, no_cursor_timeout=True)
    for entry in cursor:   
        if entry['_game'] > 0 and db['weighted_%s_%s'% (exploratory_name, for_against)].find({'_id': {'$lt':entry['_id']}, '_team':entry['_team'], '_game': {"$lt": entry['_game']}, '_game': {"$gt": entry['_game'] - 2}}).count() > 0:
            rolled_data = {'_id': entry['_id'], '_date' : entry['_date'], '_team' : entry['_team'], 'stats' : {}}   
            for stat in statlist:
                for n in [1, 5, 10, 20, 30]:
                    rolled_data['stats']['%s_game_avg_%s' % (n,stat)] = []
            for n in [1, 5, 10, 20, 30]:
                for each in db['weighted_%s_%s'% (exploratory_name, for_against)].find({'_id': {'$lt':entry['_id']}, '_team':entry['_team'], '_game': {"$lt": entry['_game']}, '_game': {"$gt": entry['_game'] - (int(n) + 1)}}):
                    if len(each['stats']) > 0:                     
                        for s in each['stats'].keys():
                            rolled_data['stats']['%s_game_avg_%s' % (n,s)].append(each['stats'][s])
            for stat_name in rolled_data['stats'].keys():
                rolled_data['stats'][stat_name] = [x for x in rolled_data['stats'][stat_name] if x != 'NULL']
                if len(rolled_data['stats'][stat_name]) > 0:
                    rolled_data['stats'][stat_name] = np.mean(rolled_data['stats'][stat_name])
                else:
                    rolled_data['stats'][stat_name]  = 'NULL'
            all_rolled_data.append(rolled_data)
        advance += 1
        if str(advance) in checkpoints:
            print('-- %s %s Rolling Avg %f percent calculated --'  % (exploratory_name, for_against, int((float(advance)/float(total)) * 100)) )
            db['%s_%s'% (exploratory_name, for_against)].insert_many(all_rolled_data)
            all_rolled_data = []
    cursor.close()