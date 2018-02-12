import pandas as pd
import numpy as np

def pull_possessions(od, cnx): 
    print('Loading Possession Data')
    if od == 'pts_scored':
        selector = ['favorite', 'underdog']
    elif od == 'pts_allowed':
        selector = ['underdog', 'favorite']

    cursor = cnx.cursor()
    query = "select oddsdate, favorite, `possessions-per-game` from oddsdata join basestats on oddsdata.oddsdate = basestats.statdate and oddsdata.%s = basestats.teamname" % (selector[0])
    labels = ["oddsdate", "favorite","possessions"]
    cursor.execute(query)
    favdata = pd.DataFrame(cursor.fetchall(), columns = labels)
    favid = []
    for date, name, score in np.array(favdata):
        favid.append(str(date)+name.replace(' ','_'))
    favdata['idx'] = favid
    favdata = favdata.set_index('idx')
    favdata = favdata['possessions']
    query = "select oddsdate, underdog,`possessions-per-game` from oddsdata join basestats on oddsdata.oddsdate = basestats.statdate and oddsdata.%s = basestats.teamname" % (selector[1])
    labels = ["oddsdate", "underdog","possessions"]
    cursor.execute(query)
    dogdata = pd.DataFrame(cursor.fetchall(), columns = labels)
    favid = []
    for date, name, score in np.array(dogdata):
        favid.append(str(date)+name.replace(' ','_'))
    dogdata['idx'] = favid
    dogdata = dogdata.set_index('idx')
    dogdata = dogdata['possessions']
    data = favdata.append(dogdata)
    print('...Possession Data Loaded')
    
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.replace('NULL', np.nan)
    data = data.dropna(how = 'any')
    return data

def pull_odds_data(cnx):
    cursor = cnx.cursor()
    query = 'select * from oddsdata'
    cursor.execute(query)
    oddsdata = pd.DataFrame(cursor.fetchall() , columns = ['date', 'fav', 'dog', 'line', 'line-juice', 'overunder', 'ou-juice', 'fav-ml', 'dog-ml', 'fav-score', 'dog-score', 'ha'])
    t1idx = []
    t2idx = []
    for d,f,dog,l,lj, ou, ouj, fml, dml, fs, ds, ha in np.array(oddsdata):
        t1idx.append(str(d)+f.replace(' ', '_'))
        t2idx.append(str(d)+dog.replace(' ', '_'))
    oddsdata['fav_idx'] = t1idx
    oddsdata['dog_idx'] = t2idx
    return oddsdata

def pull_days_rest(cnx):
    cursor = cnx.cursor()
    query = 'SELECT teamname, date, datediff(date, (select max(date) from gamedata as gd1 where gd1.teamname = gd.teamname and gd1.date < gd.date))  FROM ncaa_bb.gamedata as gd;'
    cursor.execute(query)
    oddsdata = pd.DataFrame(cursor.fetchall() , columns = ['name', 'date', 'rest'])
    idx = []
    for n,d,r in np.array(oddsdata):
        idx.append(str(d)+n.replace(' ', '_'))
    oddsdata['idx'] = idx
    oddsdata = oddsdata.set_index('idx')
    oddsdata['rest'] = oddsdata.rest.apply(lambda x: 10 if x > 10 else x)
    oddsdata = oddsdata['rest']
    return oddsdata
    
def pull_train_index(cnx):
    cursor = cnx.cursor()
    query = 'select date, teamname from gamedata where date < "2017-11-1"'
    cursor.execute(query)
    indexdata = pd.DataFrame(cursor.fetchall(), columns = ['date', 'name'])
    idx = []
    for d,n in np.array(indexdata):
        idx.append(str(d)+n.replace(' ','_'))
    return idx 
    
def offensive_points(cnx):
    cursor = cnx.cursor()
    query = 'SELECT `teamname`, `date`, `lightgbm_all`, `ridge_all`, `lasso_team`, `lightgbm_team`, `linsvm_team`, `ridge_team`, `lasso_possessions`, `lightgbm_possessions`, `ridge_possessions`, `lasso_ppp`, `lightgbm_ppp` FROM offensive_preds as op;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'lightgbm_all', 'ridge_all', 'lasso_team', 'lightgbm_team', 'linsvm_team', 'ridge_team', 'lasso_possessions', 'lightgbm_possessions', 'ridge_possessions', 'lasso_ppp', 'lightgbm_ppp'])
    idx = []
    for name, date in np.array(data[['teamname', 'date']]):
        idx.append(str(date)+name.replace(' ','_'))
    data['idx'] = idx
    data.set_index('idx')
    del data['teamname']
    del data['date']
    
    rest = pull_days_rest(cnx)
    data = data.join(rest, how = 'inner')
    points = pull_pts('offensive', cnx)
    data = data.join(points, how = 'inner')
    return data

def score(cnx):
    off_data = offensive_points(cnx)
    del off_data['idx']
    off_data = off_data.rename(columns = {i:'+'+i for i in list(off_data)})
    def_data = defensive_points(cnx)
    del def_data['idx']
    def_data = def_data.rename(columns = {i:'-'+i for i in list(def_data)})
    del def_data['-pts']
    def_data *= -1
    cursor = cnx.cursor()
    query = 'SELECT * from gamedata;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'opponent', 'location'])
    idx_switch = {}
    for t,d,o,l in np.array(data):
        idx_switch[str(d)+t.replace(' ', '_')] = str(d)+o.replace(' ', '_')
    idx = []
    for idxx in def_data.index:
        idx.append(idx_switch[idxx])
    def_data['idx'] = idx
    def_data = def_data.set_index('idx')
    data = def_data.join(off_data)   
    return data
    
def offensive_pace(cnx):
    cursor = cnx.cursor()
    query = 'SELECT `teamname`, `date`, `lasso_possessions`, `lightgbm_possessions`, `ridge_possessions` FROM offensive_preds as op;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'lasso_possessions', 'lightgbm_possessions', 'ridge_possessions'])
    idx = []
    for name, date in np.array(data[['teamname', 'date']]):
        idx.append(str(date)+name.replace(' ','_'))
    data['idx'] = idx
    data.set_index('idx')
    del data['teamname']
    del data['date']
    
    points = pull_possessions('pts_scored', cnx)
    data = data.join(points, how = 'inner')
    return data    
    
def defensive_pace(cnx):
    cursor = cnx.cursor()
    query = 'SELECT teamname, date, lasso_possessions, lightgbm_possessions, linsvm_possessions FROM defensive_preds as dp;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'lasso_possessions', 'lightgbm_possessions', 'linsvm_possessions'])
    idx = []
    for name, date in np.array(data[['teamname', 'date']]):
        idx.append(str(date)+name.replace(' ','_'))
    data['idx'] = idx
    data.set_index('idx')
    del data['teamname']
    del data['date']
    
    points = pull_possessions('pts_allowed', cnx)
    data = data.join(points, how = 'inner')
    return data 

  
def defensive_points(cnx):
    cursor = cnx.cursor()
    query = 'SELECT teamname, date, lightgbm_team, linsvm_team, linsvm_all, ridge_all, lasso_possessions, lightgbm_possessions, linsvm_possessions, lasso_ppp, lightgbm_ppp, linsvm_ppp, ridge_ppp FROM defensive_preds as dp;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'lightgbm_team', 'linsvm_team', 'linsvm_all', 'ridge_all', 'lasso_possessions', 'lightgbm_possessions', 'linsvm_possessions', 'lasso_ppp', 'lightgbm_ppp', 'linsvm_ppp', 'ridge_ppp'])
    idx = []
    for name, date in np.array(data[['teamname', 'date']]):
        idx.append(str(date)+name.replace(' ','_'))
    data['idx'] = idx
    data.set_index('idx')
    del data['teamname']
    del data['date']
    
    rest = pull_days_rest(cnx)
    data = data.join(rest, how = 'inner')
    points = pull_pts('defensive', cnx)
    data = data.join(points, how = 'inner')
    return data

def offensive_ppp(cnx):
    cursor = cnx.cursor()
    query = 'SELECT `teamname`, `date`, `lasso_team`, `lightgbm_team`, `linsvm_team`, `ridge_team`, `lasso_ppp`, `lightgbm_ppp` FROM offensive_preds as op;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'lasso_team', 'lightgbm_team', 'linsvm_team', 'ridge_team', 'lasso_ppp', 'lightgbm_ppp'])
    idx = []
    for name, date in np.array(data[['teamname', 'date']]):
        idx.append(str(date)+name.replace(' ','_'))
    data['idx'] = idx
    data.set_index('idx')
    del data['teamname']
    del data['date']
    
    rest = pull_days_rest(cnx)
    data = data.join(rest, how = 'inner')
    points = pull_ppp('offensive', cnx)
    data = data.join(points, how = 'inner')
    return data    

def defensive_ppp(cnx):
    cursor = cnx.cursor()
    query = 'SELECT teamname, date, lightgbm_team, linsvm_team, lasso_ppp, lightgbm_ppp, linsvm_ppp, ridge_ppp FROM defensive_preds as dp;'
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns = ['teamname', 'date', 'lightgbm_team', 'linsvm_team', 'lasso_ppp', 'lightgbm_ppp', 'linsvm_ppp', 'ridge_ppp'])
    idx = []
    for name, date in np.array(data[['teamname', 'date']]):
        idx.append(str(date)+name.replace(' ','_'))
    data['idx'] = idx
    data.set_index('idx')
    del data['teamname']
    del data['date']
    
    rest = pull_days_rest(cnx)
    data = data.join(rest, how = 'inner')
    points = pull_ppp('defensive', cnx)
    data = data.join(points, how = 'inner')
    return data
    
def pull_ppp(od, cnx): 
    print('Loading Target Data')
    if od == 'pts_scored':
        selector = ['favscore', 'dogscore', 'favorite', 'underdog']
    elif od == 'pts_allowed':
        selector = ['dogscore', 'favscore', 'underdog', 'favorite']

    cursor = cnx.cursor()
    query = "select oddsdate, favorite,%s/`possessions-per-game` from oddsdata join basestats on oddsdata.oddsdate = basestats.statdate and oddsdata.%s = basestats.teamname" % (selector[0], selector[2])
    labels = ["oddsdate", "favorite","ppp"]
    cursor.execute(query)
    favdata = pd.DataFrame(cursor.fetchall(), columns = labels)
    favid = []
    for date, name, score in np.array(favdata):
        favid.append(str(date)+name.replace(' ','_'))
    favdata['idx'] = favid
    favdata = favdata.set_index('idx')
    favdata = favdata['ppp']
    query = "select oddsdate, underdog,%s/`possessions-per-game` from oddsdata join basestats on oddsdata.oddsdate = basestats.statdate and oddsdata.%s = basestats.teamname" % (selector[1], selector[3])
    labels = ["oddsdate", "underdog","ppp"]
    cursor.execute(query)
    dogdata = pd.DataFrame(cursor.fetchall(), columns = labels)
    favid = []
    for date, name, score in np.array(dogdata):
        favid.append(str(date)+name.replace(' ','_'))
    dogdata['idx'] = favid
    dogdata = dogdata.set_index('idx')
    dogdata = dogdata['ppp']
    data = favdata.append(dogdata)
    print('...Target Data Loaded')
    
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.replace('NULL', np.nan)
    data = data.dropna(how = 'any')
    return data
    
def pull_pts(od, cnx): 
    print('Loading Target Data')
    if od == 'offensive':
        selector = ['favscore', 'dogscore', 'favorite', 'underdog']
    elif od == 'defensive':
        selector = ['dogscore', 'favscore', 'underdog', 'favorite']

    cursor = cnx.cursor()
    query = "select oddsdate, favorite,%s from oddsdata join basestats on oddsdata.oddsdate = basestats.statdate and oddsdata.%s = basestats.teamname" % (selector[0], selector[2])
    labels = ["oddsdate", "favorite","pts"]
    cursor.execute(query)
    favdata = pd.DataFrame(cursor.fetchall(), columns = labels)
    favid = []
    for date, name, score in np.array(favdata):
        favid.append(str(date)+name.replace(' ','_'))
    favdata['idx'] = favid
    favdata = favdata.set_index('idx')
    favdata = favdata['pts']
    query = "select oddsdate, underdog,%s from oddsdata join basestats on oddsdata.oddsdate = basestats.statdate and oddsdata.%s = basestats.teamname" % (selector[1], selector[3])
    labels = ["oddsdate", "underdog","pts"]
    cursor.execute(query)
    dogdata = pd.DataFrame(cursor.fetchall(), columns = labels)
    favid = []
    for date, name, score in np.array(dogdata):
        favid.append(str(date)+name.replace(' ','_'))
    dogdata['idx'] = favid
    dogdata = dogdata.set_index('idx')
    dogdata = dogdata['pts']
    data = favdata.append(dogdata)
    print('...Target Data Loaded')
    
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.replace('NULL', np.nan)
    data = data.dropna(how = 'any')
    return data
    
def pull_model_features(y, x, mongodb_client):
#    y,x,mongodb_client = 'pts_scored', 'offensive_stats', mongodb_client
    print('Loading Weighted Features')
    db = mongodb_client['ncaa_bb']
    weighted_stats = {}
    for each in db['%s_%s'% (y.replace('_','-'), x.replace('_','-'))].find():
        weighted_stats[each['_date']+each['_team']] = each['stats']
    weighted_stats = pd.DataFrame.from_dict(weighted_stats)
    weighted_stats = weighted_stats.T
    print('...Weighted Features Loaded')

    print('Loading Home/Away Features')
    db = mongodb_client['ncaa_bb']
    hfa_stats = {}
    for each in db['hfa-spread_%s_%s'% (y.replace('_','-'), x.replace('_','-'))].find():
        hfa_stats[each['_date']+each['_team']] = each['stats']
    hfa_stats = pd.DataFrame.from_dict(hfa_stats)
    hfa_stats = hfa_stats.T
    print('...Home/Away Features Loaded')
    
    if y == 'pts_scored':
        elo_tag = '_for'
    elif y == 'pts_allowed':
        elo_tag = '_allowed'
    print('Loading Elo Features')
    if x in ['defensive_stats', 'offensive_stats']:
        elo_stats = {}
        for each in db.elo_four_features.find():
            elo_stats[each['_date']+each['_team'].replace(' ', '_')] = each['stats'][x]
        elo_stats = pd.DataFrame.from_dict(elo_stats)
        elo_stats = elo_stats.T
        elo_stats = elo_stats.rename(columns = {i:i+elo_tag for i in list(elo_stats)})
    elif x in ['possessions', 'target']:
        elo_stats = {}
        for each in db.elo_four_features.find():
            elo_stats[each['_date']+each['_team'].replace(' ', '_')] = each['stats'][y][x]
        elo_stats = pd.DataFrame.from_dict(elo_stats)
        elo_stats = elo_stats.T
        elo_stats = elo_stats.rename(columns = {i:i+elo_tag for i in list(elo_stats)})
    print('...Elo Features Loaded')
    
    x_data = weighted_stats.join(elo_stats, how = 'inner')    
    x_data = x_data.join(hfa_stats, how = 'inner')
    x_data = x_data.replace([np.inf, -np.inf], np.nan)
    x_data = x_data.replace('NULL', np.nan)
    x_data = x_data.dropna(how = 'any')
    return x_data
