import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
while cur_path.split('/')[-1] != 'bb_preds':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(-1, os.path.join(cur_path, 'model_conf'))
sys.path.insert(-1, cur_path)


def update(cnx):
    import requests
    from lxml import html
    import datetime
    import pandas as pd
    
    print('Starting NCAA Basketball Stats Update')
    #ratinglist = ['predictive-by-other', 'home-by-other', 'away-by-other', 'neutral-by-other', 
    #      'home-adv-by-other', 'schedule-strength-by-other', 'future-sos-by-other', 
    #      'season-sos-by-other', 'sos-basic-by-other', 'in-conference-sos-by-other',
    #      'non-conference-sos-by-other', 'last-5-games-by-other', 'last-10-games-by-other',
    #      'in-conference-by-other', 'non-conference-by-other', 'luck-by-other',
    #      'consistency-by-other', 'vs-1-10-by-other', 'vs-11-25-by-other', 'vs-26-40-by-other',
    #      'vs-41-75-by-other', 'vs-76-120-by-other', 'first-half-by-other', 'second-half-by-other']
    
    #statslist = ['points-per-game', 'offensive-efficiency', 'floor-percentage', 'points-from-2-pointers', 'points-from-3-pointers', 
    #             'percent-of-points-from-2-pointers', 'percent-of-points-from-3-pointers', 'percent-of-points-from-free-throws', 
    #             'opponent-points-per-game', 'defensive-efficiency', 'opponent-floor-percentage', 'opponent-points-from-2-pointers',
    #             'opponent-points-from-3-pointers', 'opponent-percent-of-points-from-2-pointers', 'opponent-percent-of-points-from-3-pointers',
    #             'opponent-percent-of-points-from-free-throws', 'shooting-pct', 'fta-per-fga', 'ftm-per-100-possessions', 'free-throw-rate', 
    #             'three-point-rate', 'two-point-rate', 'three-pointers-made-per-game', 'effective-field-goal-pct', 'true-shooting-percentage', 
    #             'opponent-shooting-pct', 'opponent-fta-per-fga', 'opponent-ftm-per-100-possessions', 'opponent-free-throw-rate', 
    #             'opponent-three-point-rate', 'opponent-two-point-rate', 'opponent-three-pointers-made-per-game', 'opponent-effective-field-goal-pct',
    #             'opponent-true-shooting-percentage', 'offensive-rebounds-per-game', 'offensive-rebounding-pct', 'defensive-rebounds-per-game',
    #             'defensive-rebounding-pct', 'opponent-offensive-rebounds-per-game', 'opponent-offensive-rebounding-pct', 'opponent-defensive-rebounds-per-game',
    #             'opponent-defensive-rebounding-pct', 'blocks-per-game', 'steals-per-game', 'block-pct', 'steals-perpossession', 'steal-pct',
    #             'opponent-blocks-per-game', 'opponent-steals-per-game', 'opponent-block-pct', 'opponent-steals-perpossession', 'opponent-steal-pct',
    #             'assists-per-game', 'turnovers-per-game', 'turnovers-per-possession', 'assist--per--turnover-ratio', 'assists-per-fgm', 'assists-per-possession',
    #             'turnover-pct', 'opponent-assists-per-game', 'opponent-turnovers-per-game', 'opponent-turnovers-per-possession', 
    #             'opponent-assist--per--turnover-ratio', 'opponent-assists-per-fgm', 'opponent-assists-per-possession', 'opponent-turnover-pct', 
    #             'personal-fouls-per-game', 'personal-fouls-per-possession', 'personal-foul-pct', 'opponent-personal-fouls-per-game', 
    #             'opponent-personal-fouls-per-possession', 'opponent-personal-foul-pct', 'possessions-per-game', 'extra-chances-per-game', 'effective-possession-ratio',
    #             'opponent-effective-possession-ratio']
    #
    #statslabels = []
    #statslabels.append('teamname')
    #for each in statslist:
    #    statslabels.append('season-' + each)
    #    statslabels.append('threegame-' + each)
    #    statslabels.append('onegame-' + each)
    #    statslabels.append('home-' + each)
    #    statslabels.append('away-' + each)
    #    statslabels.append('lastseason-' + each)
       
    
    teamstats = ['points-per-game', 'offensive-efficiency', 'floor-percentage', 'points-from-2-pointers', 'points-from-3-pointers', 
                 'percent-of-points-from-2-pointers', 'percent-of-points-from-3-pointers', 'percent-of-points-from-free-throws', 
                 'defensive-efficiency', 'shooting-pct', 'fta-per-fga', 'ftm-per-100-possessions', 'free-throw-rate', 
                 'three-point-rate', 'two-point-rate', 'three-pointers-made-per-game', 'effective-field-goal-pct', 'true-shooting-percentage', 
                 'offensive-rebounds-per-game', 'offensive-rebounding-pct', 'defensive-rebounds-per-game',
                 'defensive-rebounding-pct', 'blocks-per-game', 'steals-per-game', 'block-pct', 'steals-perpossession', 'steal-pct',
                 'assists-per-game', 'turnovers-per-game', 'turnovers-per-possession', 'assist--per--turnover-ratio', 'assists-per-fgm', 'assists-per-possession',
                 'turnover-pct', 'personal-fouls-per-game', 'personal-fouls-per-possession', 'personal-foul-pct', 
                 'possessions-per-game', 'extra-chances-per-game', 'effective-possession-ratio']
     
    teamnames = ['Purdue', 'Duke', 'Wichita St', 'Arizona', 'Kansas', 'Xavier', 'Villanova', 'Florida', 
             'N Carolina', 'Michigan St', 'W Virginia', 'Cincinnati', 'Texas A&M', 'Virginia', 'Miami (FL)', 
             'Arkansas', 'Minnesota', 'Notre Dame', 'Gonzaga', 'Baylor', 'Texas Tech', 'Texas', 
             'TX Christian', 'Oregon', 'Seton Hall', 'Oklahoma', 'Alabama', 'USC', 'Kentucky', 'Creighton', 
             'St Marys', 'Maryland', 'Providence', 'Wisconsin', 'Kansas St', 'Louisville', 'Rhode Island', 
             'S Methodist', 'Oklahoma St', 'Florida St', 'Nevada', 'S Carolina', 'Missouri', 'VA Tech',
             'Tennessee', 'Butler', 'Mississippi', 'Utah', 'Temple', 'St Johns', 'Clemson', 'Penn State',
             'Ohio State', 'Davidson', 'Marquette', 'Michigan', 'Syracuse', 'Auburn', 'Vanderbilt', 'Iowa',
             'UCLA', 'San Diego St', 'Arizona St', 'Bucknell', 'Georgetown', 'LSU', 'Boston Col', 'Central FL', 
             'Iowa State', 'Miss State', 'Illinois', 'Vermont', 'LA Lafayette', 'Georgia', 'Northwestern',
             'Boise State', 'St Bonavent', 'Houston', 'Fla Gulf Cst', 'UNLV', 'VCU', 'Col Charlestn', 
             'GA Southern', 'BYU', 'Mercer', 'Belmont', 'TX-Arlington', 'Towson', 'Middle Tenn', 'Connecticut',
             'N Mex State', 'Georgia St', 'Fresno St', 'Valparaiso', 'Old Dominion', 'GA Tech', 'Colorado', 
             'N Iowa', 'Indiana', 'Loyola-Chi', 'Stanford', 'Wyoming', 'NC State', 'Murray St', 'St Josephs',
             'LA Tech', 'N Kentucky', 'Wake Forest', 'Rutgers', 'Illinois St', 'Dayton', 'Yale', 'New Mexico',
             'Jksnville St', 'Oakland', 'Albany', 'South Dakota', 'Missouri St', 'Iona', 'Monmouth', 'Tulsa',
             'Grand Canyon', 'Furman', 'Nebraska', 'U Mass', 'E Tenn St', 'DePaul', 'Princeton', 'NC-Grnsboro',
             'U Penn', 'NC-Asheville', 'Toledo', 'Buffalo', 'Kent State', 'Oregon St', 'California', 'San Fransco', 
             'IPFW', 'Elon', 'La Salle', 'UAB', 'Utah Val St', 'S Dakota St', 'Bradley', 'Harvard', 'Akron',
             'S Illinois', 'Utah State', 'Washington', 'Tulane', 'W Michigan', 'Troy', 'Saint Louis', 'UC Irvine',
             'E Michigan', 'Hofstra', 'Weber State', 'N Dakota St', 'NC-Wilmgton', 'Ste F Austin', 'Army', 
             'Lipscomb', 'Evansville', 'UCSB', 'N Hampshire', 'Memphis', 'Ohio', 'San Diego', 'Liberty', 'Rider',
             'Northeastrn', 'Montana', 'Ball State', 'LA Monroe', 'Coastal Car', 'Drake', 'Columbia', 'CS Bakersfld',
             'TX El Paso', 'Central Mich', 'Richmond', 'Geo Mason', 'Indiana St', 'Colgate', 'Wash State',
             'Idaho', 'Montana St', 'IL-Chicago', 'Pacific', 'Geo Wshgtn', 'Portland St', 'Samford', 'Arkansas St',
             'Campbell', 'UC Davis', 'Winthrop', 'St Fran (PA)', 'WI-Milwkee', 'Air Force', 'Stony Brook', 'Lamar', 
             'Maryland BC', 'Canisius', 'Lehigh', 'Lg Beach St', 'Hawaii', 'E Washingtn', 'Santa Clara', 'Radford', 
             'Navy', 'TN Tech', 'E Kentucky', 'TN Martin', 'Fairfield', 'Pittsburgh', 'Colorado St', 'Marshall', 
             'Niagara', 'N Illinois', 'Delaware', 'W Kentucky', 'WI-Grn Bay', 'App State', 'Denver', 'Wofford', 
             'Wright State', 'St Peters', 'E Carolina', 'Texas State', 'Gard-Webb', 'TX Southern', 'Wm & Mary',
             'IUPUI', 'Seattle', 'TN State', 'F Dickinson', 'Bowling Grn', 'Cleveland St', 'Fordham', 'Charlotte',
             'S Alabama', 'Holy Cross', 'Loyola Mymt', 'Drexel', 'Cal Poly', 'North Dakota', 'Duquesne', 
             'Massachusetts Lowell', 'Manhattan', 'Dartmouth', 'Sam Hous St', 'TX A&M-CC', 'Boston U', 'Neb Omaha', 
             'Loyola-MD', 'TX-San Ant', 'Chattanooga', 'High Point', 'NC Central', 'E Illinois', 'N Colorado',
             'Incarnate Word', 'SC Upstate', 'James Mad', 'Siena', 'American', 'W Illinois', 'S Mississippi',
             'TX-Pan Am', 'Abilene Christian', 'Miami (OH)', 'Portland', 'UC Riverside', 'Austin Peay', 'Brown',
             'LIU-Brooklyn', 'CS Fullerton', 'Binghamton', 'Rice', 'Mt St Marys', 'Cornell', 'UMKC', 'SE Missouri', 
             'Morehead St', 'Detroit', 'New Orleans', 'Rob Morris', 'Fla Atlantic', 'Charl South', 'Hampton', 'NC A&T', 
             'Central Ark', 'Youngs St', 'N Florida', 'Wagner', 'Houston Bap', 'SE Louisiana', 'Cal St Nrdge',
             'Quinnipiac', 'S Florida', 'Central Conn', 'NJIT', 'Oral Roberts', 'Sac State', 'Sacred Hrt', 'San Jose St', 
             'Kennesaw St', 'Bryant', 'North Texas', 'St Fran (NY)', 'AR Lit Rock', 'Nicholls St', 'SIU Edward', 
             'Jackson St', 'Lafayette', 'W Carolina', 'Marist', 'Norfolk St', 'S Utah', 'Pepperdine', 'Jacksonville',
             'Southern', 'Hartford', 'Stetson', 'McNeese St', 'Prairie View', 'Citadel', 'Morgan St', 'Howard', 
             'Savannah St', 'Florida Intl', 'N Arizona', 'Idaho State', 'Beth-Cook', 'VA Military', 'Alcorn State', 
             'Grambling St', 'Maine', 'Ark Pine Bl', 'Florida A&M', 'Longwood', 'S Car State', 'Maryland ES', 'Alab A&M',
             'Coppin State', 'Delaware St', 'Chicago St', 'NW State', 'Presbyterian', 'Alabama St', 'Miss Val St'] 
    
    cursor = cnx.cursor()
    cursor.execute('select max(statdate) from basestats')
    x = cursor.fetchall()
    cursor.execute('Select oddsdate, favorite, underdog from ncaa_bb.oddsdata where oddsdate > "%s-%s-%s" order by oddsdate asc;'% (x[0][0].year, x[0][0].month, x[0][0].day))
    inputs = pd.DataFrame(cursor.fetchall())
    cursor.close()
    if len(inputs) > 0:
        alldates = inputs[0]
        alldates = alldates.drop_duplicates()
        

        cursor = cnx.cursor()
        
        progress = 0
        for usedate in alldates:
            progress += 1
        
            print('%f percent complete' % (float(progress) / float(len(alldates)) * 100))    
            statsdata = pd.DataFrame(columns = teamstats)
            whichteams = None
            whichteams = list(inputs[inputs[0] == usedate][1])
            whichteams = whichteams + list(inputs[inputs[0] == usedate][2])
            statsdata['teamname'] = whichteams
            urldate = str((usedate + datetime.timedelta(days=1)).year) + '-' + str((usedate + datetime.timedelta(days=1)).month) + '-' + str((usedate + datetime.timedelta(days=1)).day)
            for each in teamstats:
                tree = None
                url = None
                names = None
                ratings = None
                rankingdict = None
                ranking = None
                url = 'https://www.teamrankings.com/ncaa-basketball/stat/%s?date=%s' % (each, urldate)
                pageContent=requests.get(url)
                tree = html.fromstring(pageContent.content)
                lastgame = []
                for team in whichteams:
                    value = None
                    formattedvalue = None
                    try:
                        value = tree.xpath('//tbody/tr[td[2]/a/text()="%s"]/td[@class="text-right"][3]/text()' % (team))[0]
                    except IndexError:
                        value = '--'
                    try:
                        formattedvalue = float(value)
                    except ValueError:
                        if value[-1] == '%':
                            formattedvalue = float(value[:-1])
                        elif value  == '--':
                            formattedvalue = 'NULL'
                    lastgame.append(formattedvalue)
                statsdata[each] = lastgame
            
            for team in whichteams:
                sqldata = []
                for label in teamstats:
                    sqldata.append(str(statsdata['%s' % label][statsdata.teamname == team].values[0]))
                sqldata = ', '.join(sqldata)
                sqldata = 'INSERT INTO ncaa_bb.basestats VALUES ("%s", "%s", ' % (team, usedate) + sqldata + ');'
                cursor.execute(sqldata)
                cnx.commit()
        #        cursor.execute('SET foreign_key_checks = 0;')
        cursor.close()
        cnx.close()
    print('Updated NCAA Basketball Stats to %s' % (str(x[0][0])))
