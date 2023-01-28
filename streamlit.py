%%writefile "C:\Users\krakl\Jupyter\machine_learning\nba_api\streamlit.py"

import pandas as pd
import streamlit as st
from time import time
from datetime import datetime, timedelta
import shutil
import os
from os.path import exists
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

image_0 = Image.open(r"C:\Users\krakl\Jupyter\machine_learning\nba_api\Logo-NBA.png")
image_1 = Image.open(r"C:\Users\krakl\Jupyter\machine_learning\output.png")

timestamp = datetime.today().strftime("%m.%d.%y")
today_preds_path = r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\hist_predictions\today_preds_" + timestamp + ".txt" 
ir_path = r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\IR\injury_report_" + timestamp + ".txt"

c=0
while True:
    if os.path.exists(ir_path):
        break
    else:
        temp_date = datetime.strptime(timestamp, "%m.%d.%y") - timedelta(days = c)
        temp_date = temp_date.strftime("%m.%d.%y")
        ir_path = r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\IR\injury_report_" + temp_date + ".txt"
        if os.path.exists(ir_path):
            break
        else:
            c+=1
            continue

source = r"C:\Users\krakl\Downloads\fanduel.csv"
csv_archive = r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\FanDuel\csv_archive\fanduel_" + timestamp + ".csv" 
archive =  r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\FanDuel\archive\FanDuel_ledger_" + timestamp + '.txt'
dest = r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\FanDuel\FanDuel_ledger.txt"

c=0
while True:
    if exists(source):
        shutil.copy(source, csv_archive)
        shutil.move(source,archive)
        break
    else:
        temp_date = datetime.strptime(timestamp,"%m.%d.%y") - timedelta(days=c)
        temp_date = temp_date.strftime("%m.%d.%y")
        archive = r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\FanDuel\archive\FanDuel_ledger_" + temp_date + '.txt'
        if exists(archive) == True:
            break
        else:
            c+=1
            continue
        continue

fd = pd.read_csv(archive)

nba = fd[fd['League']=='NBA']


nba.columns = [i.replace(' ','_') for i in nba.columns]
nba.columns = [i.lower() for i in nba.columns]
nba['bet_date_time'] = pd.to_datetime(nba.bet_date_time).dt.strftime('%m.%d.%y')
nba = nba.iloc[pd.to_datetime(nba.bet_date_time).values.argsort()]
nba = nba[nba['bet_result']!='CASHED_OUT']
nba['bet_type_desc'] = np.where(nba['bet_type']=='SGL', 
                         'single','parlay')
nba = nba[nba['bet_type_desc']=='single']
nba['bet_result_code'] = np.where(nba['bet_result']=='WON',True,False).astype(int) 

# --- ST CONFIGURATION ---
st.set_page_config(
    page_title = 'NBA Betting Dashboard',
    layout = 'wide'

)

# ---FILTER CONFIGURATION ---
page = st.sidebar.selectbox('Choose your page:',
                 ['Home','Code','Tonights Bets','Bets Catalogue'])

if page == 'Home':
    st.header("Welcome to Dalton's NBA Betting Dashboard!")
    col1,col2 = st.columns(2, gap = 'small')
    with col1:
        st.write("""
        Within this dashboard I display the results of my attempt at leveraging machine learning to predict NBA outcomes and my bets associated with those outcomes.
        The algorithm uses a Random Forest Regressor to predict scores for each night's slate of NBA games to be played. With these predicted scores I place bets on 
        three different markets: \n
        - Point Totals\n
        - Spreads\n
        - Moneylines\n
        While results vary, it's been very fun becoming familiar with the concepts of Machine Learning (and making a few dollars here and there!).\n
        Please use the navigation dropdown on the left side of the screen to navigate the dashboard. For any questions or inquiries, I can be reached at data.dalton.kraklan@gmail.com
        or at my portfolio website daltonkraklan.com.
                 """,
    )
    with col2:
        st.image(image_0)

elif page == 'Code':
    st.header('Code and Code Output')
    st.subheader('\nOutput:')
    st.image(image_1)
    
    code = r"""
# Import Modules and define key variables
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2, boxscoretraditionalv2
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from os.path import exists
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from selenium import webdriver
path = 'C:\\Program Files (x86)\\chromedriver.exe'
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support import expected_conditions
from sklearn.preprocessing import MinMaxScaler
import os
import shutil
from time import time
import geopy.distance
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import sqlalchemy
engine = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3306/nba_api')


today = datetime.today().strftime("%Y-%m-%d")
timestamp = datetime.today().strftime("%m.%d.%y")
today = datetime.today().strftime("%Y-%m-%d")
year = datetime.today().year
yesterday = datetime.today() - timedelta(days=1)
yesterday = yesterday.strftime("%m.%d.%y")
AN_yesterday = datetime.today() - timedelta(days=1)
AN_yesterday = AN_yesterday.strftime("%Y-%m-%d")
yesterday_df = datetime.today() - timedelta(days=1)
yesterday_df = yesterday_df.strftime("%Y-%m-%d")
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint64']
stamp = str(datetime.today())
stamp = stamp.replace(":",'.')

cities = pd.read_excel(r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\cities.xlsx")
cities['coords'] = list(zip(cities.lat, cities.lng))
cities = cities.set_index('team')

#Define key formulas
def distancer(row):
    coords_1 = row['PRIOR_GAME_COORDS']
    coords_2 = row['GAME_COORDS']
    return geopy.distance.geodesic(coords_1, coords_2).miles

def pd_to_sql(df,table):
    df.columns = df.columns.str.strip()
    df.to_sql(
        name = table,
        con = engine,
        index = False,
        if_exists = 'replace'
    )

def pd_read_sql(table):
    df = pd.read_sql(table , engine)
    return df

# Starting Script
print('Starting Script')
main_start = time()

print('\n\tScraping Action Network:')
start = time()
last_date = AN_yesterday

link = 'https://www.actionnetwork.com/nba/odds'
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul']
team_names = pd.read_excel(r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\nba_teams.xlsx",'Sheet2')
driver = webdriver.Chrome(path)
driver.get(link)
sleep(1)

date = driver.find_element(By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/span').text
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul']
if any(i in date for i in months):
    date_2 = date + f", {year}"
    x = datetime.strptime(date_2,'%a %b %d, %Y')
    date = x.strftime("%Y-%m-%d")
else:
    date_2 = date + f", {year-1}"
    x = datetime.strptime(date_2,'%a %b %d, %Y')
    date = x.strftime("%Y-%m-%d")

cols_0 = ['AWAY','AWAY_ML','AWAY_SPREAD','TOTAL','HOME','HOME_ML','HOME_SPREAD']
action_network = pd.DataFrame(columns = cols_0)

while date > last_date:
    WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select'))).click()
    WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select/option[1]'))).click()
    sleep(1)

    #Scrape teams by home/away
    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.CLASS_NAME,'best-odds__game-info')))
    game_box = driver.find_elements(By.CLASS_NAME,'best-odds__game-info')
    teams = []
    for i in game_box:
        team = i.find_elements(By.XPATH,'//div[@class="game-info__team--desktop"]/span')
        teams = [i.text for i in team]

    #Scrape Spread
    spread = driver.find_elements(By.XPATH,'//td[@class="best-odds__last-type-cell"][1]/div/div/div/span[1]')
    spread = [i.text for i in spread]

    #Scrape Moneyline
    WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select'))).click()
    WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select/option[3]'))).click()
    sleep(1)

    ML = driver.find_elements(By.XPATH,'//td[@class="best-odds__last-type-cell"][1]/div/div/div/span[1]')
    ML = [i.text for i in ML]

    #Scrape Total
    sleep(1)
    WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select'))).click()
    WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select/option[2]'))).click()
    sleep(1)

    total = driver.find_elements(By.XPATH,'//td[@class="best-odds__last-type-cell"][1]/div/div/div/span[1]')
    total = [i.text for i in total]

    driver.find_element(By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select').click()
    driver.find_element(By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[1]/div/div[2]/select/option[2]').click()
    sleep(1)

    date = driver.find_element(By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/span').text
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul']
    if any(i in date for i in months):
        date_2 = date + f", {year}"
        x = datetime.strptime(date_2,'%a %b %d, %Y')
        date = x.strftime("%Y-%m-%d")
    else:
        date_2 = date + f", {year-1}"
        x = datetime.strptime(date_2,'%a %b %d, %Y')
        date = x.strftime("%Y-%m-%d")

    cols = ['AWAY','AWAY_ML','AWAY_SPREAD','TOTAL','HOME','HOME_ML','HOME_SPREAD']

    key = pd.DataFrame(list(zip(teams, ML, spread, total)), columns = ['TEAMS','ML','SPREAD','TOTAL'])
    key['TEAMS'] = key.TEAMS.map(team_names.set_index('Opponent').Team)

    df = pd.DataFrame(list(zip(teams, ML, spread, total)), columns = ['TEAMS','ML','SPREAD','TOTAL'])
    df['HOME'] = df['TEAMS'].shift(-1)
    df['HOME_ML'] = df['ML'].shift(-1)
    df['HOME_SPREAD'] = df['SPREAD'].shift(-1)
    df.columns = cols
    df = df.iloc[::2, :]
    df['AWAY'] = df.AWAY.map(team_names.set_index('Opponent').Team)
    df['HOME'] = df.HOME.map(team_names.set_index('Opponent').Team)
    df['GAME_DATE'] = date
    df['MATCHUP'] = df['AWAY'].astype('str') + " v " + df['HOME'].astype('str')
    df = df.replace(['PK'],0,regex=True)
    df['TOTAL'] = df['TOTAL'].replace(['o','u'],'',regex=True)

    action_network = pd.concat([action_network,df], axis=0)

    driver.find_element(By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/button[1]').click()
    sleep(1)

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul']
    date = driver.find_element(By.XPATH,'//*[@id="__next"]/div/main/div[2]/div[2]/div[2]/div[1]/span').text
    if any(i in date for i in months):
        date_2 = date + f", {year}"
        x = datetime.strptime(date_2,'%a %b %d, %Y')
        date = x.strftime("%Y-%m-%d")
    else:
        date_2 = date + f", {year-1}"
        x = datetime.strptime(date_2,'%a %b %d, %Y')
        date = x.strftime("%Y-%m-%d")

driver.quit()
action_network = action_network.reset_index(drop=True)

hist_odds = action_network
hist_odds = hist_odds.sort_values(by=['GAME_DATE']).reset_index(drop=True)

pd_to_sql(hist_odds,'hist_odds')

end_time = time() - start
print("\n\t\tExecution Time: %0.3fs" % end_time)

# Refresh Game Log
print('\n\tRefreshing Data:')
start = time()
print('\nRefreshing Game Log:')
seasons = [2022]
game_log = pd.DataFrame()

while True:
    try:
        for i in tqdm(seasons):
            temp = leaguegamelog.LeagueGameLog(season = i).get_data_frames()[0]
            game_log = pd.concat([game_log,temp])
    except:
        continue
    break
    
# Save Game Log
game_log_games = list(game_log['GAME_ID'].unique())
game_log_games.sort()

print('\tExporting to SQL')
pd_to_sql(game_log,'game_log')

# Refresh Adv_Stats
print('''
Refreshing Advanced Stats:\n
\tImporting from SQL
''')
adv_stats_archive = pd_read_sql('adv_stats_archive')
adv_stats_games = list(adv_stats_archive['GAME_ID'].unique())
adv_stats_games = [str(i).zfill(10) for i in adv_stats_games]
adv_stats_games.sort()
adv_stats_missing = [i for i in game_log_games if i not in adv_stats_games]
halt = len(adv_stats_missing)
counter = []
c = 0

while True:
    try:
        while len(counter)<=halt:
            for i in tqdm(adv_stats_missing):
                if i not in adv_stats_games:
                    temp = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=i,timeout = 100).get_data_frames()[0]
                    adv_stats_archive = pd.concat([adv_stats_archive,temp], axis=0)
                    adv_stats_archive = adv_stats_archive.reset_index(drop = True)
                    counter.append(str(i).zfill(10))
                    c+=1
                    if c%25 == 0:
                        pd_to_sql(adv_stats_archive,'adv_stats_archive')
                    else:
                        continue
                else:
                    continue
            break
    except:
        continue
    break
    
# Save Advanced Stats
print('\tExporting to SQL')
pd_to_sql(adv_stats_archive,'adv_stats_archive')
end_time = time() - start

# Refresh Box Scores
print('''
Refreshing Box Scores:\n
\tImporting from SQL
''')
box_score_archive = pd_read_sql('box_score_archive')
box_score_games = list(box_score_archive['GAME_ID'].unique())
box_score_games = [str(i).zfill(10) for i in box_score_games]
box_score_games.sort()
box_score_missing = [i for i in game_log_games if i not in box_score_games]
halt = len(box_score_missing)
counter = []
c = 0

while True:
    try:
        while len(counter)<=halt:
            for i in tqdm(box_score_missing):
                if i not in box_score_games:
                    temp = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=i,timeout = 100).get_data_frames()[0]
                    box_score_archive = pd.concat([box_score_archive,temp], axis=0)
                    box_score_archive = box_score_archive.reset_index(drop = True)
                    counter.append(str(i).zfill(10))
                    c+=1
                    if c%25 == 0:
                        pd_to_sql(box_score_archive,'box_score_archive')
                    else:
                        continue
                else:
                    continue
            break
    except:
        continue
    break
    
# Save Box Scores
print('\tExporting to SQL')
pd_to_sql(box_score_archive,'box_score_archive')

end_time = time() - start
print("\n\t\tExecution Time: %0.3fs" % end_time)

# Manipulating Game log, advanced stats, and box scores
print('\n\tData Refreshed, Manipulating Data Frames:')
start = time()

game_log = game_log.sort_values(by=['GAME_DATE','GAME_ID'])
game_log = game_log.reset_index(drop=True)
game_log['TOTAL'] = game_log['PTS']*2 - game_log['PLUS_MINUS']
game_log['PTS_ALLOWED'] = game_log['TOTAL'] - game_log['PTS']
game_log['WL_CODE'] = game_log['WL'].astype('category').cat.codes
game_log['HA_CODE'] = np.where(game_log['MATCHUP'].str.contains('vs.'),True,False)
game_log['HA_CODE'] = game_log['HA_CODE'].astype(int,errors='ignore')
game_log['OPP'] = game_log['MATCHUP'].str[-3:]
game_log = game_log.drop(columns = ['VIDEO_AVAILABLE','TEAM_NAME','MATCHUP','WL'])
game_log = game_log.apply(pd.to_numeric, errors='ignore')
game_log['GAME_ID'] = game_log['GAME_ID'].astype(str).str.zfill(10)
game_log['KEY'] = game_log['TEAM_ID'].astype('str') + game_log['GAME_ID'].astype('str')

game_log['GAME_COORDS'] = np.where(game_log['HA_CODE'] == 1,game_log.TEAM_ABBREVIATION.map(cities['coords']),game_log.OPP.map(cities['coords']))
game_log['PRIOR_GAME_COORDS'] = game_log.groupby('TEAM_ID')['GAME_COORDS'].shift(1).fillna(game_log['GAME_COORDS'])
game_log['DIST_TRAVELED'] = game_log.apply(distancer, axis=1)
game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
game_log['DAYS_BETWEEN'] = game_log.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff() / np.timedelta64(1,'D')
game_log['DAYS_BETWEEN'] = game_log['DAYS_BETWEEN'].fillna(0)
game_log['GAME_DATE'] = game_log['GAME_DATE'].astype('str')

box_adv = adv_stats_archive
box_adv = box_adv.apply(pd.to_numeric, errors='ignore')
box_adv = box_adv.select_dtypes(include=numerics)
box_adv['GAME_ID'] = box_adv['GAME_ID'].astype(str).str.zfill(10)
box_adv = box_adv.drop(columns=['PLAYER_ID'])
box_adv['KEY'] = box_adv['TEAM_ID'].astype('str') + box_adv['GAME_ID'].astype('str')
box_adv = box_adv.drop(columns=['TEAM_ID','GAME_ID'])
box_adv_mrg = box_adv.groupby('KEY', as_index=False).max()
adv_cols = [f"P_ADV_{col}" for col in box_adv.columns if col != 'KEY']
column_indices = range(1,len(box_adv_mrg.columns))
old_names = box_adv_mrg.columns[column_indices]
box_adv_mrg.rename(columns=dict(zip(old_names, adv_cols)), inplace=True)

box_score = box_score_archive
box_score = box_score.apply(pd.to_numeric, errors='ignore')
box_score = box_score.select_dtypes(include=numerics)
box_score['GAME_ID'] = box_score['GAME_ID'].astype(str).str.zfill(10)
box_score = box_score.drop(columns=['PLAYER_ID'])
box_score['KEY'] = box_score['TEAM_ID'].astype('str') + box_score['GAME_ID'].astype('str')
box_score = box_score.drop(columns=['TEAM_ID','GAME_ID'])
box_score_mrg = box_score.groupby('KEY', as_index=False).max()
box_cols = [f"P_BOX_{col}" for col in box_score_mrg.columns]
column_indices = range(1,len(box_score_mrg.columns))
old_names = box_score_mrg.columns[column_indices]
box_score_mrg.rename(columns=dict(zip(old_names, box_cols)), inplace=True)

df = pd.merge(game_log,box_adv_mrg,on='KEY')
df = pd.merge(df,box_score_mrg,on='KEY')

removed_columns = list(df.columns[df.dtypes == 'object'])+['TARGET','SEASON_ID','TEAM_ID']
selected_columns = df.columns[~df.columns.isin(removed_columns)]

end_time = time() - start
print("\t\tExecution Time: %0.3fs" % end_time)

# Training ML Model
print('\n\tTraining Model:')

# Defining rolling averages and other key variables for ML
rolls = [25,20,15,10,5,3,1]
preds_cols = [f'PREDS_{i}' for i in rolls]
acc_scores = []

for i in rolls:
    print(f"\t\t{i}")
    start = time()
    roll = i
    def find_team_averages(team):
        rolling = team.rolling(roll).mean()
        return rolling

    df_rolling = df[list(selected_columns) + ['TEAM_ABBREVIATION','SEASON_ID']]
    df_rolling = df_rolling.groupby(['TEAM_ABBREVIATION','SEASON_ID'],group_keys=False).apply(find_team_averages)
    rolling_cols = [f"{col}_rlg" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols

    df_rl = pd.concat([df,df_rolling],axis=1)


    def shift_col(team, col_name):
        next_col = team[col_name].shift(-1)
        return next_col

    def add_col(df,col_name):
        return df.groupby('TEAM_ABBREVIATION', group_keys=False).apply(lambda x: shift_col (x,col_name))

    df_rl['HOME_NEXT'] = add_col(df_rl,'HA_CODE')
    df_rl['TEAM_OPP_NEXT'] = add_col(df_rl,'OPP')
    df_rl['DATE_NEXT'] = add_col(df_rl,'GAME_DATE')

    null = df_rl[df_rl['DATE_NEXT'].isna()]
    df_rl = df_rl.dropna()

    key = pd.read_excel(r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\nba_22-23_schedule.xlsx")
    key['GAME_DATE'] = key['GAME_DATE'].dt.strftime('%Y-%m-%d')
    key['key'] = key['GAME_DATE'].astype('str') + key['TEAM_ABBREVIATION'].astype('str')
    key['HOME_NEXT'] = add_col(key,'H/A_CODE')
    key['TEAM_OPP_NEXT'] = add_col(key,'OPP')
    key['DATE_NEXT'] = add_col(key,'GAME_DATE')

    null['key'] = null['GAME_DATE'].astype('str') + null['TEAM_ABBREVIATION'].astype('str')
    null['HOME_NEXT'] = null.key.map(key.drop_duplicates('key').set_index('key').HOME_NEXT)
    null['TEAM_OPP_NEXT'] = null.key.map(key.drop_duplicates('key').set_index('key').TEAM_OPP_NEXT)
    null['DATE_NEXT'] = null.key.map(key.drop_duplicates('key').set_index('key').DATE_NEXT)
    null = null.drop(columns = ['key'])

    temp = pd.concat([df_rl,null], axis=0)
    temp

    full = temp.merge(
        temp[rolling_cols + ['TEAM_OPP_NEXT', 'DATE_NEXT','TEAM_ABBREVIATION']],
        left_on = ['TEAM_ABBREVIATION','DATE_NEXT'],
        right_on = ['TEAM_OPP_NEXT','DATE_NEXT'])

    full = full.drop(columns=['KEY'])


    encoder = OneHotEncoder(sparse = False)

    team_cols = ['TEAM_ABBREVIATION_x','TEAM_OPP_NEXT_x','TEAM_OPP_NEXT_y','TEAM_ABBREVIATION_y']
    for col in team_cols:
        onehot = encoder.fit_transform(full[[col]])
        one = pd.DataFrame(onehot,columns = encoder.get_feature_names_out())
        full = pd.concat([full,one], axis = 1)

    full = full.sort_values(by=['GAME_ID'])


    def add_target(team):
        team['TARGET'] = team['PTS'].shift(-1)
        return team

    full = full.groupby('TEAM_ID', group_keys=False).apply(add_target)
    full = full[full['DATE_NEXT']<today]

    test = full[~full.isna().any(axis=1)]

    text_columns = list(test.select_dtypes(object).columns) + ['SEASON_ID','TEAM_ID','GAME_ID','OU_RESULT','TARGET'] + [i for i in test.columns if 'CODE' in i]
    num_columns = [i for i in test.columns if i not in text_columns]
    # num_columns = [i for i in num_columns if i in predictors]

    scaler = MinMaxScaler()

    rfr = RandomForestRegressor(random_state=42)

    test[num_columns] = scaler.fit_transform(test[num_columns])

    rfr.fit(test[num_columns], test['TARGET'])
    preds = pd.DataFrame(rfr.predict(test[num_columns]), columns=['PREDS']).reset_index(drop=True)

    full = full.reset_index(drop=True).join(preds.reset_index(drop=True))

    full['PREDS'] = full.groupby('TEAM_ID',group_keys=False)['PREDS'].shift(1)

    acc_df = full[~full.isna().any(axis=1)]
    mae = mean_absolute_error(acc_df['PTS'],acc_df['PREDS'])
    acc_scores.append(mae)
    print(f"\t\t\tMAE: {mae}")

    end_time = time() - start
    print("\t\t\tExecution Time: %0.3fs" % end_time)

mean_acc = sum(acc_scores) / len(acc_scores)
acc_weight = []
for i in acc_scores:
    acc_weight.append(1 + (mean_acc - i)/i)

cols = ['MATCHUP','GAME_DATE','FAV','FAV_ML','FAV_SPREAD','DOG','DOG_ML','DOG_SPREAD','TOTAL','WINNER','WINNER_PAYOUT','WAGER']  
rolls_df = pd.DataFrame(columns = ['MATCHUP','GAME_DATE','FAV','FAV_ML','FAV_SPREAD','DOG','DOG_ML','DOG_SPREAD','TOTAL','WINNER','WAGER'])
rolls = [25,20,15,10,5,3,1]

# Making predictions on tonights games now that model is trained
temp_preds = pd.DataFrame()

print("\n\tMaking Predictions")
for i in range(len(rolls)):
    print(f'\t\tGame Rlg Average {rolls[i]}:')
    start = time()
    roll = rolls[i]
    def find_team_averages(team):
        rolling = team.rolling(roll).mean()
        return rolling

    df_rolling = df[list(selected_columns) + ['TEAM_ABBREVIATION','SEASON_ID']]
    df_rolling = df_rolling.groupby(['TEAM_ABBREVIATION','SEASON_ID'],group_keys=False).apply(find_team_averages)
    rolling_cols = [f"{col}_rlg" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols

    df_rl = pd.concat([df,df_rolling],axis=1)

    def shift_col(team, col_name):
        next_col = team[col_name].shift(-1)
        return next_col

    def add_col(df,col_name):
        return df.groupby('TEAM_ABBREVIATION', group_keys=False).apply(lambda x: shift_col (x,col_name))

    df_rl['HOME_NEXT'] = add_col(df_rl,'HA_CODE')
    df_rl['TEAM_OPP_NEXT'] = add_col(df_rl,'OPP')
    df_rl['DATE_NEXT'] = add_col(df_rl,'GAME_DATE')

    null = df_rl[df_rl['DATE_NEXT'].isna()]
    df_rl = df_rl.dropna()

    key = pd.read_excel(r"C:\Users\krakl\Jupyter\machine_learning\nba_api\data\nba_22-23_schedule.xlsx")
    key['GAME_DATE'] = key['GAME_DATE'].dt.strftime('%Y-%m-%d')
    key['key'] = key['GAME_DATE'].astype('str') + key['TEAM_ABBREVIATION'].astype('str')
    key['HOME_NEXT'] = add_col(key,'H/A_CODE')
    key['TEAM_OPP_NEXT'] = add_col(key,'OPP')
    key['DATE_NEXT'] = add_col(key,'GAME_DATE')

    null['key'] = null['GAME_DATE'].astype('str') + null['TEAM_ABBREVIATION'].astype('str')
    null['HOME_NEXT'] = null.key.map(key.drop_duplicates('key').set_index('key').HOME_NEXT)
    null['TEAM_OPP_NEXT'] = null.key.map(key.drop_duplicates('key').set_index('key').TEAM_OPP_NEXT)
    null['DATE_NEXT'] = null.key.map(key.drop_duplicates('key').set_index('key').DATE_NEXT)
    null = null.drop(columns = ['key'])

    temp = pd.concat([df_rl,null], axis=0)
    temp = temp.sort_values(by=['GAME_ID'])


    full = temp.merge(
        temp[rolling_cols + ['TEAM_OPP_NEXT', 'DATE_NEXT','TEAM_ABBREVIATION']],
        left_on = ['TEAM_ABBREVIATION','DATE_NEXT'],
        right_on = ['TEAM_OPP_NEXT','DATE_NEXT'])

    full = full.drop(columns=['KEY'])

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse = False)

    team_cols = ['TEAM_ABBREVIATION_x','TEAM_OPP_NEXT_x','TEAM_OPP_NEXT_y','TEAM_ABBREVIATION_y']
    for col in team_cols:
        onehot = encoder.fit_transform(full[[col]])
        one = pd.DataFrame(onehot,columns = encoder.get_feature_names_out())
        full = pd.concat([full,one], axis = 1)

    full = full.sort_values(by=['GAME_ID'])

    def add_target(team):
        team['TARGET'] = team['PTS'].shift(-1)
        return team

    full = full.groupby('TEAM_ID', group_keys=False).apply(add_target)

    tonight_teams = list(full[full['DATE_NEXT']==today]['TEAM_ABBREVIATION_x'])

    tonight_games = full[full['TEAM_ABBREVIATION_x'].isin(tonight_teams)]
    tonight_games = tonight_games.groupby('TEAM_ABBREVIATION_x').tail(1)
    tonight_games = tonight_games.sort_values(by=['TEAM_ABBREVIATION_x'])

    tonight_test = full[~full.isna().any(axis=1)]
    tonight_test = tonight_test[tonight_test['TEAM_ABBREVIATION_x'].isin(tonight_teams)]
    tonight_test = tonight_test.groupby('TEAM_ABBREVIATION_x').tail(1)
    tonight_test = tonight_test.sort_values(by=['TEAM_ABBREVIATION_x'])

    text_columns = list(tonight_test.select_dtypes(object).columns) + ['SEASON_ID','TEAM_ID','GAME_ID','OU_RESULT','TARGET'] + [i for i in tonight_test.columns if 'CODE' in i]
    num_columns = [i for i in tonight_test.columns if i not in text_columns]

    tonight_test[num_columns] = scaler.fit_transform(tonight_test[num_columns])

    preds = pd.DataFrame(rfr.predict(tonight_test[num_columns]), columns=[f'PREDS_{rolls[i]}']).reset_index(drop=True)

    temp_preds = pd.concat([temp_preds,preds],axis=1)

# Manipulating prediction output and formatting for simplicity
def rounder(x):
    return round(x * 2)/2

temp_0 = temp_preds
temp_0 = temp_0.join(tonight_games[['TEAM_ABBREVIATION_x','DATE_NEXT','HOME_NEXT','TEAM_ABBREVIATION_y']].reset_index(drop=True))
temp_0['MATCHUP'] = np.where(temp_0['HOME_NEXT']==1,
                            temp_0['TEAM_ABBREVIATION_y'].astype('str') + " v " + temp_0['TEAM_ABBREVIATION_x'].astype('str'),
                            temp_0['TEAM_ABBREVIATION_x'].astype('str') + " v " + temp_0['TEAM_ABBREVIATION_y'].astype('str'))

for i in range(len(preds_cols)):
    temp_0[preds_cols[i]] = temp_0[preds_cols[i]] * acc_weight[i]

temp_0['PTS'] = (temp_0[preds_cols].sum(axis=1)) / len(preds_cols)
temp_0['PTS'] = temp_0['PTS'].apply(lambda x: rounder(x))



temp_0 = temp_0.merge(temp_0[['TEAM_ABBREVIATION_x','TEAM_ABBREVIATION_y','PTS']],left_on = ['TEAM_ABBREVIATION_x'], right_on = ['TEAM_ABBREVIATION_y'])
temp_0 = temp_0.drop(columns = ['TEAM_ABBREVIATION_x_y','TEAM_ABBREVIATION_y_y'])
temp_0 = temp_0.rename(columns = {'TEAM_ABBREVIATION_x_x':'TEAM_x','TEAM_ABBREVIATION_y_x':'TEAM_y','DATE_NEXT':'GAME_DATE'})

temp_0['WINNER'] = np.where(temp_0['PTS_x']==temp_0['PTS_y'],
                       'PK',
                       np.where(temp_0['PTS_x']>temp_0['PTS_y'],
                                temp_0['TEAM_x'],
                                temp_0['TEAM_y']))

temp_0 = temp_0.drop(columns = preds_cols + ['TEAM_x','TEAM_y','HOME_NEXT',])

temp_0 = temp_0.drop_duplicates('MATCHUP')
temp_0['PTS_H'] = np.where(temp_0['WINNER'].astype('str')==temp_0['MATCHUP'].str[-3:],
                          temp_0[['PTS_x','PTS_y']].max(axis=1),
                          temp_0[['PTS_x','PTS_y']].min(axis=1))

temp_0['PTS_A'] = np.where(temp_0['WINNER'].astype('str')==temp_0['MATCHUP'].str[:3],
                          temp_0[['PTS_x','PTS_y']].max(axis=1),
                          temp_0[['PTS_x','PTS_y']].min(axis=1))

temp_0['MY_A_SPREAD'] = np.where(temp_0['PTS_A'] - temp_0['PTS_H']<0,
                         '+' + abs(temp_0['PTS_A'] - temp_0['PTS_H']).astype('str'),
                         '-' + abs(temp_0['PTS_A'] - temp_0['PTS_H']).astype('str'))

temp_0['MY_H_SPREAD'] = np.where(temp_0['PTS_H'] - temp_0['PTS_A']<0,
                         '+' + abs(temp_0['PTS_A'] - temp_0['PTS_H']).astype('str'),
                         '-' + abs(temp_0['PTS_A'] - temp_0['PTS_H']).astype('str'))

temp_0 =temp_0.drop(columns = ['PTS_x','PTS_y'])
temp_0 = temp_0.merge(hist_odds)

temp_0 = temp_0.merge(hist_odds)
temp_0['MY_TOTAL'] = temp_0['PTS_A'] + temp_0['PTS_H']

temp_0['TOTAL_BET'] = np.where(abs(temp_0['MY_TOTAL'].astype(float) - temp_0['TOTAL'].astype(float))<mean_acc,
                         '-',
                         np.where((abs(temp_0['MY_TOTAL'].astype(float) - temp_0['TOTAL'].astype(float))>mean_acc) & (temp_0['MY_TOTAL'].astype(float) < temp_0['TOTAL'].astype(float)),
                                  'UNDER',
                                  'OVER'))
temp_0['A_SPREAD_BET'] = np.where(abs(temp_0['MY_A_SPREAD'].astype(float) - temp_0['AWAY_SPREAD'].astype(float)) < mean_acc,
                         '-',
                         np.where(temp_0['MY_A_SPREAD'].astype(float)<temp_0['AWAY_SPREAD'].astype(float),
                                 'BET',
                                 '-'))
temp_0['H_SPREAD_BET'] = np.where(abs(temp_0['MY_H_SPREAD'].astype(float) - temp_0['HOME_SPREAD'].astype(float)) < mean_acc,
                         '-',
                         np.where(temp_0['MY_H_SPREAD'].astype(float)<temp_0['HOME_SPREAD'].astype(float),
                                 'BET',
                                 '-'))

highlighter = np.where(abs(temp_0['MY_TOTAL'].astype(float) - temp_0['TOTAL'].astype(float))>mean_acc,
        'background-color: green',
        '')

l = ['OVER','UNDER','BET']
idx = temp_0[temp_0[['TOTAL_BET','A_SPREAD_BET','H_SPREAD_BET']].isin(l)].index

temp_0 = temp_0[temp_0.index.isin(idx)]

cols = ['GAME_DATE','MATCHUP','WINNER','MY_A_SPREAD','AWAY_SPREAD','A_SPREAD_BET','MY_H_SPREAD','HOME_SPREAD','H_SPREAD_BET','PTS_H','PTS_A','MY_TOTAL','TOTAL','TOTAL_BET']
today_preds = temp_0[cols]

pd_to_sql(today_preds,'today_preds')

formatted_preds = today_preds.style.apply(lambda x: ['background: green' if v in l else '' for v in x], axis=1).format(precision=1)

main_end_time = time() - main_start
print("\n\t\tExecution Time: %0.3fs" % main_end_time)

display(formatted_preds)
    """
    st.subheader('\nCode:')
    st.code(code, language = 'python')

elif page == 'Tonights Bets':
    st.subheader("Tonights Predictions:")
    try:
        today_preds = pd.read_csv(today_preds_path)
        l = ['OVER','UNDER','BET']
        formatted_preds = today_preds
        st.table(formatted_preds.style.apply(lambda x: ['background: green' if v in l else '' for v in x], axis=1).format(precision=1))

        today_teams_temp = list(today_preds['MATCHUP'])
        away = [i[:3] for i in today_teams_temp]
        home = [i[-3:] for i in today_teams_temp]
        today_teams = [away + home]
        today_teams = [x for l in today_teams for x in l]


        IR = pd.read_csv(ir_path)
        IR = IR[IR['TEAM_ABBREVIATION'].isin(today_teams)]

        st.subheader("Injury Report:")
        TEAM = st.selectbox(
            "Choose Team:",
            list(IR['TEAM_ABBREVIATION'].unique())
        )

        IR_selection = IR.query(
            'TEAM_ABBREVIATION == @TEAM'
        )

        st.dataframe(IR_selection)

    except:
        print('No Current Selections')
        
elif page == 'Bets Catalogue':

    st.sidebar.header('Filters:')

    MARKET_NAME = st.sidebar.selectbox(
        "Select Market Type:",
        ['All'] + list(nba['market_name'].unique()))

    if "All" in MARKET_NAME:
        MARKET_NAME = list(nba['market_name'].unique())

    nba_selection = nba.query(
        "market_name == @MARKET_NAME"
    )

    nba_selection['bet_result_code'] = np.where(nba_selection['bet_result']=='WON',True,False).astype(int) 
    nba_selection['divider'] = 1
    nba_selection['win_rate'] = nba_selection['bet_result_code'] / nba_selection['divider']
    nba_selection['running_win_rate'] = nba_selection['win_rate'].cumsum(axis=0)
    nba_selection['running_divider'] = nba_selection['divider'].cumsum(axis=0)
    nba_selection['running_win_rate'] = nba_selection['running_win_rate'] / nba_selection['running_divider']
    nba_selection['avg_running_win_rate'] = nba_selection['running_win_rate'].transform(lambda x: x.expanding().mean())
    nba_selection['daily_running_win_rate'] = nba_selection.groupby(['bet_date_time','market_name'])['running_win_rate'].tail(1)





    table = nba_selection[~nba_selection['daily_running_win_rate'].isna()].reset_index(drop=True)

    st.subheader("Win Rate by Bet Market")
    fig = px.line(table, 
                  x = 'bet_date_time',
                  y = 'daily_running_win_rate',
                  color = 'market_name',
                  line_group = 'market_name',
                  markers = True,
                  text = [f"{i:.2%}" for i in table['daily_running_win_rate']],
                  labels = {'bet_date_time':'Date','daily_running_win_rate':'Daily Running Win Rate'}
                 )
    # fig.update_layout(title_text = 'Win Rate by Bet Market',title_x=.5)
    fig.layout.yaxis.tickformat = ',.0%'
    fig.update(layout_yaxis_range = [0,1],
              )
    fig.update_traces(textposition = 'top center')
    st.plotly_chart(fig, use_container_width = True)

    overall_bets = len(nba_selection)
    won_bets = len(nba_selection[nba_selection['bet_result']=='WON'])
    lost_bets = len(nba_selection[nba_selection['bet_result']=='LOST'])
    net_earnings = "${:.2f}".format(nba_selection['payout'].sum() - nba_selection['bet_amount'].sum())
    roi = '{:.3%}'.format((nba_selection['payout'].sum() - nba_selection['bet_amount'].sum()) / nba_selection['bet_amount'].sum())
    try:
        win_rate = '{:.3%}'.format(len(nba_selection[nba_selection['bet_result']=='WON'])  / len(nba_selection))
    except ZeroDivisionError:
        win_rate = "-"

    ML_overall_bets = len(nba_selection[nba_selection['market_name']=='Moneyline'])
    ML_won_bets = len(nba_selection[(nba_selection['bet_result']=='WON') & (nba_selection['market_name']=='Moneyline')])
    ML_lost_bets = len(nba_selection[(nba_selection['bet_result']=='LOST') & (nba_selection['market_name']=='Moneyline')])
    ML_net_earnings = "${:.2f}".format(nba_selection[nba_selection['market_name']=='Moneyline']['payout'].sum() - nba_selection[nba_selection['market_name']=='Moneyline']['bet_amount'].sum())
    ML_roi = '{:.3%}'.format((nba_selection[nba_selection['market_name']=='Moneyline']['payout'].sum() - nba_selection[nba_selection['market_name']=='Moneyline']['bet_amount'].sum()) / nba_selection[nba_selection['market_name']=='Moneyline']['bet_amount'].sum())
    try:
        ML_win_rate = '{:.3%}'.format(len(nba_selection[(nba_selection['bet_result']=='WON') & (nba_selection['market_name']=='Moneyline')]) / len(nba_selection[nba_selection['market_name']=='Moneyline']))
    except ZeroDivisionError:
        ML_win_rate = "-"

    spread_overall_bets = len(nba_selection[nba_selection['market_name']=='Spread Betting'])
    spread_won_bets = len(nba_selection[(nba_selection['bet_result']=='WON') & (nba_selection['market_name']=='Spread Betting')])
    spread_lost_bets = len(nba_selection[(nba_selection['bet_result']=='LOST') & (nba_selection['market_name']=='Spread Betting')])
    spread_net_earnings = "${:.2f}".format(nba_selection[nba_selection['market_name']=='Spread Betting']['payout'].sum() - nba_selection[nba_selection['market_name']=='Spread Betting']['bet_amount'].sum())
    spread_roi = '{:.3%}'.format((nba_selection[nba_selection['market_name']=='Spread Betting']['payout'].sum() - nba_selection[nba_selection['market_name']=='Spread Betting']['bet_amount'].sum()) / nba_selection[nba_selection['market_name']=='Spread Betting']['bet_amount'].sum())
    try:
        spread_win_rate = '{:.3%}'.format(len(nba_selection[(nba_selection['bet_result']=='WON') & (nba_selection['market_name']=='Spread Betting')]) / len(nba_selection[nba_selection['market_name']=='Spread Betting']))
    except ZeroDivisionError:
        spread_win_rate = "-"

    total_overall_bets = len(nba_selection[nba_selection['market_name']=='Total Points'])
    total_won_bets = len(nba_selection[(nba_selection['bet_result']=='WON') & (nba_selection['market_name']=='Total Points')])
    total_lost_bets = len(nba_selection[(nba_selection['bet_result']=='LOST') & (nba_selection['market_name']=='Total Points')])
    total_net_earnings = "${:.2f}".format(nba_selection[nba_selection['market_name']=='Total Points']['payout'].sum() - nba_selection[nba_selection['market_name']=='Total Points']['bet_amount'].sum())
    total_roi = '{:.3%}'.format((nba_selection[nba_selection['market_name']=='Total Points']['payout'].sum() - nba_selection[nba_selection['market_name']=='Total Points']['bet_amount'].sum()) / nba_selection[nba_selection['market_name']=='Total Points']['bet_amount'].sum())
    try:
        total_win_rate = '{:.3%}'.format(len(nba_selection[(nba_selection['bet_result']=='WON') & (nba_selection['market_name']=='Total Points')]) / len(nba_selection[nba_selection['market_name']=='Total Points']))
    except ZeroDivisionError:
        total_win_rate = "-"

    overall_KPIs = ['overall_bets','won_bets','lost_bets','net_earnings','win_rate','roi']    

    ML_KPIs = ['ML_overall_bets','ML_won_bets','ML_lost_bets','ML_net_earnings','ML_roi','ML_win_rate']

    spread_KPIs = ['spread_overall_bets','spread_won_bets','spread_lost_bets','spread_net_earnings','spread_roi','spread_win_rate']

    totals_KPIs = ['total_overall_bets','total_won_bets','total_lost_bets','total_net_earnings','total_roi','total_win_rate']

    if MARKET_NAME == 'Moneyline':
        for s in spread_KPIs:
            globals()[s] = "-"
        for t in totals_KPIs:
            globals()[t] = "-"
    elif MARKET_NAME == 'Spread Betting':
        for m in ML_KPIs:
            globals()[m] = "-"
        for t in totals_KPIs:
            globals()[t] = "-"
    elif MARKET_NAME == 'Total Points':
        for m in ML_KPIs:
            globals()[m] = "-"
        for s in spread_KPIs:
            globals()[s] = "-"

    col1,col2,col3,col4,col5 = st.columns(5, gap = 'small')
    with col1:
        st.subheader("*KPI*")
        st.subheader(f"*Total Bets:*")
        st.subheader(f"*Won Bets:*")
        st.subheader(f"*Lost Bets:*")
        st.subheader(f"*Win Rate:*")
        st.subheader(f"*Net Earnings:*")
        st.subheader(f"*ROI:*")

    with col2:
        st.subheader('*Overall:*')
        st.subheader(overall_bets)
        st.subheader(won_bets)
        st.subheader(lost_bets)
        st.subheader(win_rate)
        st.subheader(net_earnings)
        st.subheader(roi)

    with col3:
        st.subheader('*ML:*')
        st.subheader(ML_overall_bets)
        st.subheader(ML_won_bets)
        st.subheader(ML_lost_bets)
        st.subheader(ML_win_rate)
        st.subheader(ML_net_earnings)
        st.subheader(ML_roi)
    with col4:
        st.subheader('*Spread:*')
        st.subheader(spread_overall_bets)
        st.subheader(spread_won_bets)
        st.subheader(spread_lost_bets)
        st.subheader(spread_win_rate)
        st.subheader(spread_net_earnings)
        st.subheader(spread_roi)
    with col5:
        st.subheader('*Totals:*')
        st.subheader(total_overall_bets)
        st.subheader(total_won_bets)
        st.subheader(total_lost_bets)
        st.subheader(total_win_rate)
        st.subheader(total_net_earnings)
        st.subheader(total_roi)
