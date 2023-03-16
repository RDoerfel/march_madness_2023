#%% imports
from pathlib import Path
import pandas as pd

#%% data loading
data_dir = Path(__file__).parent.parent / 'data'
kaggle_dir = data_dir / 'kaggle'
df_results = pd.read_csv(data_dir / 'submission.csv')
df_seeds = pd.read_csv(kaggle_dir / 'MNCAATourneySeeds.csv')
df_rounds = pd.read_csv(kaggle_dir / 'MNCAATourneySlots.csv')
df_team_name = pd.read_csv(kaggle_dir / 'MTeams.csv')

#%% correct results
df_results['Pred'] = 1 - df_results['Pred']

# %% get seeds and round for Season == 2023
df_seeds_2023 = df_seeds[df_seeds['Season'] == 2023].drop(columns=['Season'])
df_rounds_2023 = df_rounds[df_rounds['Season'] == 2023].drop(columns=['Season'])

# %% get names for teams in 2023
df_team_name_2023 = df_team_name[df_team_name['TeamID'].isin(df_seeds_2023['TeamID'])]
df_team_name_2023 = df_team_name_2023[['TeamID', 'TeamName']]

# mearge seeds and rounds based on strong seed
df_rounds_2023 = pd.merge(df_rounds_2023,df_seeds_2023, how='left', left_on=['StrongSeed'], right_on=['Seed']).rename(columns={'TeamID':'StrongTeamID'}).drop(columns=['Seed'])
df_rounds_2023 = pd.merge(df_rounds_2023,df_seeds_2023, how='left', left_on=['WeakSeed'], right_on=['Seed']).rename(columns={'TeamID':'WeakTeamID'}).drop(columns=['Seed'])

# add team names
df_rounds_2023 = pd.merge(df_rounds_2023,df_team_name_2023, how='left', left_on=['StrongTeamID'], right_on=['TeamID']).rename(columns={'TeamName':'StrongTeamName'}).drop(columns=['TeamID'])
df_rounds_2023 = pd.merge(df_rounds_2023,df_team_name_2023, how='left', left_on=['WeakTeamID'], right_on=['TeamID']).rename(columns={'TeamName':'WeakTeamName'}).drop(columns=['TeamID'])

# replace nan with 0
df_rounds_2023 = df_rounds_2023.fillna(0)

# convert to int
df_rounds_2023['StrongTeamID'] = df_rounds_2023['StrongTeamID'].astype(int)
df_rounds_2023['WeakTeamID'] = df_rounds_2023['WeakTeamID'].astype(int)

# combine values for each game to 2023_<lowerID>_<higherID>
df_rounds_2023['LowerTeamID'] = df_rounds_2023.apply(lambda x: min(x['StrongTeamID'], x['WeakTeamID']), axis=1)
df_rounds_2023['HigherTeamID'] = df_rounds_2023.apply(lambda x: max(x['StrongTeamID'], x['WeakTeamID']), axis=1)
df_rounds_2023['GameID'] = df_rounds_2023.apply(lambda x: f'2023_{x["LowerTeamID"]}_{x["HigherTeamID"]}', axis=1)

# add lower and higher team name
df_rounds_2023 = pd.merge(df_rounds_2023,df_team_name_2023, how='left', left_on=['LowerTeamID'], right_on=['TeamID']).rename(columns={'TeamName':'LowerTeamName'}).drop(columns=['TeamID'])
df_rounds_2023 = pd.merge(df_rounds_2023,df_team_name_2023, how='left', left_on=['HigherTeamID'], right_on=['TeamID']).rename(columns={'TeamName':'HigherTeamName'}).drop(columns=['TeamID'])

# get results for each game
df_results_2023 = pd.merge(df_rounds_2023,df_results, how='left', left_on=['GameID'], right_on=['ID']).drop(columns=['ID'])

# binarize predictions
df_results_2023['Res'] = df_results_2023['Pred'].apply(lambda x: 1 if x > 0.5 else 0)

#add region col to df
df_results_2023['Region'] = df_results_2023['StrongSeed'].apply(lambda x: x[0])

# drop rows with nan
df_results_2023 = df_results_2023.dropna()

#%% print first round
# print for seeds starting with W
def print_region(df, region):
    print(region)
    region_code = {'East': 'W', 'Midwest': 'X', 'South': 'Y', 'West': 'Z'}
    region_df = df[df['Region']==region_code[region]]
    print(region_df[['LowerTeamName', 'HigherTeamName', 'Pred', 'Res']])

print_region(df_results_2023, 'East')
print_region(df_results_2023, 'Midwest')
print_region(df_results_2023, 'South')
print_region(df_results_2023, 'West')

#%% print second round
# if res == 1, then lower team wins, else higher team wins
# print IDs of teams that won
def print_winners_first_round(df, region):
    print(region)
    region_code = {'East': 'W', 'Midwest': 'X', 'South': 'Y', 'West': 'Z'}
    region_df = df[df['Region']==region_code[region]]
    winners = []
    for index, row in region_df.iterrows():
        if row['Res'] == 1:
            winners.append(row['LowerTeamID'])
        else:
            winners.append(row['HigherTeamID'])

    # create new df with winners
    df_winners = pd.DataFrame({'TeamID': winners})
    df_winners = pd.merge(df_winners,df_team_name_2023, how='left', left_on=['TeamID'], right_on=['TeamID'])

    # print winners
    print(df_winners[['TeamID', 'TeamName']])
    
# print winners of first round
print_winners_first_round(df_results_2023, 'East')
print_winners_first_round(df_results_2023, 'Midwest')
print_winners_first_round(df_results_2023, 'South')
print_winners_first_round(df_results_2023, 'West')

#%% sec_round 
# Midwest
2023_1104_1268
2023_1158_1438
2023_1301_1364
2023_1112_1429

# South
2023_1120_1222
2023_1179_1245
2023_1244_1338
2023_1400_1401

# West
2023_1228_1242
2023_1233_1433
2023_1211_1305
2023_1129_1417

# East
2023_1192_1194
2023_1331_1418
2023_1344_1286
2023_1266_1425

# make the above into strings
east_strings = ['2023_1192_1194', '2023_1331_1418', '2023_1344_1286', '2023_1266_1425']
midwest_strings = ['2023_1104_1268', '2023_1158_1438', '2023_1301_1364', '2023_1112_1429']
south_strings = ['2023_1120_1222', '2023_1179_1245', '2023_1244_1338', '2023_1400_1401']
west_strings = ['2023_1228_1242', '2023_1233_1433', '2023_1211_1305', '2023_1129_1417']

# build df from string list with lower and higher id and team name
     
def print_results_2(strings, results, names):
    for string in strings:
        lower_id = string.split('_')[1]
        higher_id = string.split('_')[2]
        lower_name = names[names['TeamID']==int(lower_id)]['TeamName'].values[0]
        higher_name = names[names['TeamID']==int(higher_id)]['TeamName'].values[0]
        pred = results[results['ID']==string]['Pred'].to_string(index=False)
        
        print(lower_name, lower_id, higher_name, higher_id, pred)

print("east")
print_results_2(east_strings, df_results, df_team_name_2023)
print("midwest")
print_results_2(midwest_strings, df_results, df_team_name_2023)
print("south")
print_results_2(south_strings, df_results, df_team_name_2023)
print("west")
print_results_2(west_strings, df_results, df_team_name_2023)


#%% round 3
# east
2023_1194_1331
2023_1266_1344

# midwest
2023_1104_1158
2023_1112_1364

# south
2023_1222_1245
2023_1244_1400

# west
2023_1242_1433
2023_1211_1417

# convert to strings
east_strings = ['2023_1194_1331', '2023_1266_1344']
midwest_strings = ['2023_1104_1158', '2023_1112_1364']
south_strings = ['2023_1222_1245', '2023_1244_1400']
west_strings = ['2023_1242_1433', '2023_1211_1417']

# print results
print("NextRound")
print("east")
print_results_2(east_strings, df_results, df_team_name_2023)
print("midwest")
print_results_2(midwest_strings, df_results, df_team_name_2023)
print("south")
print_results_2(south_strings, df_results, df_team_name_2023)
print("west")
print_results_2(west_strings, df_results, df_team_name_2023)

#%% round 4
# east
2023_1194_1266

# midwest
2023_1112_1158

# south
2023_1222_1400

# west
2023_1242_1417

# convert to strings in one list
games = ['2023_1194_1266', '2023_1112_1158', '2023_1222_1400', '2023_1242_1417']

# print results
print("SemiRound")
print_results_2(games, df_results, df_team_name_2023)

# %% final four
2023_1158_1194
2023_1222_1417

# convert to strings in one list
games = ['2023_1158_1194', '2023_1222_1417']

# print results
print("FinalFour")
print_results_2(games, df_results, df_team_name_2023)

# %% Final 
2023_1158_1222

# convert to strings in one list
games = ['2023_1158_1222']

# print results
print("Final")
print_results_2(games, df_results, df_team_name_2023)

# %%
