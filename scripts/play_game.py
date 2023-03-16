#%% imports
from pathlib import Path
import pandas as pd
pd.set_option('mode.chained_assignment', None)

# defs
def get_game_id(row):
    return f'2023_{min(row["TeamIDA"], row["TeamIDB"])}_{max(row["TeamIDA"], row["TeamIDB"])}'

def get_predictions(df_round, df_results):
    df_round = df_round.merge(df_results, left_on='GameID', right_on='ID').drop(columns=['ID'])
    return df_round

def get_winner(row):
    if row['Pred'] > 0.5:
        return row['TeamIDA']
    else:
        return row['TeamIDB']

def get_winner_name(row):
    if row['Pred'] > 0.5:
        return row['TeamNameA']
    else:
        return row['TeamNameB']
    
def play_round(df_rounds_2023, df_results, winner_df, round):
    df_round = df_rounds_2023[df_rounds_2023['Round'] == round]
    df_round.rename(columns={'StrongSeed': 'SlotA', 'WeakSeed': 'SlotB'}, inplace=True)

    # get TeamIDA and TeamNameA from winner_df
    df_round = df_round.merge(winner_df, left_on='SlotA', right_on='Seed').drop(columns=['Seed'])
    df_round = df_round.rename(columns={'TeamName': 'TeamNameA', 'TeamID': 'TeamIDA'})

    # get TeamIDB and TeamNameB from winner_df
    df_round = df_round.merge(winner_df, left_on='SlotB', right_on='Seed').drop(columns=['Seed'])
    df_round = df_round.rename(columns={'TeamName': 'TeamNameB', 'TeamID': 'TeamIDB'})

    # compose GameID as 2023_LowerID_HigherID
    df_round['GameID'] = df_round.apply(get_game_id, axis=1)
    df_round = get_predictions(df_round, df_results)
    df_round['WinnerID'] = df_round.apply(get_winner, axis=1)
    df_round['WinnerName'] = df_round.apply(get_winner_name, axis=1)

    # TeamNameA, TeamNameB, Pred, WinnerName
    print('Round ', round)
    print(df_round[['Slot', 'TeamNameA', 'TeamNameB', 'Pred', 'WinnerName']])
    winner_df = df_round[['Slot', 'WinnerID', 'WinnerName']]
    winner_df = winner_df.rename(columns={'WinnerID': 'TeamID', 'WinnerName': 'TeamName', 'Slot': 'Seed'})
    return winner_df

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
df_seeds_2023 = df_seeds[df_seeds['Season'] == 2023].drop(columns=['Season']).reset_index(drop=True)
df_rounds_2023 = df_rounds[df_rounds['Season'] == 2023].drop(columns=['Season']).reset_index(drop=True)

df_extra = df_rounds_2023[63:67]
df_rounds_2023 = df_rounds_2023[0:63]

# add round column to df_rounds_2023
df_rounds_2023['Round'] = df_rounds_2023.apply(lambda x: x['Slot'][1:2], axis=1)

# %% merge df_seeds_2023 and df_team_name based on TeamID
df_seeds_2023 = df_seeds_2023.merge(df_team_name, left_on='TeamID', right_on='TeamID').drop(columns=['FirstD1Season', 'LastD1Season'])

#%% play game
winner1_df = play_round(df_rounds_2023, df_results, df_seeds_2023, '1')
winner2_df = play_round(df_rounds_2023, df_results, winner1_df, '2')
winner3_df = play_round(df_rounds_2023, df_results, winner2_df, '3')
winner4_df = play_round(df_rounds_2023, df_results, winner3_df, '4')
winner5_df = play_round(df_rounds_2023, df_results, winner4_df, '5')
winner6_df = play_round(df_rounds_2023, df_results, winner5_df, '6')

# %%
"""
X16a, 1369, SE Missouri St
X16b, 1394, TAM C. Christi XX

Y11a, 1280, Mississippi St
Y11b, 1338, Pittsburgh XX

Z11a, 1113, Arizona St XX
Z11b, 1305, Nevada

W16a, 1192, F Dickinson XX
W16b, 1411, TX Southern
"""
