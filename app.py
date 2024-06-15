import pandas as pd
import numpy as np
import streamlit as st
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit app configuration
st.set_page_config(page_title="Dream 11 Team Builder", page_icon="🏏", layout='wide')

# Cache the data loading function
@st.cache_data
def load_data():
    """Load the IPL ball-by-ball dataset."""
    return pd.read_csv("./IPL_BallByBall2008_2024(Updated).csv", low_memory=False)

def main():
    """Main function to run the Streamlit app."""
    st.title("Dream 11 Players Rank Optimizer - IPL")
    st.text("""
    Hi Amogh here,
    This is the first draft of my Dream 11 predictor app.
    I have added 5 KPIs, which evaluate player performance to generate aggregate
    rankings on an individual level.
            
    1. Overall Performance
    2. Recent Form 
    3. Venue Stats
    4. Innings Stats
    5. Performance vs Opponent
    """)

    # Load and preprocess the data
    df = load_data()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.drop(columns=['type of extras']).fillna(0)
    df = df[df['Innings No'] <= 2]
    df['Date'] = pd.to_datetime(df['Date'])

    extra_df = pd.read_csv("./team_performance_dataset_2008to2024.csv")
    extra_df.rename(columns={'Match_ID':'Match id'}, inplace=True)
    extra_df['Date'] = pd.to_datetime(extra_df['Date'])
    df = pd.merge(df, extra_df, on=['Match id', 'Date'])

    # Rename teams
    rename_teams = {
        "Kings XI Punjab": "Punjab Kings",
        "Delhi Daredevils": "Delhi Capitals",
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Gujarat Lions': 'Gujarat Titans',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Rising Pune Supergiant': 'Chennai Super Kings',
        'Rising Pune Supergiants': 'Chennai Super Kings'
    }

    for col in df.columns.tolist():
        try:
            df[col] = df[col].replace(rename_teams)
        except Exception as e:
            pass
            
    rename_venues = {'Arun Jaitley Stadium':"Arun Jaitley Stadium, Delhi",
                 'Feroz Shah Kotla' : "Arun Jaitley Stadium, Delhi",
                 'Brabourne Stadium': 'Brabourne Stadium, Mumbai',
                 'Dr DY Patil Sports Academy':'Dr DY Patil Sports Academy, Mumbai',
                'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam',
                'Eden Gardens':'Eden Gardens, Kolkata', 
                 'Himachal Pradesh Cricket Association Stadium':'Himachal Pradesh Cricket Association Stadium, Dharamsala',
                 'M Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
                 'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium, Bengaluru',
                 'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chepauk, Chennai',
                 'Maharashtra Cricket Association Stadium':'Maharashtra Cricket Association Stadium, Pune',
                'Punjab Cricket Association IS Bindra Stadium':'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
               'Punjab Cricket Association IS Bindra Stadium, Mohali':'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
               'Punjab Cricket Association Stadium, Mohali':'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
                'Rajiv Gandhi International Stadium':'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
               'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
                'Sawai Mansingh Stadium':'Sawai Mansingh Stadium, Jaipur',
                 'Wankhede Stadium':'Wankhede Stadium, Mumbai',
                }
    try:
        df['Venue'] = df['Venue'].replace(rename_venues)
    except Exception as e:
        pass

    # Drop inactive teams
    drop_teams = ['Kochi Tuskers Kerala', 'Pune Warriors']
    df = df[~df['Batting team'].isin(drop_teams) & ~df['Bowling team'].isin(drop_teams)]

    # Get current players and their teams
    batters_in_2024_df = df[df['Season'] == '2024'][['Striker', 'Batting team']].drop_duplicates().reset_index(drop=True).rename(columns={'Striker':'Player', 'Batting team':'Team'})
    bowlers_in_2024_df = df[df['Season'] == '2024'][['Bowler', 'Bowling team']].drop_duplicates().reset_index(drop=True).rename(columns={'Bowler':'Player', 'Bowling team':'Team'})
    curr_players_teams_df = batters_in_2024_df.merge(bowlers_in_2024_df, on=['Player', 'Team'], how='outer')

    # User inputs
    batting_team = st.selectbox("Select the Batting Team:", options=curr_players_teams_df['Team'].unique())
    bowling_team = st.selectbox("Select the Bowling Team:", options=curr_players_teams_df['Team'].unique())
    stadium = st.selectbox("Select a Venue:", options=df['Venue'].sort_values().unique())

    if batting_team == bowling_team:
        st.error("The Batting and Bowling teams must not be the same.")
    else:
        # Get players for the selected teams
        batting_team_players = curr_players_teams_df[curr_players_teams_df['Team'].str.contains(batting_team)]['Player'].tolist()
        bowling_team_players = curr_players_teams_df[curr_players_teams_df['Team'].str.contains(bowling_team)]['Player'].tolist()

        batter_dfs = {}
        bowler_dfs = {}

        # Define KPIs
        def overall_factor(df):
            """Calculate the overall performance factor for batters and bowlers."""
            x = df[df['Striker'].isin(batting_team_players + bowling_team_players)]
            x = x.groupby(['Match id', 'Striker'])['runs_scored'].sum().reset_index().groupby('Striker')['runs_scored'].mean().reset_index(name='overall_scores')
            x = x.sort_values(by='overall_scores', ascending=False).reset_index(drop=True)

            y = df[df['Bowler'].isin(batting_team_players + bowling_team_players)]
            y = y.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().reset_index().groupby('Bowler')['wicket_confirmation'].mean().reset_index(name='overall_scores')
            y = y.sort_values(by='overall_scores', ascending=False).reset_index(drop=True)

            return x.set_index('Striker'), y.set_index('Bowler')

        batter_dfs['OverallFactor'], bowler_dfs['OverallFactor'] = overall_factor(df.copy())

        def opponent_factor(df):
            """Calculate the performance factor against the opponent team."""
            x = df[(df['Striker'].isin(batting_team_players)) & (df['Bowling team'].str.contains(bowling_team))]
            x = x.groupby(['Match id', 'Striker'])['runs_scored'].sum().reset_index().groupby('Striker')['runs_scored'].mean().reset_index(name='opponent_scores')

            y = df[(df['Bowler'].isin(batting_team_players)) & (df['Batting team'].str.contains(bowling_team))]
            y = y.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().reset_index().groupby('Bowler')['wicket_confirmation'].mean().reset_index(name='opponent_scores')

            a = df[(df['Striker'].isin(bowling_team_players)) & (df['Bowling team'].str.contains(batting_team))]
            a = a.groupby(['Match id', 'Striker'])['runs_scored'].sum().reset_index().groupby('Striker')['runs_scored'].mean().reset_index(name='opponent_scores')

            b = df[(df['Bowler'].isin(bowling_team_players)) & (df['Batting team'].str.contains(batting_team))]
            b = b.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().reset_index().groupby('Bowler')['wicket_confirmation'].mean().reset_index(name='opponent_scores')

            u = pd.concat([x, a]).sort_values(by='opponent_scores', ascending=False).reset_index(drop=True)
            v = pd.concat([y, b]).sort_values(by='opponent_scores', ascending=False).reset_index(drop=True)
            return u.set_index('Striker'), v.set_index('Bowler')

        batter_dfs['OpponentFactor'], bowler_dfs['OpponentFactor'] = opponent_factor(df.copy())

        def venue_factor(df):
            """Calculate the performance factor at the selected venue."""
            a = df[(df['Striker'].isin(batting_team_players + bowling_team_players)) & df['Venue'].str.contains(stadium)]
            a = a.groupby(['Match id', 'Striker'])['runs_scored'].sum().reset_index(name='venue_runs').groupby('Striker')['venue_runs'].mean().reset_index(name='venue_score')
            a = a.sort_values(by='venue_score', ascending=False).reset_index(drop=True)

            b = df[(df['Bowler'].isin(batting_team_players + bowling_team_players)) & df['Venue'].str.contains(stadium)]
            b = b.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().reset_index(name='venue_wickets').groupby('Bowler')['venue_wickets'].mean().reset_index(name='venue_score')
            b = b.sort_values(by='venue_score', ascending=False).reset_index(drop=True)

            return a.set_index('Striker'), b.set_index('Bowler')

        batter_dfs['VenueFactor'], bowler_dfs['VenueFactor'] = venue_factor(df.copy())

        def form_factor(df):
            """Calculate the recent form factor based on the last 5 matches."""
            recent_match_ids = df[df['Batting team'].str.contains(batting_team)]['Match id'].unique()[-5:]
            r = df[df['Striker'].isin(batting_team_players) & df['Match id'].isin(recent_match_ids)] 
            r = r.groupby(['Match id', 'Striker'])['runs_scored'].sum().groupby('Striker').mean().reset_index(name='form_score')
            t = df[df['Bowler'].isin(batting_team_players) & df['Match id'].isin(recent_match_ids)] 
            t = t.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().groupby('Bowler').mean().reset_index(name='form_score')

            recent_match_ids = df[df['Batting team'].str.contains(bowling_team)]['Match id'].unique()[-5:]
            s = df[df['Striker'].isin(bowling_team_players) & df['Match id'].isin(recent_match_ids)] 
            s = s.groupby(['Match id', 'Striker'])['runs_scored'].sum().groupby('Striker').mean().reset_index(name='form_score')
            u = df[df['Bowler'].isin(bowling_team_players) & df['Match id'].isin(recent_match_ids)] 
            u = u.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().groupby('Bowler').mean().reset_index(name='form_score')
        
            a = pd.concat([r, s]).sort_values(by='form_score', ascending=False).reset_index(drop=True)
            b = pd.concat([t, u]).sort_values(by='form_score', ascending=False).reset_index(drop=True)

            return a.set_index('Striker'), b.set_index('Bowler')

        batter_dfs['FormFactor'], bowler_dfs['FormFactor'] = form_factor(df.copy())

        def innings_factor(df):
            """Calculate the innings performance factor."""
            a = df[(df['Striker'].isin(batting_team_players)) & (df['Innings No'] == 1)]
            a = a.groupby(['Match id', 'Striker'])['runs_scored'].sum().reset_index(name='first_inning_runs')
            a = a.groupby('Striker')['first_inning_runs'].mean().reset_index(name='innings_score')
            a = a.sort_values(by='innings_score', ascending=False)

            b = df[(df['Bowler'].isin(bowling_team_players)) & (df['Innings No'] == 1)]
            b = b.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().reset_index(name='first_inning_wickets')
            b = b.groupby('Bowler')['first_inning_wickets'].mean().reset_index(name='innings_score')
            b = b.sort_values(by='innings_score', ascending=False)

            c = df[(df['Striker'].isin(bowling_team_players)) & (df['Innings No'] == 2)]
            c = c.groupby(['Match id', 'Striker'])['runs_scored'].sum().reset_index(name='second_inning_runs')
            c = c.groupby('Striker')['second_inning_runs'].mean().reset_index(name='innings_score')
            c = c.sort_values(by='innings_score', ascending=False)
        
            d = df[(df['Bowler'].isin(batting_team_players)) & (df['Innings No'] == 2)]
            d = d.groupby(['Match id', 'Bowler'])['wicket_confirmation'].sum().reset_index(name='second_inning_wickets')
            d = d.groupby('Bowler')['second_inning_wickets'].mean().reset_index(name='innings_score')
            d = d.sort_values(by='innings_score', ascending=False)

            p = pd.concat([a, c]).sort_values(by='innings_score', ascending=False).reset_index(drop=True)
            q = pd.concat([b, d]).sort_values(by='innings_score', ascending=False).reset_index(drop=True)
        
            return p.set_index("Striker"), q.set_index("Bowler")

        batter_dfs['InningsFactor'], bowler_dfs['InningsFactor'] = innings_factor(df.copy())

        # Combine the factors to calculate final scores
        batters_dfs_concatinated = pd.concat(batter_dfs.values(), axis=1).fillna(0)
        bowlers_dfs_concatinated = pd.concat(bowler_dfs.values(), axis=1).fillna(0)

        batters_dfs_concatinated['final_batter_score'] = batters_dfs_concatinated['form_score']*0.30 + batters_dfs_concatinated['venue_score']*0.25 + batters_dfs_concatinated['opponent_scores']*0.20 + batters_dfs_concatinated['innings_score']*0.15 + batters_dfs_concatinated['overall_scores']*0.10
        bowlers_dfs_concatinated['final_bowler_score'] = bowlers_dfs_concatinated['form_score']*0.30 + bowlers_dfs_concatinated['venue_score']*0.25 + bowlers_dfs_concatinated['opponent_scores']*0.20 + bowlers_dfs_concatinated['innings_score']*0.15 + bowlers_dfs_concatinated['overall_scores']*0.10

        result_df = pd.concat([batters_dfs_concatinated[['final_batter_score']], bowlers_dfs_concatinated[['final_bowler_score']]], axis=1)
        result_df = result_df.rank(ascending=False).min(axis=1).reset_index(name='Final Rank')
        result_df.rename(columns={'index':'Players'}, inplace=True)
        result_df = result_df.sort_values(by='Final Rank')

        # Generate optimized ranks on button click
        gen_button = st.button("Generate Optimized Ranks")
        if gen_button:
            result_df = result_df[['Final Rank', 'Players']]
            result_df.set_index('Final Rank', inplace=True)
            st.dataframe(result_df, width=500)

if __name__ == "__main__":
    main()
