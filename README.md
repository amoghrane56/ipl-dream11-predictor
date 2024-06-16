Overview
Welcome to the Dream 11 Players Rank Optimizer for IPL! This Streamlit app helps you optimize your Dream 11 fantasy cricket team by ranking players based on various key performance indicators (KPIs). The app evaluates player performance to generate aggregate rankings on an individual level.

Features
The app considers the following KPIs to rank players:

Overall Performance: Measures the player's average performance over their career.
Recent Form: Evaluates the player's performance in the last 5 matches.
Venue Stats: Assesses the player's performance at a specific venue.
Innings Stats: Analyzes the player's performance in different innings (first and second).
Performance vs Opponent: Considers the player's performance against a specific opponent.


Usage
Load the Data:
The app loads the IPL ball-by-ball dataset (IPL_BallByBall2008_2024(Updated).csv) and the team performance dataset (team_performance_dataset_2008to2024.csv). Ensure these files are in the same directory as the app.

Configure the Inputs:
Select the batting team, bowling team, and venue from the dropdown menus.

Generate Optimized Ranks:
Click on the "Generate Optimized Ranks" button to see the optimized player rankings based on the selected KPIs.

Data Preprocessing
The data undergoes several preprocessing steps to ensure accuracy and consistency:

Strips extra spaces from strings.
Fills missing values with zeros.
Filters data to include only the first two innings.
Renames teams and venues to maintain uniformity.
Drops inactive teams.
Merges additional team performance data.
Player Ranking
The app calculates player rankings based on a weighted sum of the KPIs:

Form Score: 30%
Venue Score: 25%
Opponent Scores: 20%
Innings Score: 15%
Overall Scores: 10%
The final rankings are displayed in a table, with players ranked from highest to lowest.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the app.

Acknowledgements
Streamlit
Pandas
NumPy
Author
Amogh

