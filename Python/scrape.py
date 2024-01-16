import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# URL of the website containing the table
url = 'https://www.tennisexplorer.com/australian-open/2024/atp-men/'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with the class 'result'
table = soup.find('table', class_='result')

# Create lists to store data
players_1 = []
players_2 = []
odds_1 = []
odds_2 = []
# Function to clean player names
def clean_player_name(name):
    return re.sub(r'\(\d+\)', '', name).strip()

# Loop through rows in the table
for row in table.find_all('tr')[1:]:  # Skip the header row
    cols = row.find_all('td')
    player_1 = clean_player_name(cols[2].text.strip().split(' - ')[0])
    player_2 = clean_player_name(cols[2].text.strip().split(' - ')[1])
    odd_1 = cols[5].text.strip()
    odd_2 = cols[6].text.strip()
  # Convert odds to numeric for comparison
    odd_1 = float(odd_1)
    odd_2 = float(odd_2)

    # Swap players and odds if odds_1 > odds_2
    if odd_1 > odd_2:
        player_1, player_2 = player_2, player_1
        odd_1, odd_2 = odd_2, odd_1

    players_1.append(player_1)
    players_2.append(player_2)
    odds_1.append(odd_1)
    odds_2.append(odd_2)

# # Create a DataFrame
data = {'Player 1': players_1, 'Player 2': players_2, 'Player 1 Odds': odds_1, 'Player 2 Odds': odds_2}
df = pd.DataFrame(data)


    # players_1.append(player_1)
    # players_2.append(player_2)
    # odds_1.append(odd_1)
    # odds_2.append(odd_2)

# Create a DataFrame
# data = {'Player 1': players_1, 'Player 2': players_2, 'Player 1 Odds': odds_1, 'Player 2 Odds': odds_2}
# df = pd.DataFrame(data)

# # Write to CSV
# df.to_csv('tennis_matches.csv', index=False)


# Write to CSV
df.to_csv('tennis_matches_sorted.csv', index=False)

# # Convert odds to numeric for sorting
# df['Player 1 Odds'] = pd.to_numeric(df['Player 1 Odds'])
# df['Player 2 Odds'] = pd.to_numeric(df['Player 2 Odds'])

# # Sort the DataFrame based on odds
# df.sort_values(by=['Player 1 Odds', 'Player 2 Odds'], inplace=True)

# # Write to CSV
# df.to_csv('tennis_matches_sorted.csv', index=False)

