import csv
import os

Y_STATS = [
    "Passing Yards",
    "TD Passes",
    "Ints",
    "Rushing Yards",
    "Rushing TDs",
    "Receiving Yards",
    "Receiving TDs",
    "Fumbles",
]

X_STATS = [
    "Games Played",
    "Passes Attempted",
    "Passes Completed",
    "Completion Percentage",
    "Pass Attempts Per Game",
    "Passing Yards",
    "Passing Yards Per Attempt",
    "Passing Yards Per Game",
    "TD Passes",
    "Percentage of TDs per Attempts",
    "Ints",
    "Int Rate",
    "Longest Pass",
    "Passes Longer than 20 Yards",
    "Passes Longer than 40 Yards",
    "Sacks",
    "Sacked Yards Lost",
    "Passer Rating",
    "Rushing Attempts",
    "Rushing Attempts Per Game",
    "Rushing Yards",
    "Yards Per Carry",
    "Rushing Yards Per Game",
    "Rushing TDs",
    "Longest Rushing Run",
    "Rushing First Downs",
    "Percentage of Rushing First Downs",
    "Rushing More Than 20 Yards",
    "Rushing More Than 40 Yards",
    "Receptions",
    "Receiving Yards",
    "Yards Per Reception",
    "Yards Per Game",
    "Longest Reception",
    "Receiving TDs",
    "Receptions Longer than 20 Yards",
    "Receptions Longer than 40 Yards",
    "First Down Receptions",
    "Fumbles",
]


def create_data_csv(file_paths: list, number_of_previous_years: int = 3) -> None:
    player_data = dict()
    for file in file_paths:
        with open(file, 'r') as f:
            headers = f.readline().replace("\n", "")
            headers = headers.split(",")
            data = list(csv.reader(f, delimiter=","))
        for player in data:
            if not player_data.get(player[0]):
                player_data[player[0]] = dict()
            year_idx = headers.index("Year")
            if not player_data[player[0]].get(player[year_idx]):
                player_data[player[0]][player[year_idx]] = dict()
            for idx, stat_desc in enumerate(headers):
                if player[idx] == "--":
                    player[idx] = 0.
                if stat_desc not in ["Player Id", "Name", "Position", "Year", "Team"]:
                    if type(player[idx]) == str:
                        player[idx] = player[idx].replace(",", "").replace("T", "")  # Random 'T's are added in
                    player[idx] = float(player[idx])
                if stat_desc == "Fumbles" and player_data[player[0]][player[year_idx]].get("Fumbles"):
                    player_data[player[0]][player[year_idx]][stat_desc] += player[idx]
                    continue
                player_data[player[0]][player[year_idx]][stat_desc] = player[idx]

    processed_players = list()
    # dict of all players with yearly stats
    # sort years by max first
    players = list(player_data.keys())
    for player in players:
        years = list(player_data[player].keys())
        years.sort(reverse=True)
        for idx, year in enumerate(years):
            if idx == len(years) - 1:  # last year, meaning no prior data
                continue

            y = list()
            for stat in Y_STATS:
                y.append(player_data[player][year].get(stat, 0.0))

            x = list()
            for i in range(idx + 1, number_of_previous_years + idx + 1):
                if i > len(years) - 1:
                    for stat in X_STATS:
                        if stat == "Team":
                            x.append("")
                        else:
                            x.append(0.0)
                else:
                    for stat in X_STATS:
                        x_year = years[i]
                        x.append(player_data[player][x_year].get(stat, 0.0))

            processed_players.append(x + y)

    # write the file
    with open(f"./data/clean_data_{number_of_previous_years}.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = list()
        for i in range(1, number_of_previous_years + 1):
            for header in X_STATS:
                headers.append(f"Feature {header} Year -{i}")
        for header in Y_STATS:
            headers.append(f"Target {header}")
        csvwriter.writerow(headers)
        for player in processed_players:
            csvwriter.writerow(player)


if __name__ == "__main__":
    _file_paths = [
        os.path.join("data", "Career_Stats_Passing.csv"),
        os.path.join("data", "Career_Stats_Rushing.csv"),
        os.path.join("data", "Career_Stats_Receiving.csv"),
    ]
    create_data_csv(_file_paths, number_of_previous_years=1)
