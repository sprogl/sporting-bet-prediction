import pandas as pd
import numpy as np
import requests
import io
import datetime
import joblib
import core.config

# Load scaler fromt the disk
scaler = joblib.load(filename="archive/scaler.pkl")

# Load svm model from disk
model_svm = joblib.load(filename="archive/model_svm.pkl")

# Load random forest model from disk
model_rfc = joblib.load(filename="archive/model_rfc.pkl")


def fetch():
    # Download the data
    url_base = "http://tennis-data.co.uk/"

    sess = requests.session()
    df_list = []
    for year in range(2000, 2013):
        url = f"{url_base}/{year}/{year}.xls"
        resp = sess.get(url)
        with open(f"archive/tournoments_{year}.xls", mode="wb") as file:
            file.write(resp.content)
        df_list.append(
            pd.read_excel(
                io.BytesIO(resp.content), sheet_name=f"{year}", engine="calamine"
            )
        )

    for year in range(2013, datetime.datetime.now().year + 1):
        url = f"{url_base}/{year}/{year}.xlsx"
        resp = sess.get(url)
        with open(f"archive/tournoments_{year}.xlsx", mode="wb") as file:
            file.write(resp.content)
        df_list.append(
            pd.read_excel(
                io.BytesIO(resp.content), sheet_name=f"{year}", engine="calamine"
            )
        )
    sess.close()

    # Put the downloaded data together
    df_atp = pd.concat(df_list, axis=0).drop(labels=core.config.drop_list, axis=1)

    # Cleanup the data
    df_atp["Winner"] = df_atp["Winner"].str.strip()
    df_atp["Loser"] = df_atp["Loser"].str.strip()

    df_atp["Winner"] = df_atp["Winner"].replace(core.config.correction_list)
    df_atp["Loser"] = df_atp["Loser"].replace(core.config.correction_list)

    df_atp["Best of"] = pd.to_numeric(
        df_atp["Best of"], errors="coerce", downcast="integer"
    )
    df_atp["WPts"] = pd.to_numeric(df_atp["WPts"], errors="coerce", downcast="integer")
    df_atp["LPts"] = pd.to_numeric(df_atp["LPts"], errors="coerce", downcast="integer")

    df_atp["Date"] = pd.to_datetime(df_atp["Date"])

    # Calculate the elo rate and number of played matches for each player at the begining of the match.
    # Also calculate the probability for the winner of the game to win the game based on the elo rates
    elo_start = 1500.0
    k_factor = 32.0

    df_atp[["match_count_winner", "match_count_loser"]] = 0
    df_atp[["elo_winner", "elo_loser"]] = elo_start
    df_atp["proba_elo"] = 0.5

    df_atp = df_atp.sort_values(by=["ATP", "Date"]).reset_index(drop=True).copy()

    elo_rates = pd.DataFrame(
        data={
            "Player": pd.concat([df_atp["Winner"], df_atp["Loser"]], axis=0).unique(),
            "pts": 0,
            "match_count": 0,
            "elo": elo_start,
        }
    )

    for index, _ in df_atp.iterrows():
        # Set atp points
        elo_rates.loc[elo_rates["Player"] == df_atp.loc[index, "Winner"], "pts"] = (
            df_atp.loc[index, "WPts"].values[0]
        )
        elo_rates.loc[elo_rates["Player"] == df_atp.loc[index, "Loser"], "pts"] = (
            df_atp.loc[index, "WPts"].values[0]
        )

        # Set the current elo rate/match count for the winner and loser of the match
        df_atp.loc[index, "elo_winner"] = elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Winner"], "elo"
        ].values[0]
        df_atp.loc[index, "elo_loser"] = elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Loser"], "elo"
        ].values[0]
        df_atp.loc[index, "match_count_winner"] = elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Winner"], "match_count"
        ].values[0]
        df_atp.loc[index, "match_count_loser"] = elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Loser"], "match_count"
        ].values[0]

        # Set the probability of the outcome based on the elo rates
        df_atp.loc[index, "proba_elo"] = 1.0 / (
            1.0
            + 10.0
            ** (
                0.0025
                * (df_atp.loc[index, "elo_loser"] - df_atp.loc[index, "elo_winner"])
            )
        )

        # Update the elo scores/match counts based on the outcome of the match
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Winner"], "elo"
        ] += k_factor * (1.0 - df_atp.loc[index, "proba_elo"])
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Loser"], "elo"
        ] -= k_factor * (1.0 - df_atp.loc[index, "proba_elo"])
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Winner"], "match_count"
        ] += 1
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Loser"], "match_count"
        ] += 1

    court_surface_type = (
        df_atp[["Court", "Surface"]]
        .drop_duplicates()
        .sort_values(by="Court")
        .to_numpy()
        .tolist()
    )

    # Initialize new columns in datasets
    for pair in court_surface_type:
        col_name = f"{pair[0].lower()}_{pair[1].lower()}"
        df_atp[[f"match_count_{col_name}_winner", f"match_count_{col_name}_loser"]] = 0
        df_atp[[f"elo_{col_name}_winner", f"elo_{col_name}_loser"]] = elo_start
        df_atp[f"proba_elo_{col_name}"] = 0.5

        elo_rates[f"match_count_{col_name}"] = 0
        elo_rates[f"elo_{col_name}"] = elo_start

    # Add the field specific elo rates and played match count of the players
    for index, _ in df_atp.iterrows():
        field_type = f"{df_atp.loc[index, 'Court'].lower()}_{df_atp.loc[index, 'Surface'].lower()}"
        for pair in court_surface_type:
            col_name = f"{pair[0].lower()}_{pair[1].lower()}"

            # Set the current match count for the winner and loser of the match (court/surface specific)
            df_atp.loc[index, f"match_count_{col_name}_winner"] = elo_rates.loc[
                elo_rates["Player"] == df_atp.loc[index, "Winner"],
                f"match_count_{col_name}",
            ].values[0]
            df_atp.loc[index, f"match_count_{col_name}_loser"] = elo_rates.loc[
                elo_rates["Player"] == df_atp.loc[index, "Loser"],
                f"match_count_{col_name}",
            ].values[0]

            # Set the current elo rate of the winner and loser of the match (court/surface specific)
            df_atp.loc[index, f"elo_{col_name}_winner"] = elo_rates.loc[
                elo_rates["Player"] == df_atp.loc[index, "Winner"], f"elo_{col_name}"
            ].values[0]
            df_atp.loc[index, f"elo_{col_name}_loser"] = elo_rates.loc[
                elo_rates["Player"] == df_atp.loc[index, "Loser"], f"elo_{col_name}"
            ].values[0]

            # Set the probability of the outcome based on the elo rates (court/surface specific)
            df_atp.loc[index, f"proba_elo_{col_name}"] = 1.0 / (
                1.0
                + 10.0
                ** (
                    0.0025
                    * (
                        df_atp.loc[index, f"elo_{col_name}_loser"]
                        - df_atp.loc[index, f"elo_{col_name}_winner"]
                    )
                )
            )

        # Update match counts based on the outcome of the match (court/surface specific)
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Winner"],
            f"match_count_{field_type}",
        ] += 1
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Loser"],
            f"match_count_{field_type}",
        ] += 1

        # Update elo rates based on the outcome of the match (court/surface specific)
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Winner"], f"elo_{field_type}"
        ] += k_factor * (1.0 - df_atp.loc[index, f"proba_elo_{field_type}"])
        elo_rates.loc[
            elo_rates["Player"] == df_atp.loc[index, "Loser"], f"elo_{field_type}"
        ] -= k_factor * (1.0 - df_atp.loc[index, f"proba_elo_{field_type}"])

    # Drop the rows in which the point of the players are absent
    df_atp.dropna(subset=["WPts", "LPts"], axis=0, inplace=True)

    # Save the new dataset (enriched dataset)
    df_atp.to_csv("archive/atp_data_enriched.csv")
    elo_rates.to_csv("archive/elo_rates_enriched.csv")

    return elo_rates


def load():
    return pd.read_csv("archive/elo_rates_enriched.csv", index_col=0)


def get_feats(
    P1_name: str, P2_name: str, field_type: str, elo_rates: pd.DataFrame
) -> pd.DataFrame:
    X_unscaled = pd.DataFrame(
        {
            "P1_wins_proba_elo": 0.0,
            "P1_match_count": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count"
            ].values[0],
            "P2_match_count": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count"
            ].values[0],
            "P1_pts": elo_rates.loc[elo_rates["Player"] == P1_name, "pts"].values[0],
            "P2_pts": elo_rates.loc[elo_rates["Player"] == P2_name, "pts"].values[0],
            "field_type==indoor_hard": 1.0 if field_type == "indoor_hard" else 0.0,
            "P1_match_count_indoor_hard": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count_indoor_hard"
            ].values[0],
            "P2_match_count_indoor_hard": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count_indoor_carpet"
            ].values[0],
            "P1_wins_proba_elo_indoor_hard": 0.0,
            "field_type==indoor_carpet": 1.0 if field_type == "indoor_carpet" else 0.0,
            "P1_match_count_indoor_carpet": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count_indoor_carpet"
            ].values[0],
            "P2_match_count_indoor_carpet": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count_indoor_carpet"
            ].values[0],
            "P1_wins_proba_elo_indoor_carpet": 0.0,
            "field_type==indoor_clay": 1.0 if field_type == "indoor_clay" else 0.0,
            "P1_match_count_indoor_clay": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count_indoor_clay"
            ].values[0],
            "P2_match_count_indoor_clay": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count_indoor_clay"
            ].values[0],
            "P1_wins_proba_elo_indoor_clay": 0.0,
            "field_type==outdoor_hard": 1.0 if field_type == "outdoor_hard" else 0.0,
            "P1_match_count_outdoor_hard": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count_outdoor_hard"
            ].values[0],
            "P2_match_count_outdoor_hard": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count_outdoor_hard"
            ].values[0],
            "P1_wins_proba_elo_outdoor_hard": 0.0,
            "field_type==outdoor_clay": 1.0 if field_type == "outdoor_clay" else 0.0,
            "P1_match_count_outdoor_clay": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count_outdoor_clay"
            ].values[0],
            "P2_match_count_outdoor_clay": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count_outdoor_clay"
            ].values[0],
            "P1_wins_proba_elo_outdoor_clay": 0.0,
            "field_type==outdoor_grass": 1.0 if field_type == "outdoor_grass" else 0.0,
            "P1_match_count_outdoor_grass": elo_rates.loc[
                elo_rates["Player"] == P1_name, "match_count_outdoor_grass"
            ].values[0],
            "P2_match_count_outdoor_grass": elo_rates.loc[
                elo_rates["Player"] == P2_name, "match_count_outdoor_grass"
            ].values[0],
            "P1_wins_proba_elo_outdoor_grass": 0.0,
        },
        index=[0],
    )

    X_unscaled[
        [
            "P1_wins_proba_elo",
            "P1_wins_proba_elo_indoor_hard",
            "P1_wins_proba_elo_indoor_carpet",
            "P1_wins_proba_elo_indoor_clay",
            "P1_wins_proba_elo_outdoor_hard",
            "P1_wins_proba_elo_outdoor_clay",
            "P1_wins_proba_elo_outdoor_grass",
        ]
    ] = 1.0 / (
        1.0
        + 10.0
        ** (
            0.0025
            * (
                elo_rates.loc[
                    elo_rates["Player"] == P2_name,
                    [
                        "elo",
                        "elo_indoor_hard",
                        "elo_indoor_carpet",
                        "elo_indoor_clay",
                        "elo_outdoor_hard",
                        "elo_outdoor_clay",
                        "elo_outdoor_grass",
                    ],
                ].to_numpy()
                - elo_rates.loc[
                    elo_rates["Player"] == P1_name,
                    [
                        "elo",
                        "elo_indoor_hard",
                        "elo_indoor_carpet",
                        "elo_indoor_clay",
                        "elo_outdoor_hard",
                        "elo_outdoor_clay",
                        "elo_outdoor_grass",
                    ],
                ].to_numpy()
            )
        )
    )

    return scaler.transform(X_unscaled)


def get_advise(input: dict | pd.DataFrame, platform: str = "PS") -> pd.DataFrame:

    if isinstance(input, dict):
        input = pd.DataFrame(input)

    if set(core.config.features_list) - set(input.columns.tolist()):
        raise ValueError(f"wrong input format")

    X = input[core.config.features_list]

    if platform == "PS":
        df_result = pd.DataFrame(
            {
                "predict_P1_wins": model_rfc.predict(X),
                "certainty": np.absolute(2 * model_rfc.predict_proba(X)[:, 1] - 1.0),
            },
            index=X.index,
        )
        df_result["advise"] = df_result["predict_P1_wins"].replace(
            {True: "P1", False: "P2"}
        )
        df_result.loc[df_result["certainty"] < 0.62, "advise"] = np.nan
        return df_result[["advise", "predict_P1_wins", "certainty"]]
    elif platform == "B365":
        df_result = pd.DataFrame(
            {
                "predic_P1_wins": model_rfc.predict(X),
                "certainty": np.absolute(2 * model_rfc.predict_proba(X)[:, 1] - 1.0),
            },
            index=X.index,
        )
        df_result["advise"] = df_result["predict_P1_wins"].replace(
            {True: "P1", False: "P2"}
        )
        df_result.loc[df_result["certainty"] < 0.62, "advise"] = np.nan
        return df_result[["advise", "predict_P1_wins", "certainty"]]
    else:
        raise ValueError(f'Unkown platform "{platform}"')
