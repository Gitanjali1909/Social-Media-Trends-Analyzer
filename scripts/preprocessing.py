import pandas as pd

def load_and_clean_data(path):

    df = pd.read_csv(path)
    df["Post_Date"] = pd.to_datetime(df["Post_Date"], errors="coerce")

    df = df.dropna(subset=["Post_Date"])

    df["Platform"] = df["Platform"].str.strip().str.lower()
    df["Hashtag"] = df["Hashtag"].astype(str).str.strip().str.lower()

    for col in ["Views", "Likes", "Shares", "Comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["Total_Engagement"] = df["Likes"] + df["Shares"] + df["Comments"]
    df["Year"] = df["Post_Date"].dt.year
    df["Month"] = df["Post_Date"].dt.month
    df["Week"] = df["Post_Date"].dt.to_period("W")

    return df
