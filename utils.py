import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- POSITION GROUP STAT MAPPING & FEATURE CONFIG ---
POS_MAPPING = {
    "Quarterbacks (QB)": {
        "positions": ["QB"],
        "primary_metric": "passing_yards",
        "stat_cols": [
            "completions", "attempts", "passing_yards", "passing_tds",
            "passing_interceptions", "passing_air_yards", "passing_epa",
            "passing_cpoe", "sacks_suffered", "rushing_yards", "rushing_tds", "fantasy_points_ppr"
        ],
        "cluster_features": ["average", "guarantee", "passing_yards", "passing_tds", "fantasy_points_ppr"]
    },
    "Running Backs (RB/FB)": {
        "positions": ["RB", "FB"],
        "primary_metric": "rushing_yards",
        "stat_cols": [
            "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
            "rushing_epa", "targets", "receptions", "receiving_yards",
            "receiving_tds", "fumbles_total", "fantasy_points_ppr"
        ],
        "cluster_features": ["average", "guarantee", "rushing_yards", "rushing_tds", "receptions"]
    },
    "Receivers & Tight Ends (WR/TE)": {
        "positions": ["WR", "TE"],
        "primary_metric": "receiving_yards",
        "stat_cols": [
            "targets", "receptions", "receiving_yards", "receiving_tds",
            "receiving_air_yards", "receiving_yards_after_catch", "target_share",
            "wopr", "receiving_epa", "fantasy_points_ppr"
        ],
        "cluster_features": ["average", "guarantee", "receiving_yards", "receiving_tds", "targets"]
    },
    "Offensive Line (G/C/T)": {
        "positions": ["G", "C", "T", "LT", "RT"],
        "primary_metric": "penalties",
        "stat_cols": ["penalties", "penalty_yards", "fumble_recovery_own"],
        "cluster_features": ["average", "guarantee", "penalties", "penalty_yards"]
    },
    "Defense (LB/CB/S/DL/ED)": {
        "positions": ["LB", "CB", "S", "FS", "SS", "DL", "ED"],
        "primary_metric": "def_tackles_total",
        "stat_cols": [
            "def_tackles_solo", "def_tackles_total", "def_tackles_for_loss",
            "def_sacks", "def_qb_hits", "def_interceptions", "def_pass_defended",
            "def_fumbles_forced", "def_tds"
        ],
        "cluster_features": ["average", "guarantee", "def_tackles_total", "def_sacks", "def_interceptions"]
    },
    "Kickers (K)": {
        "positions": ["K"],
        "primary_metric": "fg_made",
        "stat_cols": [
            "fg_made", "fg_att", "fg_pct", "fg_long", "fg_made_50_59",
            "pat_made", "pat_pct", "gwfg_made"
        ],
        "cluster_features": ["average", "guarantee", "fg_made", "fg_pct"]
    },
    "Punters (P)": {
        "positions": ["P"],
        "primary_metric": "pt_yards",
        "stat_cols": [
            "pt_att", "pt_yards", "pt_long", "pt_inside_20",
            "pt_touchback", "pt_net_yards"
        ],
        "cluster_features": ["average", "guarantee", "pt_yards", "pt_inside_20"]
    }
}


def load_data(filepath="player_contracts.csv"):
    """Loads and preprocesses player contract and production data."""
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    meta_cols = [
        "player", "pos", "team", "age_at_signing",
        "start", "end", "years", "value", "average", "guarantee", "practical"
    ]
    stat_cols = [c for c in df.columns if c not in meta_cols]

    # Exclude players without statistical records (e.g. rookie contract players)
    df_vets = df[df[stat_cols].notna().any(axis=1)].copy()

    # Fill contract NaNs safely
    df_vets["guarantee"] = df_vets["guarantee"].fillna(0)
    df_vets["practical"] = df_vets["practical"].fillna(0)
    df_vets["value"] = df_vets["value"].fillna(0)
    df_vets["average"] = df_vets["average"].fillna(0)
    df_vets["years"] = df_vets["years"].fillna(1)

    # Round all float columns to 2 decimal places
    float_cols = df_vets.select_dtypes(include=['float64', 'float32']).columns
    df_vets[float_cols] = df_vets[float_cols].round(2)

    # Calculate derived contract & performance metrics
    df_vets["guarantee_pct"] = np.where(
        df_vets["value"] > 0,
        (df_vets["guarantee"] / df_vets["value"] * 100).round(2),
        0.0
    )
    df_vets["def_tackles_total"] = (
            df_vets["def_tackles_solo"].fillna(0) + df_vets["def_tackle_assists"].fillna(0)
    ).round(2)

    # Cost Efficiency Ratio: APY per Fantasy Point
    df_vets["cost_per_fpts"] = np.where(
        df_vets["fantasy_points_ppr"] > 0,
        (df_vets["average"] / df_vets["fantasy_points_ppr"]).round(2),
        np.nan
    )

    return df_vets


def filter_by_position_and_market(df, selected_group, selected_apy_range, selected_team="All Teams"):
    """Filters dataset by position group, APY range, and team."""
    group_info = POS_MAPPING[selected_group]
    allowed_pos = group_info["positions"]

    df_sub = df[df["pos"].isin(allowed_pos)].copy()

    df_sub = df_sub[
        (df_sub["average"] >= selected_apy_range[0]) &
        (df_sub["average"] <= selected_apy_range[1])
        ].copy()

    if selected_team != "All Teams":
        df_sub = df_sub[df_sub["team"] == selected_team].copy()

    df_sub["guarantee_pct"] = np.where(
        df_sub["value"] > 0,
        (df_sub["guarantee"] / df_sub["value"] * 100).round(2),
        0.0
    )

    return df_sub, group_info


def calculate_percentile_ranks(df, metrics):
    """Calculates percentile ranks (0-100%) across specified metrics."""
    pct_df = df[["player"] + metrics].copy()
    for m in metrics:
        pct_df[m] = pct_df[m].rank(pct=True) * 100
    return pct_df


def run_kmeans_clustering(df, cluster_features, n_clusters=3):
    """Executes StandardScaler + KMeans clustering on feature matrix."""
    if len(df) < n_clusters or len(cluster_features) < 2:
        return df, None

    df_cl = df.copy()
    X = df_cl[cluster_features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cl["cluster"] = kmeans.fit_predict(X_scaled)

    # Sort tier naming by average APY (Tier 1 = Highest APY)
    cluster_apy_order = df_cl.groupby("cluster")["average"].mean().sort_values(ascending=False).index
    cluster_rename = {old: f"Tier {i + 1}" for i, old in enumerate(cluster_apy_order)}
    df_cl["market_tier"] = df_cl["cluster"].map(cluster_rename)

    return df_cl, cluster_rename