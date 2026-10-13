import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NFL Veteran Player Market Analysis & ML Clustering Tool",
    page_icon="🏈",
    layout="wide"
)


# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_data():
    df = pd.read_csv("player_contracts.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Metadata columns
    meta_cols = [
        "player", "pos", "team", "age_at_signing",
        "start", "end", "years", "value", "average", "guarantee", "practical"
    ]
    stat_cols = [c for c in df.columns if c not in meta_cols]

    # Filter out players without stats (rookie contracts / unpopulated stats)
    df_vets = df[df[stat_cols].notna().any(axis=1)].copy()

    # Fill contract NaNs safely
    df_vets["guarantee"] = df_vets["guarantee"].fillna(0)
    df_vets["practical"] = df_vets["practical"].fillna(0)
    df_vets["value"] = df_vets["value"].fillna(0)
    df_vets["average"] = df_vets["average"].fillna(0)
    df_vets["years"] = df_vets["years"].fillna(1)

    # Round all float columns to 2 decimal places max
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


df_raw = load_data()

# --- POSITION GROUP STAT MAPPING ---
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

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔍 Market Controls")
st.sidebar.markdown("---")

selected_group = st.sidebar.selectbox(
    "Select Position Group",
    options=list(POS_MAPPING.keys()),
    index=0
)

group_info = POS_MAPPING[selected_group]
allowed_pos = group_info["positions"]

# Filter dataset by position group
df_filtered = df_raw[df_raw["pos"].isin(allowed_pos)].copy()

# Contract value sliders
min_apy, max_apy = int(df_filtered["average"].min()), int(df_filtered["average"].max())
selected_apy = st.sidebar.slider(
    "Average Annual Value (APY) Range ($)",
    min_value=min_apy,
    max_value=max_apy,
    value=(min_apy, max_apy),
    step=250000,
    format="$%d"
)

teams = ["All Teams"] + sorted(df_filtered["team"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Filter by Team", teams)

# Apply global filters
df_filtered = df_filtered[
    (df_filtered["average"] >= selected_apy[0]) &
    (df_filtered["average"] <= selected_apy[1])
    ].copy()

if selected_team != "All Teams":
    df_filtered = df_filtered[df_filtered["team"] == selected_team].copy()

# Ensure guarantee_pct exists and is rounded
df_filtered["guarantee_pct"] = np.where(
    df_filtered["value"] > 0,
    (df_filtered["guarantee"] / df_filtered["value"] * 100).round(2),
    0.0
)

# --- MAIN DASHBOARD HEADER ---
st.title("🏈 Veteran NFL Market Analysis Tool")
st.markdown(
    f"Evaluating **{len(df_filtered)}** veteran contracts in group **{selected_group}**. "
    "*(Rookie contracts without statistical records removed)*"
)

# --- TOP SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Veteran Contracts", len(df_filtered))
col2.metric("Mean APY", f"${df_filtered['average'].mean():,.2f}" if not df_filtered.empty else "$0.00")
col3.metric("Median Guarantee %", f"{df_filtered['guarantee_pct'].median():.2f}%" if not df_filtered.empty else "0.00%")
col4.metric("Avg Age at Signing",
            f"{df_filtered['age_at_signing'].mean():.2f} yrs" if not df_filtered.empty else "0.00 yrs")

st.markdown("---")

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Regression & Valuation",
    "📋 Positional Contract & Stat Roster",
    "👤 Player Comparison & Peer Analysis",
    "🤖 ML Market Segmentation (Clustering)"
])

# ==========================================
# TAB 1: MARKET REGRESSION & VALUATION
# ==========================================
with tab1:
    st.subheader("Market Trendline: Performance vs. Contract Valuation")
    st.caption(
        "Players above the linear regression line represent premium contracts relative to production. "
        "Players below the trendline offer surplus value and high ROI."
    )

    stat_options = [c for c in group_info["stat_cols"] if c in df_filtered.columns]

    col_x, col_y = st.columns([2, 1])
    with col_y:
        selected_x_stat = st.selectbox(
            "Select Metric for Valuation Axis (X)",
            options=stat_options,
            index=stat_options.index(group_info["primary_metric"]) if group_info[
                                                                          "primary_metric"] in stat_options else 0
        )

        contract_metric = st.selectbox(
            "Select Contract Metric Axis (Y)",
            options=["average", "value", "guarantee"],
            format_func=lambda x: {
                "average": "Average Annual Value (APY)",
                "value": "Total Contract Value",
                "guarantee": "Total Guaranteed Money"
            }[x]
        )

    with col_x:
        valid_plot_df = df_filtered.dropna(subset=[selected_x_stat, contract_metric])
        if not valid_plot_df.empty:
            fig_scatter = px.scatter(
                valid_plot_df,
                x=selected_x_stat,
                y=contract_metric,
                color="pos",
                hover_name="player",
                hover_data=["team", "age_at_signing", "years", "guarantee_pct"],
                trendline="ols",
                trendline_color_override="darkred",
                title=f"{contract_metric.title()} vs. {selected_x_stat.replace('_', ' ').title()}",
                labels={
                    selected_x_stat: selected_x_stat.replace("_", " ").title(),
                    contract_metric: "Contract " + contract_metric.title() + " ($)"
                },
                height=520
            )
            fig_scatter.update_traces(marker=dict(size=11, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No sufficient data available to construct trendline for this metric selection.")

# ==========================================
# TAB 2: POSITION-SPECIFIC ROSTER TABLE
# ==========================================
with tab2:
    st.subheader(f"Positional Stat & Contract Ledger — {selected_group}")

    base_contract_cols = [
        "player", "pos", "team", "age_at_signing",
        "years", "value", "average", "guarantee", "guarantee_pct"
    ]
    display_stat_cols = [c for c in group_info["stat_cols"] if c in df_filtered.columns]

    final_cols = base_contract_cols + display_stat_cols

    table_df = df_filtered[final_cols].sort_values(by="average", ascending=False)

    rename_dict = {
        "player": "Player", "pos": "Pos", "team": "Team",
        "age_at_signing": "Age", "years": "Yrs", "value": "Total Value ($)",
        "average": "APY ($)", "guarantee": "Guarantee ($)", "guarantee_pct": "Guarantee %"
    }
    table_df_renamed = table_df.rename(columns=rename_dict)

    format_dict = {
        "Total Value ($)": "${:,.2f}",
        "APY ($)": "${:,.2f}",
        "Guarantee ($)": "${:,.2f}",
        "Guarantee %": "{:.2f}%",
        "Age": "{:.2f}"
    }
    for col in display_stat_cols:
        format_dict[col] = "{:.2f}"

    st.dataframe(
        table_df_renamed.style.format(format_dict, na_rep="-"),
        use_container_width=True,
        height=500
    )

# ==========================================
# TAB 3: PLAYER DEEP DIVE & PEER COMPARISON
# ==========================================
with tab3:
    st.subheader("Veteran Player Peer Benchmark & Multi-Metric Analysis")

    primary_m = group_info["primary_metric"]
    primary_m_label = primary_m.replace("_", " ").title()

    # Dynamic Stat Filter for Comparison Tab
    st.markdown("#### Filter Players by Performance Threshold")
    if primary_m in df_filtered.columns and not df_filtered.empty:
        stat_min_val = float(df_filtered[primary_m].min())
        stat_max_val = float(df_filtered[primary_m].max())

        stat_filter_comp = st.slider(
            f"Filter {selected_group} by Minimum {primary_m_label}:",
            min_value=round(stat_min_val, 2),
            max_value=round(stat_max_val, 2),
            value=round(stat_min_val, 2),
            step=5.0 if stat_max_val > 100 else 0.5
        )
        df_comp_active = df_filtered[df_filtered[primary_m] >= stat_filter_comp].copy()
    else:
        df_comp_active = df_filtered.copy()

    st.markdown("---")

    if not df_comp_active.empty:
        selected_player = st.selectbox(
            "Select Primary Veteran Player:",
            options=sorted(df_comp_active["player"].unique())
        )

        p_data = df_comp_active[df_comp_active["player"] == selected_player].iloc[0]

        # --- CUSTOM HTML CARDS TO PREVENT TEXT TRUNCATION ---
        st.markdown("""
        <style>
        .player-card {
            background-color: #1e222a;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid #1f77b4;
            margin-bottom: 10px;
        }
        .card-label {
            font-size: 13px;
            color: #9aa0a6;
            margin-bottom: 2px;
        }
        .card-value-large {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            white-space: normal;
            word-wrap: break-word;
        }
        .card-value {
            font-size: 18px;
            font-weight: 600;
            color: #e8eaed;
        }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns([1.5, 1.2, 1.3, 1.0, 1.2])

        with c1:
            st.markdown(f"""
            <div class="player-card">
                <div class="card-label">Selected Veteran Player</div>
                <div class="card-value-large">{p_data['player']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="player-card">
                <div class="card-label">Contract APY</div>
                <div class="card-value">${p_data['average']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="player-card">
                <div class="card-label">Guaranteed Money</div>
                <div class="card-value">${p_data['guarantee']:,.2f} ({p_data['guarantee_pct']:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="player-card">
                <div class="card-label">Contract Term</div>
                <div class="card-value">{p_data['years']:.0f} Years</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            cost_fpts = f"${p_data['cost_per_fpts']:,.2f}/pt" if pd.notna(p_data["cost_per_fpts"]) else "N/A"
            st.markdown(f"""
            <div class="player-card">
                <div class="card-label">Cost per PPR Point</div>
                <div class="card-value">{cost_fpts}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Peer Selection & Metric Customization")

        peer_options = [p for p in df_comp_active["player"].unique() if p != selected_player]
        selected_peers = st.multiselect(
            "Select Positional Peers to Compare:",
            options=peer_options,
            default=peer_options[:2] if len(peer_options) >= 2 else peer_options
        )

        all_avail_stats = [c for c in group_info["stat_cols"] if c in df_comp_active.columns]
        selected_comp_stats = st.multiselect(
            "Select Specific Performance Metrics for Comparison:",
            options=all_avail_stats,
            default=all_avail_stats[:3] if len(all_avail_stats) >= 3 else all_avail_stats
        )

        if selected_peers:
            comp_players = [selected_player] + selected_peers
            comp_df = df_comp_active[df_comp_active["player"].isin(comp_players)].copy()

            st.markdown("##### 1. Dual-Axis Bar Comparison (APY vs Primary Production Metric)")

            fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

            fig_bar.add_trace(
                go.Bar(
                    x=comp_df["player"],
                    y=comp_df["average"],
                    name="APY ($)",
                    marker_color="rgb(26, 118, 255)"
                ),
                secondary_y=False
            )

            fig_bar.add_trace(
                go.Bar(
                    x=comp_df["player"],
                    y=comp_df[primary_m],
                    name=primary_m_label,
                    marker_color="rgb(219, 64, 82)"
                ),
                secondary_y=True
            )

            fig_bar.update_layout(
                title=f"APY vs. {primary_m_label} Comparison",
                barmode="group",
                height=450,
                legend=dict(x=1.05, y=1)
            )
            fig_bar.update_yaxes(title_text="Average Annual Value ($)", secondary_y=False)
            fig_bar.update_yaxes(title_text=primary_m_label, secondary_y=True)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("##### 2. Comprehensive Contract & Statistical Breakdown")
            comp_table_cols = ["player", "pos", "team", "age_at_signing", "years", "value", "average", "guarantee",
                               "guarantee_pct"] + selected_comp_stats
            comp_table = comp_df[comp_table_cols].rename(columns={
                "player": "Player", "pos": "Pos", "team": "Team", "age_at_signing": "Age",
                "years": "Yrs", "value": "Total Value", "average": "APY", "guarantee": "Guarantee",
                "guarantee_pct": "Guarantee %"
            })

            comp_format_dict = {
                "Total Value": "${:,.2f}",
                "APY": "${:,.2f}",
                "Guarantee": "${:,.2f}",
                "Guarantee %": "{:.2f}%",
                "Age": "{:.2f}"
            }
            for stat in selected_comp_stats:
                comp_format_dict[stat] = "{:.2f}"

            st.dataframe(
                comp_table.style.format(comp_format_dict, na_rep="-"),
                use_container_width=True
            )

            st.markdown("##### 3. Relative Profile Percentile Radar Chart")
            st.caption("Normalizes metrics across the filtered position group into percentile ranks (0–100%).")

            radar_metrics = ["average", "guarantee_pct"] + selected_comp_stats[:3]
            radar_labels = ["APY ($)", "Guarantee %"] + [m.replace("_", " ").title() for m in selected_comp_stats[:3]]

            pct_df = df_comp_active[["player"] + radar_metrics].copy()
            for m in radar_metrics:
                pct_df[m] = pct_df[m].rank(pct=True) * 100

            fig_radar = go.Figure()
            for p in comp_players:
                p_pct = pct_df[pct_df["player"] == p]
                if not p_pct.empty:
                    r_values = p_pct[radar_metrics].values[0].tolist()
                    r_values.append(r_values[0])
                    thetas = radar_labels + [radar_labels[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_values,
                        theta=thetas,
                        fill='toself',
                        name=p,
                        opacity=0.6
                    ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# TAB 4: MACHINE LEARNING MARKET SEGMENTATION
# ==========================================
with tab4:
    st.subheader(f"🤖 k-Means Market Segmentation ({selected_group})")
    st.markdown(
        "Using **scikit-learn `KMeans`** (k=3), players in this position group are algorithmically clustered into 3 distinct market tiers "
        "based on both contract commitments and on-field statistical performance."
    )

    primary_m = group_info["primary_metric"]
    primary_m_label = primary_m.replace("_", " ").title()

    # Stat Filter on ML Tab
    st.markdown("#### Filter Clustering Dataset by Production Floor")
    if primary_m in df_filtered.columns and not df_filtered.empty:
        stat_min_val_ml = float(df_filtered[primary_m].min())
        stat_max_val_ml = float(df_filtered[primary_m].max())

        stat_filter_ml = st.slider(
            f"Set Production Floor for Clustering ({primary_m_label}):",
            min_value=round(stat_min_val_ml, 2),
            max_value=round(stat_max_val_ml, 2),
            value=round(stat_min_val_ml, 2),
            step=5.0 if stat_max_val_ml > 100 else 0.5,
            key="ml_stat_slider"
        )
        df_ml_active = df_filtered[df_filtered[primary_m] >= stat_filter_ml].copy()
    else:
        df_ml_active = df_filtered.copy()

    st.markdown("---")

    cluster_feat = [f for f in group_info["cluster_features"] if f in df_ml_active.columns]

    if len(df_ml_active) >= 3 and len(cluster_feat) >= 2:
        X = df_ml_active[cluster_feat].fillna(0).copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_ml_active["cluster"] = kmeans.fit_predict(X_scaled)

        cluster_apy_order = df_ml_active.groupby("cluster")["average"].mean().sort_values(ascending=False).index
        cluster_rename = {old: f"Tier {i + 1}" for i, old in enumerate(cluster_apy_order)}
        df_ml_active["market_tier"] = df_ml_active["cluster"].map(cluster_rename)

        # --- 1. CLUSTER OVERVIEW TABLE ---
        st.markdown("#### 1. Market Tier Aggregate Profiles")

        tier_agg = df_ml_active.groupby("market_tier").agg(
            Count=("player", "count"),
            Mean_APY=("average", "mean"),
            Mean_Guarantee=("guarantee", "mean"),
            Mean_Guarantee_Pct=("guarantee_pct", "mean")
        ).reset_index()

        format_tier_dict = {
            "Mean_APY": "${:,.2f}",
            "Mean_Guarantee": "${:,.2f}",
            "Mean_Guarantee_Pct": "{:.2f}%"
        }

        for feat in cluster_feat[2:]:
            feat_means = df_ml_active.groupby("market_tier")[feat].mean().reset_index()
            feat_means.columns = ["market_tier", f"Mean_{feat}"]
            tier_agg = tier_agg.merge(feat_means, on="market_tier")
            format_tier_dict[f"Mean_{feat}"] = "{:.2f}"

        st.dataframe(
            tier_agg.style.format(format_tier_dict, na_rep="-"),
            use_container_width=True
        )

        # --- 2. CLUSTER VISUAL SCATTER PLOT ---
        st.markdown("#### 2. Tier Cluster Visual Mapping")

        fig_cluster = px.scatter(
            df_ml_active,
            x=primary_m,
            y="average",
            color="market_tier",
            symbol="market_tier",
            hover_name="player",
            hover_data=["team", "age_at_signing", "guarantee_pct"],
            title=f"Market Segmentation: APY vs. {primary_m_label}",
            labels={
                primary_m: primary_m_label,
                "average": "Average Annual Value (APY) ($)",
                "market_tier": "Market Tier"
            },
            color_discrete_map={
                "Tier 1": "#1f77b4",
                "Tier 2": "#ff7f0e",
                "Tier 3": "#2ca02c"
            },
            height=520
        )
        fig_cluster.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig_cluster, use_container_width=True)

        # --- 3. TIER CARDS BREAKDOWN ---
        st.markdown("#### 3. Strategic Market Insights")
        t1 = df_ml_active[df_ml_active["market_tier"] == "Tier 1"]
        t2 = df_ml_active[df_ml_active["market_tier"] == "Tier 2"]
        t3 = df_ml_active[df_ml_active["market_tier"] == "Tier 3"]

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.info(
                f"**Tier 1: Premium / Franchise Market ({len(t1)} Players)**\n\n"
                f"- **Avg APY**: ${t1['average'].mean():,.2f}\n"
                f"- **Avg Guarantee %**: {t1['guarantee_pct'].mean():.2f}%\n"
                f"- **Key Characteristic**: Top-tier volume production paired with maximum financial security."
            )
        with col_t2:
            st.warning(
                f"**Tier 2: Mid-Tier / Rotation Veterans ({len(t2)} Players)**\n\n"
                f"- **Avg APY**: ${t2['average'].mean():,.2f}\n"
                f"- **Avg Guarantee %**: {t2['guarantee_pct'].mean():.2f}%\n"
                f"- **Key Characteristic**: Solid starter or complementary output with moderate market commitments."
            )
        with col_t3:
            st.success(
                f"**Tier 3: Value / Depth Contracts ({len(t3)} Players)**\n\n"
                f"- **Avg APY**: ${t3['average'].mean():,.2f}\n"
                f"- **Avg Guarantee %**: {t3['guarantee_pct'].mean():.2f}%\n"
                f"- **Key Characteristic**: Low risk, budget contracts offering high upside or depth value."
            )
    else:
        st.warning("Insufficient sample size in current filter selection to compute k-Means clustering.")