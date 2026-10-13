import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    POS_MAPPING,
    load_data,
    filter_by_position_and_market,
    calculate_percentile_ranks,
    run_kmeans_clustering
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NFL Veteran Player Market Analysis Tool",
    page_icon="🏈",
    layout="wide"
)

# --- DATA INITIALIZATION ---
df_raw = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🔍 Market Controls")
st.sidebar.markdown("---")

selected_group = st.sidebar.selectbox(
    "Select Position Group",
    options=list(POS_MAPPING.keys()),
    index=0
)

# Initial slice to establish price ranges
raw_group_info = POS_MAPPING[selected_group]
df_group_raw = df_raw[df_raw["pos"].isin(raw_group_info["positions"])]

min_apy, max_apy = int(df_group_raw["average"].min()), int(df_group_raw["average"].max())
selected_apy = st.sidebar.slider(
    "Average Annual Value (APY) Range ($)",
    min_value=min_apy,
    max_value=max_apy,
    value=(min_apy, max_apy),
    step=250000,
    format="$%d"
)

teams = ["All Teams"] + sorted(df_group_raw["team"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Filter by Team", teams)

# Filter dataset via utils function
df_filtered, group_info = filter_by_position_and_market(
    df_raw, selected_group, selected_apy, selected_team
)

# --- HEADER & KEY METRICS ---
st.title("🏈 Veteran NFL Market Analysis Tool")
st.markdown(
    f"Evaluating **{len(df_filtered)}** veteran contracts in group **{selected_group}**. "
    "*(Rookie contracts excluded)*"
)

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
    st.caption("Players above the regression line represent premium contracts; players below offer surplus value.")

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
            st.info("No sufficient data available to construct trendline.")

# ==========================================
# TAB 2: POSITION-SPECIFIC ROSTER TABLE
# ==========================================
with tab2:
    st.subheader(f"Positional Stat & Contract Ledger — {selected_group}")

    base_cols = ["player", "pos", "team", "age_at_signing", "years", "value", "average", "guarantee", "guarantee_pct"]
    display_stat_cols = [c for c in group_info["stat_cols"] if c in df_filtered.columns]

    table_df = df_filtered[base_cols + display_stat_cols].sort_values(by="average", ascending=False)

    rename_dict = {
        "player": "Player", "pos": "Pos", "team": "Team",
        "age_at_signing": "Age", "years": "Yrs", "value": "Total Value ($)",
        "average": "APY ($)", "guarantee": "Guarantee ($)", "guarantee_pct": "Guarantee %"
    }
    table_df_renamed = table_df.rename(columns=rename_dict)

    fmt_dict = {
        "Total Value ($)": "${:,.2f}",
        "APY ($)": "${:,.2f}",
        "Guarantee ($)": "${:,.2f}",
        "Guarantee %": "{:.2f}%",
        "Age": "{:.2f}"
    }
    for col in display_stat_cols:
        fmt_dict[col] = "{:.2f}"

    st.dataframe(table_df_renamed.style.format(fmt_dict, na_rep="-"), use_container_width=True, height=500)

# ==========================================
# TAB 3: PLAYER DEEP DIVE & PEER COMPARISON
# ==========================================
with tab3:
    st.subheader("Veteran Player Peer Benchmark & Multi-Metric Analysis")
    primary_m = group_info["primary_metric"]
    primary_m_label = primary_m.replace("_", " ").title()

    st.markdown("#### Filter Players by Performance Threshold")
    if primary_m in df_filtered.columns and not df_filtered.empty:
        s_min, s_max = float(df_filtered[primary_m].min()), float(df_filtered[primary_m].max())
        s_filter = st.slider(
            f"Filter {selected_group} by Minimum {primary_m_label}:",
            min_value=round(s_min, 2), max_value=round(s_max, 2), value=round(s_min, 2),
            step=5.0 if s_max > 100 else 0.5
        )
        df_comp_active = df_filtered[df_filtered[primary_m] >= s_filter].copy()
    else:
        df_comp_active = df_filtered.copy()

    st.markdown("---")
    if not df_comp_active.empty:
        selected_player = st.selectbox("Select Primary Veteran Player:",
                                       options=sorted(df_comp_active["player"].unique()))
        p_data = df_comp_active[df_comp_active["player"] == selected_player].iloc[0]

        # Responsive CSS layout for long player names
        st.markdown("""
        <style>
        .player-card { background-color: #1e222a; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #1f77b4; margin-bottom: 10px; }
        .card-label { font-size: 13px; color: #9aa0a6; margin-bottom: 2px; }
        .card-value-large { font-size: 20px; font-weight: 700; color: #ffffff; white-space: normal; word-wrap: break-word; }
        .card-value { font-size: 18px; font-weight: 600; color: #e8eaed; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns([1.5, 1.2, 1.3, 1.0, 1.2])
        with c1:
            st.markdown(
                f'<div class="player-card"><div class="card-label">Selected Player</div><div class="card-value-large">{p_data["player"]}</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="player-card"><div class="card-label">Contract APY</div><div class="card-value">${p_data["average"]:,.2f}</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="player-card"><div class="card-label">Guaranteed Money</div><div class="card-value">${p_data["guarantee"]:,.2f} ({p_data["guarantee_pct"]:.2f}%)</div></div>',
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="player-card"><div class="card-label">Contract Term</div><div class="card-value">{p_data["years"]:.0f} Years</div></div>',
                unsafe_allow_html=True)
        with c5:
            cost_fpts = f"${p_data['cost_per_fpts']:,.2f}/pt" if pd.notna(p_data["cost_per_fpts"]) else "N/A"
            st.markdown(
                f'<div class="player-card"><div class="card-label">Cost per PPR Point</div><div class="card-value">{cost_fpts}</div></div>',
                unsafe_allow_html=True)

        peer_opts = [p for p in df_comp_active["player"].unique() if p != selected_player]
        selected_peers = st.multiselect("Select Positional Peers to Compare:", options=peer_opts,
                                        default=peer_opts[:2] if len(peer_opts) >= 2 else peer_opts)

        all_stats = [c for c in group_info["stat_cols"] if c in df_comp_active.columns]
        selected_stats = st.multiselect("Select Performance Metrics:", options=all_stats,
                                        default=all_stats[:3] if len(all_stats) >= 3 else all_stats)

        if selected_peers:
            comp_players = [selected_player] + selected_peers
            comp_df = df_comp_active[df_comp_active["player"].isin(comp_players)].copy()

            # Dual Axis Chart
            fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bar.add_trace(
                go.Bar(x=comp_df["player"], y=comp_df["average"], name="APY ($)", marker_color="rgb(26, 118, 255)"),
                secondary_y=False)
            fig_bar.add_trace(go.Bar(x=comp_df["player"], y=comp_df[primary_m], name=primary_m_label,
                                     marker_color="rgb(219, 64, 82)"), secondary_y=True)
            fig_bar.update_layout(title=f"APY vs. {primary_m_label} Comparison", barmode="group", height=450,
                                  legend=dict(x=1.05, y=1))
            fig_bar.update_yaxes(title_text="Average Annual Value ($)", secondary_y=False)
            fig_bar.update_yaxes(title_text=primary_m_label, secondary_y=True)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Comparison Table
            comp_table_cols = ["player", "pos", "team", "age_at_signing", "years", "value", "average", "guarantee",
                               "guarantee_pct"] + selected_stats
            comp_tbl = comp_df[comp_table_cols].rename(columns={
                "player": "Player", "pos": "Pos", "team": "Team", "age_at_signing": "Age",
                "years": "Yrs", "value": "Total Value", "average": "APY", "guarantee": "Guarantee",
                "guarantee_pct": "Guarantee %"
            })
            tbl_fmt = {"Total Value": "${:,.2f}", "APY": "${:,.2f}", "Guarantee": "${:,.2f}", "Guarantee %": "{:.2f}%",
                       "Age": "{:.2f}"}
            for s in selected_stats: tbl_fmt[s] = "{:.2f}"
            st.dataframe(comp_tbl.style.format(tbl_fmt, na_rep="-"), use_container_width=True)

            # Percentile Radar Chart
            st.markdown("##### Relative Profile Percentile Radar Chart")
            radar_m = ["average", "guarantee_pct"] + selected_stats[:3]
            radar_lbls = ["APY ($)", "Guarantee %"] + [m.replace("_", " ").title() for m in selected_stats[:3]]
            pct_df = calculate_percentile_ranks(df_comp_active, radar_m)

            fig_radar = go.Figure()
            for p in comp_players:
                p_pct = pct_df[pct_df["player"] == p]
                if not p_pct.empty:
                    r_vals = p_pct[radar_m].values[0].tolist()
                    r_vals.append(r_vals[0])
                    fig_radar.add_trace(
                        go.Scatterpolar(r=r_vals, theta=radar_lbls + [radar_lbls[0]], fill='toself', name=p,
                                        opacity=0.6))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True,
                                    height=500)
            st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# TAB 4: MACHINE LEARNING MARKET SEGMENTATION
# ==========================================
with tab4:
    st.subheader(f"🤖 k-Means Market Segmentation ({selected_group})")
    primary_m = group_info["primary_metric"]
    primary_m_label = primary_m.replace("_", " ").title()

    if primary_m in df_filtered.columns and not df_filtered.empty:
        s_min_ml, s_max_ml = float(df_filtered[primary_m].min()), float(df_filtered[primary_m].max())
        s_filter_ml = st.slider(
            f"Set Production Floor for Clustering ({primary_m_label}):",
            min_value=round(s_min_ml, 2), max_value=round(s_max_ml, 2), value=round(s_min_ml, 2),
            step=5.0 if s_max_ml > 100 else 0.5, key="ml_stat_slider"
        )
        df_ml_active = df_filtered[df_filtered[primary_m] >= s_filter_ml].copy()
    else:
        df_ml_active = df_filtered.copy()

    st.markdown("---")
    cluster_feat = [f for f in group_info["cluster_features"] if f in df_ml_active.columns]

    if len(df_ml_active) >= 3 and len(cluster_feat) >= 2:
        df_clustered, _ = run_kmeans_clustering(df_ml_active, cluster_feat, n_clusters=3)

        st.markdown("#### 1. Market Tier Aggregate Profiles")
        tier_agg = df_clustered.groupby("market_tier").agg(
            Count=("player", "count"),
            Mean_APY=("average", "mean"),
            Mean_Guarantee=("guarantee", "mean"),
            Mean_Guarantee_Pct=("guarantee_pct", "mean")
        ).reset_index()

        tier_fmt = {"Mean_APY": "${:,.2f}", "Mean_Guarantee": "${:,.2f}", "Mean_Guarantee_Pct": "{:.2f}%"}
        for feat in cluster_feat[2:]:
            feat_m = df_clustered.groupby("market_tier")[feat].mean().reset_index()
            feat_m.columns = ["market_tier", f"Mean_{feat}"]
            tier_agg = tier_agg.merge(feat_m, on="market_tier")
            tier_fmt[f"Mean_{feat}"] = "{:.2f}"

        st.dataframe(tier_agg.style.format(tier_fmt, na_rep="-"), use_container_width=True)

        st.markdown("#### 2. Tier Cluster Visual Mapping")
        fig_cluster = px.scatter(
            df_clustered, x=primary_m, y="average", color="market_tier", symbol="market_tier",
            hover_name="player", hover_data=["team", "age_at_signing", "guarantee_pct"],
            title=f"Market Segmentation: APY vs. {primary_m_label}",
            labels={primary_m: primary_m_label, "average": "Average Annual Value (APY) ($)",
                    "market_tier": "Market Tier"},
            color_discrete_map={"Tier 1": "#1f77b4", "Tier 2": "#ff7f0e", "Tier 3": "#2ca02c"},
            height=520
        )
        fig_cluster.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("#### 3. Strategic Market Insights")
        t1, t2, t3 = [df_clustered[df_clustered["market_tier"] == f"Tier {i}"] for i in range(1, 4)]
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.info(
                f"**Tier 1: Premium / Franchise ({len(t1)} Players)**\n\n- **Avg APY**: ${t1['average'].mean():,.2f}\n- **Avg Guarantee %**: {t1['guarantee_pct'].mean():.2f}%")
        with col_t2:
            st.warning(
                f"**Tier 2: Mid-Tier / Rotation ({len(t2)} Players)**\n\n- **Avg APY**: ${t2['average'].mean():,.2f}\n- **Avg Guarantee %**: {t2['guarantee_pct'].mean():.2f}%")
        with col_t3:
            st.success(
                f"**Tier 3: Value / Depth ({len(t3)} Players)**\n\n- **Avg APY**: ${t3['average'].mean():,.2f}\n- **Avg Guarantee %**: {t3['guarantee_pct'].mean():.2f}%")
    else:
        st.warning("Insufficient sample size in current filter selection to compute k-Means clustering.")