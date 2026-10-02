import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RailGuard AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

def load_css():

    css_file = Path("assets/style.css")

    if css_file.exists():

        with open(css_file, "r", encoding="utf-8") as file:
            css = file.read()

        st.html(f"<style>{css}</style>")


load_css()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_accidents():

    data = pd.read_csv(
        "data/processed/railguard_accidents.csv"
    )

    data["Consequential Train Accidents"] = pd.to_numeric(
        data["Consequential Train Accidents"],
        errors="coerce"
    )

    return data


@st.cache_data
def load_funding():

    return pd.read_csv(
        "data/raw/station_funds.csv"
    )


@st.cache_resource
def load_anomaly_model():

    return joblib.load(
        "models/isolation_forest.pkl"
    )


@st.cache_resource
def load_forecast_model():

    return joblib.load(
        "models/baseline_forecast.pkl"
    )


accidents = load_accidents()
funding = load_funding()

anomaly_model = load_anomaly_model()
forecast_model = load_forecast_model()


# ============================================================
# COMMON DATA
# ============================================================

completed = accidents[
    accidents["Year"] != "2024-25"
].copy()

completed["Accident_Change"] = (
    completed["Consequential Train Accidents"].diff()
)

completed["Percentage_Change"] = (
    completed["Consequential Train Accidents"].pct_change()
    * 100
)

average_accidents = completed[
    "Consequential Train Accidents"
].mean()

peak_accidents = completed[
    "Consequential Train Accidents"
].max()

lowest_accidents = completed[
    "Consequential Train Accidents"
].min()

peak_year = completed.loc[
    completed["Consequential Train Accidents"].idxmax(),
    "Year"
]

lowest_year = completed.loc[
    completed["Consequential Train Accidents"].idxmin(),
    "Year"
]

forecast_value = float(
    forecast_model["prediction"]
)


# ============================================================
# FUNDING CALCULATIONS
# ============================================================

funding_clean = funding[
    funding["Zonal Railway"].astype(str).str.lower() != "total"
].copy()

allocation_columns = [
    "2020-21 - Allocation",
    "2021-22 - Allocation",
    "2022-23 - Allocation"
]

expenditure_columns = [
    "2020-21 - Expenditure",
    "2021-22 - Expenditure",
    "2022-23 - Expenditure"
]

for column in allocation_columns + expenditure_columns:

    funding_clean[column] = pd.to_numeric(
        funding_clean[column],
        errors="coerce"
    )

total_allocation = funding_clean[
    allocation_columns
].sum().sum()

total_expenditure = funding_clean[
    expenditure_columns
].sum().sum()

funding_utilization = (
    total_expenditure /
    total_allocation *
    100
)


# ============================================================
# ANOMALY DATA
# ============================================================

anomaly_data = completed.copy()

anomaly_data["Accident_Change"] = (
    anomaly_data[
        "Consequential Train Accidents"
    ].diff()
)

anomaly_data["Percentage_Change"] = (
    anomaly_data[
        "Consequential Train Accidents"
    ].pct_change() * 100
)

anomaly_features = anomaly_data.dropna(
    subset=[
        "Accident_Change",
        "Percentage_Change"
    ]
).copy()

feature_columns = [
    "Consequential Train Accidents",
    "Accident_Change",
    "Percentage_Change"
]

anomaly_features["Prediction"] = (
    anomaly_model.predict(
        anomaly_features[
            feature_columns
        ]
    )
)

anomaly_features["Status"] = (
    anomaly_features["Prediction"].map(
        {
            -1: "Anomaly",
            1: "Normal"
        }
    )
)

detected = anomaly_features[
    anomaly_features["Status"] == "Anomaly"
].copy()


# ============================================================
# RISK DATA
# ============================================================

risk_data = completed.copy()


def classify_risk(value):

    if value >= average_accidents * 1.25:
        return "High"

    elif value >= average_accidents * 0.90:
        return "Medium"

    return "Low"


risk_data["Risk_Level"] = (
    risk_data[
        "Consequential Train Accidents"
    ].apply(classify_risk)
)

high_count = (
    risk_data["Risk_Level"] == "High"
).sum()

medium_count = (
    risk_data["Risk_Level"] == "Medium"
).sum()

low_count = (
    risk_data["Risk_Level"] == "Low"
).sum()


# ============================================================
# CHART STYLE
# ============================================================

def style_chart(fig, height=430):

    fig.update_layout(

        template="plotly_dark",

        height=height,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#AEB5C5",
            family="Inter, Arial"
        ),

        margin=dict(
            l=20,
            r=20,
            t=35,
            b=30
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#71798B"
        ),

        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            color="#71798B"
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),

        hoverlabel=dict(
            bgcolor="#11151F",
            font_color="#FFFFFF"
        )
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">R</div>

            <div>
                <div class="sidebar-name">
                    RAILGUARD <span>/ AI</span>
                </div>

                <div class="sidebar-sub">
                    SAFETY INTELLIGENCE
                </div>
            </div>
        </div>
        """
    )

    st.html(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            SYSTEM OPERATIONAL
        </div>
        """
    )

    st.write("")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Anomaly Intelligence",
            "Baseline Forecast",
            "Infrastructure",
            "Risk Intelligence"
        ],
        label_visibility="collapsed"
    )

    st.write("")

    st.html(
        """
        <div class="sidebar-section-label">
            INTELLIGENCE STACK
        </div>

        <div class="sidebar-module">
            <span>01</span>
            Anomaly Detection
        </div>

        <div class="sidebar-module">
            <span>02</span>
            Baseline Forecast
        </div>

        <div class="sidebar-module">
            <span>03</span>
            Infrastructure
        </div>

        <div class="sidebar-module">
            <span>04</span>
            Risk Classification
        </div>

        <div class="sidebar-footer">
            RAILGUARD AI<br>
            v1.0 / HISTORICAL MODE
        </div>
        """
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.html(
        """
        <div class="hero">

            <div class="hero-top">

                <div class="hero-label">
                    RAILWAY SAFETY / INTELLIGENCE CENTER
                </div>

                <div class="hero-live">
                    <span></span>
                    HISTORICAL ANALYSIS
                </div>

            </div>

            <div class="hero-grid">

                <div class="hero-main">

                    <div class="hero-kicker">
                        RAILGUARD <b>/ AI</b>
                    </div>

                    <h1>
                        Railway Safety<br>
                        <span>Intelligence Center</span>
                    </h1>

                    <p>
                        A data-driven platform for historical
                        safety signals, anomaly detection,
                        infrastructure intelligence and
                        transparent risk analysis.
                    </p>

                </div>

                <div class="hero-orbit">

                    <div class="orbit-ring ring-one"></div>
                    <div class="orbit-ring ring-two"></div>

                    <div class="orbit-core">
                        <strong>10</strong>
                        <small>YEARS<br>ANALYZED</small>
                    </div>

                </div>

            </div>

            <div class="hero-bottom">

                <span>GOVERNMENT OPEN DATA</span>
                <span>•</span>
                <span>MACHINE LEARNING</span>
                <span>•</span>
                <span>PYTHON ANALYTICS</span>

            </div>

        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # KPI STRIP
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="kpi-grid">

            <div class="kpi-card">
                <div class="kpi-label">
                    LATEST COMPLETED
                </div>

                <div class="kpi-value">
                    40
                </div>

                <div class="kpi-meta">
                    2023–24
                </div>
            </div>


            <div class="kpi-card accent-purple">
                <div class="kpi-label">
                    BASELINE
                </div>

                <div class="kpi-value">
                    {forecast_value:.1f}
                </div>

                <div class="kpi-meta">
                    3-YEAR HISTORICAL AVG
                </div>
            </div>


            <div class="kpi-card accent-red">
                <div class="kpi-label">
                    ML SIGNALS
                </div>

                <div class="kpi-value">
                    {len(detected):02d}
                </div>

                <div class="kpi-meta">
                    ISOLATION FOREST
                </div>
            </div>


            <div class="kpi-card accent-blue">
                <div class="kpi-label">
                    FUND UTILIZATION
                </div>

                <div class="kpi-value">
                    {funding_utilization:.2f}%
                </div>

                <div class="kpi-meta">
                    2020–23 DATA
                </div>
            </div>

        </div>
        """
    )

    st.write("")

    # --------------------------------------------------------
    # MAIN ANALYTICS GRID
    # --------------------------------------------------------

    left, right = st.columns(
        [2.15, 1],
        gap="large"
    )

    with left:

        st.html(
            """
            <div class="section-heading">
                <span>01</span>
                SAFETY TRAJECTORY
                <em>ACCIDENT HISTORY</em>
            </div>
            """
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=accidents["Year"],
                y=accidents[
                    "Consequential Train Accidents"
                ],
                mode="lines+markers",
                line=dict(
                    color="#8B5CF6",
                    width=4
                ),
                marker=dict(
                    size=8,
                    color="#A78BFA",
                    line=dict(
                        color="#08090D",
                        width=2
                    )
                ),
                fill="tozeroy",
                fillcolor="rgba(139,92,246,0.07)",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Accidents: %{y}"
                    "<extra></extra>"
                )
            )
        )

        fig.add_hline(
            y=average_accidents,
            line_dash="dot",
            line_color="#3B82F6",
            annotation_text=(
                f"AVG {average_accidents:.1f}"
            ),
            annotation_font_color="#6E7BFF"
        )

        fig = style_chart(
            fig,
            510
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

        st.caption(
            "2024–25 is partial-year data and excluded "
            "from completed-year modelling."
        )

    with right:

        st.html(
            """
            <div class="section-heading">
                <span>02</span>
                SIGNAL MONITOR
                <em>ML DETECTION</em>
            </div>
            """
        )

        for _, row in detected.iterrows():

            change = row["Percentage_Change"]

            if change < 0:

                direction = "▼"
                change_text = f"{abs(change):.1f}%"
                signal_class = "signal-purple"

            else:

                direction = "▲"
                change_text = f"{change:.1f}%"
                signal_class = "signal-red"

            st.html(
                f"""
                <div class="signal-card {signal_class}">

                    <div class="signal-top">
                        <span>{row['Year']}</span>
                        <b>ANOMALY</b>
                    </div>

                    <div class="signal-number">
                        {direction} {change_text}
                    </div>

                    <div class="signal-bottom">
                        {int(row['Consequential Train Accidents'])}
                        accidents
                        <span>
                            ISOLATION FOREST
                        </span>
                    </div>

                </div>
                """
            )

        st.write("")

        st.html(
            f"""
            <div class="mini-stat">

                <div>
                    <small>HISTORICAL PEAK</small>
                    <strong>{int(peak_accidents)}</strong>
                    <span>{peak_year}</span>
                </div>

                <div>
                    <small>HISTORICAL LOW</small>
                    <strong>{int(lowest_accidents)}</strong>
                    <span>{lowest_year}</span>
                </div>

            </div>
            """
        )

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # INTELLIGENCE STACK
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-heading">
            <span>03</span>
            INTELLIGENCE LAYER
            <em>ANALYTICS ENGINES</em>
        </div>

        <div class="module-grid">

            <div class="module-card">
                <div class="module-number">01</div>
                <div class="module-icon">◈</div>

                <h3>ANOMALY<br>DETECTION</h3>

                <p>
                    Isolation Forest identifies
                    unusual historical patterns.
                </p>

                <div class="module-tag">
                    MACHINE LEARNING
                </div>
            </div>


            <div class="module-card">
                <div class="module-number">02</div>
                <div class="module-icon">⌁</div>

                <h3>BASELINE<br>FORECAST</h3>

                <p>
                    Three-year historical average
                    creates a transparent reference.
                </p>

                <div class="module-tag">
                    STATISTICAL BASELINE
                </div>
            </div>


            <div class="module-card">
                <div class="module-number">03</div>
                <div class="module-icon">▦</div>

                <h3>INFRASTRUCTURE<br>INTELLIGENCE</h3>

                <p>
                    Zonal allocation and expenditure
                    analysis.
                </p>

                <div class="module-tag">
                    FUNDING ANALYTICS
                </div>
            </div>


            <div class="module-card">
                <div class="module-number">04</div>
                <div class="module-icon">△</div>

                <h3>RISK<br>INTELLIGENCE</h3>

                <p>
                    Transparent historical accident
                    level classification.
                </p>

                <div class="module-tag">
                    RULE-BASED
                </div>
            </div>

        </div>
        """
    )

    st.write("")
    st.divider()

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    left, right = st.columns(
        [1.5, 1],
        gap="large"
    )

    with left:

        st.html(
            """
            <div class="section-heading">
                <span>04</span>
                SYSTEM STATUS
                <em>ENGINE HEALTH</em>
            </div>

            <div class="engine-list">

                <div class="engine-row">
                    <div>
                        <span class="engine-dot"></span>
                        DATA PIPELINE
                    </div>
                    <b>READY</b>
                </div>

                <div class="engine-row">
                    <div>
                        <span class="engine-dot"></span>
                        ANOMALY ENGINE
                    </div>
                    <b>READY</b>
                </div>

                <div class="engine-row">
                    <div>
                        <span class="engine-dot"></span>
                        FORECAST ENGINE
                    </div>
                    <b>READY</b>
                </div>

                <div class="engine-row">
                    <div>
                        <span class="engine-dot"></span>
                        RISK ENGINE
                    </div>
                    <b>READY</b>
                </div>

            </div>
            """
        )

    with right:

        st.html(
            f"""
            <div class="method-card">

                <div class="method-label">
                    DATA FOUNDATION
                </div>

                <h3>
                    Government Open Data
                </h3>

                <p>
                    Accident observations from
                    2014–15 to 2024–25 and station
                    funding data from 2020–21 to 2022–23.
                </p>

                <div class="method-line">
                    <span>COMPLETED YEARS</span>
                    <b>{len(completed)}</b>
                </div>

                <div class="method-line">
                    <span>MODELS LOADED</span>
                    <b>02</b>
                </div>

            </div>
            """
        )

    st.write("")
    st.caption(
        "RAILGUARD AI • HISTORICAL RAILWAY SAFETY INTELLIGENCE • v1.0"
    )

    st.caption(
        "Analytical and educational system — not an operational "
        "railway safety system or real-time accident prediction tool."
    )


# ============================================================
# ANOMALY INTELLIGENCE
# ============================================================

elif page == "Anomaly Intelligence":

    st.html(
        """
        <div class="page-hero">

            <div class="hero-label">
                MACHINE LEARNING / SIGNAL DETECTION
            </div>

            <h1>
                Anomaly<br>
                <span>Intelligence</span>
            </h1>

            <p>
                Identification of statistically unusual
                historical accident patterns using
                Isolation Forest.
            </p>

        </div>
        """
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Signals detected",
            len(detected)
        )

    with c2:

        largest_change = (
            detected["Percentage_Change"].abs().max()
            if len(detected) > 0
            else 0
        )

        st.metric(
            "Largest signal",
            f"{largest_change:.1f}%"
        )

    with c3:
        st.metric(
            "Detection engine",
            "Isolation Forest"
        )

    st.divider()

    st.html(
        """
        <div class="section-heading">
            <span>01</span>
            SIGNAL MAP
            <em>HISTORICAL PATTERNS</em>
        </div>
        """
    )

    fig = go.Figure()

    normal = anomaly_features[
        anomaly_features["Status"] == "Normal"
    ]

    anomalies = anomaly_features[
        anomaly_features["Status"] == "Anomaly"
    ]

    fig.add_trace(
        go.Scatter(
            x=normal["Percentage_Change"],
            y=normal[
                "Consequential Train Accidents"
            ],
            mode="markers",
            name="Normal",
            marker=dict(
                size=10,
                color="#6366F1"
            ),
            text=normal["Year"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Change: %{x:.1f}%<br>"
                "Accidents: %{y}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies["Percentage_Change"],
            y=anomalies[
                "Consequential Train Accidents"
            ],
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=17,
                color="#F87171",
                symbol="diamond"
            ),
            text=anomalies["Year"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Change: %{x:.1f}%<br>"
                "Accidents: %{y}"
                "<extra></extra>"
            )
        )
    )

    fig = style_chart(
        fig,
        500
    )

    fig.update_layout(
        xaxis_title="Year-over-year change (%)",
        yaxis_title="Accidents"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.html(
        """
        <div class="section-heading">
            <span>02</span>
            DETECTION LOG
            <em>MODEL OUTPUT</em>
        </div>
        """
    )

    display_data = anomaly_features[
        [
            "Year",
            "Consequential Train Accidents",
            "Accident_Change",
            "Percentage_Change",
            "Status"
        ]
    ].copy()

    display_data.columns = [
        "Year",
        "Accidents",
        "Change",
        "Change %",
        "Status"
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Isolation Forest identifies unusual historical "
        "patterns. It does not predict future accidents."
    )


# ============================================================
# BASELINE FORECAST
# ============================================================

elif page == "Baseline Forecast":

    st.html(
        """
        <div class="page-hero">

            <div class="hero-label">
                FORECASTING / HISTORICAL BASELINE
            </div>

            <h1>
                Baseline<br>
                <span>Forecast</span>
            </h1>

            <p>
                A transparent three-year historical
                average used as a reference baseline.
            </p>

        </div>
        """
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Baseline estimate",
            f"{forecast_value:.1f}"
        )

    with c2:
        st.metric(
            "Years used",
            "3"
        )

    with c3:
        st.metric(
            "Method",
            "Historical average"
        )

    st.divider()

    recent = completed.tail(3).copy()

    st.html(
        """
        <div class="section-heading">
            <span>01</span>
            CALCULATION
            <em>REFERENCE MODEL</em>
        </div>
        """
    )

    st.code(
        """2021–22 → 35
2022–23 → 48
2023–24 → 40

(35 + 48 + 40) / 3 = 41.0""",
        language="text"
    )

    st.divider()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=recent["Year"],
            y=recent[
                "Consequential Train Accidents"
            ],
            name="Actual",
            marker_color="#6366F1"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=recent["Year"],
            y=[forecast_value] * len(recent),
            mode="lines",
            name="Baseline",
            line=dict(
                color="#A78BFA",
                width=3,
                dash="dash"
            )
        )
    )

    fig = style_chart(
        fig,
        450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "This is a historical reference baseline, not "
        "a validated production forecasting model."
    )


# ============================================================
# INFRASTRUCTURE
# ============================================================

elif page == "Infrastructure":

    st.html(
        """
        <div class="page-hero">

            <div class="hero-label">
                INFRASTRUCTURE / FUNDING INTELLIGENCE
            </div>

            <h1>
                Infrastructure<br>
                <span>Intelligence</span>
            </h1>

            <p>
                Zonal analysis of railway station
                development and maintenance allocations
                and expenditure.
            </p>

        </div>
        """
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total allocation",
            f"{total_allocation:,.2f}"
        )

    with c2:
        st.metric(
            "Total expenditure",
            f"{total_expenditure:,.2f}"
        )

    with c3:
        st.metric(
            "Utilization",
            f"{funding_utilization:.2f}%"
        )

    st.divider()

    metric_choice = st.selectbox(
        "COMPARE METRIC",
        [
            "Allocation",
            "Expenditure",
            "Utilization"
        ]
    )

    zone_data = funding_clean.copy()

    if metric_choice == "Allocation":

        zone_data["Value"] = (
            zone_data[
                allocation_columns
            ].sum(axis=1)
        )

    elif metric_choice == "Expenditure":

        zone_data["Value"] = (
            zone_data[
                expenditure_columns
            ].sum(axis=1)
        )

    else:

        zone_data["Allocation_Total"] = (
            zone_data[
                allocation_columns
            ].sum(axis=1)
        )

        zone_data["Expenditure_Total"] = (
            zone_data[
                expenditure_columns
            ].sum(axis=1)
        )

        zone_data["Value"] = (
            zone_data["Expenditure_Total"] /
            zone_data["Allocation_Total"] *
            100
        )

    zone_data = zone_data.sort_values(
        "Value",
        ascending=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=zone_data[
                "Zonal Railway"
            ],
            x=zone_data["Value"],
            orientation="h",
            marker_color="#6366F1",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Value: %{x:.2f}"
                "<extra></extra>"
            )
        )
    )

    fig = style_chart(
        fig,
        max(
            500,
            len(zone_data) * 30
        )
    )

    fig.update_layout(
        xaxis_title=metric_choice,
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.html(
        """
        <div class="section-heading">
            <span>02</span>
            FUNDING DATA
            <em>ZONAL DETAIL</em>
        </div>
        """
    )

    st.dataframe(
        funding_clean,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Funding and accident datasets have different "
        "granularities. This dashboard does not claim "
        "a causal relationship between funding and accidents."
    )


# ============================================================
# RISK INTELLIGENCE
# ============================================================

elif page == "Risk Intelligence":

    st.html(
        """
        <div class="page-hero">

            <div class="hero-label">
                RISK INTELLIGENCE / HISTORICAL CLASSIFICATION
            </div>

            <h1>
                Risk<br>
                <span>Intelligence</span>
            </h1>

            <p>
                Transparent classification of historical
                accident levels relative to the completed-year
                average.
            </p>

        </div>
        """
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Historical average",
            f"{average_accidents:.1f}"
        )

    with c2:
        st.metric(
            "High",
            high_count
        )

    with c3:
        st.metric(
            "Medium",
            medium_count
        )

    with c4:
        st.metric(
            "Low",
            low_count
        )

    st.divider()

    left, right = st.columns(
        [1, 1.7],
        gap="large"
    )

    with left:

        st.html(
            """
            <div class="section-heading">
                <span>01</span>
                DISTRIBUTION
                <em>HISTORICAL RISK</em>
            </div>
            """
        )

        risk_counts = pd.DataFrame(
            {
                "Risk": [
                    "High",
                    "Medium",
                    "Low"
                ],
                "Years": [
                    high_count,
                    medium_count,
                    low_count
                ]
            }
        )

        fig = px.pie(
            risk_counts,
            names="Risk",
            values="Years",
            hole=0.68
        )

        fig.update_traces(
            textinfo="label+value"
        )

        fig = style_chart(
            fig,
            430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.html(
            """
            <div class="section-heading">
                <span>02</span>
                RISK TIMELINE
                <em>YEAR-BY-YEAR</em>
            </div>
            """
        )

        fig = go.Figure()

        risk_colors = {
            "High": "#F87171",
            "Medium": "#F59E0B",
            "Low": "#4ADE80"
        }

        for risk_level in [
            "High",
            "Medium",
            "Low"
        ]:

            subset = risk_data[
                risk_data["Risk_Level"] == risk_level
            ]

            fig.add_trace(
                go.Scatter(
                    x=subset["Year"],
                    y=subset[
                        "Consequential Train Accidents"
                    ],
                    mode="markers",
                    name=risk_level,
                    marker=dict(
                        size=14,
                        color=risk_colors[
                            risk_level
                        ]
                    )
                )
            )

        fig.add_hline(
            y=average_accidents,
            line_dash="dash",
            line_color="#8B5CF6",
            annotation_text=(
                f"Average {average_accidents:.1f}"
            )
        )

        fig = style_chart(
            fig,
            430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    display_risk = risk_data[
        [
            "Year",
            "Consequential Train Accidents",
            "Risk_Level"
        ]
    ].copy()

    display_risk.columns = [
        "Year",
        "Accidents",
        "Historical Risk"
    ]

    st.dataframe(
        display_risk,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Risk levels are historical classifications based "
        "on accident counts relative to the completed-year "
        "average. They are not probabilities of future accidents."
    )