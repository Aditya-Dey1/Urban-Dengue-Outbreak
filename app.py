
import json
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.predict import Predictor

warnings.filterwarnings("ignore")



st.set_page_config(
    page_title="Dengue Intelligence | Bangladesh",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)



BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    st.html(
    f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"
)






TIER_COLORS = {
    "Low": "#22C55E",
    "Medium": "#EAB308",
    "High": "#F97316",
    "Critical": "#EF4444"
}

TIER_ORDER = list(C.TIER_NAMES)

TIER_ACTION = {
    "Low": "Maintain routine dengue surveillance.",
    "Medium": "Increase monitoring and inspect stagnant water.",
    "High": "Schedule larvicide treatment and alert local clinics.",
    "Critical": "Deploy emergency response teams immediately."
}



@st.cache_resource(show_spinner="Loading prediction models...")
def get_predictor():
    return Predictor()


@st.cache_data(show_spinner=False)
def overview(origin, horizon):
    return get_predictor().predict_all_wards(origin, horizon)


@st.cache_data(show_spinner=False)
def heat_data(dates, horizon):
    predictor = get_predictor()

    frames = [
        predictor.predict_all_wards(date, horizon)
        for date in dates
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_metrics():

    path = C.MODEL_DIR / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_backtest():

    path = C.MODEL_DIR / "backtest_test.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_benchmark():

    path = C.MODEL_DIR / "benchmark.json"

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)



def section_title(title, subtitle=None):

    st.html(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <p>{subtitle or ""}</p>
        </div>
        """
    )


def risk_badge(tier):

    color = TIER_COLORS.get(tier, "#64748B")

    return f"""
    <span class="risk-badge"
    style="background:{color}20;color:{color};
    border:1px solid {color}70;">
        ● {tier}
    </span>
    """


def metric_card(title, value, subtitle="", icon="📊"):

    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """
    )


def empty_message(message):

    st.info(message)



def chart_theme(chart):
    """Apply a consistent, presentation-ready dark chart theme."""
    return chart.configure_view(
        strokeOpacity=0,
        continuousWidth=620,
        continuousHeight=360
    ).configure_axis(
        labelColor="#C7D2E0",
        titleColor="#EAF0F7",
        gridColor="#34465F",
        gridOpacity=0.42,
        gridDash=[3, 5],
        domain=False,
        tickColor="#52657D",
        tickSize=4,
        labelFontSize=13,
        titleFontSize=14,
        titleFontWeight=700,
        labelFont="Segoe UI",
        titleFont="Segoe UI",
        labelLimit=220,
        labelPadding=8,
        titlePadding=12
    ).configure_legend(
        labelColor="#DCE5EF",
        titleColor="#F1F5F9",
        labelFontSize=13,
        titleFontSize=13,
        titleFontWeight=700,
        symbolSize=130,
        symbolStrokeWidth=1.5,
        padding=12,
        cornerRadius=8,
        orient="bottom"
    ).configure_title(
        color="#F8FAFC",
        fontSize=19,
        fontWeight=700,  # 750 is not supported by Altair
        font="Segoe UI",
        anchor="start",
        offset=14
    ).configure_mark(
        tooltip=True
    )





try:

    p = get_predictor()

    dates = [pd.Timestamp(d) for d in p.dates]

    valid_dates = dates[8:]

    if not valid_dates:
        st.error("Not enough historical data for forecasting.")
        st.stop()

except Exception as e:

    st.error(f"Unable to initialize prediction system: {e}")
    st.stop()



with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🦟</div>
            <div>
                <h2>DENGUE AI</h2>
                <p>INTELLIGENCE PLATFORM</p>
            </div>
        </div>
        """,
    )

    st.divider()

    st.markdown("### 📍 Location Filters")

    cities = ["All"] + sorted(set(p.city_of.values()))

    city = st.selectbox(
        "Select City",
        cities
    )

    ward_opts = [
        w for w in p.wards
        if city == "All" or p.city_of[w] == city
    ]

    if not ward_opts:
        st.error("No wards available for this city.")
        st.stop()

    ward = st.selectbox(
        "Select Ward",
        ward_opts
    )

    st.divider()

    st.markdown("### 📅 Forecast Settings")

    PRESETS = {
        "Latest available week": valid_dates[-1],
        "Monsoon 2025": min(
            valid_dates,
            key=lambda d: abs(d - pd.Timestamp("2025-07-21"))
        ),
        "Monsoon 2024": min(
            valid_dates,
            key=lambda d: abs(d - pd.Timestamp("2024-07-22"))
        )
    }

    preset = st.selectbox(
        "Quick Date Selection",
        list(PRESETS.keys())
    )

    origin = st.select_slider(
        "Forecast Observation Date",
        options=valid_dates,
        value=PRESETS[preset],
        format_func=lambda d: d.strftime("%d %b %Y"),
        key=f"origin_{preset}"
    )

    horizon = st.radio(
        "Prediction Horizon",
        list(C.HORIZONS),
        index=1,
        horizontal=True,
        format_func=lambda x: f"{x} Weeks"
    )

    st.divider()

    st.html(
        """
        <div class="sidebar-status">
            <span class="status-dot"></span>
            Prediction Engine Ready
        </div>
        """
    )

    st.caption(
        f"Historical data: {p.meta['data_start']} to "
        f"{p.meta['data_end']}"
    )

    st.caption(
        "Forecasts use historical weather and ward-level features."
    )




st.html(
    """
    <div class="main-header">

        <div class="header-top">
            <div>
                <div class="eyebrow">
                    BANGLADESH • PUBLIC HEALTH INTELLIGENCE
                </div>

                <h1>🦟 Dengue Early Warning</h1>

                <p>
                    AI-powered dengue surveillance, risk prediction
                    and decision support system.
                </p>
            </div>

            <div class="header-status">
                <span class="status-dot"></span>
                SYSTEM OPERATIONAL
            </div>
        </div>

    </div>
    """
)

st.write("")




tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Ward Intelligence",
    "📊 City Surveillance",
    "🧠 Model Performance",
    "⚡ System Efficiency"
])



with tab1:

    section_title(
        "Ward Risk Intelligence",
        "Forecast dengue cases and identify potential outbreak risks."
    )

    st.html(
        f"""
        <div class="location-banner">
            <div>
                <span class="location-label">CURRENT LOCATION</span>
                <h3>{ward}</h3>
                <p>{p.city_of[ward]}</p>
            </div>

            <div class="date-label">
                FORECAST ORIGIN<br>
                <strong>{origin.strftime('%d %B %Y')}</strong>
            </div>
        </div>
        """
    )


    try:

        results = {
            h: p.predict_ward(ward, origin, h)
            for h in C.HORIZONS
        }

    except Exception as e:

        st.error(f"Prediction error: {e}")
        st.stop()

    st.markdown("### 🔮 Future Risk Predictions")

    cols = st.columns(3)

    for col, h in zip(cols, C.HORIZONS):

        r = results[h]

        with col:

            color = TIER_COLORS.get(r["tier"], "#64748B")

            st.html(
                f"""
                <div class="forecast-card"
                style="border-top:4px solid {color};">

                    <div class="forecast-period">
                        {h} WEEKS AHEAD
                    </div>

                    <div class="forecast-date">
                        {r['target_date'].strftime('%d %b %Y')}
                    </div>

                    <div class="forecast-number">
                        {r['pred']:.1f}
                    </div>

                    <div class="forecast-caption">
                        Expected dengue cases / week
                    </div>

                    {risk_badge(r['tier'])}

                    <div class="forecast-range">
                        80% Prediction Interval<br>
                        <strong>{r['lo']:.1f} – {r['hi']:.1f}</strong>
                    </div>

                </div>
                """
            )

    sel = results[horizon]

    st.write("")

    st.html(
        f"""
        <div class="alert-box"
        style="border-left:5px solid {TIER_COLORS[sel['tier']]};">

            <div class="alert-heading">
                🚨 PUBLIC HEALTH RECOMMENDATION
            </div>

            <h3>{sel['tier']} Risk Classification</h3>

            <p>{TIER_ACTION[sel['tier']]}</p>

        </div>
        """
    )

    st.write("")

    left, right = st.columns([1, 1])


    with left:

        st.markdown("### 🎯 Main Risk Contributors")

        g = sel["groups"].reset_index()

        g.columns = ["Driver", "effect"]

        g["direction"] = np.where(
            g["effect"] >= 0,
            "Raises risk",
            "Lowers risk"
        )

        driver_chart = alt.Chart(g).mark_bar(
            cornerRadiusEnd=8,
            height=24
        ).encode(

            x=alt.X(
                "effect:Q",
                title="Effect on expected cases (%)"
            ),

            y=alt.Y(
                "Driver:N",
                sort="-x",
                title=None
            ),

            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises risk", "Lowers risk"],
                    range=["#EF4444", "#22C55E"]
                ),
                legend=None
            ),

            tooltip=[
                "Driver",
                alt.Tooltip("effect:Q", format=".2f")
            ]

        ).properties(height=360)

        st.altair_chart(
            chart_theme(driver_chart),
            width="stretch"
        )

        st.markdown("#### Individual Factors")

        for lab, value in sel["top"]:

            arrow = "▲" if value >= 0 else "▼"

            st.write(
                f"{arrow} **{lab}** — {value:+.1f}%"
            )

    with right:

        st.markdown("### 🌧️ Weather & Disease History")

        hist = p.history(
            ward,
            26,
            until=origin
        ).copy()

        if hist.empty:

            empty_message("No historical data available.")

        else:

            hist["window"] = np.where(
                hist["date"] > origin - pd.Timedelta(weeks=4),
                "Recent 4 Weeks",
                "Earlier"
            )

            rain = alt.Chart(hist).mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
                opacity=0.88
            ).encode(

                x=alt.X("date:T", title=None),

                y=alt.Y(
                    "rainfall_mm:Q",
                    title="Rainfall (mm)"
                ),

                color=alt.Color(
                    "window:N",
                    scale=alt.Scale(
                        range=["#64748B", "#38BDF8"]
                    ),
                    legend=alt.Legend(orient="bottom")
                ),

                tooltip=["date:T", "rainfall_mm:Q"]

            ).properties(height=250)

            st.altair_chart(
                chart_theme(rain),
                width="stretch"
            )

            cases = alt.Chart(hist).mark_line(
                color="#FB7185",
                point=alt.OverlayMarkDef(filled=True, size=72, color="#FB7185", stroke="#FFE4E6", strokeWidth=1.5),
                strokeWidth=3.5,
                strokeCap="round"
            ).encode(

                x=alt.X("date:T", title=None),

                y=alt.Y(
                    "dengue_cases:Q",
                    title="Recorded cases"
                ),

                tooltip=[
                    "date:T",
                    "dengue_cases:Q"
                ]

            ).properties(height=250)

            st.altair_chart(
                chart_theme(cases),
                width="stretch"
            )

            w = hist.iloc[-1]

            st.html(
                f"""
                <div class="weather-card">
                    <h4>Latest Weather Conditions</h4>
                    <div class="weather-grid">
                        <div>🌧️ Rainfall<br><strong>{w['rainfall_mm']:.1f} mm</strong></div>
                        <div>🌡️ Temperature<br><strong>{w['avg_temp_c']:.1f} °C</strong></div>
                        <div>💧 Humidity<br><strong>{w['relative_humidity_pct']:.1f}%</strong></div>
                    </div>
                    <hr>
                    <p>
                    Water logging: <strong>{w['water_logging_risk_index']:.2f}</strong><br>
                    Sanitation: <strong>{w['sanitation_score']:.2f}</strong><br>
                    Population density:
                    <strong>{int(w['population_density_sqkm']):,}/km²</strong>
                    </p>
                </div>
                """
            )

with tab2:

    section_title(
        "Citywide Dengue Surveillance",
        "Monitor risk distribution and identify high-priority wards."
    )

    try:

        ov = overview(origin, horizon)

        if city != "All":
            ov = ov[ov["city"] == city]

        if ov.empty:

            empty_message("No ward predictions available.")
        
        else:

            ov = ov.sort_values(
                "pred",
                ascending=False
            )

            counts = (
                ov["tier"]
                .value_counts()
                .reindex(TIER_ORDER)
                .fillna(0)
                .astype(int)
            )


            total = len(ov)

            critical = counts.get("Critical", 0)

            high = counts.get("High", 0)

            medium = counts.get("Medium", 0)

            low = counts.get("Low", 0)

            avg_cases = ov["pred"].mean()

            st.markdown("### 📈 Surveillance Overview")

            cols = st.columns(4)

            with cols[0]:
                metric_card(
                    "Total Wards",
                    total,
                    "Wards monitored",
                    "🏙️"
                )

            with cols[1]:
                metric_card(
                    "High + Critical",
                    high + critical,
                    "Priority attention",
                    "🚨"
                )

            with cols[2]:
                metric_card(
                    "Average Forecast",
                    f"{avg_cases:.1f}",
                    "Expected cases / week",
                    "📊"
                )

            with cols[3]:
                metric_card(
                    "Critical Wards",
                    critical,
                    "Immediate attention",
                    "🔴"
                )

            st.write("")


            st.markdown("### Risk Distribution")

            risk_df = pd.DataFrame({
                "Risk": TIER_ORDER,
                "Wards": [
                    counts[t] for t in TIER_ORDER
                ]
            })

            risk_chart = alt.Chart(risk_df).mark_bar(
                cornerRadiusEnd=9,
                height=28
            ).encode(

                x=alt.X(
                    "Wards:Q",
                    title="Number of wards"
                ),

                y=alt.Y(
                    "Risk:N",
                    sort=TIER_ORDER,
                    title=None
                ),

                color=alt.Color(
                    "Risk:N",
                    scale=alt.Scale(
                        domain=TIER_ORDER,
                        range=[
                            TIER_COLORS[t]
                            for t in TIER_ORDER
                        ]
                    ),
                    legend=None
                ),

                tooltip=["Risk", "Wards"]

            ).properties(height=270)

            st.altair_chart(
                chart_theme(risk_chart),
                width="stretch"
            )


            st.markdown("### 🏆 Highest Risk Wards")

            show = ov[
                [
                    "ward_id",
                    "city",
                    "tier",
                    "pred",
                    "lo",
                    "hi"
                ]
            ].rename(columns={
                "ward_id": "Ward",
                "city": "City",
                "tier": "Risk Level",
                "pred": "Expected Cases",
                "lo": "Lower Range",
                "hi": "Upper Range"
            })

            st.dataframe(
                show.style.format({
                    "Expected Cases": "{:.1f}",
                    "Lower Range": "{:.1f}",
                    "Upper Range": "{:.1f}"
                }).map(
                    lambda v: (
                        f"background-color:{TIER_COLORS[v]};color:white"
                        if v in TIER_COLORS else ""
                    ),
                    subset=["Risk Level"]
                ),
                hide_index=True,
                width="stretch",
                height=350
            )


            st.write("")

            st.markdown("### 🗺️ Dengue Risk Heatmap")

            last = tuple(
                valid_dates[-26:]
                if origin == valid_dates[-1]
                else [
                    d for d in valid_dates
                    if d <= origin
                ][-26:]
            )

            hd = heat_data(last, horizon)

            if city != "All":
                hd = hd[hd["city"] == city]

            if hd.empty:

                empty_message("Heatmap data unavailable.")

            else:

                hm = alt.Chart(hd).mark_rect(
                    stroke="#172438",
                    strokeWidth=1,
                    cornerRadius=2
                ).encode(

                    x=alt.X(
                        "date:T",
                        title="Forecast observation date"
                    ),

                    y=alt.Y(
                        "ward_id:N",
                        title="Ward",
                        sort=alt.SortField("city")
                    ),

                    color=alt.Color(
                        "tier:N",
                        scale=alt.Scale(
                            domain=TIER_ORDER,
                            range=[
                                TIER_COLORS[t]
                                for t in TIER_ORDER
                            ]
                        ),
                        legend=alt.Legend(
                            title="Risk Classification"
                        )
                    ),

                    tooltip=[
                        "ward_id",
                        "city",
                        "date:T",
                        alt.Tooltip("pred:Q", format=".1f"),
                        "tier"
                    ]

                ).properties(
                    height=max(
                        350,
                        18 * hd["ward_id"].nunique()
                    )
                )

                st.altair_chart(
                    chart_theme(hm),
                    width="stretch"
                )

    except Exception as e:

        st.error(f"Unable to generate city overview: {e}")


with tab3:

    section_title(
        "Machine Learning Model Validation",
        "Evaluate forecasting accuracy and reliability using chronological testing."
    )

    try:

        m = load_metrics()

        info = m["_info"]

        st.markdown("### 📅 Dataset Splitting")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Training Period",
                "TRAIN",
                info["train_period"],
                "🧠"
            )

        with c2:
            metric_card(
                "Validation Period",
                "VALIDATION",
                info["validation_period"],
                "🔍"
            )

        with c3:
            metric_card(
                "Testing Period",
                "TEST",
                info["test_period"],
                "🎯"
            )

        st.write("")

        st.info(
            f"Tier cut points: Medium ≥ {info['tier_cuts'][0]}, "
            f"High ≥ {info['tier_cuts'][1]}, "
            f"Critical ≥ {info['tier_cuts'][2]}"
        )


        st.markdown("### 📊 Model Evaluation Metrics")

        rows = []

        for h in C.HORIZONS:

            r = m[f"h{h}"]

            rows.append({

                "Horizon": f"{h} Weeks",

                "MAE": r["MAE"],

                "RMSE": r["RMSE"],

                "Tier Accuracy": r["tier_accuracy"],

                "Within 1 Tier": r["within_one_tier"],

                "High/Critical Recall": r["high_or_critical_recall"],

                "80% Interval Coverage": r["interval_80_coverage"],

                "Ward Mean Baseline": r["baseline_ward_mean_MAE"],

                "Last Week Baseline": r["baseline_persistence_MAE"]

            })

        metrics_df = pd.DataFrame(rows)

        st.dataframe(
            metrics_df.style.format({
                col: "{:.3f}"
                for col in metrics_df.columns
                if col != "Horizon"
            }),
            hide_index=True,
            width="stretch"
        )


        st.write("")

        st.markdown("### 🔬 Actual vs Predicted Cases")

        bt = load_backtest()

        b = bt[
            (bt["ward_id"] == ward)
            & (bt["horizon"] == horizon)
        ].sort_values("target_date")

        if b.empty:

            empty_message("No backtesting records for this selection.")

        else:

            band = alt.Chart(b).mark_area(
                opacity=0.18,
                color="#60A5FA",
                interpolate="monotone"
            ).encode(
                x="target_date:T",
                y="lo:Q",
                y2="hi:Q"
            )

            predicted = alt.Chart(b).mark_line(
                color="#38BDF8",
                strokeWidth=3.5,
                strokeCap="round",
                point=alt.OverlayMarkDef(filled=True, size=58, color="#38BDF8", stroke="#DBEAFE", strokeWidth=1.2)
            ).encode(

                x=alt.X(
                    "target_date:T",
                    title="Target Date"
                ),

                y=alt.Y(
                    "pred:Q",
                    title="Dengue Cases"
                )

            )

            actual = alt.Chart(b).mark_line(
                color="#FB7185",
                strokeDash=[7, 4],
                strokeWidth=2.8,
                strokeCap="round"
            ).encode(
                x="target_date:T",
                y="actual:Q"
            )

            st.altair_chart(
                chart_theme(band + predicted + actual),
                width="stretch"
            )

            st.caption(
                "Blue = prediction | Red = actual cases | "
                "Shaded area = 80% prediction interval."
            )


        st.write("")

        left, right = st.columns(2)

        with left:

            st.markdown("### 🎯 Tier Confusion Matrix")

            cm = np.array(
                m[f"h{horizon}"]["confusion_matrix"]
            )

            cmd = pd.DataFrame([
                {
                    "Actual": TIER_ORDER[i],
                    "Predicted": TIER_ORDER[j],
                    "n": int(cm[i, j])
                }
                for i in range(4)
                for j in range(4)
            ])

            base = alt.Chart(cmd).encode(
                x=alt.X(
                    "Predicted:N",
                    sort=TIER_ORDER
                ),
                y=alt.Y(
                    "Actual:N",
                    sort=TIER_ORDER
                )
            )

            heat = base.mark_rect(cornerRadius=5, stroke="#172438", strokeWidth=2).encode(
                color=alt.Color(
                    "n:Q",
                    scale=alt.Scale(
                        scheme="teals"
                    ),
                    legend=None
                )
            )

            text = base.mark_text(
                color="white",
                fontSize=17,
                fontWeight="bold"
            ).encode(
                text="n:Q"
            )

            st.altair_chart(
                chart_theme((heat + text).properties(height=340)),
                width="stretch"
            )

        with right:

            st.markdown("### 🏙️ Prediction Error by City")

            e = bt[
                bt["horizon"] == horizon
            ].copy()

            if not e.empty:

                e["abs_err"] = (
                    e["actual"] - e["pred"]
                ).abs()

                e["sq"] = (
                    e["actual"] - e["pred"]
                ) ** 2

                ec = e.groupby("city").agg(
                    MAE=("abs_err", "mean"),
                    RMSE=(
                        "sq",
                        lambda s: float(np.sqrt(s.mean()))
                    )
                ).reset_index()

                st.dataframe(
                    ec.style.format({
                        "MAE": "{:.3f}",
                        "RMSE": "{:.3f}"
                    }),
                    hide_index=True,
                    width="stretch"
                )

        r = m[f"h{horizon}"]

        st.info(
            f"Weather-only model MAE: {r['MAE']:.3f}. "
            f"Model with past case counts: "
            f"{r['with_case_history_MAE']:.3f}."
        )

    except Exception as e:

        st.error(f"Unable to load validation results: {e}")


with tab4:

    section_title(
        "System Performance & Efficiency",
        "Monitor computational resources and machine-learning deployment."
    )

    try:

        bm = load_benchmark()

        if bm is None:

            st.warning(
                "Benchmark information is unavailable. "
                "Run python -m src.benchmark to generate benchmark.json."
            )

        else:

            st.markdown("### ⚡ Benchmark Information")

            st.json(bm)

    except Exception as e:

        st.error(f"Unable to load benchmark: {e}")

    st.write("")

    st.markdown("### ⚙️ System Architecture")

    c1, c2 = st.columns(2)

    with c1:

        st.html(
            """
            <div class="info-card">
                <h3>🧠 Machine Learning</h3>
                <ul>
                    <li>Offline model training</li>
                    <li>Historical weather features</li>
                    <li>Ward-level prediction</li>
                    <li>Multi-horizon forecasting</li>
                    <li>Risk classification</li>
                </ul>
            </div>
            """
        )

    with c2:

        st.html(
            """
            <div class="info-card">
                <h3>🚀 Deployment</h3>
                <ul>
                    <li>Cached model loading</li>
                    <li>Cached predictions</li>
                    <li>Vectorized data processing</li>
                    <li>Batch ward forecasting</li>
                    <li>No external API calls at inference</li>
                </ul>
            </div>
            """
        )

    st.write("")

    st.markdown("### 📡 Platform Information")

    st.html(
        """
        <div class="info-card">
            <h3>Dengue Early Warning Platform</h3>
            <p>
            An AI-assisted public-health decision support platform
            designed to identify potential dengue hotspots and help
            public-health authorities prioritize surveillance and
            intervention activities.
            </p>

            <hr>

            <p><strong>Prediction:</strong> 2, 3 and 4 weeks ahead</p>
            <p><strong>Classification:</strong> Low / Medium / High / Critical</p>
            <p><strong>Visualization:</strong> Interactive analytical dashboard</p>
            <p><strong>Architecture:</strong> Python + Streamlit + Machine Learning</p>
        </div>
        """
    )


st.html(
    """
    <div class="footer">

        <div class="footer-line"></div>

        <h3>🦟 DENGUE EARLY WARNING SYSTEM</h3>

        <p>
            AI-powered disease surveillance and predictive analytics
            for Bangladesh.
        </p>

        <span>
            DECISION SUPPORT TOOL • NOT A CLINICAL DIAGNOSTIC SYSTEM
        </span>

    </div>
    """
)
