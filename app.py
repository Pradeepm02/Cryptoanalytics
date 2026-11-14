import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import io
import hashlib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Crypto Analytics Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background-color: #0b1020;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stMetric"] {
    background-color: #172036;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #26334d;
}

h1, h2, h3 {
    color: #f8fafc;
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1;
}

.stButton > button {
    background-color: #6366f1;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #4f46e5;
    color: white;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(file_bytes):

    data = pd.read_csv(
        io.BytesIO(file_bytes)
    )

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True
        )
        .str.strip("_")
    )

    data = data.loc[
        :,
        ~data.columns.str.startswith("unnamed")
    ]

    required = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return data

def prepare_coin(data, symbol):

    coin = data[
        data["symbol"] == symbol
    ].copy()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market_cap"
    ]:

        if col not in coin.columns:
            coin[col] = 0

        coin[col] = pd.to_numeric(
            coin[col],
            errors="coerce"
        )

    coin = coin.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    valid = (
        (coin["open"] > 0) &
        (coin["high"] > 0) &
        (coin["low"] > 0) &
        (coin["close"] > 0)
    )

    valid &= (
        coin["high"] >=
        coin[
            ["open", "close", "low"]
        ].max(axis=1)
    )

    valid &= (
        coin["low"] <=
        coin[
            ["open", "close", "high"]
        ].min(axis=1)
    )

    coin = coin.loc[valid].copy()

    coin = coin.sort_values("date")

    coin = coin.drop_duplicates(
        subset=["date"],
        keep="last"
    )

    coin["volume_missing"] = (
        coin["volume"].isna().astype(int)
    )

    coin["market_cap_missing"] = (
        coin["market_cap"].isna().astype(int)
    )

    coin["volume"] = (
        coin["volume"].ffill().fillna(0)
    )

    coin["market_cap"] = (
        coin["market_cap"].ffill().fillna(0)
    )

    return coin.reset_index(drop=True)

def create_features(coin):

    df = coin.copy()

    # Daily returns

    df["returns"] = (
        df["close"].pct_change()
    )

    df["log_returns"] = np.log(
        df["close"]
    ).diff()

    # Moving averages

    df["ma_7"] = (
        df["close"].rolling(7).mean()
    )

    df["ma_20"] = (
        df["close"].rolling(20).mean()
    )

    df["ma_50"] = (
        df["close"].rolling(50).mean()
    )

    df["ma_100"] = (
        df["close"].rolling(100).mean()
    )

    # Exponential moving averages

    df["ema_12"] = (
        df["close"].ewm(span=12).mean()
    )

    df["ema_26"] = (
        df["close"].ewm(span=26).mean()
    )

    # RSI

    delta = df["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (
        avg_loss + 1e-10
    )

    df["rsi"] = (
        100 - (100 / (1 + rs))
    )

    # MACD

    df["macd"] = (
        df["ema_12"] - df["ema_26"]
    )

    df["macd_signal"] = (
        df["macd"].ewm(span=9).mean()
    )

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # Bollinger Bands

    df["bb_middle"] = (
        df["close"].rolling(20).mean()
    )

    bb_std = df["close"].rolling(20).std()

    df["bb_upper"] = (
        df["bb_middle"] + 2 * bb_std
    )

    df["bb_lower"] = (
        df["bb_middle"] - 2 * bb_std
    )

    # Volatility

    df["volatility_10"] = (
        df["returns"].rolling(10).std()
    )

    df["volatility_20"] = (
        df["returns"].rolling(20).std()
    )

    # Momentum

    df["momentum_10"] = (
        df["close"] - df["close"].shift(10)
    )

    # Additional features

    df["range_pct"] = (
        (df["high"] - df["low"]) /
        df["close"]
    )

    df["close_to_ma20"] = (
        df["close"] / df["ma_20"] - 1
    )

    # Next-day target

    df["target_next_return"] = (
        df["close"].shift(-1) /
        df["close"] - 1
    )

    df["target_next_close"] = (
        df["close"].shift(-1)
    )

    return df

def evaluate(actual, predicted):

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "R2 Score": r2_score(
            actual,
            predicted
        ),

        "MAPE": np.mean(
            np.abs(
                (actual - predicted) /
                np.maximum(
                    np.abs(actual),
                    1e-8
                )
            )
        ) * 100
    }


def run_models(coin):

    df = create_features(coin)

    features = [

        "open",
        "high",
        "low",
        "close",

        "volume",
        "market_cap",

        "volume_missing",
        "market_cap_missing",

        "returns",
        "log_returns",

        "ma_7",
        "ma_20",
        "ma_50",
        "ma_100",

        "ema_12",
        "ema_26",

        "macd",
        "macd_signal",

        "rsi",

        "bb_upper",
        "bb_lower",

        "volatility_10",
        "volatility_20",

        "momentum_10",

        "range_pct",
        "close_to_ma20"
    ]

    model_data = df.dropna(
        subset=features + [
            "target_next_return",
            "target_next_close"
        ]
    ).copy()

    if len(model_data) < 150:

        raise ValueError(
            "Not enough historical records. "
            "Select a cryptocurrency with more data."
        )

    X = model_data[features]

    y = model_data[
        "target_next_return"
    ]

    split = int(len(model_data) * 0.80)

    X_train = X.iloc[:split]

    X_test = X.iloc[split:]

    y_train = y.iloc[:split]

    y_test = y.iloc[split:]

    # Random Forest

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_return = rf.predict(X_test)

    # XGBoost

    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(
        X_train,
        y_train
    )

    xgb_return = xgb.predict(X_test)

    # Convert returns to prices

    current_prices = X_test[
        "close"
    ].to_numpy()

    rf_price = np.maximum(
        0,
        current_prices * (1 + rf_return)
    )

    xgb_price = np.maximum(
        0,
        current_prices * (1 + xgb_return)
    )

    actual_price = model_data[
        "target_next_close"
    ].iloc[split:].to_numpy()

    baseline_price = current_prices

    # Evaluate models

    results = pd.DataFrame([

        {
            "Model": "Naive Baseline",
            **evaluate(
                actual_price,
                baseline_price
            )
        },

        {
            "Model": "Random Forest",
            **evaluate(
                actual_price,
                rf_price
            )
        },

        {
            "Model": "XGBoost",
            **evaluate(
                actual_price,
                xgb_price
            )
        }

    ])

    # Prediction history

    predictions = pd.DataFrame({

        "date": model_data[
            "date"
        ].iloc[split:].values,

        "current_close": current_prices,

        "actual_next_close": actual_price,

        "random_forest": rf_price,

        "xgboost": xgb_price,

        "baseline": baseline_price,

        "rsi": model_data[
            "rsi"
        ].iloc[split:].values,

        "macd": model_data[
            "macd"
        ].iloc[split:].values,

        "macd_signal": model_data[
            "macd_signal"
        ].iloc[split:].values,

        "volatility_20": model_data[
            "volatility_20"
        ].iloc[split:].values

    })

    predictions["expected_change"] = (
        (
            predictions["xgboost"] -
            predictions["current_close"]
        ) /
        predictions["current_close"]
    ) * 100

    predictions["signal"] = np.select(
        [
            predictions["expected_change"] > 0.5,

            predictions["expected_change"] < -0.5
        ],

        [
            "BUY",
            "SELL"
        ],

        default="HOLD"
    )

    # Fit final models using all available
    # labelled historical observations

    final_rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    final_rf.fit(X, y)

    final_xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    final_xgb.fit(X, y)

    # Latest known complete feature row

    latest = df.dropna(
        subset=features
    ).iloc[[-1]]

    current_price = float(
        latest["close"].iloc[0]
    )

    next_return = float(
        final_xgb.predict(
            latest[features]
        )[0]
    )

    predicted_price = max(
        0,
        current_price * (1 + next_return)
    )

    expected_change = next_return * 100

    if expected_change > 0.5:
        next_signal = "BUY"
    elif expected_change < -0.5:
        next_signal = "SELL"
    else:
        next_signal = "HOLD"

    # Rule-based reasoning

    latest_rsi = float(
        latest["rsi"].iloc[0]
    )

    latest_volatility = float(
        latest["volatility_20"].iloc[0]
    )

    vol_threshold = model_data[
        "volatility_20"
    ].quantile(0.90)

    if latest_volatility > vol_threshold:

        reasoning = (
            "High volatility detected. "
            "Market risk is elevated."
        )

    elif latest_rsi < 30 and expected_change > 0.5:

        reasoning = (
            "Oversold RSI and positive predicted movement."
        )

    elif latest_rsi > 70 and expected_change < -0.5:

        reasoning = (
            "Overbought RSI and negative predicted movement."
        )

    elif next_signal == "BUY":

        reasoning = (
            "Model predicts positive next-day movement."
        )

    elif next_signal == "SELL":

        reasoning = (
            "Model predicts negative next-day movement."
        )

    else:

        reasoning = (
            "Mixed market conditions. HOLD is suggested."
        )

    predictions["AI_Reasoning"] = np.where(
        predictions["volatility_20"] > vol_threshold,
        "High historical volatility detected.",
        np.where(
            predictions["signal"] == "BUY",
            "Positive predicted movement.",
            np.where(
                predictions["signal"] == "SELL",
                "Negative predicted movement.",
                "Mixed indicators. Monitor the market."
            )
        )
    )

    importance = pd.DataFrame({

        "Feature": features,

        "Importance": final_rf.feature_importances_

    }).sort_values(
        "Importance",
        ascending=False
    )

    forecast = {

        "current_price": current_price,

        "predicted_price": predicted_price,

        "expected_change": expected_change,

        "signal": next_signal,

        "reasoning": reasoning,

        "date": latest["date"].iloc[0]

    }

    return {

        "feature_data": df,

        "results": results,

        "predictions": predictions,

        "importance": importance,

        "forecast": forecast

    }

st.sidebar.title("₿ Crypto Analytics")

st.sidebar.caption(
    "Cryptocurrency Market Intelligence Platform"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Cryptocurrency Dataset",
    type=["csv"]
)

# Use uploaded file or local CSV

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

else:

    local_file = Path("all_currencies.csv")

    if local_file.exists():

        file_bytes = local_file.read_bytes()

    else:

        st.title("₿ Cryptocurrency Analytics")

        st.info(
            "Upload your all_currencies.csv file "
            "to start the dashboard."
        )

        st.stop()

try:

    data = load_data(file_bytes)

except Exception as error:

    st.error(f"Dataset loading error: {error}")

    st.stop()


symbols = sorted(
    data["symbol"].dropna().unique().tolist()
)

if not symbols:

    st.error("No cryptocurrency symbols found.")

    st.stop()


default_index = (
    symbols.index("BTC")
    if "BTC" in symbols
    else 0
)

selected_coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    symbols,
    index=default_index
)

coin = prepare_coin(
    data,
    selected_coin
)

if coin.empty:

    st.error("No valid records for selected cryptocurrency.")

    st.stop()


st.sidebar.markdown("---")

st.sidebar.write(
    f"Available currencies: {len(symbols)}"
)

st.sidebar.write(
    f"Selected: {selected_coin}"
)

st.sidebar.write(
    f"Records: {len(coin):,}"
)

st.sidebar.write(
    f"History: {coin['date'].min().date()} to "
    f"{coin['date'].max().date()}"
)

dataset_hash = hashlib.sha256(
    file_bytes
).hexdigest()[:16]

analysis_key = (
    dataset_hash + "_" + selected_coin
)

if st.sidebar.button(
    "Run Analysis & Train Models",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Processing data and training ML models..."
        ):

            result = run_models(coin)

            st.session_state["analysis"] = result

            st.session_state["analysis_key"] = analysis_key

        st.sidebar.success(
            "Analysis completed successfully!"
        )

    except Exception as error:

        st.error(
            f"Analysis failed: {error}"
        )

        st.stop()


analysis = st.session_state.get(
    "analysis"
)

saved_key = st.session_state.get(
    "analysis_key"
)

if analysis is None or saved_key != analysis_key:

    st.title("₿ Cryptocurrency Analytics Dashboard")

    st.markdown(
        "### Welcome to your Cryptocurrency Market Analysis System"
    )

    st.write(
        "Select a cryptocurrency from the sidebar "
        "and click **Run Analysis & Train Models** "
        "to generate analytics and predictions."
    )

    st.info(
        "The dashboard will display historical charts, "
        "technical indicators, ML performance, "
        "and next-day forecasts."
    )

    st.stop()

feature_data = analysis["feature_data"]

results = analysis["results"]

predictions = analysis["predictions"]

importance = analysis["importance"]

forecast = analysis["forecast"]

latest = feature_data.iloc[-1]

current_price = forecast["current_price"]

predicted_price = forecast["predicted_price"]

price_change = forecast["expected_change"]

next_signal = forecast["signal"]

st.title(
    "₿ Cryptocurrency Analytics Dashboard"
)

st.markdown(
    f"### Market Analysis for {selected_coin}"
)

st.caption(
    "Historical cryptocurrency analytics "
    "and machine learning price prediction"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Latest Closing Price",
        f"{current_price:,.4f}"
    )

with col2:

    st.metric(
        "Predicted Next-Day Price",
        f"{predicted_price:,.4f}",
        f"{price_change:+.2f}%"
    )

with col3:

    st.metric(
        "RSI",
        f"{latest['rsi']:.2f}"
    )

with col4:

    st.metric(
        "Historical Records",
        f"{len(coin):,}"
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📊 Market Overview",

    "📈 Technical Analysis",

    "🤖 ML Predictions",

    "🎯 Trading Signals",

    "📁 Dataset Explorer"

])

with tab1:

    st.subheader(
        f"{selected_coin} Price History"
    )

    chart_data = feature_data.tail(365)

    fig = px.line(
        chart_data,
        x="date",
        y="close",
        title="Historical Closing Price",
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(width=2)
    )

    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Trading Volume")

    volume_fig = px.area(
        chart_data,
        x="date",
        y="volume",
        title="Historical Trading Volume",
        template="plotly_dark"
    )

    volume_fig.update_layout(
        height=350
    )

    st.plotly_chart(
        volume_fig,
        use_container_width=True
    )

with tab2:

    st.subheader(
        "Candlestick Chart"
    )

    chart_data = feature_data.tail(365)

    candle = go.Figure()

    candle.add_trace(
        go.Candlestick(

            x=chart_data["date"],

            open=chart_data["open"],

            high=chart_data["high"],

            low=chart_data["low"],

            close=chart_data["close"],

            name="OHLC"

        )
    )

    candle.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["ma_20"],
            name="MA 20"
        )
    )

    candle.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["ma_50"],
            name="MA 50"
        )
    )

    candle.update_layout(
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
        title="Cryptocurrency OHLC Analysis"
    )

    st.plotly_chart(
        candle,
        use_container_width=True
    )

with tab1:

    st.subheader(
        "Cryptocurrency Statistics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Maximum Price",
            f"{coin['close'].max():,.4f}"
        )

    with col2:

        st.metric(
            "Minimum Price",
            f"{coin['close'].min():,.4f}"
        )

    with col3:

        st.metric(
            "Average Price",
            f"{coin['close'].mean():,.4f}"
        )

    with col4:

        st.metric(
            "Average Volume",
            f"{coin['volume'].mean():,.2f}"
        )

    st.write("Statistical Summary")

    st.dataframe(
        coin[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_cap"
            ]
        ].describe().round(4),
        use_container_width=True
    )

with tab2:

    st.subheader(
        "Candlestick Chart"
    )

    chart_data = feature_data.tail(365)

    candle = go.Figure()

    candle.add_trace(
        go.Candlestick(

            x=chart_data["date"],

            open=chart_data["open"],

            high=chart_data["high"],

            low=chart_data["low"],

            close=chart_data["close"],

            name="OHLC"

        )
    )

    candle.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["ma_20"],
            name="MA 20"
        )
    )

    candle.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["ma_50"],
            name="MA 50"
        )
    )

    candle.update_layout(
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
        title="Cryptocurrency OHLC Analysis"
    )

    st.plotly_chart(
    candle,
    use_container_width=True,
    key="crypto_candlestick_chart"
)

with tab2:

    st.subheader(
        "Relative Strength Index"
    )

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["rsi"],
            name="RSI"
        )
    )

    rsi_fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Overbought"
    )

    rsi_fig.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Oversold"
    )

    rsi_fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="RSI"
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

with tab2:

    st.subheader(
        "MACD Analysis"
    )

    macd_fig = px.line(

        chart_data,

        x="date",

        y=["macd", "macd_signal"],

        title="MACD and Signal Line",

        template="plotly_dark"

    )

    macd_fig.update_layout(
        height=400
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True
    )

with tab3:

    st.subheader(
        "Machine Learning Model Performance"
    )

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader(
        "Model RMSE Comparison"
    )

    model_fig = px.bar(

        results,

        x="Model",

        y="RMSE",

        color="Model",

        title="Random Forest vs XGBoost vs Baseline",

        template="plotly_dark"

    )

    model_fig.update_layout(
        height=400,
        showlegend=False
    )

    st.plotly_chart(
        model_fig,
        use_container_width=True
    )

with tab3:

    st.subheader(
        "Actual vs Predicted Price"
    )

    comparison_data = predictions[
        [
            "date",
            "actual_next_close",
            "random_forest",
            "xgboost",
            "baseline"
        ]
    ].melt(

        id_vars="date",

        var_name="Model",

        value_name="Price"

    )

    prediction_fig = px.line(

        comparison_data,

        x="date",

        y="Price",

        color="Model",

        title="Actual and Predicted Cryptocurrency Prices",

        template="plotly_dark"

    )

    prediction_fig.update_layout(
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )

with tab3:

    best_model = results.loc[
        results["RMSE"].idxmin()
    ]

    st.subheader(
        "Best Performing Model"
    )

    st.metric(
        "Lowest RMSE",
        f"{best_model['RMSE']:,.4f}",
        best_model["Model"]
    )

    st.caption(
        "The model with the lowest RMSE performed best "
        "on this chronological test period. "
        "Compare it with the naive baseline before "
        "concluding that machine learning adds value."
    )

with tab4:

    st.subheader(
        "Historical Trading Signals"
    )

    signal_fig = go.Figure()

    signal_fig.add_trace(

        go.Scatter(

            x=predictions["date"],

            y=predictions["current_close"],

            name="Closing Price",

            mode="lines"

        )

    )

    buy = predictions[
        predictions["signal"] == "BUY"
    ]

    sell = predictions[
        predictions["signal"] == "SELL"
    ]

    hold = predictions[
        predictions["signal"] == "HOLD"
    ]

    signal_fig.add_trace(

        go.Scatter(

            x=buy["date"],

            y=buy["current_close"],

            mode="markers",

            name="BUY",

            marker=dict(
                symbol="triangle-up",
                size=10
            )

        )

    )

    signal_fig.add_trace(

        go.Scatter(

            x=sell["date"],

            y=sell["current_close"],

            mode="markers",

            name="SELL",

            marker=dict(
                symbol="triangle-down",
                size=10
            )

        )

    )

    signal_fig.add_trace(

        go.Scatter(

            x=hold["date"],

            y=hold["current_close"],

            mode="markers",

            name="HOLD",

            marker=dict(
                symbol="circle",
                size=6
            )

        )

    )

    signal_fig.update_layout(
        template="plotly_dark",
        height=500,
        title="Cryptocurrency Trading Signals",
        hovermode="x unified"
    )

    st.plotly_chart(
        signal_fig,
        use_container_width=True
    )

with tab4:

    st.subheader(
        "Signal Distribution"
    )

    signal_counts = predictions[
        "signal"
    ].value_counts()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "BUY Signals",
            int(signal_counts.get("BUY", 0))
        )

    with col2:

        st.metric(
            "SELL Signals",
            int(signal_counts.get("SELL", 0))
        )

    with col3:

        st.metric(
            "HOLD Signals",
            int(signal_counts.get("HOLD", 0))
        )

    signal_chart = px.pie(

        names=signal_counts.index,

        values=signal_counts.values,

        hole=0.5,

        title="Trading Signal Distribution",

        template="plotly_dark"

    )

    st.plotly_chart(
        signal_chart,
        use_container_width=True
    )

with tab5:

    st.subheader(
        "Cryptocurrency Historical Dataset"
    )

    st.write(
        f"Selected cryptocurrency: {selected_coin}"
    )

    st.write(
        f"Total records: {len(coin):,}"
    )

    st.dataframe(

        coin.tail(500),

        use_container_width=True,

        hide_index=True

    )

with tab5:

    csv_data = coin.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="Download Processed Dataset",

        data=csv_data,

        file_name=f"{selected_coin}_processed.csv",

        mime="text/csv"

    )

with tab5:

    prediction_csv = predictions.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="Download Prediction Results",

        data=prediction_csv,

        file_name=f"{selected_coin}_predictions.csv",

        mime="text/csv"

    )

st.markdown("---")

footer_html = """
<div style="
    text-align: center;
    color: #94a3b8;
    padding: 25px;
    background-color: #111827;
    border-radius: 12px;
">

<h3 style="color: #818cf8;">
    ₿ Cryptocurrency Market Analytics
</h3>

<p>
    Data Analytics | Machine Learning |
    Technical Analysis | Price Forecasting
</p>

<p>
    Developed using Python, Streamlit,
    Random Forest and XGBoost
</p>

<p style="font-size: 12px;">
    Cryptocurrency Market Intelligence Platform
</p>

</div>
"""

st.markdown(
    footer_html,
    unsafe_allow_html=True
)