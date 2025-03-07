import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Stock Analysis", layout="wide")

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    # Reset index to handle date
    df = df.copy()  # Create a copy to avoid modifying original
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    volume_price = typical_price * df['Volume']
    cumulative_volume = df['Volume'].cumsum()
    cumulative_volume_price = volume_price.cumsum()
    return cumulative_volume_price / cumulative_volume

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signals(df):
    """Generate buy and sell signals"""
    df = df.copy()
    
    # MACD Line crosses above Signal Line (Buy)
    df['Buy_Signal'] = ((df['MACD'] > df['Signal']) & 
                        (df['MACD'].shift(1) <= df['Signal'].shift(1)) & 
                        (df['RSI'] < 70)).astype(int)
    
    # MACD Line crosses below Signal Line (Sell)
    df['Sell_Signal'] = ((df['MACD'] < df['Signal']) & 
                         (df['MACD'].shift(1) >= df['Signal'].shift(1)) & 
                         (df['RSI'] > 30)).astype(int)
    
    return df

def calculate_heikin_ashi(df):
    """Calculate Heikin-Ashi candlestick data"""
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize ha_open series with the first value
    ha_open = pd.Series([(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2], index=[df.index[0]])
    
    # Calculate subsequent ha_open values
    for i in range(1, len(df)):
        next_value = pd.Series(
            [(ha_open.iloc[-1] + ha_close.iloc[i-1]) / 2],
            index=[df.index[i]]
        )
        ha_open = pd.concat([ha_open, next_value])
    
    ha_high = pd.concat([df['High'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['Low'], ha_open, ha_close], axis=1).min(axis=1)
    
    return pd.DataFrame({
        'HA_Open': ha_open,
        'HA_High': ha_high,
        'HA_Low': ha_low,
        'HA_Close': ha_close
    }, index=df.index)

def forecast_sarima(data, periods=30):
    """Generate SARIMA forecast"""
    model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.forecast(steps=periods)
    return forecast

@st.cache_data
def load_stock_data(symbol, start, end):
    try:
        # Validate inputs
        if not symbol:
            st.error("Please enter a stock symbol")
            return None
            
        # Download data with explicit parameters and debug info
        st.write(f"Attempting to download data for {symbol} from {start} to {end}")
        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            progress=False
        )
        
        # Fix MultiIndex columns by selecting the first level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Debug information
        st.write(f"Downloaded data shape: {df.shape if df is not None else 'None'}")
        
        # Check if data is empty or None
        if df is None or df.empty:
            st.error(f"No data found for symbol '{symbol}' in the selected date range")
            return None
            
        # Check if we have enough data for calculations
        if len(df) < 26:  # Minimum length needed for MACD
            st.error(f"Not enough data points for '{symbol}'. Please select a longer date range")
            return None

        # Reset index to handle date
        df = df.copy()  # Create a copy to avoid modifying original
        
        # Calculate technical indicators
        df['EMA9'] = calculate_ema(df, 9)
        df['EMA20'] = calculate_ema(df, 20)
        df['VWAP'] = calculate_vwap(df)
        macd, signal = calculate_macd(df)
        df['MACD'] = macd
        df['Signal'] = signal
        df['RSI'] = calculate_rsi(df)
        
        # Add Heikin-Ashi data
        ha_df = calculate_heikin_ashi(df)
        df = pd.concat([df, ha_df], axis=1)
        
        # Generate buy/sell signals
        df = generate_signals(df)
        
        # Verify all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'EMA9', 'EMA20', 'VWAP', 'MACD', 'Signal', 'RSI']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data for '{symbol}': {str(e)}")
        st.write("Debug info:", e.__class__.__name__)
        st.write("Error details:", str(e))
        if isinstance(e, ValueError):
            st.write("Please check if the stock symbol is correct")
        return None

# Main content
st.title('Advanced Stock Analysis Dashboard')

# Date inputs with validation
today = datetime.today().date()  # Convert to date
min_date = today - timedelta(days=365*10)  # 10 years ago
max_date = today - timedelta(days=1)  # Yesterday

# Sidebar inputs with additional validation
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input("Stock Symbol", "AAPL").upper().strip()

# Validate ticker
if not ticker:
    st.error("Please enter a stock symbol")
    st.stop()

# First date input
start_date = st.sidebar.date_input(
    "Start Date", 
    value=today - timedelta(days=365),  # Default to 1 year ago
    min_value=min_date,
    max_value=today - timedelta(days=2)  # Must be at least 2 days before today
)

# Second date input
end_date = st.sidebar.date_input(
    "End Date", 
    value=today - timedelta(days=1),  # Default to yesterday
    min_value=start_date,
    max_value=today - timedelta(days=1)  # Can't be later than yesterday
)

# Validate dates
if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# Load data with progress indicator
with st.spinner(f'Loading data for {ticker}...'):
    try:
        # Convert dates to string format for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Debug information
        st.write(f"Attempting to download {ticker} data from {start_str} to {end_str}")
        
        # Download data
        df = yf.download(ticker, start=start_str, end=end_str, progress=False)
        
        # Fix MultiIndex columns by selecting the first level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Debug information
        st.write(f"Data shape: {df.shape}")
        st.write(f"Columns: {df.columns.tolist()}")
        
        if df.empty:
            st.error(f"No data available for {ticker} in the selected date range")
            st.stop()
            
        if len(df) < 26:  # Minimum required for technical indicators
            st.error(f"Not enough data points for {ticker}. Please select a longer date range")
            st.stop()
            
        # Create a copy to avoid modification warnings
        df = df.copy()
        
        # Calculate indicators with error checking
        try:
            df['EMA9'] = calculate_ema(df, 9)
            df['EMA20'] = calculate_ema(df, 20)
            df['VWAP'] = calculate_vwap(df)
            macd, signal = calculate_macd(df)
            df['MACD'] = macd
            df['Signal'] = signal
            df['RSI'] = calculate_rsi(df)
            
            # Add Heikin-Ashi data
            ha_df = calculate_heikin_ashi(df)
            df = pd.concat([df, ha_df], axis=1)
            
            # Generate signals
            df = generate_signals(df)
            
        except KeyError as ke:
            st.error(f"Error calculating indicators: Missing column {ke}")
            st.write("Available columns:", df.columns.tolist())
            st.stop()
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            st.stop()
        
        # Verify we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'EMA9', 'EMA20', 'VWAP', 'MACD', 'Signal', 'RSI']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.write("Available columns:", df.columns.tolist())
            st.stop()
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Debug info:", e.__class__.__name__)
        st.write("Full error:", str(e))
        st.stop()

if df is not None:
    # Add chart type selector
    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        ["Regular Candlestick", "Heikin-Ashi"]
    )
    
    # Add forecast option
    show_forecast = st.sidebar.checkbox("Show SARIMA Forecast")
    if show_forecast:
        forecast_periods = st.sidebar.slider("Forecast Periods", 5, 60, 30)
        forecast = forecast_sarima(df, periods=forecast_periods)

    # Create subplots
    fig = make_subplots(rows=4, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.15, 0.15, 0.15])

    # Main price chart (row 1)
    if chart_type == "Regular Candlestick":
        candlestick_data = dict(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )
    else:  # Heikin-Ashi
        candlestick_data = dict(
            x=df.index,
            open=df['HA_Open'],
            high=df['HA_High'],
            low=df['HA_Low'],
            close=df['HA_Close'],
            name='Heikin-Ashi'
        )

    fig.add_trace(go.Candlestick(
        **candlestick_data,
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # Add SARIMA forecast if enabled
    if show_forecast:
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=df.index[-1], periods=len(forecast)+1)[1:],
            y=forecast,
            name='SARIMA Forecast',
            line=dict(color='orange', dash='dash'),
            hovertemplate='Forecast: %{y:.2f}<extra></extra>',
            showlegend=True,  # Explicitly set showlegend to True
            legendgroup='forecast'  # Add a legend group
        ), row=1, col=1)

    # Get real-time current price - Modified to handle errors gracefully
    try:
        current_data = yf.Ticker(ticker)
        current_price = current_data.info.get('regularMarketPrice')
        if current_price is None:
            current_price = df['Close'].iloc[-1]  # Use last closing price if real-time price unavailable
        
        # Create a line for current price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=[current_price] * len(df.index),
            name='Current Price',
            line=dict(color='black', width=1, dash='solid'),
            hovertemplate=f'Current Price: ${current_price:.2f}<extra></extra>'
        ), row=1, col=1)

        # Add annotation for current price
        fig.add_annotation(
            x=df.index[-1],
            y=current_price,
            text=f'${current_price:.2f}',
            showarrow=False,
            yshift=10,
            xshift=50,
            font=dict(size=12, color='black'),
            row=1, col=1
        )
    except Exception as e:
        st.warning(f"Could not fetch real-time price: {str(e)}")
        current_price = df['Close'].iloc[-1]  # Use last closing price as fallback

    # Add EMAs
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA9'],
        name='9 EMA',
        line=dict(color='blue', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA20'],
        name='20 EMA',
        line=dict(color='orange', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VWAP'],
        name='VWAP',
        line=dict(color='purple', width=1)
    ), row=1, col=1)

    # Get earnings dates
    try:
        earnings_dates = current_data.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            # Convert timezones to UTC for consistent comparison
            earnings_dates.index = earnings_dates.index.tz_localize(None)
            df.index = df.index.tz_localize(None)
            
            # Filter earnings dates within our date range
            earnings_dates = earnings_dates[
                (earnings_dates.index >= df.index[0]) & 
                (earnings_dates.index <= df.index[-1])
            ]
            
            # Add earnings markers
            if not earnings_dates.empty:
                fig.add_trace(go.Scatter(
                    x=earnings_dates.index,
                    y=[df['High'].max()] * len(earnings_dates),  # Place markers at top of chart
                    mode='markers+text',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='gold',
                        line=dict(color='black', width=1)
                    ),
                    text=['📊'] * len(earnings_dates),  # Earnings emoji
                    textposition='top center',
                    name='Earnings Dates',
                    hovertemplate='Earnings Date: %{x}<extra></extra>'
                ), row=1, col=1)

                # Add annotations for earnings dates
                for date in earnings_dates.index:
                    fig.add_annotation(
                        x=date,
                        y=df['High'].max(),
                        text='Earnings',
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='gold',
                        font=dict(size=10, color='black'),
                        yshift=20,
                        row=1, col=1
                    )
    except Exception as e:
        st.warning(f"Could not load earnings dates: {str(e)}")

    # Add buy signals
    buy_mask = df['Buy_Signal'] == 1
    if buy_mask.any():
        buy_signals = df[buy_mask]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Low'] * 0.99,
            mode='markers+text',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='green',
                line=dict(width=2, color='darkgreen')
            ),
            text='BUY',
            textposition='bottom center',
            textfont=dict(size=12, color='green'),
            name='Buy Signal'
        ), row=1, col=1)

    # Add sell signals
    sell_mask = df['Sell_Signal'] == 1
    if sell_mask.any():
        sell_signals = df[sell_mask]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['High'] * 1.01,
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text='SELL',
            textposition='top center',
            textfont=dict(size=12, color='red'),
            name='Sell Signal'
        ), row=1, col=1)

    # Update the main chart y-axis
    fig.update_yaxes(title_text="Price", row=1, col=1)

    # Update layout to ensure legend is visible
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title="Price",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor='rgba(255, 255, 255, 0.8)'  # Semi-transparent white background
        )
    )

    # Make sure candlesticks are visible
    fig.update_layout(
        yaxis=dict(
            autorange=True,
            fixedrange=False
        )
    )

    # Add Volume (row 2)
    colors = ['green' if close > open else 'red' 
             for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(
            color=colors,
            line=dict(color=colors, width=1)
        )
    ), row=2, col=1)

    # Add volume moving average
    volume_ma = df['Volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=volume_ma,
        name='Volume MA (20)',
        line=dict(color='blue', width=1)
    ), row=2, col=1)

    # MACD (row 3)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=1)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        name='Signal',
        line=dict(color='orange', width=1)
    ), row=3, col=1)

    # Add MACD histogram
    colors = ['red' if val < 0 else 'green' for val in df['MACD'] - df['Signal']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD'] - df['Signal'],
        marker_color=colors,
        name='MACD Histogram'
    ), row=3, col=1)

    # Add buy signals on MACD
    buy_mask = df['Buy_Signal'] == 1
    if buy_mask.any():
        buy_signals = df[buy_mask]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['MACD'],
            mode='markers+text',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=2, color='darkgreen')
            ),
            text='BUY',
            textposition='bottom center',
            textfont=dict(size=10, color='green'),
            name='MACD Buy',
            showlegend=False
        ), row=3, col=1)

    # Add sell signals on MACD
    sell_mask = df['Sell_Signal'] == 1
    if sell_mask.any():
        sell_signals = df[sell_mask]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['MACD'],
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text='SELL',
            textposition='top center',
            textfont=dict(size=10, color='red'),
            name='MACD Sell',
            showlegend=False
        ), row=3, col=1)

    # Add annotations for crossovers
    for idx, row in buy_signals.iterrows():
        fig.add_annotation(
            x=idx,
            y=row['MACD'],
            text='↑',
            showarrow=False,
            font=dict(size=14, color='green'),
            row=3, col=1
        )

    for idx, row in sell_signals.iterrows():
        fig.add_annotation(
            x=idx,
            y=row['MACD'],
            text='↓',
            showarrow=False,
            font=dict(size=14, color='red'),
            row=3, col=1
        )

    # RSI (row 4)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple', width=1)
    ), row=4, col=1)

    # Add RSI buy signals
    if buy_mask.any():
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['RSI'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=8,
                color='green',
                line=dict(width=1, color='darkgreen')
            ),
            name='RSI Buy',
            showlegend=False
        ), row=4, col=1)

    # Add RSI sell signals
    if sell_mask.any():
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['RSI'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=8,
                color='red',
                line=dict(width=1, color='darkred')
            ),
            name='RSI Sell',
            showlegend=False
        ), row=4, col=1)

    # Add RSI levels
    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=70,
        y1=70,
        line=dict(color='red', width=1, dash='dash'),
        row=4,
        col=1
    )

    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=30,
        y1=30,
        line=dict(color='green', width=1, dash='dash'),
        row=4,
        col=1
    )

    # Update RSI axis range
    fig.update_yaxes(range=[0, 100], row=4, col=1)

    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis_title="Date",
        height=1000,  # Increased height
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update y-axes grid lines
    fig.update_yaxes(
        gridcolor='lightgrey',
        gridwidth=0.1,
        zerolinecolor='lightgrey',
        zerolinewidth=1
    )

    # Update x-axes grid lines
    fig.update_xaxes(
        gridcolor='lightgrey',
        gridwidth=0.1,
        zerolinecolor='lightgrey',
        zerolinewidth=1
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Statistics and Analysis
    st.subheader('Technical Indicators Summary')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_rsi = float(df['RSI'].iloc[-1])
        st.metric("RSI", f"{current_rsi:.2f}", 
                 "Overbought > 70, Oversold < 30")
    
    with col2:
        current_macd = float(df['MACD'].iloc[-1])
        current_signal = float(df['Signal'].iloc[-1])
        macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
        st.metric("MACD Signal", macd_signal)
    
    with col3:
        current_close = float(df['Close'].iloc[-1])
        current_ema = float(df['EMA20'].iloc[-1])
        trend = "Bullish" if current_close > current_ema else "Bearish"
        st.metric("Trend (20 EMA)", trend)

    # Add strategy performance metrics
    st.subheader('Strategy Performance Metrics')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Buy Signals", len(df[df['Buy_Signal'] == 1]))
    
    with col2:
        st.metric("Total Sell Signals", len(df[df['Sell_Signal'] == 1]))
    
    with col3:
        vwap_position = "Above VWAP" if df['Close'].iloc[-1] > df['VWAP'].iloc[-1] else "Below VWAP"
        st.metric("VWAP Position", vwap_position)
    
    with col4:
        ha_trend = "Bullish" if df['HA_Close'].iloc[-1] > df['HA_Open'].iloc[-1] else "Bearish"
        st.metric("Heikin-Ashi Trend", ha_trend)

    # Export data option
    if st.button('Export Data to CSV'):
        csv = df.to_csv()
        st.download_button(
            label="Download Data",
            data=csv,
            file_name=f'{ticker}_technical_analysis.csv',
            mime='text/csv'
        )
else:
    st.error("No data available for the selected stock symbol and date range.")