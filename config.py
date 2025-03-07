# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Application Settings
APP_CONFIG = {
    'name': 'Stock Analysis Dashboard',
    'version': '1.0.0',
    'debug': os.getenv('DEBUG_MODE', 'False').lower() == 'true',
    'cache_duration': 3600,  # Cache duration in seconds
}

# Server Settings
SERVER_CONFIG = {
    'host': os.getenv('STREAMLIT_SERVER_ADDRESS', 'localhost'),
    'port': int(os.getenv('STREAMLIT_SERVER_PORT', 8501)),
}

# Stock Analysis Settings
STOCK_CONFIG = {
    'default_ticker': 'AAPL',
    'default_period': '1y',
    'max_forecast_days': 30,
    'default_indicators': ['EMA', 'MACD', 'RSI', 'VWAP'],
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    'ema_periods': [9, 20],
    'macd_params': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'rsi_period': 14,
    'vwap_period': 'D',  # D for daily
}

# Chart Configuration
CHART_CONFIG = {
    'default_height': 800,
    'default_width': None,  # Auto-width
    'theme': 'plotly_white',
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ffbb00',
        'background': 'white',
    },
}

# Cache Settings
CACHE_CONFIG = {
    'enabled': True,
    'ttl': 3600,  # Time to live in seconds
    'max_entries': 1000,
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    },
}

# Data Processing Settings
DATA_CONFIG = {
    'date_format': '%Y-%m-%d',
    'time_format': '%H:%M:%S',
    'timezone': 'UTC',
    'decimal_places': 2,
}

# Error Messages
ERROR_MESSAGES = {
    'data_fetch': 'Error fetching data from Yahoo Finance',
    'invalid_symbol': 'Invalid stock symbol provided',
    'date_range': 'Invalid date range selected',
    'calculation': 'Error calculating technical indicators',
    'forecast': 'Error generating forecast',
    'cache': 'Cache error occurred',
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'data_load_time': 2.0,  # seconds
    'calculation_time': 1.0,  # seconds
    'rendering_time': 3.0,  # seconds
    'memory_limit': 1024,  # MB
}

# API Rate Limits
API_LIMITS = {
    'max_requests_per_minute': 60,
    'max_symbols_per_request': 1,
    'max_years_historical': 10,
}

# Feature Flags
FEATURES = {
    'enable_forecasting': True,
    'enable_backtesting': False,
    'enable_alerts': False,
    'enable_portfolio': False,
    'enable_news': True,
}

# Export Settings
EXPORT_CONFIG = {
    'formats': ['csv', 'xlsx', 'json'],
    'default_format': 'csv',
    'max_rows': 1000000,
}

# Visualization Settings
VIZ_CONFIG = {
    'candlestick': {
        'increasing_color': '#26a69a',
        'decreasing_color': '#ef5350',
        'line_width': 1,
    },
    'volume': {
        'increasing_color': '#26a69a50',
        'decreasing_color': '#ef535050',
    },
    'indicators': {
        'line_width': 1.5,
        'opacity': 0.8,
    },
}

def get_config():
    """
    Returns the complete configuration dictionary.
    """
    return {
        'app': APP_CONFIG,
        'server': SERVER_CONFIG,
        'stock': STOCK_CONFIG,
        'technical': TECHNICAL_PARAMS,
        'chart': CHART_CONFIG,
        'cache': CACHE_CONFIG,
        'logging': LOGGING_CONFIG,
        'data': DATA_CONFIG,
        'errors': ERROR_MESSAGES,
        'performance': PERFORMANCE_THRESHOLDS,
        'api_limits': API_LIMITS,
        'features': FEATURES,
        'export': EXPORT_CONFIG,
        'visualization': VIZ_CONFIG,
    }

# Example usage of configuration
if __name__ == "__main__":
    config = get_config()
    print("Application Name:", config['app']['name'])
    print("Default Ticker:", config['stock']['default_ticker'])
    print("Cache Duration:", config['cache']['ttl'])