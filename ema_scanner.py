"""
EMA Scanner - Calculates and displays EMA values for specified periods.
Follows TradingView's EMA calculation method.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sys

def calculate_ema(data, period):
    """Calculate EMA using TradingView method (pandas ewm with span)"""
    if len(data) < period:
        # Not enough data for this EMA period
        return pd.Series([float('nan')] * len(data), index=data.index)

    # Use pandas ewm() with span parameter to match TradingView's EMA
    # span = period for TradingView compatibility
    ema = data['close'].ewm(span=period, adjust=False).mean()

    return ema

def format_date(date_str):
    """Format date to DD-Mon-YYYY"""
    if isinstance(date_str, str):
        date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
    else:
        date_obj = date_str
    return date_obj.strftime("%d-%b-%Y")

def load_instrument_mapping():
    """Load instrument mapping for symbol resolution"""
    mapping_path = os.path.join(os.path.dirname(__file__), 'data_loader', 'config', 'instrument_mapping.json')
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        print("Could not load instrument mapping")
        return None

def get_instrument_key(symbol, instrument_map):
    """Get instrument key for symbol"""
    if symbol.upper() in instrument_map:
        symbol_data = instrument_map[symbol.upper()]
        if isinstance(symbol_data, dict) and 'instrument_key' in symbol_data:
            return symbol_data['instrument_key']
    return None

def fetch_timeframe_data_direct(symbol, timeframe, days_back=300):
    """Fetch data for specific timeframe using direct API calls (fallback method)"""
    instrument_map = load_instrument_mapping()
    if not instrument_map:
        return None

    instrument_key = get_instrument_key(symbol, instrument_map)
    if not instrument_key:
        print(f"Could not find instrument key for {symbol}")
        return None

    # Map timeframe to API parameters
    timeframe_mapping = {
        '5mins': ('minutes', '5'),
        '15mins': ('minutes', '15'),
        '30mins': ('minutes', '30'),
        '1hour': ('minutes', '60'),
        '2hours': ('minutes', '120'),
        '4hours': ('minutes', '240'),
        'daily': ('days', '1'),
        'weekly': ('days', '7'),
        'monthly': ('days', '30'),
        'yearly': ('days', '365')
    }

    if timeframe not in timeframe_mapping:
        print(f"Unsupported timeframe: {timeframe}")
        return None

    unit, interval = timeframe_mapping[timeframe]

    # Adjust date range based on timeframe (API limitations)
    end_date = datetime.now()

    if timeframe in ['5mins', '15mins', '30mins']:
        # Intraday data typically only available for recent periods
        days_back = min(days_back, 30)  # Max 30 days for intraday
    elif timeframe in ['1hour', '2hours', '4hours']:
        days_back = min(days_back, 90)  # Max 90 days for hourly
    elif timeframe == 'daily':
        days_back = min(days_back, 365 * 2)  # Max 2 years for daily
    elif timeframe == 'weekly':
        days_back = min(days_back, 365 * 5)  # Max 5 years for weekly
    elif timeframe == 'monthly':
        days_back = min(days_back, 365 * 10)  # Max 10 years for monthly
    elif timeframe == 'yearly':
        days_back = min(days_back, 365 * 20)  # Max 20 years for yearly

    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Fetching {timeframe} data for {symbol}: {start_str} to {end_str} (DIRECT API)")

    # Build API URL
    safe_key = instrument_key.replace('|', '%7C')
    if unit == 'days':
        url = f"https://api.upstox.com/v3/historical-candle/{safe_key}/days/{interval}/{end_str}/{start_str}"
    else:
        url = f"https://api.upstox.com/v3/historical-candle/{safe_key}/minutes/{interval}/{end_str}/{start_str}"

    headers = {'Accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        data = response.json()

        if 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            print(f"Received {len(candles)} {timeframe} data points for {symbol} (DIRECT API)")

            # Convert to DataFrame
            rows = []
            for candle in candles:
                if len(candle) >= 6:
                    rows.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })

            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            return df
        else:
            print(f"No candles in response for {symbol}")
            return None

    except Exception as e:
        print(f"Error fetching {timeframe} data for {symbol}: {e}")
        return None

def fetch_timeframe_data(symbol, timeframe, days_back=300):
    """Fetch data for specific timeframe using data_loader.py"""
    try:
        # Import data_loader functionality
        sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader'))
        from data_loader import fetch_data_for_symbol

        print(f"Using data_loader.py to fetch {days_back} days of data for {symbol}")

        # Use data_loader to fetch data
        combined_file = fetch_data_for_symbol(symbol, days_back)

        if combined_file and os.path.exists(combined_file):
            print(f"Loading data from: {combined_file}")

            # Read the combined CSV file
            df = pd.read_csv(combined_file)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            print(f"Loaded {len(df)} data points from data_loader.py")

            # Resample data to requested timeframe if different from 5-minute data
            if timeframe != '5mins':
                print(f"Resampling 5-minute data to {timeframe} timeframe")

                # Set timestamp as index for resampling
                df = df.set_index('timestamp')

                # Define resampling rules based on timeframe with market open offset
                # Indian market opens at 9:15 AM, so we offset resampling to start from market open
                market_open_offset = '9h15min'
                
                if timeframe == '15mins':
                    resampled = df.resample('15min', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '30mins':
                    resampled = df.resample('30min', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '1hour':
                    resampled = df.resample('1H', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '2hours':
                    resampled = df.resample('2H', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == '4hours':
                    resampled = df.resample('4H', offset=market_open_offset).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'daily':
                    resampled = df.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'weekly':
                    resampled = df.resample('W').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'monthly':
                    resampled = df.resample('M').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                elif timeframe == 'yearly':
                    resampled = df.resample('Y').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                else:
                    print(f"Unsupported timeframe for resampling: {timeframe}")
                    return None

                # Remove NaN rows (incomplete candles)
                resampled = resampled.dropna()

                # Reset index to get timestamp back as column
                df = resampled.reset_index()

                print(f"Resampled to {len(df)} {timeframe} data points")

            return df
        else:
            print(f"data_loader.py failed to fetch data for {symbol}")
            return None

    except Exception as e:
        print(f"Error using data_loader.py: {e}")
        print("Falling back to direct API fetch...")

        # Fallback to original direct API method
        return fetch_timeframe_data_direct(symbol, timeframe, days_back)

def run_ema_scanner():
    """Main EMA scanner function"""
    print("EMA Scanner Starting...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'ema_config.json')
    print(f"Loading config from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Config loaded successfully")
    except FileNotFoundError:
        print("Config file not found. Using default parameters.")
        # Default configuration when no config file exists
        config = {
            "symbols": ["RELIANCE"],
            "ema_periods": [9, 15, 65, 200],
            "base_timeframe": "15mins",
            "days_to_list": 2,
            "days_fallback_threshold": 200
        }
        print("Using built-in defaults")

    symbols = config['symbols']
    ema_periods = config['ema_periods']
    base_timeframe = config.get('base_timeframe', '15mins')
    days_to_list = config.get('days_to_list', 2)
    days_fallback_threshold = config.get('days_fallback_threshold', 200)

    print(f"Scanning symbols: {symbols}")
    print(f"EMA periods: {ema_periods}")
    print(f"Base timeframe: {base_timeframe}")
    print(f"Days to display: {days_to_list}")

    # Process each symbol
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")

        # Calculate required days based on EMA periods
        # EMA needs more data for longer periods to stabilize
        max_ema_period = max(ema_periods)
        ideal_days = max(max_ema_period * 2, 100)  # At least 2x the longest period or 100 days
        actual_days = min(days_fallback_threshold, ideal_days)

        print(f"EMA needs {ideal_days} days for accuracy, but limited to max {days_fallback_threshold} days")
        print(f"Fetching {actual_days} days of data")

        # Fetch data
        df = fetch_timeframe_data(symbol, base_timeframe, days_back=actual_days)
        if df is None or df.empty:
            print(f"Failed to fetch data for {symbol}")
            continue

        # Apply market open filter AFTER resampling (resampling is done in fetch_timeframe_data)
        # For resampled data, we need to filter based on the timeframe
        if base_timeframe in ['5mins', '15mins']:
            # Intraday data: filter for 9:15 and later
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            market_open_mask = (df['hour'] > 9) | ((df['hour'] == 9) & (df['minute'] >= 15))
        elif base_timeframe == '30mins':
            # 30-minute data: filter for 9:15 and later (first 30-min candle starting at 9:15)
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            market_open_mask = (df['hour'] > 9) | ((df['hour'] == 9) & (df['minute'] >= 15))
        elif base_timeframe == '1hour':
            # 1-hour data: filter for 9:15 and later (first 1-hour candle starting at 9:15)
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            market_open_mask = (df['hour'] > 9) | ((df['hour'] == 9) & (df['minute'] >= 15))
        elif base_timeframe in ['daily', 'weekly', 'monthly', 'yearly']:
            # For daily and higher timeframes, no market open filtering needed
            # These are already aggregated bars for the full trading day
            market_open_mask = pd.Series([True] * len(df))
        else:
            # For other timeframes, use a more general approach
            df['hour'] = df['timestamp'].dt.hour
            market_open_mask = df['hour'] >= 9

        df = df[market_open_mask].copy()
        df = df.drop(['hour', 'minute'], axis=1, errors='ignore')

        print(f"Loaded {len(df)} {base_timeframe} data points for {symbol}")

        # Calculate EMA for each period
        ema_results = {}
        for period in ema_periods:
            ema_series = calculate_ema(df, period)
            ema_results[f'ema_{period}'] = ema_series
            print(f"Calculated EMA({period}) using TradingView method")

        # Add EMA columns to dataframe
        for ema_col, ema_series in ema_results.items():
            df[ema_col] = ema_series

        # Filter data for the specified number of days
        if not df.empty:
            latest_date = df['timestamp'].max().date()
            start_date = latest_date - timedelta(days=days_to_list - 1)
            df = df[df['timestamp'].dt.date >= start_date]

            print(f"Showing data for last {days_to_list} days")

        # Save data with EMA calculations to CSV file
        os.makedirs(f"data/{symbol}", exist_ok=True)
        csv_path = f"data/{symbol}/{symbol}_ema_data.csv"

        # Replace NaN values with None for better CSV handling
        csv_df = df.copy()
        for col in csv_df.columns:
            if csv_df[col].dtype in ['float64', 'int64']:
                csv_df[col] = csv_df[col].fillna('N/A')

        csv_df.to_csv(csv_path, index=False)
        print(f"Data with EMA calculations saved to: {csv_path}")
        print(f"Final dataframe shape: {df.shape}")
        print(f"Final dataframe columns: {list(df.columns)}")

        # Display results in table format
        print(f"\n{'='*100}")
        print(f"EMA ANALYSIS - {symbol.upper()}")
        print(f"{'='*100}")

        # Create headers
        headers = ['Time', 'Symbol', 'CMP'] + [f'EMA{period}' for period in ema_periods]

        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)

        # Update widths based on data
        for _, row in df.iterrows():
            for header in headers:
                if header == 'Time':
                    dt = row['timestamp']
                    if pd.isna(dt):
                        value = 'N/A'
                    else:
                        hour = dt.hour if dt.hour != 0 else 12
                        time_str = f"{hour}:{dt.minute:02d}:{dt.second:02d}"
                        date_str = dt.strftime("%d-%m-%Y")
                        value = f"{date_str} {time_str}"
                elif header == 'Symbol':
                    value = symbol
                elif header == 'CMP':
                    close_val = row['close']
                    if pd.isna(close_val) or str(close_val).lower() in ['nat', 'nan']:
                        value = 'N/A'
                    else:
                        try:
                            value = f"{float(close_val):.2f}"
                        except (ValueError, TypeError):
                            value = 'N/A'
                else:
                    period = int(header.replace('EMA', ''))
                    ema_value = row[f'ema_{period}']
                    if pd.isna(ema_value) or ema_value != ema_value:
                        value = 'N/A'
                    else:
                        value = f"{ema_value:.2f}"
                col_widths[header] = max(col_widths[header], len(str(value)))

        # Print table header
        header_row = ' | '.join(header.ljust(col_widths[header]) for header in headers)
        print(header_row)
        print('-' * len(header_row))

        # Print data rows
        for _, row in df.iterrows():
            row_data = []
            for header in headers:
                if header == 'Time':
                    dt = row['timestamp']
                    if pd.isna(dt):
                        value = 'N/A'
                    else:
                        hour = dt.hour if dt.hour != 0 else 12
                        time_str = f"{hour}:{dt.minute:02d}:{dt.second:02d}"
                        date_str = dt.strftime("%d-%m-%Y")
                        value = f"{date_str} {time_str}"
                elif header == 'Symbol':
                    value = symbol
                elif header == 'CMP':
                    close_val = row['close']
                    if pd.isna(close_val) or str(close_val).lower() in ['nat', 'nan']:
                        value = 'N/A'
                    else:
                        try:
                            value = f"{float(close_val):.2f}"
                        except (ValueError, TypeError):
                            value = 'N/A'
                else:
                    period = int(header.replace('EMA', ''))
                    ema_value = row[f'ema_{period}']
                    if pd.isna(ema_value) or ema_value != ema_value:
                        value = 'N/A'
                    else:
                        value = f"{ema_value:.2f}"
                row_data.append(str(value).ljust(col_widths[header]))
            print(' | '.join(row_data))

        print(f"{'='*100}\n")

    print("EMA Scanner completed successfully!")

if __name__ == "__main__":
    run_ema_scanner()