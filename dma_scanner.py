"""
DMA Scanner - Calculates and displays DMA (Displaced Moving Average) values for specified periods.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import sys
import pytz  # Added for timezone handling
import logging
import time
import traceback
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_dma(data, period, displacement=1):
    """Calculate DMA (Displaced Moving Average) with backward displacement"""
    if len(data) < period + displacement:
        # Not enough data for this DMA period
        return pd.Series([float('nan')] * len(data), index=data.index)

    # Calculate simple moving average
    sma = data['close'].rolling(window=period).mean()

    # Apply backward displacement (shift forward in time by displacement periods)
    dma = sma.shift(displacement)

    return dma

def report_error(category: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a structured error report with categorization and diagnostics
    
    Args:
        category: Error category (data, calculation, configuration, api)
        message: Human-readable error message
        details: Additional error details and diagnostic information
        
    Returns:
        Dictionary with structured error information
    """
    error_data = {
        "category": category,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }
    
    # Add system information to help with debugging
    error_data["system_info"] = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "memory_available": "Unknown"  # Could add psutil.virtual_memory() if available
    }
    
    # Log the error with appropriate level based on category
    if category == "critical":
        logger.critical(f"{category.upper()}: {message}")
    elif category in ["data", "calculation"]:
        logger.error(f"{category.upper()}: {message}")
    else:
        logger.warning(f"{category.upper()}: {message}")
        
    return error_data

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
    logger.info(f"Attempting direct API fetch for {symbol} ({timeframe}, {days_back} days)")
    
    instrument_map = load_instrument_mapping()
    if not instrument_map:
        error = report_error("data", "Failed to load instrument mapping", 
                            {"symbol": symbol, "timeframe": timeframe})
        logger.error(f"Could not load instrument mapping")
        return None

    instrument_key = get_instrument_key(symbol, instrument_map)
    if not instrument_key:
        error = report_error("data", f"Could not find instrument key for {symbol}",
                            {"symbol": symbol, "available_symbols": list(instrument_map.keys())[:10]})
        logger.error(f"Could not find instrument key for {symbol}")
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
        error = report_error("configuration", f"Unsupported timeframe: {timeframe}",
                           {"timeframe": timeframe, "supported_timeframes": list(timeframe_mapping.keys())})
        logger.error(f"Unsupported timeframe: {timeframe}")
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

    logger.info(f"Fetching {timeframe} data for {symbol}: {start_str} to {end_str} (DIRECT API)")

    # Build API URL
    safe_key = instrument_key.replace('|', '%7C')
    if unit == 'days':
        url = f"https://api.upstox.com/v3/historical-candle/{safe_key}/days/{interval}/{end_str}/{start_str}"
    else:
        url = f"https://api.upstox.com/v3/historical-candle/{safe_key}/minutes/{interval}/{end_str}/{start_str}"

    headers = {'Accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            error = report_error("api", f"API returned non-200 status code: {response.status_code}",
                               {"url": url, "status_code": response.status_code, "response": response.text[:500]})
            logger.error(f"API error: Status {response.status_code}")
            return None
            
        data = response.json()

        if 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            logger.info(f"Received {len(candles)} {timeframe} data points for {symbol} (DIRECT API)")

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
            
            if df.empty:
                error = report_error("data", f"Empty DataFrame after processing candles for {symbol}",
                                   {"candles_count": len(candles), "timeframe": timeframe})
                logger.error(f"Empty DataFrame after processing candles")
                return None
                
            # Validate OHLC data
            invalid_data = (df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0)
            if invalid_data.any():
                invalid_count = invalid_data.sum()
                if invalid_count > len(df) * 0.1:  # If more than 10% is invalid
                    error = report_error("data", f"High percentage of invalid OHLC data: {invalid_count} records",
                                       {"invalid_count": int(invalid_count), "total_count": len(df)})
                    logger.warning(f"High percentage of invalid OHLC data: {invalid_count} records")
                else:
                    logger.warning(f"Found {invalid_count} invalid OHLC records, will be filtered out")
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            return df
        else:
            error = report_error("data", f"No candles in response for {symbol}",
                               {"api_response": str(data)[:500]})
            logger.error(f"No candles in response for {symbol}")
            return None

    except requests.exceptions.Timeout:
        error = report_error("api", f"API request timed out for {symbol}",
                           {"url": url, "timeout": 30})
        logger.error(f"API request timed out for {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        error = report_error("api", f"API request failed for {symbol}: {str(e)}",
                           {"url": url, "error": str(e)})
        logger.error(f"API request failed for {symbol}: {e}")
        return None
    except ValueError as e:
        error = report_error("data", f"Invalid JSON response for {symbol}: {str(e)}",
                           {"url": url, "error": str(e)})
        logger.error(f"Invalid JSON response for {symbol}: {e}")
        return None
    except Exception as e:
        error = report_error("critical", f"Unexpected error fetching {timeframe} data for {symbol}: {str(e)}",
                           {"url": url, "error": str(e), "traceback": traceback.format_exc()})
        logger.exception(f"Error fetching {timeframe} data for {symbol}: {e}")
        return None

def fetch_timeframe_data(symbol, timeframe, days_back=300):
    """Fetch data for specific timeframe with improved accuracy"""
    try:
        # Import data_loader functionality
        sys.path.append(os.path.join(os.path.dirname(__file__), 'data_loader'))
        from data_loader import fetch_data_for_symbol

        logger.info(f"Using data_loader.py to fetch {days_back} days of data for {symbol}")

        # Use data_loader to fetch 5-minute data
        combined_file = fetch_data_for_symbol(symbol, days_back)

        if combined_file and os.path.exists(combined_file):
            logger.info(f"Loading data from: {combined_file}")

            # Read the combined CSV file
            try:
                df = pd.read_csv(combined_file)
            except Exception as e:
                error = report_error("data", f"Failed to read CSV file for {symbol}: {str(e)}",
                                  {"file": combined_file, "error": str(e)})
                logger.error(f"Failed to read CSV file: {combined_file}")
                return None
                
            if df.empty:
                error = report_error("data", f"Empty dataframe loaded from {combined_file}",
                                  {"file": combined_file, "symbol": symbol})
                logger.error(f"Empty dataframe loaded from {combined_file}")
                return None

            # Convert timestamp to datetime and localize to IST
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
                df = df.sort_values('timestamp')
            except Exception as e:
                error = report_error("data", f"Failed to process timestamps for {symbol}: {str(e)}",
                                  {"error": str(e), "example_timestamps": str(df['timestamp'].head(3).tolist())})
                logger.error(f"Failed to process timestamps: {e}")
                return None

            logger.info(f"Loaded {len(df)} data points from data_loader.py")

            # Apply market open filter BEFORE resampling for accuracy
            if timeframe in ['5mins', '15mins', '30mins', '1hour', '4hours']:
                # Filter for market hours: 9:15 to 15:30 IST
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                market_open_mask = (
                    ((df['hour'] == 9) & (df['minute'] >= 15)) |
                    ((df['hour'] > 9) & (df['hour'] < 15)) |
                    ((df['hour'] == 15) & (df['minute'] <= 30))
                )
                df = df[market_open_mask].copy()
                df = df.drop(['hour', 'minute'], axis=1, errors='ignore')
                logger.info(f"Filtered to {len(df)} market hours data points")
                
                if df.empty:
                    error = report_error("data", f"No data points during market hours for {symbol}",
                                      {"symbol": symbol, "timeframe": timeframe})
                    logger.error(f"No data points during market hours")
                    return None
            elif timeframe in ['daily', 'weekly', 'monthly']:
                # For daily and higher, filter to trading days (exclude weekends and holidays if possible)
                df['date'] = df['timestamp'].dt.date
                # Basic weekend filter (could be enhanced with holiday API)
                df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
                trading_day_mask = df['weekday'] < 5  # Monday to Friday
                df = df[trading_day_mask].copy()
                df = df.drop(['date', 'weekday'], axis=1, errors='ignore')
                logger.info(f"Filtered to {len(df)} trading day data points")
                
                if df.empty:
                    error = report_error("data", f"No trading day data points for {symbol}",
                                      {"symbol": symbol, "timeframe": timeframe})
                    logger.error(f"No trading day data points")
                    return None

            # Resample to requested timeframe
            if timeframe != '5mins':
                logger.info(f"Resampling to {timeframe} timeframe")
                if timeframe == '15mins':
                    rule = '15min'
                elif timeframe == '30mins':
                    rule = '30min'
                elif timeframe == '1hour':
                    rule = 'h'
                elif timeframe == '4hours':
                    rule = '4h'
                elif timeframe == 'daily':
                    rule = 'D'
                elif timeframe == 'weekly':
                    rule = 'W'
                elif timeframe == 'monthly':
                    rule = 'M'
                else:
                    error = report_error("configuration", f"Unsupported timeframe for resampling: {timeframe}",
                                     {"timeframe": timeframe, "supported": ['5mins', '15mins', '30mins', '1hour', '4hours', 'daily', 'weekly', 'monthly']})
                    logger.error(f"Unsupported timeframe: {timeframe}")
                    return None

                try:
                    # Resample with proper OHLC aggregation
                    df = df.set_index('timestamp').resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna().reset_index()
                except Exception as e:
                    error = report_error("calculation", f"Failed to resample data for {symbol}: {str(e)}",
                                     {"timeframe": timeframe, "rule": rule, "error": str(e)})
                    logger.error(f"Failed to resample data: {e}")
                    return None

                logger.info(f"Resampled to {len(df)} {timeframe} data points")

            # Validate data integrity
            if df.empty:
                error = report_error("data", "No data after filtering/resampling",
                                 {"symbol": symbol, "timeframe": timeframe})
                logger.error("No data after filtering/resampling")
                return None

            # Check for basic data quality
            invalid_mask = (df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0)
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                logger.warning(f"Warning: {invalid_count} invalid OHLC records found, removing")
                df = df[~invalid_mask]
                
                if df.empty:
                    error = report_error("data", "All records were invalid after OHLC validation",
                                     {"symbol": symbol, "timeframe": timeframe, "invalid_count": int(invalid_count)})
                    logger.error("All records were invalid after OHLC validation")
                    return None

            # Remove duplicates
            df = df.drop_duplicates(subset='timestamp')

            logger.info(f"Final data: {len(df)} records")
            return df

        else:
            error = report_error("data", "Failed to fetch data from data_loader.py",
                             {"symbol": symbol, "days_back": days_back})
            logger.error("Failed to fetch data from data_loader.py")
            return None

    except ImportError as e:
        error = report_error("configuration", f"Failed to import data_loader: {str(e)}",
                         {"error": str(e), "traceback": traceback.format_exc()})
        logger.error(f"Failed to import data_loader: {e}")
        return None
    except Exception as e:
        error = report_error("critical", f"Unexpected error in fetch_timeframe_data: {str(e)}",
                         {"error": str(e), "traceback": traceback.format_exc()})
        logger.exception(f"Error in fetch_timeframe_data: {e}")
        return None

def run_dma_scanner():
    """Main DMA scanner function"""
    start_time = time.time()
    logger.info("DMA Scanner Starting...")
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Python executable: {sys.executable}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'dma_config.json')
    logger.info(f"Loading config from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info("Config loaded successfully")
    except FileNotFoundError:
        logger.warning("Config file not found. Using default parameters.")
        # Default configuration when no config file exists
        config = {
            "dma_periods": [10, 20, 50],
            "base_timeframe": "1hour",
            "days_to_list": 2,
            "days_fallback_threshold": 1600,
            "displacement": 1
        }
        logger.info("Using built-in defaults")

    # Load centralized symbols if not present in config (for standalone execution)
    if 'symbols' not in config:
        try:
            symbols_config_path = os.path.join(os.path.dirname(__file__), 'config', 'symbols.config.json')
            with open(symbols_config_path, 'r') as f:
                symbols_data = json.load(f)
            symbols = symbols_data.get('symbols', ["RELIANCE"])
            logger.info(f"Loaded centralized symbols: {len(symbols)} symbols")
        except FileNotFoundError:
            symbols = ["RELIANCE"]
            logger.warning("Centralized symbols file not found, using default symbol")
    else:
        symbols = config['symbols']
    dma_periods = config['dma_periods']
    base_timeframe = config.get('base_timeframe', '15mins')
    days_to_list = config.get('days_to_list', 2)
    days_fallback_threshold = config.get('days_fallback_threshold', 200)
    displacement = config.get('displacement', 1)

    logger.info(f"Scanning symbols: {symbols}")
    logger.info(f"DMA periods: {dma_periods}")
    logger.info(f"Base timeframe: {base_timeframe}")
    logger.info(f"Days to display: {days_to_list}")
    logger.info(f"Displacement: {displacement}")

    # Process each symbol
    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")

        # Calculate required days based on DMA periods
        # DMA needs more data for longer periods to stabilize
        max_dma_period = max(dma_periods)
        ideal_days = max(max_dma_period * 3, 200)  # At least 3x the longest period or 200 days for better accuracy
        actual_days = min(days_fallback_threshold, ideal_days)

        logger.info(f"DMA needs {ideal_days} days for accuracy, but limited to max {days_fallback_threshold} days")
        logger.info(f"NOTE: This assumes proper historical data fetching. If data spans < {ideal_days} days,")
        logger.info(f"      the data_loader may not be configured correctly for historical data retrieval.")
        logger.info(f"Fetching {actual_days} days of data")

        # Fetch data
        df = fetch_timeframe_data(symbol, base_timeframe, days_back=actual_days)
        if df is None or df.empty:
            error = report_error("data", f"Failed to fetch ANY data for {symbol}",
                            {"symbol": symbol, "timeframe": base_timeframe, "days_back": actual_days})
            logger.error(f"CRITICAL: Failed to fetch ANY data for {symbol}")
            logger.error(f"SKIPPING {symbol} - No data available")
            continue

        logger.info(f"Fetched {len(df)} {base_timeframe} data points for {symbol}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # STRICT DATA SUFFICIENCY CHECK
        insufficient_periods = []
        sufficient_periods = []

        for period in dma_periods:
            required_points = period + displacement + 20  # period + displacement + generous buffer
            if len(df) < required_points:
                insufficient_periods.append(f"DMA{period} (needs {required_points}, has {len(df)})")
            else:
                sufficient_periods.append(f"DMA{period}")

        if insufficient_periods:
            error = report_error("data", "Data insufficiency detected",
                           {"symbol": symbol, 
                            "insufficient_periods": insufficient_periods,
                            "sufficient_periods": sufficient_periods,
                            "data_points": len(df),
                            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}"})
            
            logger.error("CRITICAL DATA INSUFFICIENCY DETECTED!")
            logger.error(f"Insufficient data for: {', '.join(insufficient_periods)}")
            logger.error(f"Sufficient data for: {', '.join(sufficient_periods)}" if sufficient_periods else "None sufficient")
            logger.error("SOLUTION: The data_loader is not fetching historical data properly.")
            logger.error(f"         Current data has only {len(df)} points, but DMA calculations need:")
            for period in dma_periods:
                required_points = period + displacement + 20
                logger.error(f"         - DMA{period}: {required_points} data points minimum")
            logger.error(f"         RECOMMENDATION: Check data_loader configuration and ensure it's fetching")
            logger.error(f"         historical data, not future data. Current data spans only {len(df)} points")
            logger.error(f"         from {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.error(f"SKIPPING CALCULATIONS FOR {symbol} - INACCURATE RESULTS WOULD BE GENERATED")
            continue  # Skip to next symbol

        logger.info("DATA SUFFICIENCY CHECK PASSED - Proceeding with calculations...")

        # Calculate DMA for each period
        dma_results = {}
        for period in dma_periods:
            try:
                dma_series = calculate_dma(df, period, displacement)
                dma_results[f'dma_{period}'] = dma_series
                logger.info(f"Calculated DMA({period}) with displacement {displacement}")
            except Exception as e:
                error = report_error("calculation", f"Failed to calculate DMA({period}) for {symbol}: {str(e)}",
                               {"symbol": symbol, "period": period, "displacement": displacement, "error": str(e)})
                logger.error(f"Failed to calculate DMA({period}): {e}")
                # Continue with other periods

        # Check if any DMA calculations succeeded
        if not dma_results:
            error = report_error("calculation", f"All DMA calculations failed for {symbol}",
                           {"symbol": symbol, "periods": dma_periods})
            logger.error(f"All DMA calculations failed for {symbol}")
            continue

        # Add DMA columns to dataframe
        for dma_col, dma_series in dma_results.items():
            df[dma_col] = dma_series

        # FILTER TO VALID ROWS ONLY (NO N/A IN TABLE)
        # Find the latest index where ANY DMA value is NaN
        max_nan_index = -1
        for dma_col in dma_results.keys():
            nan_indices = df[df[dma_col].isna()].index
            if not nan_indices.empty:
                max_nan_index = max(max_nan_index, nan_indices.max())

        # Keep only rows after the last NaN
        if max_nan_index >= 0:
            before_count = len(df)
            df = df.iloc[max_nan_index + 1:].copy()
            after_count = len(df)
            logger.info(f"Filtered to {after_count} valid rows (removed {before_count - after_count} rows with NaN DMA values)")

        if df.empty:
            error = report_error("data", "No valid DMA data after filtering - insufficient data even after fetch",
                           {"symbol": symbol, "periods": dma_periods})
            logger.error("No valid DMA data after filtering - insufficient data even after fetch")
            continue

        # Filter data for the specified number of days
        if not df.empty:
            latest_date = df['timestamp'].max().date()
            start_date = latest_date - timedelta(days=days_to_list - 1)
            df = df[df['timestamp'].dt.date >= start_date]

            logger.info(f"Showing data for last {days_to_list} days")

        # Save data with DMA calculations to CSV file
        try:
            os.makedirs(f"data/{symbol}", exist_ok=True)
            csv_path = f"data/{symbol}/{symbol}_dma_data.csv"

            # Keep NaN values as NaN for proper numerical analysis
            csv_df = df.copy()
            csv_df.to_csv(csv_path, index=False)
            logger.info(f"Data with DMA calculations saved to: {csv_path}")
            logger.info(f"Final dataframe shape: {df.shape}")
            logger.debug(f"Final dataframe columns: {list(df.columns)}")
        except Exception as e:
            error = report_error("system", f"Failed to save CSV file for {symbol}: {str(e)}",
                           {"symbol": symbol, "path": f"data/{symbol}/{symbol}_dma_data.csv", "error": str(e)})
            logger.error(f"Failed to save CSV file: {e}")
            # Continue with console output even if CSV save fails

        # Display results in table format
        logger.info(f"\n{'='*100}")
        logger.info(f"DMA ANALYSIS - {symbol.upper()}")
        logger.info(f"{'='*100}")

        # Create headers
        headers = ['Time', 'Symbol', 'CMP'] + [f'DMA{period}' for period in dma_periods]

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
                    period = int(header.replace('DMA', ''))
                    dma_value = row[f'dma_{period}']
                    if pd.isna(dma_value):
                        value = 'N/A'
                    else:
                        value = f"{dma_value:.2f}"
                col_widths[header] = max(col_widths[header], len(str(value)))

        # Print table header
        header_row = ' | '.join(header.ljust(col_widths[header]) for header in headers)
        logger.info(header_row)
        logger.info('-' * len(header_row))

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
                    if pd.isna(close_val):
                        value = 'N/A'
                    else:
                        try:
                            value = f"{float(close_val):.2f}"
                        except (ValueError, TypeError):
                            value = 'N/A'
                else:
                    period = int(header.replace('DMA', ''))
                    dma_value = row[f'dma_{period}']
                    if pd.isna(dma_value):
                        value = 'N/A'
                    else:
                        value = f"{dma_value:.2f}"
                row_data.append(str(value).ljust(col_widths[header]))
            logger.info(' | '.join(row_data))

        logger.info(f"{'='*100}\n")

    # Calculate total execution time
    execution_time = time.time() - start_time
    logger.info(f"DMA Scanner completed successfully in {execution_time:.2f} seconds!")
    return {"status": "success", "execution_time": execution_time}

if __name__ == "__main__":
    try:
        result = run_dma_scanner()
        sys.exit(0)  # Success
    except Exception as e:
        error = report_error("critical", f"Unhandled exception in DMA scanner: {str(e)}",
                         {"error": str(e), "traceback": traceback.format_exc()})
        logger.exception("Unhandled exception in DMA scanner")
        sys.exit(1)  # Failure