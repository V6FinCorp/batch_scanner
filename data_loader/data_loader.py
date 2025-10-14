"""
Parameterized data loader for fetching historical data.

Adapted from batch_download.py to accept symbol and duration parameters.
"""

import os
import time
import logging
import importlib
import traceback
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, timezone
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from symbol_mapper import SymbolMapper
except Exception:
    # minimal fallback mapper: returns symbol unchanged
    class SymbolMapper:
        def __init__(self):
            pass
        def map(self, s):
            return s
    logger.warning("Could not import SymbolMapper, using fallback")

try:
    from download_historical_data import HistoricalDataDownloader
except Exception:
    HistoricalDataDownloader = None
    logger.warning("Could not import HistoricalDataDownloader, will use fallback")
    # We'll provide a simple inline downloader below if needed
    
try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests library not available - some features will be limited")

# Import config - handle both relative and absolute imports
try:
    from .config import config as base_cfg
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
    from config import config as base_cfg
    logger.info("Using direct import for config")

def report_error(category: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a structured error report
    
    Args:
        category: Error category (api, data, configuration, system)
        message: Human-readable error message
        details: Additional error details
        
    Returns:
        Dictionary with structured error information
    """
    error_data = {
        "error": message,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }
    
    # Log the error with appropriate level
    if category == "critical":
        logger.critical(f"{category.upper()}: {message}")
    elif category in ["api", "data"]:
        logger.error(f"{category.upper()}: {message}")
    else:
        logger.warning(f"{category.upper()}: {message}")
        
    return error_data

def fetch_data_for_symbol(symbol, days_back=30):
    """
    Fetch historical data for a single symbol for the specified number of days.
    
    Args:
        symbol: Symbol to download data for
        days_back: Number of trading days to fetch
        
    Returns:
        Path to the combined CSV file or None if error
    """
    if not symbol:
        return report_error("validation", "No symbol provided", {"days_back": days_back})
        
    if not isinstance(days_back, int) or days_back <= 0:
        return report_error("validation", f"Invalid days_back parameter: {days_back}", 
                         {"symbol": symbol, "days_back": days_back})
    
    logger.info(f"Fetching data for {symbol} ({days_back} days)")
    
    # Check if data already exists and is sufficient before fetching
    symbol_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', symbol.upper())
    combined_filename = f"{symbol}_5_combined.csv"
    combined_filepath = os.path.join(symbol_dir, combined_filename)
    
    if os.path.exists(combined_filepath):
        try:
            import pandas as pd
            existing_df = pd.read_csv(combined_filepath)
            
            if existing_df.empty:
                logger.warning(f"Existing data file for {symbol} is empty, will fetch new data")
            else:
                # Validate data structure
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in existing_df.columns]
                
                if missing_columns:
                    logger.warning(f"Existing data for {symbol} is missing columns: {missing_columns}, will fetch new data")
                else:
                    # Check if existing data covers the required date range
                    try:
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                        existing_start = existing_df['timestamp'].min().date()
                        existing_end = existing_df['timestamp'].max().date()

                        # Calculate required date range
                        today = datetime.now(timezone.utc).date()
                        end_date = today
                        while not is_trading_day(end_date) and (today - end_date).days < 10:
                            end_date = end_date - timedelta(days=1)

                        start_date = end_date - timedelta(days=days_back)
                        while not is_trading_day(start_date) and (start_date - (end_date - timedelta(days=days_back))).days < 10:
                            start_date = start_date + timedelta(days=1)

                        # Validate date range
                        if existing_start <= start_date and existing_end >= end_date:
                            logger.info(f"Existing data for {symbol} covers the required date range ({existing_start} to {existing_end})")
                            logger.info(f"Using existing combined file: {combined_filepath} ({len(existing_df)} records)")

                            # Check data quality
                            invalid_data = (existing_df['high'] < existing_df['low']) | (existing_df['open'] <= 0) | (existing_df['close'] <= 0)
                            invalid_pct = 0.0
                            if invalid_data.any():
                                invalid_pct = 100 * invalid_data.sum() / len(existing_df)
                                logger.warning(f"Existing data has {invalid_pct:.1f}% invalid OHLC values")

                            # Perform a freshness check for intraday data regardless of data-quality
                            try:
                                max_ts = existing_df['timestamp'].max()
                                # Make both times timezone-aware in Asia/Kolkata for proper comparison
                                try:
                                    # Convert max_ts to pandas.Timestamp if not already
                                    max_ts = pd.to_datetime(max_ts)
                                    if max_ts.tzinfo is None:
                                        # Assume CSV timestamps are in IST if they have offset or naive - localize accordingly
                                        max_ts = max_ts.tz_localize('Asia/Kolkata')
                                    else:
                                        max_ts = max_ts.tz_convert('Asia/Kolkata')
                                except Exception:
                                    # Fallback: parse without tz
                                    max_ts = pd.to_datetime(max_ts)

                                now = pd.Timestamp.now(tz='Asia/Kolkata')

                                # If the combined file has data for today but the most recent
                                # timestamp is older than configured threshold, consider it stale.
                                # Read from on-disk config if available, else fall back to 5 minutes.
                                try:
                                    stale_threshold_minutes = int(getattr(base_cfg, 'STALE_THRESHOLD_MINUTES', 5))
                                except Exception:
                                    stale_threshold_minutes = 5
                                is_stale = False
                                try:
                                    if isinstance(max_ts, pd.Timestamp):
                                        # Compare using tz-aware timestamps
                                        delta = now - max_ts
                                        if max_ts.date() == now.date() and delta > timedelta(minutes=stale_threshold_minutes):
                                            is_stale = True
                                except Exception:
                                    # Conservatively treat as not-stale on parse errors
                                    is_stale = False

                                if not is_stale and invalid_pct <= 5.0:
                                    # Clean up leftover chunk files if KEEP_CHUNK_FILES is False
                                    keep_chunk_files = getattr(base_cfg, 'KEEP_CHUNK_FILES', True)
                                    if not keep_chunk_files:
                                        import glob
                                        chunk_pattern = os.path.join(symbol_dir, f"{symbol}_5_*.csv")
                                        chunk_files_to_clean = [f for f in glob.glob(chunk_pattern) if 'combined' not in f]
                                        if chunk_files_to_clean:
                                            logger.info(f"Cleaning up {len(chunk_files_to_clean)} leftover chunk files (KEEP_CHUNK_FILES=False)")
                                            for chunk_file in chunk_files_to_clean:
                                                try:
                                                    os.remove(chunk_file)
                                                    logger.debug(f"Removed leftover chunk file: {chunk_file}")
                                                except Exception as e:
                                                    logger.warning(f"Failed to remove chunk file {chunk_file}: {e}")

                                    return combined_filepath
                                else:
                                    logger.info(f"Existing combined file not suitable (stale={is_stale}, invalid_pct={invalid_pct:.1f}) — will fetch fresh data")
                            except Exception as e:
                                logger.warning(f"Error checking freshness of existing combined file: {e}")
                                # If any error occurs while checking freshness, fallthrough and fetch new data
                        else:
                            logger.info(f"Existing data for {symbol} ({existing_start} to {existing_end}) doesn't cover required range ({start_date} to {end_date})")
                            logger.info("Fetching additional data...")
                    except Exception as e:
                        logger.warning(f"Error validating date range for {symbol}: {e}")
                        logger.info("Fetching new data due to validation error...")
                    except Exception as e:
                        logger.warning(f"Error validating date range for {symbol}: {e}")
                        logger.info("Fetching new data due to validation error...")
        except Exception as e:
            logger.warning(f"Error checking existing data for {symbol}: {e}")
            logger.info("Fetching new data due to error...")
    else:
        logger.info(f"No existing data found for {symbol}, will fetch new data")
    
    # Create a per-symbol copy of configuration
    import types
    custom_config = types.SimpleNamespace()
    
    # copy only needed settings from the module into an independent object
    required_config_keys = [
        'SYMBOL', 'UNIT', 'INTERVALS', 'INTERVAL',
        'CHUNK_DAYS', 'MAX_RETRIES', 'RETRY_BACKOFF_SECONDS',
        'API_BASE_URL', 'INTRADAY_API_URL', 'HOLIDAY_API_URL', 'OUTPUT_DIRECTORY', 'OUTPUT_FORMAT', 'ENABLE_HOLIDAY_CHECK',
        'KEEP_CHUNK_FILES'
    ]
    
    # Validate that base_cfg has the minimum required configuration
    missing_keys = [key for key in ['OUTPUT_DIRECTORY', 'INTERVAL'] if not hasattr(base_cfg, key)]
    if missing_keys:
        logger.warning(f"Configuration is missing required keys: {missing_keys}")
        # Set defaults for missing keys
        if 'OUTPUT_DIRECTORY' in missing_keys:
            setattr(base_cfg, 'OUTPUT_DIRECTORY', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
        if 'INTERVAL' in missing_keys:
            setattr(base_cfg, 'INTERVAL', '5')
    
    # Copy configuration values with validation
    for key in required_config_keys:
        if hasattr(base_cfg, key):
            setattr(custom_config, key, getattr(base_cfg, key))
        else:
            # Set defaults for missing optional keys
            if key == 'UNIT':
                setattr(custom_config, key, 'minutes')
            elif key == 'INTERVALS':
                setattr(custom_config, key, [getattr(base_cfg, 'INTERVAL', '5')])
            elif key == 'CHUNK_DAYS':
                setattr(custom_config, key, 30)
            elif key == 'MAX_RETRIES':
                setattr(custom_config, key, 3)
            elif key == 'RETRY_BACKOFF_SECONDS':
                setattr(custom_config, key, 5)
            elif key == 'API_BASE_URL':
                setattr(custom_config, key, 'https://api.upstox.com/v3/historical-candle')
            elif key == 'INTRADAY_API_URL':
                setattr(custom_config, key, 'https://api.upstox.com/v3/historical-candle/intraday')
            elif key == 'HOLIDAY_API_URL':
                setattr(custom_config, key, 'https://api.upstox.com/v2/market/holidays')
            elif key == 'OUTPUT_FORMAT':
                setattr(custom_config, key, 'csv')
            elif key == 'ENABLE_HOLIDAY_CHECK':
                setattr(custom_config, key, True)
            elif key == 'KEEP_CHUNK_FILES':
                setattr(custom_config, key, False)
    
    # set symbol and days_back
    custom_config.SYMBOL = symbol
    custom_config.DAYS_BACK = days_back
    custom_config.USE_DAYS_BACK = True
    
    # Ensure the output directory is set correctly
    default_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    custom_config.OUTPUT_DIRECTORY = getattr(custom_config, 'OUTPUT_DIRECTORY', default_output_dir)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(custom_config.OUTPUT_DIRECTORY, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {custom_config.OUTPUT_DIRECTORY}")
    except Exception as e:
        logger.error(f"Failed to create output directory {custom_config.OUTPUT_DIRECTORY}: {e}")
        # Try to use a fallback directory
        fallback_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            custom_config.OUTPUT_DIRECTORY = fallback_dir
            logger.info(f"Using fallback output directory: {fallback_dir}")
        except Exception as e2:
            return report_error("system", f"Failed to create output directories: {e2}", 
                             {"original_error": str(e), "fallback_dir": fallback_dir})

    # Create symbol mapper once for efficiency
    symbol_mapper = SymbolMapper()
    
    # Load instrument mapping (if available) once and attempt to resolve the instrument key
    instrument_map = None
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.normpath(os.path.join(module_dir, 'config', 'instrument_mapping.json'))
        if os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r', encoding='utf-8') as mf:
                instrument_map = json.load(mf)
    except Exception:
        instrument_map = None

    # Helper: resolve instrument_key for a symbol using mapping
    def resolve_instrument_key(sym: str):
        if not instrument_map:
            return None
        key = instrument_map.get(sym.upper())
        if key and isinstance(key, dict) and 'instrument_key' in key:
            return key['instrument_key']
        # fallback: search trading_symbol fields
        for k, v in instrument_map.items():
            if isinstance(v, dict) and v.get('trading_symbol', '').upper() == sym.upper():
                return v.get('instrument_key')
        return None

    # If config requests validation, resolve and skip invalid symbols to avoid wasted retries
    if getattr(custom_config, 'VALIDATE_SYMBOLS', True):
        resolved_key = resolve_instrument_key(symbol)
        if not resolved_key:
            logger.error(f"Symbol {symbol} could not be resolved in instrument mapping — skipping")
            return None
        else:
            # Attach resolved key to custom_config so downloader can use it if needed
            custom_config.INSTRUMENT_KEY = resolved_key
    
    # Helper: split the configured date range into calendar-month aligned chunks
    def split_date_range_by_month(start_str: str, end_str: str):
        """
        Split inclusive YYYY-MM-DD start/end into list of (start,end) pairs aligned to calendar months.
        """
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
        if end < start:
            raise ValueError("END_DATE must be >= START_DATE")

        chunks = []
        cur = start
        while cur <= end:
            last_day = calendar.monthrange(cur.year, cur.month)[1]
            month_end = datetime(cur.year, cur.month, last_day).date()
            chunk_end = month_end if month_end <= end else end
            chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            # move to first day of next month
            cur = month_end + timedelta(days=1)
        return chunks

    # Create downloader
    if HistoricalDataDownloader is not None:
        downloader = HistoricalDataDownloader(
            config_module=custom_config,
            symbol_mapper=symbol_mapper
        )
    else:
        # Inline simple downloader using Upstox API v3 as a fallback.
        import requests

        class SimpleDownloader:
            def __init__(self, config_module, symbol):
                self.cfg = config_module
                self.symbol = symbol
                # Try to load instrument mapping to resolve proper instrument_key
                try:
                    module_dir = os.path.dirname(os.path.abspath(__file__))
                    mapping_path = os.path.normpath(os.path.join(module_dir, 'config', 'instrument_mapping.json'))
                    if os.path.exists(mapping_path):
                        import json
                        with open(mapping_path, 'r', encoding='utf-8') as mf:
                            self.instrument_map = json.load(mf)
                    else:
                        self.instrument_map = None
                except Exception:
                    self.instrument_map = None

            def _build_url(self, start_date: str, end_date: str, interval: str):
                base = getattr(self.cfg, 'API_BASE_URL', 'https://api.upstox.com/v3/historical-candle')
                unit = getattr(self.cfg, 'UNIT', 'minutes')
                # Use resolved instrument_key from config, fallback to mapping lookup
                instrument_key = getattr(self.cfg, 'INSTRUMENT_KEY', None)
                if not instrument_key:
                    # Try to resolve from mapping
                    if self.instrument_map and self.symbol.upper() in self.instrument_map:
                        symbol_data = self.instrument_map[self.symbol.upper()]
                        if isinstance(symbol_data, dict) and 'instrument_key' in symbol_data:
                            instrument_key = symbol_data['instrument_key']
                    # Final fallback
                    if not instrument_key:
                        instrument_key = f"NSE_EQ|{self.symbol}"
                safe_key = instrument_key.replace('|', '%7C')
                to_date = end_date
                from_date = start_date

                if unit == 'days':
                    return f"{base}/{safe_key}/days/{interval}/{to_date}/{from_date}"
                else:
                    return f"{base}/{safe_key}/minutes/{interval}/{to_date}/{from_date}"

            def run(self):
                # Get dates from config
                start_date = getattr(self.cfg, 'START_DATE')
                end_date = getattr(self.cfg, 'END_DATE')
                out_dir = getattr(self.cfg, 'OUTPUT_DIRECTORY', 'data')
                os.makedirs(out_dir, exist_ok=True)

                intervals = getattr(self.cfg, 'INTERVALS', [getattr(self.cfg, 'INTERVAL', '5')])
                for interval_val in intervals:
                    url = self._build_url(start_date, end_date, interval=str(interval_val))

                    # auth token: prefer env var UPSTOX_ACCESS_TOKEN
                    token = os.environ.get('UPSTOX_ACCESS_TOKEN', '')
                    headers = {'Accept': 'application/json'}
                    if token:
                        headers['Authorization'] = f'Bearer {token}'

                    resp = requests.get(url, headers=headers, timeout=20)

                    try:
                        data = resp.json()
                    except Exception:
                        logger.exception("Failed to parse JSON response")
                        return None

                    candles = []
                    if isinstance(data, dict) and 'data' in data:
                        d = data['data']
                        if isinstance(d, dict) and 'candles' in d:
                            candles = d['candles']
                        elif isinstance(d, list):
                            candles = d
                    elif isinstance(data, list):
                        candles = data

                    if not candles:
                        logger.error(f"No candles returned for {self.symbol}")
                        return None

                    rows = []
                    for c in candles:
                        if isinstance(c, (list, tuple)):
                            rows.append(c[:6])
                        elif isinstance(c, dict):
                            ts = c.get('timestamp') or c.get('time') or c.get('date')
                            rows.append([
                                ts,
                                c.get('open', 0),
                                c.get('high', 0),
                                c.get('low', 0),
                                c.get('close', 0),
                                c.get('volume', 0)
                            ])

                    symbol_dir = os.path.join(out_dir, self.symbol.upper())
                    os.makedirs(symbol_dir, exist_ok=True)
                    # Create unique filename for each chunk using date range
                    start_date = getattr(self.cfg, 'START_DATE')
                    end_date = getattr(self.cfg, 'END_DATE')
                    fname = f"{self.symbol}_{interval_val}_{start_date}_{end_date}.csv"
                    fpath = os.path.join(symbol_dir, fname)
                    try:
                        with open(fpath, 'w', encoding='utf-8') as f:
                            f.write('timestamp,open,high,low,close,volume\n')
                            for r in rows:
                                line = ','.join([str(x) for x in r])
                                f.write(line + '\n')
                    except Exception:
                        logger.exception(f"Failed to write file {fpath}")
                        return None

                    return fpath

        downloader = SimpleDownloader(custom_config, symbol)

    # Calculate the date range for the data fetching
    end_date = get_latest_business_date(custom_config)
    if end_date is None:
        return report_error("date", "Failed to calculate latest business date", 
                          {"symbol": symbol, "days_back": days_back})
    
    # Calculate start date based on days_back
    try:
        if hasattr(custom_config, 'USE_DAYS_BACK') and custom_config.USE_DAYS_BACK:
            if not isinstance(days_back, int) or days_back <= 0:
                return report_error("parameter", "Invalid days_back parameter", 
                                  {"days_back": days_back, "required": "positive integer"})
            
            start_date = calculate_start_date(end_date, days_back)
        else:
            # Use fixed start date if provided
            if hasattr(custom_config, 'START_DATE') and custom_config.START_DATE:
                try:
                    start_date = pd.to_datetime(custom_config.START_DATE).date()
                except Exception as e:
                    return report_error("parameter", "Invalid START_DATE configuration", 
                                      {"START_DATE": custom_config.START_DATE, "error": str(e)})
            else:
                # Default to 365 days back if neither days_back nor START_DATE is specified
                start_date = calculate_start_date(end_date, 365)
                logger.warning(f"No days_back or START_DATE specified. Defaulting to 365 days back from {end_date}")
    except Exception as e:
        return report_error("date", "Error calculating start date", 
                          {"error": str(e), "end_date": end_date, "days_back": days_back})
    
    logger.info(f"Date range for {symbol}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Convert dates to strings for API and file paths
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    custom_config.START_DATE = start_date_str
    custom_config.END_DATE = end_date_str

    logger.info(f"Preparing to fetch data for {symbol} from {start_date_str} to {end_date_str}")
    
    # Helper function for splitting date range into chunks
    def split_date_range_by_month(start_str, end_str):
        """
        Split inclusive YYYY-MM-DD start/end into list of (start,end) pairs aligned to calendar months.
        
        Args:
            start_str: Start date in YYYY-MM-DD format
            end_str: End date in YYYY-MM-DD format
            
        Returns:
            List of (start, end) date string tuples
        """
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
            end = datetime.strptime(end_str, "%Y-%m-%d").date()
            
            if end < start:
                logger.error(f"Invalid date range: end date {end_str} is before start date {start_str}")
                return [(start_str, end_str)]  # Return single chunk as fallback

            chunks = []
            cur = start
            while cur <= end:
                # Get last day of current month
                last_day = calendar.monthrange(cur.year, cur.month)[1]
                month_end = datetime(cur.year, cur.month, last_day).date()
                # Limit to requested end date
                chunk_end = month_end if month_end <= end else end
                chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
                # Move to first day of next month
                cur = month_end + timedelta(days=1)
                
            return chunks
        except Exception as e:
            logger.error(f"Error splitting date range: {e}")
            # Return original range as fallback
            return [(start_str, end_str)]

    # Use monthly chunks for efficient data retrieval
    # If a combined file already exists, compute only the missing date range to avoid re-downloading
    all_chunks = []
    try:
        if os.path.exists(combined_filepath):
            # Read only timestamps to determine existing coverage
            import pandas as _pd
            try:
                existing_ts = _pd.read_csv(combined_filepath, usecols=['timestamp'])
                existing_ts['timestamp'] = _pd.to_datetime(existing_ts['timestamp'])
                existing_max_ts = existing_ts['timestamp'].max()
                # Use the date of the last available timestamp as the base for missing data
                existing_max_date = existing_max_ts.date()

                # If last timestamp is earlier on the same end_date, we still want to include that date
                # so use existing_max_date (not +1 day). The missing range is from max(requested start_date, existing_max_date)
                missing_start_date = existing_max_date
                if missing_start_date < start_date:
                    missing_start_date = start_date

                # If end_date is today, cap monthly chunks to yesterday because today's
                # intraday data is fetched separately (cheaper and more appropriate).
                today_dt = datetime.now(timezone.utc).date()
                chunk_end_date = end_date
                if end_date == today_dt:
                    chunk_end_date = end_date - timedelta(days=1)

                if missing_start_date <= chunk_end_date:
                    all_chunks = split_date_range_by_month(missing_start_date.strftime('%Y-%m-%d'), chunk_end_date.strftime('%Y-%m-%d'))
                else:
                    all_chunks = []
            except Exception:
                # If we fail to read existing combined file quickly, fall back to full range
                all_chunks = split_date_range_by_month(start_date_str, end_date_str)
        else:
            all_chunks = split_date_range_by_month(start_date_str, end_date_str)
    except Exception:
        all_chunks = split_date_range_by_month(start_date_str, end_date_str)
    chunk_count = len(all_chunks)
    
    if chunk_count == 0:
        logger.info("No monthly chunks required; will attempt intraday/current-day fetch only")
    else:
        logger.info(f"Total required chunks: {chunk_count}")
    
    # Download chunks
    chunk_files = []
    failed_chunks = []
    
    for idx, (chunk_start, chunk_end) in enumerate(all_chunks, start=1):
        logger.info(f"Chunk {idx}/{chunk_count} for {symbol}: {chunk_start} -> {chunk_end}")
        
        # Maximum retry count for each chunk
        max_retries = getattr(custom_config, 'MAX_RETRIES', 3)
        retry_backoff = getattr(custom_config, 'RETRY_BACKOFF_SECONDS', 5)
        
        # Update config for this chunk
        custom_config.START_DATE = chunk_start
        custom_config.END_DATE = chunk_end
        
        # Try to download the chunk with retries
        chunk_output = None
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries and not chunk_output:
            if retry_count > 0:
                # Add exponential backoff delay for retries
                backoff_time = retry_backoff * (2 ** (retry_count - 1))
                logger.info(f"Retry #{retry_count} for chunk {idx}/{chunk_count} after {backoff_time}s delay")
                time.sleep(backoff_time)
            
            try:
                # Create downloader for this chunk
                if HistoricalDataDownloader is not None:
                    chunk_downloader = HistoricalDataDownloader(
                        config_module=custom_config,
                        symbol_mapper=symbol_mapper
                    )
                else:
                    chunk_downloader = SimpleDownloader(custom_config, symbol)
                
                chunk_output = chunk_downloader.run()
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error downloading chunk {idx}/{chunk_count}: {e}")
                traceback.print_exc()
            
            retry_count += 1
        
        if chunk_output:
            chunk_files.append(chunk_output)
            logger.info(f"Chunk saved: {chunk_output}")
        else:
            logger.error(f"Failed to download chunk {chunk_start} -> {chunk_end} after {max_retries} attempts")
            failed_chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "error": last_error
            })
    
    # Report error if all chunks failed (only when we expected chunks)
    if chunk_count > 0 and failed_chunks and len(failed_chunks) == chunk_count:
        return report_error("api", f"All {chunk_count} data chunks failed to download", 
                          {"symbol": symbol, "failed_chunks": failed_chunks})
    
    # Log warning if some chunks failed
    if failed_chunks:
        logger.warning(f"{len(failed_chunks)}/{chunk_count} chunks failed for {symbol}")
        
    # Continue with available data if at least one chunk succeeded (only required when chunk_count>0)
    if chunk_count > 0 and not chunk_files:
        return report_error("api", "No data chunks were successfully downloaded", 
                          {"symbol": symbol, "days_back": days_back})
    
    # FETCH CURRENT DAY'S DATA IF TODAY IS A TRADING DAY
    today = datetime.now(timezone.utc).date()
    today_str = today.strftime("%Y-%m-%d")
    
    if is_trading_day(today) and requests is not None:
        logger.info(f"Today ({today_str}) is a trading day - fetching current day data")
        try:
            # Get instrument key for the symbol
            instrument_key = getattr(custom_config, 'INSTRUMENT_KEY', None)
            if not instrument_key:
                # Try to resolve from mapping
                if instrument_map:
                    im = instrument_map.get(symbol.upper())
                    if not im:
                        for k, v in instrument_map.items():
                            if isinstance(v, dict) and v.get('trading_symbol', '').upper() == symbol.upper():
                                im = v
                                break
                    if im and 'instrument_key' in im:
                        instrument_key = im['instrument_key']

            if not instrument_key:
                instrument_key = f"NSE_EQ|{symbol}"
                logger.debug(f"Using default instrument key: {instrument_key}")

            # Build intraday API URL
            intraday_url = getattr(custom_config, 'INTRADAY_API_URL', 'https://api.upstox.com/v3/historical-candle/intraday')
            unit = getattr(custom_config, 'UNIT', 'minutes')
            interval = str(getattr(custom_config, 'INTERVAL', '5'))
            safe_key = instrument_key.replace('|', '%7C')

            url = f"{intraday_url}/{safe_key}/minutes/{interval}"
            logger.debug(f"Intraday API URL: {url}")

            # Make API request
            token = os.environ.get('UPSTOX_ACCESS_TOKEN', '')
            headers = {'Accept': 'application/json'}
            if token:
                headers['Authorization'] = f'Bearer {token}'
            else:
                logger.warning("No UPSTOX_ACCESS_TOKEN found in environment variables")

            # Use max retries with backoff for intraday data
            max_retries = getattr(custom_config, 'MAX_RETRIES', 3)
            retry_backoff = getattr(custom_config, 'RETRY_BACKOFF_SECONDS', 5)
            
            response = None
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries and not response:
                if retry_count > 0:
                    # Add exponential backoff delay for retries
                    backoff_time = retry_backoff * (2 ** (retry_count - 1))
                    logger.info(f"Retry #{retry_count} for intraday data after {backoff_time}s delay")
                    time.sleep(backoff_time)
                
                try:
                    response = requests.get(url, headers=headers, timeout=20)
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Error fetching intraday data: {e}")
                
                retry_count += 1

            if not response:
                logger.error(f"Failed to fetch intraday data after {max_retries} attempts: {last_error}")
                # Continue without intraday data
            elif response.status_code == 200:
                try:
                    data = response.json()
                    candles = []

                    if isinstance(data, dict) and 'data' in data:
                        d = data['data']
                        if isinstance(d, dict) and 'candles' in d:
                            candles = d['candles']
                        elif isinstance(d, list):
                            candles = d
                    elif isinstance(data, list):
                        candles = data

                    if candles:
                        # Create today's data file
                        interval = str(getattr(custom_config, 'INTERVAL', '5'))
                        symbol_dir = os.path.join(custom_config.OUTPUT_DIRECTORY, symbol.upper())
                        os.makedirs(symbol_dir, exist_ok=True)
                        fname = f"{symbol}_{interval}_{today_str}_{today_str}.{getattr(custom_config, 'OUTPUT_FORMAT', 'csv')}"
                        fpath = os.path.join(symbol_dir, fname)

                        try:
                            with open(fpath, 'w', encoding='utf-8') as f:
                                f.write('timestamp,open,high,low,close,volume\n')
                                
                                candle_count = 0
                                for c in candles:
                                    if isinstance(c, (list, tuple)) and len(c) >= 6:
                                        line = ','.join([str(x) for x in c[:6]])
                                        f.write(line + '\n')
                                        candle_count += 1
                                    elif isinstance(c, dict):
                                        ts = c.get('timestamp') or c.get('time') or c.get('date')
                                        if ts and all(k in c for k in ['open', 'high', 'low', 'close']):
                                            line = ','.join([
                                                str(ts),
                                                str(c.get('open', 0)),
                                                str(c.get('high', 0)),
                                                str(c.get('low', 0)),
                                                str(c.get('close', 0)),
                                                str(c.get('volume', 0))
                                            ])
                                            f.write(line + '\n')
                                            candle_count += 1

                            logger.info(f"Current day data saved: {fpath} ({candle_count} candles)")
                            
                            if candle_count > 0:
                                chunk_files.append(fpath)  # Add to chunk_files list
                            else:
                                logger.warning(f"No valid candles found in intraday data")
                                # Remove empty file
                                try:
                                    os.remove(fpath)
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.warning(f"Failed to save current day data: {e}")
                    else:
                        logger.info("No current day data available")
                except Exception as e:
                    logger.warning(f"Error processing intraday data: {e}")
            else:
                logger.warning(f"Failed to fetch current day data: HTTP {response.status_code}")
                if response.text:
                    try:
                        error_data = response.json()
                        logger.debug(f"Error response: {error_data}")
                    except Exception:
                        logger.debug(f"Error response text: {response.text[:200]}...")

        except Exception as e:
            logger.warning(f"Error fetching current day data: {e}")
    else:
        if not is_trading_day(today):
            logger.info(f"Today ({today_str}) is not a trading day - skipping current day data fetch")
        if requests is None:
            logger.info("requests library not available - skipping current day data fetch")
    
    # Create combined file
    if chunk_files:
        # If combined file exists, start combined_data from its existing lines to append only new data
        combined_data = []
        header_line = None
        processed_files = 0
        error_count = 0

        # If an existing combined file exists, preload its data lines (skip header)
        if os.path.exists(combined_filepath):
            try:
                with open(combined_filepath, 'r', encoding='utf-8') as ef:
                    existing_lines = ef.readlines()
                    if existing_lines:
                        # find header and use remaining as initial combined_data
                        if existing_lines[0].strip().startswith('timestamp'):
                            header_line = existing_lines[0].strip()
                            combined_data.extend([ln.strip() for ln in existing_lines[1:] if ln.strip()])
                        else:
                            combined_data.extend([ln.strip() for ln in existing_lines if ln.strip()])
            except Exception as e:
                logger.warning(f"Failed to preload existing combined file {combined_filepath}: {e}")

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + data
                        data_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
                        combined_data.extend(data_lines)
                processed_files += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to read chunk file {chunk_file}: {e}")

        if processed_files == 0 and not combined_data:
            return report_error("system", "Failed to read any chunk files and no existing combined data available", 
                              {"total_chunks": len(chunk_files), "errors": error_count})

        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors while reading {len(chunk_files)} chunk files")

        if not combined_data:
            return report_error("data", "No data found in chunk files or existing combined file", 
                              {"processed_files": processed_files, "total_chunks": len(chunk_files)})

        # Remove duplicates and sort
        try:
            seen_timestamps = set()
            unique_data = []

            for line in combined_data:
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 6:
                    timestamp = parts[0]
                    if timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_data.append(line)

            # Determine header_line if not set
            if not header_line:
                header_line = 'timestamp,open,high,low,close,volume'

            # Sort by timestamp
            unique_data.sort(key=lambda x: x.split(',')[0])

            # Create symbol directory if it doesn't exist
            symbol_dir = os.path.join(custom_config.OUTPUT_DIRECTORY, symbol.upper())
            os.makedirs(symbol_dir, exist_ok=True)

            # Define combined file path
            combined_filename = f"{symbol}_5_combined.csv"
            combined_filepath = os.path.join(symbol_dir, combined_filename)

            # Write combined file
            with open(combined_filepath, 'w', encoding='utf-8') as f:
                f.write(f"{header_line}\n")
                for line in unique_data:
                    f.write(f"{line}\n")

            record_count = len(unique_data)
            logger.info(f"Combined file updated: {combined_filepath} ({record_count} records)")

            # Data validation
            if record_count == 0:
                return report_error("data", "Combined file has no data records", 
                                  {"filepath": combined_filepath})

            # Clean up chunk files if KEEP_CHUNK_FILES is False
            keep_chunk_files = getattr(custom_config, 'KEEP_CHUNK_FILES', False)
            if not keep_chunk_files:
                logger.info("Cleaning up chunk files (KEEP_CHUNK_FILES=False)")
                cleanup_count = 0
                for chunk_file in chunk_files:
                    try:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                            cleanup_count += 1
                            logger.debug(f"Removed chunk file: {chunk_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove chunk file {chunk_file}: {e}")

                if cleanup_count > 0:
                    logger.info(f"Cleaned up {cleanup_count}/{len(chunk_files)} chunk files")

            return combined_filepath

        except Exception as e:
            return report_error("system", f"Error processing combined data: {e}", 
                              {"processed_files": processed_files, "error": traceback.format_exc()})
    
    return report_error("data", "No chunk files available to combine", 
                      {"symbol": symbol, "days_back": days_back})

def is_trading_day(date):
    """
    Check if a date is a trading day using Upstox API or fallback to weekend check
    
    Args:
        date: The date to check
        
    Returns:
        Boolean indicating if the date is a trading day
    """
    if not date:
        logger.warning("Cannot check trading day status: date is None")
        return False
        
    # Ensure we have a date object
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Invalid date format: {date} - {e}")
            return False
    elif isinstance(date, datetime):
        date = date.date()
    elif not isinstance(date, date.__class__):
        logger.error(f"Unknown date type: {type(date)}")
        return False
    
    # Check if holiday checking is enabled
    enable_holiday_check = getattr(base_cfg, 'ENABLE_HOLIDAY_CHECK', True)

    # Quick check for weekends
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        logger.debug(f"Date {date.strftime('%Y-%m-%d')} is a weekend (day {date.weekday()})")
        return False
        
    if not enable_holiday_check or requests is None:
        # Fallback to weekend-only checking (original logic)
        return date.weekday() < 5  # Monday=0, Sunday=6

    # Cache holiday results to avoid repeated API calls
    if not hasattr(is_trading_day, '_holiday_cache'):
        is_trading_day._holiday_cache = {}
        
    date_str = date.strftime("%Y-%m-%d")
    
    # Return cached result if available
    if date_str in is_trading_day._holiday_cache:
        return is_trading_day._holiday_cache[date_str]

    try:
        url = f"{getattr(base_cfg, 'HOLIDAY_API_URL', 'https://api.upstox.com/v2/market/holidays')}/{date_str}"

        # Make request without authentication (holiday API works without auth)
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            holiday_data = data.get('data', [])

            if holiday_data:
                holiday_desc = holiday_data[0].get('description', 'Holiday')
                logger.info(f"Date {date_str} is a holiday: {holiday_desc}")
                is_trading_day._holiday_cache[date_str] = False
                return False
                
            # If no holiday data returned, it's a trading day
            is_trading_day._holiday_cache[date_str] = True
            return True
        else:
            # If API fails, log the error and fall back to weekend-only check
            logger.warning(f"Failed to check holiday for {date_str} (HTTP {response.status_code}), falling back to weekend check")
            result = date.weekday() < 5
            is_trading_day._holiday_cache[date_str] = result
            return result
    except Exception as e:
        logger.warning(f"Error checking holiday for {date_str}: {e}, falling back to weekend check")
        result = date.weekday() < 5
        is_trading_day._holiday_cache[date_str] = result
        return result

def get_latest_business_date(config):
    """
    Get the latest business (trading) date
    
    Args:
        config: Configuration namespace
        
    Returns:
        Latest business date or None if error
    """
    try:
        # Start with today in UTC
        today = datetime.now(timezone.utc).date()
        
        # Look back up to 10 days to find a trading day
        for i in range(10):
            check_date = today - timedelta(days=i)
            if is_trading_day(check_date):
                logger.debug(f"Latest business date: {check_date.strftime('%Y-%m-%d')}")
                return check_date
                
        # If we couldn't find a trading day in the last 10 days, something is wrong
        logger.error("Could not find a trading day in the last 10 days")
        return None
    except Exception as e:
        logger.error(f"Error getting latest business date: {e}")
        return None
        
def calculate_start_date(end_date, days_back):
    """
    Calculate start date based on days back from end date
    
    Args:
        end_date: End date
        days_back: Number of days to go back
        
    Returns:
        Start date
    """
    if not end_date:
        raise ValueError("End date cannot be None")
        
    if not isinstance(days_back, int) or days_back <= 0:
        raise ValueError(f"Invalid days_back parameter: {days_back}")
    
    # Ensure we have a date object
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()
        
    # Calculate the initial start date
    start_date = end_date - timedelta(days=days_back)
    
    # Find the first trading day on or after the initial start date
    # Look forward up to 10 days
    for i in range(10):
        check_date = start_date + timedelta(days=i)
        if is_trading_day(check_date):
            logger.debug(f"Adjusted start date: {check_date.strftime('%Y-%m-%d')}")
            return check_date
    
    # If we couldn't find a trading day within 10 days, use the unadjusted date
    logger.warning(f"Could not find trading day near {start_date}, using unadjusted date")
    return start_date

if __name__ == "__main__":
    # Test the data loader with a sample symbol
    import sys
    
    # Configure logging for testing
    logger.setLevel(logging.INFO)
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    else:
        symbol = "ITC"  # Default test symbol
        days_back = 30
    
    logger.info(f"Testing data loader with symbol: {symbol}, days_back: {days_back}")
    
    result = fetch_data_for_symbol(symbol, days_back)
    
    if result:
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Failed: {result['error']}")
            logger.error(f"Details: {result.get('details', {})}")
            sys.exit(1)
        else:
            logger.info(f"Success! Data saved to: {result}")
            sys.exit(0)
    else:
        logger.error("Failed to fetch data (null result)")
        sys.exit(1)