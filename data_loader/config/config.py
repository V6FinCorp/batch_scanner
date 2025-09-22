"""
Configuration file for downloading historical data from Upstox API.
"""

# Holiday checking configuration
# Enable/disable holiday API checks for accurate trading day calculation
# When disabled, falls back to weekend-only checking (faster but less accurate)
ENABLE_HOLIDAY_CHECK = False

# Chunking and retry behavior
# Number of days per chunk (the original request used 30-day chunks)
CHUNK_DAYS = 30
# Maximum attempts per chunk before marking it as missing
MAX_RETRIES = 4
# Initial backoff seconds between retries (will exponential-backoff)
RETRY_BACKOFF_SECONDS = 2

# API configuration
API_BASE_URL = "https://api.upstox.com/v3/historical-candle"
INTRADAY_API_URL = "https://api.upstox.com/v3/historical-candle/intraday"
HOLIDAY_API_URL = "https://api.upstox.com/v2/market/holidays"

# Output configuration
OUTPUT_DIRECTORY = "data"
OUTPUT_FORMAT = "csv"  # Options: csv, json

# Chunk file management
# Set to True to keep individual chunk files after creating combined file
# Set to False to delete chunk files after creating combined file (saves disk space)
KEEP_CHUNK_FILES = False

# If True, the downloader will validate each symbol against the local
# instrument mapping (config/instrument_mapping.json) before
# attempting chunked downloads. Invalid symbols are skipped to avoid
# repeated failing HTTP attempts.
VALIDATE_SYMBOLS = True

# Data fetching configuration
UNIT = "minutes"  # Options: minutes, days, weeks, months
INTERVAL = 5  # Candle interval in the specified unit
INTERVALS = [5]  # List of intervals to fetch (for multiple intervals)

# Authentication (optional)
# Set UPSTOX_ACCESS_TOKEN environment variable for authenticated requests
# If not set, requests will be made without authentication