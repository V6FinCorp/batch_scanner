# Trading Scanners Dashboard

A comprehensive web dashboard for running and visualizing technical analysis scanners including RSI, EMA, and DMA calculations.

## 🚀 Features

- **Interactive Web Interface**: Modern, responsive dashboard built with HTML, CSS, and JavaScript
- **Multiple Scanners**: Support for RSI, EMA, and DMA analysis
- **Real-time Execution**: Run scanners directly from the web interface
- **Parameter Customization**: Configure scanner parameters through the UI
- **Data Visualization**: Chart visualization of price and indicator data with Line/Candlestick toggle
- **Collapsible Parameters**: Clean, collapsible scanner parameters section
- **CSV Export**: Export results to CSV files
- **API Backend**: RESTful API for scanner execution
- **Railway.app Ready**: Configured for easy deployment on Railway.app

## 📊 Available Scanners

### RSI Scanner
- Calculates RSI using TradingView-compatible RMA (Running Moving Average) method
- Supports multiple periods (default: 15, 30, 60)
- Multi-timeframe analysis capability

### EMA Scanner
- Calculates EMA using TradingView's exact method
- Supports multiple periods (default: 9, 15, 65, 200)
- Exponential smoothing with proper alpha calculation

### DMA Scanner
- Calculates Displaced Moving Averages
- Configurable displacement (default: 1 period)
- Supports multiple periods (default: 10, 20, 50)

## 🛠️ Installation & Setup

1. **Navigate to scanners folder:**
   ```bash
   cd scanners
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Dashboard Server:**
   ```bash
   python dashboard_server.py
   ```

## 🚂 Deploying to Railway.app

1. **Prepare for Deployment:**
   - Ensure you have a valid `requirements.txt` file with all dependencies
   - Verify that the `Procfile` contains the correct startup command (`web: python dashboard_server.py`)
   - Check that `runtime.txt` specifies the correct Python version

2. **Deploy to Railway.app:**
   - Create a new project in Railway.app
   - Connect your GitHub repository or upload the code directly
   - Railway will automatically detect the Python project and install dependencies
   - The application will start using the command in the Procfile

3. **Environment Variables:**
   - Railway automatically sets the `PORT` environment variable, which the app uses
   - Set `FLASK_ENV=production` in your Railway environment variables for production mode

4. **Troubleshooting Deployment:**
   - If deployment fails with pip installation errors, check your `requirements.txt` file
   - Ensure all dependencies are correctly specified with compatible versions
   - Check Railway logs for specific error messages
   - The app is designed to handle missing modules gracefully with fallbacks

4. **Access the Dashboard:**
   Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

**Everything is now organized inside the `scanners/` folder:**

```
scanners/
├── dashboard.html              # Main dashboard HTML interface
├── dashboard_server.py        # Flask server with API endpoints
├── requirements.txt            # Python dependencies
├── railway.json               # Railway.app deployment config
├── Procfile                    # Railway deployment configuration
├── scanner_manager.py          # Central scanner management logic
├── rsi_scanner.py             # RSI scanner implementation
├── ema_scanner.py             # EMA scanner implementation
├── dma_scanner.py             # DMA scanner implementation
├── rsi.pine.txt              # TradingView Pine script
├── templates/                 # Flask templates
│   └── dashboard.html
├── config/                    # Scanner configurations
│   ├── rsi_config.json
│   ├── ema_config.json
│   ├── dma_config.json
│   └── symbols.config.json    # Centralized symbols configuration
├── data/                      # Scanner output data
│   └── [SYMBOL]/             # Symbol-specific data folders
└── data_loader/              # Data loading utilities
    ├── data_loader.py
    └── config/
        ├── instrument_mapping.json
        ├── NSE.json.gz
        └── run_summary.json
```

## 🔧 Configuration

**Centralized Symbol Management**: All symbols are now managed through a single `symbols.config.json` file. Individual scanner configuration files no longer contain symbol arrays, ensuring consistency across all scanners.

Each scanner has its own configuration file in `scanners/config/`, and all symbols are managed centrally:

### Centralized Symbols Config (`symbols.config.json`)
```json
{
    "symbols": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "WIPRO", "LT", "BAJFINANCE", "KOTAKBANK"]
}
```

### RSI Config (`rsi_config.json`)
```json
{
    "rsi_periods": [15, 30, 60],
    "base_timeframe": "15mins",
    "days_to_list": 2,
    "days_fallback_threshold": 200
}
```

### EMA Config (`ema_config.json`)
```json
{
    "ema_periods": [9, 15, 65, 200],
    "base_timeframe": "15mins",
    "days_to_list": 2,
    "days_fallback_threshold": 200
}
```

### DMA Config (`dma_config.json`)
```json
{
    "dma_periods": [10, 20, 50],
    "base_timeframe": "15mins",
    "days_to_list": 2,
    "days_fallback_threshold": 200,
    "displacement": 1
}
```

### How to change indicator periods

- Indicator period lists are authoritative when defined in the scanner config files located in `batch_scanner/config/` (for example `ema_config.json`, `rsi_config.json`, `dma_config.json`).
- To change the periods the scanners use, edit the corresponding JSON array (`ema_periods`, `rsi_periods`, or `dma_periods`) and save the file.
- After editing a config file, re-run the scanner from the dashboard or via the API. The orchestrator (`ScannerManager`) and individual scanner scripts read these files at runtime and will use the updated period lists.
- The web UI displays the configured periods but does not allow changing them at runtime — the on-disk config files are the single source of truth.

## 🌐 API Endpoints

### Run Scanner
```http
POST /api/run-scanner
Content-Type: application/json

{
    "scanner": "rsi|ema|dma",
    "symbols": ["RELIANCE"],
    "rsi_periods": [15, 30, 60],
    "baseTimeframe": "15mins",
    "daysToList": 2,
    "daysFallbackThreshold": 200
}
```

### Get Scanner Status
```http
GET /api/scanner-status
```

## 📈 Usage Examples

### Running RSI Scanner
```javascript
fetch('/api/run-scanner', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        scanner: 'rsi',
        symbols: ['RELIANCE'],  // Optional: uses centralized symbols if not provided
        rsi_periods: [15, 30, 60],
        baseTimeframe: '15mins',
        daysToList: 2
    })
});
```

### Running EMA Scanner
```javascript
fetch('/api/run-scanner', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        scanner: 'ema',
        symbols: ['RELIANCE'],  // Optional: uses centralized symbols if not provided
        ema_periods: [9, 15, 65, 200],
        baseTimeframe: '15mins',
        daysToList: 2
    })
});
```

## 🎨 Dashboard Features

- **Parameter Selection**: Interactive forms for configuring scanner parameters
- **Real-time Feedback**: Loading indicators and status messages
- **Data Visualization**: Chart.js integration for price and indicator visualization
- **Export Functionality**: Download results as CSV files
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, professional interface with Tailwind CSS

## 🔍 Data Sources

The scanners fetch data from Upstox API with support for:
- Multiple timeframes (5min, 15min, 30min, 1hour, daily, etc.)
- Historical data up to 2 years for daily data
- Real-time market data filtering (9:15 AM market open)

## 📊 Output Formats

### Console Output
Professional table format with:
- Timestamp, Symbol, Close Price
- Indicator values (RSI/EMA/DMA)
- Proper column alignment
- Market hours filtering

### CSV Export
Structured data files containing:
- OHLCV data
- Calculated indicators
- Timestamps
- Ready for further analysis

### Chart Visualization
Interactive charts showing:
- Price action
- Indicator overlays
- Multiple timeframe support
- Zoom and pan capabilities

## 🚀 Deployment

For production deployment:

1. **Railway/Render/Heroku:**
   ```bash
   # Navigate to scanners folder
   cd scanners

   # Set environment variables
   export FLASK_ENV=production
   export PORT=5000

   # Start server
   gunicorn dashboard_server:app
   ```

2. **Docker:**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY scanners/ .
   RUN pip install -r requirements.txt
   CMD ["python", "dashboard_server.py"]
   ```

3. **Railway.app (Recommended):**
   - Point Railway to the `scanners/` folder
   - All deployment files are already configured in `scanners/`
   - Railway will automatically use `Procfile` and `railway.json`
  