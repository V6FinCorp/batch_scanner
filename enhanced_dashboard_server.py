"""
Enhanced Dashboard Server with Progressive Loading and Status Tracking
"""

import os
import json
import sys
import shutil
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
dashboard_logger = logging.getLogger('dashboard')

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NaN values and numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

app = Flask(__name__, 
                static_folder='.', 
                static_url_path='/static')
app.json_encoder = CustomJSONEncoder
CORS(app)

# Import scanner modules
try:
    from scanner_manager import ScannerManager
    from progressive_scanner import ProgressiveScanner
    dashboard_logger.info("Successfully imported scanner modules")
except ImportError as e:
    dashboard_logger.error(f"Failed to import scanner modules: {e}")
    ScannerManager = None
    ProgressiveScanner = None

# Initialize scanner managers
if ScannerManager:
    scanner_manager = ScannerManager()
    progressive_scanner = ProgressiveScanner() if ProgressiveScanner else None
    dashboard_logger.info("Scanner managers initialized")
else:
    scanner_manager = None
    progressive_scanner = None
    dashboard_logger.error("Scanner manager not available")

@app.route('/')
def dashboard():
    """Serve the enhanced progressive dashboard"""
    try:
        return render_template('progressive_dashboard.html')
    except Exception as e:
        dashboard_logger.error(f"Error rendering progressive template: {e}")
        # Fallback to original dashboard
        try:
            return render_template('dashboard.html')
        except Exception as e2:
            dashboard_logger.error(f"Error rendering fallback template: {e2}")
            with open('dashboard.html', 'r', encoding='utf-8') as f:
                return f.read()

@app.route('/api/symbol-table')
def get_symbol_table():
    """Get symbol table with status - loads immediately on page load"""
    try:
        dashboard_logger.info("Loading symbol table")
        
        if progressive_scanner:
            result = progressive_scanner.get_symbol_table()
            dashboard_logger.info(f"Symbol table loaded with {len(result.get('symbols', []))} symbols")
            return jsonify(result)
        else:
            # Fallback to basic symbol list
            symbols_data = scanner_manager.get_symbols() if scanner_manager else {'symbols': []}
            symbols = symbols_data.get('symbols', [])
            
            result = {
                'success': True,
                'symbols': [
                    {
                        'Symbol': symbol,
                        'Status': 'Ready',
                        'CMP': 'Loading...',
                        'RSI_Available': False,
                        'EMA_Available': False,
                        'DMA_Available': False,
                        'LastUpdate': 'Not calculated'
                    } for symbol in symbols
                ],
                'total_symbols': len(symbols)
            }
            
            dashboard_logger.info(f"Fallback symbol table created with {len(symbols)} symbols")
            return jsonify(result)
            
    except Exception as e:
        dashboard_logger.error(f"Error getting symbol table: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/run-single-scanner', methods=['POST'])
def run_single_scanner():
    """Run scanner for single symbol with status tracking"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
        
    try:
        data = request.get_json()
        scanner_type = data.get('scanner', 'rsi')
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
            
        dashboard_logger.info(f"Starting {scanner_type.upper()} analysis for {symbol}")
        start_time = datetime.now()
        
        # Get parameters
        params = {
            'symbols': [symbol],  # Single symbol only
            'baseTimeframe': data.get('baseTimeframe', '15mins'),
            'daysToList': data.get('daysToList', 4)
        }
        
        # Add scanner-specific parameters
        if scanner_type == 'rsi':
            params.update({
                'rsiPeriods': data.get('rsiPeriods', [5, 15, 30, 60]),
                'daysFallbackThreshold': data.get('daysFallbackThreshold', 200),
                'rsiOverbought': data.get('rsiOverbought', 70),
                'rsiOversold': data.get('rsiOversold', 30)
            })
        elif scanner_type == 'ema':
            params.update({
                'emaPeriods': data.get('emaPeriods', [9, 15, 65, 200]),
                'daysFallbackThreshold': data.get('daysFallbackThreshold', 200)
            })
        elif scanner_type == 'dma':
            params.update({
                'dmaPeriods': data.get('dmaPeriods', [10, 20, 50]),
                'daysFallbackThreshold': data.get('daysFallbackThreshold', 400)
            })
        
        # Run scanner with error isolation
        try:
            result = scanner_manager.run_scanner(
                scanner_type=scanner_type,
                **params
            )
        except Exception as scanner_error:
            dashboard_logger.error(f"Scanner execution error for {symbol}: {scanner_error}")
            result = {
                'error': str(scanner_error),
                'returncode': 1,
                'output': f'Scanner failed: {str(scanner_error)}'
            }
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Log result
        if result.get('returncode') == 0:
            dashboard_logger.info(f"{scanner_type.upper()} completed for {symbol} in {execution_time:.1f}s")
        else:
            dashboard_logger.error(f"{scanner_type.upper()} failed for {symbol}: {result.get('error', 'Unknown error')}")
        
        # Add timing and metadata
        result.update({
            'execution_time': execution_time,
            'symbol': symbol,
            'scanner': scanner_type,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(result)
        
    except Exception as e:
        dashboard_logger.error(f"Error in single scanner execution: {e}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/run-scanner', methods=['POST'])
def run_scanner():
    """Enhanced original API endpoint with better error handling"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
        
    try:
        data = request.get_json()
        scanner_type = data.get('scanner', 'rsi')
        symbols = data.get('symbols', [])
        
        dashboard_logger.info(f"Running {scanner_type.upper()} scanner for {len(symbols)} symbols")
        
        # If more than 5 symbols, suggest using progressive approach
        if len(symbols) > 5:
            dashboard_logger.warning(f"Large batch request ({len(symbols)} symbols) - consider using progressive API")
        
        start_time = datetime.now()
        
        # Get parameters
        base_timeframe = data.get('baseTimeframe', '15mins')
        days_to_list = data.get('daysToList', 4)

        # Extract additional parameters
        kwargs = {
            'daysFallbackThreshold': data.get('daysFallbackThreshold', 200),
            'rsiPeriods': data.get('rsiPeriods', [5, 15, 30, 60]),
            'emaPeriods': data.get('emaPeriods', [9, 15, 65, 200]),
            'dmaPeriods': data.get('dmaPeriods', [10, 20, 50])
        }

        # Run scanner with comprehensive error handling
        result = scanner_manager.run_scanner(
            scanner_type=scanner_type,
            symbols=symbols,
            base_timeframe=base_timeframe,
            days_to_list=days_to_list,
            **kwargs
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result['execution_time'] = execution_time
        
        # Log results
        if result.get('returncode') == 0:
            dashboard_logger.info(f"{scanner_type.upper()} batch completed in {execution_time:.1f}s")
        else:
            dashboard_logger.error(f"{scanner_type.upper()} batch failed: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)

    except Exception as e:
        dashboard_logger.error(f"Error running scanner: {e}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/scanner-status')
def scanner_status():
    """Get status of available scanners"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    try:
        status = scanner_manager.get_scanner_status()
        return jsonify(status)
    except Exception as e:
        dashboard_logger.error(f"Error getting scanner status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbols')
def get_symbols():
    """Get available symbols from JSON file"""
    try:
        if scanner_manager:
            symbols = scanner_manager.get_symbols()
            dashboard_logger.info(f"Loaded {len(symbols.get('symbols', []))} symbols")
            return jsonify(symbols)
        else:
            # Fallback symbols
            fallback_symbols = {
                'symbols': ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY']
            }
            dashboard_logger.warning("Using fallback symbols")
            return jsonify(fallback_symbols)
    except Exception as e:
        dashboard_logger.error(f"Error getting symbols: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """Get application logs"""
    try:
        log_file = 'dashboard.log'
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.readlines()[-200:]  # Last 200 lines
            return jsonify({'success': True, 'logs': logs})
        else:
            return jsonify({'success': True, 'logs': []})
    except Exception as e:
        dashboard_logger.error(f"Error reading logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-logs', methods=['POST'])
def clear_logs():
    """Clear application logs"""
    try:
        log_file = 'dashboard.log'
        if os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')
        dashboard_logger.info('Logs cleared by user')
        return jsonify({'success': True, 'message': 'Logs cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scanner-results/<scanner_type>')
def get_scanner_results(scanner_type):
    """Get stored results for a specific scanner type"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    try:
        results = scanner_manager.get_scanner_results(scanner_type)
        return jsonify(results)
    except Exception as e:
        dashboard_logger.error(f"Error getting scanner results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart-data/<scanner_type>/<symbol>')
def get_chart_data(scanner_type, symbol):
    """Get chart data for a specific symbol and scanner type"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    try:
        chart_data = scanner_manager.get_symbol_chart_data(scanner_type, symbol)
        return jsonify(chart_data)
    except Exception as e:
        dashboard_logger.error(f"Error getting chart data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/<config_type>')
def get_config(config_type):
    """Get config data for a specific indicator type"""
    try:
        config_filename = f'{config_type}.config.json' if config_type == 'symbols' else f'{config_type}_config.json'
        config_path = os.path.join(os.path.dirname(__file__), 'config', config_filename)
        
        dashboard_logger.debug(f"Loading config: {config_type} from {config_path}")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return jsonify(config_data)
        else:
            # Return default values if config file doesn't exist
            defaults = {
                'rsi': {
                    'rsi_periods': [5, 15, 30, 60],
                    'base_timeframe': '15mins',
                    'days_to_list': 4,
                    'days_fallback_threshold': 200,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                },
                'ema': {
                    'ema_periods': [9, 15, 65, 200],
                    'base_timeframe': '15mins',
                    'days_to_list': 4,
                    'days_fallback_threshold': 200
                },
                'dma': {
                    'dma_periods': [10, 20, 50],
                    'base_timeframe': '1hour',
                    'days_to_list': 4,
                    'days_fallback_threshold': 400
                },
                'symbols': {
                    'symbols': ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC']
                }
            }
            return jsonify(defaults.get(config_type, {}))
    except Exception as e:
        dashboard_logger.error(f"Error loading config {config_type}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/simple-theme.css')
def serve_css():
    """Serve the CSS file"""
    try:
        css_path = os.path.join(os.path.dirname(__file__), 'simple-theme.css')
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        from flask import Response
        return Response(css_content, mimetype='text/css')
    except Exception as e:
        dashboard_logger.error(f"Error serving CSS: {e}")
        return "/* CSS file not found */", 404

if __name__ == '__main__':
    dashboard_logger.info("Starting Enhanced Trading Scanners Dashboard Server...")
    
    # Create templates directory and copy dashboard
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    # Copy progressive dashboard to templates if it exists
    progressive_src = os.path.join(os.path.dirname(__file__), 'progressive_dashboard.html')
    progressive_dst = os.path.join(templates_dir, 'progressive_dashboard.html')
    
    if os.path.exists(progressive_src):
        shutil.copy2(progressive_src, progressive_dst)
        dashboard_logger.info("Progressive dashboard template copied")
    
    # Copy original dashboard as fallback
    dashboard_src = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    dashboard_dst = os.path.join(templates_dir, 'dashboard.html')

    if os.path.exists(dashboard_src):
        shutil.copy2(dashboard_src, dashboard_dst)
        dashboard_logger.info("Original dashboard template copied as fallback")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'

    dashboard_logger.info(f"Dashboard available at: http://localhost:{port}")
    dashboard_logger.info("Enhanced features: Progressive loading, Status tracking, Retry mechanism, Logging")

    app.run(debug=debug, host='0.0.0.0', port=port)