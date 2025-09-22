"""
Trading Scanners Dashboard Server
Serves the HTML dashboard and handles scanner execution via API endpoints.
"""

import os
import json
import sys
import shutil
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

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

# Import scanner manager from current directory (repository root)
try:
    from scanner_manager import ScannerManager
except ImportError:
    print("Warning: Could not import ScannerManager")
    ScannerManager = None

# Initialize scanner manager
if ScannerManager:
    scanner_manager = ScannerManager()
else:
    scanner_manager = None
    print("Error: ScannerManager not available")

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        # Fallback: serve the file directly
        with open('dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()

@app.route('/api/run-scanner', methods=['POST'])
def run_scanner():
    """API endpoint to run a scanner"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
        
    try:
        data = request.get_json()

        scanner_type = data.get('scanner', 'rsi')
        symbols = data.get('symbols', ['RELIANCE'])
        base_timeframe = data.get('baseTimeframe', '15mins')
        days_to_list = data.get('daysToList', 2)

        # Extract additional parameters
        kwargs = {
            'daysFallbackThreshold': data.get('daysFallbackThreshold', 200),
            'rsiPeriods': data.get('rsiPeriods', [15, 30, 60]),
            'emaPeriods': data.get('emaPeriods', [9, 15, 65, 200]),
            'dmaPeriods': data.get('dmaPeriods', [10, 20, 50])
        }

        # Run scanner using scanner manager
        result = scanner_manager.run_scanner(
            scanner_type=scanner_type,
            symbols=symbols,
            base_timeframe=base_timeframe,
            days_to_list=days_to_list,
            **kwargs
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scanner-status')
def scanner_status():
    """Get status of available scanners"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    return jsonify(scanner_manager.get_scanner_status())

@app.route('/api/symbols')
def get_symbols():
    """Get available symbols from JSON file"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    return jsonify(scanner_manager.get_symbols())

@app.route('/api/scanner-results/<scanner_type>')
def get_scanner_results(scanner_type):
    """Get stored results for a specific scanner type"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    return jsonify(scanner_manager.get_scanner_results(scanner_type))

@app.route('/api/chart-data/<scanner_type>/<symbol>')
def get_chart_data(scanner_type, symbol):
    """Get chart data for a specific symbol and scanner type"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    return jsonify(scanner_manager.get_symbol_chart_data(scanner_type, symbol))

@app.route('/api/config/<config_type>')
def get_config(config_type):
    """Get config data for a specific indicator type"""
    try:
        config_filename = f'{config_type}.config.json' if config_type == 'symbols' else f'{config_type}_config.json'
        config_path = os.path.join(os.path.dirname(__file__), 'config', config_filename)
        
        print(f"Loading config: {config_type} from {config_path}")
        print(f"Config file exists: {os.path.exists(config_path)}")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"Config data loaded: {config_data}")
            return jsonify(config_data)
        else:
            # Return default values if config file doesn't exist
            defaults = {
                'rsi': {
                    'symbols': ['ITC'],
                    'rsi_periods': [15, 30, 60],
                    'base_timeframe': '15mins',
                    'days_to_list': 2,
                    'days_fallback_threshold': 200,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                },
                'ema': {
                    'symbols': ['RELIANCE'],
                    'ema_periods': [9, 15, 65, 200],
                    'base_timeframe': '15mins',
                    'days_to_list': 2,
                    'days_fallback_threshold': 200
                },
                'dma': {
                    'symbols': ['RELIANCE'],
                    'dma_periods': [10, 20, 50],
                    'base_timeframe': '1hour',
                    'days_to_list': 2,
                    'days_fallback_threshold': 1600
                },
                'symbols': {
                    'symbols': ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC']
                }
            }
            return jsonify(defaults.get(config_type, {}))
    except Exception as e:
        print(f"Error loading config {config_type}: {e}")
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
        print(f"Error serving CSS: {e}")
        return "/* CSS file not found */", 404

if __name__ == '__main__':
    print("Starting Trading Scanners Dashboard Server...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Scanners directory: {os.path.dirname(__file__)}")

    # Create templates directory and copy dashboard
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    # Copy dashboard.html to templates
    dashboard_src = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    dashboard_dst = os.path.join(templates_dir, 'dashboard.html')

    if os.path.exists(dashboard_src):
        shutil.copy2(dashboard_src, dashboard_dst)
        print("Dashboard template copied successfully")
    else:
        print("Warning: dashboard.html not found")
    
    # Check if CSS file exists
    css_file = os.path.join(os.path.dirname(__file__), 'simple-theme.css')
    if os.path.exists(css_file):
        print("CSS file found: simple-theme.css")
    else:
        print("Warning: simple-theme.css not found")

    # Get port from environment variable (Railway.app)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'

    print("Starting Trading Scanners Dashboard Server...")
    print(f"Dashboard available at: http://localhost:{port}")
    print("API endpoints:")
    print("   POST /api/run-scanner - Run a scanner")
    print("   GET  /api/scanner-status - Get scanner status")

    app.run(debug=debug, host='0.0.0.0', port=port)