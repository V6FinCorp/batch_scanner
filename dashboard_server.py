"""
Trading Scanners Dashboard Server
Serves the HTML dashboard and handles scanner execution via API endpoints.
"""

import os
import json
import sys
import time
import shutil
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

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
# -- Utilities ---------------------------------------------------------------
def _ensure_utf8_template(template_filename: str) -> None:
    """Ensure the given template file is saved as UTF-8 without BOM.

    Some editors (e.g., Notepad) save HTML as UTF-16 with a BOM (starts with 0xFF 0xFE or 0xFE 0xFF),
    which causes Jinja to fail decoding with utf-8. This converts such files in-place to UTF-8.
    """
    try:
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        file_path = os.path.join(templates_dir, template_filename)
        if not os.path.exists(file_path):
            return
        # Read raw bytes to detect BOM/encoding
        with open(file_path, 'rb') as bf:
            raw = bf.read()
        if not raw:
            return
        # Detect encodings by BOM
        decoded = None
        if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
            # UTF-16 (LE/BE)
            try:
                decoded = raw.decode('utf-16')
            except Exception:
                pass
        elif raw.startswith(b'\xef\xbb\xbf'):
            # UTF-8 with BOM
            try:
                decoded = raw.decode('utf-8-sig')
            except Exception:
                pass
        else:
            # Try plain UTF-8 first
            try:
                decoded = raw.decode('utf-8')
            except Exception:
                # Last resort, decode as latin-1 and re-encode to utf-8
                try:
                    decoded = raw.decode('latin-1')
                except Exception:
                    decoded = None
        if decoded is None:
            return
        # If we decoded using non-utf8 or utf8-sig, re-save as clean UTF-8
        try:
            # Re-encode always to ensure consistent file state
            with open(file_path, 'w', encoding='utf-8', newline='') as tf:
                tf.write(decoded)
        except Exception:
            # If write fails, leave file as-is
            pass
    except Exception:
        # Never raise from helper
        pass


# Configure logging to file for diagnostics and /api/logs endpoint
logger = logging.getLogger('dashboard_server')
logger.setLevel(logging.INFO)
try:
    log_path = os.path.join(os.path.dirname(__file__), 'dashboard.log')
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Also log to console
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info('Logging initialized')
except Exception as e:
    print(f"Warning: Failed to initialize file logging: {e}")

# Import scanner manager from current directory (repository root)
try:
    from scanner_manager import ScannerManager
    print("✅ ScannerManager loaded")
except ImportError:
    print("❌ Warning: Could not import ScannerManager")
    ScannerManager = None

# REAL CHANGE: Import progressive scanner for optimizations
try:
    from progressive_scanner import ProgressiveScanner
    progressive_available = True
    print("✅ Progressive scanner module loaded")
except ImportError:
    progressive_available = False
    print("⚠️ Progressive scanner not available, using fallback")

# Initialize scanner manager
if ScannerManager:
    scanner_manager = ScannerManager()
    progressive_scanner = ProgressiveScanner() if progressive_available else None
    print("✅ Scanner manager initialized")
else:
    scanner_manager = None
    progressive_scanner = None
    print("Error: ScannerManager not available")

@app.route('/')
def dashboard():
    """Serve the progressive dashboard via Flask templates with safe fallback."""
    try:
        _ensure_utf8_template('progressive_dashboard.html')
        # Prefer rendering from templates folder to avoid manual encoding issues
        return render_template('progressive_dashboard.html', cache_bust=int(time.time()))
    except Exception as e:
        print(f"⚠️ Failed to render progressive dashboard, falling back to legacy: {e}")
        try:
            _ensure_utf8_template('dashboard.html')
            return render_template('dashboard.html', cache_bust=int(time.time()))
        except Exception as e2:
            print(f"❌ Failed to render legacy dashboard too: {e2}")
            return f"Error rendering dashboard: {e} / {e2}", 500

@app.route('/progressive')
def progressive():
    try:
        _ensure_utf8_template('progressive_dashboard.html')
        return render_template('progressive_dashboard.html', cache_bust=int(time.time()))
    except Exception as e:
        return f"Progressive template error: {e}", 500

@app.route('/legacy')
def legacy():
    try:
        _ensure_utf8_template('dashboard.html')
        return render_template('dashboard.html', cache_bust=int(time.time()))
    except Exception as e:
        return f"Legacy template error: {e}", 500

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

@app.route('/api/symbol-analysis/<scanner_type>/<symbol>')
def get_symbol_analysis(scanner_type, symbol):
    """Get analysis for a specific symbol and scanner type with caching"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
    return jsonify(scanner_manager.get_symbol_analysis(scanner_type, symbol))

@app.route('/api/progressive-analysis', methods=['POST'])
def run_progressive_analysis():
    """Run progressive analysis for all symbols"""
    if not scanner_manager:
        return jsonify({'error': 'Scanner manager not available'}), 500
        
    try:
        data = request.get_json()
        scanner_type = data.get('scanner', 'rsi')
        base_timeframe = data.get('baseTimeframe', '15mins')
        days_to_list = data.get('daysToList', 2)
        
        # Extract additional parameters
        kwargs = {
            'daysFallbackThreshold': data.get('daysFallbackThreshold', 200),
            'rsiPeriods': data.get('rsiPeriods', [15, 30, 60]),
            'emaPeriods': data.get('emaPeriods', [9, 15, 65, 200]),
            'dmaPeriods': data.get('dmaPeriods', [10, 20, 50])
        }
        
        result = scanner_manager.run_progressive_analysis(
            scanner_type=scanner_type,
            base_timeframe=base_timeframe,
            days_to_list=days_to_list,
            **kwargs
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
                    'rsi_periods': [15, 30, 60],
                    'base_timeframe': '15mins',
                    'days_to_list': 2,
                    'days_fallback_threshold': 200,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                },
                'ema': {
                    'ema_periods': [9, 15, 65, 200],
                    'base_timeframe': '15mins',
                    'days_to_list': 2,
                    'days_fallback_threshold': 200
                },
                'dma': {
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

@app.route('/api/symbol-table')
def get_symbol_table():
    """Get symbol table for progressive loading using ProgressiveScanner."""
    try:
        print("📊 API CALL: /api/symbol-table requested")
        if progressive_scanner:
            result = progressive_scanner.get_symbol_table()
            status = 200 if result.get('success') else 500
            return jsonify(result), status
        # Fallback to basic list via ScannerManager
        if not scanner_manager:
            return jsonify({'success': False, 'error': 'Scanner manager not available'}), 500
        symbols = scanner_manager.get_symbols().get('symbols', [])
        return jsonify({
            'success': True,
            'symbols': [{'Symbol': s, 'Status': 'Pending', 'CMP': 'Loading...', 'RSI_Available': False, 'EMA_Available': False, 'DMA_Available': False, 'LastUpdate': 'Not calculated'} for s in symbols],
            'total_symbols': len(symbols)
        })
    except Exception as e:
        print(f"❌ ERROR in /api/symbol-table: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/single-analysis/<scanner_type>/<symbol>', methods=['POST'])
def run_single_analysis(scanner_type, symbol):
    """Legacy: Run analysis for a single symbol via path params."""
    try:
        if not progressive_scanner:
            return jsonify({'error': 'Progressive scanner not available'}), 500
        data = request.get_json() or {}
        params = {
            'base_timeframe': data.get('baseTimeframe', '15mins'),
            'days_to_list': data.get('daysToList', 2)
        }
        if scanner_type == 'rsi':
            params.update({
                'rsiPeriods': data.get('rsiPeriods', [15, 30, 60]),
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
        result = progressive_scanner.run_single_symbol_analysis(scanner_type, symbol, **params)
        # Normalize response to include returncode for frontend
        status_code = 200 if result.get('success') else 500
        norm = {
            **result,
            'returncode': 0 if result.get('success') else 1
        }
        return jsonify(norm), status_code
    except Exception as e:
        return jsonify({'error': str(e), 'returncode': 1}), 500

@app.route('/api/run-single-scanner', methods=['POST'])
def run_single_scanner():
    """Run a single scanner for a single symbol (used by progressive UI)."""
    try:
        if not progressive_scanner:
            return jsonify({'error': 'Progressive scanner not available', 'returncode': 1}), 500
        data = request.get_json() or {}
        scanner_type = data.get('scanner')
        symbol = data.get('symbol')
        if not scanner_type or not symbol:
            return jsonify({'error': 'scanner and symbol are required', 'returncode': 1}), 400
        params = {
            'base_timeframe': data.get('baseTimeframe', '15mins'),
            'days_to_list': data.get('daysToList', 2),
            'rsiPeriods': data.get('rsiPeriods', [15, 30, 60]),
            'emaPeriods': data.get('emaPeriods', [9, 15, 65, 200]),
            'dmaPeriods': data.get('dmaPeriods', [10, 20, 50]),
            'daysFallbackThreshold': data.get('daysFallbackThreshold', 200),
            'rsiOverbought': data.get('rsiOverbought', 70),
            'rsiOversold': data.get('rsiOversold', 30)
        }
        result = progressive_scanner.run_single_symbol_analysis(scanner_type, symbol, **params)
        status_code = 200 if result.get('success') else 500
        return jsonify({**result, 'returncode': 0 if result.get('success') else 1}), status_code
    except Exception as e:
        return jsonify({'error': str(e), 'returncode': 1}), 500

@app.route('/api/logs')
def get_logs():
    """Return recent application logs for the modal viewer."""
    try:
        log_file = os.path.join(os.path.dirname(__file__), 'dashboard.log')
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Limit to last ~10k characters to avoid huge payloads
                tail = content[-10000:]
                logs = tail.splitlines()
        else:
            # Provide a minimal log from runtime info
            logs = [
                'dashboard.log not found. Showing runtime info:',
                f"Python: {sys.version}",
                f"CWD: {os.getcwd()}",
                f"Scanners dir: {os.path.dirname(__file__)}"
            ]
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear-logs', methods=['POST'])
def clear_logs():
    """Clear the dashboard log file if present."""
    try:
        log_file = os.path.join(os.path.dirname(__file__), 'dashboard.log')
        if os.path.exists(log_file):
            open(log_file, 'w').close()
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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