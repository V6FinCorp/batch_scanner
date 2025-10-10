"""
Progressive Scanner Module
Implements optimized progressive loading for better user experience
"""

import os
import json
import asyncio
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scanner_manager import ScannerManager
import logging

logger = logging.getLogger(__name__)

class ProgressiveScanner:
    """Handles progressive analysis with optimized performance"""
    
    def __init__(self):
        self.scanner_manager = ScannerManager()
        self.analysis_cache = {}
        self.symbol_status = {}
        self.analysis_queue = []
        self.is_running = False
        
    def get_symbol_table(self) -> Dict[str, Any]:
        """Get initial symbol table with basic info"""
        try:
            symbols = self.scanner_manager.load_centralized_symbols()
            symbol_table = []
            
            for symbol in symbols:
                # Check if we have cached data
                has_rsi = self._has_cached_data('rsi', symbol)
                has_ema = self._has_cached_data('ema', symbol)
                has_dma = self._has_cached_data('dma', symbol)
                
                symbol_info = {
                    'Symbol': symbol,
                    'Status': 'Ready' if any([has_rsi, has_ema, has_dma]) else 'Pending',
                    'RSI_Available': has_rsi,
                    'EMA_Available': has_ema,
                    'DMA_Available': has_dma,
                    'LastUpdate': self._get_last_update(symbol) or 'Not calculated'
                }
                
                # Try to get basic price info if available
                try:
                    price_info = self._get_basic_price_info(symbol)
                    if price_info:
                        symbol_info.update(price_info)
                except:
                    symbol_info['CMP'] = 'Loading...'

                # Try to pull summary values from latest CSV rows for each scanner
                try:
                    rsi_row = self._load_symbol_csv('rsi', symbol)
                    if rsi_row:
                        # Common RSI periods if present
                        for p in [15, 30, 60]:
                            key = f'rsi_{p}'
                            if key in rsi_row and rsi_row[key] is not None:
                                symbol_info[key] = rsi_row[key]
                except Exception:
                    pass

                try:
                    ema_row = self._load_symbol_csv('ema', symbol)
                    if ema_row:
                        for p in [9, 15, 65, 200]:
                            key = f'ema_{p}'
                            if key in ema_row and ema_row[key] is not None:
                                symbol_info[key] = ema_row[key]
                except Exception:
                    pass

                try:
                    dma_row = self._load_symbol_csv('dma', symbol)
                    if dma_row:
                        for p in [10, 20, 50]:
                            key = f'dma_{p}'
                            if key in dma_row and dma_row[key] is not None:
                                symbol_info[key] = dma_row[key]
                except Exception:
                    pass
                
                symbol_table.append(symbol_info)
            
            return {
                'success': True,
                'symbols': symbol_table,
                'total_symbols': len(symbols),
                'available_count': sum(1 for s in symbol_table if s['Status'] == 'Ready')
            }
            
        except Exception as e:
            logger.exception('Error creating symbol table')
            return {'success': False, 'error': str(e)}
    
    def get_symbol_analysis_progressive(self, scanner_type: str, symbol: str) -> Dict[str, Any]:
        """Get analysis for specific symbol with progressive loading"""
        try:
            # Check cache first
            cache_key = f"{scanner_type}_{symbol}"
            if cache_key in self.analysis_cache:
                cached_data = self.analysis_cache[cache_key]
                # Return cached data if less than 5 minutes old
                if (datetime.now() - cached_data['timestamp']).seconds < 300:
                    return {
                        'success': True,
                        'data': cached_data['data'],
                        'cached': True,
                        'cache_age': (datetime.now() - cached_data['timestamp']).seconds
                    }
            
            # Try to load from existing CSV
            analysis_data = self._load_symbol_csv(scanner_type, symbol)
            if analysis_data:
                # Cache the result
                self.analysis_cache[cache_key] = {
                    'data': analysis_data,
                    'timestamp': datetime.now()
                }
                
                return {
                    'success': True,
                    'data': analysis_data,
                    'cached': False
                }
            
            # Return pending status if no data available
            return {
                'success': True,
                'data': {
                    'Symbol': symbol,
                    'Status': 'No data available',
                    'Message': f'{scanner_type.upper()} analysis not calculated yet'
                },
                'cached': False
            }
            
        except Exception as e:
            logger.exception(f'Error getting progressive analysis for {symbol}')
            return {'success': False, 'error': str(e)}
    
    def run_single_symbol_analysis(self, scanner_type: str, symbol: str, **params) -> Dict[str, Any]:
        """Run analysis for a single symbol (optimized for individual calculations)"""
        try:
            logger.info(f"Running {scanner_type} analysis for {symbol}")
            
            # Update status
            self.symbol_status[symbol] = {
                'status': 'Running',
                'scanner': scanner_type,
                'start_time': datetime.now()
            }
            
            # Run analysis for single symbol only
            result = self.scanner_manager.run_scanner(
                scanner_type=scanner_type,
                symbols=[symbol],  # Only analyze this symbol
                **params
            )
            
            if result.get('returncode') == 0:
                # Update status and cache
                self.symbol_status[symbol] = {
                    'status': 'Completed',
                    'scanner': scanner_type,
                    'completion_time': datetime.now()
                }
                
                # Try to get the result
                analysis_data = self._load_symbol_csv(scanner_type, symbol)
                if analysis_data:
                    cache_key = f"{scanner_type}_{symbol}"
                    self.analysis_cache[cache_key] = {
                        'data': analysis_data,
                        'timestamp': datetime.now()
                    }
                    
                return {
                    'success': True,
                    'symbol': symbol,
                    'scanner': scanner_type,
                    'data': analysis_data,
                    'message': f'{scanner_type.upper()} analysis completed for {symbol}'
                }
            else:
                # Update status with error
                self.symbol_status[symbol] = {
                    'status': 'Error',
                    'scanner': scanner_type,
                    'error': result.get('error', 'Unknown error')
                }
                
                return {
                    'success': False,
                    'symbol': symbol,
                    'scanner': scanner_type,
                    'error': result.get('error', 'Analysis failed')
                }
                
        except Exception as e:
            logger.exception(f'Error running single symbol analysis for {symbol}')
            self.symbol_status[symbol] = {
                'status': 'Error',
                'scanner': scanner_type,
                'error': str(e)
            }
            return {'success': False, 'symbol': symbol, 'error': str(e)}
    
    def _has_cached_data(self, scanner_type: str, symbol: str) -> bool:
        """Check if cached data exists for symbol and scanner type"""
        try:
            if scanner_type == 'rsi':
                csv_filename = f'{symbol}_multi_timeframe_rsi_data.csv'
            else:
                csv_filename = f'{symbol}_{scanner_type}_data.csv'
                
            csv_file = os.path.join(self.scanner_manager.scanners_dir, 'data', symbol, csv_filename)
            return os.path.exists(csv_file)
        except:
            return False
    
    def _get_last_update(self, symbol: str) -> Optional[str]:
        """Get last update time for symbol data"""
        try:
            data_dir = os.path.join(self.scanner_manager.scanners_dir, 'data', symbol)
            if not os.path.exists(data_dir):
                return None
                
            # Find the most recent CSV file
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if not csv_files:
                return None
                
            latest_time = 0
            for csv_file in csv_files:
                file_path = os.path.join(data_dir, csv_file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
            
            return datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    def _get_basic_price_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get basic price info from any available CSV"""
        try:
            data_dir = os.path.join(self.scanner_manager.scanners_dir, 'data', symbol)
            if not os.path.exists(data_dir):
                return None
                
            # Try to find any CSV with price data
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            for csv_file in csv_files:
                try:
                    csv_path = os.path.join(data_dir, csv_file)
                    df = pd.read_csv(csv_path)
                    
                    if not df.empty and 'close' in df.columns:
                        latest_price = df['close'].iloc[-1]
                        if not pd.isna(latest_price):
                            return {
                                'CMP': f"₹{latest_price:.2f}",
                                'Source': csv_file.replace('.csv', '').replace(f'{symbol}_', '').upper()
                            }
                except:
                    continue
            
            return None
        except:
            return None
    
    def _load_symbol_csv(self, scanner_type: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Load and parse CSV data for a symbol"""
        try:
            if scanner_type == 'rsi':
                csv_filename = f'{symbol}_multi_timeframe_rsi_data.csv'
            else:
                csv_filename = f'{symbol}_{scanner_type}_data.csv'
                
            csv_file = os.path.join(self.scanner_manager.scanners_dir, 'data', symbol, csv_filename)
            
            if not os.path.exists(csv_file):
                return None
                
            df = pd.read_csv(csv_file)
            df = df.replace(['N/A', 'NaN', 'nan'], None)
            
            if df.empty:
                return None
                
            # Get the latest row
            latest_row = df.iloc[-1:].copy()
            if 'Symbol' not in latest_row.columns:
                latest_row['Symbol'] = symbol
                
            # Convert to dict and clean numpy types
            row_dict = latest_row.to_dict('records')[0]
            for key, value in list(row_dict.items()):
                if pd.isna(value) or value is None:
                    row_dict[key] = None
                elif hasattr(value, 'item'):
                    row_dict[key] = value.item()
                elif isinstance(value, (np.int64, np.float64)):
                    row_dict[key] = value.item()
                    
            return row_dict
            
        except Exception as e:
            logger.error(f"Error loading CSV for {symbol}: {e}")
            return None
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status for all symbols"""
        try:
            return {
                'success': True,
                'symbol_status': self.symbol_status,
                'cache_size': len(self.analysis_cache),
                'is_running': self.is_running
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear analysis cache"""
        try:
            cleared_count = len(self.analysis_cache)
            self.analysis_cache.clear()
            self.symbol_status.clear()
            
            return {
                'success': True,
                'message': f'Cleared {cleared_count} cached entries'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}