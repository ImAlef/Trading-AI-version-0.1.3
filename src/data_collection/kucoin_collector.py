import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import sqlite3
import json
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KuCoinDataCollector:
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        self.symbols = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
            'SOL-USDT', 'AVAX-USDT', 'DOT-USDT', 
            'LINK-USDT', 'UNI-USDT'
        ]
        self.timeframes = ['1min', '5min', '15min', '1hour', '4hour', '1day']
        self.db_path = 'crypto_data.db'
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for OHLCV data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # Create table for market sentiment data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                fear_greed_index INTEGER,
                market_cap REAL,
                total_volume REAL,
                btc_dominance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def get_klines(self, symbol: str, timeframe: str, start_time: int, end_time: int) -> Optional[List]:
        """Get historical kline data from KuCoin"""
        try:
            endpoint = f"{self.base_url}/api/v1/market/candles"
            params = {
                'symbol': symbol,
                'type': timeframe,
                'startAt': start_time,
                'endAt': end_time
            }
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['code'] == '200000':
                return data['data']
            else:
                logger.error(f"API Error: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
            return None
    
    def save_ohlcv_data(self, symbol: str, timeframe: str, klines: List):
        """Save OHLCV data to database"""
        if not klines:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_to_insert = []
        for kline in klines:
            # KuCoin returns: [timestamp, open, close, high, low, volume, turnover]
            timestamp = int(kline[0])
            open_price = float(kline[1])
            close_price = float(kline[2])
            high_price = float(kline[3])
            low_price = float(kline[4])
            volume = float(kline[5])
            
            data_to_insert.append((symbol, timeframe, timestamp, open_price, high_price, low_price, close_price, volume))
        
        try:
            cursor.executemany('''
                INSERT OR IGNORE INTO ohlcv_data 
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            
            conn.commit()
            logger.info(f"Saved {len(data_to_insert)} records for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
        finally:
            conn.close()
    
    def get_fear_greed_index(self) -> Optional[Dict]:
        """Get Fear & Greed Index from Alternative.me API"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                return data['data'][0]
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None
    
    def get_global_market_data(self) -> Optional[Dict]:
        """Get global cryptocurrency market data"""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching global market data: {e}")
            return None
    
    def save_sentiment_data(self):
        """Save market sentiment data"""
        fear_greed = self.get_fear_greed_index()
        global_data = self.get_global_market_data()
        
        if not fear_greed or not global_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            timestamp = int(time.time())
            fear_greed_value = int(fear_greed['value'])
            market_cap = global_data['data']['total_market_cap']['usd']
            total_volume = global_data['data']['total_volume']['usd']
            btc_dominance = global_data['data']['market_cap_percentage']['btc']
            
            cursor.execute('''
                INSERT OR IGNORE INTO sentiment_data 
                (timestamp, fear_greed_index, market_cap, total_volume, btc_dominance)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, fear_greed_value, market_cap, total_volume, btc_dominance))
            
            conn.commit()
            logger.info("Sentiment data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
        finally:
            conn.close()
    
    def collect_historical_data(self, days_back: int = 365):
        """Collect historical data for all symbols and timeframes"""
        logger.info(f"Starting historical data collection for {days_back} days")
        
        end_time = int(time.time())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        for symbol in self.symbols:
            logger.info(f"Collecting data for {symbol}")
            
            for timeframe in self.timeframes:
                logger.info(f"  - Timeframe: {timeframe}")
                
                # KuCoin has rate limits, so we need to be careful
                time.sleep(0.1)  # 100ms delay between requests
                
                klines = self.get_klines(symbol, timeframe, start_time, end_time)
                if klines:
                    self.save_ohlcv_data(symbol, timeframe, klines)
                
                # Longer delay for higher timeframes to respect rate limits
                if timeframe in ['1hour', '4hour', '1day']:
                    time.sleep(0.5)
        
        # Collect sentiment data
        self.save_sentiment_data()
        logger.info("Historical data collection completed")
    
    def collect_realtime_data(self):
        """Collect real-time data (for live monitoring)"""
        logger.info("Collecting real-time data")
        
        end_time = int(time.time())
        start_time = end_time - (2 * 60 * 60)  # Last 2 hours
        
        for symbol in self.symbols:
            for timeframe in ['1min', '5min']:  # Only short timeframes for real-time
                time.sleep(0.1)
                
                klines = self.get_klines(symbol, timeframe, start_time, end_time)
                if klines:
                    self.save_ohlcv_data(symbol, timeframe, klines)
        
        # Update sentiment data
        self.save_sentiment_data()
        logger.info("Real-time data collection completed")
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get latest data from database for analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_data_summary(self) -> Dict:
        """Get summary of collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, timeframe, COUNT(*) as count, 
                   MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM ohlcv_data 
            GROUP BY symbol, timeframe
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {}
        for row in results:
            symbol, timeframe, count, earliest, latest = row
            if symbol not in summary:
                summary[symbol] = {}
            
            summary[symbol][timeframe] = {
                'count': count,
                'earliest': datetime.fromtimestamp(earliest).strftime('%Y-%m-%d %H:%M:%S'),
                'latest': datetime.fromtimestamp(latest).strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return summary

# Example usage
if __name__ == "__main__":
    collector = KuCoinDataCollector()
    
    # Collect historical data
    collector.collect_historical_data(days_back=365)
    
    # Print data summary
    summary = collector.get_data_summary()
    print(json.dumps(summary, indent=2))