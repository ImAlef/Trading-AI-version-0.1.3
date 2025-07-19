import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFeatureEngineer:
    """Simplified feature engineering without TA-Lib dependency"""
    
    def __init__(self, db_path: str = 'crypto_data.db'):
        self.db_path = db_path
        self.scalers = {}
        self.symbols = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
            'SOL-USDT', 'AVAX-USDT', 'DOT-USDT', 
            'LINK-USDT', 'UNI-USDT'
        ]
        self.timeframes = ['1min', '5min', '15min', '1hour', '4hour', '1day']
    
    def load_data_from_db(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """Load OHLCV data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
        
        return df
    
    def calculate_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple technical indicators without TA-Lib"""
        if df.empty or len(df) < 50:
            return df
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Simple Moving Averages
        df.loc[:, 'sma_7'] = df['close'].rolling(window=7).mean()
        df.loc[:, 'sma_14'] = df['close'].rolling(window=14).mean()
        df.loc[:, 'sma_21'] = df['close'].rolling(window=21).mean()
        df.loc[:, 'sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df.loc[:, 'ema_7'] = df['close'].ewm(span=7).mean()
        df.loc[:, 'ema_14'] = df['close'].ewm(span=14).mean()
        df.loc[:, 'ema_21'] = df['close'].ewm(span=21).mean()
        
        # Simple RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df.loc[:, 'rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df.loc[:, 'bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df.loc[:, 'bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df.loc[:, 'bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df.loc[:, 'bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df.loc[:, 'bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df.loc[:, 'volume_sma_14'] = df['volume'].rolling(window=14).mean()
        df.loc[:, 'volume_ratio'] = df['volume'] / df['volume_sma_14']
        
        # Price changes
        df.loc[:, 'price_change'] = df['close'].pct_change()
        df.loc[:, 'price_change_5'] = df['close'].pct_change(periods=5)
        df.loc[:, 'price_change_10'] = df['close'].pct_change(periods=10)
        
        # Volatility
        df.loc[:, 'returns'] = df['close'].pct_change()
        df.loc[:, 'volatility_10'] = df['returns'].rolling(window=10).std() * np.sqrt(10)
        df.loc[:, 'volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(20)
        
        # High/Low ratios
        df.loc[:, 'hl_ratio'] = df['high'] / df['low']
        df.loc[:, 'oc_ratio'] = df['open'] / df['close']
        
        # Body size
        df.loc[:, 'body_size'] = abs(df['close'] - df['open']) / df['open']
        
        return df
    
    def create_target_labels(self, df: pd.DataFrame, 
                           min_profit: float = 0.01,  # 1% minimum profit
                           max_loss: float = 0.02,    # 2% maximum loss
                           hold_periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create target labels for training"""
        if df.empty:
            return df
        
        # Initialize target columns
        df['signal'] = 0  # 0: Hold, 1: Long, -1: Short
        df['entry_price'] = df['close']
        df['take_profit'] = 0.0
        df['stop_loss'] = 0.0
        df['hold_period'] = 0
        df['future_return'] = 0.0
        
        for i in range(len(df) - max(hold_periods)):
            current_price = df['close'].iloc[i]
            
            best_signal = 0
            best_profit = 0
            best_tp = 0
            best_sl = 0
            best_period = 0
            
            # Look forward for different holding periods
            for period in hold_periods:
                if i + period >= len(df):
                    continue
                
                future_prices = df['close'].iloc[i+1:i+period+1]
                
                # Calculate potential Long signal
                max_price = future_prices.max()
                min_price = future_prices.min()
                
                long_profit = (max_price - current_price) / current_price
                long_loss = (min_price - current_price) / current_price
                
                # Calculate potential Short signal
                short_profit = (current_price - min_price) / current_price
                short_loss = (max_price - current_price) / current_price
                
                # Check Long signal conditions
                if long_profit >= min_profit and abs(long_loss) <= max_loss:
                    if long_profit > best_profit:
                        best_signal = 1
                        best_profit = long_profit
                        best_tp = max_price
                        best_sl = current_price * (1 - max_loss)
                        best_period = period
                
                # Check Short signal conditions
                if short_profit >= min_profit and abs(short_loss) <= max_loss:
                    if short_profit > best_profit:
                        best_signal = -1
                        best_profit = short_profit
                        best_tp = min_price
                        best_sl = current_price * (1 + max_loss)
                        best_period = period
            
            # Assign the best signal
            df.iloc[i, df.columns.get_loc('signal')] = best_signal
            df.iloc[i, df.columns.get_loc('take_profit')] = best_tp
            df.iloc[i, df.columns.get_loc('stop_loss')] = best_sl
            df.iloc[i, df.columns.get_loc('hold_period')] = best_period
            df.iloc[i, df.columns.get_loc('future_return')] = best_profit
        
        return df
    
    def process_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Complete feature engineering pipeline for a symbol"""
        logger.info(f"Processing features for {symbol}")
        
        # Load data for base timeframe
        base_timeframe = '5min'
        df = self.load_data_from_db(symbol, base_timeframe, limit=2000)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return df
        
        # Calculate indicators
        df = self.calculate_simple_indicators(df)
        
        # Create target labels
        df = self.create_target_labels(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Normalize features
        feature_cols = [col for col in df.columns if col not in 
                       ['signal', 'entry_price', 'take_profit', 'stop_loss', 'hold_period', 'future_return']]
        
        if len(df) > 0:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers[symbol] = scaler
        
        logger.info(f"Processed {len(df)} samples for {symbol}")
        return df
    
    def prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare training data for all symbols"""
        logger.info("Preparing training data for all symbols")
        
        training_data = {}
        for symbol in self.symbols:
            df = self.process_symbol_data(symbol)
            if not df.empty and len(df) > 100:  # Minimum samples required
                training_data[symbol] = df
        
        logger.info(f"Training data prepared for {len(training_data)} symbols")
        return training_data
    
    def process_symbol_data_for_live(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process symbol data for live trading"""
        if df.empty:
            return df
        
        # Calculate indicators
        df = self.calculate_simple_indicators(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def process_symbol_data_for_backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process symbol data for backtesting"""
        return self.process_symbol_data_for_live(df, symbol)

# Example usage
if __name__ == "__main__":
    engineer = SimpleFeatureEngineer()
    
    # Process single symbol
    df = engineer.process_symbol_data('BTC-USDT')
    print(f"Features shape: {df.shape}")
    
    if not df.empty:
        print(f"Signal distribution:")
        print(df['signal'].value_counts())