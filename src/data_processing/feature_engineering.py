import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, List, Tuple, Optional
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, db_path: str = 'crypto_data.db'):
        self.db_path = db_path
        self.scalers = {}
        self.symbols = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
            'SOL-USDT', 'MATIC-USDT', 'AVAX-USDT', 'DOT-USDT', 
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
    
    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df
        
        # Price-based indicators
        df['sma_7'] = talib.SMA(df['close'], timeperiod=7)
        df['sma_14'] = talib.SMA(df['close'], timeperiod=14)
        df['sma_21'] = talib.SMA(df['close'], timeperiod=21)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        df['ema_7'] = talib.EMA(df['close'], timeperiod=7)
        df['ema_14'] = talib.EMA(df['close'], timeperiod=14)
        df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        if df.empty:
            return df
        
        # RSI
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
        
        # ROC (Rate of Change)
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        if df.empty:
            return df
        
        # Volume SMA
        df['volume_sma_7'] = talib.SMA(df['volume'], timeperiod=7)
        df['volume_sma_14'] = talib.SMA(df['volume'], timeperiod=14)
        df['volume_ratio'] = df['volume'] / df['volume_sma_14']
        
        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=10)
        
        # Accumulation/Distribution Line
        df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = self._calculate_cmf(df)
        
        return df
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_multiplier = mf_multiplier.fillna(0)
        mf_volume = mf_multiplier * df['volume']
        cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return cmf
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        if df.empty:
            return df
        
        # Average True Range
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_7'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=7)
        
        # True Range
        df['tr'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Historical Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_10'] = df['returns'].rolling(window=10).std() * np.sqrt(10)
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(20)
        
        return df
    
    def calculate_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price patterns and candlestick patterns"""
        if df.empty:
            return df
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_10'] = df['close'].pct_change(periods=10)
        
        # High/Low ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        # Body and shadow ratios
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_shadow'] = df['upper_shadow'] + df['lower_shadow']
        
        # Candlestick patterns (selected most important ones)
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        return df
    
    def create_multi_timeframe_features(self, symbol: str) -> pd.DataFrame:
        """Create features combining multiple timeframes"""
        logger.info(f"Creating multi-timeframe features for {symbol}")
        
        # Load data for different timeframes
        timeframe_data = {}
        for tf in self.timeframes:
            df = self.load_data_from_db(symbol, tf, limit=5000)
            if not df.empty:
                # Calculate all indicators
                df = self.calculate_basic_indicators(df)
                df = self.calculate_momentum_indicators(df)
                df = self.calculate_volume_indicators(df)
                df = self.calculate_volatility_indicators(df)
                df = self.calculate_price_patterns(df)
                timeframe_data[tf] = df
        
        if not timeframe_data:
            return pd.DataFrame()
        
        # Use 1min as base timeframe
        base_tf = '1min'
        if base_tf not in timeframe_data:
            base_tf = list(timeframe_data.keys())[0]
        
        base_df = timeframe_data[base_tf].copy()
        
        # Merge higher timeframe data
        for tf, df in timeframe_data.items():
            if tf == base_tf:
                continue
            
            # Resample higher timeframe data to base timeframe
            df_resampled = self._resample_to_base_timeframe(df, base_df.index, tf)
            
            # Add suffix to column names
            df_resampled = df_resampled.add_suffix(f'_{tf}')
            
            # Merge with base dataframe
            base_df = pd.concat([base_df, df_resampled], axis=1)
        
        return base_df
    
    def _resample_to_base_timeframe(self, df: pd.DataFrame, target_index: pd.DatetimeIndex, timeframe: str) -> pd.DataFrame:
        """Resample higher timeframe data to match base timeframe"""
        # Forward fill to match the target index
        df_resampled = df.reindex(target_index, method='ffill')
        
        # Select only indicator columns (exclude OHLCV)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        indicator_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df_resampled[indicator_cols]
    
    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market sentiment features"""
        conn = sqlite3.connect(self.db_path)
        
        # Load sentiment data
        sentiment_query = '''
            SELECT timestamp, fear_greed_index, market_cap, total_volume, btc_dominance
            FROM sentiment_data
            ORDER BY timestamp ASC
        '''
        
        sentiment_df = pd.read_sql_query(sentiment_query, conn)
        conn.close()
        
        if sentiment_df.empty:
            logger.warning("No sentiment data found")
            return df
        
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], unit='s')
        sentiment_df = sentiment_df.set_index('timestamp')
        
        # Resample sentiment data to match price data frequency
        sentiment_resampled = sentiment_df.reindex(df.index, method='ffill')
        
        # Add sentiment features
        df['fear_greed_index'] = sentiment_resampled['fear_greed_index']
        df['market_cap'] = sentiment_resampled['market_cap']
        df['total_volume_market'] = sentiment_resampled['total_volume']
        df['btc_dominance'] = sentiment_resampled['btc_dominance']
        
        # Calculate sentiment indicators
        df['fear_greed_ma_7'] = df['fear_greed_index'].rolling(window=7).mean()
        df['fear_greed_change'] = df['fear_greed_index'].pct_change()
        df['btc_dominance_change'] = df['btc_dominance'].pct_change()
        
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
    
    def normalize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize features for ML model"""
        if df.empty:
            return df
        
        # Select numeric columns for normalization (exclude target columns)
        exclude_cols = ['signal', 'entry_price', 'take_profit', 'stop_loss', 'hold_period']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if fit_scaler:
            # Fit new scaler
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
            self.scalers['features'] = scaler
        else:
            # Use existing scaler
            if 'features' in self.scalers:
                df[feature_cols] = self.scalers['features'].transform(df[feature_cols].fillna(0))
            else:
                logger.warning("No fitted scaler found, fitting new one")
                self.normalize_features(df, fit_scaler=True)
        
        return df
    
    def process_symbol_data_for_live(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process symbol data for live trading (simplified version)"""
        if df.empty:
            return df
        
        # Calculate basic indicators
        df = self.calculate_basic_indicators(df)
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_volatility_indicators(df)
        df = self.calculate_price_patterns(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def process_symbol_data_for_backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process symbol data for backtesting"""
        return self.process_symbol_data_for_live(df, symbol)
        """Complete feature engineering pipeline for a symbol"""
        logger.info(f"Processing features for {symbol}")
        
        # Create multi-timeframe features
        df = self.create_multi_timeframe_features(symbol)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return df
        
        # Add sentiment features
        df = self.add_sentiment_features(df)
        
        # Create target labels
        df = self.create_target_labels(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Normalize features
        df = self.normalize_features(df)
        
        logger.info(f"Processed {len(df)} samples for {symbol}")
        return df
    
    def prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare training data for all symbols"""
        logger.info("Preparing training data for all symbols")
        
        training_data = {}
        for symbol in self.symbols:
            df = self.process_symbol_data(symbol)
            if not df.empty:
                training_data[symbol] = df
        
        logger.info(f"Training data prepared for {len(training_data)} symbols")
        return training_data

# Example usage
if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Process single symbol
    df = engineer.process_symbol_data('BTC-USDT')
    print(f"Features shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check target distribution
    if not df.empty:
        print(f"\nSignal distribution:")
        print(df['signal'].value_counts())