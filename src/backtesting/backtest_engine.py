import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    signal_type: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    take_profit: float
    stop_loss: float
    quantity: float
    leverage: float
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    status: str = 'OPEN'  # 'OPEN', 'CLOSED', 'TP_HIT', 'SL_HIT'
    confidence: float = 0.0
    commission: float = 0.0

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_balance: float = 10000.0
    leverage: float = 5.0
    position_size_percentage: float = 0.15  # 15% of balance per trade
    commission_rate: float = 0.001  # 0.1% commission
    max_open_positions: int = 5
    confidence_threshold: float = 0.45
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class BacktestEngine:
    def __init__(self, config: BacktestConfig, db_path: str = 'crypto_data.db'):
        self.config = config
        self.db_path = db_path
        self.balance = config.initial_balance
        self.initial_balance = config.initial_balance
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        self.balance_history = []
        self.trade_history = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.max_equity = config.initial_balance
        
    def load_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load historical data from database"""
        conn = sqlite3.connect(self.db_path)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        query = '''
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_timestamp, end_timestamp))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
        
        return df
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management"""
        # Calculate position value
        position_value = self.balance * self.config.position_size_percentage
        
        # Calculate quantity with leverage
        quantity = (position_value * self.config.leverage) / price
        
        return quantity
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        return quantity * price * self.config.commission_rate
    
    def open_position(self, signal_data: dict, current_price: float, 
                     current_time: datetime, symbol: str) -> bool:
        """Open a new trading position"""
        # Check if we can open more positions
        if len(self.open_trades) >= self.config.max_open_positions:
            return False
        
        # Check confidence threshold
        if signal_data['confidence'] < self.config.confidence_threshold:
            return False
        
        # Skip hold signals
        if signal_data['signal'] == 0:
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(current_price)
        
        # Calculate commission
        commission = self.calculate_commission(quantity, current_price)
        
        # Check if we have enough balance
        required_margin = (quantity * current_price) / self.config.leverage + commission
        if required_margin > self.balance:
            return False
        
        # Determine signal type
        signal_type = 'LONG' if signal_data['signal'] == 1 else 'SHORT'
        
        # Create trade
        trade = Trade(
            entry_time=current_time,
            exit_time=None,
            symbol=symbol,
            signal_type=signal_type,
            entry_price=current_price,
            exit_price=None,
            take_profit=signal_data['take_profit'],
            stop_loss=signal_data['stop_loss'],
            quantity=quantity,
            leverage=self.config.leverage,
            confidence=signal_data['confidence'],
            commission=commission
        )
        
        # Update balance
        self.balance -= required_margin
        
        # Add to open trades
        self.open_trades.append(trade)
        
        logger.info(f"Opened {signal_type} position for {symbol} at {current_price:.4f}")
        return True
    
    def check_exit_conditions(self, trade: Trade, current_high: float, 
                            current_low: float, current_close: float, 
                            current_time: datetime) -> bool:
        """Check if trade should be closed based on TP/SL"""
        if trade.signal_type == 'LONG':
            # Check take profit
            if current_high >= trade.take_profit:
                self.close_position(trade, trade.take_profit, current_time, 'TP_HIT')
                return True
            
            # Check stop loss
            if current_low <= trade.stop_loss:
                self.close_position(trade, trade.stop_loss, current_time, 'SL_HIT')
                return True
        
        elif trade.signal_type == 'SHORT':
            # Check take profit (for short, TP is lower)
            if current_low <= trade.take_profit:
                self.close_position(trade, trade.take_profit, current_time, 'TP_HIT')
                return True
            
            # Check stop loss (for short, SL is higher)
            if current_high >= trade.stop_loss:
                self.close_position(trade, trade.stop_loss, current_time, 'SL_HIT')
                return True
        
        return False
    
    def close_position(self, trade: Trade, exit_price: float, 
                      exit_time: datetime, status: str):
        """Close a trading position"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.status = status
        
        # Calculate PnL
        if trade.signal_type == 'LONG':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity * trade.leverage
        else:  # SHORT
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity * trade.leverage
        
        # Subtract exit commission
        exit_commission = self.calculate_commission(trade.quantity, exit_price)
        trade.pnl -= (trade.commission + exit_commission)
        
        # Calculate percentage return
        trade.pnl_percentage = (trade.pnl / (trade.quantity * trade.entry_price / trade.leverage)) * 100
        
        # Update balance
        margin_return = (trade.quantity * trade.entry_price) / trade.leverage
        self.balance += margin_return + trade.pnl
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update max drawdown
        if self.balance > self.max_equity:
            self.max_equity = self.balance
        
        current_drawdown = (self.max_equity - self.balance) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Move to closed trades
        self.closed_trades.append(trade)
        
        logger.info(f"Closed {trade.signal_type} position for {trade.symbol} at {exit_price:.4f}, PnL: {trade.pnl:.2f}")
    
    def run_backtest(self, model, feature_engineer, symbols: List[str], 
                    timeframe: str = '5min') -> Dict:
        """Run complete backtest"""
        logger.info("Starting backtest")
        
        # Set date range
        if self.config.end_date is None:
            self.config.end_date = datetime.now()
        if self.config.start_date is None:
            self.config.start_date = self.config.end_date - timedelta(days=30)  # Last month
        
        # Initialize tracking
        current_time = self.config.start_date
        end_time = self.config.end_date
        
        # Load data for all symbols
        symbol_data = {}
        for symbol in symbols:
            df = self.load_historical_data(symbol, timeframe, self.config.start_date, self.config.end_date)
            if not df.empty:
                symbol_data[symbol] = df
        
        if not symbol_data:
            raise ValueError("No historical data found for the specified period")
        
        # Get all unique timestamps and sort them
        all_timestamps = set()
        for df in symbol_data.values():
            all_timestamps.update(df.index.tolist())
        
        timestamps = sorted(list(all_timestamps))
        
        logger.info(f"Backtesting from {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Total timestamps: {len(timestamps)}")
        
        # Process each timestamp
        for i, timestamp in enumerate(timestamps):
            # Record equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.balance,
                'open_positions': len(self.open_trades),
                'total_pnl': self.total_pnl
            })
            
            # Check exit conditions for open trades
            trades_to_remove = []
            for trade in self.open_trades:
                if trade.symbol in symbol_data:
                    df = symbol_data[trade.symbol]
                    if timestamp in df.index:
                        row = df.loc[timestamp]
                        if self.check_exit_conditions(trade, row['high'], row['low'], row['close'], timestamp):
                            trades_to_remove.append(trade)
            
            # Remove closed trades from open trades
            for trade in trades_to_remove:
                if trade in self.open_trades:
                    self.open_trades.remove(trade)
            
            # Look for new signals
            for symbol in symbols:
                if symbol not in symbol_data:
                    continue
                
                df = symbol_data[symbol]
                if timestamp not in df.index:
                    continue
                
                # Prepare sequence for prediction (last 60 periods)
                end_idx = df.index.get_loc(timestamp)
                if end_idx < model.sequence_length:
                    continue
                
                start_idx = end_idx - model.sequence_length + 1
                sequence_data = df.iloc[start_idx:end_idx + 1]
                
                # Process features (this should match training preprocessing)
                try:
                    # Create a simple dataframe slice for feature processing
                    features_df = feature_engineer.process_symbol_data_for_backtest(sequence_data, symbol)
                    if features_df.empty or len(features_df) < model.sequence_length:
                        continue
                    
                    # Check if feature_columns exist
                    if not hasattr(model, 'feature_columns') or not model.feature_columns:
                        logger.warning(f"Model feature_columns not found")
                        continue
                    
                    # Get feature sequence
                    available_features = [col for col in model.feature_columns if col in features_df.columns]
                    if len(available_features) < len(model.feature_columns) * 0.8:  # At least 80% of features
                        logger.warning(f"Not enough features available for {symbol}")
                        continue
                    
                    feature_sequence = features_df[available_features].values[-model.sequence_length:]
                    
                    # Pad with zeros if necessary
                    if len(available_features) < len(model.feature_columns):
                        padding = np.zeros((model.sequence_length, len(model.feature_columns) - len(available_features)))
                        feature_sequence = np.concatenate([feature_sequence, padding], axis=1)
                    
                    # Make prediction
                    prediction = model.predict_single(feature_sequence)
                    
                    # Try to open position
                    current_price = df.loc[timestamp, 'close']
                    opened = self.open_position(prediction, current_price, timestamp, symbol)
                    
                    if opened:
                        logger.info(f"Opened position for {symbol} with confidence {prediction['prediction_confidence']:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol} at {timestamp}: {e}")
                    continue
            
            # Progress logging
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(timestamps)} timestamps, Balance: {self.balance:.2f}")
        
        # Close any remaining open trades at market close
        for trade in self.open_trades[:]:
            if trade.symbol in symbol_data:
                df = symbol_data[trade.symbol]
                final_price = df.iloc[-1]['close']
                self.close_position(trade, final_price, timestamps[-1], 'MARKET_CLOSE')
        
        self.open_trades = []
        
        # Calculate final metrics
        results = self.calculate_performance_metrics()
        
        logger.info("Backtest completed")
        return results
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        win_rate = (self.winning_trades / self.total_trades) * 100
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate returns for each trade
        trade_returns = [trade.pnl_percentage for trade in self.closed_trades]
        
        # Profit factor
        gross_profit = sum([trade.pnl for trade in self.closed_trades if trade.pnl > 0])
        gross_loss = abs(sum([trade.pnl for trade in self.closed_trades if trade.pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        if len(trade_returns) > 1:
            returns_std = np.std(trade_returns)
            avg_return = np.mean(trade_returns)
            sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average trade metrics
        avg_win = np.mean([trade.pnl for trade in self.closed_trades if trade.pnl > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([trade.pnl for trade in self.closed_trades if trade.pnl < 0]) if self.losing_trades > 0 else 0
        
        # Time-based metrics
        if self.closed_trades and self.config.end_date and self.config.start_date:
            total_time = self.config.end_date - self.config.start_date
            if total_time.days > 0:
                monthly_return = (total_return / total_time.days) * 30.44  # Average days per month
            else:
                monthly_return = 0
        else:
            monthly_return = 0
        
        metrics = {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'max_drawdown': self.max_drawdown * 100,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_pnl': self.total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': self.total_pnl / self.total_trades if self.total_trades > 0 else 0,
            'best_trade': max([trade.pnl for trade in self.closed_trades]) if self.closed_trades else 0,
            'worst_trade': min([trade.pnl for trade in self.closed_trades]) if self.closed_trades else 0
        }
        
        return metrics
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.closed_trades:
            trade_data.append({
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'signal_type': trade.signal_type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_percentage': trade.pnl_percentage,
                'status': trade.status,
                'confidence': trade.confidence
            })
        
        return pd.DataFrame(trade_data)
    
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        df_equity = pd.DataFrame(self.equity_curve)
        axes[0, 0].plot(df_equity['timestamp'], df_equity['balance'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].grid(True)
        
        # Trade distribution
        if self.closed_trades:
            trade_pnl = [trade.pnl for trade in self.closed_trades]
            axes[0, 1].hist(trade_pnl, bins=20, alpha=0.7)
            axes[0, 1].set_title('Trade PnL Distribution')
            axes[0, 1].set_xlabel('PnL ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Monthly returns
        axes[1, 0].bar(['Wins', 'Losses'], [self.winning_trades, self.losing_trades])
        axes[1, 0].set_title('Win/Loss Count')
        axes[1, 0].set_ylabel('Number of Trades')
        
        # Drawdown
        axes[1, 1].plot(df_equity['timestamp'], 
                       [(self.initial_balance - balance) / self.initial_balance * 100 
                        for balance in df_equity['balance']])
        axes[1, 1].set_title('Drawdown (%)')
        axes[1, 1].set_ylabel('Drawdown %')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, file_path: str):
        """Export backtest results to JSON"""
        results = {
            'config': {
                'initial_balance': self.config.initial_balance,
                'leverage': self.config.leverage,
                'position_size_percentage': self.config.position_size_percentage,
                'commission_rate': self.config.commission_rate,
                'confidence_threshold': self.config.confidence_threshold,
                'start_date': self.config.start_date.isoformat() if self.config.start_date else None,
                'end_date': self.config.end_date.isoformat() if self.config.end_date else None
            },
            'performance_metrics': self.calculate_performance_metrics(),
            'equity_curve': self.equity_curve,
            'total_trades': len(self.closed_trades)
        }
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {file_path}")

# Example usage
if __name__ == "__main__":
    # Create backtest configuration
    config = BacktestConfig(
        initial_balance=10000.0,
        leverage=5.0,
        position_size_percentage=0.15,
        confidence_threshold=0.45
    )
    
    # Initialize backtest engine
    backtest = BacktestEngine(config)
    
    # Run backtest (requires trained model and feature engineer)
    # results = backtest.run_backtest(model, feature_engineer, symbols)
    
    print("Backtest engine ready")