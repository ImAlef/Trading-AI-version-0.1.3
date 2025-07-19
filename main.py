#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Main Application
Entry point for training, backtesting, and live monitoring
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.data_collection.kucoin_collector import KuCoinDataCollector
from src.data_processing.simple_feature_engineering import SimpleFeatureEngineer
from src.models.deep_learning_model import TradingSignalModel
from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.live_trading.live_monitor import LiveTradingMonitor, MonitorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBotManager:
    """Main trading bot manager"""
    
    def __init__(self):
        self.symbols = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
            'SOL-USDT', 'AVAX-USDT', 'DOT-USDT', 
            'LINK-USDT', 'UNI-USDT'
        ]
        self.timeframes = ['1min', '5min', '15min', '1hour', '4hour', '1day']
        
    def collect_data(self, days_back: int = 365):
        """Collect historical data"""
        logger.info("=== DATA COLLECTION PHASE ===")
        
        collector = KuCoinDataCollector()
        
        # Collect historical data
        logger.info(f"Collecting {days_back} days of historical data...")
        collector.collect_historical_data(days_back=days_back)
        
        # Show data summary
        summary = collector.get_data_summary()
        logger.info("Data collection completed!")
        
        # Print summary
        for symbol in self.symbols:
            if symbol in summary:
                logger.info(f"\n{symbol}:")
                for tf, info in summary[symbol].items():
                    logger.info(f"  {tf}: {info['count']} records from {info['earliest']} to {info['latest']}")
        
        return True
    
    def train_model(self, epochs: int = 100, batch_size: int = 32):
        """Train the deep learning model"""
        logger.info("=== MODEL TRAINING PHASE ===")
        
        # Initialize components
        feature_engineer = SimpleFeatureEngineer()
        model = TradingSignalModel(sequence_length=60, confidence_threshold=0.55)
        
        # Prepare training data
        logger.info("Preparing training data...")
        training_data = feature_engineer.prepare_training_data()
        
        if not training_data:
            logger.error("No training data available!")
            return False
        
        logger.info(f"Training data prepared for {len(training_data)} symbols")
        
        # Train model
        logger.info("Starting model training...")
        history = model.train(
            training_data=training_data,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size
        )
        
        logger.info("Model training completed!")
        
        # Save training history
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, values in history.items():
            if isinstance(values, list):
                history_serializable[key] = [float(v) for v in values]
            else:
                history_serializable[key] = float(values)
        
        with open('training_history.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        return True
    
    def run_backtest(self, days_back: int = 30, initial_balance: float = 10000):
        """Run backtest on trained model"""
        logger.info("=== BACKTESTING PHASE ===")
        
        # Initialize components
        feature_engineer = SimpleFeatureEngineer()
        model = TradingSignalModel(confidence_threshold=0.55)
        
        try:
            # Load trained model
            model.load_model()
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error("Trained model not found! Please run training first.")
            return False
        
        # Configure backtest
        config = BacktestConfig(
            initial_balance=initial_balance,
            leverage=5.0,
            position_size_percentage=0.15,
            commission_rate=0.001,
            confidence_threshold=0.55,
            start_date=datetime.now() - timedelta(days=days_back),
            end_date=datetime.now()
        )
        
        # Run backtest
        backtest_engine = BacktestEngine(config)
        results = backtest_engine.run_backtest(
            model=model,
            feature_engineer=feature_engineer,
            symbols=self.symbols[:5],  # Test with first 5 symbols
            timeframe='5min'
        )
        
        # Display results
        logger.info("=== BACKTEST RESULTS ===")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Monthly Return: {results['monthly_return']:.2f}%")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        
        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'backtest_results_{timestamp}.json'
        backtest_engine.export_results(results_file)
        
        # Plot results
        plot_file = f'backtest_plot_{timestamp}.png'
        backtest_engine.plot_results(save_path=plot_file)
        
        # Get trade summary
        trade_summary = backtest_engine.get_trade_summary()
        if not trade_summary.empty:
            trade_summary.to_csv(f'trade_summary_{timestamp}.csv', index=False)
            logger.info(f"Trade summary saved to trade_summary_{timestamp}.csv")
        
        return True
    
    def start_live_monitoring(self):
        """Start live trading monitor"""
        logger.info("=== LIVE MONITORING PHASE ===")
        
        # Check if model exists
        try:
            test_model = TradingSignalModel()
            test_model.load_model()
            logger.info("Model found and ready for live trading")
        except FileNotFoundError:
            logger.error("Trained model not found! Please run training first.")
            return False
        
        # Configure live monitor
        config = MonitorConfig(
            symbols=self.symbols,
            timeframes=['5min', '15min'],
            check_interval_minutes=5,
            confidence_threshold=0.55,
            min_signals_per_hour=1,
            max_signals_per_hour=10
        )
        
        # Start monitoring
        monitor = LiveTradingMonitor(config)
        
        try:
            logger.info("Starting live trading monitor...")
            monitor.start_monitoring()
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Error in live monitoring: {e}")
            monitor.stop_monitoring()
            return False
        
        return True
    
    def show_status(self):
        """Show system status"""
        logger.info("=== SYSTEM STATUS ===")
        
        # Check data availability
        collector = KuCoinDataCollector()
        summary = collector.get_data_summary()
        
        if summary:
            logger.info("✅ Historical data available")
            logger.info(f"   Symbols: {len(summary)} available")
        else:
            logger.info("❌ No historical data found")
        
        # Check model availability
        try:
            model = TradingSignalModel()
            model.load_model()
            logger.info("✅ Trained model available")
        except FileNotFoundError:
            logger.info("❌ No trained model found")
        
        # Check recent signals
        try:
            with open(f"trading_signals_{datetime.now().strftime('%Y%m%d')}.json", 'r') as f:
                signals = json.load(f)
                logger.info(f"📊 Today's signals: {len(signals)}")
        except FileNotFoundError:
            logger.info("📊 No signals found for today")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Crypto Trading Bot')
    parser.add_argument('command', choices=[
        'collect-data', 'train', 'backtest', 'live', 'status', 'full-pipeline'
    ], help='Command to execute')
    
    # Optional arguments
    parser.add_argument('--days', type=int, default=365, help='Days of historical data to collect')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance for backtesting')
    parser.add_argument('--backtest-days', type=int, default=30, help='Days to backtest')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = TradingBotManager()
    
    logger.info("🤖 AI Crypto Trading Bot Started")
    logger.info(f"Command: {args.command}")
    
    try:
        if args.command == 'collect-data':
            success = manager.collect_data(days_back=args.days)
            
        elif args.command == 'train':
            success = manager.train_model(epochs=args.epochs, batch_size=args.batch_size)
            
        elif args.command == 'backtest':
            success = manager.run_backtest(days_back=args.backtest_days, initial_balance=args.balance)
            
        elif args.command == 'live':
            success = manager.start_live_monitoring()
            
        elif args.command == 'status':
            manager.show_status()
            success = True
            
        elif args.command == 'full-pipeline':
            logger.info("Running full pipeline: collect -> train -> backtest -> live")
            
            # Step 1: Collect data
            success = manager.collect_data(days_back=args.days)
            if not success:
                logger.error("Data collection failed!")
                return 1
            
            # Step 2: Train model
            success = manager.train_model(epochs=args.epochs, batch_size=args.batch_size)
            if not success:
                logger.error("Model training failed!")
                return 1
            
            # Step 3: Run backtest
            success = manager.run_backtest(days_back=args.backtest_days, initial_balance=args.balance)
            if not success:
                logger.error("Backtesting failed!")
                return 1
            
            # Step 4: Start live monitoring (if backtest results are good)
            logger.info("Pipeline completed successfully! Starting live monitoring...")
            success = manager.start_live_monitoring()
        
        if success:
            logger.info("✅ Command completed successfully!")
            return 0
        else:
            logger.error("❌ Command failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)