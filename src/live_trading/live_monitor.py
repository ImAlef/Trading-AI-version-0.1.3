import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import logging
import json
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import schedule
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'HOLD'
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float
    prediction_confidence: float
    timeframe: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class MonitorConfig:
    """Live monitor configuration"""
    symbols: List[str]
    timeframes: List[str] = None
    check_interval_minutes: int = 5
    confidence_threshold: float = 0.55
    min_signals_per_hour: int = 1
    max_signals_per_hour: int = 10
    
    # Email settings
    sender_email: str = "arianakbari043@gmail.com"
    sender_password: str = "oaew ptyi krgu edly"
    recipient_email: str = "arianakbari043@gmail.com"
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1min', '5min', '15min']

class EmailNotifier:
    """Handle email notifications for trading signals"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.signal_count_today = 0
        self.last_signal_time = {}  # Track last signal time per symbol
        
    def create_signal_email(self, signals: List[TradingSignal]) -> str:
        """Create HTML email content for trading signals"""
        if not signals:
            return ""
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .signal {{ margin: 15px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #ecf0f1; }}
                .long {{ border-left-color: #27ae60; }}
                .short {{ border-left-color: #e74c3c; }}
                .details {{ margin: 10px 0; }}
                .price {{ font-weight: bold; color: #2c3e50; }}
                .confidence {{ font-size: 0.9em; color: #7f8c8d; }}
                .footer {{ margin-top: 20px; font-size: 0.8em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚀 Trading Signals Alert</h2>
                <p>AI has detected {len(signals)} new trading opportunity(s)</p>
            </div>
        """
        
        for signal in signals:
            signal_class = "long" if signal.signal_type == "LONG" else "short"
            emoji = "📈" if signal.signal_type == "LONG" else "📉"
            
            # Calculate potential profit percentage
            if signal.signal_type == "LONG":
                profit_pct = ((signal.take_profit - signal.entry_price) / signal.entry_price) * 100
                risk_pct = ((signal.entry_price - signal.stop_loss) / signal.entry_price) * 100
            else:
                profit_pct = ((signal.entry_price - signal.take_profit) / signal.entry_price) * 100
                risk_pct = ((signal.stop_loss - signal.entry_price) / signal.entry_price) * 100
            
            risk_reward = profit_pct / risk_pct if risk_pct > 0 else 0
            
            html_content += f"""
            <div class="signal {signal_class}">
                <h3>{emoji} {signal.symbol} - {signal.signal_type} Signal</h3>
                <div class="details">
                    <p><strong>Entry Price:</strong> <span class="price">${signal.entry_price:.6f}</span></p>
                    <p><strong>Take Profit:</strong> <span class="price">${signal.take_profit:.6f}</span> (+{profit_pct:.2f}%)</p>
                    <p><strong>Stop Loss:</strong> <span class="price">${signal.stop_loss:.6f}</span> (-{risk_pct:.2f}%)</p>
                    <p><strong>Risk/Reward Ratio:</strong> 1:{risk_reward:.2f}</p>
                    <p><strong>Timeframe:</strong> {signal.timeframe}</p>
                    <p class="confidence"><strong>AI Confidence:</strong> {signal.confidence:.1%} | <strong>Prediction Confidence:</strong> {signal.prediction_confidence:.1%}</p>
                    <p class="confidence"><strong>Signal Time:</strong> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
            </div>
            """
        
        html_content += f"""
            <div class="footer">
                <p>⚠️ <strong>Risk Warning:</strong> This is an AI-generated signal. Always do your own research and risk management.</p>
                <p>📊 Leverage: 5x | Position Size: 10-20% of portfolio</p>
                <p>🤖 Generated by AI Trading Bot | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def send_signal_email(self, signals: List[TradingSignal]) -> bool:
        """Send trading signal email"""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🚨 {len(signals)} New Trading Signal(s) Detected!"
            message["From"] = self.config.sender_email
            message["To"] = self.config.recipient_email
            
            # Create HTML content
            html_content = self.create_signal_email(signals)
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Create text content (fallback)
            text_content = f"Trading Signals Alert!\n\n"
            for signal in signals:
                text_content += f"""
{signal.symbol} - {signal.signal_type}
Entry: ${signal.entry_price:.6f}
Take Profit: ${signal.take_profit:.6f}
Stop Loss: ${signal.stop_loss:.6f}
Confidence: {signal.confidence:.1%}
Time: {signal.timestamp}
---
"""
            text_part = MIMEText(text_content, "plain")
            message.attach(text_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    self.config.recipient_email,
                    message.as_string()
                )
            
            logger.info(f"Email sent successfully with {len(signals)} signals")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def should_send_signal(self, symbol: str, signal_type: str) -> bool:
        """Check if signal should be sent based on rate limiting"""
        now = datetime.now()
        
        # Check daily limit
        if self.signal_count_today >= self.config.max_signals_per_hour * 24:
            return False
        
        # Check symbol-specific rate limiting (avoid spam)
        symbol_key = f"{symbol}_{signal_type}"
        if symbol_key in self.last_signal_time:
            time_diff = now - self.last_signal_time[symbol_key]
            if time_diff < timedelta(minutes=30):  # Minimum 30 minutes between same signals
                return False
        
        self.last_signal_time[symbol_key] = now
        return True

class LiveTradingMonitor:
    """Main live trading monitoring system"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.data_collector = None
        self.feature_engineer = None
        self.model = None
        self.email_notifier = EmailNotifier(config)
        self.is_running = False
        self.last_check_time = {}
        self.signal_history = []
        self.performance_stats = {
            'signals_sent': 0,
            'last_signal_time': None,
            'uptime_start': datetime.now(),
            'errors': 0
        }
        
    def initialize_components(self):
        """Initialize all required components"""
        try:
            # Import components
            from src.data_collection.kucoin_collector import KuCoinDataCollector
            from src.data_processing.simple_feature_engineering import SimpleFeatureEngineer
            from src.models.deep_learning_model import TradingSignalModel
            
            # Initialize components
            self.data_collector = KuCoinDataCollector()
            self.feature_engineer = SimpleFeatureEngineer()
            self.model = TradingSignalModel(confidence_threshold=self.config.confidence_threshold)
            
            # Load trained model
            self.model.load_model()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def collect_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Collect latest data for analysis"""
        try:
            # Collect real-time data
            self.data_collector.collect_realtime_data()
            
            # Get latest data for the symbol
            df = self.data_collector.get_latest_data(symbol, timeframe, limit=200)
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Analyze a single symbol for trading signals"""
        try:
            # Get latest data
            df = self.collect_latest_data(symbol, timeframe)
            if df.empty or len(df) < self.model.sequence_length:
                return None
            
            # Process features
            features_df = self.feature_engineer.process_symbol_data_for_live(df, symbol)
            if features_df.empty or len(features_df) < self.model.sequence_length:
                return None
            
            # Prepare sequence for prediction
            feature_sequence = features_df[self.model.feature_columns].values[-self.model.sequence_length:]
            
            # Make prediction
            prediction = self.model.predict_single(feature_sequence)
            
            # Check if signal meets criteria
            if (prediction['prediction_confidence'] >= self.config.confidence_threshold and 
                prediction['signal'] != 0):
                
                signal_type = 'LONG' if prediction['signal'] == 1 else 'SHORT'
                current_price = df.iloc[-1]['close']
                
                # Create trading signal
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_price=current_price,
                    take_profit=prediction['take_profit'],
                    stop_loss=prediction['stop_loss'],
                    confidence=prediction['confidence'],
                    prediction_confidence=prediction['prediction_confidence'],
                    timeframe=timeframe
                )
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
            self.performance_stats['errors'] += 1
            return None
    
    def scan_all_symbols(self) -> List[TradingSignal]:
        """Scan all symbols for trading signals"""
        signals = []
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                signal = self.analyze_symbol(symbol, timeframe)
                if signal and self.email_notifier.should_send_signal(symbol, signal.signal_type):
                    signals.append(signal)
                    logger.info(f"New signal: {signal.symbol} {signal.signal_type} at {signal.entry_price:.6f}")
        
        return signals
    
    def process_signals(self, signals: List[TradingSignal]):
        """Process and send trading signals"""
        if not signals:
            return
        
        # Save signals to history
        self.signal_history.extend(signals)
        
        # Send email notification
        if self.email_notifier.send_signal_email(signals):
            self.performance_stats['signals_sent'] += len(signals)
            self.performance_stats['last_signal_time'] = datetime.now()
            
            # Log signals to file
            self.log_signals_to_file(signals)
    
    def log_signals_to_file(self, signals: List[TradingSignal]):
        """Log signals to file for record keeping"""
        try:
            filename = f"trading_signals_{datetime.now().strftime('%Y%m%d')}.json"
            
            signal_data = []
            for signal in signals:
                signal_data.append(signal.to_dict())
            
            # Append to daily file
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []
            
            existing_data.extend(signal_data)
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
            
            logger.info(f"Signals logged to {filename}")
            
        except Exception as e:
            logger.error(f"Error logging signals: {e}")
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                logger.info("Scanning for trading signals...")
                
                # Scan for signals
                signals = self.scan_all_symbols()
                
                # Process any found signals
                if signals:
                    logger.info(f"Found {len(signals)} new signals")
                    self.process_signals(signals)
                else:
                    logger.info("No new signals found")
                
                # Wait for next check
                time.sleep(self.config.check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.performance_stats['errors'] += 1
                time.sleep(60)  # Wait 1 minute before retrying
    
    def start_monitoring(self):
        """Start the live monitoring system"""
        logger.info("Starting live trading monitor")
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        # Send startup notification
        startup_message = f"""
        🤖 AI Trading Bot Started!
        
        Configuration:
        - Symbols: {', '.join(self.config.symbols)}
        - Timeframes: {', '.join(self.config.timeframes)}
        - Check Interval: {self.config.check_interval_minutes} minutes
        - Confidence Threshold: {self.config.confidence_threshold:.0%}
        
        The bot is now monitoring the market for trading opportunities.
        """
        
        try:
            message = MIMEText(startup_message)
            message["Subject"] = "🚀 AI Trading Bot Started"
            message["From"] = self.config.sender_email
            message["To"] = self.config.recipient_email
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(self.config.sender_email, self.config.recipient_email, message.as_string())
            
            logger.info("Startup notification sent")
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")
        
        # Start monitoring
        self.is_running = True
        self.monitoring_loop()
        
        return True
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        logger.info("Stopping live trading monitor")
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        uptime = datetime.now() - self.performance_stats['uptime_start']
        
        status = {
            'is_running': self.is_running,
            'uptime_hours': uptime.total_seconds() / 3600,
            'signals_sent_today': self.performance_stats['signals_sent'],
            'last_signal_time': self.performance_stats['last_signal_time'],
            'error_count': self.performance_stats['errors'],
            'monitored_symbols': len(self.config.symbols),
            'monitored_timeframes': len(self.config.timeframes)
        }
        
        return status

# Main execution
def main():
    """Main function to start the live monitor"""
    # Configuration
    symbols = [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
        'SOL-USDT', 'MATIC-USDT', 'AVAX-USDT', 'DOT-USDT', 
        'LINK-USDT', 'UNI-USDT'
    ]
    
    config = MonitorConfig(
        symbols=symbols,
        timeframes=['5min', '15min'],
        check_interval_minutes=5,
        confidence_threshold=0.55
    )
    
    # Start monitor
    monitor = LiveTradingMonitor(config)
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()