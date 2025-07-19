#!/usr/bin/env python3
"""
Railway Deployment Configuration
Optimized for cloud deployment with resource management
"""

import os
import logging
import sys
from datetime import datetime
import psutil
import gc
import threading
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live_trading.live_monitor import LiveTradingMonitor, MonitorConfig

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Railway captures stdout
    ]
)
logger = logging.getLogger(__name__)

class ProductionTradingBot:
    """Production-ready trading bot for Railway deployment"""
    
    def __init__(self):
        self.monitor = None
        self.is_running = False
        self.setup_signal_handlers()
        
        # Environment variables for Railway
        self.symbols = [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 
            'SOL-USDT', 'MATIC-USDT', 'AVAX-USDT', 'DOT-USDT', 
            'LINK-USDT', 'UNI-USDT'
        ]
        
        # Get configuration from environment variables
        self.config = self.load_config_from_env()
        
    def load_config_from_env(self) -> MonitorConfig:
        """Load configuration from environment variables"""
        
        # Email configuration from environment
        sender_email = os.getenv('SENDER_EMAIL', 'arianakbari043@gmail.com')
        sender_password = os.getenv('SENDER_PASSWORD', 'oaew ptyi krgu edly')
        recipient_email = os.getenv('RECIPIENT_EMAIL', 'arianakbari043@gmail.com')
        
        # Bot configuration
        check_interval = int(os.getenv('CHECK_INTERVAL_MINUTES', '5'))
        confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.45'))
        max_signals_per_hour = int(os.getenv('MAX_SIGNALS_PER_HOUR', '10'))
        
        # Timeframes - optimized for cloud resources
        timeframes = os.getenv('TIMEFRAMES', '5min,15min').split(',')
        
        config = MonitorConfig(
            symbols=self.symbols,
            timeframes=timeframes,
            check_interval_minutes=check_interval,
            confidence_threshold=confidence_threshold,
            max_signals_per_hour=max_signals_per_hour,
            sender_email=sender_email,
            sender_password=sender_password,
            recipient_email=recipient_email
        )
        
        logger.info(f"Configuration loaded: {len(config.symbols)} symbols, {len(config.timeframes)} timeframes")
        return config
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def monitor_resources(self):
        """Monitor system resources and optimize usage"""
        def resource_monitor():
            while self.is_running:
                try:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    logger.info(f"Resources - Memory: {memory.percent:.1f}%, CPU: {cpu_percent:.1f}%")
                    
                    # Force garbage collection if memory usage is high
                    if memory.percent > 80:
                        logger.warning("High memory usage detected, running garbage collection")
                        gc.collect()
                    
                    # Sleep for 5 minutes before next check
                    import time
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error monitoring resources: {e}")
                    import time
                    time.sleep(60)
        
        # Start resource monitoring in background thread
        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitor_thread.start()
    
    def health_check_server(self):
        """Simple health check server for Railway"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import threading
        
        class HealthCheckHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    # Health check endpoint
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    status = {
                        'status': 'healthy' if self.server.bot.is_running else 'stopped',
                        'timestamp': datetime.now().isoformat(),
                        'uptime': str(datetime.now() - self.server.start_time),
                        'memory_percent': psutil.virtual_memory().percent,
                        'cpu_percent': psutil.cpu_percent()
                    }
                    
                    self.wfile.write(json.dumps(status).encode())
                    
                elif self.path == '/status':
                    # Detailed status endpoint
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    if self.server.bot.monitor:
                        status = self.server.bot.monitor.get_status()
                    else:
                        status = {'error': 'Monitor not initialized'}
                    
                    self.wfile.write(json.dumps(status, default=str).encode())
                    
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress HTTP server logs
                pass
        
        # Start health check server
        port = int(os.getenv('PORT', '8080'))  # Railway provides PORT env variable
        server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
        server.bot = self
        server.start_time = datetime.now()
        
        def run_server():
            logger.info(f"Health check server started on port {port}")
            server.serve_forever()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return server
    
    def send_startup_notification(self):
        """Send startup notification"""
        try:
            import smtplib
            import ssl
            from email.mime.text import MIMEText
            
            startup_message = f"""
🚀 AI Trading Bot Deployed Successfully!

Deployment Details:
- Platform: Railway
- Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- Symbols: {len(self.symbols)} cryptocurrencies
- Timeframes: {', '.join(self.config.timeframes)}
- Check Interval: {self.config.check_interval_minutes} minutes
- Confidence Threshold: {self.config.confidence_threshold:.0%}

System Resources:
- Memory: {psutil.virtual_memory().percent:.1f}%
- CPU: {psutil.cpu_percent():.1f}%

The bot is now live and monitoring the market 24/7!
            """
            
            message = MIMEText(startup_message)
            message["Subject"] = "🚀 AI Trading Bot Deployed on Railway"
            message["From"] = self.config.sender_email
            message["To"] = self.config.recipient_email
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    self.config.recipient_email,
                    message.as_string()
                )
            
            logger.info("Startup notification sent successfully")
            
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")
    
    def start(self):
        """Start the production trading bot"""
        logger.info("=== STARTING PRODUCTION TRADING BOT ===")
        logger.info(f"Environment: Railway Cloud")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Start Time: {datetime.now()}")
        
        try:
            # Start health check server
            health_server = self.health_check_server()
            
            # Start resource monitoring
            self.monitor_resources()
            
            # Send startup notification
            self.send_startup_notification()
            
            # Initialize and start live monitor
            self.monitor = LiveTradingMonitor(self.config)
            self.is_running = True
            
            logger.info("Starting live trading monitor...")
            self.monitor.start_monitoring()
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down trading bot...")
        
        self.is_running = False
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Send shutdown notification
        try:
            import smtplib
            import ssl
            from email.mime.text import MIMEText
            
            shutdown_message = f"""
🛑 AI Trading Bot Shutdown

Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Reason: Graceful shutdown requested

The bot has been stopped and is no longer monitoring the market.
            """
            
            message = MIMEText(shutdown_message)
            message["Subject"] = "🛑 AI Trading Bot Shutdown"
            message["From"] = self.config.sender_email
            message["To"] = self.config.recipient_email
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    self.config.recipient_email,
                    message.as_string()
                )
            
            logger.info("Shutdown notification sent")
            
        except Exception as e:
            logger.warning(f"Failed to send shutdown notification: {e}")
        
        logger.info("Trading bot shutdown complete")

def main():
    """Main entry point for Railway deployment"""
    logger.info("🤖 AI Trading Bot - Railway Deployment")
    
    # Check if we're in Railway environment
    if os.getenv('RAILWAY_ENVIRONMENT'):
        logger.info(f"Running on Railway - Environment: {os.getenv('RAILWAY_ENVIRONMENT')}")
    
    try:
        # Create and start production bot
        bot = ProductionTradingBot()
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()