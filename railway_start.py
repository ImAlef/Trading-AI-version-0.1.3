#!/usr/bin/env python3
"""
Railway Auto-Start Script
Automatically runs: collect data (6 months) -> train -> backtest (30 days) -> live monitor
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingBotManager

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def railway_auto_pipeline():
    """Complete automated pipeline for Railway deployment"""
    logger.info("🚀 Starting Railway Auto-Pipeline")
    logger.info("=" * 50)
    
    manager = TradingBotManager()
    
    try:
        # Step 1: Collect 6 months of data
        logger.info("📊 Step 1: Collecting 6 months of historical data...")
        success = manager.collect_data(days_back=180)  # 6 months
        if not success:
            logger.error("❌ Data collection failed!")
            return False
        
        # Step 2: Train model
        logger.info("🧠 Step 2: Training AI model...")
        success = manager.train_model(epochs=50, batch_size=16)  # More training
        if not success:
            logger.error("❌ Model training failed!")
            return False
        
        # Step 3: Backtest on last 30 days
        logger.info("📈 Step 3: Running backtest on last 30 days...")
        success = manager.run_backtest(days_back=30, initial_balance=20)
        if not success:
            logger.error("❌ Backtesting failed!")
            return False
        
        # Wait a bit before starting live trading
        logger.info("⏳ Waiting 30 seconds before starting live monitoring...")
        time.sleep(30)
        
        # Step 4: Start live monitoring
        logger.info("🔴 Step 4: Starting live monitoring...")
        logger.info("Bot is now LIVE and monitoring the market!")
        success = manager.start_live_monitoring()
        
        return success
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {e}")
        return False

def health_check_endpoint():
    """Simple health check for Railway"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import threading
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'AI Crypto Trading Bot',
                'stage': 'running'
            }
            
            self.wfile.write(json.dumps(status).encode())
        
        def log_message(self, format, *args):
            pass  # Suppress HTTP logs
    
    # Start health server in background
    port = int(os.getenv('PORT', '8080'))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    
    def run_server():
        logger.info(f"🌐 Health check server running on port {port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    return server

if __name__ == "__main__":
    logger.info("🤖 AI Crypto Trading Bot - Railway Deployment")
    logger.info(f"🕐 Start Time: {datetime.now()}")
    
    # Start health check server
    health_server = health_check_endpoint()
    
    # Check if we're in Railway environment
    if os.getenv('RAILWAY_ENVIRONMENT'):
        logger.info("🚂 Detected Railway environment")
        
        # Run the automated pipeline
        success = railway_auto_pipeline()
        
        if success:
            logger.info("✅ Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline failed!")
            sys.exit(1)
    else:
        logger.info("💻 Local environment detected")
        logger.info("🔧 Use: python main.py full-pipeline")
        
        # Just keep health server running for local testing
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")