#!/usr/bin/env python3
"""
Railway Pre-trained Deployment Script
Uses locally trained model and cached data
"""

import os
import sys
import logging
import time
from datetime import datetime
import requests

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

def download_pretrained_assets():
    """Download pre-trained model and data from GitHub releases"""
    logger.info("📥 Downloading pre-trained assets...")
    
    # URLs to your GitHub release assets (شما باید اینو set کنید)
    model_urls = {
        "model_weights.h5": "https://github.com/your-username/ai-trading-bot/releases/download/v1.0/model_weights.h5",
        "model_architecture.json": "https://github.com/your-username/ai-trading-bot/releases/download/v1.0/model_architecture.json", 
        "model_metadata.pkl": "https://github.com/your-username/ai-trading-bot/releases/download/v1.0/model_metadata.pkl",
        "crypto_data.db": "https://github.com/your-username/ai-trading-bot/releases/download/v1.0/crypto_data.db"
    }
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    for filename, url in model_urls.items():
        try:
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = f"models/{filename}" if filename.endswith(('.h5', '.json', '.pkl')) else filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False
    
    return True

def check_pretrained_assets():
    """Check if pre-trained assets exist locally"""
    required_files = [
        "models/model_weights.h5",
        "models/model_architecture.json", 
        "models/model_metadata.pkl",
        "crypto_data.db"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        return False
    
    logger.info("✅ All pre-trained assets found!")
    return True

def railway_pretrained_pipeline():
    """Railway pipeline using pre-trained model"""
    logger.info("🚀 Starting Railway PRE-TRAINED Pipeline")
    logger.info("=" * 50)
    
    manager = TradingBotManager()
    
    try:
        # Check if pre-trained assets exist
        if not check_pretrained_assets():
            logger.info("📥 Pre-trained assets not found, downloading...")
            if not download_pretrained_assets():
                logger.error("❌ Failed to download pre-trained assets!")
                return False
        
        # Step 1: Skip data collection (use cached data)
        logger.info("📊 Step 1: Using cached historical data ✅")
        
        # Step 2: Skip training (use pre-trained model)
        logger.info("🧠 Step 2: Using pre-trained model ✅")
        
        # Step 3: Quick validation backtest
        logger.info("📈 Step 3: Running validation backtest...")
        success = manager.run_backtest(days_back=7, initial_balance=20)
        if not success:
            logger.warning("⚠️ Backtest failed, but continuing with live trading...")
        
        # Step 4: Start live monitoring immediately
        logger.info("🔴 Step 4: Starting live monitoring...")
        logger.info("🎉 Pre-trained bot is now LIVE!")
        success = manager.start_live_monitoring()
        
        return success
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        return False

def health_check_endpoint():
    """Health check server"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import threading
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check if model exists
            model_status = "loaded" if check_pretrained_assets() else "missing"
            
            status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'AI Crypto Trading Bot',
                'mode': 'pre-trained',
                'model_status': model_status
            }
            
            self.wfile.write(json.dumps(status).encode())
        
        def log_message(self, format, *args):
            pass
    
    port = int(os.getenv('PORT', '8080'))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    
    def run_server():
        logger.info(f"🌐 Health server running on port {port}")
        server.serve_forever()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    return server

if __name__ == "__main__":
    logger.info("🤖 AI Crypto Trading Bot - PRE-TRAINED MODE")
    logger.info(f"🕐 Start Time: {datetime.now()}")
    
    # Start health check server
    health_server = health_check_endpoint()
    
    # Check if we're in cloud environment
    if os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('PORT'):
        logger.info("☁️ Cloud environment detected")
        
        # Run pre-trained pipeline
        success = railway_pretrained_pipeline()
        
        if success:
            logger.info("✅ Pre-trained pipeline completed!")
        else:
            logger.error("❌ Pipeline failed!")
            logger.info("🔄 Keeping health server alive...")
            try:
                while True:
                    time.sleep(300)
            except KeyboardInterrupt:
                logger.info("👋 Shutting down...")
    else:
        logger.info("💻 Local environment - use: python main.py")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")