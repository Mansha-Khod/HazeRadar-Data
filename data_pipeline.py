import os
import requests
import pandas as pd
import schedule
import time
import logging
from datetime import datetime
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables (set in Railway)
NASA_API_KEY = os.getenv('NASA_API_KEY', '4fd11fa7e1e313db40a1d963e9c4c7c0')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '83d1e9d7156a44ae4aaa3355855a1cea')
WAQI_API_TOKEN = os.getenv('WAQI_API_TOKEN', 'd5922d5ac41e8a89495f776fdc48a469213bc5c1')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://daxrnmvkpikjvvzgrhko.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRheHJubXZrcGlranZ2emdyaGtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2OTkyNjEsImV4cCI6MjA3NjI3NTI2MX0.XWJ_aWUh5Eci5tQSRAATqDXmQ5nh2eHQGzYu6qMcsvQ')

# Rest of your data collection classes (NASAFIRMSCollector, WeatherCollector, etc.)
# [Include all the class definitions from the previous code here]

def run_pipeline():
    """Run one complete data collection cycle"""
    logger.info("🚀 Starting cloud data collection...")
    
    # Initialize collectors
    nasa_collector = NASAFIRMSCollector()
    weather_collector = WeatherCollector()
    aqi_collector = AQICollector()
    gnn_generator = GNNDataGenerator()
    db_manager = SupabaseManager()
    
    try:
        # Collect and store all data
        fire_data = nasa_collector.collect_fire_data()
        if fire_data:
            db_manager.insert_data('fire_hotspots', fire_data)
        
        weather_data = weather_collector.collect_weather_data()
        if weather_data:
            db_manager.insert_data('weather_data', weather_data)
        
        aqi_data = aqi_collector.collect_aqi_data()
        if aqi_data:
            db_manager.insert_data('air_quality', aqi_data)
        
        gnn_data = gnn_generator.generate_training_data(fire_data, weather_data, aqi_data)
        if gnn_data:
            db_manager.insert_data('gnn_training_data', gnn_data)
        
        # Generate graph structure once per day
        if datetime.now().hour == 0:
            graph_generator = GraphStructureGenerator()
            graph_data = graph_generator.generate_graph_data()
            if graph_data:
                db_manager.insert_data('city_graph_structure', graph_data)
        
        logger.info("✅ Cloud collection completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cloud collection failed: {e}")
        return False

def main():
    """Main cloud scheduler"""
    logger.info("🌐 HazeRadar Cloud Service Started - 24/7 Operation")
    
    # Schedule runs every 3 hours
    schedule.every(3).hours.do(run_pipeline)
    
    # Run immediately on startup
    run_pipeline()
    
    # Keep running forever
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
