"""
HazeRadar Automated Data Collection Pipeline
Collects data from NASA FIRMS, Weather APIs, SIPONGI, and WAQI
Cleans and stores in PostgreSQL database for team access
"""

import os
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hazeradar_pipeline.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'hazeradar_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# API Keys
NASA_FIRMS_KEY = os.getenv('NASA_FIRMS_MAP_KEY')
OPENWEATHER_KEY = os.getenv('OPENWEATHER_API_KEY')
WAQI_TOKEN = os.getenv('WAQI_API_TOKEN')

# Indonesia bounding box
INDONESIA_BBOX = {
    'min_lat': -11.0,
    'max_lat': 6.0,
    'min_lon': 95.0,
    'max_lon': 141.0
}


class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self):
        self.conn = None
        self.connect()
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            logging.info("✓ Database connection established")
        except Exception as e:
            logging.error(f"✗ Database connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        commands = [
            """
            CREATE EXTENSION IF NOT EXISTS postgis;
            """,
            """
            CREATE TABLE IF NOT EXISTS fire_hotspots (
                id SERIAL PRIMARY KEY,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                brightness FLOAT,
                confidence VARCHAR(20),
                frp FLOAT,
                acquisition_date TIMESTAMP NOT NULL,
                satellite VARCHAR(20),
                geom GEOMETRY(Point, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS weather_data (
                id SERIAL PRIMARY KEY,
                city VARCHAR(100),
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                wind_speed FLOAT,
                wind_direction FLOAT,
                temperature FLOAT,
                humidity FLOAT,
                pressure FLOAT,
                timestamp TIMESTAMP NOT NULL,
                geom GEOMETRY(Point, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS air_quality (
                id SERIAL PRIMARY KEY,
                city VARCHAR(100),
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                aqi INTEGER,
                pm25 FLOAT,
                pm10 FLOAT,
                timestamp TIMESTAMP NOT NULL,
                geom GEOMETRY(Point, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_fire_date 
            ON fire_hotspots(acquisition_date);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_fire_geom 
            ON fire_hotspots USING GIST(geom);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_weather_geom 
            ON weather_data USING GIST(geom);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_weather_time 
            ON weather_data(timestamp);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_aqi_time 
            ON air_quality(timestamp);
            """
        ]
        
        cur = self.conn.cursor()
        try:
            for cmd in commands:
                cur.execute(cmd)
            self.conn.commit()
            logging.info("✓ Database tables created/verified")
        except Exception as e:
            logging.error(f"✗ Error creating tables: {e}")
            self.conn.rollback()
            raise
        finally:
            cur.close()
    
    def insert_fire_data(self, df):
        """Insert fire hotspot data"""
        if df.empty:
            logging.info("No fire data to insert")
            return 0
        
        cur = self.conn.cursor()
        try:
            values = [
                (row['latitude'], row['longitude'], row['brightness'], 
                 row['confidence'], row['frp'], row['acq_date'], 
                 row['satellite'], f"POINT({row['longitude']} {row['latitude']})")
                for _, row in df.iterrows()
            ]
            
            query = """
            INSERT INTO fire_hotspots 
            (latitude, longitude, brightness, confidence, frp, 
             acquisition_date, satellite, geom)
            VALUES %s
            """
            
            execute_values(cur, query, values)
            self.conn.commit()
            logging.info(f"✓ Inserted {len(df)} fire hotspot records")
            return len(df)
        except Exception as e:
            logging.error(f"✗ Error inserting fire data: {e}")
            self.conn.rollback()
            return 0
        finally:
            cur.close()
    
    def insert_weather_data(self, df):
        """Insert weather data"""
        if df.empty:
            logging.info("No weather data to insert")
            return 0
        
        cur = self.conn.cursor()
        try:
            values = [
                (row['city'], row['lat'], row['lon'], row['wind_speed'], 
                 row['wind_deg'], row['temp'], row['humidity'], 
                 row['pressure'], row['timestamp'],
                 f"POINT({row['lon']} {row['lat']})")
                for _, row in df.iterrows()
            ]
            
            query = """
            INSERT INTO weather_data 
            (city, latitude, longitude, wind_speed, wind_direction, 
             temperature, humidity, pressure, timestamp, geom)
            VALUES %s
            """
            
            execute_values(cur, query, values)
            self.conn.commit()
            logging.info(f"✓ Inserted {len(df)} weather records")
            return len(df)
        except Exception as e:
            logging.error(f"✗ Error inserting weather data: {e}")
            self.conn.rollback()
            return 0
        finally:
            cur.close()
    
    def insert_air_quality(self, df):
        """Insert air quality data"""
        if df.empty:
            logging.info("No air quality data to insert")
            return 0
        
        cur = self.conn.cursor()
        try:
            values = [
                (row['city'], row['lat'], row['lon'], row['aqi'], 
                 row['pm25'], row['pm10'], row['timestamp'],
                 f"POINT({row['lon']} {row['lat']})")
                for _, row in df.iterrows()
            ]
            
            query = """
            INSERT INTO air_quality 
            (city, latitude, longitude, aqi, pm25, pm10, timestamp, geom)
            VALUES %s
            """
            
            execute_values(cur, query, values)
            self.conn.commit()
            logging.info(f"✓ Inserted {len(df)} air quality records")
            return len(df)
        except Exception as e:
            logging.error(f"✗ Error inserting air quality data: {e}")
            self.conn.rollback()
            return 0
        finally:
            cur.close()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")


class NASAFIRMSCollector:
    """Collects fire hotspot data from NASA FIRMS"""
    
    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def collect_data(self, days=1):
        """Fetch fire hotspots for the last N days"""
        if not self.api_key:
            logging.warning("⚠ NASA FIRMS API key not configured - skipping")
            return pd.DataFrame()
        
        try:
            # VIIRS S-NPP satellite data
            url = f"{self.BASE_URL}/{self.api_key}/VIIRS_SNPP_NRT/{INDONESIA_BBOX['min_lat']},{INDONESIA_BBOX['min_lon']},{INDONESIA_BBOX['max_lat']},{INDONESIA_BBOX['max_lon']}/{days}"
            
            logging.info(f"Fetching NASA FIRMS data for last {days} days...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            if df.empty:
                logging.info("No fire hotspots detected")
                return pd.DataFrame()
            
            # Clean and format data
            df['acq_date'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H%M')
            df = df[['latitude', 'longitude', 'brightness', 'confidence', 'frp', 'acq_date', 'satellite']]
            
            # Data cleaning
            df = df.dropna(subset=['latitude', 'longitude', 'acq_date'])
            df['brightness'] = pd.to_numeric(df['brightness'], errors='coerce')
            df['frp'] = pd.to_numeric(df['frp'], errors='coerce')
            
            logging.info(f"✓ Collected {len(df)} fire hotspots")
            return df
            
        except Exception as e:
            logging.error(f"✗ Error fetching NASA FIRMS data: {e}")
            return pd.DataFrame()


class WeatherCollector:
    """Collects weather data from OpenWeatherMap"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    # Major Indonesian cities for weather monitoring
    CITIES = [
        {'name': 'Jakarta', 'lat': -6.2088, 'lon': 106.8456},
        {'name': 'Bandung', 'lat': -6.9175, 'lon': 107.6191},
        {'name': 'Surabaya', 'lat': -7.2575, 'lon': 112.7521},
        {'name': 'Medan', 'lat': 3.5952, 'lon': 98.6722},
        {'name': 'Palembang', 'lat': -2.9761, 'lon': 104.7754},
        {'name': 'Pontianak', 'lat': -0.0263, 'lon': 109.3425},
        {'name': 'Banjarmasin', 'lat': -3.3194, 'lon': 114.5905},
        {'name': 'Pekanbaru', 'lat': 0.5071, 'lon': 101.4478},
        {'name': 'Jambi', 'lat': -1.6101, 'lon': 103.6131},
        {'name': 'Palangkaraya', 'lat': -2.2089, 'lon': 113.9213}
    ]
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def collect_data(self):
        """Fetch current weather for Indonesian cities"""
        if not self.api_key:
            logging.warning("⚠ OpenWeather API key not configured - skipping")
            return pd.DataFrame()
        
        weather_data = []
        
        for city in self.CITIES:
            try:
                params = {
                    'lat': city['lat'],
                    'lon': city['lon'],
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                weather_data.append({
                    'city': city['name'],
                    'lat': city['lat'],
                    'lon': city['lon'],
                    'wind_speed': data['wind']['speed'],
                    'wind_deg': data['wind'].get('deg', 0),
                    'temp': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'timestamp': datetime.now()
                })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logging.error(f"✗ Error fetching weather for {city['name']}: {e}")
        
        df = pd.DataFrame(weather_data)
        if not df.empty:
            logging.info(f"✓ Collected weather data for {len(df)} locations")
        return df


class AirQualityCollector:
    """Collects air quality data from WAQI"""
    
    BASE_URL = "https://api.waqi.info/feed"
    
    CITIES = ['jakarta', 'bandung', 'surabaya', 'medan', 'palembang', 
              'pontianak', 'pekanbaru', 'jambi', 'banjarmasin', 'palangkaraya']
    
    def __init__(self, token):
        self.token = token
    
    def collect_data(self):
        """Fetch air quality data for Indonesian cities"""
        if not self.token:
            logging.warning("⚠ WAQI token not configured - skipping")
            return pd.DataFrame()
        
        aqi_data = []
        
        for city in self.CITIES:
            try:
                url = f"{self.BASE_URL}/{city}/?token={self.token}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == 'ok':
                    aqi_data.append({
                        'city': city.capitalize(),
                        'lat': data['data']['city']['geo'][0],
                        'lon': data['data']['city']['geo'][1],
                        'aqi': data['data']['aqi'],
                        'pm25': data['data']['iaqi'].get('pm25', {}).get('v', None),
                        'pm10': data['data']['iaqi'].get('pm10', {}).get('v', None),
                        'timestamp': datetime.now()
                    })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logging.error(f"✗ Error fetching AQI for {city}: {e}")
        
        df = pd.DataFrame(aqi_data)
        if not df.empty:
            logging.info(f"✓ Collected AQI data for {len(df)} cities")
        return df


def run_pipeline():
    """Main pipeline execution"""
    logging.info("\n" + "="*70)
    logging.info("🌋 HazeRadar Data Pipeline Started")
    logging.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*70 + "\n")
    
    db = DatabaseManager()
    db.create_tables()
    
    total_records = 0
    
    try:
        # Collect NASA FIRMS fire data
        logging.info("📡 Collecting NASA FIRMS fire data...")
        firms_collector = NASAFIRMSCollector(NASA_FIRMS_KEY)
        fire_data = firms_collector.collect_data(days=1)
        total_records += db.insert_fire_data(fire_data)
        
        # Collect weather data
        logging.info("\n🌤️  Collecting weather data...")
        weather_collector = WeatherCollector(OPENWEATHER_KEY)
        weather_data = weather_collector.collect_data()
        total_records += db.insert_weather_data(weather_data)
        
        # Collect air quality data
        logging.info("\n💨 Collecting air quality data...")
        aqi_collector = AirQualityCollector(WAQI_TOKEN)
        aqi_data = aqi_collector.collect_data()
        total_records += db.insert_air_quality(aqi_data)
        
        logging.info("\n" + "="*70)
        logging.info(f"✅ Pipeline completed successfully!")
        logging.info(f"📊 Total records collected: {total_records}")
        logging.info("="*70 + "\n")
        
    except Exception as e:
        logging.error(f"\n❌ Pipeline failed: {e}\n")
    finally:
        db.close()


if __name__ == "__main__":
    run_pipeline()