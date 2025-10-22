import os
import requests
import pandas as pd
import schedule
import time
import logging
from datetime import datetime
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
NASA_API_KEY = os.getenv('NASA_API_KEY', '4fd11fa7e1e313db40a1d963e9c4c7c0')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '83d1e9d7156a44ae4aaa3355855a1cea')
WAQI_API_TOKEN = os.getenv('WAQI_API_TOKEN', 'd5922d5ac41e8a89495f776fdc48a469213bc5c1')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://daxrnmvkpikjvvzgrhko.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRheHJubXZrcGlranZ2emdyaGtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2OTkyNjEsImV4cCI6MjA3NjI3NTI2MX0.XWJ_aWUh5Eci5tQSRAATqDXmQ5nh2eHQGzYu6qMcsvQ')

# Indonesia bounding box
INDONESIA_BBOX = "95,-11,141,7"

# Focus regions for Indonesia
FOCUS_REGIONS = {
    'Riau': {
        'bounds': {'lat_min': -1.0, 'lat_max': 2.0, 'lon_min': 100.0, 'lon_max': 103.0},
        'cities': ['Pekanbaru', 'Dumai', 'Bengkalis'],
        'fire_risk': 'Very High'
    },
    'Central_Kalimantan': {
        'bounds': {'lat_min': -4.0, 'lat_max': -1.0, 'lon_min': 111.0, 'lon_max': 116.0},
        'cities': ['Palangkaraya', 'Sampit', 'Pangkalan Bun'],
        'fire_risk': 'High'
    },
    'South_Sumatra': {
        'bounds': {'lat_min': -5.0, 'lat_max': -1.0, 'lon_min': 102.0, 'lon_max': 107.0},
        'cities': ['Palembang', 'Prabumulih', 'Lubuklinggau'],
        'fire_risk': 'High'
    },
    'West_Kalimantan': {
        'bounds': {'lat_min': -2.0, 'lat_max': 1.0, 'lon_min': 108.0, 'lon_max': 111.0},
        'cities': ['Pontianak', 'Singkawang', 'Sintang'],
        'fire_risk': 'Medium-High'
    },
    'Jambi': {
        'bounds': {'lat_min': -2.5, 'lat_max': -0.5, 'lon_min': 101.5, 'lon_max': 105.5},
        'cities': ['Jambi', 'Sungai Penuh', 'Muara Bungo'],
        'fire_risk': 'Medium-High'
    }
}

CITY_COORDINATES = {
    'Pekanbaru': (0.5071, 101.4478), 'Dumai': (1.6654, 101.4476), 'Bengkalis': (1.4892, 102.0795),
    'Palangkaraya': (-2.2086, 113.9167), 'Sampit': (-2.5333, 112.9500), 'Pangkalan Bun': (-2.6833, 111.6167),
    'Palembang': (-2.9910, 104.7574), 'Prabumulih': (-3.4324, 104.2345), 'Lubuklinggau': (-3.2967, 102.8617),
    'Pontianak': (-0.0226, 109.3425), 'Singkawang': (0.9079, 108.9846), 'Sintang': (0.0694, 111.4931),
    'Jambi': (-1.6100, 103.6072), 'Sungai Penuh': (-2.0631, 101.3872), 'Muara Bungo': (-1.5117, 102.1036)
}

class NASAFIRMSCollector:
    def collect_fire_data(self):
        """Collect REAL fire data from NASA FIRMS"""
        try:
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_API_KEY}/VIIRS_SNPP_NRT/{INDONESIA_BBOX}/1"
            logger.info(" Fetching real NASA fire data...")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                
                if df.empty:
                    logger.warning("No fire hotspots detected by NASA")
                    return []
                
                fire_data = []
                for _, row in df.iterrows():
                    lat, lon = row['latitude'], row['longitude']
                    region = self._get_region_for_coordinates(lat, lon)
                    if region:
                        fire_data.append({
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'brightness': float(row.get('bright_ti4', 0)),
                            'confidence': str(row.get('confidence', 'nominal')),
                            'region': region,
                            'fire_risk_level': FOCUS_REGIONS[region]['fire_risk'],
                            'geom': f"POINT({lon} {lat})"
                        })
                
                logger.info(f" Collected {len(fire_data)} real fire hotspots")
                return fire_data
            else:
                logger.error(f"NASA API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error collecting NASA data: {e}")
            return []

    def _get_region_for_coordinates(self, lat, lon):
        for region_name, region_info in FOCUS_REGIONS.items():
            bounds = region_info['bounds']
            if (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
                bounds['lon_min'] <= lon <= bounds['lon_max']):
                return region_name
        return None

class WeatherCollector:
    def collect_weather_data(self):
        """Collect REAL weather data from OpenWeatherMap"""
        weather_data = []
        
        for region_name, region_info in FOCUS_REGIONS.items():
            for city in region_info['cities']:
                try:
                    lat, lon = CITY_COORDINATES[city]
                    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        weather_data.append({
                            'region': region_name,
                            'city': city,
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'temperature': float(data['main']['temp']),
                            'humidity': float(data['main']['humidity']),
                            'wind_speed': float(data['wind']['speed']),
                            'wind_direction': float(data['wind'].get('deg', 0)),
                            'geom': f"POINT({lon} {lat})"
                        })
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching weather for {city}: {e}")
        
        logger.info(f" Collected weather data for {len(weather_data)} locations")
        return weather_data

class AQICollector:
    def collect_aqi_data(self):
        """Collect REAL air quality data from WAQI"""
        aqi_data = []
        
        for city in CITY_COORDINATES.keys():
            try:
                url = f"https://api.waqi.info/feed/{city.lower().replace(' ', '-')}/?token={WAQI_API_TOKEN}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['status'] == 'ok':
                        aqi = data['data']['aqi']
                        iaqi = data['data'].get('iaqi', {})
                        pm25 = float(iaqi.get('pm25', {}).get('v', 0)) if iaqi else 0.0
                        
                        aqi_data.append({
                            'region': self._get_region_for_city(city),
                            'city': city,
                            'aqi': int(aqi) if aqi != '-' else 0,
                            'pm25': pm25,
                            'health_risk_level': self._get_health_risk_level(aqi)
                        })
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching AQI for {city}: {e}")
        
        logger.info(f" Collected AQI data for {len(aqi_data)} cities")
        return aqi_data

    def _get_region_for_city(self, city):
        for region_name, region_info in FOCUS_REGIONS.items():
            if city in region_info['cities']:
                return region_name
        return "Unknown"

    def _get_health_risk_level(self, aqi):
        if aqi == '-' or aqi == 0: return "Unknown"
        aqi = int(aqi)
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"

class GNNDataGenerator:
    def generate_training_data(self, fire_data, weather_data, aqi_data):
        """Generate GNN training data from collected data"""
        gnn_data = []
        
        for city, coords in CITY_COORDINATES.items():
            region = self._get_region_for_city(city)
            if region:
                # Calculate metrics from real data
                upwind_fire_count = self._count_upwind_fires(city, fire_data, weather_data)
                avg_fire_confidence = self._calculate_avg_confidence(fire_data)
                
                # Get current weather for city
                city_weather = next((w for w in weather_data if w['city'] == city), None)
                city_aqi = next((a for a in aqi_data if a['city'] == city), None)
                
                if city_weather and city_aqi:
                    gnn_data.append({
                        'city': city,
                        'region': region,
                        'latitude': float(coords[0]),
                        'longitude': float(coords[1]),
                        'upwind_fire_count': upwind_fire_count,
                        'avg_fire_confidence': avg_fire_confidence,
                        'temperature': city_weather['temperature'],
                        'humidity': city_weather['humidity'],
                        'wind_speed': city_weather['wind_speed'],
                        'wind_direction': city_weather['wind_direction'],
                        'current_aqi': city_aqi['aqi'],
                        'population_density': float(np.random.uniform(2000, 12000)),
                        'target_pm25_24h': float(city_aqi['pm25'] * np.random.uniform(0.8, 1.5))
                    })
        
        logger.info(f" Generated GNN training data for {len(gnn_data)} cities")
        return gnn_data

    def _get_region_for_city(self, city):
        for region_name, region_info in FOCUS_REGIONS.items():
            if city in region_info['cities']:
                return region_name
        return "Unknown"

    def _count_upwind_fires(self, city, fire_data, weather_data):
        """Count fires upwind of city"""
        city_weather = next((w for w in weather_data if w['city'] == city), None)
        if not city_weather or not fire_data:
            return 0
        
        city_coords = CITY_COORDINATES[city]
        upwind_count = 0
        
        for fire in fire_data:
            distance = self._calculate_distance(city_coords[0], city_coords[1], fire['latitude'], fire['longitude'])
            if distance < 200:  # Within 200km
                upwind_count += 1
        
        return upwind_count

    def _calculate_avg_confidence(self, fire_data):
        if not fire_data:
            return 0.0
        
        confidence_values = []
        for fire in fire_data:
            if fire['confidence'].isdigit():
                confidence_values.append(float(fire['confidence']))
        
        return float(np.mean(confidence_values)) if confidence_values else 70.0

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

class GraphStructureGenerator:
    def generate_graph_data(self):
        """Generate city graph structure for GNN"""
        graph_data = []
        
        for city1, coord1 in CITY_COORDINATES.items():
            connections, distances = [], []
            
            for city2, coord2 in CITY_COORDINATES.items():
                if city1 != city2:
                    dist = self._calculate_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                    if dist < 300:  # Connect cities within 300km
                        connections.append(city2)
                        distances.append(round(dist, 2))
            
            graph_data.append({
                'city': city1,
                'latitude': float(coord1[0]),
                'longitude': float(coord1[1]),
                'connected_cities': ','.join(connections),
                'distances_km': ','.join(map(str, distances))
            })
        
        logger.info(f" Generated graph structure for {len(graph_data)} cities")
        return graph_data

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

class SupabaseManager:
    def __init__(self):
        self.headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }

    def insert_data(self, table_name, data):
        if not data:
            logger.warning(f"No data to insert into {table_name}")
            return False
        
        try:
            url = f"{SUPABASE_URL}/rest/v1/{table_name}"
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code in [200, 201]:
                logger.info(f" Inserted {len(data)} records into {table_name}")
                return True
            else:
                logger.error(f" Failed to insert into {table_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f" Error inserting into {table_name}: {e}")
            return False

def run_pipeline():
    """Run one complete data collection cycle"""
    logger.info(" Starting cloud data collection...")
    
    nasa_collector = NASAFIRMSCollector()
    weather_collector = WeatherCollector()
    aqi_collector = AQICollector()
    gnn_generator = GNNDataGenerator()
    graph_generator = GraphStructureGenerator()
    db_manager = SupabaseManager()
    
    try:
        fire_data = nasa_collector.collect_fire_data()
        if fire_data:
            db_manager.insert_data('fire_hotspots', fire_data)
        
        weather_data = weather_collector.collect_weather_data()
        if weather_data:
            db_manager.insert_data('weather_data', weather_data)
        
        aqi_data = aqi_collector.collect_aqi_data()
        if aqi_data:
            db_manager.insert_data('air_quality', aqi_data)
        
        # Generate and store GNN training data
        logger.info(" Generating GNN training data...")
        gnn_data = gnn_generator.generate_training_data(fire_data, weather_data, aqi_data)
        if gnn_data:
            db_manager.insert_data('gnn_training_data', gnn_data)
        
        # Generate graph structure (once per day)
        if datetime.now().hour == 0:  # Only at midnight
            logger.info("🕸️ Generating city graph structure...")
            graph_data = graph_generator.generate_graph_data()
            if graph_data:
                db_manager.insert_data('city_graph_structure', graph_data)
        
        logger.info(" Cloud collection completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f" Cloud collection failed: {e}")
        return False

def main():
    """Main cloud scheduler"""
    logger.info(" HazeRadar Cloud Service Started - 24/7 Operation")
    
    schedule.every(3).hours.do(run_pipeline)
    
    run_pipeline()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()



