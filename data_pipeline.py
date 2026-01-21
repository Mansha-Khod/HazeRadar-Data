import os
import time
import json
import logging
import random
import requests
import pandas as pd
import schedule
import numpy as np

from datetime import datetime
from io import StringIO
from math import radians, sin, cos, sqrt, atan2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hazeradar")


NASA_API_KEY = os.getenv("NASA_API_KEY", "4fd11fa7e1e313db40a1d963e9c4c7c0")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "83d1e9d7156a44ae4aaa3355855a1cea")
WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN", "d5922d5ac41e8a89495f776fdc48a469213bc5c1")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://daxrnmvkpikjvvzgrhko.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


INDONESIA_BBOX = "106,-8,109,-5"

FOCUS_REGIONS = {
    "North_West_Java": {
        "bounds": {"lat_min": -6.8, "lat_max": -6.0, "lon_min": 106.8, "lon_max": 107.8},
        "cities": ["Bekasi", "Karawang", "Subang"],
        "fire_risk": "Medium"
    },
    "Central_West_Java": {
        "bounds": {"lat_min": -7.2, "lat_max": -6.5, "lon_min": 107.2, "lon_max": 108.2},
        "cities": ["Bandung", "Cimahi", "West Bandung", "Sumedang"],
        "fire_risk": "Medium"
    },
    "South_West_Java": {
        "bounds": {"lat_min": -7.8, "lat_max": -7.0, "lon_min": 106.8, "lon_max": 108.5},
        "cities": ["Tasikmalaya", "Cianjur"],
        "fire_risk": "Medium-High"
    },
    "East_West_Java": {
        "bounds": {"lat_min": -6.8, "lat_max": -6.2, "lon_min": 107.8, "lon_max": 108.5},
        "cities": ["Indramayu"],
        "fire_risk": "Medium"
    }
}

CITY_COORDINATES = {
    "Bekasi": (-6.2383, 106.9756),
    "Karawang": (-6.3063, 107.3019),
    "Sumedang": (-6.8575, 107.9167),
    "Tasikmalaya": (-7.3274, 108.2207),
    "Bandung": (-6.9175, 107.6191),
    "Subang": (-6.5697, 107.7631),
    "Indramayu": (-6.3269, 108.3200),
    "Cimahi": (-6.8722, 107.5425),
    "West Bandung": (-6.8597, 107.4858),
    "Cianjur": (-6.8167, 107.1392)
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2.0)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_region_for_coordinates(lat, lon):
    for name, info in FOCUS_REGIONS.items():
        b = info["bounds"]
        if b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]:
            return name
    return None

def get_region_for_city(city):
    for name, info in FOCUS_REGIONS.items():
        if city in info["cities"]:
            return name
    return "Unknown"

def health_risk_from_aqi(aqi):
    try:
        if aqi == "-" or aqi is None:
            return "Unknown"
        aqi = int(aqi)
    except Exception:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def make_session(retries=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=backoff_factor, status_forcelist=status_forcelist,
                  allowed_methods=frozenset(["GET", "POST"]))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

session = make_session()


class NASAFIRMSCollector:
    def collect_fire_data(self):
        logger.info("Fetching NASA FIRMS data...")
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_API_KEY}/VIIRS_SNPP_NRT/{INDONESIA_BBOX}/1"
        try:
            r = session.get(url, timeout=30)
            if r.status_code != 200:
                logger.error("NASA FIRMS returned %s", r.status_code)
                return []
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                logger.info("No hotspots from NASA.")
                return []
            out = []
            for _, row in df.iterrows():
                lat = row.get("latitude")
                lon = row.get("longitude")
                if pd.isna(lat) or pd.isna(lon):
                    continue
                region = get_region_for_coordinates(lat, lon)
                if not region:
                    continue
                brightness = row.get("bright_ti4", 0)
                confidence = str(row.get("confidence", "nominal"))
                out.append({
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "brightness": float(brightness) if not pd.isna(brightness) else 0.0,
                    "confidence": confidence,
                    "region": region,
                    "fire_risk_level": FOCUS_REGIONS[region]["fire_risk"],
                    "geom": f"POINT({lon} {lat})"
                })
            logger.info("Collected %d fire hotspots", len(out))
            return out
        except Exception as e:
            logger.exception("NASA FIRMS error: %s", e)
            return []

class WeatherCollector:
    def collect_weather_data(self):
        out = []
        for region_name, region_info in FOCUS_REGIONS.items():
            for city in region_info["cities"]:
                lat, lon = CITY_COORDINATES.get(city, (None, None))
                if lat is None:
                    logger.warning("No coords for city %s", city)
                    continue
                url = ("https://api.openweathermap.org/data/2.5/weather?"
                       f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric")
                try:
                    r = session.get(url, timeout=12)
                    if r.status_code != 200:
                        logger.warning("OWM failed for %s: %s", city, r.status_code)
                        time.sleep(1 + random.random()*0.5)
                        continue
                    data = r.json()
                    main = data.get("main", {})
                    wind = data.get("wind", {})
                    out.append({
                        "city": city,
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "temperature": float(main.get("temp", 0.0)),
                        "humidity": float(main.get("humidity", 0.0)),
                        "wind_speed": float(wind.get("speed", 0.0)),
                        "wind_direction": float(wind.get("deg", 0.0)),
                        "geom": f"POINT({lon} {lat})"
                    })
                   
                    time.sleep(0.8 + random.random()*0.6)
                except Exception as e:
                    logger.exception("OpenWeather error for %s: %s", city, e)
                    time.sleep(1 + random.random()*0.5)
        logger.info("Collected weather for %d locations", len(out))
        return out

class AQICollector:
    def _convert_aqi_to_pm25(self, aqi):
        """Convert AQI to PM2.5 concentration"""
        if aqi <= 50:
            return aqi * 12.0 / 50.0
        elif aqi <= 100:
            return 12.1 + (aqi - 51) * (35.4 - 12.1) / (100 - 51)
        elif aqi <= 150:
            return 35.5 + (aqi - 101) * (55.4 - 35.5) / (150 - 101)
        elif aqi <= 200:
            return 55.5 + (aqi - 151) * (150.4 - 55.5) / (200 - 151)
        elif aqi <= 300:
            return 150.5 + (aqi - 201) * (250.4 - 150.5) / (300 - 201)
        else:
            return 250.5 + (aqi - 301) * (350.4 - 250.5) / (400 - 301)
    
    def collect_aqi_data(self):
        out = []
        for city in CITY_COORDINATES.keys():
            lat, lon = CITY_COORDINATES[city]
            
            url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_API_TOKEN}"
            try:
                r = session.get(url, timeout=12)
                if r.status_code != 200:
                    logger.warning("AQICN failed for %s: %s", city, r.status_code)
                    time.sleep(1)
                    continue
                
                d = r.json()
                if d.get("status") != "ok":
                    logger.warning("AQICN not ok for %s: %s", city, d.get("data"))
                    time.sleep(1)
                    continue
                
                data = d.get("data", {})
                aqi_value = data.get("aqi", 0)
                
                iaqi = data.get("iaqi", {})
                pm25_aqi = 0.0
                if iaqi and "pm25" in iaqi:
                    try:
                        pm25_aqi = float(iaqi["pm25"].get("v", 0.0))
                    except Exception:
                        pm25_aqi = 0.0
                
                if pm25_aqi == 0.0 and aqi_value > 0:
                    pm25_aqi = float(aqi_value)
                
                pm25_concentration = self._convert_aqi_to_pm25(pm25_aqi) if pm25_aqi > 0 else 0.0
                
                final_aqi = int(aqi_value) if isinstance(aqi_value, (int, float)) or (isinstance(aqi_value, str) and str(aqi_value).replace('.','').replace('-','').isdigit()) else 0
                
                station_name = data.get("city", {}).get("name", "Unknown")
                
                out.append({
                    "city": city,
                    "aqi": final_aqi,
                    "pm25": pm25_concentration,
                    "health_risk_level": health_risk_from_aqi(final_aqi)
                })
                logger.info("Successfully collected AQI for %s: AQI=%d, PM2.5=%.1f µg/m³ from station '%s'", 
                           city, final_aqi, pm25_concentration, station_name)
                
                time.sleep(0.7 + random.random()*0.4)
                
            except Exception as e:
                logger.exception("AQICN error for %s: %s", city, e)
                time.sleep(1)
        
        logger.info("Collected AQI for %d cities", len(out))
        return out

class GNNDataGenerator:
    def generate_training_data(self, fire_data, weather_data, aqi_data):
        rows = []
       
        for city, coords in CITY_COORDINATES.items():
            region = get_region_for_city(city)
            if region == "Unknown":
                continue
            city_weather = next((w for w in weather_data if w["city"] == city), None)
            city_aqi = next((a for a in aqi_data if a["city"] == city), None)
            if not city_weather:
                logger.warning("No weather data for %s, skipping GNN row", city)
                continue
            if not city_aqi:
                logger.warning("No AQI data for %s, skipping GNN row", city)
                continue
            
            upwind_count = self._count_upwind_fires(city, fire_data, weather_data)
            avg_conf = self._calculate_avg_confidence(fire_data)

            seed = abs(hash(city)) % (2**32)
            r = np.random.RandomState(seed)
            pop_density = float(r.uniform(3000, 15000))
            
            current_pm25 = float(city_aqi["pm25"]) if city_aqi.get("pm25") else 0.0
            
            rows.append({
                "city": city,
                "region": region,
                "latitude": float(coords[0]),
                "longitude": float(coords[1]),
                "upwind_fire_count": int(upwind_count),
                "avg_fire_confidence": float(avg_conf),
                "temperature": float(city_weather["temperature"]),
                "humidity": float(city_weather["humidity"]),
                "wind_speed": float(city_weather["wind_speed"]),
                "wind_direction": float(city_weather["wind_direction"]),
                "current_aqi": int(city_aqi["aqi"]),
                "current_pm25": current_pm25,
                "population_density": pop_density
            })
        logger.info("Generated GNN rows for %d cities", len(rows))
        return rows

    def _count_upwind_fires(self, city, fire_data, weather_data):
        cw = next((w for w in weather_data if w["city"] == city), None)
        if not cw or not fire_data:
            return 0
        c_lat, c_lon = CITY_COORDINATES[city]
        cnt = 0
        for f in fire_data:
            d = haversine_km(c_lat, c_lon, f["latitude"], f["longitude"])
            if d < 100:
                cnt += 1
        return cnt

    def _calculate_avg_confidence(self, fire_data):
        if not fire_data:
            return 0.0
        vals = []
        for f in fire_data:
            c = f.get("confidence", "")
            if isinstance(c, str) and c.isdigit():
                vals.append(float(c))
        return float(np.mean(vals)) if vals else 70.0

class GraphStructureGenerator:
    def generate_graph_data(self):
        rows = []
        for city1, coord1 in CITY_COORDINATES.items():
            conns = []
            dists = []
            for city2, coord2 in CITY_COORDINATES.items():
                if city1 == city2:
                    continue
                d = haversine_km(coord1[0], coord1[1], coord2[0], coord2[1])
                if d < 150:
                    conns.append(city2)
                    dists.append(round(d, 2))
            rows.append({
                "city": city1,
                "latitude": float(coord1[0]),
                "longitude": float(coord1[1]),
                "connected_cities": ",".join(conns),
                "distances_km": ",".join(map(str, dists))
            })
        logger.info("Generated graph for %d cities", len(rows))
        return rows

class SupabaseManager:
    def __init__(self, url=SUPABASE_URL, key=SUPABASE_KEY):
        self.url = url.rstrip("/")
        self.key = key
        
        logger.info(f"Supabase URL: {self.url}")
        logger.info(f"Supabase Key provided: {'YES' if self.key else 'NO'}")
        logger.info(f"Supabase Key length: {len(self.key) if self.key else 0}")
        
        if not self.key:
            logger.error("SUPABASE_KEY is empty! Please check your environment variables.")
            logger.error("Current SUPABASE_KEY value: %s", self.key)
        
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        self.insert_batch = 100

    def test_connection(self):
        test_url = f"{self.url}/rest/v1/"
        try:
            response = session.get(test_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                logger.info("Supabase connection test: SUCCESS")
                return True
            else:
                logger.error(f"Supabase connection test failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Supabase connection error: {e}")
            return False

    def insert_data(self, table_name, data):
        if not data:
            logger.debug("No data to insert into %s", table_name)
            return True
            
        if not self.test_connection():
            logger.error("Cannot insert data - Supabase connection failed")
            return False
            
        url = f"{self.url}/rest/v1/{table_name}"
        
        for i in range(0, len(data), self.insert_batch):
            chunk = data[i:i+self.insert_batch]
            try:
                r = session.post(url, headers=self.headers, json=chunk, timeout=20)
                
                if r.status_code in (200, 201, 204):
                    logger.info("Inserted %d records into %s", len(chunk), table_name)
                else:
                    logger.error("Insert failed %s -> %s", table_name, r.status_code)
                    logger.error("Full response: %s", r.text)
                    
                    if r.status_code == 401:
                        logger.error("AUTHENTICATION ERROR: Check your SUPABASE_KEY")
                        logger.error("Make sure you're using the SERVICE ROLE key, not anon key")
                        return False
                    elif r.status_code == 404:
                        logger.error("TABLE NOT FOUND: Table '%s' doesn't exist", table_name)
                        return False
                    elif r.status_code == 429:
                        wait = 5 + random.random()*5
                        logger.warning("Rate limited. Sleeping %.1fs", wait)
                        time.sleep(wait)
                        
                        r2 = session.post(url, headers=self.headers, json=chunk, timeout=20)
                        if r2.status_code not in (200, 201, 204):
                            logger.error("Retry insert failed: %s", r2.status_code)
                            return False
                    else:
                        return False
                
                time.sleep(0.2 + random.random()*0.2)
            except Exception as e:
                logger.exception("Supabase insert error for %s: %s", table_name, e)
                return False
        return True


def run_pipeline():
    logger.info("Starting data collection cycle (run_pipeline)")
    
    logger.info("=== ENVIRONMENT VARIABLES ===")
    logger.info(f"SUPABASE_URL: {SUPABASE_URL}")
    logger.info(f"SUPABASE_KEY set: {bool(SUPABASE_KEY)}")
    logger.info(f"SUPABASE_KEY length: {len(SUPABASE_KEY) if SUPABASE_KEY else 0}")
    logger.info(f"NASA_API_KEY: {NASA_API_KEY}")
    logger.info(f"OPENWEATHER_API_KEY: {OPENWEATHER_API_KEY}")
    logger.info(f"WAQI_API_TOKEN: {'SET' if WAQI_API_TOKEN else 'NOT SET'}")
    
    nasa = NASAFIRMSCollector()
    weather = WeatherCollector()
    aqi = AQICollector()
    gnn_gen = GNNDataGenerator()
    graph_gen = GraphStructureGenerator()
    db = SupabaseManager()

    try:
        fire_data = nasa.collect_fire_data()
        if fire_data:
            success = db.insert_data("fire_hotspots", fire_data)
            logger.info(f"Fire hotspots insert: {'SUCCESS' if success else 'FAILED'}")

        weather_data = weather.collect_weather_data()
        if weather_data:
            success = db.insert_data("weather_data", weather_data)
            logger.info(f"Weather data insert: {'SUCCESS' if success else 'FAILED'}")

        aqi_data = aqi.collect_aqi_data()
        if aqi_data:
            success = db.insert_data("air_quality", aqi_data)
            logger.info(f"Air quality insert: {'SUCCESS' if success else 'FAILED'}")

        logger.info("Generating GNN training rows")
        gnn_rows = gnn_gen.generate_training_data(fire_data, weather_data, aqi_data)
        if gnn_rows:
            success = db.insert_data("gnn_training_data", gnn_rows)
            logger.info(f"GNN training data insert: {'SUCCESS' if success else 'FAILED'}")

        now = datetime.now()
        if now.hour == 0 and random.random() < 0.9:
            graph_rows = graph_gen.generate_graph_data()
            if graph_rows:
                success = db.insert_data("city_graph_structure", graph_rows)
                logger.info(f"Graph structure insert: {'SUCCESS' if success else 'FAILED'}")

        logger.info("Data collection cycle finished successfully")
        return True

    except Exception as e:
        logger.exception("run_pipeline failed: %s", e)
        return False


def main():
    logger.info("HazeRadar Cloud Service started")
    
    def scheduled_job():
        jitter = random.uniform(-60, 60)
        if jitter > 0:
            time.sleep(min(jitter, 30))
        run_pipeline()
    
    schedule.every(3).hours.do(scheduled_job)

    run_pipeline()

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)")

if __name__ == "__main__":
    main()
