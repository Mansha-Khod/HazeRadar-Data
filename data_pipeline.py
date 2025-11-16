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


INDONESIA_BBOX = "95,-11,141,7"

FOCUS_REGIONS = {
    "Riau": {"bounds": {"lat_min": -1.0, "lat_max": 2.0, "lon_min": 100.0, "lon_max": 103.0},
             "cities": ["Pekanbaru", "Dumai", "Bengkalis"], "fire_risk": "Very High"},
    "Central_Kalimantan": {"bounds": {"lat_min": -4.0, "lat_max": -1.0, "lon_min": 111.0, "lon_max": 116.0},
                           "cities": ["Palangkaraya", "Sampit", "Pangkalan Bun"], "fire_risk": "High"},
    "South_Sumatra": {"bounds": {"lat_min": -5.0, "lat_max": -1.0, "lon_min": 102.0, "lon_max": 107.0},
                      "cities": ["Palembang", "Prabumulih", "Lubuklinggau"], "fire_risk": "High"},
    "West_Kalimantan": {"bounds": {"lat_min": -2.0, "lat_max": 1.0, "lon_min": 108.0, "lon_max": 111.0},
                        "cities": ["Pontianak", "Singkawang", "Sintang"], "fire_risk": "Medium-High"},
    "Jambi": {"bounds": {"lat_min": -2.5, "lat_max": -0.5, "lon_min": 101.5, "lon_max": 105.5},
              "cities": ["Jambi", "Sungai Penuh", "Muara Bungo"], "fire_risk": "Medium-High"}
}

CITY_COORDINATES = {
    "Pekanbaru": (0.5071, 101.4478), "Dumai": (1.6654, 101.4476), "Bengkalis": (1.4892, 102.0795),
    "Palangkaraya": (-2.2086, 113.9167), "Sampit": (-2.5333, 112.95), "Pangkalan Bun": (-2.6833, 111.6167),
    "Palembang": (-2.9910, 104.7574), "Prabumulih": (-3.4324, 104.2345), "Lubuklinggau": (-3.2967, 102.8617),
    "Pontianak": (-0.0226, 109.3425), "Singkawang": (0.9079, 108.9846), "Sintang": (0.0694, 111.4931),
    "Jambi": (-1.61, 103.6072), "Sungai Penuh": (-2.0631, 101.3872), "Muara Bungo": (-1.5117, 102.1036)
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
                        "region": region_name,
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
    def collect_aqi_data(self):
        out = []
        for city in CITY_COORDINATES.keys():
            qcity = city.lower().replace(" ", "-")
            url = f"https://api.waqi.info/feed/{qcity}/?token={WAQI_API_TOKEN}"
            try:
                r = session.get(url, timeout=12)
                if r.status_code != 200:
                    logger.warning("WAQI failed for %s: %s", city, r.status_code)
                    time.sleep(0.8 + random.random()*0.5)
                    continue
                d = r.json()
                if d.get("status") != "ok":
                   
                    logger.debug("WAQI not ok for %s: %s", city, d.get("data"))
                    time.sleep(0.4 + random.random()*0.5)
                    continue
                aqi = d["data"].get("aqi", 0)
                iaqi = d["data"].get("iaqi", {})
                pm25 = 0.0
                if iaqi and "pm25" in iaqi:
                    try:
                        pm25 = float(iaqi["pm25"].get("v", 0.0))
                    except Exception:
                        pm25 = 0.0
                out.append({
                    "region": get_region_for_city(city),
                    "city": city,
                    "aqi": int(aqi) if isinstance(aqi, (int, float)) or (isinstance(aqi, str) and aqi.isdigit()) else 0,
                    "pm25": pm25,
                    "health_risk_level": health_risk_from_aqi(aqi)
                })
                time.sleep(0.7 + random.random()*0.4)
            except Exception as e:
                logger.exception("WAQI error for %s: %s", city, e)
                time.sleep(1 + random.random()*0.5)
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
            if not city_weather or not city_aqi:
                continue
            upwind_count = self._count_upwind_fires(city, fire_data, weather_data)
            avg_conf = self._calculate_avg_confidence(fire_data)

            seed = abs(hash(city)) % (2**32)
            r = np.random.RandomState(seed)
            pop_density = float(r.uniform(2000, 12000))
            target_pm25 = float(city_aqi["pm25"] * r.uniform(0.8, 1.5)) if city_aqi.get("pm25") is not None else 0.0
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
                "population_density": pop_density,
                "target_pm25_24h": target_pm25
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
            if d < 200:
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
                if d < 300:
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
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}" if self.key else "",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        self.insert_batch = 100

    def insert_data(self, table_name, data):
        if not data:
            logger.debug("No data to insert into %s", table_name)
            return True
        url = f"{self.url}/rest/v1/{table_name}"
        
        for i in range(0, len(data), self.insert_batch):
            chunk = data[i:i+self.insert_batch]
            try:
                r = session.post(url, headers=self.headers, json=chunk, timeout=20)
                if r.status_code in (200, 201, 204):
                    logger.info("Inserted %d records into %s", len(chunk), table_name)
                else:
                    
                    logger.warning("Insert failed %s -> %s / resp: %s", table_name, r.status_code, r.text[:200])
                    
                    if r.status_code == 429:
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
    nasa = NASAFIRMSCollector()
    weather = WeatherCollector()
    aqi = AQICollector()
    gnn_gen = GNNDataGenerator()
    graph_gen = GraphStructureGenerator()
    db = SupabaseManager()

    try:
        fire_data = nasa.collect_fire_data()
        if fire_data:
            db.insert_data("fire_hotspots", fire_data)

        weather_data = weather.collect_weather_data()
        if weather_data:
            db.insert_data("weather_data", weather_data)

        aqi_data = aqi.collect_aqi_data()
        if aqi_data:
            db.insert_data("air_quality", aqi_data)

        logger.info("Generating GNN training rows")
        gnn_rows = gnn_gen.generate_training_data(fire_data, weather_data, aqi_data)
        if gnn_rows:
            db.insert_data("gnn_training_data", gnn_rows)

        now = datetime.now()
        if now.hour == 0 and random.random() < 0.9:
            graph_rows = graph_gen.generate_graph_data()
            if graph_rows:
                db.insert_data("city_graph_structure", graph_rows)

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
