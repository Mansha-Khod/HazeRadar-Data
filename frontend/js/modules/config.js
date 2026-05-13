// js/modules/config.js

// ✅ URL SUDAH BENAR (Pakai /api)
export const API_ROOT = "https://haze-radargnnmodelstimulation-production.up.railway.app/api";

export const SIMULATION_CONFIG = {
    MAP_CENTER: [-1.5, 113],
    MAP_ZOOM: 5,
    INITIAL_PM25: 150,
    HAZE_DECAY: 0.98,
    DEGREE_PER_KMH: 0.00005,
    MAX_RADIUS_KM: 300
};


// --- DATA TRANSLASI BAHASA ---
export const TRANSLATIONS = {
    en: {
        'title': 'HazeRadar: 3-Day Smoke Prediction',
        'nav_haze_prediction': 'Haze Prediction',
        'nav_key_matrix': 'Key Matrix',
        'nav_cloud_vision': 'CrowdVision',
        'timeline_header': 'Prediction Timeline',
        'time_now': 'Now',
        'time_4hr': '4hr',
        'time_8hr': '8hr',
        'time_12hr': '12hr',
        'time_16hr': '16hr',
        'time_20hr': '20hr',
        'time_24hr': '24hr',
        'area_list_header': 'Affected Areas (Click to View)',
        'area_list_empty': 'No locations have been liked yet.',
        'metrics_header': 'Haze Metrics',
        'metric_current_status': 'Current Status',
        'metric_predicted_aqi': 'Predicted AQI',
        'metric_wind_speed': 'PM2.5',
        'metric_visibility': 'Temperature',
        'cv_header': 'CrowdVision Insight',
        'cv_explanation': 'Use CrowdVision to see if you are in a haze zone.',
        'footer_mission_header': 'Our Team & Mission',
        'footer_mission_text': 'HazeRadar is built by a dedicated team focused on environmental technology and data science. We strive to provide accurate, predictive modeling for smoke and haze events across Southeast Asia.',
        'footer_back_to_map': 'Back to Map'
    },
    id: {
        'title': 'HazeRadar: Prediksi Asap 3 Hari',
        'nav_haze_prediction': 'Prediksi Asap',
        'nav_key_matrix': 'Matriks Kunci',
        'nav_cloud_vision': 'Visi Kerumunan',
        'timeline_header': 'Linimasa Prediksi',
        'time_now': 'Sekarang',
        'time_4hr': '4 jam',
        'time_8hr': '8 jam',
        'time_12hr': '12 jam',
        'time_16hr': '16 jam',
        'time_20hr': '20 jam',
        'time_24hr': '24 jam',
        'area_list_header': 'Area Terdampak (Klik untuk Lihat)',
        'area_list_empty': 'Belum ada lokasi yang disukai.',
        'metrics_header': 'Metrik Asap',
        'metric_current_status': 'Status Saat Ini',
        'metric_predicted_aqi': 'Prediksi AQI',
        'metric_wind_speed': 'PM2.5',
        'metric_visibility': 'Suhu',
        'cv_header': 'Wawasan Visi Kerumunan',
        'cv_explanation': 'Gunakan Visi Kerumunan untuk melihat apakah Anda berada di zona asap.',
        'footer_mission_header': 'Tim & Misi Kami',
        'footer_mission_text': 'HazeRadar dibangun oleh tim yang berdedikasi pada teknologi lingkungan dan sains data. Kami berusaha memberikan pemodelan prediktif yang akurat untuk kejadian asap dan kabut di Asia Tenggara.',
        'footer_back_to_map': 'Kembali ke Peta'
    }
};