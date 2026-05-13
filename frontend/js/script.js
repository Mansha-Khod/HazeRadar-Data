// --- CONFIGURATION ---
// Gunakan URL Backend Railway Anda
const HAZE_API_BASE_URL = window.NEXT_PUBLIC_API_URL || 'https://haze-radargnnmodelrealtime-production.up.railway.app';

// Konfigurasi Supabase (Untuk Fallback/Cadangan jika Railway down)
const SUPABASE_URL = 'https://daxrnmvkpikjvvzgrhko.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRheHJubXZrcGlranZ2emdyaGtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2OTkyNjEsImV4cCI6MjA3NjI3NTI2MX0.XWJ_aWUh5Eci5tQSRAATqDXmQ5nh2eHQGzYu6qMcsvQ';

// Daftar kota resmi (Jawa Barat)
const OFFICIAL_CITIES = [
    "Bekasi", "Karawang", "Sumedang", "Tasikmalaya", "Bandung",
    "Subang", "Indramayu", "Cimahi", "West Bandung", "Cianjur"
];

// Peta fokus ke Jawa Barat
const MAP_CENTER = [-6.9175, 107.6191]; // Bandung
const MAP_ZOOM = 9;

// --- LANGUAGE TRANSLATION DATA ---
const translations = {
    en: {
        'title': 'HazeRadar: West Java Air Quality',
        'nav_haze_prediction': 'Haze Prediction',
        'nav_key_matrix': 'Key Matrix',
        'nav_cloud_vision': 'CrowdVision',
        'timeline_header': 'Prediction Timeline',
        'time_now': 'Now',
        'time_12hr': '12hr',
        'time_24hr': '24hr',
        'time_36hr': '36hr',
        'time_48hr': '48hr',
        'time_60hr': '60hr',
        'area_list_header': 'Monitored Cities (West Java)',
        'area_list_empty': 'Loading city data...',
        'metrics_header': 'Air Quality Metrics',
        'metric_current_status': 'Status',
        'metric_predicted_aqi': 'AQI',
        'metric_wind_speed': 'PM2.5',
        'metric_visibility': 'Temperature',
        
        'cv_header': 'CrowdVision Insight',
        'cv_explanation': 'Use CrowdVision to see if you are in a safe environment.',
        'cv_browse_link': 'Click to browse',
        'cv_drag_drop': 'or drag and drop',
        'cv_file_type': 'PNG, JPG, or JPEG (Max 5MB per file)',
        'cv_button_analyze': 'Analyze Photo',
        'cv_button_analyze_count': 'Analyze {count} Photo',
        'api_title_analyzing': 'Analyzing...',
        
        'haze_insight_header': 'Haze Insight',
        'haze_insight_text': 'Simulate your own smoke prediction.',
        'haze_insight_button': 'Start Simulation Tool',
        'footer_mission_header': 'Our Team & Mission',
        'footer_mission_text': 'HazeRadar provides accurate, predictive modeling for air quality in West Java using GNN technology.',
        'footer_back_to_map': 'Back to Map',
        'modal_close': 'Close',
        'ews_title': 'Early Warning System',
        'ews_desc': 'Get notifications when air quality drops.',
        'ews_name_label': 'Full Name',
        'ews_name_placeholder': 'e.g. John Doe',
        'ews_phone_label': 'WhatsApp Number',
        'ews_phone_placeholder': 'e.g. +628123456789',
        'ews_submit_btn': 'Activate Alert',
        'ews_success_msg': 'Thank you, {name}! Registered.',
        'ews_tooltip_text': '👋 Want Early Warning alerts?',
    },
    id: {
        'title': 'HazeRadar: Kualitas Udara Jawa Barat',
        'nav_haze_prediction': 'Prediksi Asap',
        'nav_key_matrix': 'Matriks Kunci',
        'nav_cloud_vision': 'CrowdVision',
        'timeline_header': 'Garis Waktu Prediksi',
        'time_now': 'Sekarang',
        'time_12hr': '12 jam',
        'time_24hr': '24 jam',
        'time_36hr': '36 jam',
        'time_48hr': '48 jam',
        'time_60hr': '60 jam',
        'area_list_header': 'Kota Terpantau (Jawa Barat)',
        'area_list_empty': 'Memuat data kota...',
        'metrics_header': 'Metrik Kualitas Udara',
        'metric_current_status': 'Status',
        'metric_predicted_aqi': 'AQI',
        'metric_wind_speed': 'PM2.5',
        'metric_visibility': 'Suhu',
        
        'cv_header': 'Wawasan CrowdVision',
        'cv_explanation': 'Gunakan CrowdVision untuk melihat apakah lingkungan Anda aman.',
        'cv_browse_link': 'Klik untuk mencari',
        'cv_drag_drop': 'atau seret foto ke sini',
        'cv_file_type': 'PNG, JPG, atau JPEG (Maks 5MB)',
        'cv_button_analyze': 'Analisis Foto',
        'cv_button_analyze_count': 'Analisis {count} Foto',
        'api_title_analyzing': 'Menganalisis...',
        
        'haze_insight_header': 'Wawasan Kabut',
        'haze_insight_text': 'Simulasikan prediksi asap Anda.',
        'haze_insight_button': 'Mulai Simulasi',
        'footer_mission_header': 'Misi Kami',
        'footer_mission_text': 'HazeRadar menyediakan pemodelan prediksi akurat untuk kualitas udara di Jawa Barat menggunakan teknologi GNN.',
        'footer_back_to_map': 'Kembali ke Peta',
        'modal_close': 'Tutup',
        'ews_title': 'Sistem Peringatan Dini',
        'ews_desc': 'Dapatkan notifikasi saat kualitas udara memburuk.',
        'ews_name_label': 'Nama Lengkap',
        'ews_name_placeholder': 'Contoh: Budi Santoso',
        'ews_phone_label': 'Nomor WhatsApp',
        'ews_phone_placeholder': 'Contoh: 08123456789',
        'ews_submit_btn': 'Aktifkan',
        'ews_success_msg': 'Terima kasih, {name}! Terdaftar.',
        'ews_tooltip_text': '👋 Mau dapat notifikasi peringatan dini?',
    },
    jv: {
        'title': 'HazeRadar: Kualitas Hawa Jawa Kulon',
        'nav_haze_prediction': 'Predhiksi Asep',
        'nav_key_matrix': 'Matriks Utama',
        'nav_cloud_vision': 'CrowdVision',
        'timeline_header': 'Garis Wektu Predhiksi',
        'time_now': 'Saiki',
        'time_12hr': '12 jam',
        'time_24hr': '24 jam',
        'time_36hr': '36 jam',
        'time_48hr': '48 jam',
        'time_60hr': '60 jam',
        'area_list_header': 'Kutha Sing Dipantau',
        'area_list_empty': 'Nunggu data...',
        'metrics_header': 'Metrik Kualitas Hawa',
        'metric_current_status': 'Status',
        'metric_predicted_aqi': 'AQI',
        'metric_wind_speed': 'PM2.5',
        'metric_visibility': 'Suhu',
        
        'cv_header': 'Wawasan CrowdVision',
        'cv_explanation': 'Gunakake CrowdVision kanggo ndeleng lingkungan sampeyan aman apa ora.',
        'cv_browse_link': 'Klik kanggo nggoleki',
        'cv_drag_drop': 'utawa seret foto mrene',
        'cv_file_type': 'PNG, JPG, utawa JPEG (Maks 5MB)',
        'cv_button_analyze': 'Analisis Foto',
        'cv_button_analyze_count': 'Analisis {count} Foto',
        'api_title_analyzing': 'Nganalisis...',
        
        'haze_insight_header': 'Wawasan Kabut',
        'haze_insight_text': 'Simulasi prediksi asep.',
        'haze_insight_button': 'Mulai Simulasi',
        'footer_mission_header': 'Misi Kita',
        'footer_mission_text': 'HazeRadar nyedhiyakake prediksi akurat kanggo kualitas hawa ing Jawa Kulon nggunakake teknologi GNN.',
        'footer_back_to_map': 'Bali menyang Peta',
        'modal_close': 'Tutup',
        'ews_title': 'Sistem Pelingeling Dini',
        'ews_desc': 'Oleh notifikasi nalika kualitas hawa elek.',
        'ews_name_label': 'Jeneng Lengkap',
        'ews_name_placeholder': 'Conto: Slamet',
        'ews_phone_label': 'Nomer WhatsApp',
        'ews_phone_placeholder': 'Conto: 08123456789',
        'ews_submit_btn': 'Aktifake',
        'ews_success_msg': 'Matur nuwun, {name}! Wis kadhaptar.',
        'ews_tooltip_text': '👋 Apa sampeyan pengin entuk notifikasi?',
    }
};

// --- GLOBAL VARIABLES ---
let map;
let currentTileLayer;
let gnnMarkerLayer = null;
let currentPredictions = []; 
let currentForecastCache = {}; 
let currentSelectedCity = null;

const SLIDER_TO_HOURS = {
    1: 0, 2: 12, 3: 24, 4: 36, 5: 48, 6: 60
};

const TIME_LABELS = {
    1: "time_now", 2: "time_12hr", 3: "time_24hr", 
    4: "time_36hr", 5: "time_48hr", 6: "time_60hr"
};

// --- INIT ---
const html = document.documentElement;

document.addEventListener('DOMContentLoaded', function () {
    loadTheme();
    loadLanguage();
    initializeMap();
    setupEventListeners();
    
    // Load Data Awal
    loadCurrentPredictions();
});

function initializeMap() {
    const mapContainer = document.getElementById('map');
    if (!mapContainer) return;

    map = L.map('map').setView(MAP_CENTER, MAP_ZOOM);
    updateMapTheme();
}

function updateMapTheme() {
    if (!map) return;
    const isDarkMode = html.classList.contains('dark');
    const url = isDarkMode 
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
    
    if (currentTileLayer) map.removeLayer(currentTileLayer);
    currentTileLayer = L.tileLayer(url, {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        maxZoom: 19
    }).addTo(map);
}

// --- API FETCHING LOGIC ---

// 1. Load Data "Current" (Coba Railway dulu, kalau gagal -> Supabase)
async function loadCurrentPredictions() {
    showLoading(true);
    try {
        console.log("Fetching from Railway Backend...");
        const response = await fetch(`${HAZE_API_BASE_URL}/api/predictions/current`);
        
        if (!response.ok) {
            throw new Error(`Railway Error: ${response.status}`);
        }
        
        const data = await response.json();
        processData(data);

    } catch (error) {
        console.error("Backend Railway Error:", error);
        console.log("Switching to Supabase Fallback...");
        await loadFallbackFromSupabase();
    } finally {
        showLoading(false);
    }
}


// Helper untuk memproses data agar tampil
function processData(data) {
    // Filter hanya kota resmi jika perlu
    const filtered = data.filter(d => 
        OFFICIAL_CITIES.some(city => d.city.toLowerCase().includes(city.toLowerCase())) || 
        OFFICIAL_CITIES.includes(d.city)
    );
    
    currentPredictions = filtered.length > 0 ? filtered : data;
    
    renderMapMarkers(currentPredictions);
    renderSidebarList(currentPredictions);
    
    // Select kota pertama default
    if (!currentSelectedCity && currentPredictions.length > 0) {
        selectCity(currentPredictions[0]);
    }
}

// --- FUNGSI LOAD GEOJSON (PETA WILAYAH) ---
async function loadGeoJson() {
    try {
        // Memanggil file yang ada di root folder kamu
        const response = await fetch('kota-kabupaten-jawa.json');

        if (!response.ok) {
            throw new Error(`Gagal memuat file JSON! Status: ${response.status}`);
        }

        const data = await response.json();

        // Membuat layer peta dari data JSON
        geoJsonLayer = L.geoJSON(data, {
            style: defaultStyle,       // Pastikan variabel defaultStyle ada di atas
            onEachFeature: onEachFeature // Pastikan fungsi onEachFeature ada di atas
        }).addTo(map);

        // Zoom otomatis ke area Jawa
        if (map) {
            map.fitBounds(geoJsonLayer.getBounds());
            
            // Batasi agar user tidak geser kejauhan (Optional)
            // map.setMaxBounds(geoJsonLayer.getBounds().pad(0.5)); 
        }

        // Sembunyikan loading overlay (spinner) karena peta sudah muncul
        const overlay = document.getElementById('message-overlay');
        if (overlay) overlay.style.display = 'none';

        console.log("Sukses memuat Peta Jawa!");

    } catch (error) {
        console.error('Error loading GeoJSON:', error);
        // Tampilkan pesan error di layar jika gagal
        const msgText = document.getElementById('message-text');
        if (msgText) msgText.innerText = "Gagal memuat peta wilayah. Pastikan file json ada.";
    }
}
// 2. Load Forecast (Dari Backend)
async function loadForecastForCity(cityName) {
    if (currentForecastCache[cityName]) return currentForecastCache[cityName];

    try {
        const response = await fetch(`${HAZE_API_BASE_URL}/api/forecast/${encodeURIComponent(cityName)}`);
        if (!response.ok) throw new Error('Forecast API Error');
        
        const data = await response.json();
        currentForecastCache[cityName] = data; 
        return data;
    } catch (error) {
        console.error(`Error loading forecast for ${cityName}:`, error);
        return null;
    }
}

// --- SUPABASE FALLBACK LOGIC (Cadangan) ---
async function loadFallbackFromSupabase() {
    const promises = OFFICIAL_CITIES.map(city => fetchSingleCityPredictionFromSupabase(city));
    const results = await Promise.all(promises);
    const data = results.filter(r => r !== null);
    
    if (data.length > 0) {
        processData(data);
    } else {
        document.getElementById('metric-current-haze').innerText = "Offline";
    }
}

async function fetchSingleCityPredictionFromSupabase(cityName) {
    if (!cityName) return null;
    const trimmed = cityName.trim();

    // ERROR FIX: Menggunakan 'created_at' bukan 'timestamp'
    const params = new URLSearchParams({
        select: '*',
        city: `ilike.*${trimmed}*`,
        order: `created_at.desc`,  // <--- INI PERBAIKANNYA
        limit: '1'
    });

    const url = `${SUPABASE_URL}/rest/v1/gnn_training_data?${params.toString()}`;

    try {
        const resp = await fetch(url, {
            method: 'GET',
            headers: {
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
                'Content-Type': 'application/json'
            },
        });

        if (!resp.ok) return null;

        const rows = await resp.json();
        if (!Array.isArray(rows) || !rows.length) return null;

        const row = rows[0];
        // Normalize object keys to match frontend expectation
        return {
            city: row.city || cityName,
            pm25: row.target_pm25_24h,
            aqi: row.current_aqi,
            status: estimateStatus(row.current_aqi),
            temperature: row.temperature,
            latitude: row.latitude,
            longitude: row.longitude,
            created_at: row.created_at // Konsisten pakai created_at
        };
    } catch (err) {
        console.error('Supabase Fallback Failed:', err);
        return null;
    }
}

// Helper status AQI jika data raw
function estimateStatus(aqi) {
    if (!aqi) return 'Unknown';
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Moderate';
    if (aqi <= 150) return 'Unhealthy (Sensitive)';
    if (aqi <= 200) return 'Unhealthy';
    if (aqi <= 300) return 'Very Unhealthy';
    return 'Hazardous';
}

// --- RENDERING UI ---

function renderMapMarkers(predictions) {
    if (!map) return;
    if (!gnnMarkerLayer) gnnMarkerLayer = L.layerGroup().addTo(map);
    gnnMarkerLayer.clearLayers();

    predictions.forEach(pred => {
        if (!pred.latitude || !pred.longitude) return;

        const color = getStatusColor(pred.status);
        
        const marker = L.circleMarker([pred.latitude, pred.longitude], {
            radius: 10 + ((pred.pm25 || 10) / 10),
            color: '#fff',
            weight: 1,
            fillColor: color,
            fillOpacity: 0.8
        });

        const popupContent = `
            <div class="popup-header border-b pb-2 mb-2 font-bold">${pred.city}</div>
            <ul class="popup-metric-list text-sm">
                <li class="flex justify-between"><span>AQI:</span> <b>${pred.aqi}</b></li>
                <li class="flex justify-between"><span>PM2.5:</span> <b>${pred.pm25} µg/m³</b></li>
                <li class="flex justify-between"><span>Status:</span> <span style="color:${color}">${pred.status}</span></li>
                <li class="flex justify-between"><span>Temp:</span> <b>${pred.temperature}°C</b></li>
            </ul>
        `;

        marker.bindPopup(popupContent);
        marker.on('click', () => selectCity(pred));
        marker.addTo(gnnMarkerLayer);
    });
}

function renderSidebarList(predictions) {
    const listContainer = document.getElementById('affected-areas-list');
    const emptyMsg = document.getElementById('no-cards-message');
    
    if (!listContainer) return;
    listContainer.innerHTML = '';

    if (!predictions || predictions.length === 0) {
        if(emptyMsg) emptyMsg.classList.remove('hidden');
        return;
    }
    if(emptyMsg) emptyMsg.classList.add('hidden');

    predictions.forEach(pred => {
        const colorClass = getStatusTextColorClass(pred.status);
        
        const card = document.createElement('div');
        card.className = "card-item group hover:shadow-lg p-3 mb-2 rounded cursor-pointer bg-white dark:bg-gray-700 flex justify-between items-center transition-all";
        card.innerHTML = `
            <div>
                <p class="font-semibold text-gray-800 dark:text-gray-100">${pred.city}</p>
                <p class="text-xs text-gray-500">Jawa Barat</p>
            </div>
            <div class="text-right">
                <p class="text-sm font-bold ${colorClass}">${pred.status}</p>
                <p class="text-xs text-gray-500">AQI: ${pred.aqi}</p>
            </div>
        `;
        
        card.addEventListener('click', () => {
            selectCity(pred, true);
        });
        
        listContainer.appendChild(card);
    });
}

function selectCity(data, shouldZoom = true) {
    currentSelectedCity = data;
    updateMetricsUI(data.aqi, data.pm25, data.status || data.category, data.temperature);

    if (shouldZoom && map && data.latitude && data.longitude) {
        map.flyTo([data.latitude, data.longitude], 10, { duration: 1.2 });
    }

    const slider = document.getElementById('notch-slider');
    if (slider && slider.value != 1) {
        slider.value = 1;
        updateSliderLabel(1);
    }
    loadForecastForCity(data.city);
}

function updateMetricsUI(aqi, pm25, status, temp) {
    const statusEl = document.getElementById('metric-current-haze');
    const aqiEl = document.getElementById('metric-predicted-aqi');
    const pmEl = document.getElementById('metric-wind-speed'); 
    const tempEl = document.getElementById('metric-visibility'); 

    if (statusEl) {
        statusEl.textContent = status || '-';
        statusEl.className = `metric-value ${getStatusTextColorClass(status)}`;
    }
    if (aqiEl) aqiEl.textContent = aqi ?? '-';
    if (pmEl) pmEl.textContent = pm25 ? `${pm25} µg/m³` : '-';
    if (tempEl) tempEl.textContent = temp ? `${temp}°C` : '-';
}

// --- SLIDER & UI HELPERS ---

async function handleSliderChange(e) {
    const val = parseInt(e.target.value);
    const hour = SLIDER_TO_HOURS[val];
    updateSliderLabel(val);
    
    if (val === 1) {
        if (currentSelectedCity) {
            const fresh = currentPredictions.find(p => p.city === currentSelectedCity.city) || currentSelectedCity;
            updateMetricsUI(fresh.aqi, fresh.pm25, fresh.status, fresh.temperature);
        }
        return;
    }

    if (!currentSelectedCity) return;
    document.getElementById('metric-current-haze').textContent = "...";
    
    let forecastList = await loadForecastForCity(currentSelectedCity.city);
    if (forecastList) {
        const point = forecastList.find(p => p.hour === hour);
        if (point) {
            updateMetricsUI(point.aqi, point.pm25, point.category, point.temperature);
        } else {
            document.getElementById('metric-current-haze').textContent = "No Data";
        }
    }
}

function updateSliderLabel(val) {
    const labelKey = TIME_LABELS[val];
    const labelEl = document.getElementById('time-label');
    if (labelEl) labelEl.textContent = t(labelKey);
}

function getStatusColor(status) {
    if (!status) return '#9ca3af'; 
    const s = status.toLowerCase();
    if (s.includes('good')) return '#10b981'; 
    if (s.includes('moderate')) return '#f59e0b'; 
    if (s.includes('sensitive')) return '#f97316'; 
    if (s.includes('very')) return '#7c3aed'; 
    if (s.includes('hazard')) return '#7f1d1d'; 
    if (s.includes('unhealthy')) return '#ef4444'; 
    return '#9ca3af';
}

function getStatusTextColorClass(status) {
    if (!status) return 'text-gray-500';
    const s = status.toLowerCase();
    if (s.includes('good')) return 'text-teal';
    if (s.includes('moderate')) return 'text-yellow';
    if (s.includes('sensitive')) return 'text-orange-500';
    if (s.includes('very')) return 'text-purple';
    if (s.includes('hazard')) return 'text-red-900';
    if (s.includes('unhealthy')) return 'text-red';
    return 'text-gray-500';
}

function showLoading(show) {
    const spinner = document.getElementById('loading-spinner');
    const overlay = document.getElementById('message-overlay');
    if (spinner && overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

// --- SETUP LISTENERS ---
function setupEventListeners() {
    const slider = document.getElementById('notch-slider');
    if (slider) {
        slider.addEventListener('input', (e) => updateSliderLabel(e.target.value));
        slider.addEventListener('change', handleSliderChange);
    }
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
    const langLinks = document.querySelectorAll('[data-lang]');
    langLinks.forEach(l => l.addEventListener('click', (e) => {
        e.preventDefault();
        setLanguage(e.target.getAttribute('data-lang'));
    }));
    const uploadBtn = document.getElementById('upload-button');
    if(uploadBtn) uploadBtn.addEventListener('click', handleUpload);
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');
    if(dropArea && fileElem) {
        dropArea.addEventListener('click', () => fileElem.click());
        fileElem.addEventListener('change', (e) => handleFiles(e.target.files));
    }
    setupEWS();
}

// --- CROWDVISION ---
let uploadedFiles = [];
function handleFiles(files) {
    uploadedFiles = Array.from(files);
    renderFileList();
}
function renderFileList() {
    const list = document.getElementById('file-list');
    const btn = document.getElementById('upload-button');
    if(!list) return;
    list.innerHTML = '';
    uploadedFiles.forEach(f => {
        list.innerHTML += `<div class="text-sm p-1 border-b">${f.name}</div>`;
    });
    if(btn) {
        btn.disabled = uploadedFiles.length === 0;
        btn.textContent = t('cv_button_analyze');
    }
}
async function handleUpload() {
    if(uploadedFiles.length === 0) return;
    const btn = document.getElementById('upload-button');
    const resultBox = document.getElementById('result');
    btn.textContent = t('api_title_analyzing');
    btn.disabled = true;
    const formData = new FormData();
    formData.append('image', uploadedFiles[0]);
    try {
        const response = await fetch("https://haze-radarcrowdvision-production.up.railway.app/predict", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if(resultBox) {
            resultBox.classList.remove('hidden');
            resultBox.innerHTML = `
                <div class="font-bold text-green-600">Prediction: ${data.prediction}</div>
                <div class="text-xs text-gray-500">Confidence: ${data.confidence}%</div>
            `;
        }
    } catch (e) {
        console.error(e);
        if(resultBox) {
            resultBox.classList.remove('hidden');
            resultBox.innerHTML = `<div class="text-red-500">Analysis failed.</div>`;
        }
    } finally {
        btn.textContent = t('cv_button_analyze');
        btn.disabled = false;
        uploadedFiles = [];
        renderFileList();
    }
}

// --- LANGUAGE & THEME ---
let currentLang = 'en';
function t(key, params = {}) {
    let str = translations[currentLang][key] || key;
    for (let k in params) str = str.replace(`{${k}}`, params[k]);
    return str;
}
function loadLanguage() {
    const saved = localStorage.getItem('lang');
    if (saved && translations[saved]) currentLang = saved;
    applyLanguage();
}
function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('lang', lang);
    applyLanguage();
}
function applyLanguage() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        el.textContent = t(el.getAttribute('data-i18n'));
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    const slider = document.getElementById('notch-slider');
    if(slider) updateSliderLabel(slider.value);
}
function loadTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        html.classList.add('dark');
        updateIcons(true);
    } else {
        html.classList.remove('dark');
        updateIcons(false);
    }
}
function toggleTheme() {
    html.classList.toggle('dark');
    const isDark = html.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateMapTheme();
    updateIcons(isDark);
}
function updateIcons(isDark) {
    const moon = document.getElementById('theme-icon-moon');
    const sun = document.getElementById('theme-icon-sun');
    if(moon && sun) {
        if(isDark) { moon.classList.remove('hidden'); sun.classList.add('hidden'); }
        else { moon.classList.add('hidden'); sun.classList.remove('hidden'); }
    }
}
function setupEWS() {
    const fab = document.getElementById('ews-fab-btn');
    const modal = document.getElementById('ews-modal');
    const closeBtn = document.getElementById('ews-modal-close-x');
    const form = document.getElementById('ews-form');
    if(fab && modal) {
        fab.addEventListener('click', () => {
            modal.classList.remove('invisible', 'opacity-0');
            modal.classList.add('visible', 'opacity-100');
        });
    }
    const closeModal = () => {
        if(modal) {
            modal.classList.remove('visible', 'opacity-100');
            modal.classList.add('invisible', 'opacity-0');
        }
    };
    if(closeBtn) closeBtn.addEventListener('click', closeModal);
    if(form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            alert(t('ews_success_msg', {name: document.getElementById('ews-name').value}));
            closeModal();
            form.reset();
        });
    }
}