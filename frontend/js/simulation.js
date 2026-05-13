// ------------------------------------------
// --- MAP AND CONSTANTS ---
// ------------------------------------------

// Endpoint to fetch the list of city coordinates 
const API_ROOT = "https://haze-radargnnmodelstimulation-production.up.railway.app";
const map = L.map("map").setView([-1.5, 113], 5);

// Use a dark tile layer for better visualization contrast
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18,
    noWrap: true,
    bounds: mapBounds,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
});

// Option B: Satellite View (Esri Imagery + Labels)
const satelliteBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
    maxZoom: 18,
    noWrap: true,
    bounds: mapBounds
});

const satelliteLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18,
    noWrap: true,
    bounds: mapBounds,
    pane: 'overlayPane'
});

// Group Satellite + Labels together so they toggle as one
const satelliteGroup = L.layerGroup([satelliteBase, satelliteLabels]);


// --- 2. Initialize Map with Default Layer ---

const map = L.map("map", {
    maxBounds: mapBounds,
    maxBoundsViscosity: 1.0,
    minZoom: 2,
    maxZoom: 18,
    layers: [satelliteGroup] // Start with Satellite by default
}).setView([-1.5, 113], 5);


// --- 3. Add the Toggle Control (Top-Right) ---

const baseMaps = {
    "Satellite View": satelliteGroup,
    "Dark Mode": darkMapLayer
};

L.control.layers(baseMaps, null, { position: 'topright' }).addTo(map);


// Removed cityCoordinates/cityMarkers as we are now using dynamic search
let hazeLayers = [];
let predictions = [];
let autoplayInterval = null;

// The simulation will now rely entirely on these "Custom" coordinates
// whether they come from a Search or a Click.
let customStartLat = null;
let customStartLon = null;
let customStartMarker = null;
let customSourceName = "Unknown Location"; // Store the name of the searched place
let hourlyWindForecast = [];

// --- Simulation Constants ---
const INITIAL_INTENSITY = 100; 
const HAZE_DECAY = 0.98;
const DEGREE_PER_KMH = 0.00005; 
const MAX_HOURS = 120; 
const DEG_PER_METER = 1 / 111320;
const MIN_RADIUS_M = 200;
const MAX_RADIUS_M = 1000;
const NUM_SATELLITES = 8;
const SATELLITE_RADIUS_RATIO = 2.0;

// --- PM2.5 Realistic Logic Constants ---
const BASE_PM25 = 1000;          // Starting PM2.5 at source
const PM25_NATURAL_DECAY = 0.99; // Baseline decay per hour (settling)
const PM25_WIND_DISPERSION_COEFF = 0.005; // Impact of wind speed on dilution

function toast(msg) { document.getElementById("statusBox").innerText = msg; }

function interpolateColor(color1, color2, factor) {
    function hexToRgb(hex) {
        const bigint = parseInt(hex.slice(1), 16);
        return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
    }
    function rgbToHex(r, g, b) {
        return '#' + [r, g, b].map(x => {
            const hex = Math.round(x).toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        }).join('');
    }
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);
    const r = rgb1[0] + (rgb2[0] - rgb1[0]) * factor;
    const g = rgb1[1] + (rgb2[1] - rgb1[1]) * factor;
    const b = rgb1[2] + (rgb2[2] - rgb1[2]) * factor;
    return rgbToHex(r, g, b);
}

// --------------------------------------------------------------------------
// --- COLOR MAPPING: Using Generic Intensity Value for Future PM2.5 Hook ---
// --------------------------------------------------------------------------

/**
 * Maps generic intensity value to color, interpolating between fixed hazard levels.
 * * NOTE: This function currently uses a synthetic 'intensityValue' derived from the 
 * simulation's decay rate. In the future, 'intensityValue' should be replaced by 
 * the actual PM2.5 data fetched from an API or model result.
 */
function colorForIntensity(intensityValue) {
    // Renamed PM constants to generic Intensity constants
    const INTENSITY_MAX = 150;
    const INTENSITY_RED_THRESHOLD = 120;
    const INTENSITY_ORANGE_THRESHOLD = 80;
    const INTENSITY_YELLOW_THRESHOLD = 40;

    const COLOR_RED = "#FF0000";
    const COLOR_ORANGE = "#FF8C00";
    const COLOR_YELLOW = "#FFFF00";
    const COLOR_GREY = "#808080";

    if (intensityValue >= INTENSITY_RED_THRESHOLD) {
        return COLOR_RED;
    } else if (intensityValue >= INTENSITY_ORANGE_THRESHOLD) {
        const factor = (INTENSITY_RED_THRESHOLD - intensityValue) / (INTENSITY_RED_THRESHOLD - INTENSITY_ORANGE_THRESHOLD);
        return interpolateColor(COLOR_RED, COLOR_ORANGE, factor);
    } else if (intensityValue >= INTENSITY_YELLOW_THRESHOLD) {
        const factor = (INTENSITY_ORANGE_THRESHOLD - intensityValue) / (INTENSITY_ORANGE_THRESHOLD - INTENSITY_YELLOW_THRESHOLD);
        return interpolateColor(COLOR_ORANGE, COLOR_YELLOW, factor);
    } else {
        const clampedIntensity = Math.max(5, intensityValue);
        const factor = (INTENSITY_YELLOW_THRESHOLD - clampedIntensity) / (INTENSITY_YELLOW_THRESHOLD - 5);
        return interpolateColor(COLOR_YELLOW, COLOR_GREY, factor);
    }
}

function radiusForHour(h) {
    const factor = (h - 1) / (MAX_HOURS - 1);
    return MIN_RADIUS_M + factor * (MAX_RADIUS_M - MIN_RADIUS_M);
}

function clearHaze() {
    hazeLayers.forEach(l => map.removeLayer(l));
    hazeLayers = [];
}

// Clears the single simulation marker
function clearCustomMarker() {
    if (customStartMarker) {
        map.removeLayer(customStartMarker);
        customStartMarker = null;
        customStartLat = null;
        customStartLon = null;
    }
}

// ----------------------------------------------------
// --- Single Wind Forecast Fetch (Open-Meteo Logic) ---
// ----------------------------------------------------

/**
 * Fetches the 5-day (120 hour) wind forecast for the selected location.
 */
async function fetch24HourWindForecast(lat, lon, sourceName) {
    toast(`Fetching wind forecast for ${sourceName}...`);
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=windspeed_10m,winddirection_10m&forecast_days=5`;

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`API error ${response.status}`);
        const data = await response.json();

        const speeds = data.hourly.windspeed_10m.slice(0, MAX_HOURS);
        const directions = data.hourly.winddirection_10m.slice(0, MAX_HOURS);

        hourlyWindForecast = speeds.map((speed, index) => ({
            hour: index + 1,
            speed: speed,
            direction: directions[index]
        }));

            // Update display with the new full forecast
            let html = `<strong>5-Day (120 Hour) Wind Forecast (${sourceName} - Open-Meteo):</strong><ul class="list-unstyled text-xs p-1 m-0">`;
            hourlyWindForecast.forEach(w => {
                const speed = typeof w.speed === 'number' ? w.speed.toFixed(1) : w.speed;
                html += `<li class="d-flex justify-content-between"><span>Hour ${w.hour.toString().padStart(3, '0')}:</span> <span>${speed} km/h @ ${w.direction}°</span></li>`;
            });
            html += '</ul>';
            document.getElementById("windForecastDisplay").innerHTML = html;

            toast(`5-day wind forecast successfully loaded for ${sourceName}.`);
            return true;

        } catch (error) {
            if (attempt < MAX_RETRIES - 1) {
                await new Promise(resolve => setTimeout(resolve, DELAY_MS * Math.pow(2, attempt)));
            } else {
                console.error("Wind Forecast API Error:", error);
                document.getElementById("windForecastDisplay").innerHTML = `
                        <p class="text-danger text-center mt-2">Error: Could not fetch 5-day wind forecast from Open-Meteo API after ${MAX_RETRIES} attempts. Check console.</p>
                    `;
                toast(`API Error: ${error.message}`);
                hourlyWindForecast = [];
                return false;
            }
        }
    }
}


// ----------------------------------------------------------
// --- CORE LOGIC: Local Plume Spread (Dynamic Visuals) ---
// ----------------------------------------------------------

/**
 * Simulates the movement of 1 core plume and 8 randomized satellite plumes over 120 hours.
 */
function simulateLocalHazeSpread(centerLat, centerLon, initialPmValue) {
    console.log(`Starting local simulation from Lat: ${centerLat}, Lon: ${centerLon} with initial PM: ${initialPmValue}`);

    const results = [];
    let currentPlumeCenterLat = centerLat;
    let currentPlumeCenterLon = centerLon;
    let currentIntensity = initialIntensity;
    
    // Initialize cumulative PM2.5 tracking
    let currentPm25 = BASE_PM25;

    for (let h = 1; h <= MAX_HOURS; h++) {
        if (h > hourlyWindForecast.length) break;

        const wind = hourlyWindForecast[h - 1];
        const windRad = wind.direction * (Math.PI / 180);
        const hourlyWindShift = wind.speed * DEGREE_PER_KMH;

        currentPlumeCenterLat += hourlyWindShift * Math.cos(windRad);
        currentPlumeCenterLon += hourlyWindShift * Math.sin(windRad);

        // 1. Intensity decay
        currentIntensity *= HAZE_DECAY;
        let baseIntensity = Math.max(5, currentIntensity);

        // 2. Cumulative PM2.5 Calculation Logic
        const windImpact = 1 / (1 + (wind.speed * PM25_WIND_DISPERSION_COEFF));
        currentPm25 = currentPm25 * PM25_NATURAL_DECAY * windImpact;
        
        // Clamp min PM2.5 to 1.0 (typical baseline air quality)
        const finalPm25 = Math.max(1.0, currentPm25);

        // Core Plume (Unified data)
        results.push({
            is_core: true,
            latitude: currentPlumeCenterLat,
            longitude: currentPlumeCenterLon,
            intensity: baseIntensity,
            pm25: finalPm25,
            forecast_hour: h
        });

        const currentVisualRadiusM = radiusForHour(h);
        const maxScatterDistanceDeg = (currentVisualRadiusM * SATELLITE_RADIUS_RATIO) * DEG_PER_METER;

        for (let i = 0; i < NUM_SATELLITES; i++) {
            const randomAngle = Math.random() * 2 * Math.PI;
            const randomDistDeg = Math.random() * maxScatterDistanceDeg;
            const satelliteLat = currentPlumeCenterLat + randomDistDeg * Math.cos(randomAngle);
            const satelliteLon = currentPlumeCenterLon + randomDistDeg * Math.sin(randomAngle);

            // Use exact same intensity and PM2.5 for satellites to ensure consistency
            results.push({
                is_core: false,
                latitude: satelliteLat,
                longitude: satelliteLon,
                intensity: baseIntensity,
                pm25: finalPm25,
                forecast_hour: h
            });
        }
    }
    return results;
}

// --- NEW SEARCH LOGIC WITH AUTOCOMPLETE ---

/**
 * Loads cities from the API, populates the dropdown, and draws the initial city dots.
 */
async function loadCities() {
    try {
        toast("Loading cities...");
        const r = await fetch(API_ROOT + "/api/cities");

        if (!r.ok) {
            toast(`Error loading cities: Status ${r.status}`);
            throw new Error(`HTTP error! status: ${r.status}`);
        }

        const d = await r.json();

        const dropdown = document.getElementById("cityDropdown");

        // Store the currently loaded cities, keeping the "Select a City" and "Custom" options
        const selectOptions = dropdown.querySelector('option[value=""]').outerHTML;
        const customOption = dropdown.querySelector('option[value="custom"]').outerHTML;
        dropdown.innerHTML = selectOptions;

        if (!d.cities || d.cities.length === 0) {
            toast("Error: Could not load cities from API.");
            dropdown.innerHTML += customOption; // Still allow custom input
            return;
        }

        clearMarkers();
        clearHaze();

        d.cities.forEach(c => {
            cityCoordinates[c.city] = c;

            const opt = document.createElement("option");
            opt.value = c.city;
            opt.innerText = c.city;
            dropdown.appendChild(opt);

            // Create city marker on the map
            const marker = L.circleMarker([c.latitude, c.longitude], {
                radius: 8,
                color: "#06b6d4",
                fillColor: "#22d3ee",
                fillOpacity: 0.8,
                weight: 1,
                // IMPORTANT: Explicitly disable interaction to prevent hover effects and clicking
                interactive: false
            }).addTo(map);

            // Marker dot no longer needs popup if non-interactive, but keeping bindPopup for completeness
            marker.bindPopup(`<b>${c.city}</b><br>Preset Haze Source Point.`);
            cityMarkers.push(marker);
        });

        dropdown.innerHTML += customOption; // Add custom option back in

        toast(`Cities loaded. Select a city, or choose 'Custom Location' to click the map.`);

    } catch (error) {
        console.error("Error loading cities:", error);
        toast(`Error loading cities. Please check the network connection. (${error.message})`);
    }
}

/**
 * Runs the simulation: fetches the forecast and initiates the haze spread.
 */
async function runSimulation() {
    let lat, lon, sourceName;
    const selectedValue = document.getElementById("cityDropdown").value;

    if (selectedValue === "custom") {
        // Case 1: Custom map location
        if (customStartLat === null || customStartLon === null) {
            toast("Error: Please click on the map to set the custom starting point first.");
            return;
        }
        lat = customStartLat;
        lon = customStartLon;
        sourceName = `Custom (${lat.toFixed(2)}, ${lon.toFixed(2)})`;

    } else if (selectedValue && cityCoordinates[selectedValue]) {
        // Case 2: Preset City selected
        const c = cityCoordinates[selectedValue];
        lat = c.latitude;
        lon = c.longitude;
        sourceName = selectedValue;
    } else {
        // Case 3: Nothing selected
        toast("Error: Please select a starting city or a custom location.");
        return;
    }

    // Clear previous animation state
    if (autoplayInterval) {
        clearInterval(autoplayInterval);
        autoplayInterval = null;
    }
    clearHaze();

    // Close the custom marker popup before simulation starts
    if (customStartMarker && customStartMarker.getPopup() && customStartMarker.isPopupOpen()) {
        customStartMarker.closePopup();
    }

    // --- PM FETCH LOGIC ---
    let initialPm25 = INITIAL_PM25;

    // Fetch real current PM from backend
    try {
        const pmResp = await fetch(`${API_ROOT}/predict?city=${sourceName}`);

        // Added check for response success to prevent parsing errors on 404/500
        if (pmResp.ok) {
            const pmData = await pmResp.json();
            if (pmData && typeof pmData.predicted_pm25 === 'number') {
                // This is the value that will be used by the simulation's decay model
                initialPm25 = parseFloat(pmData.predicted_pm25);
                console.log(`Initial PM data fetched and used: ${initialPm25.toFixed(2)} µg/m³.`);
            } else {
                console.warn(`PM data fetch successful but 'predicted_pm25' not found or invalid. Using default ${INITIAL_PM25}.`);
            }
        } else {
            console.warn(`Could not fetch initial PM data from backend. Status: ${pmResp.status}. Using default ${INITIAL_PM25}.`);
        }
    } catch (error) {
        // This catches network errors (like server being down)
        console.error("Network error during PM fetch:", error);
        console.warn(`PM fetch failed due to network error. Using default ${INITIAL_PM25}.`);
    }
    // --- END PM FETCH LOGIC ---


    // 1. Fetch the 5-day wind forecast using Geo coordinates
    const fetchSuccessful = await fetch24HourWindForecast(lat, lon, sourceName);
    if (!fetchSuccessful) return;

    toast(`Running 5-day haze simulation from ${sourceName} with 120-hour forecast... 💨`);

    // --- EXECUTE LOCAL MODEL ---
    // Pass the fetched/default PM value to the simulation function
    predictions = simulateLocalHazeSpread(lat, lon, initialPm25);
    // ---------------------------

    // Fly in on the selected location with zoom 12
    map.flyTo([lat, lon], 12, { animate: true, duration: 1.5 });

    if (predictions.length > 0) {
        toast("Simulation complete. Playing haze spread... 🌬️");
        playAnimation();
    } else {
        toast("Local simulation failed to generate data.");
    }
}

/**
 * Starts the 120-hour animation loop (like a video player).
 */
function playAnimation() {
    clearHaze();
    let hour = 1;
    document.getElementById("sliderPanel").style.display = "flex"; // Show controls

    if (autoplayInterval) {
        clearInterval(autoplayInterval);
    }

    showHour(hour);
    hour++;

    autoplayInterval = setInterval(() => {
        showHour(hour);
        hour++;
        if (hour > MAX_HOURS) {
            clearInterval(autoplayInterval);
            autoplayInterval = null;
            toast("Simulation playback finished.");
        }
    }, 100);
}

function showHour(h) {
    clearHaze();
    document.getElementById("hourSlider").value = h;

    // DYNAMIC HOUR INDICATOR: Convert total hours to Day/Hour format
    const day = Math.floor((h - 1) / 24) + 1;
    const hourInDay = (h % 24 === 0) ? 24 : h % 24;
    document.getElementById("hourLabel").innerText = `Day ${day}, Hour ${hourInDay} (Total: ${h})`;

    const isCumulative = document.getElementById("viewModeToggle").checked;

    const hourData = predictions.filter(p =>
        isCumulative ? p.forecast_hour <= h : p.forecast_hour === h
    );

    hourData.forEach(p => {
        const intensityValue = p.predicted_pm25;
        const col = colorForIntensity(intensityValue); // Color now based on intensity
        const dynamicRadius = radiusForHour(p.forecast_hour);
        const wind = hourlyWindForecast[p.forecast_hour - 1];
        const speed = wind ? wind.speed.toFixed(1) : 'N/A';

        const ring = L.circle([p.latitude, p.longitude], {
            radius: dynamicRadius, color: col, fillColor: col, fillOpacity: 0.125, weight: 0
        }).addTo(map);

        // Add popup with details for inspection
        ring.bindPopup(`
                <b>Hour ${p.forecast_hour} Prediction</b><br>
                <span style="color: #e63946;"><b>PM2.5:</b> ${p.pm25.toFixed(1)} µg/m³</span><br>
                <span style="color: #457b9d;"><b>Wind:</b> ${speed} km/h</span><br>
                <span style="color: #666; font-size: 0.8em;">Intensity Score: ${p.intensity.toFixed(2)}</span>
            </div>
        `, { sticky: true, opacity: 0.9 });

        hazeLayers.push(ring);
    });
}

// --- Event Handlers ---

// Handler for the "Run Haze Simulation" button
document.getElementById("simulateLocal").onclick = runSimulation;

// Handler for City Dropdown selection
document.getElementById("cityDropdown").addEventListener("change", async e => {
    const selectedValue = e.target.value;
    const c = cityCoordinates[selectedValue];

    // Clear previous state
    if (autoplayInterval) { clearInterval(autoplayInterval); autoplayInterval = null; }
    clearHaze();
    hourlyWindForecast = [];

    if (selectedValue === "custom") {
        // Enable map clicks and show instructions
        isMapClickEnabled = true;
        document.getElementById("customInstructions").style.display = 'block';
        toast("Mode: Custom Location. Click on the map to set a point.");
    } else {
        // Disable map clicks and clear custom marker
        isMapClickEnabled = false;
        document.getElementById("customInstructions").style.display = 'none';
        clearCustomMarker();

        if (c) {
            // Zoom to 12 when selecting from dropdown 
            map.flyTo([c.latitude, c.longitude], 12, { duration: 1.0 });
            document.getElementById("windForecastDisplay").innerHTML = `
                    <p class="text-center text-light-override">Selected ${selectedValue}. Click 'Run Simulation' to fetch forecast.</p>
                `;
            toast(`City changed to ${selectedValue}. Ready to run simulation.`);
        } else {
            document.getElementById("windForecastDisplay").innerHTML = `<p class="text-center text-light-override">Select a city or custom location, then press 'Run Haze Simulation'.</p>`;
            toast("Ready.");
        }
    }
});

// Handler for Map Click (Custom Source Point) - NOW GUARDED BY isMapClickEnabled
map.on('click', (e) => {
    if (!isMapClickEnabled) {
        toast("Select 'Set Custom Location on Map' from the dropdown to enable map clicking.");
        return;
    }
    // Set point
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    // 1. Clear previous custom marker
    clearCustomMarker();

    // 2. Store coordinates
    customStartLat = lat;
    customStartLon = lon;

    // 3. Add new custom marker
    customStartMarker = L.marker([lat, lon], {
        icon: L.divIcon({
            className: 'custom-start-icon',
            html: `<div style="background-color: ${'var(--accent-color)'}; width: 15px; height: 15px; border-radius: 50%; border: 3px solid white;"></div>`,
            iconSize: [15, 15],
            iconAnchor: [7.5, 7.5]
        })
    }).addTo(map);

    // Bind the popup so the user can click the marker to see coordinates, but don't open it immediately
    customStartMarker.bindPopup(`<b>Custom Source Set</b><br>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`).openPopup();

    // Zoom to the new location
    map.flyTo([lat, lon], 12, { duration: 0.5 });

    // Update UI status
    document.getElementById("windForecastDisplay").innerHTML = `
            <p class="text-center text-light-override">Custom location set. Click 'Run Haze Simulation' to fetch forecast.</p>
        `;
    toast("Custom location set. Click 'Run Haze Simulation'.");
});

document.getElementById("viewModeToggle").addEventListener("change", e => {
    const isChecked = e.target.checked;
    document.getElementById("toggleLabel").innerText = isChecked ? "Cumulative View" : "Hourly View";

    // Re-render the current hour based on the new view mode
    const currentHour = parseInt(document.getElementById("hourSlider").value);
    if (predictions.length > 0) {
        showHour(currentHour);
    }

    // Stop autoplay if running
    if (autoplayInterval) {
        clearInterval(autoplayInterval);
        autoplayInterval = null;
        toast(`Autoplay paused. Switched to ${isChecked ? 'Cumulative' : 'Hourly'} View.`);
    }
});


// Handle slider interaction (stops autoplay)
document.getElementById("hourSlider").addEventListener("input", e => {
    if (autoplayInterval) {
        clearInterval(autoplayInterval);
        autoplayInterval = null;
        toast("Autoplay paused. Inspecting manually.");
    }
    const hour = parseInt(e.target.value);
    if (predictions.length > 0) {
        showHour(hour);
    } else {
        document.getElementById("hourLabel").innerText = "Hour: " + hour;
    }
});

// Replay button handler
document.getElementById("replayBtn").onclick = () => {
    if (predictions.length > 0) playAnimation();
};

// --- Initialization ---
window.onload = loadCities;
