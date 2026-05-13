import { API_ROOT } from './config.js';

// --- FUNGSI 1: AMBIL DATA KOTA ---
export async function fetchCities() {
    try {
        // Ini memanggil: .../api/cities
        const response = await fetch(`${API_ROOT}/cities`);

        if (!response.ok) throw new Error("Gagal mengambil data kota");
        const data = await response.json();

        // Backend Python kamu mengembalikan format: { cities: [...] }
        return data.cities || [];
    } catch (error) {
        console.error("API Error (fetchCities):", error);
        return [];
    }
}

// --- FUNGSI 2: PREDIKSI (UBAH TOTAL DISINI) ---
// --- FUNGSI 2: PREDIKSI (DENGAN PERBAIKAN FORMAT DATA) ---
export async function fetchPrediction(lat, lon) {
    try {
        const url = `${API_ROOT}/simulate`;

        console.log(`Mengirim simulasi ke: ${url}`);

        const requestBody = {
            fire_zone_coords: [
                {
                    latitude: parseFloat(lat),
                    longitude: parseFloat(lon)
                }
            ],
            simulation_hours: 72,
            radius_km: 100.0
        };

        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Server Error ${response.status}: ${errText}`);
        }

        const data = await response.json();

        // 🔥 PERBAIKAN UTAMA DISINI: Normalisasi Data
        // Python kirim "latitude", JS butuh "lat". Kita konversi.
        if (data.predictions && Array.isArray(data.predictions)) {
            data.predictions = data.predictions.map(p => ({
                ...p, // Salin data lain (timestamp, PM2.5, dll)

                // Ambil latitude (prioritas: latitude -> lat -> 0)
                lat: p.latitude !== undefined ? p.latitude : (p.lat !== undefined ? p.lat : 0),

                // Ambil longitude (prioritas: longitude -> lon -> 0)
                lon: p.longitude !== undefined ? p.longitude : (p.lon !== undefined ? p.lon : 0)
            }));
        }

        return data;

    } catch (error) {
        console.error("API Error (fetchPrediction):", error);
        alert(`Gagal simulasi: ${error.message}`);
        return null;
    }
}

// --- FUNGSI 3: CUACA (Tetap Sama) ---
export async function fetchWeather(lat, lon) {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=windspeed_10m,winddirection_10m&timezone=auto`;
    try {
        const res = await fetch(url);
        return await res.json();
    } catch (e) { return null; }
}
