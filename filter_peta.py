import json

# Proses ulang filter agar link download muncul fresh
input_file = 'kota-kabupaten.json'
output_file = 'kota-kabupaten-jawa.json'

java_keywords = ["JAKARTA", "JAWA BARAT", "BANTEN", "JAWA TENGAH", "YOGYAKARTA", "JAWA TIMUR"]

def is_java_province(prov_name):
    if not prov_name: return False
    return any(k in prov_name.upper() for k in java_keywords)

try:
    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered_features = [f for f in data['features'] if is_java_province(f.get('properties', {}).get('NAME_1', ''))]
    data['features'] = filtered_features
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
    print("File siap diunduh.")

except Exception as e:
    print(e)