from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)  


@app.route('/api/geocode', methods=['POST'])
def geocode():
    data = request.get_json()
    lat = data.get('latitude')
    lng = data.get('longitude')

    if not lat or not lng:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    geocode_data = response.json()

    if geocode_data['status'] != 'OK':
        return jsonify({'error': 'Geocoding failed', 'details': geocode_data['status']}), 500

    address = geocode_data['results'][0]['formatted_address']
    return jsonify({
        'latitude': lat,
        'longitude': lng,
        'address': address
    })

if __name__ == '__main__':
    app.run(debug=True)

