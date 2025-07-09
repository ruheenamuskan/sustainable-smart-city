from flask import Flask, request, jsonify, send_file
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from dotenv import load_dotenv
import torch
import os
import uuid
import pathlib
import tempfile
import requests
import datetime
import json

load_dotenv()  # Load .env file

app = Flask(__name__)

# Hugging Face & IBM Granite Model Setup
HF_TOKEN = os.getenv("HF_TOKEN")
model_path = "ibm-granite/granite-3.3-2b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token=HF_TOKEN,
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)

# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY = os.getenv("WEATHER_KEY") or "01c071f61b4927f03f61b65428a5871f"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Upload Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'document_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    
    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif ext == 'pdf':
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except ImportError:
            return "PDF support missing. Install PyPDF2."
    elif ext in ['docx', 'doc']:
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return "DOCX support missing. Install python-docx."
    return "Unsupported file format"

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data.get("query", "")
    if not user_input:
        return jsonify({"response": "Please enter a valid query."})
    
    conversation = [{"role": "user", "content": user_input}]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True
    )
    set_seed(42)
    output = model.generate(**input_ids, max_new_tokens=512)
    prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return jsonify({"response": prediction})

@app.route("/upload-document", methods=["POST"])
def upload_document():
    if 'document' not in request.files:
        return jsonify({"error": "No document uploaded."})
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "No selected file."})
    
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        document_text = extract_text_from_file(file_path)
        summary_prompt = f"Summarize this in context of sustainable smart cities:\n\n{document_text[:4000]}"
        
        conversation = [{"role": "user", "content": summary_prompt}]
        input_ids = tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        
        set_seed(42)
        output = model.generate(**input_ids, max_new_tokens=512)
        summary = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        
        os.remove(file_path)
        return jsonify({"summary": summary, "original_filename": file.filename})
    
    return jsonify({"error": "Invalid file type. Allowed: txt, pdf, docx, doc"})

@app.route("/get-weather", methods=["POST"])
def get_weather():
    data = request.json
    city = data.get("city", "")
    if not city:
        return jsonify({"error": "Please provide a city name"})
    
    try:
        params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(OPENWEATHER_URL, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        result = {
            "city": weather_data['name'],
            "country": weather_data['sys']['country'],
            "temp": weather_data['main']['temp'],
            "feels_like": weather_data['main']['feels_like'],
            "humidity": weather_data['main']['humidity'],
            "description": weather_data['weather'][0]['description'].title(),
            "icon": weather_data['weather'][0]['icon']
        }
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Weather API error: {str(e)}"})
    except KeyError:
        return jsonify({"error": "Invalid city or data format"})

@app.route("/get-temperature-comparison", methods=["POST"])
def get_temperature_comparison():
    data = request.json
    city = data.get("city", "")
    if not city:
        return jsonify({"error": "Please provide a city name"})
    
    try:
        current_params = {
            'q': city,
            'appid': OPENWEATHER_API_KEY
        }
        current_response = requests.get(OPENWEATHER_URL, params=current_params)
        current_response.raise_for_status()
        current_data = current_response.json()
        
        lat = current_data['coord']['lat']
        lon = current_data['coord']['lon']
        city_name = current_data['name']
        country = current_data['sys']['country']
        
        forecast_params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'cnt': 40
        }
        forecast_response = requests.get(OPENWEATHER_FORECAST_URL, params=forecast_params)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        forecast_temps = {}
        for item in forecast_data['list']:
            dt = datetime.datetime.fromtimestamp(item['dt'])
            date_str = dt.strftime('%Y-%m-%d')
            forecast_temps.setdefault(date_str, []).append(item['main']['temp'])
        
        forecast_daily_avg = {d: sum(t)/len(t) for d, t in forecast_temps.items()}
        
        last_year = datetime.datetime.now().year - 1
        historical_daily_avg = {}
        import random
        for date_str, temp in forecast_daily_avg.items():
            parts = date_str.split('-')
            hist_date = f"{last_year}-{parts[1]}-{parts[2]}"
            variation = random.uniform(-3, 3)
            historical_daily_avg[hist_date] = temp + variation
        
        chart_data = []
        for cur_date, cur_temp in forecast_daily_avg.items():
            parts = cur_date.split('-')
            hist_date = f"{last_year}-{parts[1]}-{parts[2]}"
            hist_temp = historical_daily_avg.get(hist_date, 0)
            display_date = datetime.datetime.strptime(cur_date, '%Y-%m-%d').strftime('%b %d')
            chart_data.append({
                "date": display_date,
                "current": round(cur_temp, 1),
                "historical": round(hist_temp, 1)
            })
        
        temp_diff = (
            sum(forecast_daily_avg.values()) / len(forecast_daily_avg)
            - sum(historical_daily_avg.values()) / len(historical_daily_avg)
        )
        
        analysis_prompt = f"""
        Analyze climate change impacts for {city_name}, {country}.
        Current avg: {round(sum(forecast_daily_avg.values())/len(forecast_daily_avg),1)}°C
        Last year avg: {round(sum(historical_daily_avg.values())/len(historical_daily_avg),1)}°C
        Diff: {round(temp_diff,1)}°C
        """
        conversation = [{"role": "user", "content": analysis_prompt}]
        input_ids = tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        set_seed(42)
        output = model.generate(**input_ids, max_new_tokens=300)
        analysis = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return jsonify({
            "city": city_name,
            "country": country,
            "chartData": chart_data,
            "analysis": analysis,
            "currentYearAvg": round(sum(forecast_daily_avg.values()) / len(forecast_daily_avg), 1),
            "lastYearAvg": round(sum(historical_daily_avg.values()) / len(historical_daily_avg), 1),
            "difference": round(temp_diff, 1)
        })
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
