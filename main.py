from flask import Flask, request, jsonify
import PyPDF2
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import requests
import io
from airtable import airtable

app = Flask(__name__)

# Configuration de l'API OpenAI
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)



class CVFit(BaseModel):
    score: float = Field(description="Score donné au CV en fonction de sa correspondance avec la description du poste (entre 0 et 1)")
    justify: str = Field(description="Justification du score atribué au CV")

def download_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        raise Exception("Failed to download PDF")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_cv(job_description, cv_text):
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Vous êtes un expert RH chargé d'évaluer la correspondance entre un CV et une description de poste."},
                {"role": "user", "content": f"""Description du poste : {job_description}\n\nCV : {cv_text}\n\n
                Évaluez la correspondance entre ce CV et la description du poste. 
                Donnez le résultat avec le format suivant : 
                - Score : Score donné au CV en fonction de sa correspondance avec la description du poste (entre 0 et 1)
                - Justify :  Justification du score atribué au CV
                """}
            ], 
            response_format=CVFit
        )
        return response.choices[0].message.parsed
    except Exception as e:
        return None

@app.route('/Analyse', methods=['POST'])
def analyse():
    data = request.get_json()
    
    if not data or 'job_description' not in data or 'cv_url' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    job_description = data['job_description']
    cv_url = data['cv_url']
    
    try:
        # Download and process the CV
        pdf_file = download_pdf_from_url(cv_url)
        cv_text = extract_text_from_pdf(pdf_file)
        
        # Analyze the CV
        result = analyze_cv(job_description, cv_text)
        
        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500
            
        return jsonify({
            'score': result.score,
            'justification': result.justify
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the CV Analyzer API!"

@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()
    return jsonify({'message': 'Test successful', 'data': data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
