from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
import os
import google.gemini as gemini  # Placeholder for actual Gemini API import

app = Flask(__name__)

# Initialize Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages])
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return " ".join([para.text for para in doc.paragraphs])

def get_summary(text):
    # Placeholder for Gemini API call
    return "Summary: " + text[:200] + "..."  # Mock implementation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'resume' not in request.files:
        return "No file uploaded", 400
        
    file = request.files['resume']
    job_desc = request.form.get('job_desc', '')
    
    if file.filename == '':
        return "No selected file", 400
        
    if not job_desc:
        return "Job description is required", 400
        
    # Save file temporarily
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    # Extract text based on file type
    if file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(filepath)
    elif file.filename.endswith('.docx'):
        resume_text = extract_text_from_docx(filepath)
    else:
        return "Unsupported file type", 400
        
    # Generate embeddings
    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(job_desc)
    
    # Calculate similarity
    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    score = round(similarity * 100, 2)
    
    # Generate summary (mock implementation)
    summary = get_summary(resume_text)
    
    # Clean up
    os.remove(filepath)
    
    return render_template('results.html', 
                         score=score,
                         summary=summary,
                         top_skills=["Python", "Machine Learning", "NLP", "Flask", "Data Analysis"])

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)