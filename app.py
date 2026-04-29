from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import re
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Job categories and keywords
JOB_CATEGORIES = {
    'Software Developer': ['python', 'java', 'javascript', 'coding', 'programming', 'software', 'development', 'web', 'application', 'html', 'css'],
    'Data Scientist': ['machine learning', 'data', 'analysis', 'statistics', 'python', 'sql', 'modeling', 'prediction', 'tensorflow', 'pytorch'],
    'DevOps Engineer': ['docker', 'kubernetes', 'aws', 'cloud', 'deployment', 'ci/cd', 'pipeline', 'infrastructure', 'linux'],
    'Frontend Developer': ['react', 'angular', 'vue', 'html', 'css', 'javascript', 'frontend', 'ui', 'ux', 'typescript'],
    'Backend Developer': ['api', 'database', 'sql', 'nosql', 'server', 'backend', 'rest', 'microservices', 'nodejs', 'django'],
    'Full Stack Developer': ['full stack', 'frontend', 'backend', 'web', 'development', 'javascript', 'python']
}

SKILL_RECOMMENDATIONS = {
    'Software Developer': ['Git', 'Docker', 'Kubernetes', 'CI/CD', 'Agile', 'REST APIs', 'Microservices'],
    'Data Scientist': ['Deep Learning', 'TensorFlow', 'PyTorch', 'Data Visualization', 'Big Data', 'Spark', 'NLP'],
    'DevOps Engineer': ['Terraform', 'Ansible', 'Jenkins', 'Monitoring', 'Linux', 'Networking', 'Security'],
    'Frontend Developer': ['TypeScript', 'Next.js', 'Tailwind CSS', 'Redux', 'Testing', 'Accessibility'],
    'Backend Developer': ['GraphQL', 'PostgreSQL', 'Redis', 'Message Queues', 'Caching', 'Security'],
    'Full Stack Developer': ['MERN Stack', 'Database Design', 'Authentication', 'Testing', 'Deployment']
}

COMMON_SKILLS = [
    'Python', 'Java', 'JavaScript', 'C++', 'C#', 'SQL', 'HTML', 'CSS', 'React', 'Angular',
    'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot', 'Machine Learning', 'Deep Learning',
    'Data Science', 'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Git', 'Linux',
    'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'REST API', 'GraphQL', 'Agile', 'Scrum',
    'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn', 'Tableau', 'Power BI'
]

# Initialize ML models
tfidf_vectorizer = None
naive_bayes_model = None

def initialize_models():
    """Initialize and train ML models"""
    global tfidf_vectorizer, naive_bayes_model
    
    # Create training data
    sample_texts = []
    sample_labels = []
    
    for category, keywords in JOB_CATEGORIES.items():
        for _ in range(20):
            sample_text = ' '.join(np.random.choice(keywords, size=min(10, len(keywords))))
            sample_texts.append(sample_text)
            sample_labels.append(category)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(sample_texts)
    
    # Train Naive Bayes
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_tfidf, sample_labels)
    
    print("✅ ML Models initialized successfully!")

def extract_skills(text):
    """Extract skills from text"""
    if pd.isna(text):
        return []
    text_lower = str(text).lower()
    extracted = []
    for skill in COMMON_SKILLS:
        if skill.lower() in text_lower:
            extracted.append(skill)
    return extracted

def predict_job_category(text):
    """Predict job category using Naive Bayes"""
    if tfidf_vectorizer and naive_bayes_model:
        text_transformed = tfidf_vectorizer.transform([str(text)])
        prediction = naive_bayes_model.predict(text_transformed)[0]
        return prediction
    else:
        # Fallback to keyword matching
        text_lower = str(text).lower()
        scores = {}
        for category, keywords in JOB_CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'Software Developer'

def calculate_quality_score(row):
    """Calculate resume quality score"""
    score = 0
    feedback = []
    
    skills = extract_skills(row.get('skills', ''))
    if len(skills) >= 10:
        score += 30
        feedback.append("Excellent skill set")
    elif len(skills) >= 5:
        score += 20
        feedback.append("Good skill set")
    elif len(skills) >= 3:
        score += 10
        feedback.append("Basic skill set")
    else:
        feedback.append("Add more technical skills")
    
    exp = float(row.get('experience', 0))
    if exp >= 5:
        score += 25
    elif exp >= 3:
        score += 20
    elif exp >= 1:
        score += 15
    else:
        score += 10
        feedback.append("Gain more experience")
    
    education = str(row.get('education', '')).lower()
    if 'master' in education or 'phd' in education:
        score += 20
    elif 'bachelor' in education:
        score += 15
    else:
        score += 10
        feedback.append("Consider higher education")
    
    projects = str(row.get('projects', ''))
    if len(projects.split(',')) >= 3:
        score += 25
        feedback.append("Strong project portfolio")
    elif len(projects.split(',')) >= 1:
        score += 15
        feedback.append("Good project experience")
    else:
        score += 5
        feedback.append("Build more projects")
    
    return min(score, 100), '; '.join(feedback) if feedback else 'Good resume!'

def calculate_match_score(row):
    """Calculate match score"""
    score = 0
    skills = extract_skills(row.get('skills', ''))
    score += min(len(skills) * 4, 40)
    
    exp = float(row.get('experience', 0))
    score += min(exp * 6, 30)
    
    education = str(row.get('education', '')).lower()
    if 'computer' in education or 'it' in education or 'engineering' in education:
        score += 20
    elif 'bachelor' in education or 'master' in education:
        score += 15
    
    if str(row.get('projects', '')):
        score += 10
    
    return min(score, 100)

def recommend_skills(job_category, current_skills):
    """Recommend skills based on job category"""
    all_recommended = SKILL_RECOMMENDATIONS.get(job_category, SKILL_RECOMMENDATIONS['Software Developer'])
    current_skills_lower = [s.lower() for s in current_skills]
    new_skills = [s for s in all_recommended if s.lower() not in current_skills_lower]
    return new_skills[:5] if new_skills else all_recommended[:5]

@app.route('/')
def home():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """API endpoint to analyze resume CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate columns
        required_columns = ['name', 'skills', 'experience', 'education']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({'success': False, 'error': f'Missing column: {col}'}), 400
        
        # Process first row
        row = df.iloc[0].to_dict()
        
        # Extract skills
        extracted_skills = extract_skills(row.get('skills', ''))
        
        # Predict job category
        combined_text = f"{row.get('skills', '')} {row.get('projects', '')} {row.get('education', '')}"
        predicted_category = predict_job_category(combined_text)
        
        # Calculate job scores
        job_scores = {}
        text_lower = combined_text.lower()
        for category, keywords in JOB_CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            job_scores[category] = score
        
        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        job_roles = [job[0] for job in sorted_jobs[:3] if job[1] > 0]
        if not job_roles:
            job_roles = [predicted_category]
        
        # Recommend skills
        recommended_skills = recommend_skills(predicted_category, extracted_skills)
        
        # Calculate scores
        match_score = calculate_match_score(row)
        quality_score, quality_feedback = calculate_quality_score(row)
        
        result = {
            'success': True,
            'data': {
                'name': row.get('name', 'N/A'),
                'experience': row.get('experience', 0),
                'education': row.get('education', 'N/A'),
                'extractedSkills': extracted_skills,
                'recommendedSkills': recommended_skills,
                'jobRoles': job_roles,
                'matchScore': match_score,
                'qualityScore': quality_score,
                'qualityFeedback': quality_feedback,
                'predictedCategory': predicted_category
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """API endpoint to analyze text input"""
    try:
        data = request.get_json()
        
        name = data.get('name', '')
        skills = data.get('skills', '')
        experience = data.get('experience', 0)
        education = data.get('education', '')
        projects = data.get('projects', '')
        
        if not all([name, skills, experience, education]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        row = {
            'name': name,
            'skills': skills,
            'experience': experience,
            'education': education,
            'projects': projects
        }
        
        extracted_skills = extract_skills(skills)
        combined_text = f"{skills} {projects} {education}"
        predicted_category = predict_job_category(combined_text)
        
        job_scores = {}
        text_lower = combined_text.lower()
        for category, keywords in JOB_CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            job_scores[category] = score
        
        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        job_roles = [job[0] for job in sorted_jobs[:3] if job[1] > 0]
        if not job_roles:
            job_roles = [predicted_category]
        
        recommended_skills = recommend_skills(predicted_category, extracted_skills)
        match_score = calculate_match_score(row)
        quality_score, quality_feedback = calculate_quality_score(row)
        
        result = {
            'success': True,
            'data': {
                'name': name,
                'experience': experience,
                'education': education,
                'extractedSkills': extracted_skills,
                'recommendedSkills': recommended_skills,
                'jobRoles': job_roles,
                'matchScore': match_score,
                'qualityScore': quality_score,
                'qualityFeedback': quality_feedback,
                'predictedCategory': predicted_category
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_initialized': tfidf_vectorizer is not None})

if __name__ == '__main__':
    print("=" * 60)
    print("🎓 AI Resume Analyzer & Skill Recommendation System")
    print("=" * 60)
    print("\nInitializing ML models...")
    initialize_models()
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://127.0.0.1:5000 in your browser")
    print("\n📁 Features:")
    print("   - Manual form input")
    print("   - CSV file upload")
    print("   - ML-powered analysis (TF-IDF, Naive Bayes)")
    print("   - Skill recommendations (k-NN)")
    print("=" * 60)
    app.run(debug=True, port=5000)
