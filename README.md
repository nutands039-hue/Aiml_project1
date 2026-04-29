# AI-Based Resume Analyzer & Skill Recommendation System

## 🎓 College Project - Python ML Implementation

A complete AI-powered resume analysis system with both **Python Flask backend** and **client-side JavaScript** implementation. Works on GitHub Pages!

---

## ✨ Features

### Two Ways to Use:
1. **📝 Manual Form Input** - Enter details directly in the website
2. **📄 CSV File Upload** - Upload a CSV file with resume data

### ML-Powered Analysis:
- **TF-IDF Vectorization** - Text feature extraction
- **Naive Bayes Classification** - Job category prediction
- **Cosine Similarity** - Match score calculation
- **k-NN Recommendations** - Skill suggestions

---

## 📊 Project Details

### 1) Problem Statement
Traditional resume screening is time-consuming and prone to human bias. Our AI-based system automatically analyzes resumes, extracts key skills, matches candidates with job requirements, and provides intelligent skill recommendations to improve employability.

### 2) Type of Problem
**Classification Problem** - We classify resumes into job categories and predict skill compatibility using supervised learning techniques.

### 3) Algorithms Used
- **TF-IDF (Term Frequency-Inverse Document Frequency)** - For text vectorization and feature extraction
- **Naive Bayes Classifier** - For job category classification
- **Cosine Similarity** - For resume-job matching score calculation
- **K-Nearest Neighbors (k-NN)** - For skill recommendation based on similar profiles

### 4) Why These Algorithms?
- **TF-IDF:** Efficiently converts text to numerical features, handles word importance
- **Naive Bayes:** Fast, works well with text classification, requires less training data
- **Cosine Similarity:** Measures document similarity effectively in high-dimensional space
- **k-NN:** Simple, effective for recommendation systems based on similar profiles

### 5) Data Used
- **Source:** Public resume datasets (Kaggle, UCI Repository)
- **Features:**
  - Skills (programming languages, tools, technologies)
  - Experience years
  - Educational qualifications
  - Project descriptions
- **Preprocessing:** Text cleaning, stopword removal, feature extraction

### 6) Model Building Approach
- **Training:** Dataset split (80% training, 20% testing), cross-validation
- **Testing:** Accuracy, precision, recall, F1-score metrics evaluation

---

## 🚀 Installation & Setup

### Option 1: Python/Flask (Local Testing)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the Flask server
python app.py

# 3. Open in browser
http://127.0.0.1:5000
```

### Option 2: GitHub Pages (Static Deployment)

```bash
# Just upload index.html to GitHub Pages
# No server needed - works entirely in browser!
```

---

## 📄 CSV Format

Upload a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| name | Full Name | John Doe |
| skills | Comma-separated skills | Python, Machine Learning, SQL |
| experience | Years of experience | 3 |
| education | Highest qualification | Bachelor's in Computer Science |
| projects | Project descriptions (optional) | ML Chatbot, Web Scraper |

### Example CSV:
```csv
name,skills,experience,education,projects
John Doe,Python Machine Learning SQL,3,Bachelor's in CS,ML Chatbot
Jane Smith,Java Spring Boot REST APIs,5,Master's in SE,E-commerce Platform
```

---

## 📁 Project Structure

```
resume-analyzer/
├── index.html          # Main website (works on GitHub Pages)
├── app.py              # Python Flask backend
├── requirements.txt    # Python dependencies
├── sample_resume.csv   # Sample CSV file
├── README.md           # This file
└── .gitignore          # Git ignore file
```

---

## 🎯 How It Works

1. **Data Input:** User enters details or uploads CSV
2. **Text Preprocessing:** System cleans and normalizes text
3. **Feature Extraction:** TF-IDF converts text to numerical features
4. **Classification:** Naive Bayes predicts job category
5. **Similarity Matching:** Cosine similarity calculates match score
6. **Recommendation:** k-NN suggests skills based on similar profiles
7. **Results:** Display analysis and recommendations

---

## 🔬 Python Implementation Details

### TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf.fit_transform(resume_texts)
```

### Naive Bayes Classifier
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_tfidf, job_categories)
prediction = model.predict(new_resume_vector)
```

### Cosine Similarity
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(resume_vector, job_profile_vector)
```

---

## 🌐 Deployment Options

### GitHub Pages (Recommended for College Demo)
1. Create a GitHub repository
2. Upload `index.html`
3. Go to Settings → Pages
4. Select main branch and save
5. Your site is live!

### Heroku / Python Anywhere
1. Deploy `app.py` with Flask
2. Add `requirements.txt`
3. Configure environment variables
4. Deploy and test

---

## 📊 Results Display

The system shows:
- ✅ **Candidate Information** - Name, experience, education
- ✅ **Match Score** - Percentage match with ideal profile
- ✅ **Extracted Skills** - Skills detected from input
- ✅ **Recommended Skills** - Skills to learn for career growth
- ✅ **Job Roles** - Suggested job positions
- ✅ **Quality Score** - Overall resume quality (0-100)

---

## 🎨 Technologies Used

| Component | Technology |
|-----------|-----------|
| Frontend | HTML5, CSS3, JavaScript (ES6+) |
| Backend (Optional) | Python 3, Flask |
| ML Libraries | scikit-learn, pandas, numpy |
| Algorithms | TF-IDF, Naive Bayes, k-NN, Cosine Similarity |
| Deployment | GitHub Pages, Flask Server |

---

## 📈 Future Enhancements

- PDF resume upload support
- NLP-based skill extraction (spaCy)
- Job market API integration
- Real-time chatbot guidance
- Database for storing analyses
- Export results as PDF

---

✅ **Explains ML concepts clearly**  
✅ **Shows practical implementation**  
✅ **Demonstrates full ML pipeline**  
✅ **Easy to present and explain**  
✅ **Works without internet/server**  
✅ **Impressive visual results**  

---

