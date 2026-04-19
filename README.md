# 🚀 Job Portfolio Evaluation using Gen AI + ML

An AI-powered Resume + GitHub + Portfolio Analyzer that evaluates resumes against job descriptions using Machine Learning, NLP, GitHub profile analysis, Portfolio scraping, and Ollama LLM feedback.

---

# 🔥 Features

✅ ATS Resume Score Calculation
✅ Resume vs Job Description Matching
✅ GitHub Profile Evaluation
✅ Portfolio Website Analysis
✅ LinkedIn PDF Parsing
✅ AI Suggestions for Improvement
✅ Resume Rewrite Generator
✅ Skill Gap Radar Chart
✅ User Login & Saved History using MongoDB
✅ Interactive Dashboard with Charts

---

# 🛠️ Tech Stack

* Python
* Streamlit
* MongoDB
* Ollama (Phi3)
* Sentence Transformers
* Scikit-Learn
* SpaCy
* Plotly

---

# 📥 Complete Installation Guide (For New Users)

If you are downloading this project for the first time, follow all steps below.

---

## 1️⃣ Download Project

### Option A: Clone Using Git


git clone https://github.com/Kaushal2306/Job-Portfolio-Evaluator.git
cd Job-Portfolio-Evaluator


### Option B: Download ZIP

* Open GitHub repository
* Click **Code → Download ZIP**
* Extract folder
* Open terminal inside project folder

---

## 2️⃣ Install Python

Download Python 3.11+ from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

⚠️ While installing, tick:


Add Python to PATH


Verify installation:


python --version


---

## 3️⃣ Create Virtual Environment


python -m venv venv


Activate it:

### Windows


venv\Scripts\activate


### Mac / Linux


source venv/bin/activate


---

## 4️⃣ Install Project Libraries


pip install -r requirements.txt


---

## 5️⃣ Install SpaCy Language Model


python -m spacy download en_core_web_sm


---

# 🧠 Install Ollama (Required for AI Feedback)

This project uses a local LLM through Ollama.

## 6️⃣ Download Ollama

Install from:

[https://ollama.com/download](https://ollama.com/download)

After installation, verify:


ollama --version


---

## 7️⃣ Download Phi3 Model


ollama pull phi3

This may take a few minutes.

---

# 🗄️ Install MongoDB (Required for Login + History)

## 8️⃣ Download MongoDB Community Server

Install from:

[https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)

Choose:

* Community Server
* Default Setup

---

## 9️⃣ Start MongoDB

After installation:

### Windows

Open Services and start:


MongoDB Server


OR use terminal:


mongod


Default database used:


mongodb://localhost:27017/


---

# ▶️ Run Project

## 🔟 Start Streamlit App

Inside project folder:


streamlit run app.py


Browser opens automatically.

If not:


http://localhost:8501


---

# 👩‍💻 Team Usage

Each teammate must do once:


git clone YOUR_REPO_URL
cd Job-Portfolio-Evaluator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull phi3
streamlit run app.py


---

# 📁 Project Structure


Job-Portfolio-Evaluator/
│── app.py
│── requirements.txt
│── README.md


---

# 🛠 Troubleshooting
## Python Not Found
Reinstall Python and enable:
Add Python to PATH
## Ollama Not Working
Restart system after installing Ollama.

## MongoDB Connection Error
Ensure MongoDB service is running.

## Streamlit Not Found
Run:
pip install streamlit

---

# 🤝 Contributors

* A.Kaushal
* Dharshini S
* Mounish Raja

---

# ⭐ Future Improvements

* Cloud Deployment
* GPT API Integration
* Better UI Design
* Resume Templates
* Email Report Generation
* Admin Dashboard

---

# 📌 Notes

This project runs mostly on your local machine:

* MongoDB stores users/history
* Ollama runs local AI model
* Streamlit runs frontend dashboard

No paid API required.
