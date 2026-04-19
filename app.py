import datetime
import hashlib
import io
import json
import re
import subprocess
import time
from urllib.parse import urlparse

import numpy as np
import ollama
import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import requests
import spacy
import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_ollama_running():
    try:
        requests.get("http://127.0.0.1:11434", timeout=2)
    except Exception:
        subprocess.Popen(["ollama", "serve"])
        time.sleep(3)


def init_session_state():
    defaults = {
        "logged_in": False,
        "user": None,
        "selected_report_id": None,
        "current_loaded_report_id": None,
        "local_users": {},
        "history_search": "",
        "form_name": "",
        "form_email": "",
        "form_github_url": "",
        "form_portfolio_url": "",
        "form_jd": "",
        "rewrite_output": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def reset_form_to_user_defaults():
    if st.session_state.user:
        st.session_state.form_name = st.session_state.user.get("name", "")
        st.session_state.form_email = st.session_state.user.get("email", "")
    else:
        st.session_state.form_name = ""
        st.session_state.form_email = ""
    st.session_state.form_github_url = ""
    st.session_state.form_portfolio_url = ""
    st.session_state.form_jd = ""
    st.session_state.current_loaded_report_id = None


def load_report_into_form(report):
    if not report:
        return
    st.session_state.form_name = report.get("candidate_name", "")
    st.session_state.form_email = report.get("candidate_email", "")
    st.session_state.form_github_url = report.get("github_url", "")
    st.session_state.form_portfolio_url = report.get("portfolio_url", "")
    st.session_state.form_jd = report.get("job_description", "")
    st.session_state.current_loaded_report_id = report.get("_id")


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.lower(), pages


def fetch_github_data(github_url: str) -> dict:
    result = {
        "username": "",
        "bio": "",
        "public_repos": 0,
        "followers": 0,
        "following": 0,
        "total_stars": 0,
        "top_languages": [],
        "repo_names": [],
        "repo_descriptions": [],
        "pinned_topics": [],
        "raw_text": "",
        "error": None,
    }
    try:
        parsed = urlparse(github_url)
        path_parts = [part for part in parsed.path.split("/") if part]
        if not path_parts:
            result["error"] = "Invalid GitHub URL"
            return result

        username = path_parts[0]
        result["username"] = username
        headers = {"Accept": "application/vnd.github+json"}

        user_resp = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=8)
        if user_resp.status_code != 200:
            result["error"] = f"GitHub API error: {user_resp.status_code}"
            return result

        user_data = user_resp.json()
        result["bio"] = user_data.get("bio", "") or ""
        result["public_repos"] = user_data.get("public_repos", 0)
        result["followers"] = user_data.get("followers", 0)
        result["following"] = user_data.get("following", 0)

        repos_resp = requests.get(
            f"https://api.github.com/users/{username}/repos?per_page=30&sort=stars",
            headers=headers,
            timeout=8,
        )
        if repos_resp.status_code == 200:
            repos = repos_resp.json()
            lang_counter = {}
            total_stars = 0
            for repo in repos:
                if repo.get("fork"):
                    continue
                total_stars += repo.get("stargazers_count", 0)
                result["repo_names"].append(repo.get("name", ""))
                description = repo.get("description", "") or ""
                result["repo_descriptions"].append(description)
                language = repo.get("language", "")
                if language:
                    lang_counter[language] = lang_counter.get(language, 0) + 1
                for topic in repo.get("topics", []):
                    if topic not in result["pinned_topics"]:
                        result["pinned_topics"].append(topic)
            result["total_stars"] = total_stars
            result["top_languages"] = sorted(lang_counter, key=lang_counter.get, reverse=True)[:6]

        result["raw_text"] = f"""
GitHub Profile: {username}
Bio: {result['bio']}
Public Repos: {result['public_repos']} | Followers: {result['followers']} | Total Stars: {result['total_stars']}
Top Languages: {', '.join(result['top_languages'])}
Topics/Skills: {', '.join(result['pinned_topics'])}
Recent Repos: {', '.join(result['repo_names'][:10])}
Repo Descriptions: {' | '.join([desc for desc in result['repo_descriptions'][:10] if desc])}
""".strip()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def fetch_portfolio_text(url: str) -> str:
    if not url:
        return ""
    try:
        if not url.startswith("http"):
            url = "https://" + url
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return f"Could not fetch portfolio (HTTP {response.status_code})"
        text = re.sub(r"<[^>]+>", " ", response.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000].lower()
    except Exception as exc:
        return f"Portfolio fetch error: {exc}"


def extract_experience_lines(text):
    lines = text.split("\n")
    matches = []
    for line in lines:
        if len(line) > 25 and re.search(
            r"\b(developed|built|designed|implemented|created|led|managed|optimized|trained|analyzed)\b",
            line,
        ):
            matches.append(line.strip())
    return matches[:40]


def extract_jd_lines(job_description):
    return [line.strip() for line in job_description.split("\n") if len(line.strip()) > 30][:40]


def parse_llm_json(output):
    try:
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1:
            return None
        clean = output[start : end + 1]
        clean = re.sub(r",\s*}", "}", clean)
        clean = re.sub(r",\s*]", "]", clean)
        return json.loads(clean)
    except Exception:
        return None


def keyword_match(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    return round((tfidf * tfidf.T).toarray()[0][1] * 100, 2)


def experience_similarity_score(resume_text, jd_text):
    resume_lines = extract_experience_lines(resume_text)
    jd_lines = extract_jd_lines(jd_text)
    if not resume_lines or not jd_lines:
        return 0
    resume_embeddings = embed_model.encode(resume_lines, normalize_embeddings=True)
    jd_embeddings = embed_model.encode(jd_lines, normalize_embeddings=True)
    similarity_matrix = np.matmul(resume_embeddings, jd_embeddings.T)
    return round(float(np.mean(similarity_matrix.max(axis=1))) * 100, 2)


def formatting_score(text, pages):
    score = 100
    if pages > 2:
        score -= 15
    if "education" not in text:
        score -= 10
    if "experience" not in text:
        score -= 20
    if len(text.split()) < 250:
        score -= 15
    return max(score, 0)


def impact_score(text):
    count = len(re.findall(r"\d+%|\d+x|\$\d+|\d+\s*(users|clients)", text))
    if count >= 8:
        return 100
    if count >= 4:
        return 70
    if count >= 1:
        return 40
    return 10


def role_match_score(resume_text, jd_text):
    roles = [
        "data scientist",
        "ml engineer",
        "software engineer",
        "ai engineer",
        "backend developer",
        "frontend developer",
        "full stack",
        "devops",
        "data engineer",
    ]
    return min(sum(20 for role in roles if role in resume_text and role in jd_text), 100)


def github_score(github_data: dict) -> float:
    if github_data.get("error") or not github_data.get("username"):
        return 0
    score = 0
    score += min(github_data["public_repos"] * 2, 30)
    score += min(github_data["total_stars"] * 3, 30)
    score += min(len(github_data["top_languages"]) * 5, 20)
    score += min(len(github_data["pinned_topics"]) * 2, 20)
    return round(min(score, 100), 2)


def portfolio_score(portfolio_text: str) -> float:
    if not portfolio_text or "error" in portfolio_text:
        return 0
    signals = ["project", "experience", "skill", "about", "work", "contact", "built", "developed", "deployed", "github"]
    return min(sum(10 for signal in signals if signal in portfolio_text), 100)


def final_score(resume_text, jd, pages, github_data, portfolio_text, linkedin_text):
    combined = resume_text + " " + linkedin_text
    keyword = keyword_match(combined, jd)
    context = experience_similarity_score(combined, jd)
    formatting = formatting_score(combined, pages)
    impact = impact_score(combined)
    role_match = role_match_score(combined, jd)
    github = github_score(github_data)
    portfolio = portfolio_score(portfolio_text)
    total = keyword * 0.22 + context * 0.28 + formatting * 0.15 + impact * 0.10 + role_match * 0.10 + github * 0.10 + portfolio * 0.05
    return round(total, 2), keyword, context, formatting, impact, role_match, github, portfolio

def llm_resume_feedback(resume_text, jd, score, name, github_data, portfolio_text, linkedin_text):
    github_summary = github_data.get("raw_text", "Not provided") if github_data else "Not provided"
    linkedin_snippet = linkedin_text[:600] if linkedin_text else "Not provided"
    portfolio_snippet = portfolio_text[:400] if portfolio_text else "Not provided"
    prompt = f"""
You are an expert technical recruiter and career advisor. Analyze the candidate holistically using all sources provided.

Return ONLY valid JSON in this exact format with no extra text:
{{
  "overall_match": "High/Medium/Low with one-line reason",
  "strengths": ["...", "..."],
  "missing_skills": ["...", "..."],
  "weak_areas": ["...", "..."],
  "quick_improvements": ["...", "..."],
  "suggestions": ["...", "..."],
  "github_feedback": "Paragraph on GitHub profile quality and relevance to JD",
  "portfolio_feedback": "Paragraph on portfolio website quality",
  "linkedin_feedback": "Paragraph on LinkedIn profile completeness",
  "sample_bullet": {{"before": "...", "after": "..."}},
  "skill_radar": {{
    "labels": ["Technical Skills", "Domain Knowledge", "Projects & Portfolio", "Communication", "Leadership", "Tools & Frameworks"],
    "candidate_scores": [70, 60, 50, 65, 40, 55],
    "jd_required": [85, 80, 70, 60, 50, 75]
  }},
  "action_plan": {{
    "30_days": ["...", "..."],
    "60_days": ["...", "..."],
    "90_days": ["...", "..."]
  }}
}}

Candidate: {name}
ATS Score: {score}/100

JOB DESCRIPTION:
{jd[:700]}

RESUME:
{resume_text[:700]}

LINKEDIN PROFILE:
{linkedin_snippet}

GITHUB PROFILE:
{github_summary}

PORTFOLIO WEBSITE:
{portfolio_snippet}
"""
    try:
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 1500},
        )
        return response["message"]["content"]
    except Exception:
        return "{}"


def generate_resume_rewrite(resume_text, jd, feedback):
    prompt = f"""
You are a resume optimization assistant.

Based on the resume, job description, and previous feedback, generate concise, stronger bullet-point rewrites.
Return ONLY valid JSON in this exact format:
{{
  "summary": "One short paragraph about the main rewrite approach",
  "rewrites": [
    {{"before": "...", "after": "...", "why": "..."}},
    {{"before": "...", "after": "...", "why": "..."}},
    {{"before": "...", "after": "...", "why": "..."}}
  ]
}}

RESUME:
{resume_text[:1500]}

JOB DESCRIPTION:
{jd[:1000]}

FEEDBACK:
{json.dumps(feedback)[:1200]}
"""
    try:
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 1200},
        )
        parsed = parse_llm_json(response["message"]["content"])
        if parsed:
            return parsed
    except Exception:
        pass

    return {
        "summary": "Focus the resume on quantified outcomes, stronger action verbs, and keywords that match the target role.",
        "rewrites": [
            {
                "before": "Worked on machine learning models for business problems",
                "after": "Built and improved machine learning models for business use cases, increasing prediction quality and supporting faster decision-making.",
                "why": "Uses stronger action verbs and highlights impact.",
            },
            {
                "before": "Created dashboards and reports for the team",
                "after": "Designed dashboards and automated reporting workflows that improved visibility into KPIs for stakeholders.",
                "why": "Makes the work sound more specific and outcome-oriented.",
            },
            {
                "before": "Collaborated with developers on project tasks",
                "after": "Partnered with cross-functional teams to deliver project milestones on time and align technical work with product goals.",
                "why": "Shows teamwork and delivery more clearly.",
            },
        ],
    }


def fallback_feedback():
    return {
        "overall_match": "Medium - profile partially matches the JD",
        "strengths": ["Resume uploaded and parsed successfully"],
        "missing_skills": ["Could not extract skills - ensure resume is text-based PDF"],
        "weak_areas": ["Limited quantified impact metrics found"],
        "quick_improvements": ["Add measurable achievements (%, $, users)"],
        "suggestions": ["Tailor resume keywords to JD", "Complete GitHub profile bio and pin repos"],
        "github_feedback": "GitHub profile could not be fully evaluated. Ensure URL is correct.",
        "portfolio_feedback": "Portfolio website could not be evaluated. Check if URL is public.",
        "linkedin_feedback": "LinkedIn PDF not uploaded or could not be parsed.",
        "sample_bullet": {
            "before": "Worked on machine learning models",
            "after": "Built and deployed ML classification model achieving 92% accuracy on 50K records",
        },
        "skill_radar": {
            "labels": ["Technical Skills", "Domain Knowledge", "Projects & Portfolio", "Communication", "Leadership", "Tools & Frameworks"],
            "candidate_scores": [50, 45, 40, 55, 35, 45],
            "jd_required": [85, 80, 70, 60, 50, 75],
        },
        "action_plan": {
            "30_days": [
                "Review and tailor resume to match JD keywords",
                "Set up or clean up GitHub profile with pinned repos",
                "Add quantifiable metrics to top 3 bullet points",
            ],
            "60_days": [
                "Build or polish a portfolio project aligned with JD requirements",
                "Complete an online course for a missing skill identified in the JD",
                "Optimize LinkedIn headline and summary section",
            ],
            "90_days": [
                "Apply to 10+ roles with tailored resumes",
                "Contribute to an open-source project relevant to the domain",
                "Request a mock interview or peer resume review",
            ],
        },
    }


def register_user(name: str, email: str, password: str):
    email = email.strip().lower()
    if not name.strip() or not email or not password:
        return False, "All fields are required."
    if users_col is not None:
        if users_col.find_one({"email": email}):
            return False, "An account with this email already exists."
        users_col.insert_one(
            {
                "name": name.strip(),
                "email": email,
                "password_hash": hash_password(password),
                "created_at": datetime.datetime.now(),
            }
        )
        return True, "Account created successfully. Please log in."
    if email in st.session_state.local_users:
        return False, "An account with this email already exists."
    st.session_state.local_users[email] = {
        "name": name.strip(),
        "email": email,
        "password_hash": hash_password(password),
    }
    return True, "Account created for this session. Please log in."


def authenticate_user(email: str, password: str):
    email = email.strip().lower()
    password_hash = hash_password(password)
    if users_col is not None:
        user = users_col.find_one({"email": email})
        if user and user.get("password_hash") == password_hash:
            return {"name": user.get("name", ""), "email": user.get("email", email)}
        return None
    user = st.session_state.local_users.get(email)
    if user and user.get("password_hash") == password_hash:
        return {"name": user.get("name", ""), "email": user.get("email", email)}
    return None


def get_user_history(user_email: str, search_text: str = ""):
    if reports_col is None:
        return []
    history = list(reports_col.find({"user_email": user_email}).sort("timestamp", -1).limit(50))
    query = search_text.strip().lower()
    if not query:
        return history
    filtered = []
    for item in history:
        haystack = " ".join(
            [
                str(item.get("candidate_name", "")),
                str(item.get("candidate_email", "")),
                str(item.get("report_title", "")),
                str(item.get("job_title", "")),
                str(item.get("github_url", "")),
                str(item.get("portfolio_url", "")),
                str(item.get("job_description", ""))[:500],
            ]
        ).lower()
        if query in haystack:
            filtered.append(item)
    return filtered


def get_report_by_id(report_id):
    if reports_col is None or not report_id:
        return None
    return reports_col.find_one({"_id": report_id})


def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.selected_report_id = None
    reset_form_to_user_defaults()


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_simple_pdf_bytes(title: str, lines):
    cleaned_lines = []
    for line in lines:
        line = line.replace("\t", " ").strip()
        if line:
            cleaned_lines.append(line[:110])
        else:
            cleaned_lines.append(" ")

    objects = []
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")

    content_lines = ["BT", "/F1 12 Tf", "50 760 Td", f"({pdf_escape(title)}) Tj"]
    vertical_step = 16
    for index, line in enumerate(cleaned_lines[:40], start=1):
        content_lines.append(f"0 -{vertical_step} Td")
        content_lines.append(f"({pdf_escape(line)}) Tj")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines)
    objects.append(f"<< /Length {len(content_stream.encode('utf-8'))} >>\nstream\n{content_stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = ["%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part.encode("utf-8")) for part in pdf))
        pdf.append(f"{index} 0 obj\n{obj}\nendobj\n")

    xref_offset = sum(len(part.encode("utf-8")) for part in pdf)
    pdf.append(f"xref\n0 {len(objects) + 1}\n")
    pdf.append("0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.append(f"{offset:010d} 00000 n \n")
    pdf.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF")
    return "".join(pdf).encode("utf-8")


def build_report_export_lines(report):
    feedback = report.get("feedback", {}) or {}
    breakdown = report.get("score_breakdown", {}) or {}
    lines = [
        f"Candidate: {report.get('candidate_name', '')}",
        f"Email: {report.get('candidate_email', '')}",
        f"Overall ATS Score: {report.get('ats_score', 0)}/100",
        f"Overall Match: {feedback.get('overall_match', '')}",
        " ",
        "Score Breakdown:",
        f"Keyword Match: {breakdown.get('keyword', 0)}",
        f"Context Similarity: {breakdown.get('context_similarity', 0)}",
        f"Formatting: {breakdown.get('formatting', 0)}",
        f"Impact: {breakdown.get('impact', 0)}",
        f"Role Match: {breakdown.get('role_match', 0)}",
        f"GitHub: {breakdown.get('github', 0)}",
        f"Portfolio: {breakdown.get('portfolio', 0)}",
        " ",
        "Strengths:",
    ]
    lines.extend(feedback.get("strengths", []))
    lines.append(" ")
    lines.append("Missing Skills:")
    lines.extend(feedback.get("missing_skills", []))
    lines.append(" ")
    lines.append("Quick Improvements:")
    lines.extend(feedback.get("quick_improvements", []))
    return lines

def render_auth_screen():
    st.title("Job Portfolio Evaluation using Gen AI + ML")
    st.caption("Login or create an account to save reports and view analysis history in the sidebar.")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", use_container_width=True)
        if submit_login:
            user = authenticate_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                reset_form_to_user_defaults()
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            submit_register = st.form_submit_button("Create Account", use_container_width=True)
        if submit_register:
            ok, message = register_user(name, email, password)
            if ok:
                st.success(message)
            else:
                st.error(message)


def render_sidebar_history():
    with st.sidebar:
        if mongo_connected:
            st.success("MongoDB Connected")
        else:
            st.warning("MongoDB not connected. History is available only for the current session.")

        st.header("Your Workspace")
        st.caption(f"Signed in as {st.session_state.user['name']}")
        st.text_input("Search history", key="history_search", placeholder="Search name, email, role...")

        if st.button("Start New Analysis", use_container_width=True):
            st.session_state.selected_report_id = None
            reset_form_to_user_defaults()
            st.rerun()
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()

        st.divider()
        st.subheader("History")
        history = get_user_history(st.session_state.user["email"], st.session_state.history_search)
        if not history:
            st.caption("No saved analyses match the current search.")
            return

        for item in history:
            score = item.get("ats_score", "-")
            label = item.get("job_title") or item.get("report_title") or "Untitled analysis"
            if st.button(f"{label[:24]} | {score}/100", key=f"history_{item['_id']}", use_container_width=True):
                st.session_state.selected_report_id = item["_id"]
                load_report_into_form(item)
                st.rerun()
            timestamp = item.get("timestamp")
            if isinstance(timestamp, datetime.datetime):
                st.caption(timestamp.strftime("%d %b %Y %I:%M %p"))


def render_sources_card(report):
    github_data = report.get("github_data", {})
    portfolio_text = report.get("portfolio_text", "")
    st.markdown("**Sources Evaluated**")
    st.write("Resume (PDF)")
    st.write("LinkedIn PDF" if report.get("linkedin_uploaded") else "LinkedIn PDF (not provided)")
    st.write("GitHub" if github_data.get("username") else "GitHub (not provided)")
    st.write("Portfolio" if portfolio_text and "error" not in portfolio_text else "Portfolio (not provided)")


def render_resume_rewrite_tab(report):
    report_key = str(report.get("_id", report.get("timestamp", "latest")))
    stored = st.session_state.rewrite_output.get(report_key)

    st.markdown("### Resume Rewrite Assistant")
    st.caption("Generate stronger resume bullets tailored to the selected job description.")

    if st.button("Generate Rewrite Suggestions", key=f"rewrite_btn_{report_key}", use_container_width=True):
        resume_text = report.get("resume_text", "")
        job_description = report.get("job_description", "")
        feedback = report.get("feedback", {}) or {}
        st.session_state.rewrite_output[report_key] = generate_resume_rewrite(resume_text, job_description, feedback)
        stored = st.session_state.rewrite_output.get(report_key)

    if not stored:
        sample = report.get("feedback", {}).get("sample_bullet", {})
        if sample:
            st.info("A sample rewrite is already available below. Generate more suggestions for additional tailored bullets.")
            st.markdown(f"**Before:** {sample.get('before', '')}")
            st.markdown(f"**After:** {sample.get('after', '')}")
        return

    st.write(stored.get("summary", ""))
    for item in stored.get("rewrites", []):
        st.markdown(f"**Before:** {item.get('before', '')}")
        st.markdown(f"**After:** {item.get('after', '')}")
        st.caption(item.get("why", ""))
        st.divider()


def render_report(report, read_only=False):
    feedback = report.get("feedback") or fallback_feedback()
    total = report.get("ats_score", 0)
    breakdown = report.get("score_breakdown", {})

    st.divider()
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.subheader("ATS Score Breakdown")
    with top_right:
        pdf_bytes = create_simple_pdf_bytes(
            f"Resume Analysis - {report.get('candidate_name', 'Candidate')}",
            build_report_export_lines(report),
        )
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"resume_report_{report.get('candidate_name', 'candidate').replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"pdf_{str(report.get('_id', 'current'))}",
        )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Overall Score", f"{total}/100")
        st.progress(min(max(float(total) / 100, 0.0), 1.0))
        st.markdown(f"**{feedback.get('overall_match', '')}**")
    with col2:
        render_sources_card(report)
    with col3:
        score_df = pd.DataFrame(
            {
                "Component": ["Keyword Match", "Context Similarity", "Formatting", "Impact Metrics", "Role Match", "GitHub", "Portfolio"],
                "Weight": ["22%", "28%", "15%", "10%", "10%", "10%", "5%"],
                "Score": [
                    breakdown.get("keyword", 0),
                    breakdown.get("context_similarity", 0),
                    breakdown.get("formatting", 0),
                    breakdown.get("impact", 0),
                    breakdown.get("role_match", 0),
                    breakdown.get("github", 0),
                    breakdown.get("portfolio", 0),
                ],
            }
        )
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.markdown("### Restored Session Details")
    st.caption(f"Job Description loaded: {'Yes' if report.get('job_description') else 'No'} | Resume file: {report.get('resume_filename', 'Not available')} | LinkedIn file: {report.get('linkedin_filename', 'Not available')}")

    github_data = report.get("github_data", {})
    if github_data.get("username"):
        st.divider()
        st.subheader("GitHub Profile Analysis")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Public Repos", github_data.get("public_repos", 0))
        g2.metric("Total Stars", github_data.get("total_stars", 0))
        g3.metric("Followers", github_data.get("followers", 0))
        g4.metric("GitHub Score", f"{breakdown.get('github', 0)}/100")
        if github_data.get("top_languages"):
            st.markdown(f"**Top Languages:** {' | '.join(github_data['top_languages'])}")
        if github_data.get("pinned_topics"):
            st.markdown(f"**Topics:** {' | '.join(github_data['pinned_topics'][:10])}")
        if github_data.get("repo_names"):
            st.markdown(f"**Recent Repos:** {', '.join(github_data['repo_names'][:8])}")

    st.divider()
    if read_only:
        st.caption("Viewing a saved analysis from your history.")
    st.subheader("AI Feedback")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Strengths & Gaps", "Improvements", "Profile Feedback", "Skill Gap Radar", "Action Plan", "Rewrite Assistant"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.markdown("### Strengths")
            for item in feedback.get("strengths", []):
                st.success(item)
        with right:
            st.markdown("### Missing Skills")
            for item in feedback.get("missing_skills", []):
                st.error(item)
        st.markdown("### Weak Areas")
        for item in feedback.get("weak_areas", []):
            st.warning(item)

    with tab2:
        st.markdown("### Quick Improvements")
        for item in feedback.get("quick_improvements", []):
            st.info(item)
        st.markdown("### Detailed Suggestions")
        for item in feedback.get("suggestions", []):
            st.success(item)

    with tab3:
        st.markdown("### LinkedIn Feedback")
        st.write(feedback.get("linkedin_feedback", "LinkedIn PDF not provided."))
        st.markdown("### GitHub Feedback")
        st.write(feedback.get("github_feedback", "GitHub URL not provided."))
        st.markdown("### Portfolio Feedback")
        st.write(feedback.get("portfolio_feedback", "Portfolio URL not provided."))

    with tab4:
        st.markdown("### Skill Gap Radar Chart")
        radar = feedback.get("skill_radar", {})
        labels = radar.get("labels", ["Technical Skills", "Domain Knowledge", "Projects & Portfolio", "Communication", "Leadership", "Tools & Frameworks"])
        candidate_scores = radar.get("candidate_scores", [50, 45, 40, 55, 35, 45])
        jd_required = radar.get("jd_required", [85, 80, 70, 60, 50, 75])
        labels_closed = labels + [labels[0]]
        candidate_closed = candidate_scores + [candidate_scores[0]]
        jd_closed = jd_required + [jd_required[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=jd_closed, theta=labels_closed, fill="toself", name="JD Required", line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)"))
        fig.add_trace(go.Scatterpolar(r=candidate_closed, theta=labels_closed, fill="toself", name="Your Profile", line=dict(color="#22c55e", width=2), fillcolor="rgba(34,197,94,0.2)"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)), angularaxis=dict(tickfont=dict(size=12))),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=40, b=80, l=60, r=60),
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### 30 / 60 / 90 Day Action Plan")
        action_plan = feedback.get("action_plan", {})
        c30, c60, c90 = st.columns(3)
        with c30:
            st.markdown("#### First 30 Days")
            for item in action_plan.get("30_days", []):
                st.info(item)
        with c60:
            st.markdown("#### Days 31-60")
            for item in action_plan.get("60_days", []):
                st.info(item)
        with c90:
            st.markdown("#### Days 61-90")
            for item in action_plan.get("90_days", []):
                st.info(item)

    with tab6:
        render_resume_rewrite_tab(report)

    st.divider()
    st.caption("ATS + LLM Powered Portfolio Analyzer | Resume | LinkedIn | GitHub | Portfolio")


def build_report_document(current_user, candidate_name, candidate_email, job_description, github_url, github_data, portfolio_url, portfolio_text, linkedin_pdf, linkedin_text, resume, resume_text, pages, total, scores, feedback):
    first_line = next((line.strip() for line in job_description.splitlines() if line.strip()), "New analysis")
    return {
        "user_email": current_user["email"],
        "user_name": current_user["name"],
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "report_title": candidate_name or "Analysis",
        "job_title": first_line[:60],
        "job_description": job_description,
        "resume_filename": resume.name if resume else None,
        "resume_text": resume_text,
        "resume_pages": pages,
        "github_url": github_url,
        "github_data": github_data,
        "portfolio_url": portfolio_url,
        "portfolio_text": portfolio_text,
        "linkedin_uploaded": bool(linkedin_pdf),
        "linkedin_filename": linkedin_pdf.name if linkedin_pdf else None,
        "linkedin_text": linkedin_text,
        "ats_score": total,
        "score_breakdown": scores,
        "feedback": feedback,
        "timestamp": datetime.datetime.now(),
    }


ensure_ollama_running()
st.set_page_config(page_title="Job Portfolio Evaluation", layout="wide")
init_session_state()

try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    db = client["ats_resume_advisor"]
    reports_col = db["reports"]
    users_col = db["users"]
    mongo_connected = True
except Exception:
    reports_col = None
    users_col = None
    mongo_connected = False


@st.cache_resource
def load_models():
    embed = SentenceTransformer("BAAI/bge-small-en-v1.5")
    model = spacy.load("en_core_web_sm")
    return embed, model


embed_model, nlp = load_models()

if not st.session_state.logged_in:
    render_auth_screen()
    st.stop()

if not st.session_state.form_email and st.session_state.user:
    reset_form_to_user_defaults()

selected_report = get_report_by_id(st.session_state.selected_report_id)
if selected_report and st.session_state.current_loaded_report_id != selected_report.get("_id"):
    load_report_into_form(selected_report)

render_sidebar_history()
st.title("Job Portfolio Evaluation using Gen AI + ML")
st.caption("Analyze resumes, restore prior sessions, export PDF reports, and generate rewrite suggestions.")

selected_report = get_report_by_id(st.session_state.selected_report_id)
if selected_report:
    render_report(selected_report, read_only=True)
    st.divider()

with st.sidebar:
    st.header("Candidate Details")
    name = st.text_input("Full Name *", key="form_name")
    email = st.text_input("Email *", key="form_email")
    st.divider()
    st.subheader("Profile Links")
    github_url = st.text_input("GitHub URL", key="form_github_url", placeholder="https://github.com/username")
    portfolio_url = st.text_input("Portfolio Website", key="form_portfolio_url", placeholder="https://myportfolio.com")
    linkedin_pdf = st.file_uploader("LinkedIn Profile PDF (optional)", type=["pdf"])
    if selected_report and selected_report.get("linkedin_filename"):
        st.caption(f"Saved LinkedIn PDF: {selected_report.get('linkedin_filename')}")
    if selected_report and selected_report.get("resume_filename"):
        st.caption(f"Saved Resume PDF: {selected_report.get('resume_filename')}")
    st.caption("To export LinkedIn as PDF: Profile -> More -> Save to PDF")

left_col, right_col = st.columns([1, 1])
with left_col:
    resume = st.file_uploader("Upload Resume (PDF) *", type=["pdf"])
with right_col:
    jd = st.text_area("Paste Job Description *", height=220, key="form_jd")

if selected_report and selected_report.get("job_description"):
    st.caption("The job description field has been restored from the selected session. Upload a new resume only if you want to run a fresh analysis.")

if st.button("Analyze Portfolio", type="primary", use_container_width=True):
    if not jd:
        st.error("Job Description is required.")
        st.stop()
    if not name or not email:
        st.error("Name and Email are required.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    stored_resume_text = selected_report.get("resume_text", "") if selected_report else ""
    stored_pages = selected_report.get("resume_pages", 1) if selected_report else 1
    if resume:
        status.text("Parsing resume...")
        resume_text, pages = extract_text_from_pdf(resume)
    elif stored_resume_text:
        status.text("Using restored resume from selected session...")
        resume_text, pages = stored_resume_text, stored_pages
    else:
        st.error("Please upload a resume or select a saved session with a stored resume.")
        st.stop()
    progress.progress(15)

    stored_linkedin_text = selected_report.get("linkedin_text", "") if selected_report else ""
    linkedin_text = ""
    if linkedin_pdf:
        status.text("Parsing LinkedIn PDF...")
        linkedin_text, _ = extract_text_from_pdf(linkedin_pdf)
    else:
        linkedin_text = stored_linkedin_text
    progress.progress(30)

    github_data = selected_report.get("github_data", {}).copy() if selected_report and github_url == selected_report.get("github_url", "") else {
        "username": "",
        "raw_text": "",
        "error": "Not provided",
        "top_languages": [],
        "public_repos": 0,
        "total_stars": 0,
        "followers": 0,
        "pinned_topics": [],
        "repo_names": [],
    }
    if github_url.strip() and not github_data.get("username"):
        status.text("Fetching GitHub data...")
        github_data = fetch_github_data(github_url.strip())
        if github_data.get("error") and not github_data.get("username"):
            st.warning(f"GitHub fetch issue: {github_data['error']}")
    progress.progress(55)

    portfolio_text = selected_report.get("portfolio_text", "") if selected_report and portfolio_url == selected_report.get("portfolio_url", "") else ""
    if portfolio_url.strip() and not portfolio_text:
        status.text("Scraping portfolio website...")
        portfolio_text = fetch_portfolio_text(portfolio_url.strip())
    progress.progress(70)

    status.text("Computing ATS scores...")
    total, keyword, context, formatting, impact, role_match, github, portfolio = final_score(resume_text, jd, pages, github_data, portfolio_text, linkedin_text)
    progress.progress(80)

    status.text("Generating AI feedback...")
    raw_feedback = llm_resume_feedback(resume_text, jd, total, name, github_data, portfolio_text, linkedin_text)
    feedback = parse_llm_json(raw_feedback) or fallback_feedback()
    progress.progress(95)

    score_breakdown = {
        "keyword": keyword,
        "context_similarity": context,
        "formatting": formatting,
        "impact": impact,
        "role_match": role_match,
        "github": github,
        "portfolio": portfolio,
    }

    report_document = build_report_document(
        st.session_state.user,
        name,
        email,
        jd,
        github_url,
        github_data,
        portfolio_url,
        portfolio_text,
        linkedin_pdf,
        linkedin_text,
        resume,
        resume_text,
        pages,
        total,
        score_breakdown,
        feedback,
    )

    if reports_col is not None:
        try:
            result = reports_col.insert_one(report_document)
            report_document["_id"] = result.inserted_id
            st.session_state.selected_report_id = result.inserted_id
            st.session_state.current_loaded_report_id = result.inserted_id
            st.sidebar.success("Report saved to MongoDB")
        except Exception as exc:
            st.sidebar.error(f"DB save error: {exc}")

    progress.progress(100)
    status.empty()
    render_report(report_document)
