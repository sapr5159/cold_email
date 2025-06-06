import re
import pdfplumber
from docx import Document

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_text_from_resume(file):

    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        raise ValueError("Unsupported file format.")

def extract_resume_sections(text):
    sections = {
        "experience": "",
        "education": "",
        "skills": "",
        "projects": "",
    }

    current_section = None
    for line in text.split("\n"):
        line = line.strip().lower()
        if "experience" in line:
            current_section = "experience"
        elif "education" in line:
            current_section = "education"
        elif "skill" in line:
            current_section = "skills"
        elif "project" in line:
            current_section = "projects"
        elif current_section:
            sections[current_section] += line + "\n"
    return sections

def extract_skills_from_text(text):
    # You can make this list more comprehensive
    # common_skills = [
    #     "python", "java", "sql", "excel", "pandas", "numpy", "machine learning",
    #     "deep learning", "data analysis", "communication", "leadership", "tableau",
    #     "power bi", "aws", "git", "tensorflow", "keras", "flask", "django"
    # ]
    common_skills = [
        # Machine Learning
        "Supervised Learning", "Unsupervised Learning", "Scikit-learn", "TensorFlow", "Keras",
        "PyTorch", "Model Evaluation", "Feature Engineering", "Cross-Validation", "Hyperparameter Tuning",
        "XGBoost", "LightGBM", "Gradient Descent", "Overfitting", "Regularization", "ML Pipelines",

        # Data Science
        "Python", "R", "SQL", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Jupyter Notebook",
        "Data Cleaning", "Exploratory Data Analysis", "Statistics", "Hypothesis Testing", "Probability",
        "A/B Testing", "Predictive Modeling", "Time Series Analysis", "Business Intelligence",

        # Data Analysis
        "Excel", "Power BI", "Tableau", "Pivot Tables", "VLOOKUP", "XLOOKUP", "Data Wrangling",
        "Descriptive Statistics", "Report Writing", "Google Sheets", "Data Visualization", "KPI Analysis",
        "SQL Joins", "SQL Aggregations",

        # Data Engineering
        "PostgreSQL", "MySQL", "MongoDB", "Apache Spark", "Apache Kafka", "ETL", "ELT Pipelines",
        "Airflow", "Data Warehousing", "Redshift", "Snowflake", "BigQuery", "dbt", "AWS S3", "AWS Lambda",
        "AWS Glue", "Azure Data Factory", "Data Lake", "Docker", "Kubernetes",

        # Natural Language Processing
        "Text Preprocessing", "Tokenization", "Lemmatization", "TF-IDF", "Word2Vec", "FastText",
        "Transformers", "BERT", "RoBERTa", "GPT", "Named Entity Recognition", "NER", "Sentiment Analysis",
        "Text Classification", "Hugging Face Transformers", "spaCy", "NLTK",

        # LLMs & Agents
        "OpenAI GPT", "LLaMA", "Claude", "Gemini", "LangChain", "Prompt Engineering",
        "Retrieval-Augmented Generation", "RAG", "Pinecone", "ChromaDB", "Weaviate", "Vector Embeddings",
        "Agent Frameworks", "LangGraph", "CrewAI", "AutoGen", "Tool Use", "Tool Calling",
        "Memory Management", "Guardrails", "Output Parsing", "Chain of Thought", "ReAct", "Streaming Outputs",
        "Chat Completion APIs",

        # General Tools
        "Git", "GitHub", "Docker", "APIs", "REST", "GraphQL", "FastAPI", "Flask", "Linux", "Bash",
        "CI/CD", "Cloud Platforms", "AWS", "Azure", "GCP", "VS Code", "Testing", "Pytest", "Unit Testing"
    ]

    found_skills = []
    text_lower = text.lower()
    print(f"Extracting skills from text: {text_lower[:100]}...")  # Debugging line to check the text
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    return sorted(set(found_skills))

def calculate_fit_percentage(resume_skills, job_skills):
    resume_skills_set = set(skill.lower() for skill in resume_skills)
    job_skills_set = set(skill.lower() for skill in job_skills)

    matched = resume_skills_set.intersection(job_skills_set)
    fit_score = len(matched) / len(job_skills_set) * 100 if job_skills_set else 0
    return round(fit_score, 2), list(matched)