# 🤖 ResuIntelGenerator: Smart Cold Emails with Resume + Job Match Intelligence

**ResuIntelGenerator** is a Streamlit-powered application that helps you generate personalized cold emails based on your resume and a specific job posting. It also provides detailed skill matching, resume improvement suggestions, and visual analytics to enhance your job application success.

---

## 🚀 Features

* ✅ Upload your resume (PDF or DOCX)
* ✅ Enter a job post URL
* ✅ Extract structured job info and required skills
* ✅ Extract and analyze skills from your resume
* ✅ Match resume skills with job skills & calculate fit %
* ✅ Get explainable skill match analysis (where a skill was found)
* ✅ Smart resume suggestions from an LLM
* ✅ Cold email generated using job + resume context
* ✅ Visual resume radar chart (skill gaps & strengths)
* ✅ What-If skill simulator (see how adding skills improves fit)

---

## 📂 Folder Structure

```
.
├── main.py               # Streamlit entry point
├── chains.py             # LangChain-based LLM interface
├── utils.py              # Resume parsing, cleaning, etc.
├── portfolio.py          # (Optional) portfolio link matcher
├── requirements.txt      # Dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # (Optional) UI settings
```

---

## 🔑 API Key Required

This app uses the **Groq API** via `langchain-groq`. You must provide your key on the app interface.

* Visit [https://console.groq.com/keys](https://console.groq.com/keys) to get one.
* The key is never stored — it is used only for your session.

---

## ▶️ How to Run Locally

```bash
# Clone this repo
https://github.com/sapr5159/cold_email.git
cd resuintelgenerator

# Create virtual env
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

---

## 📦 Key Dependencies

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-groq
pypdf
pdfplumber
python-docx
plotly
```

---

## 🤝 Contributions

Contributions, feedback, and ideas are welcome! Open a PR or issue if you'd like to collaborate.

---

## 📄 License

MIT License. Feel free to fork and build on top.

---

## 🙋‍♂️ Creator

Built by [Sathish Kumar P.](https://github.com/sapr5159) — powered by Groq, LangChain, and Streamlit.
