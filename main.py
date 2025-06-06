import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text, extract_text_from_resume, calculate_fit_percentage


def create_streamlit_app(portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    st.title("Cold Mail Generator")

    # Sidebar: Enter Grow API Key
    with st.sidebar:
        st.header("🔑 API Configuration")
        GROQ_API_KEY = st.text_input("Enter your Grow API Key", type="password")
        if GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
            st.success("API key set for this session.")
    
    resume_file = st.file_uploader("📄 Upload Your Resume", type=["pdf", "docx"])
    url_input = st.text_input("Enter a Job Post URL:", value="https://www.amazon.jobs/en/jobs/2993489/data-scientist-data-and-machine-learning-wwps-proserve?cmpid=SPLICX0248M&ss=paid&utm_campaign=cxro&utm_content=job_posting&utm_medium=social_media&utm_source=linkedin.com")
    submit_button = st.button("Generate Email")

    if submit_button:
        # api_key = os.getenv("GROQ_API_KEY")
        if not resume_file:
            st.warning("Please upload your resume.")
            return
        if not os.getenv("GROQ_API_KEY"):
            st.warning("Please enter your Grow API Key in the sidebar.")
            return
        try:
            # ✅ Instantiate Chain after API key is set
            llm = Chain()  # pass explicitly if your Chain class accepts it
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            # portfolio.load_portfolio()
            resume_text = extract_text_from_resume(resume_file)
            # st.subheader("📌 Resume Preview")
            # st.text_area("Extracted Text", resume_text, height=300)

            parsed_resume = llm.extract_resume_sections(resume_text)
            # st.json(parsed_resume)
            skills = parsed_resume.get("skills", [])
            st.subheader("Top Skills")
            for skill in skills:
                st.markdown(f"`{skill}` ", unsafe_allow_html=True)
            # skills = llm.extract_skills_from_text(resume_text)

            # st.subheader("🧩 Extracted Sections")
            # for key, value in sections.items():
            #     st.markdown(f"**{key.title()}**")
            #     st.code(value.strip() or "N/A", language="markdown")

            # st.subheader("🎯 Top Skills Identified")
            # st.success(", ".join(skills) if skills else "No known skills found.")

            jobs = llm.extract_jobs(data)

            for job in jobs:
                job_skills = job.get('skills', [])
                fit_score, matched = calculate_fit_percentage(skills, job_skills)
                skill_match = llm.skill_matching(skills, job_skills)
                st.subheader(f"📋 Job Role: {job.get('role', 'N/A')}")
                st.metric(label="🎯 Skill Fit Percentage", value=f"{skill_match.get('fit_percentage', 0)}%")
                matched_skills = skill_match.get('matched_skills', [])
                if matched_skills:
                    st.caption(f"✅ Matched Skills: {', '.join(matched_skills)}")
                else:
                    st.caption("⚠️ No matched skills found.")
                if skill_match.get('fit_percentage', 0) < 50:
                    st.warning(f"⚠️ Low Fit Score: {fit_score}%")
                    improvise = llm.improve_resume(
                        resume_text=resume_text,
                        job_description=job.get('description', ''),
                        job_skills=job_skills
                    )
                    st.subheader("📝 Suggested Improvements")
                    st.code(improvise, language="markdown")
                else:
                    st.success(f"✅ High Fit Score: {fit_score}%")
                # links = portfolio.query_links(skills)
                # email = llm.write_mail(job, links)
                email = llm.write_mail(job, resume_text=resume_text)
                st.subheader("📧 Generated Cold Email")
                st.code(email, language="markdown")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")
            st.stop()


if __name__ == "__main__":
    portfolio = Portfolio()
    create_streamlit_app(portfolio, clean_text)
