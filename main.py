import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
# from portfolio import Portfolio
from utils import clean_text, extract_text_from_resume
import plotly.graph_objects as go
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def show_resume_radar_chart(category_scores):
    labels = list(category_scores.keys())
    values = list(category_scores.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Resume Strength'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, max(max(values), 10)]
        )),
      showlegend=False,
      title="📊 Resume Strength Radar"
    )

    st.plotly_chart(fig, use_container_width=True)

def create_streamlit_app():
    """Streamlit entry‑point for the Cold Email Generator."""
    # ---------- Page & Sidebar ---------- #
    st.set_page_config(
        layout="wide",
        page_title="Cold Email Generator",
        page_icon="📧",
    )
    st.title("ResuIntelGenAI: Smart Cold Emails with Resume + Job Match Intelligence")

    # --- Sidebar: API key --- #
    # with st.sidebar:
    #     st.header("🔑 API Configuration")
    #     groq_api_key = st.text_input("Enter your GROQ API Key", type="password")
    #     if groq_api_key:
    #         os.environ["GROQ_API_KEY"] = groq_api_key
    #         st.success("API key set for this session.")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    st.subheader("🔑 Enter Your GROQ API Key")
    api_key_input = st.text_input("GROQ API Key", type="password", value=st.session_state.api_key)

    if st.button("Save API Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("API key saved successfully!")
        else:
            st.warning("Please enter a valid API key.")
    # ---------- Main Inputs ---------- #
    resume_file = st.file_uploader("📄 Upload Your Resume", type=["pdf", "docx"])
    url_input = st.text_input(
        "Enter a Job Post URL:",
        value="https://www.amazon.jobs/en/jobs/2993489/data-scientist-data-and-machine-learning-wwps-proserve",
    )
    if st.button("Generate Analysis & Email"):
        if not resume_file:
            st.warning("Please upload your resume before generating.")
            st.stop()
        if "GROQ_API_KEY" not in os.environ:
            st.warning("Please enter your GROQ API Key in the sidebar.")
            st.stop()

        # ---------- Instantiate LLM Chain ---------- #
        llm = Chain()

        # ---------- Parse inputs ---------- #
        resume_text = extract_text_from_resume(resume_file)
        job_html = WebBaseLoader([url_input]).load().pop().page_content
        job_clean = clean_text(job_html)

        parsed_resume = llm.extract_resume_sections(resume_text)
        user_skills = parsed_resume.get("skills", [])
        jobs = llm.extract_jobs(job_clean)

        # ---------- Display Top Resume Skills ---------- #
        st.subheader("🎯 Top Skills in Resume")
        skills = [s.strip().title() for s in parsed_resume.get("skills", []) if isinstance(s, str)]
        skills = sorted(set(skills))
        # st.subheader("🎯 Top Resume Skills")
        if skills:
            st.markdown(" ".join([f"`{s}`" for s in skills]))
        else:
            st.warning("No skills were extracted from the resume.")
        # for sk in user_skills:
        #     st.markdown(f"`{sk}` ", unsafe_allow_html=True)

        # ---------- Iterate over extracted jobs ---------- #
        for idx, job in enumerate(jobs, start=1):
            st.markdown("---")
            st.header(f"📝 Analysis for Job #{idx}: {job.get('role', 'N/A')}")

            # --- Skill Matching & Explanation via LLM --- #
            skill_match = llm.skill_matching(user_skills, job.get("skills", []))
            explanation = llm.explain_skill_match(resume_text, job.get("skills", []))
            improvements = llm.improve_resume(resume_text, job_description=job.get("description", ""), job_skills=job.get("skills", []))
            email_body = llm.write_mail(job, resume_text=resume_text)

            # ---------- Tabbed UI ---------- #
            summary_tab, explain_tab, improve_tab, email_tab, resume_radar, skill_simulator = st.tabs([
                "📊 Summary",
                "🔍 Skill Attribution",
                "🛠️ Resume Suggestions",
                "✉️ Cold Email",
                "📈 Resume Radar Chart",
                "🧪\"What-If\" Skill Simulator"
            ])

            with summary_tab:
                st.subheader("📋 Match Summary")
                st.metric(
                    label="Skill Fit Percentage",
                    value=f"{skill_match.get('fit_percentage', 0)}%",
                )
                matched = skill_match.get("matched_skills", [])
                if matched:
                    st.caption("✅ Matched Skills: " + ", ".join(matched))
                else:
                    st.caption("⚠️ No matched skills found.")

            with explain_tab:
                st.subheader("🔍 Skill Attribution")
                for m in explanation.get("matched_skills", []):
                    st.markdown(f"✅ **{m['skill']}** – Found in _{m['location']}_")
                if explanation.get("unmatched_skills"):
                    st.markdown("#### ❌ Unmatched Skills & Suggestions")
                    for miss in explanation["unmatched_skills"]:
                        st.markdown(f"🔧 **{miss['skill']}** – _{miss['suggestion']}_")

            with improve_tab:
                st.subheader("🛠️ Resume Improvement Suggestions")
                for suggestion in improvements.get("suggested_changes", []):
                    st.markdown(f"▪️ {suggestion}")
                enhancer = llm.improve_resume(
                    resume_text=resume_text,
                    job_description=job.get("description", ""),
                    job_skills=job.get("skills", [])
                )

                st.subheader("💡 Smart Resume Enhancement Suggestions")

                st.markdown("### ❌ Missing Skills")
                for skill in enhancer.get("missing_skills", []):
                    st.markdown(f"- {skill}")

                st.markdown("### ✍️ Suggested Changes")
                for change in enhancer.get("suggested_changes", []):
                    st.markdown(f"- {change}")

                st.markdown("### ➕ New Section Ideas")
                for idea in enhancer.get("new_section_ideas", []):
                    st.markdown(f"- {idea}")

            with email_tab:
                st.subheader("✉️ Generated Cold Email")
                st.code(email_body, language="markdown")
            
            with resume_radar:
                st.subheader("📈 Resume Strength Radar Chart")
                category_scores = llm.analyze_resume_categories(resume_text, skills)
                show_resume_radar_chart(category_scores)
            
            with skill_simulator:
                st.subheader("🧪 What-If Skill Simulator")
                if "added_skills" not in st.session_state:
                    st.session_state.added_skills = []
                missing = [m["skill"] for m in explanation.get("unmatched_skills", [])]
                added_skills = st.multiselect(
                    "Add hypothetical skills to your resume:",
                    options=missing,
                    default=st.session_state.added_skills,
                    key="skill_sim_input"
                )
                st.session_state.added_skills = added_skills
                simulated_resume_skills = list(set(skills + added_skills))  # original resume + what-if
                if added_skills:
                    sim_fit = llm.skill_matching(simulated_resume_skills, job.get("skills", []))
                    st.metric("🔁 Simulated Fit %", f"{sim_fit['fit_percentage']}%")
                    matched_skills = sim_fit.get("matched_skills", [])
                    st.markdown("✅ Matched Skills With Simulation:")
                    st.markdown(", ".join(matched_skills))

                    if st.button("Generate Email with Simulated Skills"):
                        simulated_email = llm.write_mail(job, resume_text="; ".join(simulated_resume_skills))
                        st.subheader("📧 Simulated Cold Email")
                        st.code(simulated_email, language="markdown")




if __name__ == "__main__":
    # portfolio = Portfolio()  # kept for future portfolio querying if needed
    create_streamlit_app()
