import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
# from dotenv import load_dotenv

# load_dotenv()

class Chain:
    def __init__(self, api_key=None):
        # if api_key:
        #     os.environ["GROQ_API_KEY"] = api_key
        # elif not os.getenv("GROQ_API_KEY"):
        #     raise ValueError("GROQ_API_KEY must be set either as an environment variable or passed as an argument.")
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Each job posting should include the following keys:
            - `role`
            - `experience`
            - `description`
            - `skills`: a **JSON list** of atomic, technical skills like ["Python", "TensorFlow", "Docker"]

            For `skills`, parse from both job description and skill requirements.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, resume_text):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### MY RESUME:
            {resume_text}

            ### INSTRUCTION:
            You are writing a **cold email** to express interest in the job opportunity described above.

            Write the email in **first-person voice**, from the perspective of the applicant (me). Use the resume content provided to:
            - Clearly state who I am and what position or background I have.
            - Explain how my skills, experience, and accomplishments make me a strong fit for this job.
            - Show alignment with the job requirements or mission.
            - Be clear, confident, professional, and concise.
            - Optionally refer to a few relevant projects or experience highlights that support my application.

            Do not add any preamble, headings, or external commentary.

            ### EMAIL:
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
        "job_description": str(job),
        # "link_list": links,
        "resume_text": resume_text
        })
        return res.content
    def extract_resume_sections(self, resume_text):
        prompt = PromptTemplate.from_template(
            """
            ### RESUME TEXT:
            {resume_text}

            ### INSTRUCTION:
            Extract the following fields from the resume text and return as JSON:
            - name
            - email
            - phone
            - skills (list)
            - education (list of degrees + institutes + years)
            - experience (list of roles, companies, durations)
            - certifications (if any)
            - projects (if any)

            Do not add any preamble, headings, or external commentary.
            
            ### FORMAT:
            {{
                "name": "...",
                "email": "...",
                "phone": "...",
                "skills": ["...", "..."],
                "education": [...],
                "experience": [...],
                "certifications": [...],
                "projects": [...]
            }}
            ONLY return valid JSON.
            """
        )
        chain = prompt | self.llm
        response = chain.invoke({"resume_text": resume_text})
        
        parser = JsonOutputParser()
        return parser.parse(response.content)

    def skill_matching(self, resume_skills, job_skills):
            prompt_extract = PromptTemplate.from_template(
                """
                ### SKILLS FROM RESUME:
                {resume_skills}
                ### SKILLS FROM JOB DESCRIPTION:
                {job_skills}
                ### INSTRUCTION:
                Calculate the percentage of skills from the resume that match the job description.
                Return a JSON object with the following keys:
                - `fit_percentage`: the percentage of skills from the resume that match the job description.
                - `matched_skills`: a list of skills that matched.
                Do not add any preamble, headings, or external commentary.
                ### FORMAT:
                {{
                    "fit_percentage": 85,
                    "matched_skills": ["Python", "TensorFlow", "Docker"]
                }}
                ONLY return valid JSON.
                """
            )
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={"resume_skills": resume_skills, "job_skills": job_skills})
            try:
                json_parser = JsonOutputParser()
                res = json_parser.parse(res.content)
            except OutputParserException:
                raise OutputParserException("Context too big. Unable to parse jobs.")
            return res
    def improve_resume(self, resume_text, job_description, job_skills):
        prompt_improve = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            ### JOB SKILLS:
            {job_skills}

            ### RESUME:
            {resume_text}

            ### INSTRUCTION:
            Identify missing or weakly represented job skills in the resume. Suggest exact changes to the resume to improve the match, such as:
            - Rewording existing bullets to better align
            - Adding new projects/skills (if relevant)
            - Reordering sections for impact
            Do not add any preamble, headings, or external commentary.
            ### FORMAT:
            {{
                "missing_skills": ["Python", "TensorFlow"],
                "suggested_changes": [
                    "Add a project on machine learning with TensorFlow",
                    "Reword the Python experience to highlight data analysis"
                ]
            }}

            """
        )
        chain_improve = prompt_improve | self.llm
        res = chain_improve.invoke(input={"resume_text": resume_text, "job_description": job_description,"job_skills": job_skills})
        try:
                json_parser = JsonOutputParser()
                res = json_parser.parse(res.content)
        except OutputParserException:
                raise OutputParserException("Context too big. Unable to parse jobs.")
        return res

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))