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
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
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


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))