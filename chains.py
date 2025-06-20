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
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="meta-llama/llama-4-maverick-17b-128e-instruct")

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
    def explain_skill_match(self, resume_text, job_skills):
        prompt = PromptTemplate.from_template(
            """
            ### RESUME TEXT:
            {resume_text}

            ### JOB SKILLS:
            {job_skills}

            ### INSTRUCTION:
            Identify which of the job-required skills are present in the resume and where they are mentioned
            (e.g., Skills section, Project section, Experience line, etc.).

            Also identify which job-required skills are missing, and provide suggestions on how they could be added or better represented.

            Return a valid JSON with:
            {{
                "matched_skills": [
                    {{"skill": "Python", "location": "Skills section"}},
                    {{"skill": "TensorFlow", "location": "Project: CNN Image Classifier"}}
                ],
                "unmatched_skills": [
                    {{"skill": "JAX", "suggestion": "Add a project using JAX or mention online coursework"}},
                    {{"skill": "Numerical Optimization", "suggestion": "Mention relevant optimization techniques used in projects"}}
                ]
            }}

            ONLY return valid JSON.
            """
        )


        chain = prompt | self.llm
        result = chain.invoke({"resume_text": resume_text, "job_skills": job_skills})

        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(result.content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse skill match explanation: {e}")
        return result
    def improve_resume(self, resume_text, job_description, job_skills):
        prompt = PromptTemplate.from_template(
            """
            ### RESUME TEXT:
            {resume_text}

            ### JOB DESCRIPTION:
            {job_description}

            ### JOB SKILLS:
            {job_skills}

            ### INSTRUCTION:
            Analyze the resume in context of the job. Suggest specific improvements:
            - List missing but relevant skills that should be added
            - Suggest rewording of any vague or generic lines
            - Identify if any new sections (e.g., projects, certifications) should be added

            Return structured JSON like:
            {{
                "missing_skills": ["JAX", "Numerical Optimization"],
                "suggested_changes": [
                    "Mention JAX in project section using a small case study.",
                    "Reword 'Worked on AI projects' to 'Built a CNN using PyTorch for image classification'."
                ],
                "new_section_ideas": ["Add a project using distributed computing tools like Spark"]
            }}

            Only return valid JSON.
            """
        )
        chain = prompt | self.llm
        response = chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description,
            "job_skills": job_skills
        })
        try:
            parser = JsonOutputParser()
            return parser.parse(response.content)
        except Exception as e:
            raise RuntimeError(f"Unable to parse resume improvement suggestions: {e}")
    def analyze_resume_categories(self, resume_text, skills_list):
        categories = {
            "Technical Skills": 0,
            "Tools & Frameworks": 0,
            "Cloud Experience": 0,
            "NLP / LLM Relevance": 0,
            "Soft Skills": 0,
            "Projects": 0
        }

        resume_lower = resume_text.lower()

        # Define category-specific keywords
        tools = ["tensorflow", "pytorch", "keras", "docker", "git", "onnx", "scikit", "opencv"]
        cloud = ["aws", "azure", "gcp", "google cloud", "s3", "lambda"]
        nlp = ["nlp", "bert", "gpt", "llm", "langchain", "transformers"]
        soft = ["communication", "leadership", "collaboration", "team", "initiative"]
        
        # Count matches
        categories["Technical Skills"] = len(skills_list)
        categories["Tools & Frameworks"] = sum(t in resume_lower for t in tools)
        categories["Cloud Experience"] = sum(c in resume_lower for c in cloud)
        categories["NLP / LLM Relevance"] = sum(n in resume_lower for n in nlp)
        categories["Soft Skills"] = sum(s in resume_lower for s in soft)
        categories["Projects"] = resume_lower.count("project")  # basic proxy

        return categories


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))