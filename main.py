from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Response, HTTPException
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.agents import create_openai_functions_agent, create_tool_calling_agent, AgentExecutor
import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
import tempfile
import subprocess
from pathlib import Path
from tools.latex_converter import LaTeXResumeConverter
from typing import List, Optional

load_dotenv()
app=FastAPI()


os.environ["OPEN_AI_KEY"] = os.getenv("OPEN_AI_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["REDIS_URI"] = os.getenv("REDIS_URI")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Enable Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://neural-frontend-alpha.vercel.app/", "https://neural-frontend-alpha.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PersonalInfo(BaseModel):
    name: str = Field(..., description="Full name of the person")
    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number")
    github: str = Field(..., description="GitHub profile link")
    linkedin: str = Field(..., description="LinkedIn profile link")

class Experience(BaseModel):
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    start_date: str = Field(..., description="Start date of employment (YYYY-MM)")
    end_date: Optional[str] = Field(None, description="End date of employment (YYYY-MM), None if still employed")
    job_type: Optional[str] = Field("Not Specified", description="Job Type (Remote, On-site, Hybrid)")
    responsibilities: List[str] = Field(..., description="List of job responsibilities")

class Education(BaseModel):
    institution: str = Field(..., description="Educational institution name")
    location: Optional[str] = Field(None, description="Location of the institution")
    degree: str = Field(..., description="Degree obtained")
    graduation_date: str = Field(..., description="Graduation date (YYYY-MM)")

class Projects(BaseModel):
    title: str = Field(..., description="Title of the project")
    tech_stack: List[str] = Field(..., description="Technologies used in this project")
    features: List[str] = Field(..., description="List of key features in the project")
    duration: str = Field(..., description="Duration of the project (e.g., 3 months)")

# Structured Skill Categories
class Skills(BaseModel):
    languages: List[str] = Field(..., description="List of programming languages")
    frameworks: List[str] = Field(..., description="List of frameworks")
    developer_tools: List[str] = Field(..., description="List of developer tools")
    libraries: List[str] = Field(..., description="List of libraries")

class Resume(BaseModel):
    personal_info: PersonalInfo
    summary: str = Field(..., description="Professional summary")
    experience: List[Experience]
    education: List[Education]
    projects: List[Projects]
    skills: Skills  # Updated to use structured categories



@app.post("/chat")
async def read_root(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        model = AzureChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPEN_AI_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION")
        )

        system_message = """
You are an expert resume builder assistant. Your role is to:
1. Collect all required information from the user through a series of questions.
2. When receiving content for "experience" or "projects" (especially responsibilities and features), use the format_responsibilities tool to break down the text into well-structured bullet points.
3. Optimize the content for ATS systems using the optimize_resume_for_ats tool.
4. Create a structured JSON resume using the format_resume_json tool.
5. Convert the resume to LaTeX format using the convert_to_latex tool.
6. Generate a PDF from the LaTeX content using the generate_pdf tool and save it locally.

Follow the provided JSON schema and ensure all information is complete before formatting.
If any information is missing, ask the user for it before proceeding.

The JSON schema should follow this structure:
{{
    "personal_info": {{
        "name": "string",
        "email": "string",
        "phone": "string",
        "github": "string",
        "linkedin": "string"
    }},
    "summary": "string",
    "experience": [{{
        "company": "string",
        "title": "string",
        "start_date": "string",
        "end_date": "string or null",
        "job_type": "string",
        "responsibilities": ["string"]  # Use format_responsibilities tool here
    }}],
    "education": [{{
        "institution": "string",
        "location": "string or null",
        "degree": "string",
        "graduation_date": "string"
    }}],
    "projects": [{{
        "title": "string",
        "tech_stack": ["string"],
        "features": ["string"],  # Use format_responsibilities tool here
        "duration": "string"
    }}],
    "skills": {{
        "languages": ["string"],
        "frameworks": ["string"],
        "developer_tools": ["string"],
        "libraries": ["string"]
    }}
}}
"""

        redis_history = RedisChatMessageHistory(
            url=os.getenv("REDIS_URI"), 
            session_id="1234",
            ttl=3600
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=redis_history, 
            return_messages=True
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

        @tool
        def format_responsibilities(text: str) -> List[str]:
            """Breaks down long text into well-structured, impactful bullet points for better readability."""

            try:
                format_large_prompt = PromptTemplate.from_template("""
                Transform the following text into **at least 5 strong, impactful bullet points** based on these rules:
                - **Start with powerful action verbs** (e.g., "Developed", "Optimized", "Implemented").
                - **Include measurable impact** (e.g., "Increased efficiency by 30%").
                - **Focus on achievements rather than generic tasks**.
                - **Use present tense for current roles, past tense for previous roles**.
                - **Keep each bullet concise (10-15 words max)**.

                ### **Example Before & After**
                
                **Before (Weak Statement):**  
                "Managed backend tasks for API development."  
                
                **After (Optimized with Metrics & Action Verbs):**  
                - "Developed and optimized **RESTful APIs** in **NestJS**, reducing response time by **40%**."
                - "Integrated **Kafka-based event-driven architecture**, improving data consistency across microservices."
                - "Implemented **GraphQL endpoints**, enhancing frontend query efficiency by **25%**."
                - "Designed and deployed **secure authentication flows** using **OAuth & JWT**."
                - "Collaborated with cross-functional teams to deliver features **2 weeks ahead of schedule**."

                **Now, transform the given text into optimized bullet points:**
                
                Text to transform:  
                {text}
                
                **Return only the bullet points, one per line, starting with '- '**
                """)

                result = model.invoke(format_large_prompt.format(text=text))
                return result.content.split("\n")  # Ensure bullet points are returned as a list

            except Exception as e:
                return [f"Error breaking into bullet points: {str(e)}"]

        @tool
        def format_resume_json(resume_data: Resume) -> str:
            """Format resume data into a structured JSON. Expects data in the Resume model format."""
            try:
                # Convert to dict and then to JSON
                return json.dumps(resume_data.model_dump(), indent=2)
            except Exception as e:
                return f"Error formatting JSON: {str(e)}"

        @tool
        def convert_to_latex(json_str: str) -> str:
            """Convert JSON resume to LaTeX format.
                Args:
                    json_str: JSON string or dict containing resume data
        
                Returns:
                    str: LaTeX formatted resume
            """
            try:
                if isinstance(json_str, str):
                    resume_data = json.loads(json_str)
                else:
                    resume_data = json_str
                # Validate the data using the Resume model
                resume = Resume(**resume_data)
                converter = LaTeXResumeConverter()
                latex_output = converter.convert_json_to_latex(resume)
                return latex_output
            except json.JSONDecodeError as e:
                return f"Error parsing JSON: {str(e)}"
            except ValueError as e:
                return f"Error validating resume data: {str(e)}"
            except Exception as e:
                return f"Error converting to LaTeX: {str(e)}"

        @tool
        def optimize_resume_for_ats(resume_data: Resume) -> str:
            """Optimize resume content for ATS systems with a 95%+ score"""

            optimise_resume_template = PromptTemplate.from_template("""
                You are an **expert in Applicant Tracking Systems (ATS) optimization** and a **resume writing specialist**.
                Your goal is to transform the given resume to achieve at least a **95% ATS score** by:
                - **Optimizing phrasing** for clarity and impact
                - **Adding industry-relevant keywords** based on the job role
                - **Ensuring readability** with proper bullet points and formatting
                - **Removing redundant or weak wording**
                - **Keeping the content concise and impactful**
                
                ### **Examples of Improvements:**
                
                **Before (Weak Statement):**  
                - "Worked on backend APIs"  
                
                **After (Optimized for ATS):**  
                - "Designed and developed **scalable RESTful APIs** using **Node.js, NestJS, and MongoDB**, improving response times by **40%**."

                **Before (Generic Description):**  
                - "Experienced in cloud services"  

                **After (Keyword-Rich & Impactful):**  
                - "Expert in **AWS (EC2, S3, Lambda)** and **Google Cloud Platform (GCP)**, optimizing cloud infrastructure for **high availability** and **cost efficiency**."

                **Transform the given resume into an optimized version following these best practices:**

                **Original Resume Data:**  
                {resume_data}

                **Optimized ATS-Ready Resume:**
            """)

            result = model.invoke(optimise_resume_template.format(resume_data=resume_data))
            return result.content

        @tool
        def generate_pdf(latex_content: str) -> str:
            """Generate a PDF from LaTeX content and save it in the local directory"""
            try:
                # Create a complete standalone LaTeX document
                full_latex_content = r"""
        \documentclass{article}
        \usepackage[left=0.4in,top=0.4in,right=0.4in,bottom=0.4in]{geometry}
        \usepackage{hyperref}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}

        % Define commands for resume formatting
        \newcommand{\name}[1]{{\huge\textbf{#1}}\\\vspace{0.5em}}
        \newcommand{\address}[1]{\textbf{#1}\\}
        \newcommand{\tab}[1]{\hspace{.2667\textwidth}\rlap{#1}}
        \newcommand{\itab}[1]{\hspace{0em}\rlap{#1}}

        % Define resume section environment
        \newenvironment{rSection}[1]{
            \vspace{0.5em}
            \textbf{\Large{#1}}
            \vspace{0.5em}
            \hrule
            \begin{list}{}{
                \setlength{\leftmargin}{0em}
            }
            \item[]
        }{
            \end{list}
            \vspace{0.5em}
        }

        \begin{document}
        """ + latex_content + r"""
        \end{document}
        """
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Write LaTeX content to temporary file
                    tex_file = temp_path / "resume.tex"
                    with open(tex_file, "w", encoding='utf-8') as f:
                        f.write(full_latex_content)
                    
                    # Try different possible pdflatex paths
                    pdflatex_paths = [
                        "pdflatex",
                        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
                        r"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\pdflatex.exe"
                    ]
                    
                    pdflatex_exe = None
                    for path in pdflatex_paths:
                        try:
                            subprocess.run([path, "--version"], capture_output=True)
                            pdflatex_exe = path
                            break
                        except Exception:
                            continue
                    
                    if not pdflatex_exe:
                        return "Error: pdflatex not found. Please install MiKTeX and ensure it's in your system PATH"
                    
                    # Run pdflatex
                    try:
                        process = subprocess.run(
                            [pdflatex_exe, "-interaction=nonstopmode", "resume.tex"],
                            cwd=temp_dir,
                            capture_output=True,
                            text=True,
                            encoding='utf-8'
                        )
                        
                        # Create output directory if it doesn't exist
                        output_dir = Path("generated_pdfs")
                        output_dir.mkdir(exist_ok=True)
                        
                        # Copy the generated PDF to our output directory
                        pdf_file = temp_path / "resume.pdf"
                        if pdf_file.exists():
                            output_path = output_dir / "resume.pdf"
                            with open(pdf_file, "rb") as src, open(output_path, "wb") as dst:
                                dst.write(src.read())
                            return f"PDF generated successfully and saved to {output_path}"
                        else:
                            return f"PDF generation failed. LaTeX Error: {process.stderr}"
                            
                    except Exception as e:
                        return f"Error running pdflatex: {str(e)}"
                        
            except Exception as e:
                return f"Error generating PDF: {str(e)}"

        tools = [format_resume_json, convert_to_latex, optimize_resume_for_ats, generate_pdf, format_responsibilities]
        
        agent = create_tool_calling_agent(
            llm=model,
            tools=tools,
            prompt=prompt,
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True
        )
        
        response = agent_executor.invoke({
            "input": query,
            # "chat_history": messages
        })
        
        return Response(content=response["output"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))