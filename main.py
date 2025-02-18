from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Response, HTTPException
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
import json
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


load_dotenv()
app=FastAPI()

os.environ["OPEN_AI_KEY"] = os.getenv("OPEN_AI_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Enable Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://neural-frontend-alpha.vercel.app/", "https://neural-frontend-alpha.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# LaTeX helper functions
def safe_get(data: dict, *keys, default="") -> str:
    """Safely get nested dictionary values, returning default if not found"""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current if current is not None else default

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters"""
    chars = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
        '\\': '\\textbackslash{}'
    }
    return ''.join(chars.get(c, c) for c in str(text))

def format_text_with_bullets(text: str) -> str:
    """Format text into LaTeX bullet points"""
    if not text:
        return ""
    lines = text.split('\n') if '\n' in text else [text]
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = line.lstrip('•-*').strip()
        formatted_lines.append(f"    \\item {escape_latex(line)}")
    return '\n'.join(formatted_lines)

class PersonalInfo(BaseModel):
    name: str = Field(..., description="Full name of the person")
    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number")
    location: str = Field(..., description="Current location")

class Experience(BaseModel):
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    start_date: str = Field(..., description="Start date of employment")
    end_date: Optional[str] = Field(default="Present", description="End date of employment")
    location: Optional[str] = Field(default="", description="Job location")
    responsibilities: str = Field(..., description="Job responsibilities")

class Education(BaseModel):
    institution: str = Field(..., description="Educational institution name")
    degree: str = Field(..., description="Degree obtained")
    graduation_date: str = Field(..., description="Graduation date")

class Resume(BaseModel):
    personal_info: PersonalInfo
    summary: str = Field(..., description="Professional summary")
    experience: List[Experience]
    education: List[Education]
    skills: List[str] = Field(..., description="List of skills")

def json_to_latex_resume(json_data: dict) -> str:
    """Convert JSON resume data to LaTeX maintaining exact template style"""
    try:
        latex_content = []
        
        # Document setup - exactly as provided
        latex_content.extend([
            "\\documentclass{resume}",
            "\\usepackage[left=0.4 in,top=0.4in,right=0.4 in,bottom=0.4in]{geometry}",
            "\\newcommand{\\tab}[1]{\\hspace{.2667\\textwidth}\\rlap{#1}}",
            "\\newcommand{\\itab}[1]{\\hspace{0em}\\rlap{#1}}",
            ""
        ])
        
        # Personal Information
        personal = json_data.get('personal_info', {})
        name = safe_get(personal, 'name')
        if name:
            latex_content.append(f"\\name{{{escape_latex(name)}}}")
        
        # Split contact info into two address blocks exactly like template
        phone_location = f"{safe_get(personal, 'phone')} \\\\ {safe_get(personal, 'location')}"
        email = safe_get(personal, 'email')
        email_block = f"\\href{{mailto:{email}}}{{{escape_latex(email)}}}"
        
        latex_content.extend([
            f"\\address{{{phone_location}}}",
            f"\\address{{{email_block}}}",
            "",
            "\\begin{document}",
            ""
        ])
        
        # Summary section formatted as OBJECTIVE
        if summary := safe_get(json_data, 'summary'):
            latex_content.extend([
                "\\begin{rSection}{OBJECTIVE}",
                escape_latex(summary),
                "\\end{rSection}",
                ""
            ])
        
        # Education section with exact formatting
        if education := json_data.get('education', []):
            latex_content.append("\\begin{rSection}{Education}")
            for edu in education:
                degree = safe_get(edu, 'degree')
                institution = safe_get(edu, 'institution')
                grad_date = safe_get(edu, 'graduation_date')
                latex_content.extend([
                    f"{{\\bf {escape_latex(degree)}}}, {escape_latex(institution)} \\hfill {{{escape_latex(grad_date)}}}",
                    ""
                ])
            latex_content.extend(["\\end{rSection}", ""])
        
        # Skills section with tabular format
        if skills := json_data.get('skills', []):
            latex_content.extend([
                "\\begin{rSection}{SKILLS}",
                "\\begin{tabular}{ @{} >{{\\bfseries}}l @{\\hspace{6ex}} l }",
                f"Technical Skills & {', '.join(escape_latex(skill) for skill in skills)}\\\\",
                "\\end{tabular}\\\\",
                "\\end{rSection}",
                ""
            ])
        
        # Experience section with exact spacing and formatting
        if experience := json_data.get('experience', []):
            latex_content.append("\\begin{rSection}{EXPERIENCE}")
            for exp in experience:
                title = safe_get(exp, 'title')
                company = safe_get(exp, 'company')
                start = safe_get(exp, 'start_date')
                end = safe_get(exp, 'end_date', 'Present')
                location = safe_get(exp, 'location')
                
                latex_content.extend([
                    f"\\textbf{{{escape_latex(title)}}} \\hfill {escape_latex(start)} - {escape_latex(end)}\\\\",
                    f"{escape_latex(company)} \\hfill \\textit{{{escape_latex(location)}}}",
                    "\\begin{itemize}",
                    "    \\itemsep -3pt {}"
                ])
                
                if responsibilities := safe_get(exp, 'responsibilities'):
                    latex_content.append(format_text_with_bullets(responsibilities))
                
                latex_content.extend([
                    "\\end{itemize}",
                    ""
                ])
            
            latex_content.extend(["\\end{rSection}", ""])
        
        latex_content.append("\\end{document}")
        return "\n".join(latex_content)
    
    except Exception as e:
        return f"% Error generating LaTeX: {str(e)}"

@app.post("/chat")
async def read_root(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        messages = data.get("messages")
        model = AzureChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPEN_AI_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION")
        )

        system_message = """
        You are an expert resume builder assistant. Your role is to:
        1. Collect all required information from the user through a series of questions
        2. Create a structured JSON resume using the format_resume_json tool
        3. Convert the resume to LaTeX format using the convert_to_latex tool
        4. Optimize the content for ATS systems using the optimize_resume_for_ats tool
        
        Follow the provided JSON schema and ensure all information is complete before formatting.
        If any information is missing, ask the user for it before proceeding.
        
        The JSON schema should follow this structure:
         {{
                "personal_info": {{
                    "name": "string",
                    "email": "string",
                    "phone": "string",
                    "location": "string"
                }},
                "summary": "string",
                "experience": [{{
                    "company": "string",
                    "title": "string",
                    "start_date": "string",
                    "end_date": "string",
                    "responsibilities": "string"
                }}],
                "education": [{{
                    "institution": "string",
                    "degree": "string",
                    "graduation_date": "string"
                }}],
                "skills": ["string"]
            }}
        """
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

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
            """Convert JSON resume to LaTeX format"""
            try:
                if isinstance(json_str, str):
                    resume_data = json.loads(json_str)
                else:
                    resume_data = json_str
                # Validate the data using the Resume model
                resume = Resume(**resume_data)
                return json_to_latex_resume(resume.model_dump())
            except Exception as e:
                return f"Error converting to LaTeX: {str(e)}"

        @tool
        def optimize_resume_for_ats(text: str) -> str:
            """Optimize resume content for ATS systems"""
            return text

        tools = [format_resume_json, convert_to_latex, optimize_resume_for_ats]
        
        agent = create_openai_functions_agent(
            llm=model,
            tools=tools,
            prompt=prompt,
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        response = agent_executor.invoke({
            "input": query,
            "chat_history": messages
        })
        
        return Response(content=response["output"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))