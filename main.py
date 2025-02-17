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

# def safe_get(data: dict, *keys, default="") -> str:
#     """Safely get nested dictionary values, returning default if not found"""
#     current = data
#     for key in keys:
#         if not isinstance(current, dict):
#             return default
#         current = current.get(key, default)
#     return current if current is not None else default

# def escape_latex(text: str) -> str:
#     """Escape special LaTeX characters"""
#     chars = {
#         '&': '\\&',
#         '%': '\\%',
#         '$': '\\$',
#         '#': '\\#',
#         '_': '\\_',
#         '{': '\\{',
#         '}': '\\}',
#         '~': '\\textasciitilde{}',
#         '^': '\\textasciicircum{}',
#         '\\': '\\textbackslash{}'
#     }
#     return ''.join(chars.get(c, c) for c in str(text))

# def format_text_with_bullets(text: str) -> str:
#     """Format text into LaTeX bullet points"""
#     if not text:
#         return ""
#     lines = text.split('\n') if '\n' in text else [text]
#     formatted_lines = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         line = line.lstrip('•-*').strip()
#         formatted_lines.append(f"    \\item {escape_latex(line)}")
#     return '\n'.join(formatted_lines)

# def json_to_latex_resume(json_data: dict) -> str:
#     """Convert JSON resume data to LaTeX using the custom resume class"""
#     try:
#         latex_content = []
        
#         # Document setup
#         latex_content.extend([
#             "\\documentclass{resume}",
#             "\\begin{document}",
#             ""
#         ])
        
#         # Personal Information
#         personal = json_data.get('personal_info', {})
#         if name := safe_get(personal, 'name'):
#             latex_content.append(f"\\name{{{escape_latex(name)}}}")
        
#         # Address fields
#         address_fields = [
#             safe_get(personal, 'email'),
#             safe_get(personal, 'phone'),
#             safe_get(personal, 'location')
#         ]
#         for field in address_fields:
#             if field:
#                 latex_content.append(f"\\address{{{escape_latex(field)}}}")
#         latex_content.append("")
        
#         # Sections
#         if summary := safe_get(json_data, 'summary'):
#             latex_content.extend([
#                 "\\begin{rSection}{Summary}",
#                 escape_latex(summary),
#                 "\\end{rSection}",
#                 ""
#             ])
        
#         # Experience section
#         if experience := json_data.get('experience', []):
#             latex_content.append("\\begin{rSection}{Experience}")
#             for exp in experience:
#                 if company := safe_get(exp, 'company'):
#                     latex_content.extend([
#                         "\\begin{rSubsection}",
#                         f"    {{{escape_latex(company)}}}",
#                         f"    {{{escape_latex(safe_get(exp, 'start_date', default=''))} -- {escape_latex(safe_get(exp, 'end_date', default='Present'))}}}",
#                         f"    {{{escape_latex(safe_get(exp, 'title'))}}}",
#                         f"    {{{escape_latex(safe_get(exp, 'location'))}}}",
#                     ])
#                     if responsibilities := safe_get(exp, 'responsibilities'):
#                         latex_content.append(format_text_with_bullets(responsibilities))
#                     latex_content.extend([
#                         "\\end{rSubsection}",
#                         ""
#                     ])
#             latex_content.append("\\end{rSection}\n")
        
#         # Education section
#         if education := json_data.get('education', []):
#             latex_content.append("\\begin{rSection}{Education}")
#             for edu in education:
#                 if institution := safe_get(edu, 'institution'):
#                     latex_content.extend([
#                         "\\begin{rSubsection}",
#                         f"    {{{escape_latex(institution)}}}",
#                         f"    {{{escape_latex(safe_get(edu, 'graduation_date'))}}}",
#                         f"    {{{escape_latex(safe_get(edu, 'degree'))}}}",
#                         f"    {{}}",
#                         "\\end{rSubsection}",
#                         ""
#                     ])
#             latex_content.append("\\end{rSection}\n")
        
#         # Skills section
#         if skills := json_data.get('skills', []):
#             latex_content.extend([
#                 "\\begin{rSection}{Skills}",
#                 " \\textbullet\\ " + " \\textbullet\\ ".join(escape_latex(skill) for skill in skills if skill),
#                 "\\end{rSection}",
#                 ""
#             ])
        
#         latex_content.append("\\end{document}")
#         return "\n".join(latex_content)
    
#     except Exception as e:
#         return f"% Error generating LaTeX: {str(e)}\n\\documentclass{resume}\\begin{document}\n\\end{document}"


# @app.post("/chat")
# async def read_root(request: Request):
#     try:
#         data = await request.json()
#         query = data.get("query")
#         messages = data.get("messages")
#         model = AzureChatOpenAI(model="gpt-4o", api_key=os.getenv("OPEN_AI_KEY"), api_version=os.getenv("OPENAI_API_VERSION"))

#         # Define the system message content directly
#         system_message = """
#             You are an expert resume builder assistant. Your role is to:
#             1. Systematically collect all required information from the user through a series of questions
#             2. Ensure you gather complete details for each section:
#             - Personal Information (name, email, phone, location)
#             - Professional Summary
#             - Work Experience (company, title, dates, responsibilities)
#             - Education (institution, degree, dates)
#             - Skills
#             - Certifications (optional)
#             3. Once all information is collected, use the format_resume_json tool to create a structured JSON
#             4. Validate that all required fields are filled before formatting
#             5. Guide the user through any missing or incomplete information
            
#             Required Information Structure:
#             {{
#                 "personal_info": {{
#                     "name": "string",
#                     "email": "string",
#                     "phone": "string",
#                     "location": "string"
#                 }},
#                 "summary": "string",
#                 "experience": [{{
#                     "company": "string",
#                     "title": "string",
#                     "start_date": "string",
#                     "end_date": "string",
#                     "responsibilities": "string"
#                 }}],
#                 "education": [{{
#                     "institution": "string",
#                     "degree": "string",
#                     "graduation_date": "string"
#                 }}],
#                 "skills": ["string"]
#             }}
            
#             Always maintain a professional tone and ensure all required information is collected before formatting the JSON.
#         """
        
#         memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True
#         )  

#       # Create the prompt template with all required variables
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", system_message),
#             MessagesPlaceholder("chat_history", optional=True),
#             ("human", "{input}"),
#             MessagesPlaceholder(variable_name='agent_scratchpad')
#         ])

#         # print(memory.load_memory_variables({}))
        
#         @tool
#         def magic_function(input: int) -> int:
#             """Applies a magic function to an input."""
#             return input + 2

#         tools = [magic_function]
    
        
#         # Create the agent with all required parameters
#         agent = create_openai_functions_agent(
#             llm=model,
#             tools=tools,
#             prompt=prompt,
#         )
        
#         # Create the agent executor
#         agent_executor = AgentExecutor(
#             agent=agent,
#             tools=tools,
#             verbose=True,
#             handle_parsing_errors=True
#         )

#         print(messages, 'messages')
        
#         # Run the agent with the required input
#         response = agent_executor.invoke({
#             "input": query,
#             "chat_history":messages
#         })  
        
#         # Return the response
#         return Response(content=response["output"])
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



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
    """Convert JSON resume data to LaTeX using the custom resume class"""
    try:
        latex_content = []
        
        # Document setup
        latex_content.extend([
            "\\documentclass{resume}",
            "\\begin{document}",
            ""
        ])
        
        # Personal Information
        personal = json_data.get('personal_info', {})
        if name := safe_get(personal, 'name'):
            latex_content.append(f"\\name{{{escape_latex(name)}}}")
        
        # Address fields
        address_fields = [
            safe_get(personal, 'email'),
            safe_get(personal, 'phone'),
            safe_get(personal, 'location')
        ]
        for field in address_fields:
            if field:
                latex_content.append(f"\\address{{{escape_latex(field)}}}")
        latex_content.append("")
        
        # Sections
        if summary := safe_get(json_data, 'summary'):
            latex_content.extend([
                "\\begin{rSection}{Summary}",
                escape_latex(summary),
                "\\end{rSection}",
                ""
            ])
        
        # Experience section
        if experience := json_data.get('experience', []):
            latex_content.append("\\begin{rSection}{Experience}")
            for exp in experience:
                if company := safe_get(exp, 'company'):
                    latex_content.extend([
                        "\\begin{rSubsection}",
                        f"    {{{escape_latex(company)}}}",
                        f"    {{{escape_latex(safe_get(exp, 'start_date', default=''))} -- {escape_latex(safe_get(exp, 'end_date', default='Present'))}}}",
                        f"    {{{escape_latex(safe_get(exp, 'title'))}}}",
                        f"    {{{escape_latex(safe_get(exp, 'location'))}}}",
                    ])
                    if responsibilities := safe_get(exp, 'responsibilities'):
                        latex_content.append(format_text_with_bullets(responsibilities))
                    latex_content.extend([
                        "\\end{rSubsection}",
                        ""
                    ])
            latex_content.append("\\end{rSection}\n")
        
        # Education section
        if education := json_data.get('education', []):
            latex_content.append("\\begin{rSection}{Education}")
            for edu in education:
                if institution := safe_get(edu, 'institution'):
                    latex_content.extend([
                        "\\begin{rSubsection}",
                        f"    {{{escape_latex(institution)}}}",
                        f"    {{{escape_latex(safe_get(edu, 'graduation_date'))}}}",
                        f"    {{{escape_latex(safe_get(edu, 'degree'))}}}",
                        f"    {{}}",
                        "\\end{rSubsection}",
                        ""
                    ])
            latex_content.append("\\end{rSection}\n")
        
        # Skills section
        if skills := json_data.get('skills', []):
            latex_content.extend([
                "\\begin{rSection}{Skills}",
                " \\textbullet\\ " + " \\textbullet\\ ".join(escape_latex(skill) for skill in skills if skill),
                "\\end{rSection}",
                ""
            ])
        
        latex_content.append("\\end{document}")
        return "\n".join(latex_content)
    
    except Exception as e:
        return f"% Error generating LaTeX:="

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