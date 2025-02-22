from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
import os
from pydantic import BaseModel, Field
from typing import Dict, List, Union
from customstore import CustomChatMessageHistory
from tools import PersonalInformation,  ExperienceTool, EducationTool, SkillsTool, ProjectsTool
import subprocess
import logging


load_dotenv()
app=FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ResumeResponse(BaseModel):
    content: str
    resume_data: dict

class ChatRequest(BaseModel):
    query: str
    session_id:str

class ResumeConversion(BaseModel):
    resume_data:dict
    

@app.post("/chat")
async def read_root(request: ChatRequest):
    try:
        query = request.query
        session_id = request.session_id
        model = AzureChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPEN_AI_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION")
        )

        custom_history = CustomChatMessageHistory(session_id=session_id, ttl=2)
        store = custom_history.store
        resume_object = store.get_resume(session_id)
        
        system_message = """
You are an AI-powered resume builder assistant. Your role is to:

1. Collect all required information from the user through a series of questions.
2. Ensure completeness and ask for missing details when necessary.  
3. Format responsibilities and project details into structured bullet points using the `format_responsibilities` tool.  

Resume structure (for reference):

{{
    "personal_info": {{
        "name": "string",
        "email": "string",
        "phone": "string",
        "github": "string",
        "linkedin": "string"
    }},
    "experience": [{{
        "company": "string",
        "title": "string",
        "start_date": "string",
        "end_date": "string or null",
        "job_type": "string",
        "responsibilities": ["string"]
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
        "features": ["string"],
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

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=custom_history, 
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

                Text to transform:  
                {text}
                
                **Return only the bullet points, one per line, starting with '- '**
                """)
                result = model.invoke(format_large_prompt.format(text=text))
                return result.content.split("\n")
            except Exception as e:
                return [f"Error breaking into bullet points: {str(e)}"]

        tools = [
            PersonalInformation(resume=resume_object),
            ExperienceTool(resume=resume_object),
            EducationTool(resume=resume_object),
            SkillsTool(resume=resume_object),
            ProjectsTool(resume=resume_object),
            format_responsibilities
        ]
        
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
        
        response = agent_executor.invoke({"input": query})

        # Save updated resume data back to Redis
        store.update_resume(session_id, resume_object.resume_data)
        print("Stored Messages:", [str(msg) for msg in custom_history.messages])
        print("Raw Stored Data:", store.get_messages(session_id))      
        try:
            return {"content": response['output'], "resume_data": resume_object.resume_data}
        except json.JSONDecodeError:
            return {"content": response['output'], "resume_data": resume_object.resume_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    temp_dir = tempfile.mkdtemp()  # Temporary directory for files
    try:
        latex_content = generate_latex_content(request.resume_data)
        tex_file = os.path.join(temp_dir, "resume.tex")
        pdf_file = os.path.join(temp_dir, "resume.pdf")
        
        logger.info("Writing LaTeX content to %s", tex_file)
        with open(tex_file, "w") as f:
            f.write(latex_content)
        
        logger.info("Compiling LaTeX to PDF")
        result = subprocess.run(
            ["pdflatex", "-output-directory", temp_dir, tex_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30  # 30-second timeout
        )
        
        if not os.path.exists(pdf_file):
            raise Exception(f"PDF file not generated: {result.stderr.decode()}")
        
        logger.info("Reading PDF content")
        with open(pdf_file, "rb") as f:
            pdf_content = f.read()
        
        return {
            "message": "PDF generated successfully",
            "pdf": pdf_content.hex()
        }
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail=f"PDF generation timed out: {e.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("Cleaning up temporary directory %s", temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)