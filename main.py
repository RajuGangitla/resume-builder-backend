import os
from fastapi import FastAPI, Request, Response, HTTPException
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory


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


@app.post("/chat")
async def read_root(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        messages = data.get("messages", [])
        model = AzureChatOpenAI(model="gpt-4o", api_key=os.getenv("OPEN_AI_KEY"), api_version=os.getenv("OPENAI_API_VERSION"))
        memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )  
              # Define the system message content directly
        system_message = """
            You are an expert resume builder assistant. Your role is to:
            1. Systematically collect all required information from the user through a series of questions
            2. Ensure you gather complete details for each section:
            - Personal Information (name, email, phone, location)
            - Professional Summary
            - Work Experience (company, title, dates, responsibilities)
            - Education (institution, degree, dates)
            - Skills
            - Certifications (optional)
            3. Once all information is collected, use the format_resume_json tool to create a structured JSON
            4. Validate that all required fields are filled before formatting
            5. Guide the user through any missing or incomplete information
            
            Required Information Structure:
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
            
            Always maintain a professional tone and ensure all required information is collected before formatting the JSON.
        """
        
      # Create the prompt template with all required variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Think about this step-by-step:"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Add the required tool names
        tools = []
        tool_names = [tool.name for tool in tools]
        
        # Create the agent with all required parameters
        agent = create_react_agent(
            llm=model,
            tools=tools,
            prompt=prompt
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Run the agent with the required input
        response = agent_executor.invoke({
            "input": query,
            "tool_names": tool_names,
            "tools": tools,
            "agent_scratchpad": ""
        })
        
        # Return the response
        return Response(content=response["output"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))