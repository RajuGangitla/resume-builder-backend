from typing import Type, Optional, List
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from resume_builder import Resume

class PersonalInformationSchema(BaseModel):
    name: Optional[str] = Field("", description="Full name of the person")
    email: Optional[str] = Field("", description="Email address")
    phone: Optional[str] = Field("", description="Phone number")
    github: Optional[str] = Field("", description="GitHub profile link")
    linkedin: Optional[str] = Field("", description="LinkedIn profile link")

class PersonalInformation(BaseTool):
    name: str = "AddPersonalInformation"
    description: str = "Use this tool to add personal information of the user to the resume."
    args_schema: Type[BaseModel] = PersonalInformationSchema
    return_direct: bool = False
    resume: Resume = None

    def _run(self, name: Optional[str] = "", email: Optional[str] = "", phone: Optional[str] = "", 
             github: Optional[str] = "", linkedin: Optional[str] = "",
             run_manager: Optional[CallbackManagerForToolRun] = None):
        self.resume.resume_data["personal_section"] = {
            "name": name or "",
            "email": email or "",
            "phone": phone or "",
            "github": github or "",
            "linkedin": linkedin or ""
        }
        return {"output": "Successfully added personal information"}

class ExperienceSchema(BaseModel):
    company: Optional[str] = Field("", description="Company name")
    job_title: Optional[str] = Field("", description="Job title")
    start_date: Optional[str] = Field("", description="Start date of employment (YYYY-MM)")
    end_date: Optional[str] = Field(None, description="End date of employment (YYYY-MM), None if still employed")
    job_type: Optional[str] = Field("Not Specified", description="Job Type (Remote, On-site, Hybrid)")
    responsibilities: Optional[List[str]] = Field([], description="List of job responsibilities")

class ExperienceTool(BaseTool):
    name: str = "AddExperience"
    description: str = "Tool to capture work experience details."
    args_schema: Type[BaseModel] = ExperienceSchema
    return_direct: bool = False
    resume: Resume = None

    def _run(self, company: Optional[str] = "", job_title: Optional[str] = "", start_date: Optional[str] = "",
             end_date: Optional[str] = None, job_type: Optional[str] = "Not Specified", 
             responsibilities: Optional[List[str]] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        print("experience started")
        responsibilities = responsibilities or []
        print(self.resume.resume_data)
        existing_data = self.resume.resume_data["experience_section"]
        print({
            "job_title": job_title or "",
            "company": company or "",
            "start_date": start_date or "",
            "end_date": end_date,
            "job_type": job_type or "Not Specified",
            "responsibilities": responsibilities
        })
        existing_data.append({
            "job_title": job_title or "",
            "company": company or "",
            "start_date": start_date or "",
            "end_date": end_date,
            "job_type": job_type or "Not Specified",
            "responsibilities": responsibilities
        })
        print("experiece ended")
        self.resume.resume_data["experience_section"] = existing_data
        return {"output": "Successfully added experience."}

class EducationSchema(BaseModel):
    institution: Optional[str] = Field("", description="Educational institution name")
    location: Optional[str] = Field(None, description="Location of the institution")
    degree: Optional[str] = Field("", description="Degree obtained")
    graduation_date: Optional[str] = Field("", description="Graduation date (YYYY-MM)")

class EducationTool(BaseTool):
    name: str = "AddEducation"
    description: str = "Tool to capture latest education details."
    args_schema: Type[BaseModel] = EducationSchema
    return_direct: bool = False
    resume: Resume = None

    def _run(self, institution: Optional[str] = "", degree: Optional[str] = "", 
             graduation_date: Optional[str] = "", location: Optional[str] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        self.resume.resume_data["education_section"] = {
            "degree": degree or "",
            "institution": institution or "",
            "graduation_date": graduation_date or "",
            "location": location
        }
        return {"output": "Successfully added education."}

class ProjectsSchema(BaseModel):
    title: Optional[str] = Field("", description="Title of the project")
    tech_stack: Optional[List[str]] = Field([], description="Technologies used in this project")
    features: Optional[List[str]] = Field([], description="List of key features in the project")
    duration: Optional[str] = Field("", description="Duration of the project (e.g., 3 months)")

class ProjectsTool(BaseTool):
    name: str = "AddProjects"
    description: str = "Tool to capture project details."
    args_schema: Type[BaseModel] = ProjectsSchema
    return_direct: bool = False
    resume: Resume = None

    def _run(self, title: Optional[str] = "", tech_stack: Optional[List[str]] = None, 
             features: Optional[List[str]] = None, duration: Optional[str] = "",
             run_manager: Optional[CallbackManagerForToolRun] = None):
        tech_stack = tech_stack or []
        features = features or []
        existing_data = self.resume.resume_data.get("projects_section", [])
        existing_data.append({
            "title": title or "",
            "tech_stack": tech_stack,
            "features": features,
            "duration": duration or ""
        })
        self.resume.resume_data["projects_section"] = existing_data
        return {"output": "Successfully added project details."}

class SkillsSchema(BaseModel):
    languages: Optional[List[str]] = Field([], description="List of programming languages")
    frameworks: Optional[List[str]] = Field([], description="List of frameworks")
    developer_tools: Optional[List[str]] = Field([], description="List of developer tools")
    libraries: Optional[List[str]] = Field([], description="List of libraries")

class SkillsTool(BaseTool):
    name: str = "AddSkills"
    description: str = "Tool to capture user skills."
    args_schema: Type[BaseModel] = SkillsSchema
    return_direct: bool = False
    resume: Resume = None

    def _run(self, languages: Optional[List[str]] = None, frameworks: Optional[List[str]] = None, 
             developer_tools: Optional[List[str]] = None, libraries: Optional[List[str]] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        languages = languages or []
        frameworks = frameworks or []
        developer_tools = developer_tools or []
        libraries = libraries or []
        self.resume.resume_data["skills_section"] = {
            "languages": languages,
            "frameworks": frameworks,
            "developer_tools": developer_tools,
            "libraries": libraries
        }
        return {"output": "Successfully added skills."}