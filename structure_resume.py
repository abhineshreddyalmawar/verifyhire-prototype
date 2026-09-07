import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

from extract_text import extract_text_from_pdf

resume_tool = {
    "name": "extract_resume_data",
    "description": "Extract structured candidate information from a resume text",
    "input_schema": {
        "type": "object",
        "properties": {
            "full_name": {"type": "string"},
            "email": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "summary": {"type": ["string", "null"]},
            "job_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "employer": {"type": "string"},
                        "start_date": {"type": "string", "description": "Format as 'Month YYYY', e.g. 'March 2021'. Convert other formats like '03/2021' to this."},
                        "end_date": {"type": ["string", "null"], "description": "Format as 'Month YYYY', e.g. 'March 2021'. Convert other formats like '03/2021' to this. Null if current/present."}
                    },
                    "required": ["title", "employer", "start_date"]
                }
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "degree": {"type": "string"},
                        "institution": {"type": "string"},
                        "graduation_date": {"type": ["string", "null"], "description": "Format as 'Month YYYY' if the month is known. If only a year is given, use 'January YYYY' as a placeholder month. Null only if no date is given at all."}
                    },
                    "required": ["degree", "institution"]
                }
            }
        },
        "required": ["full_name", "email", "phone", "skills", "job_history", "summary", "education"]
    }
}

def structure_resume(resume_text):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[resume_tool],
        tool_choice={"type": "tool", "name": "extract_resume_data"},
        messages=[
            {"role": "user", "content": f"Extract structured data from this resume:\n\n{resume_text}"}
        ]
    )
    return response.content[0].input


if __name__ == "__main__":
    resume_text = extract_text_from_pdf("sample_resume_jordan_lee.pdf")
    print(structure_resume(resume_text))