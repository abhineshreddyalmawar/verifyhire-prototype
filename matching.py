import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

job_description_tool = {
    "name": "structure_job_description",
    "description": "Extract structured requirements from a job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "required_skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "preferred_skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "min_experience_years": {"type": ["number", "null"]},
            "keywords": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["required_skills", "preferred_skills", "keywords"]
    }
}

def structure_job_description(job_text):
    prompt = f"""Extract structured hiring requirements from this job description.

Job description text:
{job_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[job_description_tool],
        tool_choice={"type": "tool", "name": "structure_job_description"},
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].input


matching_tool = {
    "name": "score_candidate_match",
    "description": "Score how well a candidate's resume matches a job description across four categories, with reasoning per category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "required_skills_match": {"type": "number"},
            "required_skills_explanation": {"type": "string"},
            "preferred_skills_match": {"type": "number"},
            "preferred_skills_explanation": {"type": "string"},
            "experience_match": {"type": "number"},
            "experience_explanation": {"type": "string"},
            "keyword_match": {"type": "number"},
            "keyword_explanation": {"type": "string"}
        },
        "required": [
            "required_skills_match", "required_skills_explanation",
            "preferred_skills_match", "preferred_skills_explanation",
            "experience_match", "experience_explanation",
            "keyword_match", "keyword_explanation"
        ]
    }
}

def compute_overall_score(scores):
    return (
        scores["required_skills_match"] * 0.50 +
        scores["preferred_skills_match"] * 0.25 +
        scores["experience_match"] * 0.15 +
        scores["keyword_match"] * 0.10
    )

def score_candidate_match(resume_data, job_description):
    prompt = f"""Score how well this candidate matches the job, across four categories, each 0-100.

Candidate resume:
Skills: {resume_data['skills']}
Summary: {resume_data['summary']}
Job history: {resume_data['job_history']}
Education: {resume_data['education']}

Job requirements:
Required skills: {job_description['required_skills']}
Preferred skills: {job_description['preferred_skills']}
Minimum experience (years): {job_description['min_experience_years']}
Keywords: {job_description['keywords']}

Score:
- required_skills_match: how well the candidate's skills cover the required list
- preferred_skills_match: how well the candidate's skills, and any relevant education (e.g. a degree explicitly listed as preferred), cover the preferred list
- experience_match: whether the candidate's career length meets the minimum (if stated)
- keyword_match: how many of the job's keywords appear in the candidate's resume

Explain your reasoning."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[matching_tool],
        tool_choice={"type": "tool", "name": "score_candidate_match"},
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].input


if __name__ == "__main__":
    sample_jd_text = """
    We're looking for a Backend Software Engineer to join our growing team.

    Requirements:
    - 3+ years of experience with Python
    - Strong knowledge of Flask or Django
    - Experience with AWS (EC2, Lambda, S3)
    - Familiarity with PostgreSQL or similar relational databases

    Nice to have:
    - Experience with Docker and containerization
    - Familiarity with CI/CD pipelines (Jenkins, GitHub Actions)
    - Prior experience in a fast-paced startup environment

    We value collaboration, clean code, and a strong sense of ownership over your work.
    """
    print("\nStructured JD:", structure_job_description(sample_jd_text))

    jordan_resume_data = {
        "full_name": "Jordan Lee",
        "summary": "Backend-focused software engineer with 5 years of experience building REST APIs and cloud-deployed services. Comfortable across Python, Java, and AWS infrastructure.",
        "skills": ["Python", "Java", "Flask", "Spring Boot", "AWS (Lambda, EC2, S3)", "Docker", "Jenkins", "PostgreSQL", "REST APIs", "Git", "Selenium", "pytest"],
        "job_history": [
            {"title": "Senior Software Engineer", "employer": "Brightpath Systems", "start_date": "June 2022", "end_date": None},
            {"title": "Software Engineer", "employer": "Alden Data Co.", "start_date": "August 2019", "end_date": "May 2022"},
            {"title": "Junior Developer", "employer": "Alden Data Co.", "start_date": "July 2018", "end_date": "August 2019"}
        ]
    }

    structured_jd = structure_job_description(sample_jd_text)
    scores = score_candidate_match(jordan_resume_data, structured_jd)
    print("\nJordan match scores:", scores)
    print("Overall score:", compute_overall_score(scores))

    weak_candidate_data = {
        "full_name": "Casey Rivera",
        "summary": "Frontend developer with 1 year of experience building user interfaces.",
        "skills": ["JavaScript", "React", "CSS", "HTML"],
        "job_history": [
            {"title": "Junior Frontend Developer", "employer": "PixelCraft Studio", "start_date": "August 2024", "end_date": None}
        ]
    }

    weak_scores = score_candidate_match(weak_candidate_data, structured_jd)
    print("\nWeak candidate scores:", weak_scores)
    print("Weak overall score:", compute_overall_score(weak_scores))

    ambiguous_candidate_data = {
        "full_name": "Morgan Ellis",
        "summary": "Backend developer with 3 years of experience building web applications using Python and Django.",
        "skills": ["Python", "Django", "MySQL", "Docker", "Git", "Unit Testing"],
        "job_history": [
            {"title": "Backend Developer", "employer": "Fieldstone Software", "start_date": "September 2022", "end_date": None}
        ]
    }

    ambiguous_scores = score_candidate_match(ambiguous_candidate_data, structured_jd)
    print("\nAmbiguous candidate scores:", ambiguous_scores)
    print("Ambiguous overall score:", compute_overall_score(ambiguous_scores))