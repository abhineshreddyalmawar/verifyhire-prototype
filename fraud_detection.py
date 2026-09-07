import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

from datetime import datetime

def parse_date(date_str):
    if date_str is None:
        return datetime.max
    try:
        return datetime.strptime(date_str, "%B %Y")
    except (ValueError, TypeError):
        return None

def has_valid_start_date(job):
    return parse_date(job["start_date"]) is not None

def months_between(date_a, date_b):
    delta = date_b - date_a
    return delta.days / 30.44

def check_title_progression(job_history):
    sorted_jobs = sorted(job_history, key=lambda job: parse_date(job["start_date"]))
    progression_data = []
    for i in range(len(sorted_jobs) - 1):
        job_a = sorted_jobs[i]
        job_b = sorted_jobs[i + 1]
        gap_months = months_between(parse_date(job_a["start_date"]), parse_date(job_b["start_date"]))
        progression_data.append({
            "from_title": job_a["title"],
            "to_title": job_b["title"],
            "months_between": round(gap_months, 1)
        })
    return progression_data


def total_career_years(job_history):
    job_history = [job for job in job_history if has_valid_start_date(job)]
    if not job_history:
        return None
    earliest_start = min(parse_date(job["start_date"]) for job in job_history)
    latest_end = max(
        datetime.now() if job["end_date"] is None else parse_date(job["end_date"])
        for job in job_history
    )
    return (latest_end - earliest_start).days / 365.25

def jobs_overlap(job_a, job_b):
    a_start = parse_date(job_a["start_date"])
    a_end = parse_date(job_a["end_date"])
    b_start = parse_date(job_b["start_date"])
    b_end = parse_date(job_b["end_date"])

    return a_start < b_end and b_start < a_end

def check_all_overlaps(job_history):
    job_history = [job for job in job_history if has_valid_start_date(job)]
    overlapping_pairs = []
    for i in range(len(job_history)):
        for j in range(i + 1, len(job_history)):
            if jobs_overlap(job_history[i], job_history[j]):
                overlapping_pairs.append((job_history[i], job_history[j]))
    return overlapping_pairs

tech_stack_tool = {
    "name": "check_tech_stack_plausibility",
    "description": "Check whether claimed technologies are plausible given job dates and career duration.",
    "input_schema": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "issue_type": {
                            "type": "string",
                            "enum": ["technology_predates_release", "implausible_duration"]
                        },
                        "explanation": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        }
                    },
                    "required": ["skill", "issue_type", "explanation", "confidence"]
                }
            }
        },
        "required": ["flags"]
    }
}

def check_tech_stack_plausibility(skills, job_history, summary, career_years):
    if career_years is None:
        career_context = "Career span could not be determined (no job history extracted) — do not attempt to judge claimed years of experience."
    else:
        career_context = f"Total career span (already calculated): {career_years:.1f} years"

    prompt = f"""Given this candidate's claimed skills, summary, and job history, identify any implausible claims.

Skills: {skills}
Summary: {summary}
Job History: {job_history}
{career_context}

Check for:
1. A technology claimed that didn't exist yet at the time of a job (technology_predates_release) — only flag if the technology could NOT plausibly have been used at ANY point in the candidate's career. If it could plausibly have been used later in their career (even if not in their earliest roles), do NOT create a flag entry.
2. A claimed duration in the summary or skills that EXCEEDS the career span (implausible_duration) — only flag OVERSTATED experience. If the claimed duration is EQUAL TO or LESS THAN the actual career span, this is not an issue at all — do not create a flag entry for it, do not mention it, treat it exactly as if it were never noticed.

Only flag genuine issues. If everything is plausible, return an empty flags list."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[tech_stack_tool],
        tool_choice={"type": "tool", "name": "check_tech_stack_plausibility"},
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].input

title_progression_tool = {
    "name": "check_title_progression",
    "description": "Flag unusually fast title progressions given precomputed month gaps between roles.",
    "input_schema": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_title": {"type": "string"},
                        "to_title": {"type": "string"},
                        "months_between": {"type": "number"},
                        "explanation": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        }
                    },
                    "required": ["from_title", "to_title", "months_between", "explanation", "confidence"]
                }
            }
        },
        "required": ["flags"]
    }
}
def check_title_progression_flags(job_history):
    job_history = [job for job in job_history if has_valid_start_date(job)]
    progression_data = check_title_progression(job_history)

    prompt = f"""Given this sequence of consecutive job title transitions with precomputed month gaps, flag any that represent an unusually fast or implausible progression.

Progression data: {progression_data}

A fast promotion is not automatically suspicious — context like company size or exceptional performance can explain it. Only flag transitions that would genuinely warrant a closer look, and explain your reasoning for the confidence level you assign. If you conclude a transition does NOT warrant scrutiny, do NOT create a flag entry for it at all — do not include it "for completeness" or as a non-issue.

If everything looks reasonable, return an empty flags list."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[title_progression_tool],
        tool_choice={"type": "tool", "name": "check_title_progression"},
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].input

def generate_fraud_scorecard(resume_data):
    job_history = resume_data["job_history"]
    career_years = total_career_years(job_history)

    overlaps = check_all_overlaps(job_history)
    tech_flags = check_tech_stack_plausibility(
        skills=resume_data["skills"],
        job_history=job_history,
        summary=resume_data["summary"],
        career_years=career_years
    )["flags"]
    title_flags = check_title_progression_flags(job_history)["flags"] if len(job_history) >= 2 else []

    high_confidence_flags = [f for f in tech_flags + title_flags if f["confidence"] == "high"]
    overall_risk = "review_recommended" if (overlaps or high_confidence_flags) else "clear"

    return {
        "candidate": resume_data["full_name"],
        "overall_risk": overall_risk,
        "career_years_known": career_years is not None,
        "timeline_overlaps": overlaps,
        "tech_stack_flags": tech_flags,
        "title_progression_flags": title_flags
    }


if __name__ == "__main__":
    jordan_jobs = [
        {"title": "Senior Software Engineer", "employer": "Brightpath Systems", "start_date": "June 2022", "end_date": None},
        {"title": "Software Engineer", "employer": "Alden Data Co.", "start_date": "August 2019", "end_date": "May 2022"},
        {"title": "Junior Developer", "employer": "Alden Data Co.", "start_date": "July 2018", "end_date": "August 2019"}
    ]
    print("Jordan overlaps (expect []):", check_all_overlaps(jordan_jobs))

    test_history_with_overlap = [
        {"title": "Role 1", "employer": "Company X", "start_date": "January 2020", "end_date": "December 2020"},
        {"title": "Role 2", "employer": "Company Y", "start_date": "June 2020", "end_date": "May 2021"},
        {"title": "Role 3", "employer": "Company Z", "start_date": "June 2021", "end_date": None}
    ]
    print("Test overlaps (expect 1 pair, Role 1 vs Role 2):", check_all_overlaps(test_history_with_overlap))

    single_job_history = [
        {"title": "Only Role", "employer": "Company Q", "start_date": "January 2022", "end_date": None}
    ]
    print("Single job overlaps (expect []):", check_all_overlaps(single_job_history))

    double_overlap_history = [
        {"title": "Role A", "employer": "Company 1", "start_date": "January 2020", "end_date": "December 2020"},
        {"title": "Role B", "employer": "Company 2", "start_date": "September 2020", "end_date": "August 2021"},
        {"title": "Role C", "employer": "Company 3", "start_date": "June 2021", "end_date": None}
    ]
    print("Double overlap (expect 2 pairs, A-B and B-C):", check_all_overlaps(double_overlap_history))

    print("\nJordan tech stack check (expect []):")
    print(check_tech_stack_plausibility(
        skills=["Python", "Java", "Flask", "Spring Boot", "AWS (Lambda, EC2, S3)", "Docker", "Jenkins", "PostgreSQL", "REST APIs", "Git", "Selenium", "pytest"],
        job_history=jordan_jobs,
        summary="Backend-focused software engineer with 5 years of experience building REST APIs and cloud-deployed services. Comfortable across Python, Java, and AWS infrastructure.",
        career_years=total_career_years(jordan_jobs)
    ))

    print("\nSuspicious tech stack check (expect implausible_duration, high confidence):")
    suspicious_jobs = [
        {"title": "Software Engineer", "employer": "StartupCo", "start_date": "January 2024", "end_date": None}
    ]
    print(check_tech_stack_plausibility(
        skills=["Python"],
        job_history=suspicious_jobs,
        summary="Software engineer with 10 years of experience in Python development.",
        career_years=total_career_years(suspicious_jobs)
    ))

    print("\nRelease-date check, well-known tech (expect technology_predates_release):")
    old_tech_jobs = [
        {"title": "Software Engineer", "employer": "Old Corp", "start_date": "January 2010", "end_date": "December 2012"}
    ]
    print(check_tech_stack_plausibility(
        skills=["Kubernetes"],
        job_history=old_tech_jobs,
        summary="Software engineer with experience in container orchestration.",
        career_years=total_career_years(old_tech_jobs)
    ))

    print("\nRelease-date check, lesser-known tech (expect technology_predates_release; confidence field known to be unverified/self-reported, not calibrated):")
    obscure_tech_jobs = [
        {"title": "Backend Developer", "employer": "Legacy Systems Inc", "start_date": "January 2019", "end_date": "December 2021"}
    ]
    print(check_tech_stack_plausibility(
        skills=["Bun"],
        job_history=obscure_tech_jobs,
        summary="Backend developer with experience in modern JavaScript tooling.",
        career_years=total_career_years(obscure_tech_jobs)
    ))

    print("\nFast promotion check (expect flagged, high confidence):")
    fast_promotion_jobs = [
        {"title": "Junior Developer", "employer": "Company A", "start_date": "January 2023", "end_date": "March 2023"},
        {"title": "Senior Software Engineer", "employer": "Company A", "start_date": "March 2023", "end_date": None}
    ]
    print(check_title_progression_flags(fast_promotion_jobs))

    print("\nJordan full scorecard (expect overall_risk: clear):")
    jordan_resume_data = {
        "full_name": "Jordan Lee",
        "summary": "Backend-focused software engineer with 5 years of experience building REST APIs and cloud-deployed services. Comfortable across Python, Java, and AWS infrastructure.",
        "skills": ["Python", "Java", "Flask", "Spring Boot", "AWS (Lambda, EC2, S3)", "Docker", "Jenkins", "PostgreSQL", "REST APIs", "Git", "Selenium", "pytest"],
        "job_history": jordan_jobs
    }
    print(generate_fraud_scorecard(jordan_resume_data))

    print("\nFast promotion candidate scorecard (expect overall_risk: review_recommended):")
    fast_promotion_resume = {
        "full_name": "Test Candidate",
        "summary": "Experienced engineer.",
        "skills": ["Python"],
        "job_history": fast_promotion_jobs
    }
    print(generate_fraud_scorecard(fast_promotion_resume))