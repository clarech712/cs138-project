import os
from typing import List, Dict, Any
from model_manager import get_model

class SkillExtractor:
    """A class to extract skills from job descriptions using Google's Gemini API."""
    
    def __init__(self, model_name: str = 'gemini-2.0-flash-lite', model_type: str = 'gemini'):
        """
        Initialize the SkillExtractor with model name and type.
        
        Args:
            model_name (str): Name of the model to use (default: 'gemini-2.0-flash-lite')
            model_type (str): Type of model to use ('gemini' or 'openai')
        """
        # Initialize model using the model manager
        self.client = get_model(model_name, model_type)
        self.model_name = model_name
    
    def extract_job_data(self, jobs: dict) -> Dict[str, List[str]]:
        """
        Extract structured data from a job description.
        
        Args:
            jobs (dict): Either a single job object or a dictionary of job objects
            
        Returns:
            Dict[str, List[str]]: Dictionary containing technical skills, soft skills, and experience requirements
        """
        # Craft the prompt
        prompt = f"""
        You are a system for data extraction.
        I will provide you with a job description, and you will mine its data.
        
        For every job description, scan the text and produce three lists:
        - List of technical skills: skills that are specific to a job and are not general skills. These would be names of programming languages, techniques, frameworks, etc. The elements of this list will generally be nouns.
        - List of soft skills: skills that are not technical in nature. These would be things like communication, teamwork, leadership, etc. These can also be soft skills that relate to work environment, company culture, such as: highly driven, fast paced, open ended problems, etc. The elements of this list will generally be short phrases.
        - List of experience requirements: requirements for experience in the job. This would be things like "3 years of experience in Python" or "Bachelor's degree in Computer Science". The elements of this list will generally be short sentences.

        Your output should be a Python dictionary of dictionaries (JSON) with the following structure:
        {{
            "job_id_1": {{
                "technical_skills": List[str],
                "soft_skills": List[str],
                "experience_requirements": List[str]
            }},
            "job_id_2": {{
                "technical_skills": List[str],
                "soft_skills": List[str],
                "experience_requirements": List[str]
            }},
            # ... more job_id entries
        }}
        Jobs:
        {jobs}
        """
        
        try:
            # Generate response using the client
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=prompt
            )
            
            # Extract the text from the response
            result_text = response.text.strip()
            
            # Remove markdown code formatting if present
            result_text = result_text.replace('```json\n', '').replace('```python\n', '').replace('```', '')
            
            # Use json.loads instead of eval for safer parsing
            import json
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                # If json.loads fails, try to clean up the text and try again
                # Remove any potential trailing commas or other syntax issues
                result_text = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                try:
                    result_dict = json.loads(result_text)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    print(f"Cleaned text: {result_text}")
                    # Fall back to eval as a last resort
                    result_dict = eval(result_text)
            
            # Ensure we got a dictionary with the expected structure
            if not isinstance(result_dict, dict):
                raise ValueError("Invalid response format from Gemini")
                
            return result_dict
            
        except Exception as e:
            print(f"Error extracting job data: {str(e)}")
            return {
                "technical_skills": [],
                "soft_skills": [],
                "experience_requirements": []
            }
    
    def analyze_resume_fit(self, job_description: str, resume: str) -> Dict[str, bool]:
        """
        Analyze how well a resume fits a job description.
        
        Args:
            job_description (str): The job description text
            resume (str): The resume text to analyze
            
        Returns:
            Dict[str, bool]: Dictionary containing feedback on resume fit
        """
        # Craft the prompt
        prompt = f"""
        You are a system for resume analysis.
        I will provide you with a job description and a resume, and you will analyze how well the resume fits the job.
        
        You will give a four item feedback on the resume:
        - Technical Skill Match (boolean): Whether the resume matches the technical skills required for the job.
        - Soft Skill Match (boolean): Whether the resume matches the soft skills required for the job.
        - Experience Match (boolean): Whether the resume matches the experience requirements for the job.
        - Good Fit (boolean): Whether the resume is a good fit for the job.

        Your output should be a Python dictionary with the following structure:
        {{
            "technical_skill_match": boolean,
            "soft_skill_match": boolean,
            "experience_match": boolean,
            "good_fit": boolean
        }}
        
        Job Description:
        {job_description}
        
        Resume:
        {resume}
        """
        
        try:
            # Generate response using the client
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=prompt
            )
            
            # Convert response to string and evaluate as Python dictionary
            result_text = response.text.strip()
            # Remove markdown code formatting if present
            result_text = result_text.replace('```python\n', '').replace('```', '')
            
            # Safely evaluate the string as a Python dictionary
            result_dict = eval(result_text)
            
            # Ensure we got a dictionary with the expected structure
            if not isinstance(result_dict, dict) or not all(key in result_dict for key in ["technical_skill_match", "soft_skill_match", "experience_match", "good_fit"]):
                raise ValueError("Invalid response format from Gemini")
                
            return result_dict
            
        except Exception as e:
            print(f"Error analyzing resume fit: {str(e)}")
            return {
                "technical_skill_match": False,
                "soft_skill_match": False,
                "experience_match": False,
                "good_fit": False
            }
    
    def extract_skills(self, job_description: str) -> List[str]:
        """
        Extract skills from a job description (legacy method for backward compatibility).
        
        Args:
            job_description (str): The job description text to analyze
            
        Returns:
            List[str]: List of skills extracted from the job description
        """
        # Get job data
        job_data = self.extract_job_data(job_description)
        
        # Combine all skills into a single list
        all_skills = (
            job_data["technical_skills"] + 
            job_data["soft_skills"] + 
            job_data["experience_requirements"]
        )
        
        return all_skills