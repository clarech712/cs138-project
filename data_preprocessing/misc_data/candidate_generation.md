Generate professional bio and a structured resume for a hypothetical job applicant based on the specifications provided below. The applicant should be looking for jobs related to data science or data analytics.

The output should contain three main components: a detailed **BIO**, a comprehensive **RESUME**, and a structured **JSON** object containing key information.

**Candidate Specifications:**
- **Target Role/Focus:** {e.g., "Mid-level Data Scientist with NLP experience", "Senior Frontend Developer focusing on React and TypeScript", "Entry-level Marketing Associate with social media skills"}
- **Experience Level:** {e.g., "2-4 years of professional experience", "Recent graduate with strong internship experience", "Over 10 years of leadership experience"}
- **Key Skills/Technologies:** {List specific skills relevant to the role, e.g., "Python, spaCy, TensorFlow, PyTorch, SQL, AWS Sagemaker", "React, TypeScript, Redux, Webpack, Jest, Node.js", "Content Creation, Social Media Marketing, SEO basics, Google Analytics"}
- **Industry Preference/Background (Optional):** {e.g., "Experience in the Fintech industry", "Academic research background in Biology", "Worked in non-profit sector"}
- **Education Highlights (Optional):** {e.g., "Master's degree in Computer Science", "Bootcamp graduate in Web Development"}
- **Specific Achievements/Focus (Optional):** {e.g., "Led a project from conception to deployment", "Contributed to open source libraries", "Strong communication and presentation skills"}

### BIO ###
Generate a professional bio (approximately 400-800 words) that:
- Summarizes the candidate's background comprehensively
- Aligns with the specified experience level and skills
- Highlights their unique value proposition for the target role
- Incorporates relevant industry-specific keywords and terminology 
- Presents a compelling narrative of their career journey
- Emphasizes key achievements that demonstrate expertise in the field
- Includes details about educational background and professional development
- Maintains a professional yet personable tone

### RESUME ###
Generate a detailed resume using realistic but placeholder information for personal details. Structure it as follows:

**1. Contact Information:** 
   - Include standard placeholders: [Candidate Name], [Phone Number], [Email Address], [City, State], [LinkedIn Profile]
   - Format consistently and professionally

**2. Professional Summary:**
   - Write a focused 3-4 sentence statement summarizing core qualifications and career goals
   - Highlight unique value proposition and specialized expertise
   - Align closely with the content in the generated bio
   - Emphasize most impressive achievements or skills

**3. Work Experience:**
   - Include 1-4 relevant positions aligned with the specified experience level
   - For each position, provide:
     * Descriptive job title reflecting actual responsibilities
     * Company name (use realistic placeholder names)
     * Location (city, state)
     * Precise employment dates (MM/YYYY - MM/YYYY format)
     * 4-6 detailed bullet points describing:
       - Key responsibilities (using action verbs)
       - Specific, quantifiable achievements (e.g., "Increased user engagement by 35% by implementing...")
       - Technologies/methodologies used
       - Impact on team/organization/clients
       - Problem-solving approaches
   - Ensure chronological consistency and appropriate progression

**4. Education:**
   - List relevant degrees in reverse chronological order
   - For each degree, include:
     * Full degree name and major
     * University name (use realistic placeholder)
     * Location (city, state)
     * Graduation year (consistent with overall timeline)
     * GPA (if impressive, e.g., 3.5+/4.0)
     * Honors or distinctions (if applicable)
     * 2-3 relevant courses (especially for recent graduates)

**5. Skills:**
   - Organize into clear categories such as:
     * Programming Languages
     * Frameworks & Libraries
     * Database Technologies
     * Cloud Platforms & DevOps
     * Tools & Software
     * Methodologies & Processes
     * Soft Skills & Competencies
   - Align with Key Skills/Technologies specified earlier
   - Include proficiency level indicators where appropriate

**6. Projects (Optional):**
   - Include 1-2 detailed project descriptions if relevant to experience level
   - For each project, provide:
     * Clear, descriptive title
     * Purpose and business context
     * Technologies/tools used
     * Individual contributions and role
     * Measurable outcomes or results
     * Link to repository/demo (use placeholder)

**7. Certifications & Awards (Optional):**
   - List relevant professional certifications
   - Include any notable industry awards or recognition
   - Provide certification dates consistent with timeline

Ensure all experience and education timelines form a coherent professional narrative. The resume should follow standard industry formatting conventions.

### JSON OUTPUT ###
Additionally, provide a structured JSON output containing the candidate's information in the following format:

```json
{
    "bio": "Complete text of the candidate's professional biography",
    "resume": "Complete text of the candidate's structured resume",
    "hard_skills": ["Array of all hard skills extracted from both bio and resume"],
    "soft_skills": ["Array of all soft skills extracted from both bio and resume"]
}
```

The skills should be categorized as follows:

- **Hard skills**: Technical, job-specific abilities that are acquired through education, training, and practice. These include programming languages, frameworks, tools, technologies, and specialized techniques. Hard skills should be presented as nouns or short technical phrases (e.g., "Python", "TensorFlow", "React", "SQL", "AWS", "Data Analysis", "UX Design", "Financial Modeling").

- **Soft skills**: Non-technical abilities related to how you work and interact with others. These encompass interpersonal attributes, character traits, and professional attitudes. Soft skills should be presented as concise phrases (e.g., "Team Leadership", "Effective Communication", "Problem-solving", "Adaptability", "Critical Thinking", "Time Management", "Conflict Resolution").
