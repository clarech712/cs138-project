# Dataset Sources Documentation

This document provides information about the datasets used in this project, including their sources, descriptions, and any relevant details for future reference.

## Datasets

### 1. gsearch_jobs.csv
- **Source**: [Kaggle - Data Analyst Job Postings Google Search](https://www.kaggle.com/datasets/lukebarousse/data-analyst-job-postings-google-search)
- **Description**: Dataset containing job postings scraped from Google Jobs search results for data analyst positions. This dataset includes job titles, descriptions, companies, locations, and other job-related information. The data was collected by searching for "data analyst" jobs on Google Jobs and includes information from various job boards and company websites.
- **Size**: 287MB (about 61.3K entries)
- **Date Retrieved**: April 6, 2024
- **Fields**: Job titles, descriptions, companies, locations, and other job-related information
- **Usage in Project**: Used for training and testing the skill extraction and resume matching models

### 2. postings.csv
- **Source**: [Kaggle - LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- **Description**: Comprehensive dataset of job postings scraped from LinkedIn. This dataset includes detailed information about job requirements, responsibilities, and qualifications across various industries and roles. The data was collected by scraping LinkedIn job postings and includes structured information about job titles, descriptions, required skills, experience levels, and company information.
- **Size**: 556MB (about 124K entries)
- **Date Retrieved**: April 6, 2024
- **Fields**: Detailed job descriptions, required skills, experience levels, and company information
- **Usage in Project**: Used as a primary dataset for skill extraction and analysis

### 3. DataAnalyst.csv
- **Source**: [Kaggle - Data Analyst Jobs](https://www.kaggle.com/datasets/andrewmvd/data-analyst-jobs)
- **Description**: Dataset focused specifically on data analyst positions. This dataset contains job postings for data analyst roles, including information about required skills, experience, education, and job responsibilities. The data was collected from various job boards and company websites, focusing specifically on positions with "data analyst" in the title or description.
- **Size**: 7.57MB (2253 entries)
- **Date Retrieved**: April 6, 2024
- **Fields**: Job titles, descriptions, required skills, and company information for data analyst roles
- **Usage in Project**: Used for specialized analysis of data analyst positions

## Data Processing Notes

- All datasets have been cleaned and preprocessed for use in the project
- Duplicate entries have been removed
- Text data has been standardized (lowercase, punctuation removal, etc.)
- Missing values have been handled appropriately

## Future Dataset Additions

When adding new datasets to this project, please update this documentation with the following information:
- Dataset name and file path
- Source URL or origin
- Brief description
- Size and date retrieved
- Key fields
- How it's used in the project
- Any special processing notes 