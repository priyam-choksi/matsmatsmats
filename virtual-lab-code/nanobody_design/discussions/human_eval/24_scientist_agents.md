## User

This is the beginning of an individual meeting with Principal Investigator to discuss your research project.

Here is the agenda for the meeting:

AI for Identifying Subtypes in Heterogeneous Diseases
    
Problem: Diseases like autism or depression are clinically heterogeneous, making diagnosis and treatment inconsistent.
    
Research Question: Can unsupervised learning models identify meaningful subtypes of heterogeneous diseases using multi-modal patient data?

You need to select a team of scientists to help you with this project. Please select the team members that you would like to invite to a discussion on this topic. Please list the team members in the following format, using the team member below as an example. You should not include yourself (Principal Investigator) in the list. Please write the full team of agents as a Python list of Agent objects. (Note that every agent should have "model=model" as the last parameter.)

Agent(
    title="Principal Investigator",
    expertise="applying artificial intelligence to biomedical research",
    goal="perform research in your area of expertise that maximizes the scientific impact of the work",
    role="lead a team of experts to solve an important problem in artificial intelligence for biomedicine, make key decisions about the project direction based on team member input, and manage the project timeline and resources",
    model=model,
)

Principal Investigator, please provide your response to the agenda.

## Principal Investigator

```python
team_members = [
    Agent(
        title="Data Scientist",
        expertise="unsupervised learning and clustering algorithms",
        goal="develop and optimize unsupervised learning models to identify patterns in multi-modal data",
        role="design and implement clustering algorithms, conduct experiments to assess model performance, and collaborate with domain experts for interpretation",
        model=model,
    ),
    Agent(
        title="Biomedical Data Analyst",
        expertise="multi-modal biomedical data integration",
        goal="integrate diverse data types to create a comprehensive dataset for analysis",
        role="collect, preprocess, and integrate patient data from various sources, ensuring data quality and consistency",
        model=model,
    ),
    Agent(
        title="Clinical Researcher",
        expertise="heterogeneous diseases such as autism and depression",
        goal="provide clinical insights to guide the identification of disease subtypes",
        role="interpret the clinical relevance of identified subtypes, validate findings with clinical data, and provide domain-specific knowledge",
        model=model,
    ),
    Agent(
        title="Bioinformatician",
        expertise="genomic and proteomic data analysis",
        goal="analyze genetic and proteomic data to find correlations with identified disease subtypes",
        role="conduct in-depth analyses of genomic and proteomic datasets, collaborate with data scientists to integrate findings into machine learning models",
        model=model,
    ),
    Agent(
        title="Software Engineer",
        expertise="machine learning infrastructure and deployment",
        goal="ensure robust and scalable implementation of machine learning models",
        role="develop software tools for data processing and model deployment, optimize computational efficiency, and maintain the project’s technical infrastructure",
        model=model,
    )
]
```

