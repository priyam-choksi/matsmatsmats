## User

This is the beginning of an individual meeting with Principal Investigator to discuss your research project.

Here is the agenda for the meeting:

Self-Supervised Learning for Biomedical Signal Processing
    
Problem: Annotated biomedical signal data (e.g., ECG, EEG) are scarce, limiting the effectiveness of supervised learning models.
    
Research Question: How can self-supervised learning be applied to biomedical time series to improve downstream task performance with minimal labeled data?

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

To effectively address the research question of applying self-supervised learning to biomedical signal processing, I will assemble a team with expertise in various relevant fields. This will ensure a comprehensive approach to the project, leveraging diverse skills and knowledge. Here is the proposed team:

```python
team = [
    Agent(
        title="Machine Learning Expert",
        expertise="self-supervised learning and time series analysis",
        goal="develop novel self-supervised learning techniques tailored for biomedical signals",
        role="design and implement self-supervised learning algorithms, and evaluate their performance on biomedical time series data",
        model=model,
    ),
    Agent(
        title="Biomedical Engineer",
        expertise="biomedical signal processing and analysis",
        goal="ensure that the machine learning models are effectively aligned with biomedical signal characteristics",
        role="provide insights into the nuances of ECG and EEG signals to guide the adaptation of learning methods",
        model=model,
    ),
    Agent(
        title="Data Scientist",
        expertise="data preprocessing and feature extraction",
        goal="optimize the use of available biomedical data through effective preprocessing and feature engineering strategies",
        role="implement data cleaning, transformation, and feature extraction pipelines for biomedical time series data",
        model=model,
    ),
    Agent(
        title="Clinical Researcher",
        expertise="clinical applications of biomedical signal data",
        goal="ensure that the research aligns with clinical needs and potential applications",
        role="provide expertise on real-world clinical challenges and validate the clinical relevance of the developed models",
        model=model,
    ),
    Agent(
        title="AI Ethicist",
        expertise="ethical implications of AI in healthcare",
        goal="ensure the ethical deployment and use of AI technologies in biomedical signal processing",
        role="assess and guide the team on ethical considerations and compliance with healthcare regulations",
        model=model,
    )
]
```

This team will collaborate to tackle the challenges of self-supervised learning in biomedical signal processing, ensuring that the developed solutions are both scientifically robust and clinically relevant.

