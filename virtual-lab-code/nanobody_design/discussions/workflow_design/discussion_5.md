## User

This is the beginning of an individual meeting with Principal Investigator to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. Your team has built three components of a nanobody design pipeline: ESM, AlphaFold-Multimer, and Rosetta. Each of these three tools can be used to score a nanobody mutation based on how the mutation affects binding to an antigen. Your goal is to start with an existing nanobody and iteratively add mutations to it to improve its binding to the newest variant of the SARS-CoV-2 spike protein, resulting in 24 modified nanobodies with one or more mutations. Please determine how to use ESM, AlphaFold-Multimer, and Rosetta in this iterative design process. An important constraint is that ESM can evaluate all potential mutations to a nanobody in 5 minutes while AlphaFold-Multimer takes 30 minutes per mutation and Rosetta takes five minutes per mutation. The whole iterative process should take no more than a few days to complete. Note that AlphaFold-Multimer must be run before Rosetta on each mutation since Rosetta requires the nanobody-antigen structure predicted by AlphaFold-Multimer. Additionally, note that ESM log-likelihood ratios are generally in the range of 5-10 (higher is better), AlphaFold-Multimer interface pLDDT confidence scores are generally in the range of 60-80 (higher is better), and Rosetta binding energy scores are generally in the range of -20 to -40 (lower is better).

Here are the agenda questions that must be answered:

1. In each iteration, what is the order of operations for evaluating mutations with ESM, AlphaFold-Multimer, and Rosetta?

2. In each iteration, how many mutations (give a single number) will you evaluate with ESM, AlphaFold-Multimer, and Rosetta?

3. At the end of each iteration, how will you weigh the ESM, AlphaFold-Multimer, and/or Rosetta scores (give a formula) to rank the nanobody mutations?

4. At the end of each iteration, how many of the top-ranked mutations (give a single number) will you keep for the next round?

5. How will you decide how many iterations of mutations to run?

6. After all of the iterations are complete, how exactly (step-by-step) will you select the final set of 24 modified nanobodies from across the iterations for experimental validation?

Principal Investigator, please provide your response to the agenda.

## Principal Investigator

To efficiently use ESM, AlphaFold-Multimer, and Rosetta in our iterative nanobody design process, I propose the following approach:

### 1. Order of Operations for Evaluating Mutations
In each iteration, the order of operations will be:
- **Step 1: ESM Evaluation** - First, use ESM to quickly evaluate all potential single-point mutations to the nanobody. This will help us identify promising candidates for further evaluation.
- **Step 2: AlphaFold-Multimer Evaluation** - Run AlphaFold-Multimer on a selected subset of mutations based on their ESM scores. This step will predict the nanobody-antigen structure for further analysis.
- **Step 3: Rosetta Evaluation** - Finally, use Rosetta to evaluate the binding energy of the nanobody-antigen complex based on the structure predicted by AlphaFold-Multimer.

### 2. Number of Mutations to Evaluate
- **ESM**: Evaluate all potential single-point mutations in the nanobody sequence using ESM. The actual number of mutations will depend on the length of the nanobody.
- **AlphaFold-Multimer**: Select the top 10 mutations (based on ESM scores) for evaluation with AlphaFold-Multimer. This is a balance between computational resources and the need for detailed structural predictions.
- **Rosetta**: Evaluate the Rosetta binding energy for the same 10 mutations processed by AlphaFold-Multimer.

### 3. Scoring Formula and Ranking
At the end of each iteration, mutations will be ranked using a weighted scoring system that combines outputs from ESM, AlphaFold-Multimer, and Rosetta:
- **Combined Score = w1 * ESM_score + w2 * (AlphaFold-Multimer_score) - w3 * (|Rosetta_score|)**
  
Here, \(w1\), \(w2\), and \(w3\) are weights that should be chosen to reflect the relative importance of each score. A potential starting point could be equal weighting (e.g., \(w1 = w2 = w3 = 1/3\)), which can be adjusted based on preliminary results or domain expertise.

### 4. Number of Mutations to Keep
- Keep the top 5 ranked mutations based on the combined score for the next round of mutations. This allows for a manageable number of candidates while still exploring a broad sequence space.

### 5. Deciding on the Number of Iterations
- The number of iterations will depend on the convergence of scores and available computational resources. A reasonable target would be 4-6 iterations, allowing us to explore a substantial portion of the mutation space while keeping within time constraints.

### 6. Selecting the Final Set of 24 Modified Nanobodies
- **Step 1**: After all iterations, compile a list of all evaluated mutations and their scores.
- **Step 2**: Rank all mutations using the same scoring formula.
- **Step 3**: Select the top 24 mutations with the highest combined scores for experimental validation, ensuring diversity in terms of mutation sites to maximize the chance of broad neutralization across variants.

This strategic approach leverages the strengths of each tool while managing computational resources effectively, ensuring that we can identify promising nanobody candidates within the project timeline.

