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

To address the agenda for our research project on developing modified nanobodies for the newest variant of the SARS-CoV-2 spike protein, I propose the following plan:

### 1. Order of Operations for Evaluating Mutations
In each iteration, the order of operations will be:
- **ESM**: First, evaluate all potential mutations for a nanobody using ESM to quickly narrow down the most promising candidates. This step is computationally efficient and will take a total of 5 minutes for all mutations.
- **AlphaFold-Multimer**: After filtering the mutations based on ESM scores, use AlphaFold-Multimer to predict the structure of the top-ranked mutations. This provides detailed structural insights necessary for subsequent steps.
- **Rosetta**: Finally, use Rosetta on the structures predicted by AlphaFold-Multimer to evaluate the binding energy of the mutations.

### 2. Number of Mutations Evaluated
- **ESM**: Evaluate all possible mutations (potentially hundreds depending on the size of the nanobody) in each iteration.
- **AlphaFold-Multimer and Rosetta**: Due to time constraints, evaluate the top 20 mutations from the ESM step using AlphaFold-Multimer. Then, evaluate all 20 with Rosetta using the structures from AlphaFold-Multimer.

### 3. Weighing the Scores to Rank Mutations
To rank the nanobody mutations, combine the scores from ESM, AlphaFold-Multimer, and Rosetta using the following formula:
\[ \text{Score} = w_1 \times \text{(ESM Log-Likelihood Ratio)} + w_2 \times \text{(AlphaFold-Multimer Interface pLDDT)} - w_3 \times \text{(Rosetta Binding Energy)} \]
Where \( w_1, w_2, \) and \( w_3 \) are weights that can be tuned based on empirical testing, with an initial suggestion of \( w_1 = 0.2, w_2 = 0.5, \) and \( w_3 = 0.3 \).

### 4. Number of Top-Ranked Mutations Kept
Keep the top 5 mutations from each iteration to carry forward into the next round of iterations.

### 5. Deciding Number of Iterations
Plan to run the process for up to 4 iterations, allowing for sufficient exploration of the sequence space while staying within time constraints. Monitor convergence of scores and diversity of mutations; if improvements plateau or all mutations appear similar, consider stopping early.

### 6. Selecting the Final Set of Modified Nanobodies
After completing all iterations:
- Collect all top mutations from each round.
- Rank all mutations from all iterations based on the combined score.
- Select the top 24 unique mutations with the highest scores, ensuring coverage of diverse mutational profiles to enhance potential cross-variant activity.
- If there are ties or close scores, prioritize based on diversity of sequences to ensure broad coverage of the mutational landscape.

This plan balances computational efficiency with scientific rigor, ensuring that we can effectively identify promising nanobody candidates within our constraints. Let me know if there are any questions or further adjustments needed.

