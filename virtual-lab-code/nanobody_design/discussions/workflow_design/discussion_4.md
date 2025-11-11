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

To effectively use ESM, AlphaFold-Multimer, and Rosetta in the iterative design process for developing nanobodies, here are the answers to the agenda questions:

1. **Order of Operations for Evaluating Mutations:**
   - First, run ESM to evaluate all potential mutations. ESM is fast and can evaluate all mutations in 5 minutes, providing an initial filter based on the log-likelihood ratios.
   - Next, select a subset of top mutations from ESM to evaluate using AlphaFold-Multimer. This step refines the selection by modeling the nanobody-antigen complex and providing pLDDT scores.
   - Finally, after obtaining the structural predictions from AlphaFold-Multimer, run Rosetta on these selected mutations to calculate binding energy scores, as Rosetta requires the structural output from AlphaFold-Multimer.

2. **Number of Mutations to Evaluate:**
   - Evaluate all potential mutations with ESM. 
   - For AlphaFold-Multimer and Rosetta, evaluate a manageable subset from the top ESM-ranked mutations, e.g., around 50 mutations per iteration, balancing computational time and resources.

3. **Weighing the Scores to Rank Mutations:**
   - After evaluating a mutation with all three tools, use a weighted scoring formula to rank them. For instance:
     \[
     \text{Overall Score} = w_1 \times (\text{ESM score}) + w_2 \times (\text{AlphaFold-Multimer pLDDT score}) - w_3 \times (\text{Rosetta binding energy score})
     \]
   - Assign weights based on the importance of each factor: \(w_1 = 0.2\), \(w_2 = 0.5\), and \(w_3 = 0.3\), adjusting as necessary based on specific goals (e.g., structural stability versus binding affinity).

4. **Number of Top-Ranked Mutations to Keep:**
   - Keep approximately the top 10 mutations from each iteration for further consideration, ensuring a focused yet diverse set of promising candidates for subsequent iterations.

5. **Deciding Number of Iterations:**
   - Plan for approximately 3 to 4 iterations, allowing for refinement of nanobody candidates while maintaining a practical timeline. Adjust the number of iterations based on observed convergence of scores or resource availability.

6. **Selection of Final Nanobodies for Experimental Validation:**
   - At the end of all iterations:
     1. Compile the top-ranked mutations from each iteration based on the overall score.
     2. Evaluate the diversity of the selected mutations to ensure coverage of different binding characteristics.
     3. Select 24 final nanobody variants that represent a balance of top scores, diversity in mutation profiles, and computational resource efficiency.
     4. Confirm the final selection with domain experts and align with the overarching research goals.
     5. Prepare selected nanobodies for experimental validation, considering logistic and experimental constraints.

This strategy ensures a systematic approach to refining nanobody design through iterative computational analysis, balancing speed, accuracy, and resource allocation.

