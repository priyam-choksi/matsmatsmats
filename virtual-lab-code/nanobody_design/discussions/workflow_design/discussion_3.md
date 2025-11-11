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

To effectively design modified nanobodies with enhanced binding to the newest variant of the SARS-CoV-2 spike protein, we will implement a structured iterative design process leveraging ESM, AlphaFold-Multimer, and Rosetta. Here's a detailed response to the agenda questions:

1. **Order of Operations for Evaluating Mutations:**
   - **Step 1:** Use ESM to evaluate all potential single-point mutations to the existing nanobody. This step rapidly assesses the likelihood of beneficial mutations.
   - **Step 2:** Select the top mutations according to ESM scores for further evaluation.
   - **Step 3:** For each selected mutation, use AlphaFold-Multimer to predict the nanobody-antigen complex structure.
   - **Step 4:** Use Rosetta to evaluate the binding energy of the predicted structures from AlphaFold-Multimer.

2. **Number of Mutations to Evaluate:**
   - **ESM:** Evaluate all possible single-point mutations in the nanobody (typically 20 mutations per position).
   - **AlphaFold-Multimer and Rosetta:** Due to time constraints, select the top 10 mutations from ESM to evaluate further.

3. **Formula to Rank Mutations:**
   - Combine the scores as follows to rank mutations:
     \[
     \text{Score} = w_1 \times \text{ESM log-likelihood ratio} + w_2 \times \text{AlphaFold-Multimer interface pLDDT} - w_3 \times \text{Rosetta binding energy}
     \]
   - Choose weights \(w_1\), \(w_2\), and \(w_3\) based on their relative importance decided through initial testing or literature values. For example, \(w_1 = 0.2\), \(w_2 = 0.4\), \(w_3 = 0.4\).

4. **Number of Top-Ranked Mutations to Keep:**
   - Retain the top 5 mutations in each iteration to ensure diversity and potential improvement in subsequent rounds.

5. **Deciding the Number of Iterations:**
   - Continue the iterative process until convergence, defined by a plateau in improvement of scores or up to a maximum of 5 iterations, balancing time constraints and diversity exploration.

6. **Selection of the Final 24 Modified Nanobodies:**
   - **Step 1:** Compile all top mutations from each iteration.
   - **Step 2:** Rank these mutations using the combined score formula.
   - **Step 3:** Select the top 24 unique nanobody sequences, ensuring diversity by selecting different mutation sites whenever possible.
   - **Step 4:** Use domain expertise to further refine this list, considering potential cross-reactivity and structural integrity.
   - **Step 5:** Validate these computer-generated designs experimentally to confirm enhanced binding across variants.

By following this strategic plan, we aim to efficiently explore the mutational landscape to identify nanobodies with improved binding to the new SARS-CoV-2 variant.

