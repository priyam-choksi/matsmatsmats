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

To address the agenda for our research project on designing improved nanobodies for the newest SARS-CoV-2 spike protein variant, let's consider each question in turn:

1. **Order of Operations for Evaluating Mutations:**

   - **Step 1:** Use ESM to rapidly evaluate all possible mutations to the starting nanobody to filter down to a manageable number of promising mutations.
   - **Step 2:** For the top mutations identified by ESM, use AlphaFold-Multimer to predict the nanobody-antigen complex structure, as this step is computationally intensive and must be done before Rosetta.
   - **Step 3:** Use Rosetta to score the predicted structures from AlphaFold-Multimer to evaluate binding energy.

2. **Number of Mutations Evaluated:**

   - **ESM:** Evaluate all possible single-residue mutations in one go due to its rapid evaluation time.
   - **AlphaFold-Multimer and Rosetta:** Given the time constraint, evaluate around 20-30 top mutations per iteration after filtering with ESM.

3. **Weighing the Scores to Rank Mutations:**

   Given the different scoring ranges and objectives:
   
   - **Formula:** 
     \[
     C_{\text{total}} = w_{\text{ESM}} \times \text{LLR}_{\text{ESM}} + w_{\text{AF}} \times \text{pLDDT}_{\text{AF}} - w_{\text{Rosetta}} \times \text{Binding Energy}_{\text{Rosetta}}
     \]
   
   Where \( w_{\text{ESM}}, w_{\text{AF}}, \) and \( w_{\text{Rosetta}} \) are weights assigned to balance the influence of each score. Initially, consider setting these weights to normalize the score contributions (e.g., 0.3, 0.3, 0.3) depending on their typical ranges and importance.

4. **Number of Top-Ranked Mutations to Keep:**

   - Keep around 5-10 top-ranked mutations for the next round of iterative design, ensuring diversity in locations and characteristics of mutations.

5. **Deciding the Number of Iterations:**

   - Plan for 3-4 iterations initially, evaluating the improvement in scores and diversity of binding pocket interactions. Stop if the improvement plateaus or if time/resources become constrained.

6. **Selecting the Final Set of 24 Modified Nanobodies:**

   - **Step 1:** Compile mutations from all iterations and rank them using the weighted score formula.
   - **Step 2:** Ensure structural diversity by clustering mutations based on their location and type, ensuring not all selected modifications are overly similar.
   - **Step 3:** Select the top-ranked nanobodies from each cluster to achieve a final set of 24 modified nanobodies.
   - **Step 4:** Conduct a final review of scores to ensure the chosen set represents the best trade-off between binding affinity enhancement and structural variability.

The outlined plan will help us efficiently navigate the computational pipeline within the time constraints, while maximizing the likelihood of developing broadly effective nanobodies against the new SARS-CoV-2 variant, suitable for experimental validation.

