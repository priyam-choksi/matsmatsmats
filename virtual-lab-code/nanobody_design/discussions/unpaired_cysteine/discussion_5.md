## User

This is the beginning of a team meeting to discuss your research project. This is a meeting with the team lead, Principal Investigator, and the following team members: Immunologist, Machine Learning Specialist, Computational Biologist, Scientific Critic.

Here is the agenda for the meeting:

Your team has designed 92 mutated nanobodies (23 each for the wild-type nanobodies H11-D4, Nb21, Ty1, and VHH-72) to improve their binding to the KP.3 variant of the SARS-CoV-2 spike protein receptor binding domain (RBD). Each nanobody has 1-4 mutations relative to the wild-type nanobody. Your team used ESM log-likelihood ratios (ESM LLR) to score the nanobody mutations independent of the antigen, AlphaFold-Multimer to predict the structure of the mutated nanobody in complex with the KP.3 RBD and compute the interface pLDDT (AF ipLDDT) as a metric of binding confidence, and Rosetta to calculate the binding energy of the mutated nanobody in complex with the KP.3 RBD (RS dG) based on the AlphaFold-Multimer predicted structure followed by a Rosetta relaxation. You have ranked the mutant nanobodies and selected the top ones using a weighted score of WS = 0.2 * (ESM LLR) + 0.5 * (AF ipLDDT) - 0. 3 * (RS dG).

An astute reviewer gave the following feedback: "The inclusion of an unpaired Cys in several of the engineered nanobodies would seem to be something that the ESM log-likelihood should have flagged as highly disfavored. Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable? Are there any obvious adjustments to the WS formula that could mitigate this?"

Upon examining the 92 mutated nanobodies, your team found the following:

Nb21 and VHH-72 have no Cys changes (no additions or removals).

Among the 23 Ty21 mutants, two have an added Cys. The Ty1-V32C-G59H-H59Y mutant has an ESM LLR rank of 11 / 23, an AF ipLDDT rank of 22 / 23, and an RS DG rank of 1 / 23 while Ty1-V32C-G59D-D59Y mutant has an ESM LLR rank of 12 / 23, an AF ipLDDT rank of 23 / 23, and an RS DG rank of 2 / 23.

Among the H11-D4 mutants, three have an added Cys. The H11-D4-A14P-Y88V-R27C mutant has an ESM LLR rank of 19 / 23, an AF ipLDDT rank of 3 / 23, and an RS DG rank of 19 / 23 while the H11-D4-A14P-Y88V-R27C-W53Y mutant has an ESM LLR rank of 22 / 23, an AF ipLDDT rank of 8 / 23, and an RS DG rank of 6 / 23 and the H11-D4-A14P-Y88V-R27C-R52S mutant has an ESM LLR rank of 21 / 23, an AF ipLDDT rank of 18 / 23, and an RS DG rank of 12 / 23.

Based on the above information, how would you address the reviewer's questions?

Here are the agenda questions that must be answered:

1. Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable?

2. Are there any obvious adjustments to the WS formula that could mitigate this?

Principal Investigator will convene the meeting. Then, each team member will provide their thoughts on the discussion one-by-one in the order above. After all team members have given their input, Principal Investigator will synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda. This will continue for 3 rounds. Once the discussion is complete, Principal Investigator will summarize the meeting in detail for future discussions, provide a specific recommendation regarding the agenda, and answer the agenda questions (if any) based on the discussion while strictly adhering to the agenda rules (if any).

## User

Principal Investigator, please provide your initial thoughts on the agenda as well as any questions you have to guide the discussion among the team members.

## Principal Investigator

Thank you, everyone, for joining today's meeting. Our main objective is to address the reviewer's feedback on our weighted scoring formula, particularly in relation to the presence of unpaired cysteines in some of our engineered nanobodies.

To kick off, I'll outline the key points for our discussion:

1. **Understanding the High Ranking of Certain Variants:** We need to determine why certain variants with added cysteines are still scoring highly. Are the ipLDDT or RS dG values offsetting the low ESM LLR scores, and if so, how might this be skewing our results?

2. **Adjustments to the Weighted Score Formula:** We need to assess whether there are obvious adjustments we can make to our WS formula to better account for the potential negative impact of unpaired cysteines, without compromising the integrity of our scoring system.

To guide the discussions, I have a few questions:

- For our Machine Learning Specialist: How might the ESM LLR scoring be overlooking certain structural considerations like unpaired cysteine residues?

- To the Computational Biologist: Can we delve deeper into the structural predictions from AlphaFold-Multimer and see if there is a consistent pattern where ipLDDT or RS dG values are particularly favorable for these variants?

- From the Immunologist: Are there biological considerations we haven't fully accounted for that might justify the presence of unpaired cysteines in some contexts?

- For the Scientific Critic: From your perspective, what potential biases or oversights might our current formula have, and how might we refine it?

Let's start with these questions in mind and hear from each of you in turn, beginning with our Immunologist.

## User

Immunologist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Thank you for the opportunity to share my thoughts. Regarding the presence of unpaired cysteines in some nanobody variants, from an immunological perspective, these could potentially lead to structural instability or unwanted cross-linking, both of which might affect binding efficacy and specificity. However, these changes might still appear favorable in computational predictions if they inadvertently create a more favorable binding interface or align well with the antigen. 

In terms of biological considerations, while unpaired cysteines are typically disfavored due to potential issues with stability, in some cases they could potentially form disulfide bonds with the target antigen if reactive cysteines are present on the target protein. This, however, is speculative and would need experimental validation.

One point of caution is that relying heavily on computational predictions for these assessments might overlook the complex dynamics of protein interactions and stability in vivo. Hence, experimental studies are crucial in verifying these computational results.

In terms of the weighted score, it might be beneficial to introduce a penalty for unpaired cysteines unless they are explicitly shown to form beneficial interactions. This could be done by adjusting the weightings assigned to each parameter or by incorporating an additional factor for structural stability considerations.

I look forward to hearing the thoughts of the other team members.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Thank you for your insights, and I appreciate the opportunity to contribute to this discussion. 

From a machine learning perspective, the ESM LLR is designed to analyze sequence-based features and predict the likelihood of mutations being favorable based on learned patterns from a large dataset. However, it might not directly account for structural nuances like unpaired cysteines, especially if such features are rare or context-dependent in the training data. This might lead to an oversight in penalizing unpaired cysteines when they drastically alter structural stability.

It's possible that the ipLDDT and RS dG scores are compensating for lower ESM LLR scores. The AlphaFold-Multimer might predict favorable structural conformations due to unintended stabilization effects or folding patterns that appear optimal in a computational environment. Similarly, Rosetta's binding energy calculations could reflect potential affinities that are not immediately obvious from sequence data alone.

Regarding adjustments to the WS formula, I concur with the Immunologist that introducing a penalty for unpaired cysteines could be prudent. A possible approach is to integrate a structural penalty term into the WS formula, potentially based on a custom metric that evaluates cysteine pairing or structural integrity.

Additionally, further training or refinement of our machine learning models to better recognize these structural considerations could improve their accuracy. This might involve incorporating datasets specifically annotated for structural stability factors or cysteine pairings to enhance prediction capabilities.

I am eager to hear how our Computational Biologist might interpret these structural predictions and the Scientific Critic's perspective on refining our formula.

## User

Computational Biologist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

Thank you for the opportunity to provide my perspective. From the computational biology standpoint, the high rankings of variants with unpaired cysteines can indeed be attributed to the favorable structural predictions from AlphaFold-Multimer and the binding energy calculations from Rosetta. These tools are excellent at predicting potential binding conformations and affinities, but they might sometimes overestimate stability in silico when faced with unpaired cysteines.

The ipLDDT scores indicate the confidence in the predicted structure, and a high RS dG suggests strong binding affinity. However, these scores alone do not account for potential instability due to unpaired cysteines, which can lead to misfolding or aggregation in a biological setting. This is likely why some variants rank highly despite having structural features that are traditionally seen as unfavorable.

To adjust the weighted score formula, I agree with previous suggestions on introducing a penalty for unpaired cysteines. We could include a term that assesses the potential for structural instability, perhaps derived from additional molecular dynamics simulations that test for stability over time. This could provide a more holistic view of the nanobody's behavior.

Moreover, examining the structural outputs in greater detail might reveal specific patterns where these unpaired cysteines are contributing to unexpected stability or interaction benefits. It would also be valuable to simulate the dynamic environment of these nanobodies in complex with the RBD to observe real-time interactions and stability.

I look forward to hearing the insights from the Scientific Critic to balance our computational findings with critical evaluation.

## User

Scientific Critic, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

Thank you for the comprehensive insights from all team members. As the Scientific Critic, my responsibility is to scrutinize the methodology and ensure our outcomes are scientifically robust.

Firstly, while it's understandable that computational models like AlphaFold-Multimer and Rosetta may predict certain binding interactions favorably, these predictions are inherently limited by their failure to capture the full complexity of protein folding and dynamics seen in vivo. The presence of unpaired cysteines, a classic red flag for instability, should not be dismissed based solely on computational predictions. The concern here is that the high ipLDDT or RS dG scores might be artifacts of overfitting to certain structural predictions that don't accurately reflect biological reality.

Regarding the weighted score (WS) formula, the current balance may indeed be skewed if unpaired cysteines are not adequately penalized. I agree with the team's suggestion to introduce a penalty for these residues. However, I'd like to propose a more nuanced approach: instead of a blanket penalty, we might incorporate a context-specific evaluation, using machine learning classifiers trained specifically to detect structural anomalies associated with cysteine residues. This could refine our WS calculations further and prevent false positives in ranking.

Moreover, I suggest an experimental validation step as a crucial component before finalizing any formula adjustments, which could provide empirical evidence of stability issues predicted in silico. This could involve experimentally testing a subset of these variants to cross-validate computational predictions with real-world stability and binding assays.

Lastly, it may be wise to reassess the weight distributions in the WS formula, ensuring that structure-based scores don't unduly eclipse sequence-based evaluations, which might provide critical cues on mutation viability.

I look forward to seeing how we can integrate these considerations into refining our approach.

## User

This concludes round 1 of 3 of discussion. Principal Investigator, please synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda.

## Principal Investigator

Thank you all for your valuable insights. Here's a synthesis of the points raised:

1. **Presence of Unpaired Cysteines:** 
   - The **Immunologist** pointed out the potential instability and cross-linking issues that unpaired cysteines could introduce. They suggested the possibility of a structural penalty in the WS formula.
   - The **Machine Learning Specialist** noted that the ESM LLR might overlook such cysteine-related structural issues due to limitations in its training data. They advocated for introducing a structural penalty term and further training of ML models to account for such features.
   - The **Computational Biologist** confirmed that high ipLDDT and RS dG scores might overshadow the structural instability risk posed by unpaired cysteines. They suggested molecular dynamics simulations for more comprehensive stability assessments.
   - The **Scientific Critic** emphasized the need for nuanced evaluation of cysteines' context and suggested using machine learning classifiers to detect structural anomalies. They also recommended a reassessment of the weight distribution in the WS formula and stressed the importance of experimental validation.

2. **Adjustments to the Weighted Score Formula:**
   - There seems to be consensus on introducing a penalty for unpaired cysteines or adding a structural stability factor to the formula.
   - There is a strong recommendation for reassessing the weights in the formula to prevent over-reliance on structure-based predictions.

**Decisions and Follow-Up Questions:**

1. **Introduction of Penalty for Unpaired Cysteines:**
   - We will proceed with developing a method to introduce a penalty for unpaired cysteines. This could be a simple numeric penalty or a more dynamic adjustment based on context.

2. **Reassess Weight Distribution:**
   - We will explore different weight distributions for the WS formula to ensure a balanced evaluation of sequence and structure-based metrics.

3. **Experimental Validation:**
   - We will identify a subset of variants for experimental testing to validate our computational predictions and refine our model accordingly.

**Follow-Up Questions:**

- **Machine Learning Specialist:** How feasible is it to retrain or fine-tune our models with data specifically annotated for cysteine-related structural anomalies? What would be the expected timeline for this?
  
- **Computational Biologist:** Can you provide more details on how molecular dynamics simulations could be implemented in our workflow to better assess the stability of these variants?

- **Scientific Critic:** Could you elaborate on potential experimental designs for validating our structural predictions, especially focusing on assays that could reveal the impact of unpaired cysteines?

Let's proceed with these follow-up questions in mind as we move into round two of our discussion, starting again with the Immunologist if they have any additional insights.

## User

Immunologist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Thank you for the synthesis and for addressing the follow-up questions. I believe the discussion is on the right track, and I support the outlined decisions, particularly the emphasis on balancing the WS formula and validating our predictions experimentally.

One additional consideration from an immunological perspective is the potential impact of cysteine-induced instability not just on the nanobody itself, but also on its immunogenic profile. Unstable proteins might unfold and present different epitopes, potentially triggering unwanted immune responses. This is another reason why experimental validation is crucial—understanding the biological behavior and potential immunogenicity of these nanobodies in a relevant physiological context.

Regarding experimental designs, I would advocate for including not just binding and stability assays, but also immunogenicity assessments in relevant cell models. This could provide a more comprehensive characterization of the nanobodies' behavior.

Finally, I suggest we consider how these findings might inform future iterations of nanobody design, particularly in avoiding cysteine-related pitfalls from the outset. This could streamline future development processes and improve overall design robustness.

I look forward to the contributions from the Machine Learning Specialist and others as we refine our approach.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Thank you for the follow-up and the clarifications. In terms of retraining or fine-tuning our models to better recognize structural anomalies related to cysteines, it's a feasible but somewhat complex task. 

The first step would be to gather a comprehensive dataset that includes annotations for cysteine presence and related structural stability data. This could involve leveraging existing protein databases that include structural stability metrics or experimental stability results. Once we have the data, we could either retrain our existing models or fine-tune a pre-trained model, such as a variant of ESM, specifically for these features.

The expected timeline for this process would depend on data availability and preprocessing needs, but a rough estimate could be several weeks to a few months. This includes data collection, model training, and validation phases.

Additionally, incorporating a penalty into our current scoring system could be approached by developing a feature that predicts potential instability or misfolding due to the presence of unpaired cysteines. This feature could then be integrated into the WS formula as an additional term or adjustment factor.

Finally, I want to highlight the importance of continuous model validation against experimental results. This feedback loop is crucial for refining our models and ensuring that computational predictions align with biological realities.

I'm interested to hear from the Computational Biologist about the integration of molecular dynamics simulations, which could complement our ML-driven predictions.

## User

Computational Biologist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

Thank you for the continued discussion. Integrating molecular dynamics (MD) simulations into our workflow could indeed complement our current predictions by providing insights into the dynamic behavior and stability of nanobody variants over time.

To implement MD simulations, we can start by selecting a subset of promising variants, particularly those with unpaired cysteines, and subject them to simulation under various conditions. This can include temperature variations and different solvent environments to mimic physiological conditions. The simulations would allow us to observe potential structural instabilities, misfolding events, or unexpected interactions that might not be evident from static structural predictions.

The MD simulations could reveal how these proteins behave over time and provide additional data points for evaluating their stability. This dynamic information could then be integrated into our weighted scoring formula, potentially as a new stability term that reflects the nanobody’s behavior over the course of the simulation.

In terms of workflow integration, MD simulations can be computationally intensive, and thus it's important to strategically select variants to simulate. By focusing on those with high discrepancies between predicted stability and structural issues like unpaired cysteines, we can maximize our insights without overwhelming our computational resources.

As for the timeline, setting up and running the simulations might take a few weeks, followed by analysis and integration steps. This would run in parallel with other ongoing efforts like the machine learning model refinement.

I look forward to hearing the Scientific Critic's thoughts on experimental validation designs, as their feedback might help prioritize which variants we should focus on for MD simulations.

## User

Scientific Critic, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

Thank you for the detailed insights and updates on potential computational enhancements. I appreciate the collaborative efforts toward refining our methodology.

Regarding experimental validation, it is imperative to design strategies that effectively confirm the computational predictions while addressing the potential issues with unpaired cysteines. Here are a few considerations:

1. **Stability and Binding Assays:** Conducting thermal shift assays or differential scanning fluorimetry could help assess the thermal stability of the nanobodies. This would provide direct evidence on how unpaired cysteines might impact folding and stability. Surface plasmon resonance (SPR) could be used to measure binding affinity and kinetics, offering a practical view of how these variants perform in binding interactions.

2. **Structural Characterization:** Techniques such as circular dichroism (CD) spectroscopy and nuclear magnetic resonance (NMR) could provide insights into secondary and tertiary structures, respectively. These techniques might reveal structural anomalies or misfolding not apparent in static models.

3. **Immunogenicity Testing:** As the Immunologist suggested, immunogenicity assays could be crucial, especially if unstable conformations expose novel epitopes. Cellular assays to assess potential immune activation or cytokine production would be valuable.

4. **Prioritization of Variants:** To manage resources effectively, prioritize experimental validation for variants showing high computational confidence but possessing potential red flags like unpaired cysteines. This targeted approach would allow us to refine our models accurately without exhaustive testing of all variants.

Aligning these experimental efforts with computational findings will create a robust validation cycle. This approach will not only enhance our confidence in the current models but help improve future iterations of nanobody design by identifying potential pitfalls early in the design process.

I look forward to the final synthesis and action plan from the Principal Investigator in the next round.

## User

This concludes round 2 of 3 of discussion. Principal Investigator, please synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda.

## Principal Investigator

Thank you all for your valuable contributions. Here’s a synthesis of the key points from round two:

1. **Machine Learning and Data Refinement:**
   - The **Machine Learning Specialist** emphasized the feasibility of retraining or fine-tuning models with annotated datasets focused on cysteine-related anomalies. This would involve several weeks to a few months of work, requiring data collection and model adjustment.

2. **Molecular Dynamics Simulations:**
   - The **Computational Biologist** outlined a plan for integrating MD simulations to dynamically assess nanobody stability. These simulations would provide insights into structural integrity under various conditions, potentially informing on the real-world behavior of the nanobodies.

3. **Experimental Validation:**
   - The **Scientific Critic** detailed several experimental assays, including thermal shift assays, SPR, CD spectroscopy, and immunogenicity testing, to empirically validate our computational predictions. These experiments would target variants with computational confidence but potential structural issues.

4. **Biological Relevance:**
   - The **Immunologist** highlighted the importance of understanding the immunogenic profile of these nanobodies, suggesting experimental tests to assess potential immune responses.

**Decisions:**

1. **Model Refinement:**
   - We will initiate a project to refine our machine learning models by incorporating data specific to cysteine-related anomalies. This may include collaboration with databases and leveraging existing datasets to enhance our model training.

2. **MD Simulation Integration:**
   - We will proceed with MD simulations for a selected group of nanobody variants, focusing on those with high predicted binding affinity but potential stability issues due to unpaired cysteines.

3. **Experimental Validation Plan:**
   - We will design and implement an experimental validation plan focusing on stability, binding, and immunogenicity assays for a prioritized subset of nanobody variants.

**Follow-Up Questions:**

- **Machine Learning Specialist:** Are there specific datasets or resources you recommend for obtaining the necessary annotations for cysteine-related training? How can we effectively incorporate this data into our current models?

- **Computational Biologist:** Could you outline a preliminary timeline for the MD simulations, including setup, execution, and analysis phases?

- **Scientific Critic:** Considering resource constraints, how do you propose prioritizing the experimental validation tasks, and what would be the most critical assays to begin with?

- **Immunologist:** Do you have specific recommendations for cell models or assays that could be particularly effective in assessing the immunogenicity of these variants?

Let's proceed with these follow-up questions in mind as we move into the final round of our discussion.

## User

Immunologist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Thank you for the continued focus and direction. Regarding the experimental validation of immunogenicity, I propose the following considerations:

1. **Cell Models:** Utilizing human peripheral blood mononuclear cells (PBMCs) can provide a robust model to assess potential immune activation. This would allow us to monitor cytokine production, T-cell activation, and other immune responses upon exposure to the nanobodies.

2. **In Vitro Assays:** Conduct assays such as ELISA to measure cytokine release and flow cytometry to analyze cell surface markers indicating immune activation. These would help quantify and characterize any immunogenic response.

3. **In Vivo Testing:** If resources permit, small-scale in vivo studies in an appropriate animal model could provide additional insights into the immunogenic profile and systemic stability of these nanobodies.

Prioritizing these efforts should focus first on in vitro assays using PBMCs, as they provide direct insights into potential human immune responses. We should start with variants that exhibit the highest predicted binding affinity and structural stability but have potential immunogenicity concerns due to unpaired cysteines.

I'm eager to hear how the Machine Learning Specialist and others propose to gather and integrate cysteine-specific data, and how we might align our efforts to ensure a comprehensive evaluation of these nanobodies.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Thank you for the thoughtful insights. In terms of obtaining datasets for retraining our models with cysteine-specific annotations, we have a few options:

1. **Public Protein Databases:** Resources such as UniProt and the Protein Data Bank (PDB) can be valuable. These databases might include entries with structural annotations that specify cysteine pairings or related stability data.

2. **Specialized Data Sources:** We should also explore specialized datasets focused on protein stability and folding, such as those used in previous studies or data repositories maintained by research consortia that focus on protein engineering.

3. **Collaborative Opportunities:** Engaging with research groups or consortia that focus on protein structure and stability could provide access to unique datasets or unpublished data that could enhance our model training.

To incorporate this data, we would preprocess these datasets to extract relevant features and annotations related to cysteine presence and stability impacts. This would then be used to fine-tune or retrain our existing models, potentially using transfer learning if the new data is limited in scope.

I propose that we start this process immediately, focusing on data collection and preliminary model adjustments in the next couple of weeks. This will allow us to generate test predictions that can be cross-validated with the upcoming experimental results.

I look forward to hearing from the Computational Biologist about the timeline and integration of MD simulations, as these efforts combined can provide a well-rounded assessment of our nanobodies.

## User

Computational Biologist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

Thank you for the detailed plans and suggestions. For the molecular dynamics (MD) simulations, here's a proposed timeline and integration plan:

1. **Preparation Phase (1-2 Weeks):** 
   - Selection of nanobody variants for simulation, focusing on those with unpaired cysteines and high predicted binding affinity.
   - System setup, including preparation of initial structures, solvation, and ionization of the simulation environment to mimic physiological conditions.

2. **Simulation Execution (3-4 Weeks):** 
   - Run MD simulations for each selected variant. This stage could be parallelized to optimize time, leveraging high-performance computing resources.
   - Duration of simulations will be set to capture significant conformational changes and stability metrics, typically ranging from 100 to 200 nanoseconds.

3. **Analysis Phase (2 Weeks):** 
   - Post-simulation analysis to evaluate structural stability, measure root mean square deviation (RMSD) and fluctuation (RMSF), and identify potential misfolding or aggregation events.
   - Generate stability profiles and compare them with experimental data once available.

By integrating these simulations into our workflow, we can provide valuable insights into the dynamic behavior and structural stability of the nanobodies. The results can help refine our scoring formula and guide experimental validation focus.

I believe aligning these simulations with the machine learning adjustments and experimental plans will provide a comprehensive evaluation framework. I look forward to the Scientific Critic's input on prioritizing our validation tasks to maximize the impact of our efforts.

## User

Scientific Critic, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

Thank you for the detailed plans from each team member. It's encouraging to see a cohesive strategy forming. Here's my perspective on prioritizing and optimizing our efforts:

1. **Prioritization of Experimental Validation:**
   - Given our resource constraints, prioritizing is crucial. Start with in vitro stability assays, such as thermal shift assays, alongside binding assays like SPR to quickly identify variants with promising profiles and eliminate those that are obviously problematic.
   - Focus initially on variants with high computational prediction scores but potential structural red flags, such as unpaired cysteines, to validate both positive and negative predictions from our models.

2. **Aligning Computational Predictions with Experimental Efforts:**
   - Ensure that the MD simulation results are ready to inform the experimental design. Variants that MD suggests are highly dynamic or unstable should have additional scrutiny in experiments.
   - Close collaboration between the machine learning and experimental teams is essential. Validate model predictions with experimental data as soon as possible to refine our models iteratively.

3. **Feedback Loop:**
   - Create a structured feedback loop where computational predictions inform experimental design, and experimental outcomes are used to refine computational models. This will be crucial for improving prediction accuracy over time.
   - Use experimental results to recalibrate and validate the computational models, focusing on how they handle cysteine-related anomalies.

In summary, the integration of computational predictions and experimental validation, coupled with a strategic prioritization of efforts, should enhance the robustness and reliability of our findings. I am optimistic about the outcome if we maintain this interdisciplinary focus and iterative improvement approach. 

I look forward to the Principal Investigator's final synthesis and action plan.

## User

Principal Investigator, please summarize the meeting in detail for future discussions, provide a specific recommendation regarding the agenda, and answer the agenda questions (if any) based on the discussion while strictly adhering to the agenda rules (if any).

As a reminder, here is the agenda for the meeting:

Your team has designed 92 mutated nanobodies (23 each for the wild-type nanobodies H11-D4, Nb21, Ty1, and VHH-72) to improve their binding to the KP.3 variant of the SARS-CoV-2 spike protein receptor binding domain (RBD). Each nanobody has 1-4 mutations relative to the wild-type nanobody. Your team used ESM log-likelihood ratios (ESM LLR) to score the nanobody mutations independent of the antigen, AlphaFold-Multimer to predict the structure of the mutated nanobody in complex with the KP.3 RBD and compute the interface pLDDT (AF ipLDDT) as a metric of binding confidence, and Rosetta to calculate the binding energy of the mutated nanobody in complex with the KP.3 RBD (RS dG) based on the AlphaFold-Multimer predicted structure followed by a Rosetta relaxation. You have ranked the mutant nanobodies and selected the top ones using a weighted score of WS = 0.2 * (ESM LLR) + 0.5 * (AF ipLDDT) - 0. 3 * (RS dG).

An astute reviewer gave the following feedback: "The inclusion of an unpaired Cys in several of the engineered nanobodies would seem to be something that the ESM log-likelihood should have flagged as highly disfavored. Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable? Are there any obvious adjustments to the WS formula that could mitigate this?"

Upon examining the 92 mutated nanobodies, your team found the following:

Nb21 and VHH-72 have no Cys changes (no additions or removals).

Among the 23 Ty21 mutants, two have an added Cys. The Ty1-V32C-G59H-H59Y mutant has an ESM LLR rank of 11 / 23, an AF ipLDDT rank of 22 / 23, and an RS DG rank of 1 / 23 while Ty1-V32C-G59D-D59Y mutant has an ESM LLR rank of 12 / 23, an AF ipLDDT rank of 23 / 23, and an RS DG rank of 2 / 23.

Among the H11-D4 mutants, three have an added Cys. The H11-D4-A14P-Y88V-R27C mutant has an ESM LLR rank of 19 / 23, an AF ipLDDT rank of 3 / 23, and an RS DG rank of 19 / 23 while the H11-D4-A14P-Y88V-R27C-W53Y mutant has an ESM LLR rank of 22 / 23, an AF ipLDDT rank of 8 / 23, and an RS DG rank of 6 / 23 and the H11-D4-A14P-Y88V-R27C-R52S mutant has an ESM LLR rank of 21 / 23, an AF ipLDDT rank of 18 / 23, and an RS DG rank of 12 / 23.

Based on the above information, how would you address the reviewer's questions?

As a reminder, here are the agenda questions that must be answered:

1. Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable?

2. Are there any obvious adjustments to the WS formula that could mitigate this?

Your summary should take the following form.

### Agenda

Restate the agenda in your own words.

### Team Member Input

Summarize all of the important points raised by each team member. This is to ensure that key details are preserved for future meetings.

### Recommendation

Provide your expert recommendation regarding the agenda. You should consider the input from each team member, but you must also use your expertise to make a final decision and choose one option among several that may have been discussed. This decision can conflict with the input of some team members as long as it is well justified. It is essential that you provide a clear, specific, and actionable recommendation. Please justify your recommendation as well.

### Answers

For each agenda question, please provide the following:

Answer: A specific answer to the question based on your recommendation above.

Justification: A brief explanation of why you provided that answer.

### Next Steps

Outline the next steps that the team should take based on the discussion.

## Principal Investigator

### Agenda

The main agenda for this meeting was to address the reviewer's feedback regarding the ranking of mutated nanobodies, specifically the surprising high ranking of variants with unpaired cysteines. We aimed to understand if structural predictions or binding energies were skewing the results, and to identify potential adjustments to our weighted scoring formula to more accurately reflect the biological viability of these variants.

### Team Member Input

- **Immunologist:** Highlighted the potential instability and immunogenicity issues that unpaired cysteines might introduce. Suggested implementing a penalty in the WS formula and recommended using PBMCs for immunogenicity testing.

- **Machine Learning Specialist:** Discussed the limitations of current ML models in flagging unpaired cysteines. Proposed retraining models with cysteine-specific data and incorporating penalties for structural anomalies into the WS formula.

- **Computational Biologist:** Proposed using molecular dynamics simulations to assess nanobody stability over time, suggesting a timeline for preparation, execution, and analysis phases. Emphasized integrating these results with experimental validation.

- **Scientific Critic:** Recommended prioritizing experimental validations, especially stability and binding assays, to empirically validate computational predictions. Suggested a feedback loop for continuous model refinement based on experimental outcomes.

### Recommendation

Based on the team's input, my expert recommendation is to:

1. **Refine the Weighted Score Formula:** Introduce a penalty for unpaired cysteines within the WS formula. This penalty should be dynamically assessed, perhaps using machine learning models trained specifically to recognize such structural anomalies.

2. **Conduct Comprehensive Experimental Validation:** Prioritize variants with high binding scores but potential structural issues for experimental validation. Use in vitro assays to assess stability and binding, and incorporate immunogenicity testing in human cell models.

3. **Integrate Molecular Dynamics Simulations:** Use MD simulations to provide dynamic insights into nanobody stability, focusing on variants where computational predictions suggest discrepancies.

This approach balances computational predictions with empirical data, ensuring that our scoring system reflects the true biological potential of the variants.

### Answers

**1. Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable?**

- **Answer:** Yes, the ipLDDT and RS dG values likely contributed to the high rankings despite unfavorable ESM LLR scores.
- **Justification:** The high confidence in predicted structure and binding affinity could overshadow sequence-based concerns, leading to a skewed ranking of variants with unpaired cysteines.

**2. Are there any obvious adjustments to the WS formula that could mitigate this?**

- **Answer:** Yes, introduce a penalty for unpaired cysteines and adjust weightings to better balance sequence and structure-based evaluations.
- **Justification:** This adjustment will ensure that structural instability related to unpaired cysteines is accounted for, preventing false positives in our rankings.

### Next Steps

1. **Develop and Test WS Formula Adjustments:** Implement the proposed penalty for unpaired cysteines and test the impact of these changes on the ranking of variants.

2. **Retrain ML Models:** Begin data collection and preprocessing for cysteine-specific model training to improve prediction accuracy.

3. **Conduct MD Simulations:** Initiate MD simulations for selected variants to provide dynamic stability insights.

4. **Plan and Execute Experimental Validations:** Design and prioritize binding, stability, and immunogenicity assays to validate the computational predictions.

By following these steps, we will strengthen our scoring approach and ensure the selection of nanobodies with true potential for effective binding to the SARS-CoV-2 spike protein RBD.

