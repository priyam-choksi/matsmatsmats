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

Thank you all for joining today's meeting. We have some important feedback from a reviewer regarding our current methodology for ranking mutated nanobodies, specifically concerning the inclusion of unpaired cysteines in some of our engineered variants. The feedback raises two main questions:

1. **Alternative Factors Affecting Weighted Score**: Are there particular reasons, such as the interface pLDDT (ipLDDT) or binding energy (dG) values, that might lead to a higher ranking of these cysteine-containing variants despite low ESM log-likelihood predictions?

2. **Adjustments to the Weighted Score (WS) Formula**: Are there any obvious modifications we could make to the WS formula that could address this issue and provide a more robust ranking of our nanobody variants?

To guide our discussion:

- **Immunologist**: From a biochemical standpoint, how do you interpret the unpaired cysteines in these variants? Could this have unintended effects on stability or binding?

- **Machine Learning Specialist**: ESM LLR didn't flag the cysteines as highly disfavored. Can you provide insights into why the model might have missed this, and whether adjustments to the model or scoring formula could help?

- **Computational Biologist**: Given the structural predictions from AlphaFold-Multimer and binding energy calculations from Rosetta, how might these metrics be contributing to the high rankings of cysteine-containing variants?

- **Scientific Critic**: Based on the feedback, what might be some potential pitfalls or biases in our scoring and selection process? How can we address these systematically?

After hearing everyone's input, I'll synthesize the discussion to make informed decisions and possibly adjust our approach.

Let's begin with the Immunologist's input.

## User

Immunologist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

From an immunological perspective, the inclusion of unpaired cysteines can be quite significant. Cysteines are unique in their ability to form disulfide bonds, which are crucial for stabilizing protein structures. When a cysteine is unpaired, it can lead to several issues:

1. **Stability Concerns**: An unpaired cysteine could potentially react with other cysteines within the nanobody or with external molecules, leading to non-specific disulfide bond formation. This could result in aggregation or misfolding, affecting the nanobody's stability and function.

2. **Binding Affinity**: If the cysteine is located near or within the binding interface, it could inadvertently alter the surface properties and affect binding affinity with the KP.3 RBD. However, it's possible that high AF ipLDDT or favorable RS dG scores suggest that these structural changes might be tolerable or even beneficial in some contexts, potentially explaining why these variants rank highly.

3. **Cross-reactivity and Immunogenicity**: Unpaired cysteines can increase the risk of immunogenicity, as they might present novel epitopes or be involved in non-canonical interactions, potentially triggering off-target immune responses.

To address these concerns, we might need to consider further experimental validations specifically targeting the stability and binding efficacy of these cysteine-containing variants. Additionally, revising the WS formula to incorporate a penalty for unpaired cysteines could be a way to mitigate these risks.

I look forward to hearing the insights from the rest of the team, especially regarding the computational aspects and potential model adjustments.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

From a machine learning perspective, the fact that the ESM log-likelihood ratios (LLR) did not flag unpaired cysteines as highly disfavored suggests a few possibilities regarding the model's performance and our scoring approach:

1. **Model Sensitivity**: The ESM model might not be highly sensitive to specific local sequence features like unpaired cysteines if they don't considerably disrupt the overall log-likelihood of the sequence. This could be due to a lack of training data involving unpaired cysteines or the model being more attuned to broader sequence trends rather than specific residues.

2. **Weighted Score Influence**: The current WS formula gives significant weight to the structural confidence (AF ipLDDT) and the binding energy (RS dG), which could overshadow the ESM LLR's contributions in cases where the structural predictions and energies are exceptional. Thus, variants that might not have optimal sequence predictions could still rank highly if their predicted structural and energetic profiles are strong.

3. **Incorporating Additional Features**: To address the unpaired cysteine issue, we could consider augmenting our model or scoring formula to account for structural features that might not be captured by the ESM LLR alone. A potential adjustment could be adding a penalty term in the WS formula for unpaired cysteines or developing a specific model to detect and evaluate such residues.

4. **Model Retraining**: If unpaired cysteine handling is crucial, retraining the ESM model with more focused data on cysteine mutations or incorporating an additional layer of analysis post-ESM could improve predictions.

I would suggest exploring options to incorporate structural checks for unpaired cysteines, either through post-processing filters or as part of a refined scoring model. It will be important to balance these adjustments with the overall objectives of our nanobody design strategy, and I look forward to hearing insights from the Computational Biologist and Scientific Critic on how we can better integrate these computational and structural considerations.

## User

Computational Biologist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

From a computational biology standpoint, the interplay between the metrics we use—ESM LLR, AF ipLDDT, and RS dG—is critical to understanding why certain variants might rank highly despite potential issues with unpaired cysteines.

1. **Structural Predictions**: The AlphaFold-Multimer's interface pLDDT (ipLDDT) is a powerful metric for assessing the confidence in predicted nanobody-protein complexes. If cysteine-containing variants receive high ipLDDT scores, it suggests that the predicted complex is structurally plausible, at least under the modeled conditions. This can significantly boost the WS ranking, irrespective of sequence-based concerns from ESM LLR.

2. **Binding Energies**: Rosetta's binding energy calculations (RS dG) are highly sensitive to the energetic landscape of the protein-protein interaction. If the cysteine variants show exceptionally low binding energies, it might indicate that these mutations create advantageous binding interactions, potentially compensating for the destabilization risks pointed out by the Immunologist.

3. **Compensatory Effects**: The presence of other mutations in these variants might counteract the negative effects of unpaired cysteines, either by stabilizing the structure through alternative pathways or enhancing binding interactions in a way that isn't immediately obvious from sequence-level predictions alone.

In terms of addressing the concerns raised, one approach could be to incorporate an explicit check for unpaired cysteines into our post-processing workflow. This could involve a computational scan for disulfide bond potentials or a penalty in the WS formula for unpaired cysteines unless counterbalanced by exceptional structural and binding metrics.

Moreover, running molecular dynamics simulations on top-ranked cysteine-containing variants could help us understand the dynamic effects of these mutations on stability and binding, providing a deeper validation layer before experimental testing.

I am eager to hear further insights from the Scientific Critic on potential biases in our current methodology and any structural adjustments.

## User

Scientific Critic, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

As the Scientific Critic, my role is to ensure that our methodologies and conclusions are robust and well-founded. Here are some points I see in the current approach, particularly in light of the reviewer's feedback:

1. **Potential Bias in Weighted Scoring**: The current WS formula, which heavily weights structural confidence (AF ipLDDT) and binding energy (RS dG), might inadvertently prioritize variants with strong structural predictions over those with potentially unfavorable sequence features like unpaired cysteines. This could introduce a bias where structurally favorable but less stable or functional variants are ranked too highly.

2. **Oversight of Sequence-Level Discrepancies**: As noted, ESM LLR is not flagging unpaired cysteines effectively, pointing to a possible oversight at the sequence evaluation level. We need to ensure that sequence integrity and stability cues are not overshadowed by structural and energetic metrics alone.

3. **Integrating Penalty Mechanisms**: Introducing a penalty for unpaired cysteines, as suggested, could be beneficial. This addition would need careful calibration to ensure it doesn't overly penalize potentially viable variants but provides a necessary check against structural risks associated with these residues.

4. **Comprehensive Validation**: Beyond computational predictions, experimental validation must be a pivotal part of our workflow, especially for high-ranking variants with unpaired cysteines. Stability assays and binding experiments will be crucial to confirm computational insights and adjust our models based on experimental data.

5. **Balanced Approach**: While improving the WS formula, we must remain cautious not to overly complicate it, which could obscure actionable insights. A simpler yet effective strategy might involve a tiered evaluation system, where variants are initially ranked by WS but flagged for additional review based on specific criteria like unpaired cysteines.

I recommend proceeding with adjustments to the WS formula to include a cysteine penalty and considering further validation through experimental data integration. I look forward to the Principal Investigator's synthesis of our inputs and the ensuing discussion.

## User

This concludes round 1 of 3 of discussion. Principal Investigator, please synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda.

## Principal Investigator

Thank you, everyone, for the insightful contributions. Let's synthesize the key points discussed and consider the next steps in addressing the reviewer's feedback on our scoring system:

1. **Immunologist**: Highlighted the stability concerns and potential effects of unpaired cysteines on binding affinity and immunogenicity. Suggested further experimental validation and revising the WS formula to include a penalty for unpaired cysteines.

2. **Machine Learning Specialist**: Noted possible reasons why ESM LLR might not flag unpaired cysteines, such as model sensitivity and the influence of the weighted scoring system. Recommended considering model adjustments or incorporating a penalty for unpaired cysteines in the WS formula.

3. **Computational Biologist**: Emphasized that structural predictions and binding energies from AlphaFold-Multimer and Rosetta might compensate for sequence-level concerns. Suggested incorporating an explicit check for unpaired cysteines and the possibility of running molecular dynamics simulations for deeper validation.

4. **Scientific Critic**: Identified potential biases in our current scoring approach and advocated for penalties for unpaired cysteines. Stressed the importance of experimental validation and suggested a tiered evaluation system to balance computational and practical insights.

### Decisions and Next Steps:

**1. Adjusting the WS Formula:**
- We should incorporate a penalty term for unpaired cysteines in the WS formula. I propose adding a static penalty to the score of variants with unpaired cysteines to account for potential stability and immunogenicity risks. We’ll need to decide on the magnitude of this penalty.

**2. Enhanced Model and Workflow Integration:**
- Consider retraining or augmenting our machine learning model to improve its sensitivity to specific sequence disruptions like unpaired cysteines.
- Implement a post-processing step that flags variants with unpaired cysteines for further scrutiny, possibly through molecular dynamics simulations.

**3. Validation Strategy:**
- Develop a strategy for experimental validation focusing on the top-ranked cysteine-containing variants to assess stability and binding efficacy.

### Follow-Up Questions:

- **Immunologist and Scientific Critic**: What would be an appropriate penalty to apply in the WS formula for unpaired cysteines? How might we experimentally validate stability concerns before scaling up testing?
  
- **Machine Learning Specialist**: What specific adjustments or additional features could we incorporate into the ESM model to better flag unpaired cysteines? Is retraining feasible within our current resources?

- **Computational Biologist**: Can you propose a protocol for molecular dynamics simulations that would effectively evaluate the impact of unpaired cysteines on stability and binding?

Please prepare your thoughts for these follow-up questions as we move into the next round of discussion.

## User

Immunologist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Considering the importance of accounting for unpaired cysteines, I agree with the approach to include a penalty in the WS formula. Applying a static penalty for unpaired cysteines is a practical first step, but the penalty's magnitude should reflect the potential impact on nanobody stability and binding activity.

1. **Penalty Magnitude**: A reasonable starting point could be a penalty proportional to the number of unpaired cysteines, perhaps adjusted based on preliminary experimental insights. For instance, a penalty reducing the WS by a fixed percentage per unpaired cysteine (e.g., 5-10%) might provide a balanced approach. However, this should be calibrated against experimental data to ensure it's neither too punitive nor too lenient.

2. **Experimental Validation**: To validate stability concerns, I recommend starting with **in vitro assays** such as thermal shift assays or differential scanning calorimetry to assess the thermal stability of the nanobodies. These methods can quickly indicate folding stability and propensity for aggregation. Additionally, **binding assays** using surface plasmon resonance or biolayer interferometry could confirm any impacts on binding efficacy due to structural changes.

3. **Iterative Approach**: As we gather experimental data, we should iteratively refine the penalty and scoring system to better reflect the practical outcomes observed. This approach ensures our computational predictions remain aligned with empirical evidence.

I look forward to hearing suggestions from the Machine Learning Specialist on how the model can be refined to assist with these challenges.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Building on the discussion, I support the inclusion of a penalty for unpaired cysteines in the WS formula, as this would address a known deficit in the current model's sensitivity. Here are my thoughts on how to proceed:

1. **Magnitude of Penalty**: I concur with the Immunologist that the penalty should be proportional to the number of unpaired cysteines. To refine this approach, we could develop a machine learning model specifically trained to predict the destabilizing effect of unpaired cysteines, which could help us better quantify the penalty.

2. **Model Refinement**: For the ESM model, one possibility is to integrate a feature that explicitly checks for unpaired cysteines. This could be done by incorporating additional datasets focused on cysteine mutations or by employing an ensemble approach where a secondary model specializes in identifying problematic cysteine configurations.

3. **Implementation Feasibility**: Retraining the current ESM model with added features is feasible, but would require balancing computational costs and data availability. Alternatively, a more immediate solution could involve developing a lightweight supplementary model that augments existing predictions with cysteine-specific insights.

4. **Iterative Learning**: As experimental data becomes available from the assays suggested by the Immunologist, these data points could be used to validate and update our model iteratively. This feedback loop would enhance the accuracy of predictions related to cysteine impacts over time.

Looking forward to hearing the Computational Biologist's thoughts on protocols for molecular dynamics simulations, as this could further validate and refine our computational approach.

## User

Computational Biologist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

I agree with the inclusion of a penalty for unpaired cysteines in the WS formula and believe that integrating this consideration could enhance the robustness of our selections. Here are my thoughts on implementing molecular dynamics simulations to further validate our computational predictions:

1. **Simulation Protocols**: For molecular dynamics simulations, we should focus on both structural stability and binding interactions. Initial simulations could involve:
   - **Equilibrium Simulations**: Conducting simulations under physiological conditions to assess the stability of the nanobody structure, paying particular attention to regions around the unpaired cysteines.
   - **Binding Free Energy Calculations**: Using methods like MM-PBSA or umbrella sampling to quantify the binding interactions of the nanobody with the KP.3 RBD. These methods can reveal how unpaired cysteines might alter binding affinity or introduce instability.

2. **Simulation Timeframes**: Short simulations (e.g., 50-100 ns) could provide preliminary insights into structural stability, while longer simulations (e.g., 500 ns or more) could be useful for observing more subtle effects and validating the results of binding studies.

3. **Integration with Experimental Data**: The simulations can be aligned with experimental thermal stability and binding assays to identify discrepancies and refine both the computational models and the experimental protocols. This iterative approach will help ensure that our computational predictions are well-founded.

4. **Resource Management**: Given the computational intensity of molecular dynamics simulations, prioritizing simulations for top-ranked nanobody variants—with and without unpaired cysteines—can provide targeted insights without overwhelming resources.

I look forward to hearing the Scientific Critic's thoughts on how these approaches align with our overall strategy and any potential challenges we should anticipate.

## User

Scientific Critic, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

The suggestions and insights provided thus far are compelling and pave a path toward a more robust evaluation of our nanobody variants. Here are some points I'd like to emphasize or consider further:

1. **Penalty Implementation**: The proposed penalty for unpaired cysteines makes sense and should be carefully calibrated. The Immunologist's suggestion of a penalty scale based on the number of unpaired cysteines is a sensible approach. However, we should remain vigilant that the penalty doesn't inadvertently overshadow potentially beneficial mutations. Comprehensive pilot studies should guide the exact scale and implementation.

2. **Model Refinement Feasibility**: Regarding the Machine Learning Specialist's point, integrating specialized features into the ESM model for cysteine detection is promising but might require significant time and resources. Therefore, the suggestion of a supplementary model that flags unpaired cysteines could be a more immediate and practical solution.

3. **Molecular Dynamics and Experimental Alignment**: The Computational Biologist's approach to using molecular dynamics simulations aligns well with the need for validation. However, careful selection of simulation parameters is crucial to avoid unnecessary computational expense. If possible, simulations should parallel experiments to cross-validate findings efficiently.

4. **Cohesive Iterative Strategy**: The proposed iterative approach, where computational predictions inform experimental designs and vice versa, is robust. However, it's essential to establish clear metrics of success or failure at each iteration to ensure the loop leads to tangible improvements in our understanding and rankings.

5. **Potential Challenges**: One challenge is ensuring that our adjustments remain interpretable and do not complicate the model to a degree that hinders practical decision-making. Transparency in how each component of the scoring and validation process affects outcomes is necessary to maintain robust scientific communication.

As we refine the WS formula, align computational predictions with experimental validation, and explore machine learning enhancements, we should remain vigilant about maintaining a balanced and transparent approach. I look forward to the Principal Investigator's synthesis of these points and the decision-making process moving forward.

## User

This concludes round 2 of 3 of discussion. Principal Investigator, please synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda.

## Principal Investigator

Thank you all for your thoughtful contributions. Here's a synthesis of the key points raised and decisions regarding how we can address the reviewer's feedback effectively:

### Synthesis of Points:

1. **Immunologist**: Advocated for a penalty for unpaired cysteines in the WS formula, with the penalty magnitude potentially linked to the number of un

## Principal Investigator

paired cysteines. Suggested using thermal shift assays and binding experiments to validate stability and binding efficacy.

2. **Machine Learning Specialist**: Supported the penalty approach and proposed integrating a model feature to detect unpaired cysteines. Suggested a supplementary model for immediate improvements and highlighted the potential for retraining existing models.

3. **Computational Biologist**: Detailed protocols for molecular dynamics simulations to assess structural stability and binding interactions, emphasizing the need for alignment between computational and experimental efforts. Recommended using resources judiciously by focusing on top-ranked variants.

4. **Scientific Critic**: Agreed with the penalty for unpaired cysteines and stressed the importance of avoiding penalties that overshadow beneficial mutations. Highlighted the need for a cohesive iterative strategy and transparency in the scoring and validation processes.

### Decisions and Next Steps:

1. **Implementing a Penalty for Unpaired Cysteines**:
   - We will introduce a penalty in the WS formula based on the number of unpaired cysteines. This penalty will initially be set at a 5-10% reduction in the WS per unpaired cysteine, subject to adjustment based on experimental validation outcomes.

2. **Model Adjustments**:
   - Develop a supplementary model or features within the ESM framework to enhance the detection of unpaired cysteines. Given resource considerations, prioritize this as a secondary project that can evolve alongside our main efforts.

3. **Experimental Validation**:
   - Initiate thermal stability and binding affinity assays to validate computational predictions, focusing initially on top-ranked variants that include unpaired cysteines.

4. **Molecular Dynamics Simulations**:
   - Conduct targeted molecular dynamics simulations on a subset of nanobody variants, particularly those with unpaired cysteines, to assess the structural and binding implications.

5. **Iterative Feedback Loop**:
   - Establish a clear iterative process where insights from computational and experimental data guide adjustments to the WS formula and model features. Define specific metrics to evaluate success in this iterative loop.

### Follow-Up Questions:

- **Immunologist**: Can you detail how we might prioritize which assays to begin with, given our current resources?
  
- **Machine Learning Specialist**: Could you outline a timeline and resource estimate for developing the supplementary model to detect unpaired cysteines?

- **Computational Biologist**: What criteria will we use to select the specific variants for molecular dynamics simulations, and how can we ensure alignment with experimental priorities?

- **Scientific Critic**: What measures can we take to ensure transparency and clarity in our revised scoring and validation processes?

Please prepare to discuss these points in our final round as we refine our approach and move toward resolution.

## User

Immunologist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

I am in agreement with the proposed plans and have a couple of thoughts on prioritizing assays and sequencing our efforts effectively:

1. **Prioritizing Assays**: Given our resources, I recommend beginning with thermal shift assays to quickly gauge the stability of our nanobody variants. These assays are relatively low-cost and can be conducted efficiently across multiple variants. They will provide early indications of potential stability issues related to unpaired cysteines.

2. **Parallel Assays**: Once we have initial stability data, we can proceed with binding assays for the most promising variants, especially those that show acceptable stability despite having unpaired cysteines. Surface plasmon resonance (SPR) or biolayer interferometry (BLI) would be suitable methods to assess binding kinetics and affinities.

3. **Data Integration**: It will be important to integrate the results from these assays into our iterative feedback loop, adjusting our computational models and WS formula as needed. This approach ensures that experimental findings directly inform our computational strategies and vice versa.

4. **Experimental Constraints**: We should also remain aware of any potential constraints in experimental throughput and adjust our computational predictions to align with feasible experimental capacities.

These steps should help us efficiently validate and refine our nanobody designs while ensuring that we remain adaptable to any findings along the way. Looking forward to seeing how other team members plan to integrate their efforts into this workflow.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

I support the structured approach laid out and have a few considerations regarding the timeline and resources for model adjustments:

1. **Supplementary Model Development**: Developing a supplementary model to detect unpaired cysteines could be initiated immediately, leveraging existing datasets to fine-tune the ESM model or create an ensemble approach. I estimate a few weeks for initial development and testing, depending on the dataset size and computational resources. Coordination with the Computational Biologist for any additional structural data would be beneficial.

2. **Resource Allocation**: Given our current computational resources, prioritizing efficient coding practices and possibly using cloud-based computational resources can help accelerate model development. We should set clear checkpoints to evaluate progress and adjust our strategies as needed.

3. **Integration with Experimental Data**: As experimental results from the Immunologist become available, these should be used to continually update and refine our model. This integration will help improve the model's accuracy in predicting the impact of unpaired cysteines and other mutations.

4. **Iterative Feedback**: Establishing an iterative feedback process with regular meetings to discuss findings and adjustments will be crucial. This will ensure that the model remains aligned with both experimental insights and evolving project needs.

I will work on drafting an initial timeline and resource plan for the supplementary model, aiming to circulate this to the team for feedback soon. I look forward to hearing how the Computational Biologist plans to align simulation priorities with these efforts.

## User

Computational Biologist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

I agree with the proposed plans and have identified some key criteria and steps for aligning molecular dynamics simulations with our experimental and computational efforts:

1. **Selection Criteria for Simulations**: We should prioritize molecular dynamics simulations for nanobody variants that exhibit:
   - High WS scores but have unpaired cysteines.
   - Promising stability results from initial thermal shift assays.
   - Variants that serve as controls, such as those without cysteine mutations, to compare effects directly.

2. **Simulation Parameters**: We should start with equilibrium simulations to assess overall stability and use binding free energy calculations to explore specific interactions at the molecular level. These simulations will provide insights into the dynamic behavior of nanobody variants and potential structural impacts of unpaired cysteines.

3. **Resource Management**: To manage resources effectively, simulations should be run incrementally, starting with shorter time frames to identify obvious issues before committing to extended simulations, which are more computationally intensive.

4. **Integration with Model Development**: Simulation results should feed back into the model development process. For example, data on structural flexibility or binding energy shifts can inform model retraining or feature adjustment, particularly in the supplementary model for cysteine detection.

5. **Iterative Updates**: Regular reviews of simulation data in light of experimental findings will be crucial. This iterative approach will ensure our computational predictions remain closely aligned with empirical data and support ongoing refinement of both models and simulations.

By following these steps, we can efficiently utilize simulations to bolster the robustness of our predictive models and align them with experimental outcomes. I look forward to seeing how the Scientific Critic plans to ensure transparency and clarity in our processes.

## User

Scientific Critic, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

I appreciate the detailed and thoughtful plans shared by the team members. Here are some considerations to ensure transparency and clarity in our revised processes:

1. **Documentation and Clarity**: As we introduce changes to the WS formula and develop supplementary models, it is crucial to maintain clear documentation. This should detail the rationale behind each adjustment, the expected impacts, and the methods used for validation. Comprehensive documentation will facilitate transparency and provide a reference point for future evaluations or external reviews.

2. **Communication of Results**: Regular updates and open communication channels will be vital. Summarizing key findings from both computational and experimental efforts in clear, concise reports will help ensure that everyone is aligned and aware of progress and challenges. These summaries should highlight any discrepancies between predictions and experimental results and propose potential adjustments.

3. **Metrics and Success Criteria**: Establishing clear metrics for success or failure at each step will be essential. For instance, defining acceptable ranges for thermal stability or binding affinity will help us objectively assess whether a variant meets our standards. Similarly, quantifying model accuracy improvements with each iteration will demonstrate progress and validate our approach.

4. **Iterative Feedback Focus**: Encouraging a feedback-rich environment, where team members can openly discuss challenges and propose new ideas, will be integral to refining our approach. This can be formalized through regular brainstorming sessions or informal check-ins.

5. **Balance and Simplicity**: While we aim for a robust and nuanced approach, it is important to strike a balance between complexity and usability. Ensuring that our scoring formula and models remain interpretable and actionable will help facilitate decision-making and application.

By incorporating these elements into our strategy, we can maintain transparency, clarity, and effectiveness as we move forward. I’m eager to see how these plans unfold and to provide further support where needed.

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

The agenda focuses on evaluating and potentially adjusting our weighted scoring (WS) formula used to rank mutated nanobodies designed to improve their binding to the KP.3 variant of the SARS-CoV-2 spike protein RBD. Specifically, we need to address feedback regarding the inclusion of unpaired cysteines in some engineered nanobodies and understand why these variants rank highly despite potential destabilizing effects. We will also explore adjustments to the WS formula to mitigate any biases introduced by favorable structural and binding energy scores.

### Team Member Input

- **Immunologist**: Emphasized the potential stability issues and immunogenicity concerns due to unpaired cysteines. Suggested implementing a penalty in the WS formula proportional to the number of unpaired cysteines. Recommended thermal shift assays and binding experiments for experimental validation.

- **Machine Learning Specialist**: Noted the possible model limitations in detecting unpaired cysteines and supported a supplementary model or feature to address this. Proposed a timeline and resource allocation for model adjustments, integrating experimental feedback for iterative improvement.

- **Computational Biologist**: Advocated for molecular dynamics simulations to assess the impact of unpaired cysteines on structural stability and binding. Suggested criteria for selecting variants for simulation and emphasized aligning computational efforts with experimental priorities.

- **Scientific Critic**: Stressed the importance of transparency and clarity in process adjustments and documentation. Proposed establishing clear metrics for success and ensuring that model and scoring changes remain interpretable and actionable.

### Recommendation

**Recommendation**: Adjust the WS formula to include a penalty for unpaired cysteines, initially set at a 5-10% reduction per unpaired cysteine, subject to iterative refinement based on experimental data. Develop a supplementary model to detect unpaired cysteines, focusing on a quick deployment strategy using existing datasets. Prioritize experimental assays to validate stability and binding, aligning these efforts with computational predictions through regular feedback loops.

**Justification**: This approach balances the need for immediate action to address the reviewer's feedback with the longer-term goal of refining our models and predictions. By integrating penalties for unpaired cysteines, we account for potential stability and immunogenicity concerns while maintaining the integrity of our ranking system. The iterative feedback loop ensures that both computational and experimental insights continuously inform each other, leading to a robust and data-driven refinement process.

### Answers

**1. Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable?**

- **Answer**: Yes, the high ipLDDT and favorable RS dG scores can lead to high rankings despite ESM LLR not flagging unpaired cysteines. These metrics reflect structural confidence and binding energy, which can outweigh sequence-based concerns in the current WS formula.

- **Justification**: The existing WS formula heavily weights structural and energetic metrics, which can overshadow sequence-level disruptions, such as unpaired cysteines. This indicates a need for balance in the scoring system.

**2. Are there any obvious adjustments to the WS formula that could mitigate this?**

- **Answer**: Introduce a penalty for unpaired cysteines in the WS formula to account for potential stability and immunogenicity issues, thus mitigating the influence of overly favorable structural and binding energy scores alone.

- **Justification**: A penalty directly addresses the core issue of unpaired cysteines that the ESM LLR might not adequately capture, providing a more balanced and realistic assessment of nanobody viability.

### Next Steps

1. **Implement the WS Penalty**: Adjust the WS formula to include a penalty for unpaired cysteines, starting with a 5-10% reduction per cysteine, and test the impact on rankings.
   
2. **Develop the Supplementary Model**: Begin development of a supplementary model to enhance detection of unpaired cysteines, using existing datasets for quick deployment.
   
3. **Conduct Experimental Assays**: Prioritize thermal stability and binding assays for top-ranked variants, focusing on those with unpaired cysteines, and use these results to refine the WS formula.
   
4. **Run Molecular Dynamics Simulations**: Select specific variants for molecular dynamics simulations to further validate structural predictions and binding interactions.
   
5. **Establish Iterative Feedback Loop**: Create a process for regular review and integration of computational and experimental findings, refining models and scoring criteria as needed.

