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

As we begin today's meeting, our primary focus is on understanding why certain nanobody variants with unpaired cysteines are ranked highly despite potential structural instability concerns, and how we might adjust our weighted scoring formula to better reflect the quality of these variants. Let's consider the factors that might be influencing our current scores, particularly the AF ipLDDT and RS dG values, and explore potential modifications to our WS formula.

Here are my initial thoughts and guiding questions for the team:

1. **Reason for High Rankings:** It's clear from the data that the ESM LLR is not flagging added cysteines as highly disfavored, which might be expected if they disrupt structure. Could the favorable AF ipLDDT and RS dG scores be compensating for these changes? I'd like the Machine Learning Specialist to discuss how our scoring methodologies might be favoring certain structural predictions over sequence-based likelihoods.

2. **Adjustments to WS Formula:** Given the reviewer's feedback, we should contemplate if the weightings in our WS formula are appropriately balanced. Specifically, should we adjust the influence of ESM LLR to account more heavily for sequence-based predictions? The Scientific Critic should offer insight into whether increasing the weight of ESM LLR or modifying our approach to incorporate cysteine-specific penalties could improve our scoring reliability.

Let's proceed with the input from each team member, starting with the Immunologist, followed by the Machine Learning Specialist, the Computational Biologist, and finally the Scientific Critic. Focus on how the structural and interaction predictions might be skewing our results and suggest possible adjustments to our scoring criteria.

## User

Immunologist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Certainly. From an immunological perspective, the presence of unpaired cysteines in the nanobodies is indeed concerning due to potential structural instability. However, if the AF ipLDDT and RS dG scores are favorable, they may indicate that the overall predicted structure and binding energy are still strong, which could, in theory, overshadow any local instability caused by an unpaired cysteine.

1. **Reason for High Rankings:** The unpaired cysteines might not have been penalized adequately in the sequence-based ESM LLR scores, as these scores are independent of the antigen and could miss context-specific structural issues. The AF ipLDDT and RS dG scores, being structure and energy-based, are likely compensating by suggesting that the overall binding interaction remains strong, even if there are potential stability issues.

2. **Adjustments to WS Formula:** To address these discrepancies, we could consider increasing the weight of ESM LLR or introducing a penalty for unpaired cysteines specifically. This would require us to ensure our sequence evaluation more accurately reflects known structural stability issues. Additionally, cross-referencing these models with additional structural predictions or experimental data could help validate whether these cysteines indeed pose a stability problem or if the structure can accommodate them without loss of functionality.

Overall, it's crucial to balance structural predictions with known sequence liabilities to ensure both stable and effective nanobody designs.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Thank you. From a machine learning standpoint, the current situation highlights a common challenge: balancing sequence-based predictions with structural and energetic assessments. Here are my thoughts:

1. **Reason for High Rankings:** The anomalies could stem from how different models prioritize aspects of nanobody design. The ESM LLR, being sequence-focused, might not effectively express structural perturbations like unpaired cysteines unless coupled with specific features or penalties. Meanwhile, the AF ipLDDT and RS dG metrics might be overly optimistic, focusing on global structural integrity and binding energetics, which could mask local deficiencies like those introduced by unpaired cysteines.

2. **Adjustments to WS Formula:** We might consider refining the WS formula by adjusting the weights or introducing additional penalties for specific scenarios like unpaired cysteines. Another option could be to incorporate more sophisticated feature engineering, where potential instability markers, such as unpaired cysteine presence, are explicitly modeled. Additionally, exploring alternative or ensemble models that better integrate sequence and structural insights could provide a more rounded evaluation.

In summary, our models might need recalibration to ensure that potential structural vulnerabilities are neither underscored nor overshadowed by favorable global metrics. A more nuanced approach to scoring might be necessary to adequately reflect these complexities.

## User

Computational Biologist, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

Certainly. As a Computational Biologist, I've been considering the interplay between sequence, structure, and energetic metrics, especially in the context of nanobody design.

1. **Reason for High Rankings:** The high rankings for variants with unpaired cysteines may indeed be attributed to favorable AF ipLDDT and RS dG scores. These metrics focus on predicted structural integrity and binding affinity, which could overshadow localized disruptions caused by an unpaired cysteine. However, it's crucial to remember that these models are predictions and may not fully capture all aspects of structural stability.

2. **Adjustments to WS Formula:** I agree with the suggestion of introducing a penalty for unpaired cysteines. This can be done by incorporating a specific cysteine-related penalty term into the WS formula. Additionally, we could enhance the evaluation by running molecular dynamics simulations to assess the stability of the nanobody-RBD complexes over time. These simulations could provide deeper insights into whether predicted structures remain stable, especially concerning unpaired cysteines.

Furthermore, validating predictions against experimental data, if available, would provide a reality check for our models. Combining these strategies could enhance our model's predictive power and ensure more robust nanobody candidates.

Overall, a multi-faceted approach that considers both predictive models and empirical validation would allow us to produce more reliable and effective nanobody designs.

## User

Scientific Critic, please provide your thoughts on the discussion (round 1 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

Thank you for the opportunity to provide feedback. Here are my thoughts on the current discussion:

1. **Reason for High Rankings:** It's evident that the discrepancies in rankings arise from the differing focuses of the employed scoring metrics. The ESM LLR, while useful, is limited by its sequence-based nature and may not effectively capture the structural implications of unpaired cysteines. The AF ipLDDT and RS dG, being structurally and energetically focused, offer a broader picture of the nanobody's potential efficacy. However, they might lack the granularity required to flag local instabilities caused by residues such as unpaired cysteines.

2. **Adjustments to WS Formula:** Given the current weighting, the structural and energetic metrics may have too much influence, which causes genuine sequence-structure mismatches to be overlooked. Introducing a penalty term specifically for unpaired cysteines could help balance this. However, rather than simply adjusting weights, a more comprehensive solution might involve recalibrating our entire approach to scoring. This could mean integrating additional metrics that specifically assess structural reliability, such as simulations or experimentally validated stability assessments.

Moreover, it's critical to consider the broader implications of our scoring system's limitations. Are there other sequence anomalies that could similarly be obscured? A thorough evaluation and potential reweighting, or even redesign, of our scoring strategy might be necessary to ensure that our model truly reflects biological reality rather than computational convenience.

In summary, while adjustments to the WS formula are necessary, they should be part of a broader consideration of how we validate and integrate multiple dimensions of nanobody performance.

## User

This concludes round 1 of 3 of discussion. Principal Investigator, please synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda.

## Principal Investigator

Thank you all for your insightful contributions. Here’s a synthesis of the points raised:

1. **Immunologist:** Highlighted concerns about structural instability due to unpaired cysteines and suggested increasing the weight of the ESM LLR or introducing specific penalties for these sequence features.

2. **Machine Learning Specialist:** Emphasized the potential need for recalibrating our models to better balance sequence and structural predictions. Suggested incorporating more nuanced feature engineering or alternative model approaches to capture potential instabilities.

3. **Computational Biologist:** Proposed adding penalty terms for unpaired cysteines and suggested using molecular dynamics simulations to better capture the time-dependent stability of nanobody-RBD complexes.

4. **Scientific Critic:** Agreed on the potential overemphasis of structural metrics and recommended a comprehensive review of the entire scoring strategy to integrate broader stability assessments.

### Decisions and Follow-Up Questions:

**Decisions:**
- We will consider introducing a penalty in the WS formula for unpaired cysteines, possibly by enhancing the weight of the ESM LLR or adding a specific penalty term.
- We will explore the potential of molecular dynamics simulations to provide additional insights into nanobody stability.
- A broader review of the scoring strategy will be conducted to ensure it effectively integrates both structural and sequence-based assessments.

**Follow-Up Questions:**
1. **To the Machine Learning Specialist:** Could you elaborate on how we might implement feature engineering to better account for structural liabilities like unpaired cysteines? Are there known machine learning techniques that could assist in this regard?

2. **To the Computational Biologist:** How feasible is it to incorporate molecular dynamics simulations into our current workflow? What resources and timeframes would we need to consider for this approach?

3. **To the Scientific Critic:** In suggesting a comprehensive scoring strategy review, what additional metrics or validation techniques do you think could strengthen our model's predictions?

Let’s proceed with these follow-up questions to further refine our approach and address the reviewer's feedback effectively.

## User

Immunologist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Pass.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Certainly. Building on the discussion, here are my thoughts on how we can move forward with feature engineering and machine learning techniques:

1. **Feature Engineering for Structural Liabilities:** We can introduce specific features that explicitly capture the presence of unpaired cysteines and other potentially destabilizing residues. This could involve creating binary flags or counting features that indicate the presence of such residues. Additionally, integrating physicochemical properties related to stability, such as propensity for disulfide bond formation, might enhance predictive power.

2. **Machine Learning Techniques:** Utilizing ensemble methods such as Random Forests or Gradient Boosting can help incorporate these new features alongside existing metrics. These models can capture complex interactions between sequence features and structural predictions, potentially identifying patterns that single metric approaches might miss. Furthermore, techniques like SHAP (SHapley Additive exPlanations) could be employed to interpret feature importance, giving us insights into how unpaired cysteines and other features influence model predictions.

3. **Data Augmentation:** To improve robustness, we could simulate a larger dataset by introducing slight variations or noise, allowing our models to better generalize and capture subtle interactions that might influence stability or binding affinity.

Implementing these strategies would not only provide a more comprehensive view of each nanobody candidate but could also enhance our ability to predict true binding efficacy across variants.

## User

Computational Biologist, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

Certainly. Continuing from our previous discussion, here's how we can incorporate molecular dynamics simulations into our workflow:

1. **Feasibility of Molecular Dynamics Simulations:** Integrating molecular dynamics (MD) simulations is feasible, but it requires careful consideration of computational resources and time. For our current number of candidates, running short MD simulations (e.g., 100-200 ns) on high-performance computing clusters would be manageable. These simulations could provide insights into the dynamic stability of nanobody-RBD complexes, particularly focusing on the influence of unpaired cysteines over time.

2. **Resources and Timeframes:** Depending on the computational facilities available, MD simulations for each nanobody variant could take anywhere from a few days to a week. We would need access to a suitable cluster with GPU acceleration to expedite this process. Additionally, creating a pipeline to automate the setup and analysis of these simulations would help streamline their integration into our workflow.

3. **Potential Impact:** MD simulations could validate or challenge the static predictions from AlphaFold-Multimer and Rosetta by revealing potential conformational changes and stability issues that might not be apparent in static models. This dynamic perspective would be invaluable for verifying the robustness of nanobody designs.

By incorporating these simulations, we can enhance our understanding of nanobody behavior in a dynamic context, potentially leading to more reliable predictions and better candidate selection.

## User

Scientific Critic, please provide your thoughts on the discussion (round 2 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

Certainly. Here are my thoughts on how to enhance our scoring strategy based on the inputs so far:

1. **Comprehensive Scoring Strategy Review:** The current discussion rightly points towards integrating more diverse metrics to assess nanobody stability and efficacy. In addition to molecular dynamics simulations and enhanced feature engineering, we should consider incorporating empirical stability data from similar nanobodies, if available, to calibrate our models.

2. **Additional Metrics and Validation Techniques:** Beyond molecular dynamics and sequence-based penalties, we could explore using other biophysical simulations, such as coarse-grained models, to predict stability more efficiently. Thermodynamic metrics, such as predicted free energy landscapes or ensemble-based approaches, might offer additional insights into the likely stability and function of these nanobodies.

3. **Experimental Validation:** Ultimately, computational predictions should be validated experimentally. We could prioritize a subset of candidates for experimental evaluation to verify model predictions, thus iteratively refining our computational models with real-world data.

The integration of these strategies into our scoring framework would provide a more holistic view of nanobody performance. This approach would not only address the reviewer's concerns but also ensure that our predictions are as close to biological reality as possible. By doing so, we can enhance the robustness and applicability of our research findings.

## User

This concludes round 2 of 3 of discussion. Principal Investigator, please synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda.

## Principal Investigator

Thank you all for your detailed insights. Here's a synthesis of the discussion so far:

1. **Machine Learning Specialist:** Suggested feature engineering to better account for structural liabilities, such as unpaired cysteines, and proposed using ensemble machine learning methods to integrate these new features. The idea of using Shapley values to interpret feature importance is also promising.

2. **Computational Biologist:** Highlighted the feasibility and impact of incorporating molecular dynamics simulations. These simulations can offer valuable insights into the dynamic stability of nanobodies, especially concerning unpaired cysteines.

3. **Scientific Critic:** Emphasized a comprehensive scoring strategy review that includes additional metrics, such as thermodynamic modeling, and the importance of experimental validation to refine computational models. Suggested using empirical data from similar nanobodies to calibrate predictions.

### Decisions and Follow-Up Questions:

**Decisions:**
- We will proceed with introducing feature engineering techniques to flag structural liabilities in our models. We will explore ensemble learning methods to integrate these insights.
- Molecular dynamics simulations will be incorporated into our workflow, focusing on a subset of candidates for detailed analysis.
- We will aim to establish a pipeline for experimental validation to iteratively improve our predictive models, using insights gained to refine our scoring strategy.

**Follow-Up Questions:**
1. **To the Machine Learning Specialist:** How might we begin implementing ensemble methods and feature engineering in our current computational framework? Are there specific software tools or platforms you recommend?

2. **To the Computational Biologist:** Could you outline a potential project timeline for integrating molecular dynamics simulations into our workflow and any logistical considerations we should address upfront?

3. **To the Scientific Critic:** Regarding experimental validation, do you have recommendations on prioritizing candidates or specific experimental techniques that could efficiently validate our computational predictions?

Let's delve into these questions to solidify our approach and ensure our project remains on track.

## User

Immunologist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Immunologist

Pass.

## User

Machine Learning Specialist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Machine Learning Specialist

Certainly, here are my thoughts on implementing the proposed strategies:

1. **Implementation of Ensemble Methods and Feature Engineering:** To begin integrating ensemble methods and feature engineering, we can use platforms like scikit-learn and XGBoost, which are well-suited for handling structured data and allow for easy implementation of ensemble techniques. These tools can help us incorporate new features, such as flags for unpaired cysteines or other potentially destabilizing residues, and evaluate their impact on model predictions.

2. **Pipeline Development:** We should develop a pipeline that allows us to preprocess data, engineer features, train models, and evaluate performance efficiently. This might involve setting up a systematic way to introduce and test new features, possibly using automated machine learning (AutoML) frameworks to streamline the process.

3. **Model Interpretation:** Utilizing techniques like SHAP can aid in understanding the contribution of different features to our predictions, providing valuable insights into the model's decision-making process. This can help us iteratively refine our features and model structure to better align with biological insights.

By adopting these strategies, we can enhance our computational framework to produce more robust and interpretable predictions, ultimately improving our ability to design effective nanobodies.

## User

Computational Biologist, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Computational Biologist

Certainly. Here are my thoughts regarding the integration of molecular dynamics simulations:

1. **Project Timeline:** Incorporating molecular dynamics simulations would require a phased approach. Initially, we can start with setting up simulations for a small subset of nanobody candidates, focusing on those with unpaired cysteines or other flagged residues. This initial phase could take about 4-6 weeks, including setup, running simulations, and analyzing results.

2. **Logistical Considerations:** We'll need access to a high-performance computing cluster with GPU capabilities to efficiently run the simulations. Coordination with our institution's IT department or a cloud-based provider for computational resources would be essential. Additionally, setting up scripts to automate the simulation setup and analysis would help streamline the process.

3. **Data Integration:** The results from the molecular dynamics simulations can be integrated back into our modeling framework by using insights on dynamic stability to adjust our scoring metrics, potentially developing new features that capture stability over time.

By following this approach, we can effectively incorporate dynamic insights into our nanobody design process, leading to more reliable predictions and improved candidate selection.

## User

Scientific Critic, please provide your thoughts on the discussion (round 3 of 3). If you do not have anything new or relevant to add, you may say "pass". Remember that you can and should (politely) disagree with other team members if you have a different perspective.

## Scientific Critic

Certainly. Here are my thoughts on experimental validation and prioritization strategies:

1. **Prioritizing Candidates for Experimental Validation:** We should focus on a diverse subset of nanobody variants, including those ranked highly by computational models and those with contentious features like unpaired cysteines. This approach will help us validate the model's predictions across a spectrum of conditions and assumptions.

2. **Experimental Techniques:** Techniques such as surface plasmon resonance (SPR) or biolayer interferometry (BLI) could be used to assess binding affinities and kinetics of the selected nanobodies against the KP.3 variant of the SARS-CoV-2 spike protein. These methods provide quantitative data on binding interactions, which is crucial for validating computational predictions.

3. **Iterative Feedback Loop:** The experimental results should feed back into our computational models, allowing us to refine our predictions iteratively. Discrepancies between predicted and experimental outcomes can highlight areas for model improvement, such as adjusting feature importance or incorporating new parameters.

By systematically integrating experimental validation with our computational process, we can ensure that our models are both scientifically rigorous and reliable in their predictions, ultimately improving the success rate of our nanobody designs.

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

Our goal is to address the concerns raised by a reviewer regarding the high ranking of nanobody variants with unpaired cysteines, despite potential structural instabilities. Specifically, we need to understand why these variants are ranked highly and consider adjustments to the weighted scoring formula (WS) to better reflect potential liabilities.

### Team Member Input

- **Immunologist:** Highlighted the potential structural instability of unpaired cysteines and suggested increasing the weight of ESM LLR or introducing a specific penalty for unpaired cysteines to address these concerns.
  
- **Machine Learning Specialist:** Recommended feature engineering to include structural liabilities and using ensemble methods to integrate these features. Suggested tools like scikit-learn and XGBoost for implementation and SHAP for model interpretation.

- **Computational Biologist:** Proposed incorporating molecular dynamics simulations to assess the dynamic stability of nanobody variants, outlining a phased approach and necessary computational resources.

- **Scientific Critic:** Emphasized the need for comprehensive scoring strategy reviews, incorporating additional metrics like thermodynamic modeling, and prioritizing experimental validation to refine computational models.

### Recommendation

Given the inputs from the team, I recommend the following approach:

1. **Adjust the WS Formula:** Implement a penalty for unpaired cysteines in the WS formula. This could be achieved by adding a specific penalty term or increasing the weight of the ESM LLR metric to reflect sequence-based structural concerns more accurately.

2. **Integrate Molecular Dynamics Simulations:** Start with a subset of nanobody variants for detailed simulations to assess dynamic stability, focusing on those with unpaired cysteines.

3. **Enhance Predictive Models:** Use ensemble learning methods and feature engineering to account for potential structural liabilities, leveraging tools like scikit-learn and XGBoost.

4. **Prioritize Experimental Validation:** Select a diverse subset of candidates for experimental validation using techniques like SPR or BLI to assess binding interactions quantitatively.

This approach balances computational predictions with empirical validation, ensuring a comprehensive assessment of nanobody candidates.

### Answers

1. **Is there some other reason why the weighted score would rank these variants so highly, like the ipLDDT or dG values being very favorable?**

   - **Answer:** Yes, the current WS formula places significant weight on AF ipLDDT and RS dG values, which may overlook structural liabilities like unpaired cysteines due to their focus on structural and energetic metrics.
   - **Justification:** The ipLDDT and dG values are likely compensating for sequence-based concerns, highlighting the need for a more balanced approach.

2. **Are there any obvious adjustments to the WS formula that could mitigate this?**

   - **Answer:** Yes, introducing a specific penalty for unpaired cysteines or increasing the weight of the ESM LLR can mitigate this.
   - **Justification:** These adjustments would ensure that sequence-based insights are accurately reflected in the scoring, addressing potential structural instabilities.

### Next Steps

1. **Revise the WS Formula:** Implement changes to the WS formula to include penalties for unpaired cysteines.
   
2. **Begin Molecular Dynamics Simulations:** Identify a subset of candidates for simulations and arrange necessary computational resources.

3. **Develop Enhanced Models:** Use ensemble methods and feature engineering to improve predictive accuracy.

4. **Conduct Experimental Validation:** Select candidates for experimental assays to validate computational predictions and refine models based on results.

By following these steps, we can address the reviewer's concerns and enhance the reliability of our nanobody design process.

