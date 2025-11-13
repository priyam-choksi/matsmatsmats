## User

This is the beginning of an individual meeting with Computational Biologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "AlphaFold-Multimer" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on AlphaFold-Multimer to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on AlphaFold-Multimer using the query "AlphaFold-Multimer SARS-CoV-2 nanobody binders" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to AlphaFold-Multimer. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Computational Biologist, please provide your response to the agenda.

## Computational Biologist

Based on the abstracts provided, I have selected the following papers that are directly relevant to AlphaFold-Multimer and its application in designing SARS-CoV-2 nanobody binders:

```python
{
    "11405125": "AI-based IsAb2.0 for antibody design",
    "11788755": "Nanobody engineering: computational modelling and design for biomedical and therapeutic applications",
    "11783119": "Epitope mapping via in vitro deep mutational scanning methods and its applications",
    "11800653": "Unveiling the new chapter in nanobody engineering: advances in traditional construction and AI-driven optimization"
}
```

These papers focus on computational approaches, including AlphaFold-Multimer, for antibody and nanobody design, which are crucial for your research project on SARS-CoV-2 spike protein interactions.

## User

Tool Output:

Here are the top 19 articles on PubMed Central for the query "AlphaFold-Multimer SARS-CoV-2 nanobody binders":

[begin paper 1]

PMCID = 11788755

Title = Nanobody engineering: computational modelling and design for biomedical and therapeutic applications

Nanobodies, the smallest functional antibody fragment derived from camelid heavy‐chain‐only antibodies, have emerged as powerful tools for diverse biomedical applications. In this comprehensive review, we discuss the structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We explore their unique antigen‐binding domains, highlighting the critical role of complementarity‐determining regions in target recognition and specificity. This review further underscores the advantages of nanobodies over conventional antibodies from a biosynthesis perspective, including their small size, stability, and solubility, which make them ideal candidates for economical antigen capture in diagnostics, therapeutics, and biosensing. We discuss the recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. Finally, we examine a direct example of how computational design strategies were implemented for improving a nanobody‐based immunosensor, known as a Quenchbody. Through combining experimental findings and computational insights, this review elucidates the transformative impact of nanobodies in biotechnology and biomedical research, offering a roadmap for future advancements and applications in healthcare and diagnostics.

This review focuses on structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We discuss recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. It underscores their advantages over conventional antibodies from a biosynthesis perspective in diagnostics, therapeutics, and biosensing. 

[end paper 1]

[begin paper 2]

PMCID = 11405125

Title = AI-based IsAb2.0 for antibody design

Therapeutic antibody design has garnered widespread attention, highlighting its interdisciplinary importance. Advancements in technology emphasize the critical role of designing nanobodies and humanized antibodies in antibody engineering. However, current experimental methods are costly and time-consuming. Computational approaches, while progressing, faced limitations due to insufficient structural data and the absence of a standardized protocol. To tackle these challenges, our lab previously developed IsAb1.0, an in silico antibody design protocol. Yet, IsAb1.0 lacked accuracy, had a complex procedure, and required extensive antibody bioinformation. Moreover, it overlooked nanobody and humanized antibody design, hindering therapeutic antibody development. Building upon IsAb1.0, we enhanced our design protocol with artificial intelligence methods to create IsAb2.0. IsAb2.0 utilized AlphaFold-Multimer (2.3/3.0) for accurate modeling and complex construction without templates and employed the precise FlexddG method for in silico antibody optimization. Validated through optimization of a humanized nanobody J3 (HuJ3) targeting HIV-1 gp120, IsAb2.0 predicted five mutations that can improve HuJ3-gp120 binding affinity. These predictions were confirmed by commercial software and validated through binding and neutralization assays. IsAb2.0 streamlined antibody design, offering insights into future techniques to accelerate immunotherapy development.

[end paper 2]

[begin paper 3]

PMCID = 10886287

Title = Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions

This perspective sheds light on the transformative impact of recent computational advancements in the field of protein therapeutics, with a particular focus on the design and development of antibodies. Cutting-edge computational methods have revolutionized our understanding of protein–protein interactions (PPIs), enhancing the efficacy of protein therapeutics in preclinical and clinical settings. Central to these advancements is the application of machine learning and deep learning, which offers unprecedented insights into the intricate mechanisms of PPIs and facilitates precise control over protein functions. Despite these advancements, the complex structural nuances of antibodies pose ongoing challenges in their design and optimization. Our review provides a comprehensive exploration of the latest deep learning approaches, including language models and diffusion techniques, and their role in surmounting these challenges. We also present a critical analysis of these methods, offering insights to drive further progress in this rapidly evolving field. The paper includes practical recommendations for the application of these computational techniques, supplemented with independent benchmark studies. These studies focus on key performance metrics such as accuracy and the ease of program execution, providing a valuable resource for researchers engaged in antibody design and development. Through this detailed perspective, we aim to contribute to the advancement of antibody design, equipping researchers with the tools and knowledge to navigate the complexities of this field.

[end paper 3]

[begin paper 4]

PMCID = 10129313

Title = Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies

Antibodies have the capacity to bind a diverse set of antigens, and they have become critical therapeutics and diagnostic molecules. The binding of antibodies is facilitated by a set of six hypervariable loops that are diversified through genetic recombination and mutation. Even with recent advances, accurate structural prediction of these loops remains a challenge. Here, we present IgFold, a fast deep learning method for antibody structure prediction. IgFold consists of a pre-trained language model trained on 558 million natural antibody sequences followed by graph networks that directly predict backbone atom coordinates. IgFold predicts structures of similar or better quality than alternative methods (including AlphaFold) in significantly less time (under 25 s). Accurate structure prediction on this timescale makes possible avenues of investigation that were previously infeasible. As a demonstration of IgFold’s capabilities, we predicted structures for 1.4 million paired antibody sequences, providing structural insights to 500-fold more antibodies than have experimentally determined structures.

Prediction of antibody structures is critical for understanding and designing novel therapeutic and diagnostic molecules. Here, the authors present IgFold: a fast, accurate method for antibody structure prediction using an end-to-end deep learning model.

[end paper 4]

[begin paper 5]

PMCID = 11221890

Title = Antibody design using deep learning: from sequence and structure design to affinity maturation

Deep learning has achieved impressive results in various fields such as computer vision and natural language processing, making it a powerful tool in biology. Its applications now encompass cellular image classification, genomic studies and drug discovery. While drug development traditionally focused deep learning applications on small molecules, recent innovations have incorporated it in the discovery and development of biological molecules, particularly antibodies. Researchers have devised novel techniques to streamline antibody development, combining in vitro and in silico methods. In particular, computational power expedites lead candidate generation, scaling and potential antibody development against complex antigens. This survey highlights significant advancements in protein design and optimization, specifically focusing on antibodies. This includes various aspects such as design, folding, antibody–antigen interactions docking and affinity maturation.

[end paper 5]

[begin paper 6]

PMCID = 11783119

Title = Epitope mapping via in vitro deep mutational scanning methods and its applications

Epitope mapping is a technique employed to define the region of an antigen that elicits an immune response, providing crucial insight into the structural architecture of the antigen as well as epitope-paratope interactions. With this breadth of knowledge, immunotherapies, diagnostics, and vaccines are being developed with a rational and data-supported design. Traditional epitope mapping methods are laborious, time-intensive, and often lack the ability to screen proteins in a high-throughput manner or provide high resolution. Deep mutational scanning (DMS), however, is revolutionizing the field as it can screen all possible single amino acid mutations and provide an efficient and high-throughput way to infer the structures of both linear and three-dimensional epitopes with high resolution. Currently, more than 50 publications take this approach to efficiently identify enhancing or escaping mutations, with many then employing this information to rapidly develop broadly neutralizing antibodies, T-cell immunotherapies, vaccine platforms, or diagnostics. We provide a comprehensive review of the approaches to accomplish epitope mapping while also providing a summation of the development of DMS technology and its impactful applications.

[end paper 6]

[begin paper 7]

PMCID = 11788748

Title = Structure‐based computational design of antibody mimetics: challenges and perspectives

The design of antibody mimetics holds great promise for revolutionizing therapeutic interventions by offering alternatives to conventional antibody therapies. Structure‐based computational approaches have emerged as indispensable tools in the rational design of those molecules, enabling the precise manipulation of their structural and functional properties. This review covers the main classes of designed antigen‐binding motifs, as well as alternative strategies to develop tailored ones. We discuss the intricacies of different computational protein–protein interaction design strategies, showcased by selected successful cases in the literature. Subsequently, we explore the latest advancements in the computational techniques including the integration of machine and deep learning methodologies into the design framework, which has led to an augmented design pipeline. Finally, we verse onto the current challenges that stand in the way between high‐throughput computer design of antibody mimetics and experimental realization, offering a forward‐looking perspective into the field and the promises it holds to biotechnology.

Computer graphics representation of an artificial intelligence–driven protein design pipeline, where building blocks are fed into the birth of a novel protein structure. 

[end paper 7]

[begin paper 8]

PMCID = 11800653

Title = Unveiling the new chapter in nanobody engineering: advances in traditional construction and AI-driven optimization

Nanobodies (Nbs), miniature antibodies consisting solely of the variable region of heavy chains, exhibit unique properties such as small size, high stability, and strong specificity, making them highly promising for disease diagnosis and treatment. The engineering production of Nbs has evolved into a mature process, involving library construction, screening, and expression purification. Different library types, including immune, naïve, and synthetic/semi-synthetic libraries, offer diverse options for various applications, while display platforms like phage display, cell surface display, and non-surface display provide efficient screening of target Nbs. Recent advancements in artificial intelligence (AI) have opened new avenues in Nb engineering. AI’s exceptional performance in protein structure prediction and molecular interaction simulation has introduced novel perspectives and tools for Nb design and optimization. Integrating AI with traditional experimental methods is anticipated to enhance the efficiency and precision of Nb development, expediting the transition from basic research to clinical applications. This review comprehensively examines the latest progress in Nb engineering, emphasizing library construction strategies, display platform technologies, and AI applications. It evaluates the strengths and weaknesses of various libraries and display platforms and explores the potential and challenges of AI in predicting Nb structure, antigen-antibody interactions, and optimizing physicochemical properties.

[end paper 8]

[begin paper 9]

PMCID = 8950575

Title = Next-Generation Molecular Discovery: From Bottom-Up In Vivo and In Vitro Approaches to In Silico Top-Down Approaches for Therapeutics Neogenesis

Protein and drug engineering comprises a major part of the medical and research industries, and yet approaches to discovering and understanding therapeutic molecular interactions in biological systems rely on trial and error. The general approach to molecular discovery involves screening large libraries of compounds, proteins, or antibodies, or in vivo antibody generation, which could be considered “bottom-up” approaches to therapeutic discovery. In these bottom-up approaches, a minimal amount is known about the therapeutics at the start of the process, but through meticulous and exhaustive laboratory work, the molecule is characterised in detail. In contrast, the advent of “big data” and access to extensive online databases and machine learning technologies offers promising new avenues to understanding molecular interactions. Artificial intelligence (AI) now has the potential to predict protein structure at an unprecedented accuracy using only the genetic sequence. This predictive approach to characterising molecular structure—when accompanied by high-quality experimental data for model training—has the capacity to invert the process of molecular discovery and characterisation. The process has potential to be transformed into a top-down approach, where new molecules can be designed directly based on the structure of a target and the desired function, rather than performing screening of large libraries of molecular variants. This paper will provide a brief evaluation of bottom-up approaches to discovering and characterising biological molecules and will discuss recent advances towards developing top-down approaches and the prospects of this.

[end paper 9]

[begin paper 10]

PMCID = 10484509

Title = Exploring rigid-backbone protein docking in biologics discovery: a test using the DARPin scaffold

Accurate protein-protein docking remains challenging, especially for artificial biologics not coevolved naturally against their protein targets, like antibodies and other engineered scaffolds. We previously developed ProPOSE, an exhaustive docker with full atomistic details, which delivers cutting-edge performance by allowing side-chain rearrangements upon docking. However, extensive protein backbone flexibility limits its practical applicability as indicated by unbound docking tests. To explore the usefulness of ProPOSE on systems with limited backbone flexibility, here we tested the engineered scaffold DARPin, which is characterized by its relatively rigid protein backbone. A prospective screening campaign was undertaken, in which sequence-diversified DARPins were docked and ranked against a directed epitope on the target protein BCL-W. In this proof-of-concept study, only a relatively small set of 2,213 diverse DARPin interfaces were selected for docking from the huge theoretical library from mutating 18 amino-acid positions. A computational selection protocol was then applied for enrichment of binders based on normalized computed binding scores and frequency of binding modes against the predefined epitope. The top-ranked 18 designed DARPin interfaces were selected for experimental validation. Three designs exhibited binding affinities to BCL-W in the nanomolar range comparable to control interfaces adopted from known DARPin binders. This result is encouraging for future screening and engineering campaigns of DARPins and possibly other similarly rigid scaffolds against targeted protein epitopes. Method limitations are discussed and directions for future refinements are proposed.

[end paper 10]

[begin paper 11]

PMCID = 9631452

Title = Advances in antibody discovery from human BCR repertoires

Antibodies make up an important and growing class of compounds used for the diagnosis or treatment of disease. While traditional antibody discovery utilized immunization of animals to generate lead compounds, technological innovations have made it possible to search for antibodies targeting a given antigen within the repertoires of B cells in humans. Here we group these innovations into four broad categories: cell sorting allows the collection of cells enriched in specificity to one or more antigens; BCR sequencing can be performed on bulk mRNA, genomic DNA or on paired (heavy-light) mRNA; BCR repertoire analysis generally involves clustering BCRs into specificity groups or more in-depth modeling of antibody-antigen interactions, such as antibody-specific epitope predictions; validation of antibody-antigen interactions requires expression of antibodies, followed by antigen binding assays or epitope mapping. Together with innovations in Deep learning these technologies will contribute to the future discovery of diagnostic and therapeutic antibodies directly from humans.

[end paper 11]

[begin paper 12]

PMCID = 11679646

Title = The Application of Machine Learning on Antibody Discovery and Optimization

Antibodies play critical roles in modern medicine, serving as diagnostics and therapeutics for various diseases due to their ability to specifically bind to target antigens. Traditional antibody discovery and optimization methods are time-consuming and resource-intensive, though they have successfully generated antibodies for diagnosing and treating diseases. The advancements in protein data, computational hardware, and machine learning (ML) models have the opportunity to disrupt antibody discovery and optimization research. Machine learning models have demonstrated their abilities in antibody design. These machine learning models enable rapid in silico design of antibody candidates within a few days, achieving approximately a 60% reduction in time and a 50% reduction in cost compared to traditional methods. This review focuses on the latest machine learning-based antibody discovery and optimization developments. We briefly discuss the limitations of traditional methods and then explore the machine learning-based antibody discovery and optimization methodologies. We also focus on future research directions, including developing Antibody Design AI Agents and data foundries, alongside the ethical and regulatory considerations essential for successfully adopting machine learning-driven antibody designs.

[end paper 12]

[begin paper 13]

PMCID = 10502012

Title = Molecular basis of TASL recruitment by the peptide/histidine transporter 1, PHT1

PHT1 is a histidine /oligopeptide transporter with an essential role in Toll-like receptor innate immune responses. It can act as a receptor by recruiting the adaptor protein TASL which leads to type I interferon production via IRF5. Persistent stimulation of this signalling pathway is known to be involved in the pathogenesis of systemic lupus erythematosus (SLE). Understanding how PHT1 recruits TASL at the molecular level, is therefore clinically important for the development of therapeutics against SLE and other autoimmune diseases. Here we present the Cryo-EM structure of PHT1 stabilized in the outward-open conformation. By combining biochemical and structural modeling techniques we propose a model of the PHT1-TASL complex, in which the first 16 N-terminal TASL residues fold into a helical structure that bind in the central cavity of the inward-open conformation of PHT1. This work provides critical insights into the molecular basis of PHT1/TASL mediated type I interferon production.

The peptide/histidine transporter 1, PHT1 (SLC15A4), is required for TLR-IRF5 activation via the adaptor protein TASL. Here, the authors determined the structure of PHT1 in the outward-open conformation and present a model of the PHT1-TASL complex where the first 16 residues of TASL bind into the central cavity of PHT1.

[end paper 13]

[begin paper 14]

PMCID = 9989197

Title = Learned features of antibody-antigen binding affinity

Defining predictors of antigen-binding affinity of antibodies is valuable for engineering therapeutic antibodies with high binding affinity to their targets. However, this task is challenging owing to the huge diversity in the conformations of the complementarity determining regions of antibodies and the mode of engagement between antibody and antigen. In this study, we used the structural antibody database (SAbDab) to identify features that can discriminate high- and low-binding affinity across a 5-log scale. First, we abstracted features based on previously learned representations of protein-protein interactions to derive ‘complex’ feature sets, which include energetic, statistical, network-based, and machine-learned features. Second, we contrasted these complex feature sets with additional ‘simple’ feature sets based on counts of contacts between antibody and antigen. By investigating the predictive potential of 700 features contained in the eight complex and simple feature sets, we observed that simple feature sets perform comparably to complex feature sets in classification of binding affinity. Moreover, combining features from all eight feature-sets provided the best classification performance (median cross-validation AUROC and F1-score of 0.72). Of note, classification performance is substantially improved when several sources of data leakage (e.g., homologous antibodies) are not removed from the dataset, emphasizing a potential pitfall in this task. We additionally observe a classification performance plateau across diverse featurization approaches, highlighting the need for additional affinity-labeled antibody-antigen structural data. The findings from our present study set the stage for future studies aimed at multiple-log enhancement of antibody affinity through feature-guided engineering.

[end paper 14]

[begin paper 15]

PMCID = 11409006

Title = Mass spectrometry-complemented molecular modeling predicts the interaction interface for a camelid single-domain antibody targeting the Plasmodium falciparum circumsporozoite protein’s C-terminal domain

Bioanalytical methods that enable rapid and high-detail characterization of binding specificities and strengths of protein complexes with low sample consumption are highly desired. The interaction between a camelid single domain antibody (sdAbCSP1) and its target antigen (PfCSP-Cext) was selected as a model system to provide proof-of-principle for the here described methodology.

The structure of the sdAbCSP1 – PfCSP-Cext complex was modeled using AlphaFold2. The recombinantly expressed proteins, sdAbCSP1, PfCSP-Cext, and the sdAbCSP1 – PfCSP-Cext complex, were subjected to limited proteolysis and mass spectrometric peptide analysis. ITEM MS (Intact Transition Epitope Mapping Mass Spectrometry) and ITC (Isothermal Titration Calorimetry) were applied to determine stoichiometry and binding strength.

The paratope of sdAbCSP1 mainly consists of its CDR3 (aa100–118). PfCSP-Cext’s epitope is assembled from its α-helix (aa40–52) and opposing loop (aa83–90). PfCSP-Cext’s GluC cleavage sites E46 and E58 were shielded by complex formation, confirming the predicted epitope. Likewise, sdAbCSP1’s tryptic cleavage sites R105 and R108 were shielded by complex formation, confirming the predicted paratope. ITEM MS determined the 1:1 stoichiometry and the high complex binding strength, exemplified by the gas phase dissociation reaction enthalpy of 50.2 kJ/mol. The in-solution complex dissociation constant is 5 × 10-10 M.

Combining AlphaFold2 modeling with mass spectrometry/limited proteolysis generated a trustworthy model for the sdAbCSP1 – PfCSP-Cext complex interaction interface.

[end paper 15]

[begin paper 16]

PMCID = 11763998

Title = Proximity Labeling: Precise Proteomics Technology for Mapping Receptor Protein Neighborhoods at the Cancer Cell Surface

Cell surface proteins and receptors on cancer cells, important as disease markers and drug targets, are highly dynamic, with their biological functions controlled by constantly changing protein–protein interaction networks. Studying these transient macromolecular neighborhoods is technically challenging, and traditional methods often fall short. Proximity labeling, a cutting-edge proteomics technology, offers a more precise approach that overcomes the limitations of existing methods. This review highlights the strategies, applications, benefits, and limitations of different proximity labeling platforms used to map the receptor protein microenvironment on the surface of cancer cells.

Cell surface receptors are pivotal to cancer cell transformation, disease progression, metastasis, early detection, targeted therapy, drug responses, and clinical outcomes. Since they coordinate complex signaling communication networks in the tumor microenvironment, mapping the physical interaction partners of cell surface receptors in vivo is vital for understanding their roles, functional states, and suitability as therapeutic targets. Yet traditional methods like immunoprecipitation and affinity purification–mass spectrometry often fail to detect key but weak or transient receptor–protein interactions. Proximity labeling, a cutting-edge proteomics technology, addresses these technical challenges by enabling precise mapping of protein neighborhoods around a receptor target on the cell surface of cancer cells. This technique has been successfully applied in vitro and in vivo for proteomic mapping across various model systems. This review explores the fundamental principles, technologies, advantages, limitations, and applications of proximity labeling in cancer biology, focusing on mapping receptor microenvironments. By advancing mechanistic insights into cancer cell receptor signaling mechanisms, proximity labeling is poised to transform cancer research, improve targeted therapies, and illuminate avenues to overcome drug resistance.

[end paper 16]

[begin paper 17]

PMCID = 9912273

Title = Advancing Targeted Protein Degradation via Multiomics Profiling and Artificial Intelligence

Only around 20% of the human proteome is considered to be druggable with small-molecule antagonists. This leaves some of the most compelling therapeutic targets outside the reach of ligand discovery. The concept of targeted protein degradation (TPD) promises to overcome some of these limitations. In brief, TPD is dependent on small molecules that induce the proximity between a protein of interest (POI) and an E3 ubiquitin ligase, causing ubiquitination and degradation of the POI. In this perspective, we want to reflect on current challenges in the field, and discuss how advances in multiomics profiling, artificial intelligence, and machine learning (AI/ML) will be vital in overcoming them. The presented roadmap is discussed in the context of small-molecule degraders but is equally applicable for other emerging proximity-inducing modalities.

[end paper 17]

[begin paper 18]

PMCID = 10299731

Title = Challenges and frontiers of computational modelling of biomolecular recognition

Biomolecular recognition including binding of small molecules, peptides and proteins to their target receptors plays a key role in cellular function and has been targeted for therapeutic drug design. However, the high flexibility of biomolecules and slow binding and dissociation processes have presented challenges for computational modelling. Here, we review the challenges and computational approaches developed to characterise biomolecular binding, including molecular docking, molecular dynamics simulations (especially enhanced sampling) and machine learning. Further improvements are still needed in order to accurately and efficiently characterise binding structures, mechanisms, thermodynamics and kinetics of biomolecules in the future.

[end paper 18]

[begin paper 19]

PMCID = 9994104

Title = Protein–protein interfaces in molecular glue-induced ternary complexes: classification, characterization, and prediction

Molecular glues are a class of small molecules that stabilize the interactions between proteins. Naturally occurring molecular glues are present in many areas of biology where they serve as central regulators of signaling pathways. Importantly, several clinical compounds act as molecular glue degraders that stabilize interactions between E3 ubiquitin ligases and target proteins, leading to their degradation. Molecular glues hold promise as a new generation of therapeutic agents, including those molecular glue degraders that can redirect the protein degradation machinery in a precise way. However, rational discovery of molecular glues is difficult in part due to the lack of understanding of the protein–protein interactions they stabilize. In this review, we summarize the structures of known molecular glue-induced ternary complexes and the interface properties. Detailed analysis shows different mechanisms of ternary structure formation. Additionally, we also review computational approaches for predicting protein–protein interfaces and highlight the promises and challenges. This information will ultimately help inform future approaches for rational molecular glue discovery.

This review surveys molecular glue-induced ternary complexes in the PDB and provides an overview of computational methods that can be utilized to predict them.

[end paper 19]



