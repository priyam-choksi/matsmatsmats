## User

This is the beginning of an individual meeting with Computational Biologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "AlphaFold-Multimer" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on AlphaFold-Multimer to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on AlphaFold-Multimer using the query "AlphaFold-Multimer nanobody design" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to AlphaFold-Multimer. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Computational Biologist, please provide your response to the agenda.

## Computational Biologist

Based on the abstracts provided, here are the papers that are directly relevant to AlphaFold-Multimer and nanobody design, which I would like to fine-tune myself on:

```python
{
    "11405125": "AI-based IsAb2.0 for antibody design",
    "10638423": "Enhancing alphafold-multimer-based protein complex structure prediction with MULTICOM in CASP15",
    "11788755": "Nanobody engineering: computational modelling and design for biomedical and therapeutic applications",
    "9278006": "Benchmarking AlphaFold for protein complex modeling reveals accuracy determinants",
    "10227038": "ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins",
    "11799971": "Physical-aware model accuracy estimation for protein complex using deep learning method",
    "10349958": "Evaluation of AlphaFold Antibody-Antigen Modeling with Implications for Improving Predictive Accuracy",
    "10402144": "Reliable protein-protein docking with AlphaFold, Rosetta, and replica-exchange",
    "11461844": "Unmasking AlphaFold to integrate experiments and predictions in multimeric complexes",
    "10886287": "Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions",
    "10751731": "Evaluation of AlphaFold antibody–antigen modeling with implications for improving predictive accuracy",
    "11788754": "An outlook on structural biology after AlphaFold: tools, limits and perspectives",
    "11483107": "Towards the accurate modelling of antibody−antigen complexes from sequence using machine learning and information-driven docking",
    "11474105": "ABodyBuilder3: improved and scalable antibody structure predictions",
    "11578886": "MassiveFold: unveiling AlphaFold’s hidden potential with optimized and parallelized massive sampling"
}
```

These papers focus on the use of AlphaFold-Multimer in the context of nanobody and antibody design, which is directly relevant to my research project.

## User

Tool Output:

Here are the top 92 articles on PubMed Central for the query "AlphaFold-Multimer nanobody design":

[begin paper 1]

PMCID = 11392808

Title = Docking a flexible basket onto the core of the nuclear pore complex

The nuclear basket attaches to the nucleoplasmic side of the nuclear pore complex (NPC), coupling transcription to mRNA quality control and export. The basket expands the functional repertoire of a subset of NPCs in Saccharomyces cerevisiae by drawing a unique RNA/protein interactome. Yet, how the basket docks onto the NPC core remains unknown. By integrating AlphaFold-based interaction screens, electron microscopy and membrane-templated reconstitution, we uncovered a membrane-anchored tripartite junction between basket and NPC core. The basket subunit Nup60 harbours three adjacent short linear motifs, which connect Mlp1, a parallel homodimer consisting of coiled-coil segments interrupted by flexible hinges, and the Nup85 subunit of the Y-complex. We reconstituted the Y-complex•Nup60•Mlp1 assembly on a synthetic membrane and validated the protein interfaces in vivo. Here we explain how a short linear motif-based protein junction can substantially reshape NPC structure and function, advancing our understanding of compositional and conformational NPC heterogeneity.

Stankunas and Köhler define how the nucleoplasmic portion of the nuclear pore complex (NPC), the basket, docks onto the NPC core by integrating AlphaFold-based interaction screens, electron microscopy, and membrane-templated reconstitutions.

[end paper 1]

[begin paper 2]

PMCID = 11383924

Title = Modelling protein complexes with crosslinking mass spectrometry and deep learning

Scarcity of structural and evolutionary information on protein complexes poses a challenge to deep learning-based structure modelling. We integrate experimental distance restraints obtained by crosslinking mass spectrometry (MS) into AlphaFold-Multimer, by extending AlphaLink to protein complexes. Integrating crosslinking MS data substantially improves modelling performance on challenging targets, by helping to identify interfaces, focusing sampling, and improving model selection. This extends to single crosslinks from whole-cell crosslinking MS, opening the possibility of whole-cell structural investigations driven by experimental data. We demonstrate this by revealing the molecular basis of iron homoeostasis in Bacillus subtilis.

Elucidating the structure of protein complexes is key to understanding life at the molecular level. Here, the authors improve modelling performance on challenging targets by integrating experimental distance restraints from crosslinking mass spectrometry into AlphaFold-Multimer.

[end paper 2]

[begin paper 3]

PMCID = 11405125

Title = AI-based IsAb2.0 for antibody design

Therapeutic antibody design has garnered widespread attention, highlighting its interdisciplinary importance. Advancements in technology emphasize the critical role of designing nanobodies and humanized antibodies in antibody engineering. However, current experimental methods are costly and time-consuming. Computational approaches, while progressing, faced limitations due to insufficient structural data and the absence of a standardized protocol. To tackle these challenges, our lab previously developed IsAb1.0, an in silico antibody design protocol. Yet, IsAb1.0 lacked accuracy, had a complex procedure, and required extensive antibody bioinformation. Moreover, it overlooked nanobody and humanized antibody design, hindering therapeutic antibody development. Building upon IsAb1.0, we enhanced our design protocol with artificial intelligence methods to create IsAb2.0. IsAb2.0 utilized AlphaFold-Multimer (2.3/3.0) for accurate modeling and complex construction without templates and employed the precise FlexddG method for in silico antibody optimization. Validated through optimization of a humanized nanobody J3 (HuJ3) targeting HIV-1 gp120, IsAb2.0 predicted five mutations that can improve HuJ3-gp120 binding affinity. These predictions were confirmed by commercial software and validated through binding and neutralization assays. IsAb2.0 streamlined antibody design, offering insights into future techniques to accelerate immunotherapy development.

[end paper 3]

[begin paper 4]

PMCID = 10638423

Title = Enhancing alphafold-multimer-based protein complex structure prediction with MULTICOM in CASP15

To enhance the AlphaFold-Multimer-based protein complex structure prediction, we developed a quaternary structure prediction system (MULTICOM) to improve the input fed to AlphaFold-Multimer and evaluate and refine its outputs. MULTICOM samples diverse multiple sequence alignments (MSAs) and templates for AlphaFold-Multimer to generate structural predictions by using both traditional sequence alignments and Foldseek-based structure alignments, ranks structural predictions through multiple complementary metrics, and refines the structural predictions via a Foldseek structure alignment-based refinement method. The MULTICOM system with different implementations was blindly tested in the assembly structure prediction in the 15th Critical Assessment of Techniques for Protein Structure Prediction (CASP15) in 2022 as both server and human predictors. MULTICOM_qa ranked 3rd among 26 CASP15 server predictors and MULTICOM_human ranked 7th among 87 CASP15 server and human predictors. The average TM-score of the first predictions submitted by MULTICOM_qa for CASP15 assembly targets is ~0.76, 5.3% higher than ~0.72 of the standard AlphaFold-Multimer. The average TM-score of the best of top 5 predictions submitted by MULTICOM_qa is ~0.80, about 8% higher than ~0.74 of the standard AlphaFold-Multimer. Moreover, the Foldseek Structure Alignment-based Multimer structure Generation (FSAMG) method outperforms the widely used sequence alignment-based multimer structure generation.

A protein complex structure prediction system (MULTICOM3) improved AlphaFold-Multimer-based prediction by 5–10% in the 15th Critical Assessment of Techniques for Protein Structure Prediction (CASP15).

[end paper 4]

[begin paper 5]

PMCID = 11788755

Title = Nanobody engineering: computational modelling and design for biomedical and therapeutic applications

Nanobodies, the smallest functional antibody fragment derived from camelid heavy‐chain‐only antibodies, have emerged as powerful tools for diverse biomedical applications. In this comprehensive review, we discuss the structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We explore their unique antigen‐binding domains, highlighting the critical role of complementarity‐determining regions in target recognition and specificity. This review further underscores the advantages of nanobodies over conventional antibodies from a biosynthesis perspective, including their small size, stability, and solubility, which make them ideal candidates for economical antigen capture in diagnostics, therapeutics, and biosensing. We discuss the recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. Finally, we examine a direct example of how computational design strategies were implemented for improving a nanobody‐based immunosensor, known as a Quenchbody. Through combining experimental findings and computational insights, this review elucidates the transformative impact of nanobodies in biotechnology and biomedical research, offering a roadmap for future advancements and applications in healthcare and diagnostics.

This review focuses on structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We discuss recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. It underscores their advantages over conventional antibodies from a biosynthesis perspective in diagnostics, therapeutics, and biosensing. 

[end paper 5]

[begin paper 6]

PMCID = 9278006

Title = Benchmarking AlphaFold for protein complex modeling reveals accuracy determinants

High‐resolution experimental structural determination of protein–protein interactions has led to valuable mechanistic insights, yet due to the massive number of interactions and experimental limitations there is a need for computational methods that can accurately model their structures. Here we explore the use of the recently developed deep learning method, AlphaFold, to predict structures of protein complexes from sequence. With a benchmark of 152 diverse heterodimeric protein complexes, multiple implementations and parameters of AlphaFold were tested for accuracy. Remarkably, many cases (43%) had near‐native models (medium or high critical assessment of predicted interactions accuracy) generated as top‐ranked predictions by AlphaFold, greatly surpassing the performance of unbound protein–protein docking (9% success rate for near‐native top‐ranked models), however AlphaFold modeling of antibody–antigen complexes within our set was unsuccessful. We identified sequence and structural features associated with lack of AlphaFold success, and we also investigated the impact of multiple sequence alignment input. Benchmarking of a multimer‐optimized version of AlphaFold (AlphaFold‐Multimer) with a set of recently released antibody–antigen structures confirmed a low rate of success for antibody–antigen complexes (11% success), and we found that T cell receptor–antigen complexes are likewise not accurately modeled by that algorithm, showing that adaptive immune recognition poses a challenge for the current AlphaFold algorithm and model. Overall, our study demonstrates that end‐to‐end deep learning can accurately model many transient protein complexes, and highlights areas of improvement for future developments to reliably model any protein–protein interaction of interest.

[end paper 6]

[begin paper 7]

PMCID = 11230293

Title = A Comparison of Antibody-Antigen Complex Sequence-to-Structure Prediction Methods and their Systematic Biases



[end paper 7]

[begin paper 8]

PMCID = 10227038

Title = ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins

Immune receptor proteins play a key role in the immune system and have shown great promise as biotherapeutics. The structure of these proteins is critical for understanding their antigen binding properties. Here, we present ImmuneBuilder, a set of deep learning models trained to accurately predict the structure of antibodies (ABodyBuilder2), nanobodies (NanoBodyBuilder2) and T-Cell receptors (TCRBuilder2). We show that ImmuneBuilder generates structures with state of the art accuracy while being far faster than AlphaFold2. For example, on a benchmark of 34 recently solved antibodies, ABodyBuilder2 predicts CDR-H3 loops with an RMSD of 2.81Å, a 0.09Å improvement over AlphaFold-Multimer, while being over a hundred times faster. Similar results are also achieved for nanobodies, (NanoBodyBuilder2 predicts CDR-H3 loops with an average RMSD of 2.89Å, a 0.55Å improvement over AlphaFold2) and TCRs. By predicting an ensemble of structures, ImmuneBuilder also gives an error estimate for every residue in its final prediction. ImmuneBuilder is made freely available, both to download (https://github.com/oxpig/ImmuneBuilder) and to use via our webserver (http://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred). We also make available structural models for ~150 thousand non-redundant paired antibody sequences (10.5281/zenodo.7258553).

ImmuneBuilder is a set of deep learning models trained to predict the structure of antibodies, nanobodies, and T-Cell receptors with state-of-the-art accuracy while being much faster than AlphaFold2 and AlphaFoldMultimer.

[end paper 8]

[begin paper 9]

PMCID = 10129313

Title = Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies

Antibodies have the capacity to bind a diverse set of antigens, and they have become critical therapeutics and diagnostic molecules. The binding of antibodies is facilitated by a set of six hypervariable loops that are diversified through genetic recombination and mutation. Even with recent advances, accurate structural prediction of these loops remains a challenge. Here, we present IgFold, a fast deep learning method for antibody structure prediction. IgFold consists of a pre-trained language model trained on 558 million natural antibody sequences followed by graph networks that directly predict backbone atom coordinates. IgFold predicts structures of similar or better quality than alternative methods (including AlphaFold) in significantly less time (under 25 s). Accurate structure prediction on this timescale makes possible avenues of investigation that were previously infeasible. As a demonstration of IgFold’s capabilities, we predicted structures for 1.4 million paired antibody sequences, providing structural insights to 500-fold more antibodies than have experimentally determined structures.

Prediction of antibody structures is critical for understanding and designing novel therapeutic and diagnostic molecules. Here, the authors present IgFold: a fast, accurate method for antibody structure prediction using an end-to-end deep learning model.

[end paper 9]

[begin paper 10]

PMCID = 11799971

Title = Physical-aware model accuracy estimation for protein complex using deep learning method

With the breakthrough of AlphaFold2 on monomers, the research focus of structure prediction has shifted to protein complexes, driving the continued development of new methods for multimer structure prediction. Therefore, it is crucial to accurately estimate quality scores for the multimer model independent of the used prediction methods. In this work, we propose a physical-aware deep learning method, DeepUMQA-PA, to evaluate the residue-wise quality of protein complex models. Given the input protein complex model, the residue-based contact area and orientation features were first constructed using Voronoi tessellation, representing the potential physical interactions and hydrophobic properties. Then, the relationship between local residues and the overall complex topology as well as the inter-residue evolutionary information are characterized by geometry-based features, protein language model embedding representation, and knowledge-based statistical potential features. Finally, these features are fed into a fused network architecture employing equivalent graph neural network and ResNet network to estimate residue-wise model accuracy. Experimental results on the CASP15 test set demonstrate that our method outperforms the state-of-the-art method DeepUMQA3 by 3.69 % and 3.49 % on Pearson and Spearman, respectively. Notably, our method achieved 16.8 % and 15.5 % improvement in Pearson and Spearman, respectively, for the evaluation of nanobody-antigens. In addition, DeepUMQA-PA achieved better MAE scores than AlphaFold-Multimer and AlphaFold3 self-assessment methods on 43 % and 50 % of the targets, respectively. All these results suggest that physical-aware information based on the area and orientation of atom-atom and atom-solvent contacts has the potential to capture sequence-structure-quality relationships of proteins, especially in the case of flexible proteins. The DeepUMQA-PA server is freely available at http://zhanglab-bioinf.com/DeepUMQA-PA/.

[end paper 10]

[begin paper 11]

PMCID = 10470007

Title = Development of a V5-tag–directed nanobody and its implementation as an intracellular biosensor of GPCR signaling

Protein–protein interactions (PPIs) form the foundation of any cell signaling network. Considering that PPIs are highly dynamic processes, cellular assays are often essential for their study because they closely mimic the biological complexities of cellular environments. However, incongruity may be observed across different PPI assays when investigating a protein partner of interest; these discrepancies can be partially attributed to the fusion of different large functional moieties, such as fluorescent proteins or enzymes, which can yield disparate perturbations to the protein’s stability, subcellular localization, and interaction partners depending on the given cellular assay. Owing to their smaller size, epitope tags may exhibit a diminished susceptibility to instigate such perturbations. However, while they have been widely used for detecting or manipulating proteins in vitro, epitope tags lack the in vivo traceability and functionality needed for intracellular biosensors. Herein, we develop NbV5, an intracellular nanobody binding the V5-tag, which is suitable for use in cellular assays commonly used to study PPIs such as BRET, NanoBiT, and Tango. The NbV5:V5 tag system has been applied to interrogate G protein–coupled receptor signaling, specifically by replacing larger functional moieties attached to the protein interactors, such as fluorescent or luminescent proteins (∼30 kDa), by the significantly smaller V5-tag peptide (1.4 kDa), and for microscopy imaging which is successfully detected by NbV5-based biosensors. Therefore, the NbV5:V5 tag system presents itself as a versatile tool for live-cell imaging and a befitting adaptation to existing cellular assays dedicated to probing PPIs.

[end paper 11]

[begin paper 12]

PMCID = 11226077

Title = Protein loop structure prediction by community-based deep learning and its application to antibody CDR H3 loop modeling

As of now, more than 60 years have passed since the first determination of protein structures through crystallography, and a significant portion of protein structures can be predicted by computers. This is due to the groundbreaking enhancement in protein structure prediction achieved through neural network training utilizing extensive sequence and structure data. However, substantial challenges persist in structure prediction due to limited data availability, with antibody structure prediction standing as one such challenge. In this paper, we propose a novel neural network architecture that effectively enables structure prediction by reflecting the inherent combinatorial nature involved in protein structure formation. The core idea of this neural network architecture is not solely to track and generate a single structure but rather to form a community of multiple structures and pursue accurate structure prediction by exchanging information among community members. Applying this concept to antibody CDR H3 loop structure prediction resulted in improved structure sampling. Such an approach could be applied in the structural and functional studies of proteins, particularly in exploring various physiological processes mediated by loops. Moreover, it holds potential in addressing various other types of combinatorial structure prediction and design problems.

In this paper, we propose a new architecture that aims to improve upon protein structure prediction algorithms like AlphaFold or RoseTTAFold by considering the combinatorial nature of protein structure formation. Such an architecture, reflecting the physical principles of nature, is expected to yield beneficial results, particularly in scenarios with limited structure and sequence information. Named ComMat, this architecture does not focus on a single structure but rather on a set of multiple structures—a community—simultaneously. In this process, combinatorial exploration of protein structure is encouraged through information exchange among community members. ComMat is an instance that integrates this idea within the structure module of AlphaFold. Applying ComMat to antibody CDR H3 loop structure prediction yielded outstanding results in structure sampling and prediction when tested on the IgFold set and compared with IgFold and AlphaFold-Multimer. It confirmed that improved structure sampling stems from effective structural exploration. The proposed concept here could potentially be used in the development of various other combinatorial protein structure prediction and protein design methods.

[end paper 12]

[begin paper 13]

PMCID = 9020716

Title = A modified TurboID approach identifies tissue-specific centriolar components in C. elegans

Proximity-dependent labeling approaches such as BioID have been a great boon to studies of protein-protein interactions in the context of cytoskeletal structures such as centrosomes which are poorly amenable to traditional biochemical approaches like immunoprecipitation and tandem affinity purification. Yet, these methods have so far not been applied extensively to invertebrate experimental models such as C. elegans given the long labeling times required for the original promiscuous biotin ligase variant BirA*. Here, we show that the recently developed variant TurboID successfully probes the interactomes of both stably associated (SPD-5) and dynamically localized (PLK-1) centrosomal components. We further develop an indirect proximity labeling method employing a GFP nanobody-TurboID fusion, which allows the identification of protein interactors in a tissue-specific manner in the context of the whole animal. Critically, this approach utilizes available endogenous GFP fusions, avoiding the need to generate multiple additional strains for each target protein and the potential complications associated with overexpressing the protein from transgenes. Using this method, we identify homologs of two highly conserved centriolar components, Cep97 and BLD10/Cep135, which are present in various somatic tissues of the worm. Surprisingly, neither protein is expressed in early embryos, likely explaining why these proteins have escaped attention until now. Our work expands the experimental repertoire for C. elegans and opens the door for further studies of tissue-specific variation in centrosome architecture.

‘You can tell a lot about a person by the company they keep.’ This is as true for proteins as it is for people. Unfortunately, the methods traditionally used to probe protein-protein interactions, which rely on isolating stable multimolecular complexes, fail where such complexes do not exist or cannot be isolated from the cell. Proximity-dependent labeling methods such as TurboID, where interacting proteins are marked in vivo before isolation and which therefore do not require complexes to be maintained during extract preparation, have consequently become invaluable tools in cell biology. Normally, this requires the protein of interest to be tagged with an enzymatic marker such as the biotin ligase BirA*. Here, we developed a variant of this method, whereby the enzyme is targeted to the protein of interest using a genetically-encoded GFP nanobody, making existing fluorescent strains immediately available for interaction biochemistry. We show that this method successfully probes protein-protein interactions in a tissue-specific manner, even where those interactions occur in only a few cells in the context of the entire animal. While this work was conducted in the nematode C. elegans, the same method should be applicable to other genetic experimental models, as well as potentially vertebrate cultured cells.

[end paper 13]

[begin paper 14]

PMCID = 11735616

Title = Herpesviruses mimic zygotic genome activation to promote viral replication

Zygotic genome activation (ZGA) is crucial for maternal to zygotic transition at the 2-8-cell stage in order to overcome silencing of genes and enable transcription from the zygotic genome. In humans, ZGA is induced by DUX4, a pioneer factor that drives expression of downstream germline-specific genes and retroelements. Here we show that herpesviruses from all subfamilies, papillomaviruses and Merkel cell polyomavirus actively induce DUX4 expression to promote viral transcription and replication. Analysis of single-cell sequencing data sets from patients shows that viral DUX4 activation is of relevance in vivo. Herpes-simplex virus 1 (HSV-1) immediate early proteins directly induce expression of DUX4 and its target genes, which mimics zygotic genome activation. Upon HSV-1 infection, DUX4 directly binds to the viral genome and promotes viral transcription. DUX4 is functionally required for infection, since genetic depletion by CRISPR/Cas9 as well as degradation of DUX4 by nanobody constructs abrogates HSV-1 replication. Our results show that DNA viruses including herpesviruses mimic an embryonic-like transcriptional program that prevents epigenetic silencing of the viral genome and facilitates herpesviral gene expression.

Neugebauer et al. show activation of the usually silenced embryonic factor DUX4 by herpesviruses in vitro and in patients, and demonstrate that depletion of DUX4 by nanobody degraders abrogates viral replication.

[end paper 14]

[begin paper 15]

PMCID = 10220908

Title = Structural Modeling of Nanobodies: A Benchmark of State-of-the-Art Artificial Intelligence Programs

The number of applications for nanobodies is steadily expanding, positioning these molecules as fast-growing biologic products in the biotechnology market. Several of their applications require protein engineering, which in turn would greatly benefit from having a reliable structural model of the nanobody of interest. However, as with antibodies, the structural modeling of nanobodies is still a challenge. With the rise of artificial intelligence (AI), several methods have been developed in recent years that attempt to solve the problem of protein modeling. In this study, we have compared the performance in nanobody modeling of several state-of-the-art AI-based programs, either designed for general protein modeling, such as AlphaFold2, OmegaFold, ESMFold, and Yang-Server, or specifically designed for antibody modeling, such as IgFold, and Nanonet. While all these programs performed rather well in constructing the nanobody framework and CDRs 1 and 2, modeling CDR3 still represents a big challenge. Interestingly, tailoring an AI method for antibody modeling does not necessarily translate into better results for nanobodies.

[end paper 15]

[begin paper 16]

PMCID = 10392983

Title = Allosteric activation or inhibition of PI3Kγ mediated through conformational changes in the p110γ helical domain

PI3Kγ is a critical immune signaling enzyme activated downstream of diverse cell surface molecules, including Ras, PKCβ activated by the IgE receptor, and Gβγ subunits released from activated GPCRs. PI3Kγ can form two distinct complexes, with the p110γ catalytic subunit binding to either a p101 or p84 regulatory subunit, with these complexes being differentially activated by upstream stimuli. Here, using a combination of cryo electron microscopy, HDX-MS, and biochemical assays, we have identified novel roles of the helical domain of p110γ in regulating lipid kinase activity of distinct PI3Kγ complexes. We defined the molecular basis for how an allosteric inhibitory nanobody potently inhibits kinase activity through rigidifying the helical domain and regulatory motif of the kinase domain. The nanobody did not block either p110γ membrane recruitment or Ras/Gβγ binding, but instead decreased ATP turnover. We also identified that p110γ can be activated by dual PKCβ helical domain phosphorylation leading to partial unfolding of an N-terminal region of the helical domain. PKCβ phosphorylation is selective for p110γ-p84 compared to p110γ-p101, driven by differential dynamics of the helical domain of these different complexes. Nanobody binding prevented PKCβ-mediated phosphorylation. Overall, this work shows an unexpected allosteric regulatory role of the helical domain of p110γ that is distinct between p110γ-p84 and p110γ-p101 and reveals how this can be modulated by either phosphorylation or allosteric inhibitory binding partners. This opens possibilities of future allosteric inhibitor development for therapeutic intervention.

[end paper 16]

[begin paper 17]

PMCID = 10349958

Title = Evaluation of AlphaFold Antibody-Antigen Modeling with Implications for Improving Predictive Accuracy

High resolution antibody-antigen structures provide critical insights into immune recognition and can inform therapeutic design. The challenges of experimental structural determination and the diversity of the immune repertoire underscore the necessity of accurate computational tools for modeling antibody-antigen complexes. Initial benchmarking showed that despite overall success in modeling protein-protein complexes, AlphaFold and AlphaFold-Multimer have limited success in modeling antibody-antigen interactions. In this study, we performed a thorough analysis of AlphaFold’s antibody-antigen modeling performance on 429 nonredundant antibody-antigen complex structures, identifying useful confidence metrics for predicting model quality, and features of complexes associated with improved modeling success. We show the importance of bound-like component modeling in complex assembly accuracy, and that the current version of AlphaFold improves near-native modeling success to over 30%, versus approximately 20% for a previous version. With this improved success, AlphaFold can generate accurate antibody-antigen models in many cases, while additional training may further improve its performance.

[end paper 17]

[begin paper 18]

PMCID = 10120615

Title = Allosteric activation or inhibition of PI3Kγ mediated through conformational changes in the p110γ helical domain

PI3Kγ is a critical immune signaling enzyme activated downstream of diverse cell surface molecules, including Ras, PKCβ activated by the IgE receptor, and Gβγ subunits released from activated GPCRs. PI3Kγ can form two distinct complexes, with the p110γ catalytic subunit binding to either a p101 or p84 regulatory subunit, with these complexes being differentially activated by upstream stimuli. Here using a combination of cryo electron microscopy, HDX-MS, and biochemical assays we have identified novel roles of the helical domain of p110γ in regulating lipid kinase activity of distinct PI3Kγ complexes. We defined the molecular basis for how an allosteric inhibitory nanobody potently inhibits kinase activity through rigidifying the helical domain and regulatory motif of the kinase domain. The nanobody did not block either p110γ membrane recruitment or Ras/Gβγ binding, but instead decreased ATP turnover. We also identified that p110γ can be activated by dual PKCβ helical domain phosphorylation leading to partial unfolding of an N-terminal region of the helical domain. PKCβ phosphorylation is selective for p110γ-p84 compared to p110γ-p101, driven by differential dynamics of the helical domain of these different complexes. Nanobody binding prevented PKCβ mediated phosphorylation. Overall, this works shows an unexpected allosteric regulatory role of the helical domain of p110γ that is distinct between p110γ-p84 and p110γ-p101 and reveals how this can be modulated by either phosphorylation or allosteric inhibitory binding partners. This opens possibilities of future allosteric inhibitor development for therapeutic intervention.

[end paper 18]

[begin paper 19]

PMCID = 11800653

Title = Unveiling the new chapter in nanobody engineering: advances in traditional construction and AI-driven optimization

Nanobodies (Nbs), miniature antibodies consisting solely of the variable region of heavy chains, exhibit unique properties such as small size, high stability, and strong specificity, making them highly promising for disease diagnosis and treatment. The engineering production of Nbs has evolved into a mature process, involving library construction, screening, and expression purification. Different library types, including immune, naïve, and synthetic/semi-synthetic libraries, offer diverse options for various applications, while display platforms like phage display, cell surface display, and non-surface display provide efficient screening of target Nbs. Recent advancements in artificial intelligence (AI) have opened new avenues in Nb engineering. AI’s exceptional performance in protein structure prediction and molecular interaction simulation has introduced novel perspectives and tools for Nb design and optimization. Integrating AI with traditional experimental methods is anticipated to enhance the efficiency and precision of Nb development, expediting the transition from basic research to clinical applications. This review comprehensively examines the latest progress in Nb engineering, emphasizing library construction strategies, display platform technologies, and AI applications. It evaluates the strengths and weaknesses of various libraries and display platforms and explores the potential and challenges of AI in predicting Nb structure, antigen-antibody interactions, and optimizing physicochemical properties.

[end paper 19]

[begin paper 20]

PMCID = 11461844

Title = Unmasking AlphaFold to integrate experiments and predictions in multimeric complexes

Since the release of AlphaFold, researchers have actively refined its predictions and attempted to integrate it into existing pipelines for determining protein structures. These efforts have introduced a number of functionalities and optimisations at the latest Critical Assessment of protein Structure Prediction edition (CASP15), resulting in a marked improvement in the prediction of multimeric protein structures. However, AlphaFold’s capability of predicting large protein complexes is still limited and integrating experimental data in the prediction pipeline is not straightforward. In this study, we introduce AF_unmasked to overcome these limitations. Our results demonstrate that AF_unmasked can integrate experimental information to build larger or hard to predict protein assemblies with high confidence. The resulting predictions can help interpret and augment experimental data. This approach generates high quality (DockQ score > 0.8) structures even when little to no evolutionary information is available and imperfect experimental structures are used as a starting point. AF_unmasked is developed and optimised to fill incomplete experimental structures (structural inpainting), which may provide insights into protein dynamics. In summary, AF_unmasked provides an easy-to-use method that efficiently integrates experiments to predict large protein complexes more confidently.

Integrating AlphaFold (AF) predictions with experimental data is not straightforward. Here, authors introduce AF_unmasked, a tool to integrate AF with experimental information to predict large or challenging protein assemblies with high confidence.

[end paper 20]

[begin paper 21]

PMCID = 10356834

Title = GPCRome-wide analysis of G-protein-coupling diversity using a computational biology approach

GPCRs are master regulators of cell signaling by transducing extracellular stimuli into the cell via selective coupling to intracellular G-proteins. Here we present a computational analysis of the structural determinants of G-protein-coupling repertoire of experimental and predicted 3D GPCR-G-protein complexes. Interface contact analysis recapitulates structural hallmarks associated with G-protein-coupling specificity, including TM5, TM6 and ICLs. We employ interface contacts as fingerprints to cluster Gs vs Gi complexes in an unsupervised fashion, suggesting that interface residues contribute to selective coupling. We experimentally confirm on a promiscuous receptor (CCKAR) that mutations of some of these specificity-determining positions bias the coupling selectivity. Interestingly, Gs-GPCR complexes have more conserved interfaces, while Gi/o proteins adopt a wider number of alternative docking poses, as assessed via structural alignments of representative 3D complexes. Binding energy calculations demonstrate that distinct structural properties of the complexes are associated to higher stability of Gs than Gi/o complexes. AlphaFold2 predictions of experimental binary complexes confirm several of these structural features and allow us to augment the structural coverage of poorly characterized complexes such as G12/13.

Selective GPCR-G protein complexes formation is critical for signal transduction regulation. Here, the authors use a data-driven approach to show that the structures of experimental and predicted complex interfaces inform, at least partially, on G protein binding preferences.

[end paper 21]

[begin paper 22]

PMCID = 11221890

Title = Antibody design using deep learning: from sequence and structure design to affinity maturation

Deep learning has achieved impressive results in various fields such as computer vision and natural language processing, making it a powerful tool in biology. Its applications now encompass cellular image classification, genomic studies and drug discovery. While drug development traditionally focused deep learning applications on small molecules, recent innovations have incorporated it in the discovery and development of biological molecules, particularly antibodies. Researchers have devised novel techniques to streamline antibody development, combining in vitro and in silico methods. In particular, computational power expedites lead candidate generation, scaling and potential antibody development against complex antigens. This survey highlights significant advancements in protein design and optimization, specifically focusing on antibodies. This includes various aspects such as design, folding, antibody–antigen interactions docking and affinity maturation.

[end paper 22]

[begin paper 23]

PMCID = 11370418

Title = High-throughput affinity measurements of direct interactions between activation domains and co-activators

Sequence-specific activation by transcription factors is essential for gene regulation. Key to this are activation domains, which often fall within disordered regions of transcription factors and recruit co-activators to initiate transcription. These interactions are difficult to characterize via most experimental techniques because they are typically weak and transient. Consequently, we know very little about whether these interactions are promiscuous or specific, the mechanisms of binding, and how these interactions tune the strength of gene activation. To address these questions, we developed a microfluidic platform for expression and purification of hundreds of activation domains in parallel followed by direct measurement of co-activator binding affinities (STAMMPPING, for Simultaneous Trapping of Affinity Measurements via a Microfluidic Protein-Protein INteraction Generator). By applying STAMMPPING to quantify direct interactions between eight co-activators and 204 human activation domains (>1,500 Kds), we provide the first quantitative map of these interactions and reveal 334 novel binding pairs. We find that the metazoan-specific co-activator P300 directly binds >100 activation domains, potentially explaining its widespread recruitment across the genome to influence transcriptional activation. Despite sharing similar molecular properties (e.g. enrichment of negative and hydrophobic residues), activation domains utilize distinct biophysical properties to recruit certain co-activator domains. Co-activator domain affinity and occupancy are well-predicted by analytical models that account for multivalency, and in vitro affinities quantitatively predict activation in cells with an ultrasensitive response. Not only do our results demonstrate the ability to measure affinities between even weak protein-protein interactions in high throughput, but they also provide a necessary resource of over 1,500 activation domain/co-activator affinities which lays the foundation for understanding the molecular basis of transcriptional activation.

[end paper 23]

[begin paper 24]

PMCID = 11788748

Title = Structure‐based computational design of antibody mimetics: challenges and perspectives

The design of antibody mimetics holds great promise for revolutionizing therapeutic interventions by offering alternatives to conventional antibody therapies. Structure‐based computational approaches have emerged as indispensable tools in the rational design of those molecules, enabling the precise manipulation of their structural and functional properties. This review covers the main classes of designed antigen‐binding motifs, as well as alternative strategies to develop tailored ones. We discuss the intricacies of different computational protein–protein interaction design strategies, showcased by selected successful cases in the literature. Subsequently, we explore the latest advancements in the computational techniques including the integration of machine and deep learning methodologies into the design framework, which has led to an augmented design pipeline. Finally, we verse onto the current challenges that stand in the way between high‐throughput computer design of antibody mimetics and experimental realization, offering a forward‐looking perspective into the field and the promises it holds to biotechnology.

Computer graphics representation of an artificial intelligence–driven protein design pipeline, where building blocks are fed into the birth of a novel protein structure. 

[end paper 24]

[begin paper 25]

PMCID = 10886287

Title = Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions

This perspective sheds light on the transformative impact of recent computational advancements in the field of protein therapeutics, with a particular focus on the design and development of antibodies. Cutting-edge computational methods have revolutionized our understanding of protein–protein interactions (PPIs), enhancing the efficacy of protein therapeutics in preclinical and clinical settings. Central to these advancements is the application of machine learning and deep learning, which offers unprecedented insights into the intricate mechanisms of PPIs and facilitates precise control over protein functions. Despite these advancements, the complex structural nuances of antibodies pose ongoing challenges in their design and optimization. Our review provides a comprehensive exploration of the latest deep learning approaches, including language models and diffusion techniques, and their role in surmounting these challenges. We also present a critical analysis of these methods, offering insights to drive further progress in this rapidly evolving field. The paper includes practical recommendations for the application of these computational techniques, supplemented with independent benchmark studies. These studies focus on key performance metrics such as accuracy and the ease of program execution, providing a valuable resource for researchers engaged in antibody design and development. Through this detailed perspective, we aim to contribute to the advancement of antibody design, equipping researchers with the tools and knowledge to navigate the complexities of this field.

[end paper 25]

[begin paper 26]

PMCID = 10751731

Title = Evaluation of AlphaFold antibody–antigen modeling with implications for improving predictive accuracy

High resolution antibody–antigen structures provide critical insights into immune recognition and can inform therapeutic design. The challenges of experimental structural determination and the diversity of the immune repertoire underscore the necessity of accurate computational tools for modeling antibody–antigen complexes. Initial benchmarking showed that despite overall success in modeling protein–protein complexes, AlphaFold and AlphaFold‐Multimer have limited success in modeling antibody–antigen interactions. In this study, we performed a thorough analysis of AlphaFold's antibody–antigen modeling performance on 427 nonredundant antibody–antigen complex structures, identifying useful confidence metrics for predicting model quality, and features of complexes associated with improved modeling success. Notably, we found that the latest version of AlphaFold improves near‐native modeling success to over 30%, versus approximately 20% for a previous version, while increased AlphaFold sampling gives approximately 50% success. With this improved success, AlphaFold can generate accurate antibody–antigen models in many cases, while additional training or other optimization may further improve performance.

[end paper 26]

[begin paper 27]

PMCID = 11695057

Title = The co-receptor Tetraspanin12 directly captures Norrin to promote ligand-specific β-catenin signaling

Wnt/β-catenin signaling directs animal development and tissue renewal in a tightly controlled, cell- and tissue-specific manner. In the mammalian central nervous system, the atypical ligand Norrin controls angiogenesis and maintenance of the blood-brain barrier and blood-retina barrier through the Wnt/β-catenin pathway. Like Wnt, Norrin activates signaling by binding and heterodimerizing the receptors Frizzled (Fzd) and low-density lipoprotein receptor-related protein 5 or 6 (LRP5/6), leading to membrane recruitment of the intracellular transducer Dishevelled (Dvl) and ultimately stabilizing the transcriptional coactivator β-catenin. Unlike Wnt, the cystine knot ligand Norrin only signals through Fzd4 and additionally requires the co-receptor Tetraspanin12 (Tspan12); however, the mechanism underlying Tspan12-mediated signal enhancement is unclear. It has been proposed that Tspan12 integrates into the Norrin-Fzd4 complex to enhance Norrin-Fzd4 affinity or otherwise allosterically modulate Fzd4 signaling. Here, we measure direct, high-affinity binding between purified Norrin and Tspan12 in a lipid environment and use AlphaFold models to interrogate this interaction interface. We find that Tspan12 and Fzd4 can simultaneously bind Norrin and that a pre-formed Tspan12/Fzd4 heterodimer, as well as cells co-expressing Tspan12 and Fzd4, more efficiently capture low concentrations of Norrin than Fzd4 alone. We also show that Tspan12 competes with both heparan sulfate proteoglycans and LRP6 for Norrin binding and that Tspan12 does not impact Fzd4-Dvl affinity in the presence or absence of Norrin. Our findings suggest that Tspan12 does not allosterically enhance Fzd4 binding to Norrin or Dvl, but instead functions to directly capture Norrin upstream of signaling.

[end paper 27]

[begin paper 28]

PMCID = 11801797

Title = Molecular determinants of Neu5Ac binding to a tripartite ATP independent periplasmic (TRAP) transporter

N -Acetylneuraminic acid (Neu5Ac) is a negatively charged nine-carbon amino sugar that is often the peripheral sugar in human cell-surface glycoconjugates. Some bacteria scavenge, import, and metabolize Neu5Ac or redeploy it on their cell surfaces for immune evasion. The import of Neu5Ac by many bacteria is mediated by tripartite ATP-independent periplasmic (TRAP) transporters. We have previously reported the structures of SiaQM, a membrane-embedded component of the Haemophilus influenzae TRAP transport system, (Currie et al., 2024). However, none of the published structures contain Neu5Ac bound to SiaQM. This information is critical for defining the transport mechanism and for further structure-activity relationship studies. Here, we report the structures of Fusobacterium nucleatum SiaQM with and without Neu5Ac. Both structures are in an inward (cytoplasmic side) facing conformation. The Neu5Ac-bound structure reveals the interactions of Neu5Ac with the transporter and its relationship with the Na+ binding sites. Two of the Na+-binding sites are similar to those described previously. We identify a third metal-binding site that is further away and buried in the elevator domain. Ser300 and Ser345 interact with the C1-carboxylate group of Neu5Ac. Proteoliposome-based transport assays showed that Ser300-Neu5Ac interaction is critical for transport, whereas Ser345 is dispensable. Neu5Ac primarily interacts with residues in the elevator domain of the protein, thereby supporting the elevator with an operator mechanism. The residues interacting with Neu5Ac are conserved, providing fundamental information required to design inhibitors against this class of proteins.

[end paper 28]

[begin paper 29]

PMCID = 10402144

Title = Reliable protein-protein docking with AlphaFold, Rosetta, and replica-exchange

Despite the recent breakthrough of AlphaFold (AF) in the field of protein sequence-to-structure prediction, modeling protein interfaces and predicting protein complex structures remains challenging, especially when there is a significant conformational change in one or both binding partners. Prior studies have demonstrated that AF-multimer (AFm) can predict accurate protein complexes in only up to 43% of cases. In this work, we combine AlphaFold as a structural template generator with a physics-based replica exchange docking algorithm. Using a curated collection of 254 available protein targets with both unbound and bound structures, we first demonstrate that AlphaFold confidence measures (pLDDT) can be repurposed for estimating protein flexibility and docking accuracy for multimers. We incorporate these metrics within our ReplicaDock 2.0 protocol to complete a robust in-silico pipeline for accurate protein complex structure prediction. AlphaRED (AlphaFold-initiated Replica Exchange Docking) successfully docks failed AF predictions including 97 failure cases in Docking Benchmark Set 5.5. AlphaRED generates CAPRI acceptable-quality or better predictions for 66% of benchmark targets. Further, on a subset of antigen-antibody targets, which is challenging for AFm (19% success rate), AlphaRED demonstrates a success rate of 51%. This new strategy demonstrates the success possible by integrating deep-learning based architectures trained on evolutionary information with physics-based enhanced sampling. The pipeline is available at github.com/Graylab/AlphaRED.

[end paper 29]

[begin paper 30]

PMCID = 10700138

Title = The social and structural architecture of the yeast protein interactome

Cellular functions are mediated by protein–protein interactions, and mapping the interactome provides fundamental insights into biological systems. Affinity purification coupled to mass spectrometry is an ideal tool for such mapping, but it has been difficult to identify low copy number complexes, membrane complexes and complexes that are disrupted by protein tagging. As a result, our current knowledge of the interactome is far from complete, and assessing the reliability of reported interactions is challenging. Here we develop a sensitive high-throughput method using highly reproducible affinity enrichment coupled to mass spectrometry combined with a quantitative two-dimensional analysis strategy to comprehensively map the interactome of Saccharomyces cerevisiae. Thousand-fold reduced volumes in 96-well format enabled replicate analysis of the endogenous GFP-tagged library covering the entire expressed yeast proteome. The 4,159 pull-downs generated a highly structured network of 3,927 proteins connected by 31,004 interactions, doubling the number of proteins and tripling the number of reliable interactions compared with existing interactome maps. This includes very-low-abundance epigenetic complexes, organellar membrane complexes and non-taggable complexes inferred by abundance correlation. This nearly saturated interactome reveals that the vast majority of yeast proteins are highly connected, with an average of 16 interactors. Similar to social networks between humans, the average shortest distance between proteins is 4.2 interactions. AlphaFold-Multimer provided novel insights into the functional roles of previously uncharacterized proteins in complexes. Our web portal (www.yeast-interactome.org) enables extensive exploration of the interactome dataset.

A protein interaction network constructed with data from high-throughput affinity enrichment coupled to mass spectrometry provides a highly saturated yeast interactome with 31,004 interactions, including low-abundance complexes, membrane protein complexes and non-taggable protein complexes.

[end paper 30]

[begin paper 31]

PMCID = 11561318

Title = PPP1R2 stimulates protein phosphatase-1 through stabilisation of dynamic subunit interactions

Protein Ser/Thr phosphatase PP1 is always associated with one or two regulatory subunits or RIPPOs. One of the earliest evolved RIPPOs is PPP1R2, also known as Inhibitor-2. Since its discovery nearly 5 decades ago, PPP1R2 has been variously described as an inhibitor, activator or (metal) chaperone of PP1, but it is still unknown how PPP1R2 affects the function of PP1 in intact cells. Here, using specific research tools, we demonstrate that PPP1R2 stabilises a subgroup of PP1 holoenzymes, exemplified by PP1:RepoMan, thereby promoting the dephosphorylation of their substrates. Mechanistically, the recruitment of PPP1R2 disrupts an inhibitory, fuzzy interaction between the C-terminal tail and catalytic domain of PP1, and generates an additional C-terminal RepoMan-interaction site. The resulting holoenzyme is further stabilized by a direct PPP1R2:RepoMan interaction, which renders it refractory to competitive disruption by RIPPOs that do not interact with PPP1R2. Our data demonstrate that PPP1R2 modulates the function of PP1 by altering the balance between holoenzymes through stabilisation of specific subunit interactions.

PPP1R2, also known as Inhibitor-2, is a regulator of PP1 with an elusive function. Here, the authors show that PPP1R2 functions as an additional subunit of a subset of PP1 holoenzymes, thereby promoting the dephosphorylation of their substrates through stabilization of subunit interactions.

[end paper 31]

[begin paper 32]

PMCID = 11734925

Title = Multi-tissue characterization of the constitutive heterochromatin proteome in Drosophila identifies a link between satellite DNA organization and transposon repression

Noncoding satellite DNA repeats are abundant at the pericentromeric heterochromatin of eukaryotic chromosomes. During interphase, sequence-specific DNA-binding proteins cluster these repeats from multiple chromosomes into nuclear foci known as chromocenters. Despite the pivotal role of chromocenters in cellular processes like genome encapsulation and gene repression, the associated proteins remain incompletely characterized. Here, we use 2 satellite DNA-binding proteins, D1 and Prod, as baits to characterize the chromocenter-associated proteome in Drosophila embryos, ovaries, and testes through quantitative mass spectrometry. We identify D1- and Prod-associated proteins, including known heterochromatin proteins as well as proteins previously unlinked to satellite DNA or chromocenters, thereby laying the foundation for a comprehensive understanding of cellular functions enabled by satellite DNA repeats and their associated proteins. Interestingly, we find that multiple components of the transposon-silencing piRNA pathway are associated with D1 and Prod in embryos. Using genetics, transcriptomics, and small RNA profiling, we show that flies lacking D1 during embryogenesis exhibit transposon expression and gonadal atrophy as adults. We further demonstrate that this gonadal atrophy can be rescued by mutating the checkpoint kinase, Chk2, which mediates germ cell arrest in response to transposon mobilization. Thus, we reveal that a satellite DNA-binding protein functions during embryogenesis to silence transposons, in a manner that is heritable across later stages of development.

Pericentromeric heterochromatin clusters into nuclear foci known as chromocenters, the proteome of which remains largely unknown. This study uses quantitative mass spectrometry to characterize the chromocenter-associated proteome in multiple tissues, identifying a link between satellite DNA-binding proteins and transposon repression during embryogenesis.

[end paper 32]

[begin paper 33]

PMCID = 11066466

Title = Recent advances and challenges in protein complex model accuracy estimation

Estimation of model accuracy plays a crucial role in protein structure prediction, aiming to evaluate the quality of predicted protein structure models accurately and objectively. This process is not only key to screening candidate models that are close to the real structure, but also provides guidance for further optimization of protein structures. With the significant advancements made by AlphaFold2 in monomer structure, the problem of single-domain protein structure prediction has been widely solved. Correspondingly, the importance of assessing the quality of single-domain protein models decreased, and the research focus has shifted to estimation of model accuracy of protein complexes. In this review, our goal is to provide a comprehensive overview of the reference and statistical metrics, as well as representative methods, and the current challenges within four distinct facets (Topology Global Score, Interface Total Score, Interface Residue-Wise Score, and Tertiary Residue-Wise Score) in the field of complex EMA.

[end paper 33]

[begin paper 34]

PMCID = 10942596

Title = Mapping the architecture of the initiating phosphoglycosyl transferase from S. enterica O-antigen biosynthesis in a liponanoparticle

Bacterial cell surface glycoconjugates are critical for cell survival and for interactions between bacteria and their hosts. Consequently, the pathways responsible for their biosynthesis have untapped potential as therapeutic targets. The localization of many glycoconjugate biosynthesis enzymes to the membrane represents a significant challenge for expressing, purifying, and characterizing these enzymes. Here, we leverage cutting-edge detergent-free methods to stabilize, purify, and structurally characterize WbaP, a phosphoglycosyl transferase (PGT) from the Salmonella enterica (LT2) O-antigen biosynthesis. From a functional perspective, these studies establish WbaP as a homodimer, reveal the structural elements responsible for dimerization, shed light on the regulatory role of a domain of unknown function embedded within WbaP, and identify conserved structural motifs between PGTs and functionally unrelated UDP-sugar dehydratases. From a technological perspective, the strategy developed here is generalizable and provides a toolkit for studying other classes of small membrane proteins embedded in liponanoparticles beyond PGTs.

[end paper 34]

[begin paper 35]

PMCID = 10219652

Title = Insights into cargo sorting by SNX32 and its role in neurite outgrowth

Sorting nexins (SNX) are a family of proteins containing the Phox homology domain, which shows a preferential endo-membrane association and regulates cargo sorting processes. Here, we established that SNX32, an SNX-BAR (Bin/Amphiphysin/Rvs) sub-family member associates with SNX4 via its BAR domain and the residues A226, Q259, E256, R366 of SNX32, and Y258, S448 of SNX4 that lie at the interface of these two SNX proteins mediate this association. SNX32, via its PX domain, interacts with the transferrin receptor (TfR) and Cation-Independent Mannose-6-Phosphate Receptor (CIMPR), and the conserved F131 in its PX domain is important in stabilizing these interactions. Silencing of SNX32 leads to a defect in intracellular trafficking of TfR and CIMPR. Further, using SILAC-based differential proteomics of the wild-type and the mutant SNX32, impaired in cargo binding, we identified Basigin (BSG), an immunoglobulin superfamily member, as a potential interactor of SNX32 in SHSY5Y cells. We then demonstrated that SNX32 binds to BSG through its PX domain and facilitates its trafficking to the cell surface. In neuroglial cell lines, silencing of SNX32 leads to defects in neuronal differentiation. Moreover, abrogation in lactate transport in the SNX32-depleted cells led us to propose that SNX32 may contribute to maintaining the neuroglial coordination via its role in BSG trafficking and the associated monocarboxylate transporter activity. Taken together, our study showed that SNX32 mediates the trafficking of specific cargo molecules along distinct pathways.

[end paper 35]

[begin paper 36]

PMCID = 9309271

Title =  Plasmodium 6-Cysteine Proteins: Functional Diversity, Transmission-Blocking Antibodies and Structural Scaffolds

The 6-cysteine protein family is one of the most abundant surface antigens that are expressed throughout the Plasmodium falciparum life cycle. Many members of the 6-cysteine family have critical roles in parasite development across the life cycle in parasite transmission, evasion of the host immune response and host cell invasion. The common feature of the family is the 6-cysteine domain, also referred to as s48/45 domain, which is conserved across Aconoidasida. This review summarizes the current approaches for recombinant expression for 6-cysteine proteins, monoclonal antibodies against 6-cysteine proteins that block transmission and the growing collection of crystal structures that provide insights into the functional domains of this protein family.

[end paper 36]

[begin paper 37]

PMCID = 11679646

Title = The Application of Machine Learning on Antibody Discovery and Optimization

Antibodies play critical roles in modern medicine, serving as diagnostics and therapeutics for various diseases due to their ability to specifically bind to target antigens. Traditional antibody discovery and optimization methods are time-consuming and resource-intensive, though they have successfully generated antibodies for diagnosing and treating diseases. The advancements in protein data, computational hardware, and machine learning (ML) models have the opportunity to disrupt antibody discovery and optimization research. Machine learning models have demonstrated their abilities in antibody design. These machine learning models enable rapid in silico design of antibody candidates within a few days, achieving approximately a 60% reduction in time and a 50% reduction in cost compared to traditional methods. This review focuses on the latest machine learning-based antibody discovery and optimization developments. We briefly discuss the limitations of traditional methods and then explore the machine learning-based antibody discovery and optimization methodologies. We also focus on future research directions, including developing Antibody Design AI Agents and data foundries, alongside the ethical and regulatory considerations essential for successfully adopting machine learning-driven antibody designs.

[end paper 37]

[begin paper 38]

PMCID = 10502012

Title = Molecular basis of TASL recruitment by the peptide/histidine transporter 1, PHT1

PHT1 is a histidine /oligopeptide transporter with an essential role in Toll-like receptor innate immune responses. It can act as a receptor by recruiting the adaptor protein TASL which leads to type I interferon production via IRF5. Persistent stimulation of this signalling pathway is known to be involved in the pathogenesis of systemic lupus erythematosus (SLE). Understanding how PHT1 recruits TASL at the molecular level, is therefore clinically important for the development of therapeutics against SLE and other autoimmune diseases. Here we present the Cryo-EM structure of PHT1 stabilized in the outward-open conformation. By combining biochemical and structural modeling techniques we propose a model of the PHT1-TASL complex, in which the first 16 N-terminal TASL residues fold into a helical structure that bind in the central cavity of the inward-open conformation of PHT1. This work provides critical insights into the molecular basis of PHT1/TASL mediated type I interferon production.

The peptide/histidine transporter 1, PHT1 (SLC15A4), is required for TLR-IRF5 activation via the adaptor protein TASL. Here, the authors determined the structure of PHT1 in the outward-open conformation and present a model of the PHT1-TASL complex where the first 16 residues of TASL bind into the central cavity of PHT1.

[end paper 38]

[begin paper 39]

PMCID = 10933040

Title = Investigating the ability of deep learning-based structure prediction to extrapolate and/or enrich the set of antibody CDR canonical forms

Deep learning models have been shown to accurately predict protein structure from sequence, allowing researchers to explore protein space from the structural viewpoint. In this paper we explore whether “novel” features, such as distinct loop conformations can arise from these predictions despite not being present in the training data. Here we have used ABodyBuilder2, a deep learning antibody structure predictor, to predict the structures of ~1.5M paired antibody sequences. We examined the predicted structures of the canonical CDR loops and found that most of these predictions fall into the already described CDR canonical form structural space. We also found a small number of “new” canonical clusters composed of heterogeneous sequences united by a common sequence motif and loop conformation. Analysis of these novel clusters showed their origins to be either shapes seen in the training data at very low frequency or shapes seen at high frequency but at a shorter sequence length. To evaluate explicitly the ability of ABodyBuilder2 to extrapolate, we retrained several models whilst withholding all antibody structures of a specific CDR loop length or canonical form. These “starved” models showed evidence of generalisation across CDRs of different lengths, but they did not extrapolate to loop conformations which were highly distinct from those present in the training data. However, the models were able to accurately predict a canonical form even if only a very small number of examples of that shape were in the training data. Our results suggest that deep learning protein structure prediction methods are unable to make completely out-of-domain predictions for CDR loops. However, in our analysis we also found that even minimal amounts of data of a structural shape allow the method to recover its original predictive abilities. We have made the ~1.5 M predicted structures used in this study available to download at https://doi.org/10.5281/zenodo.10280181.

[end paper 39]

[begin paper 40]

PMCID = 11065263

Title = Beyond the VSG layer: Exploring the role of intrinsic disorder in the invariant surface glycoproteins of African trypanosomes

In the bloodstream of mammalian hosts, African trypanosomes face the challenge of protecting their invariant surface receptors from immune detection. This crucial role is fulfilled by a dense, glycosylated protein layer composed of variant surface glycoproteins (VSGs), which undergo antigenic variation and provide a physical barrier that shields the underlying invariant surface glycoproteins (ISGs). The protective shield’s limited permeability comes at the cost of restricted access to the extracellular host environment, raising questions regarding the specific function of the ISG repertoire. In this study, we employ an integrative structural biology approach to show that intrinsically disordered membrane-proximal regions are a common feature of members of the ISG super-family, conferring the ability to switch between compact and elongated conformers. While the folded, membrane-distal ectodomain is buried within the VSG layer for compact conformers, their elongated counterparts would enable the extension beyond it. This dynamic behavior enables ISGs to maintain a low immunogenic footprint while still allowing them to engage with the host environment when necessary. Our findings add further evidence to a dynamic molecular organization of trypanosome surface antigens wherein intrinsic disorder underpins the characteristics of a highly flexible ISG proteome to circumvent the constraints imposed by the VSG coat.

In the blood of their human and animal hosts, single-celled parasites of the species Trypanosoma brucei need to hide from the body’s defense system to survive. These parasites have a special armor made of proteins (known as VSGs) that changes regularly, helping them stay undetected by the immune system. This armor also covers other important proteins (ISGs) that the parasites need in order to interact with their host, but must hide to avoid being attacked. However, this protective layer makes it hard for the parasites to reach outside their shield. In our study, we looked at these hidden proteins in detail and discovered that they can change shape. Some can stretch out beyond their protective layer when necessary, while others stay concealed inside. This ability to change shape lets these otherwise hidden proteins interact with their surroundings without getting caught by the immune system. Our findings provide further evidence for a multi-layered defense strategy in these parasites.

[end paper 40]

[begin paper 41]

PMCID = 8983713

Title = Structural validation and assessment of AlphaFold2 predictions for centrosomal and centriolar proteins and their complexes

Obtaining the high-resolution structures of proteins and their complexes is a crucial aspect of understanding the mechanisms of life. Experimental structure determination methods are time-consuming, expensive and cannot keep pace with the growing number of protein sequences available through genomic DNA sequencing. Thus, the ability to accurately predict the structure of proteins from their sequence is a holy grail of structural and computational biology that would remove a bottleneck in our efforts to understand as well as rationally engineer living systems. Recent advances in protein structure prediction, in particular the breakthrough with the AI-based tool AlphaFold2 (AF2), hold promise for achieving this goal, but the practical utility of AF2 remains to be explored. Focusing on proteins with essential roles in centrosome and centriole biogenesis, we demonstrate the quality and usability of the AF2 prediction models and we show that they can provide important insights into the modular organization of two key players in this process, CEP192 and CEP44. Furthermore, we used the AF2 algorithm to elucidate and then experimentally validate previously unknown prime features in the structure of TTBK2 bound to CEP164, as well as the Chibby1-FAM92A complex for which no structural information was available to date. These findings have important implications in understanding the regulation and function of these complexes. Finally, we also discuss some practical limitations of AF2 and anticipate the implications for future research approaches in the centriole/centrosome field.

Using experimental data, the authors assess the quality and discuss the limitations of AlphaFold2 predictions of protein structures and protein-protein interactions essential for centrosome and centriole biogenesis.

[end paper 41]

[begin paper 42]

PMCID = 11483107

Title = Towards the accurate modelling of antibody−antigen complexes from sequence using machine learning and information-driven docking

Antibody−antigen complex modelling is an important step in computational workflows for therapeutic antibody design. While experimentally determined structures of both antibody and the cognate antigen are often not available, recent advances in machine learning-driven protein modelling have enabled accurate prediction of both antibody and antigen structures. Here, we analyse the ability of protein−protein docking tools to use machine learning generated input structures for information-driven docking.

In an information-driven scenario, we find that HADDOCK can generate accurate models of antibody−antigen complexes using an ensemble of antibody structures generated by machine learning tools and AlphaFold2 predicted antigen structures. Targeted docking using knowledge of the complementary determining regions on the antibody and some information about the targeted epitope allows the generation of high-quality models of the complex with reduced sampling, resulting in a computationally cheap protocol that outperforms the ZDOCK baseline.

The source code of HADDOCK3 is freely available at github.com/haddocking/haddock3. The code to generate and analyse the data is available at github.com/haddocking/ai-antibodies. The full runs, including docking models from all modules of a workflow have been deposited in our lab collection (data.sbgrid.org/labs/32/1139) at the SBGRID data repository.

[end paper 42]

[begin paper 43]

PMCID = 11128484

Title = Accurate prediction of antibody function and structure using bio-inspired antibody language model

In recent decades, antibodies have emerged as indispensable therapeutics for combating diseases, particularly viral infections. However, their development has been hindered by limited structural information and labor-intensive engineering processes. Fortunately, significant advancements in deep learning methods have facilitated the precise prediction of protein structure and function by leveraging co-evolution information from homologous proteins. Despite these advances, predicting the conformation of antibodies remains challenging due to their unique evolution and the high flexibility of their antigen-binding regions. Here, to address this challenge, we present the Bio-inspired Antibody Language Model (BALM). This model is trained on a vast dataset comprising 336 million 40% nonredundant unlabeled antibody sequences, capturing both unique and conserved properties specific to antibodies. Notably, BALM showcases exceptional performance across four antigen-binding prediction tasks. Moreover, we introduce BALMFold, an end-to-end method derived from BALM, capable of swiftly predicting full atomic antibody structures from individual sequences. Remarkably, BALMFold outperforms those well-established methods like AlphaFold2, IgFold, ESMFold and OmegaFold in the antibody benchmark, demonstrating significant potential to advance innovative engineering and streamline therapeutic antibody development by reducing the need for unnecessary trials. The BALMFold structure prediction server is freely available at https://beamlab-sh.com/models/BALMFold.

[end paper 43]

[begin paper 44]

PMCID = 10594513

Title = Structure and Dynamics Guiding Design of Antibody Therapeutics and Vaccines

Antibodies and other new antibody-like formats have emerged as one of the most rapidly growing classes of biotherapeutic proteins. Understanding the structural features that drive antibody function and, consequently, their molecular recognition is critical for engineering antibodies. Here, we present the structural architecture of conventional IgG antibodies alongside other formats. We emphasize the importance of considering antibodies as conformational ensembles in solution instead of focusing on single-static structures because their functions and properties are strongly governed by their dynamic nature. Thus, in this review, we provide an overview of the unique structural and dynamic characteristics of antibodies with respect to their antigen recognition, biophysical properties, and effector functions. We highlight the numerous technical advances in antibody structure prediction and design, enabled by the vast number of experimentally determined high-quality structures recorded with cryo-EM, NMR, and X-ray crystallography. Lastly, we assess antibody and vaccine design strategies in the context of structure and dynamics.

[end paper 44]

[begin paper 45]

PMCID = 9597551

Title = Structural basis of activation and antagonism of receptor signaling mediated by interleukin-27

Interleukin-27 (IL-27) uniquely assembles p28 and EBI3 subunits to a heterodimeric cytokine that signals via IL-27Rα and gp130. To provide the structural framework for receptor activation by IL-27 and its emerging therapeutic targeting, we report here crystal structures of mouse IL-27 in complex with IL-27Rα and of human IL-27 in complex with SRF388, a monoclonal antibody undergoing clinical trials with oncology indications. One face of the helical p28 subunit interacts with EBI3, while the opposite face nestles into the interdomain elbow of IL-27Rα to juxtapose IL-27Rα to EBI3. This orients IL-27Rα for paired signaling with gp130, which only uses its immunoglobulin domain to bind to IL-27. Such a signaling complex is distinct from those mediated by IL-12 and IL-23. The SRF388 binding epitope on IL-27 overlaps with the IL-27Rα interaction site explaining its potent antagonistic properties. Collectively, our findings will facilitate the mechanistic interrogation, engineering, and therapeutic targeting of IL-27.

Mouse and human IL-27 adopt highly similar structures

IL-27Rα interacts both with the p28 and EBI3 subunits of IL-27

SRF388 and IL-27Rα occupy mutually exclusive binding sites on IL-27

IL-27 mediates receptor assemblies distinct from IL-12 and IL-23

Interleukin-27 (IL-27) activates its cognate receptors on cells to regulate innate and adaptive immune responses in T cell populations, an important area for basic and applied biomedicine. Składanowska and Bloch et al. report the structure and mechanism by which IL-27 binds its signaling receptors and show how a therapeutic antibody antagonizes IL-27 activity.

[end paper 45]

[begin paper 46]

PMCID = 10841881

Title = Impact of AlphaFold on structure prediction of protein complexes: The CASP15‐CAPRI experiment

We present the results for CAPRI Round 54, the 5th joint CASP‐CAPRI protein assembly prediction challenge. The Round offered 37 targets, including 14 homodimers, 3 homo‐trimers, 13 heterodimers including 3 antibody–antigen complexes, and 7 large assemblies. On average ~70 CASP and CAPRI predictor groups, including more than 20 automatics servers, submitted models for each target. A total of 21 941 models submitted by these groups and by 15 CAPRI scorer groups were evaluated using the CAPRI model quality measures and the DockQ score consolidating these measures. The prediction performance was quantified by a weighted score based on the number of models of acceptable quality or higher submitted by each group among their five best models. Results show substantial progress achieved across a significant fraction of the 60+ participating groups. High‐quality models were produced for about 40% of the targets compared to 8% two years earlier. This remarkable improvement is due to the wide use of the AlphaFold2 and AlphaFold2‐Multimer software and the confidence metrics they provide. Notably, expanded sampling of candidate solutions by manipulating these deep learning inference engines, enriching multiple sequence alignments, or integration of advanced modeling tools, enabled top performing groups to exceed the performance of a standard AlphaFold2‐Multimer version used as a yard stick. This notwithstanding, performance remained poor for complexes with antibodies and nanobodies, where evolutionary relationships between the binding partners are lacking, and for complexes featuring conformational flexibility, clearly indicating that the prediction of protein complexes remains a challenging problem.

[end paper 46]

[begin paper 47]

PMCID = 10944524

Title = SpyMask enables combinatorial assembly of bispecific binders

Bispecific antibodies are a successful and expanding therapeutic class. Standard approaches to generate bispecifics are complicated by the need for disulfide reduction/oxidation or specialized formats. Here we present SpyMask, a modular approach to bispecifics using SpyTag/SpyCatcher spontaneous amidation. Two SpyTag-fused antigen-binding modules can be precisely conjugated onto DoubleCatcher, a tandem SpyCatcher where the second SpyCatcher is protease-activatable. We engineer a panel of structurally-distinct DoubleCatchers, from which binders project in different directions. We establish a generalized methodology for one-pot assembly and purification of bispecifics in 96-well plates. A panel of binders recognizing different HER2 epitopes were coupled to DoubleCatcher, revealing unexpected combinations with anti-proliferative or pro-proliferative activity on HER2-addicted cancer cells. Bispecific activity depended sensitively on both binder orientation and DoubleCatcher scaffold geometry. These findings support the need for straightforward assembly in different formats. SpyMask provides a scalable tool to discover synergy in bispecific activity, through modulating receptor organization and geometry.

Bispecific antibody architecture is often important for function but rarely optimized. Here, authors present a modular approach to assemble bispecifics in varied formats using a SpyTag/SpyCatcher approach called SpyMask, and build anti-HER2 bispecifics whose activities depend on binder orientation and bispecific geometry.

[end paper 47]

[begin paper 48]

PMCID = 11006224

Title = Combinatorially restricted computational design of protein-protein interfaces to produce IgG heterodimers

Redesigning protein-protein interfaces is an important tool for developing therapeutic strategies. Interfaces can be redesigned by in silico screening, which allows for efficient sampling of a large protein space before experimental validation. However, computational costs limit the number of combinations that can be reasonably sampled. Here, we present combinatorial tyrosine (Y)/serine (S) selection (combYSelect), a computational approach combining in silico determination of the change in binding free energy (ΔΔG) of an interface with a highly restricted library composed of just two amino acids, tyrosine and serine. We used combYSelect to design two immunoglobulin G (IgG) heterodimers—combYSelect1 (L368S/D399Y-K409S/T411Y) and combYSelect2 (D399Y/K447S-K409S/T411Y)—that exhibit near-optimal heterodimerization, without affecting IgG stability or function. We solved the crystal structures of these heterodimers and found that dynamic π-stacking interactions and polar contacts drive preferential heterodimeric interactions. Finally, we demonstrated the utility of our combYSelect heterodimers by engineering both a bispecific antibody and a cytokine trap for two unique therapeutic applications.

combYSelect is an in silico screening strategy for redesigning protein-protein interfaces using a restricted amino acid library.

[end paper 48]

[begin paper 49]

PMCID = 11703149

Title = Structural basis for a nucleoporin exportin complex between RanBP2, SUMO1-RanGAP1, the E2 Ubc9, Crm1 and the Ran GTPase

The human nucleoporin RanBP2/Nup358 interacts with SUMO1-modified RanGAP1 and the SUMO E2 Ubc9 at the nuclear pore complex (NPC) to promote export and disassembly of exportin Crm1/Ran(GTP)/cargo complexes. In mitosis, RanBP2/SUMO1-RanGAP1/Ubc9 remains intact after NPC disassembly and is recruited to kinetochores and mitotic spindles by Crm1 where it contributes to mitotic progression. Interestingly, RanBP2 binds SUMO1-RanGAP1/Ubc9 via motifs that also catalyze SUMO E3 ligase activity. Here, we resolve cryo-EM structures of a RanBP2 C-terminal fragment in complex with Crm1, SUMO1-RanGAP1/Ubc9, and two molecules of Ran(GTP). These structures reveal several unanticipated interactions with Crm1 including a nuclear export signal (NES) for RanGAP1, the deletion of which mislocalizes RanGAP1 and the Ran GTPase in cells. Our structural and biochemical results support models in which RanBP2 E3 ligase activity is dependent on Crm1, the RanGAP1 NES and Ran GTPase cycling.

[end paper 49]

[begin paper 50]

PMCID = 11626814

Title = How to target membrane proteins for degradation: Bringing GPCRs into the TPD fold

We are now in the middle of a so-called “fourth wave” of drug innovation: multispecific medicines aimed at diseases and targets previously thought to be “undruggable”; by inducing proximity between two or more proteins, for example, a target and an effector that do not naturally interact, such modalities have potential far beyond the scope of conventional drugs. In particular, targeted protein degradation (TPD) strategies to destroy disease-associated proteins have emerged as an exciting pipeline in drug discovery. Most efforts are focused on intracellular proteins, whereas membrane proteins have been less thoroughly explored despite the fact that they comprise roughly a quarter of the human proteome with G-protein coupled receptors (GPCRs) notably dysregulated in many diseases. Here, we discuss the opportunities and challenges of developing degraders for membrane proteins with a focus on GPCRs. We provide an overview of different TPD platforms in the context of membrane-tethered targets, and we present recent degradation technologies highlighting their potential application to GPCRs.

[end paper 50]

[begin paper 51]

PMCID = 10416210

Title = Crystal Structure of de Novo Designed Coiled-Coil Protein Origami Triangle

Coiled-coil protein origami (CCPO) uses modular coiled-coil building blocks and topological principles to design polyhedral structures distinct from those of natural globular proteins. While the CCPO strategy has proven successful in designing diverse protein topologies, no high-resolution structural information has been available about these novel protein folds. Here we report the crystal structure of a single-chain CCPO in the shape of a triangle. While neither cyclization nor the addition of nanobodies enabled crystallization, it was ultimately facilitated by the inclusion of a GCN2 homodimer. Triangle edges are formed by the orthogonal parallel coiled-coil dimers P1:P2, P3:P4, and GCN2 connected by short linkers. A triangle has a large central cavity and is additionally stabilized by side-chain interactions between neighboring segments at each vertex. The crystal lattice is densely packed and stabilized by a large number of contacts between triangles. Interestingly, the polypeptide chain folds into a trefoil-type protein knot topology, and AlphaFold2 fails to predict the correct fold. The structure validates the modular CC-based protein design strategy, providing molecular insight underlying CCPO stabilization and new opportunities for the design.

[end paper 51]

[begin paper 52]

PMCID = 8950575

Title = Next-Generation Molecular Discovery: From Bottom-Up In Vivo and In Vitro Approaches to In Silico Top-Down Approaches for Therapeutics Neogenesis

Protein and drug engineering comprises a major part of the medical and research industries, and yet approaches to discovering and understanding therapeutic molecular interactions in biological systems rely on trial and error. The general approach to molecular discovery involves screening large libraries of compounds, proteins, or antibodies, or in vivo antibody generation, which could be considered “bottom-up” approaches to therapeutic discovery. In these bottom-up approaches, a minimal amount is known about the therapeutics at the start of the process, but through meticulous and exhaustive laboratory work, the molecule is characterised in detail. In contrast, the advent of “big data” and access to extensive online databases and machine learning technologies offers promising new avenues to understanding molecular interactions. Artificial intelligence (AI) now has the potential to predict protein structure at an unprecedented accuracy using only the genetic sequence. This predictive approach to characterising molecular structure—when accompanied by high-quality experimental data for model training—has the capacity to invert the process of molecular discovery and characterisation. The process has potential to be transformed into a top-down approach, where new molecules can be designed directly based on the structure of a target and the desired function, rather than performing screening of large libraries of molecular variants. This paper will provide a brief evaluation of bottom-up approaches to discovering and characterising biological molecules and will discuss recent advances towards developing top-down approaches and the prospects of this.

[end paper 52]

[begin paper 53]

PMCID = 9906597

Title = Targeted photodynamic neutralization of SARS-CoV-2 mediated by singlet oxygen

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) virus has been on a rampage for more than two years. Vaccines in combination with neutralizing antibodies (NAbs) against SARS-CoV-2 carry great hope in the treatment and final elimination of coronavirus disease 2019 (COVID-19). However, the relentless emergence of variants of concern (VOC), including the most recent Omicron variants, presses for novel measures to counter these variants that often show immune evasion. Hereby we developed a targeted photodynamic approach to neutralize SARS-CoV-2 by engineering a genetically encoded photosensitizer (SOPP3) to a diverse list of antibodies targeting the wild-type (WT) spike protein, including human antibodies isolated from a 2003 Severe acute respiratory syndrome (SARS) patient, potent monomeric and multimeric nanobodies targeting receptor-binding domain (RBD), and non-neutralizing antibodies (non-NAbs) targeting the more conserved N-terminal domain (NTD). As confirmed by pseudovirus neutralization assay, this targeted photodynamic approach significantly increased the efficacy of these antibodies, especially that of non-NAbs, against not only the WT but also the Delta strain and the heavily immune escape Omicron strain (BA.1). Subsequent measurement of infrared phosphorescence at 1270 nm confirmed the generation of singlet oxygen (1O2) in the photodynamic process. Mass spectroscopy assay uncovered amino acids in the spike protein targeted by 1O2. Impressively, Y145 and H146 form an oxidization “hotspot”, which overlaps with the antigenic “supersite” in NTD. Taken together, our study established a targeted photodynamic approach against the SARS-CoV-2 virus and provided mechanistic insights into the photodynamic modification of protein molecules mediated by 1O2.

The online version contains supplementary material available at 10.1007/s43630-023-00381-w.

[end paper 53]

[begin paper 54]

PMCID = 9817007

Title = Protein-protein interaction-Gaussian accelerated molecular dynamics (PPI-GaMD): Characterization of protein binding thermodynamics and kinetics

Protein-protein interactions (PPIs) play key roles in many fundamental biological processes such as cellular signaling and immune responses. However, it has proven challenging to simulate repetitive protein association and dissociation in order to calculate binding free energies and kinetics of PPIs, due to long biological timescales and complex protein dynamics. To address this challenge, we have developed a new computational approach to all-atom simulations of PPIs based on a robust Gaussian accelerated molecular dynamics (GaMD) technique. The method, termed “PPI-GaMD”, selectively boosts interaction potential energy between protein partners to facilitate their slow dissociation. Meanwhile, another boost potential is applied to the remaining potential energy of the entire system to effectively model the protein’s flexibility and rebinding. PPI-GaMD has been demonstrated on a model system of the ribonuclease barnase interactions with its inhibitor barstar. Six independent 2 μs PPI-GaMD simulations have captured repetitive barstar dissociation and rebinding events, which enable calculations of the protein binding thermodynamics and kinetics simultaneously. The calculated binding free energies and kinetic rate constants agree well with the experimental data. Furthermore, PPI-GaMD simulations have provided mechanistic insights into barstar binding to barnase, which involve long-range electrostatic interactions and multiple binding pathways, being consistent with previous experimental and computational findings of this model system. In summary, PPI-GaMD provides a highly efficient and easy-to-use approach for binding free energy and kinetics calculations of PPIs.

All-atom protein-protein interaction-Gaussian accelerated molecular dynamics (PPI-GaMD) simulations captured repeptitive binding and dissociaiton of barstar to the barnase model system, which enabled us to characterize the protein binding pathways, thermodynamcis and kinetics.

[end paper 54]

[begin paper 55]

PMCID = 11783119

Title = Epitope mapping via in vitro deep mutational scanning methods and its applications

Epitope mapping is a technique employed to define the region of an antigen that elicits an immune response, providing crucial insight into the structural architecture of the antigen as well as epitope-paratope interactions. With this breadth of knowledge, immunotherapies, diagnostics, and vaccines are being developed with a rational and data-supported design. Traditional epitope mapping methods are laborious, time-intensive, and often lack the ability to screen proteins in a high-throughput manner or provide high resolution. Deep mutational scanning (DMS), however, is revolutionizing the field as it can screen all possible single amino acid mutations and provide an efficient and high-throughput way to infer the structures of both linear and three-dimensional epitopes with high resolution. Currently, more than 50 publications take this approach to efficiently identify enhancing or escaping mutations, with many then employing this information to rapidly develop broadly neutralizing antibodies, T-cell immunotherapies, vaccine platforms, or diagnostics. We provide a comprehensive review of the approaches to accomplish epitope mapping while also providing a summation of the development of DMS technology and its impactful applications.

[end paper 55]

[begin paper 56]

PMCID = 11180690

Title = One step forward towards deep‐learning protein complex structure prediction by precise multiple sequence alignment construction



[end paper 56]

[begin paper 57]

PMCID = 10820826

Title = Interrogation and validation of the interactome of neuronal Munc18-interacting Mint proteins with AlphaFold2

Munc18-interacting proteins (Mints) are multidomain adaptors that regulate neuronal membrane trafficking, signaling, and neurotransmission. Mint1 and Mint2 are highly expressed in the brain with overlapping roles in the regulation of synaptic vesicle fusion required for neurotransmitter release by interacting with the essential synaptic protein Munc18-1. Here, we have used AlphaFold2 to identify and then validate the mechanisms that underpin both the specific interactions of neuronal Mint proteins with Munc18-1 as well as their wider interactome. We found that a short acidic α-helical motif within Mint1 and Mint2 is necessary and sufficient for specific binding to Munc18-1 and binds a conserved surface on Munc18-1 domain3b. In Munc18-1/2 double knockout neurosecretory cells, mutation of the Mint-binding site reduces the ability of Munc18-1 to rescue exocytosis, and although Munc18-1 can interact with Mint and Sx1a (Syntaxin1a) proteins simultaneously in vitro, we find that they have mutually reduced affinities, suggesting an allosteric coupling between the proteins. Using AlphaFold2 to then examine the entire cellular network of putative Mint interactors provides a structural model for their assembly with a variety of known and novel regulatory and cargo proteins including ADP-ribosylation factor (ARF3/ARF4) small GTPases and the AP3 clathrin adaptor complex. Validation of Mint1 interaction with a new predicted binder TJAP1 (tight junction–associated protein 1) provides experimental support that AlphaFold2 can correctly predict interactions across such large-scale datasets. Overall, our data provide insights into the diversity of interactions mediated by the Mint family and show that Mints may help facilitate a key trigger point in SNARE (soluble N-ethylmaleimide-sensitive factor attachment receptor) complex assembly and vesicle fusion.

[end paper 57]

[begin paper 58]

PMCID = 9989197

Title = Learned features of antibody-antigen binding affinity

Defining predictors of antigen-binding affinity of antibodies is valuable for engineering therapeutic antibodies with high binding affinity to their targets. However, this task is challenging owing to the huge diversity in the conformations of the complementarity determining regions of antibodies and the mode of engagement between antibody and antigen. In this study, we used the structural antibody database (SAbDab) to identify features that can discriminate high- and low-binding affinity across a 5-log scale. First, we abstracted features based on previously learned representations of protein-protein interactions to derive ‘complex’ feature sets, which include energetic, statistical, network-based, and machine-learned features. Second, we contrasted these complex feature sets with additional ‘simple’ feature sets based on counts of contacts between antibody and antigen. By investigating the predictive potential of 700 features contained in the eight complex and simple feature sets, we observed that simple feature sets perform comparably to complex feature sets in classification of binding affinity. Moreover, combining features from all eight feature-sets provided the best classification performance (median cross-validation AUROC and F1-score of 0.72). Of note, classification performance is substantially improved when several sources of data leakage (e.g., homologous antibodies) are not removed from the dataset, emphasizing a potential pitfall in this task. We additionally observe a classification performance plateau across diverse featurization approaches, highlighting the need for additional affinity-labeled antibody-antigen structural data. The findings from our present study set the stage for future studies aimed at multiple-log enhancement of antibody affinity through feature-guided engineering.

[end paper 58]

[begin paper 59]

PMCID = 11409006

Title = Mass spectrometry-complemented molecular modeling predicts the interaction interface for a camelid single-domain antibody targeting the Plasmodium falciparum circumsporozoite protein’s C-terminal domain

Bioanalytical methods that enable rapid and high-detail characterization of binding specificities and strengths of protein complexes with low sample consumption are highly desired. The interaction between a camelid single domain antibody (sdAbCSP1) and its target antigen (PfCSP-Cext) was selected as a model system to provide proof-of-principle for the here described methodology.

The structure of the sdAbCSP1 – PfCSP-Cext complex was modeled using AlphaFold2. The recombinantly expressed proteins, sdAbCSP1, PfCSP-Cext, and the sdAbCSP1 – PfCSP-Cext complex, were subjected to limited proteolysis and mass spectrometric peptide analysis. ITEM MS (Intact Transition Epitope Mapping Mass Spectrometry) and ITC (Isothermal Titration Calorimetry) were applied to determine stoichiometry and binding strength.

The paratope of sdAbCSP1 mainly consists of its CDR3 (aa100–118). PfCSP-Cext’s epitope is assembled from its α-helix (aa40–52) and opposing loop (aa83–90). PfCSP-Cext’s GluC cleavage sites E46 and E58 were shielded by complex formation, confirming the predicted epitope. Likewise, sdAbCSP1’s tryptic cleavage sites R105 and R108 were shielded by complex formation, confirming the predicted paratope. ITEM MS determined the 1:1 stoichiometry and the high complex binding strength, exemplified by the gas phase dissociation reaction enthalpy of 50.2 kJ/mol. The in-solution complex dissociation constant is 5 × 10-10 M.

Combining AlphaFold2 modeling with mass spectrometry/limited proteolysis generated a trustworthy model for the sdAbCSP1 – PfCSP-Cext complex interaction interface.

[end paper 59]

[begin paper 60]

PMCID = 10299731

Title = Challenges and frontiers of computational modelling of biomolecular recognition

Biomolecular recognition including binding of small molecules, peptides and proteins to their target receptors plays a key role in cellular function and has been targeted for therapeutic drug design. However, the high flexibility of biomolecules and slow binding and dissociation processes have presented challenges for computational modelling. Here, we review the challenges and computational approaches developed to characterise biomolecular binding, including molecular docking, molecular dynamics simulations (especially enhanced sampling) and machine learning. Further improvements are still needed in order to accurately and efficiently characterise binding structures, mechanisms, thermodynamics and kinetics of biomolecules in the future.

[end paper 60]

[begin paper 61]

PMCID = 10659397

Title = Evolutionary origin and structural ligand mimicry by the inserted domain of alpha-integrin proteins

Heterodimeric integrin proteins transmit signals through conformational changes upon ligand binding between their alpha (α) and beta (β) subunits. Early in chordate evolution, some α subunits acquired an “inserted” (I) domain, which expanded their ligand binding capacity but simultaneously obstructed the ancestral ligand-binding pocket. While this would seemingly impede conventional ligand-mediated integrin activation, it was proposed that the I domain itself could serve both as a ligand replacement and an activation trigger. Here, we provide compelling evidence in support of this longstanding hypothesis using high-resolution cryo-electron microscopy structures of two distinct integrin complexes: the ligand-free and E-cadherin-bound states of the αEβ7 integrin with the I domain, as well as the α4β7 integrin lacking the I domain in both a ligand-free state and bound to MadCAM-1. We trace the evolutionary origin of the I domain to an ancestral collagen-collagen interaction domain. Our analyses illuminate how the I domain intrinsically mimics an extrinsic ligand, enabling integrins to undergo the canonical allosteric cascade of conformational activation and dramatically expanding the range of cellular communication mechanisms in vertebrates.

[end paper 61]

[begin paper 62]

PMCID = 11604545

Title = Intra-channel bi-epitopic crosslinking unleashes ultrapotent antibodies targeting NaV1.7 for pain alleviation

Crucial for cell activities, ion channels are key drug discovery targets. Although small-molecule and peptide modulators dominate ion channel drug discovery, antibodies are emerging as an alternative modality. However, challenges persist in generating potent antibodies, especially for channels with limited extracellular epitopes. We herein present a bi-epitopic crosslinking strategy to overcome these challenges, focusing on NaV1.7, a potential analgesic target. Aiming to crosslink two non-overlapping epitopes on voltage-sensing domains II and IV, we construct bispecific antibodies and ligand-antibody conjugates. Enhanced affinity and potency are observed in comparison to the monospecific controls. Among them, a ligand-antibody conjugate (1080-PEG7-ACDTB) displays a two-orders-of-magnitude improvement in potency (IC50 of 0.06 ± 0.01 nM) and over 1,000-fold selectivity for NaV1.7. Additionally, this conjugate demonstrates robust analgesic effects in mouse pain models. Our study introduces an approach to developing effective antibodies against NaV1.7, thereby initiating a promising direction for the advancement of pain therapeutics.

Bi-epitopic crosslinking enhances the potency of antibodies targeting ion channels

A ligand-antibody conjugate shows ultrapotent and highly selective NaV1.7 inhibition

The ligand-antibody conjugate exhibits potent analgesic effects in mouse pain models

Zhang et al. introduce a bi-epitopic crosslinking strategy to enhance antibody potency against ion channels, particularly for NaV1.7, a pain relief target. This approach, using bispecific antibodies and ligand-antibody conjugates, significantly improves affinity and potency, with one conjugate showing a 1,000-fold selectivity for NaV1.7 and strong analgesic effects in mice, offering a promising direction for pain therapeutics.

[end paper 62]

[begin paper 63]

PMCID = 9254107

Title = The Toxoplasma glucan phosphatase TgLaforin utilizes a distinct functional mechanism that can be exploited by therapeutic inhibitors

Toxoplasma gondii is an intracellular parasite that generates amylopectin granules (AGs), a polysaccharide associated with bradyzoites that define chronic T. gondii infection. AGs are postulated to act as an essential energy storage molecule that enable bradyzoite persistence, transmission, and reactivation. Importantly, reactivation can result in the life-threatening symptoms of toxoplasmosis. T. gondii encodes glucan dikinase and glucan phosphatase enzymes that are homologous to the plant and animal enzymes involved in reversible glucan phosphorylation and which are required for efficient polysaccharide degradation and utilization. However, the structural determinants that regulate reversible glucan phosphorylation in T. gondii are unclear. Herein, we define key functional aspects of the T. gondii glucan phosphatase TgLaforin (TGME49_205290). We demonstrate that TgLaforin possesses an atypical split carbohydrate-binding-module domain. AlphaFold2 modeling combined with hydrogen–deuterium exchange mass spectrometry and differential scanning fluorimetry also demonstrate the unique structural dynamics of TgLaforin with regard to glucan binding. Moreover, we show that TgLaforin forms a dual specificity phosphatase domain–mediated dimer. Finally, the distinct properties of the glucan phosphatase catalytic domain were exploited to identify a small molecule inhibitor of TgLaforin catalytic activity. Together, these studies define a distinct mechanism of TgLaforin activity, opening up a new avenue of T. gondii bradyzoite biology as a therapeutic target.

[end paper 63]

[begin paper 64]

PMCID = 11745815

Title = Recent Advances in Mass Spectrometry-Based Protein Interactome Studies

The foundation of all biological processes is the network of diverse and dynamic protein interactions with other molecules in cells known as the interactome. Understanding the interactome is crucial for elucidating molecular mechanisms but has been a longstanding challenge. Recent developments in mass spectrometry (MS)-based techniques, including affinity purification, proximity labeling, cross-linking, and co-fractionation mass spectrometry (MS), have significantly enhanced our abilities to study the interactome. They do so by identifying and quantifying protein interactions yielding profound insights into protein organizations and functions. This review summarizes recent advances in MS-based interactomics, focusing on the development of techniques that capture protein-protein, protein-metabolite, and protein-nucleic acid interactions. Additionally, we discuss how integrated MS-based approaches have been applied to diverse biological samples, focusing on significant discoveries that have leveraged our understanding of cellular functions. Finally, we highlight state-of-the-art bioinformatic approaches for predictions of interactome and complex modeling, as well as strategies for combining experimental interactome data with computation methods, thereby enhancing the ability of MS-based techniques to identify protein interactomes. Indeed, advances in MS technologies and their integrations with computational biology provide new directions and avenues for interactome research, leveraging new insights into mechanisms that govern the molecular architecture of living cells and, thereby, our comprehension of biological processes.

Recent advances in MS-based techniques for comprehensive protein interactome mapping.

Integration of experimental MS methods with cutting-edge computational tools.

Emerging strategies for high-resolution interactome analysis in various biological contexts.

This review highlights recent advancements in mass spectrometry-based techniques for mapping protein interactomes, including affinity purification, proximity labeling, cross-linking, and co-fractionation approaches. It discusses the integration of these methods with cutting-edge computational tools, emphasizing their synergistic potential in elucidating complex cellular networks. The review explores how these technologies are revolutionizing our understanding of protein interactions, providing unprecedented insights into cellular processes and opening new avenues for biological research, drug discovery, and disease diagnostics.

[end paper 64]

[begin paper 65]

PMCID = 9994104

Title = Protein–protein interfaces in molecular glue-induced ternary complexes: classification, characterization, and prediction

Molecular glues are a class of small molecules that stabilize the interactions between proteins. Naturally occurring molecular glues are present in many areas of biology where they serve as central regulators of signaling pathways. Importantly, several clinical compounds act as molecular glue degraders that stabilize interactions between E3 ubiquitin ligases and target proteins, leading to their degradation. Molecular glues hold promise as a new generation of therapeutic agents, including those molecular glue degraders that can redirect the protein degradation machinery in a precise way. However, rational discovery of molecular glues is difficult in part due to the lack of understanding of the protein–protein interactions they stabilize. In this review, we summarize the structures of known molecular glue-induced ternary complexes and the interface properties. Detailed analysis shows different mechanisms of ternary structure formation. Additionally, we also review computational approaches for predicting protein–protein interfaces and highlight the promises and challenges. This information will ultimately help inform future approaches for rational molecular glue discovery.

This review surveys molecular glue-induced ternary complexes in the PDB and provides an overview of computational methods that can be utilized to predict them.

[end paper 65]

[begin paper 66]

PMCID = 7616833

Title = One-step nanoscale expansion microscopy reveals individual protein shapes

The attainable resolution of fluorescence microscopy has reached the subnanometer range, but this technique still fails to image the morphology of single proteins or small molecular complexes. Here, we expand the specimens at least tenfold, label them with conventional fluorophores and image them with conventional light microscopes, acquiring videos in which we analyze fluorescence fluctuations. One-step nanoscale expansion (ONE) microscopy enables the visualization of the shapes of individual membrane and soluble proteins, achieving around 1-nm resolution. We show that conformational changes are readily observable, such as those undergone by the ~17-kDa protein calmodulin upon Ca2+ binding. ONE is also applied to clinical samples, analyzing the morphology of protein aggregates in cerebrospinal fluid from persons with Parkinson disease, potentially aiding disease diagnosis. This technology bridges the gap between high-resolution structural biology techniques and light microscopy, providing new avenues for discoveries in biology and medicine.

[end paper 66]

[begin paper 67]

PMCID = 11370311

Title = SART3 reads methylarginine-marked glycine- and arginine-rich motifs

Glycine- and arginine-rich (GAR) motifs, commonly found in RNA-binding and -processing proteins, can be symmetrically (SDMA) or asymmetrically (ADMA) dimethylated at the arginine residue by protein arginine methyltransferases. Arginine-methylated protein motifs are usually read by Tudor domain-containing proteins. Here, using a GFP-Trap, we identify a non-Tudor domain protein, squamous cell carcinoma antigen recognized by T cells 3 (SART3), as a reader for SDMA-marked GAR motifs. Structural analysis and mutagenesis of SART3 show that aromatic residues lining a groove between two adjacent aromatic-rich half-a-tetratricopeptide (HAT) repeat domains are essential for SART3 to recognize and bind to SDMA-marked GAR motif peptides, as well as for the interaction between SART3 and the GAR-motif-containing proteins fibrillarin and coilin. Further, we show that the loss of this reader ability affects RNA splicing. Overall, our findings broaden the range of potential SDMA readers to include HAT domains.

Using a composite arginine methylation substrate, Wang et al. identify SART3 as a “reader” of symmetrically dimethylated arginine (SDMA) motifs. This binding is mediated by a groove formed between two adjacent aromatic-rich half-a-tetratricopeptide (HAT) repeat domains. The integrity of this methyl-dependent interaction is required for normal splicing.

[end paper 67]

[begin paper 68]

PMCID = 9939385

Title = Toward real-world automated antibody design with combinatorial Bayesian optimization

Antibodies are multimeric proteins capable of highly specific molecular recognition. The complementarity determining region 3 of the antibody variable heavy chain (CDRH3) often dominates antigen-binding specificity. Hence, it is a priority to design optimal antigen-specific CDRH3 to develop therapeutic antibodies. The combinatorial structure of CDRH3 sequences makes it impossible to query binding-affinity oracles exhaustively. Moreover, antibodies are expected to have high target specificity and developability. Here, we present AntBO, a combinatorial Bayesian optimization framework utilizing a CDRH3 trust region for an in silico design of antibodies with favorable developability scores. The in silico experiments on 159 antigens demonstrate that AntBO is a step toward practically viable in vitro antibody design. In under 200 calls to the oracle, AntBO suggests antibodies outperforming the best binding sequence from 6.9 million experimentally obtained CDRH3s. Additionally, AntBO finds very-high-affinity CDRH3 in only 38 protein designs while requiring no domain knowledge.

AntBO is a sample-efficient combinatorial Bayesian optimization tool for antibody design

Provides an easy interface to benchmark optimization methods for antibody design

Utilizes trust region to design antibodies with favorable developability scores

Consistently outperforms baseline approaches, including experimentally known CDRH3s

Designing antibody sequences that bind to an antigen of interest is a fundamental problem in therapeutics design. In addition to the binding toward the target antigen, the antibodies of clinical interest should have favorable biophysical (i.e., developability) properties. The simulation of antigen-antibody binding affinity is a complex process that requires a model to generate a structure of antibody and antigen from their respective sequences and then simulate the binding affinity. We view this intricate process as a black-box oracle that takes a pair of antibody and antigen sequences as an input and returns their binding affinity. The design process to find an optimal antibody sequence is essentially a combinatorial search in the antibody sequence space to find an instance that maximizes the target function of a black oracle (e.g., a function that determines binding affinity). The combinatorial nature of the antibody sequence space makes it impossible to query the oracle function exhaustively, both computationally and experimentally. Therefore, we need a computationally efficient mechanism to search for an antibody sequence that maximizes the oracle’s output to achieve strong affinity with an antigen and has desired biophysical properties.

Khan et al. develop AntBO, a combinatorial Bayesian optimization tool for antibody design. AntBO provides a sample-efficient strategy for an in silico design of antibodies with a diverse range of favorable developability properties on various antigens of interest, including a lethal SARS-CoV pathogen.

[end paper 68]

[begin paper 69]

PMCID = 11042291

Title = PAbFold: Linear Antibody Epitope Prediction using AlphaFold2

Defining the binding epitopes of antibodies is essential for understanding how they bind to their antigens and perform their molecular functions. However, while determining linear epitopes of monoclonal antibodies can be accomplished utilizing well-established empirical procedures, these approaches are generally labor- and time-intensive and costly. To take advantage of the recent advances in protein structure prediction algorithms available to the scientific community, we developed a calculation pipeline based on the localColabFold implementation of AlphaFold2 that can predict linear antibody epitopes by predicting the structure of the complex between antibody heavy and light chains and target peptide sequences derived from antigens. We found that this AlphaFold2 pipeline, which we call PAbFold, was able to accurately flag known epitope sequences for several well-known antibody targets (HA / Myc) when the target sequence was broken into small overlapping linear peptides and antibody complementarity determining regions (CDRs) were grafted onto several different antibody framework regions in the single-chain antibody fragment (scFv) format. To determine if this pipeline was able to identify the epitope of a novel antibody with no structural information publicly available, we determined the epitope of a novel anti-SARS-CoV-2 nucleocapsid targeted antibody using our method and then experimentally validated our computational results using peptide competition ELISA assays. These results indicate that the AlphaFold2-based PAbFold pipeline we developed is capable of accurately identifying linear antibody epitopes in a short time using just antibody and target protein sequences. This emergent capability of the method is sensitive to methodological details such as peptide length, AlphaFold2 neural network versions, and multiple-sequence alignment database. PAbFold is available at https://github.com/jbderoo/PAbFold.

[end paper 69]

[begin paper 70]

PMCID = 11195458

Title = Toward enhancement of antibody thermostability and affinity by computational design in the absence of antigen

Over the past two decades, therapeutic antibodies have emerged as a rapidly expanding domain within the field of biologics. In silico tools that can streamline the process of antibody discovery and optimization are critical to support a pipeline that is growing more numerous and complex every year. High-quality structural information remains critical for the antibody optimization process, but antibody-antigen complex structures are often unavailable and in silico antibody docking methods are still unreliable. In this study, DeepAb, a deep learning model for predicting antibody Fv structure directly from sequence, was used in conjunction with single-point experimental deep mutational scanning (DMS) enrichment data to design 200 potentially optimized variants of an anti-hen egg lysozyme (HEL) antibody. We sought to determine whether DeepAb-designed variants containing combinations of beneficial mutations from the DMS exhibit enhanced thermostability and whether this optimization affected their developability profile. The 200 variants were produced through a robust high-throughput method and tested for thermal and colloidal stability (Tonset, Tm, Tagg), affinity (KD) relative to the parental antibody, and for developability parameters (nonspecific binding, aggregation propensity, self-association). Of the designed clones, 91% and 94% exhibited increased thermal and colloidal stability and affinity, respectively. Of these, 10% showed a significantly increased affinity for HEL (5- to 21-fold increase) and thermostability (>2.5C increase in Tm1), with most clones retaining the favorable developability profile of the parental antibody. Additional in silico tests suggest that these methods would enrich for binding affinity even without first collecting experimental DMS measurements. These data open the possibility of in silico antibody optimization without the need to predict the antibody–antigen interface, which is notoriously difficult in the absence of crystal structures.

[end paper 70]

[begin paper 71]

PMCID = 10792694

Title = Systematic investigation of machine learning on limited data: A study on predicting protein-protein binding strength

The application of machine learning techniques in biological research, especially when dealing with limited data availability, poses significant challenges. In this study, we leveraged advancements in method development for predicting protein-protein binding strength to conduct a systematic investigation into the application of machine learning on limited data. The binding strength, quantitatively measured as binding affinity, is vital for understanding the processes of recognition, association, and dysfunction that occur within protein complexes. By incorporating transfer learning, integrating domain knowledge, and employing both deep learning and traditional machine learning algorithms, we mitigated the impact of data limitations and made significant advancements in predicting protein-protein binding affinity. In particular, we developed over 20 models, ultimately selecting three representative best-performing ones that belong to distinct categories. The first model is structure-based, consisting of a random forest regression and thirteen handcrafted features. The second model is sequence-based, employing an architecture that combines transferred embedding features with a multilayer perceptron. Finally, we created an ensemble model by averaging the predictions of the two aforementioned models. The comparison with other predictors on three independent datasets confirms the significant improvements achieved by our models in predicting protein-protein binding affinity. The programs for running these three models are available at https://github.com/minghuilab/BindPPI.

[end paper 71]

[begin paper 72]

PMCID = 10776854

Title = Doa10/MARCH6 architecture interconnects E3 ligase activity with lipid-binding transmembrane channel to regulate SQLE

Transmembrane E3 ligases play crucial roles in homeostasis. Much protein and organelle quality control, and metabolic regulation, are determined by ER-resident MARCH6 E3 ligases, including Doa10 in yeast. Here, we present Doa10/MARCH6 structural analysis by cryo-EM and AlphaFold predictions, and a structure-based mutagenesis campaign. The majority of Doa10/MARCH6 adopts a unique circular structure within the membrane. This channel is established by a lipid-binding scaffold, and gated by a flexible helical bundle. The ubiquitylation active site is positioned over the channel by connections between the cytosolic E3 ligase RING domain and the membrane-spanning scaffold and gate. Here, by assaying 95 MARCH6 variants for effects on stability of the well-characterized substrate SQLE, which regulates cholesterol levels, we reveal crucial roles of the gated channel and RING domain consistent with AlphaFold-models of substrate-engaged and ubiquitylation complexes. SQLE degradation further depends on connections between the channel and RING domain, and lipid binding sites, revealing how interconnected Doa10/MARCH6 elements could orchestrate metabolic signals, substrate binding, and E3 ligase activity.

Transmembrane E3 ligases are crucial in cellular homeostasis and metabolic regulation. Here, the authors provide the structural details of the ER-resident E3 ligase MARCH6/Doa10, uncovering its unique circular membrane structure and its role in ubiquitylation processes, essential for protein quality control.

[end paper 72]

[begin paper 73]

PMCID = 9936529

Title = Development of a Syrian hamster anti-PD-L1 monoclonal antibody enables oncolytic adenoviral immunotherapy modelling in an immunocompetent virus replication permissive setting

Immune checkpoint inhibitors (ICIs) have revolutionized the treatment of cancer, but preclinical testing of hypotheses such as combination therapies has been complicated, in part due to species incompatibility issues. For example, one of few known permissive animal models for oncolytic adenoviruses is the Syrian hamster, for which an ICI, mainly an anti-PD-L1 monoclonal antibody (mAb) was not previously available. In this study, we developed an anti-Syrian hamster PD-L1 mAb to enable the evaluation of safety and efficacy, when combining anti-PD-L1 with an oncolytic adenovirus encoding tumour necrosis factor alpha (TNFα) and interleukin-2 (IL-2) (Ad5/3-E2F-D24-hTNFα-IRES-hIL-2 or TILT-123).

Recombinant Syrian hamster PD-L1 was expressed and mice immunized for mAb formation using hybridoma technology. Clonal selection through binding and functional studies in vitro, in silico and in vivo identified anti-PD-L1 clone 11B12-1 as the primary mAb candidate for immunotherapy modelling. The oncolytic virus (OV) and ICI combination approach was then evaluated using 11B12-1 and TILT-123 in a Syrian hamster model of pancreatic ductal adenocarcinoma (PDAC).

Supernatants from hybridoma parent subclone 11B12B4 provided the highest positive PD-L1 signal, on Syrian hamster PBMCs and three cancer cell lines (HT100, HapT1 and HCPC1). In vitro co-cultures revealed superior immune modulated profiles of cell line matched HT100 tumour infiltrating lymphocytes when using subclones of 7G2, 11B12 and 12F1. Epitope binning and epitope prediction using AlphaFold2 and ColabFold revealed two distinct functional epitopes for clone 11B12-1 and 12F1-1. Treatment of Syrian hamsters bearing HapT1 tumours, with 11B12-1 induced significantly better (p<0.05) tumour growth control than isotype control by day 12. 12F1-1 did not induce significant tumour growth control. The combination of 11B12-1 with oncolytic adenovirus TILT-123 improved tumour growth control further, when compared to monotherapy (p<0.05) by day 26.

Novel Syrian hamster anti-PD-L1 clone 11B12-1 induces tumour growth control in a hamster model of PDAC. Combining 11B12-1 with oncolytic adenovirus TILT-123 improves tumour growth control further and demonstrates good safety and toxicity profiles.

[end paper 73]

[begin paper 74]

PMCID = 11788754

Title = An outlook on structural biology after AlphaFold: tools, limits and perspectives

AlphaFold and similar groundbreaking, AI‐based tools, have revolutionized the field of structural bioinformatics, with their remarkable accuracy in ab‐initio protein structure prediction. This success has catalyzed the development of new software and pipelines aimed at incorporating AlphaFold's predictions, often focusing on addressing the algorithm's remaining challenges. Here, we present the current landscape of structural bioinformatics shaped by AlphaFold, and discuss how the field is dynamically responding to this revolution, with new software, methods, and pipelines. While the excitement around AI‐based tools led to their widespread application, it is essential to acknowledge that their practical success hinges on their integration into established protocols within structural bioinformatics, often neglected in the context of AI‐driven advancements. Indeed, user‐driven intervention is still as pivotal in the structure prediction process as in complementing state‐of‐the‐art algorithms with functional and biological knowledge.

Rendering of a GFP‐like fluorescent protein, starting from its AF‐generated PDB coordinates. Like AI algorithms for image generation focus on creating realistic and detailed content, AI‐based algorithms for protein structure generation aim to accurately generate/predict complex molecular arrangements. Figure generated using CGDream (https://cgdream.ai). 

[end paper 74]

[begin paper 75]

PMCID = 11763998

Title = Proximity Labeling: Precise Proteomics Technology for Mapping Receptor Protein Neighborhoods at the Cancer Cell Surface

Cell surface proteins and receptors on cancer cells, important as disease markers and drug targets, are highly dynamic, with their biological functions controlled by constantly changing protein–protein interaction networks. Studying these transient macromolecular neighborhoods is technically challenging, and traditional methods often fall short. Proximity labeling, a cutting-edge proteomics technology, offers a more precise approach that overcomes the limitations of existing methods. This review highlights the strategies, applications, benefits, and limitations of different proximity labeling platforms used to map the receptor protein microenvironment on the surface of cancer cells.

Cell surface receptors are pivotal to cancer cell transformation, disease progression, metastasis, early detection, targeted therapy, drug responses, and clinical outcomes. Since they coordinate complex signaling communication networks in the tumor microenvironment, mapping the physical interaction partners of cell surface receptors in vivo is vital for understanding their roles, functional states, and suitability as therapeutic targets. Yet traditional methods like immunoprecipitation and affinity purification–mass spectrometry often fail to detect key but weak or transient receptor–protein interactions. Proximity labeling, a cutting-edge proteomics technology, addresses these technical challenges by enabling precise mapping of protein neighborhoods around a receptor target on the cell surface of cancer cells. This technique has been successfully applied in vitro and in vivo for proteomic mapping across various model systems. This review explores the fundamental principles, technologies, advantages, limitations, and applications of proximity labeling in cancer biology, focusing on mapping receptor microenvironments. By advancing mechanistic insights into cancer cell receptor signaling mechanisms, proximity labeling is poised to transform cancer research, improve targeted therapies, and illuminate avenues to overcome drug resistance.

[end paper 75]

[begin paper 76]

PMCID = 11118328

Title = Disulfi de constrained Fabs overcome target size limitation for high-resolution single-particle cryo-EM

High-resolution structures of proteins are critical to understanding molecular mechanisms of biological processes and in the discovery of therapeutic molecules. Cryo-EM has revolutionized structure determination of large proteins and their complexes, but a vast majority of proteins that underlie human diseases are small (< 50 kDa) and usually beyond its reach due to low signal-to-noise images and difficulties in particle alignment. Current strategies to overcome this problem increase the overall size of small protein targets using scaffold proteins that bind to the target, but are limited by inherent flexibility and not being bound to their targets in a rigid manner, resulting in the target being poorly resolved compared to the scaffolds. Here we present an iteratively engineered molecular design for transforming Fabs (antibody fragments), into conformationally rigid scaffolds (Rigid-Fabs) that, when bound to small proteins (~20 kDa), can enable high-resolution structure determination using cryo-EM. This design introduces multiple disulfide bonds at strategic locations, generates a well-folded Fab constrained into a rigid conformation and can be applied to Fabs from various species, isotypes and chimeric Fabs. We present examples of the Rigid Fab design enabling high-resolution (2.3–2.5 Å) structures of small proteins, Ang2 (26 kDa) and KRAS (21 kDa) by cryo-EM. The strategies for designing disulfide constrained Rigid Fabs in our work thus establish a general approach to overcome the target size limitation of single particle cryo-EM.

[end paper 76]

[begin paper 77]

PMCID = 9912273

Title = Advancing Targeted Protein Degradation via Multiomics Profiling and Artificial Intelligence

Only around 20% of the human proteome is considered to be druggable with small-molecule antagonists. This leaves some of the most compelling therapeutic targets outside the reach of ligand discovery. The concept of targeted protein degradation (TPD) promises to overcome some of these limitations. In brief, TPD is dependent on small molecules that induce the proximity between a protein of interest (POI) and an E3 ubiquitin ligase, causing ubiquitination and degradation of the POI. In this perspective, we want to reflect on current challenges in the field, and discuss how advances in multiomics profiling, artificial intelligence, and machine learning (AI/ML) will be vital in overcoming them. The presented roadmap is discussed in the context of small-molecule degraders but is equally applicable for other emerging proximity-inducing modalities.

[end paper 77]

[begin paper 78]

PMCID = 10484509

Title = Exploring rigid-backbone protein docking in biologics discovery: a test using the DARPin scaffold

Accurate protein-protein docking remains challenging, especially for artificial biologics not coevolved naturally against their protein targets, like antibodies and other engineered scaffolds. We previously developed ProPOSE, an exhaustive docker with full atomistic details, which delivers cutting-edge performance by allowing side-chain rearrangements upon docking. However, extensive protein backbone flexibility limits its practical applicability as indicated by unbound docking tests. To explore the usefulness of ProPOSE on systems with limited backbone flexibility, here we tested the engineered scaffold DARPin, which is characterized by its relatively rigid protein backbone. A prospective screening campaign was undertaken, in which sequence-diversified DARPins were docked and ranked against a directed epitope on the target protein BCL-W. In this proof-of-concept study, only a relatively small set of 2,213 diverse DARPin interfaces were selected for docking from the huge theoretical library from mutating 18 amino-acid positions. A computational selection protocol was then applied for enrichment of binders based on normalized computed binding scores and frequency of binding modes against the predefined epitope. The top-ranked 18 designed DARPin interfaces were selected for experimental validation. Three designs exhibited binding affinities to BCL-W in the nanomolar range comparable to control interfaces adopted from known DARPin binders. This result is encouraging for future screening and engineering campaigns of DARPins and possibly other similarly rigid scaffolds against targeted protein epitopes. Method limitations are discussed and directions for future refinements are proposed.

[end paper 78]

[begin paper 79]

PMCID = 10620067

Title = The current role and evolution of X-ray crystallography in drug discovery and development

Macromolecular X-ray crystallography and cryo-EM are currently the primary techniques used to determine the three-dimensional structures of proteins, nucleic acids, and viruses. Structural information has been critical to drug discovery and structural bioinformatics. The integration of artificial intelligence (AI) into X-ray crystallography has shown great promise in automating and accelerating the analysis of complex structural data, further improving the efficiency and accuracy of structure determination.

This review explores the relationship between X-ray crystallography and other modern structural determination methods. It examines the integration of data acquired from diverse biochemical and biophysical techniques with those derived from structural biology. Additionally, the paper offers insights into the influence of AI on X-ray crystallography, emphasizing how integrating AI with experimental approaches can revolutionize our comprehension of biological processes and interactions.

Investing in science is crucially emphasized due to its significant role in drug discovery and advancements in healthcare. X-ray crystallography remains an essential source of structural biology data for drug discovery. Recent advances in biochemical, spectroscopic, and bioinformatic methods, along with the integration of AI techniques, hold the potential to revolutionize drug discovery when effectively combined with robust data management practices.

[end paper 79]

[begin paper 80]

PMCID = 10774428

Title = Contextualising the developability risk of antibodies with lambda light chains using enhanced therapeutic antibody profiling

Antibodies with lambda light chains (λ-antibodies) are generally considered to be less developable than those with kappa light chains (κ-antibodies). Though this hypothesis has not been formally established, it has led to substantial systematic biases in drug discovery pipelines and thus contributed to kappa dominance amongst clinical-stage therapeutics. However, the identification of increasing numbers of epitopes preferentially engaged by λ-antibodies shows there is a functional cost to neglecting to consider them as potential lead candidates. Here, we update our Therapeutic Antibody Profiler (TAP) tool to use the latest data and machine learning-based structure prediction, and apply it to evaluate developability risk profiles for κ-antibodies and λ-antibodies based on their surface physicochemical properties. We find that while human λ-antibodies on average have a higher risk of developability issues than κ-antibodies, a sizeable proportion are assigned lower-risk profiles by TAP and should represent more tractable candidates for therapeutic development. Through a comparative analysis of the low- and high-risk populations, we highlight opportunities for strategic design that TAP suggests would enrich for more developable λ-antibodies. Overall, we provide context to the differing developability of κ- and λ-antibodies, enabling a rational approach to incorporate more diversity into the initial pool of immunotherapeutic candidates.

A computational analysis of human kappa and lambda antibodies suggests that lambda antibodies are frequently less therapeutically developable due to their physicochemical properties, but that rational design strategies could reduce this risk.

[end paper 80]

[begin paper 81]

PMCID = 10589233

Title = A conformation-locking inhibitor of SLC15A4 with TASL proteostatic anti-inflammatory activity

Dysregulation of pathogen-recognition pathways of the innate immune system is associated with multiple autoimmune disorders. Due to the intricacies of the molecular network involved, the identification of pathway- and disease-specific therapeutics has been challenging. Using a phenotypic assay monitoring the degradation of the immune adapter TASL, we identify feeblin, a chemical entity which inhibits the nucleic acid-sensing TLR7/8 pathway activating IRF5 by disrupting the SLC15A4-TASL adapter module. A high-resolution cryo-EM structure of feeblin with SLC15A4 reveals that the inhibitor binds a lysosomal outward-open conformation incompatible with TASL binding on the cytoplasmic side, leading to degradation of TASL. This mechanism of action exploits a conformational switch and converts a target-binding event into proteostatic regulation of the effector protein TASL, interrupting the TLR7/8-IRF5 signaling pathway and preventing downstream proinflammatory responses. Considering that all components involved have been genetically associated with systemic lupus erythematosus and that feeblin blocks responses in disease-relevant human immune cells from patients, the study represents a proof-of-concept for the development of therapeutics against this disease.

The authors identify feeblin, an inhibitory compound of the proinflammatory TLR7/8/9-IRF5 pathway with therapeutic potential, which acts by binding SLC15A4 via an allosteric mechanism mediating degradation of its signaling partner TASL.

[end paper 81]

[begin paper 82]

PMCID = 11463455

Title = Unified Sampling and Ranking for Protein Docking with DFMDock

Diffusion models have shown promise in addressing the protein docking problem. Traditionally, these models are used solely for sampling docked poses, with a separate confidence model for ranking. We introduce DFMDock (Denoising Force Matching Dock), a diffusion model that unifies sampling and ranking within a single framework. DFMDock features two output heads: one for predicting forces and the other for predicting energies. The forces are trained using a denoising force matching objective, while the energy gradients are trained to align with the forces. This design enables our model to sample using the predicted forces and rank poses using the predicted energies, thereby eliminating the need for an additional confidence model. Our approach outperforms the previous diffusion model for protein docking, DiffDock-PP, with a sampling success rate of 44% compared to its 8%, and a Top- 1 ranking success rate of 16% compared to 0% on the Docking Benchmark 5.5 test set. In successful decoy cases, the DFMDock Energy forms a binding funnel similar to the physics-based Rosetta Energy, suggesting that DFMDock can capture the underlying energy landscape.

[end paper 82]

[begin paper 83]

PMCID = 9515332

Title = Reduced B cell antigenicity of Omicron lowers host serologic response

The SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. However, the mechanism remains unclear. Here, we find using a geometric deep-learning model that Omicron’s extensively mutated receptor binding site (RBS) features reduced antigenicity compared with previous variants. Mice immunization experiments with different recombinant receptor binding domain (RBD) variants confirm that the serological response to Omicron is drastically attenuated and less potent. Analyses of serum cross-reactivity and competitive ELISA reveal a reduction in antibody response across both variable and conserved RBD epitopes. Computational modeling confirms that the RBS has a potential for further antigenicity reduction while retaining efficient receptor binding. Finally, we find a similar trend of antigenicity reduction over decades for hCoV229E, a common cold coronavirus. Thus, our study explains the reduced antibody titers associated with Omicron infection and reveals a possible trajectory of future viral evolution.

SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. Tubiana et al. investigate the underlying mechanism using geometric deep learning, mice immunization experiments, and biochemical assays. Mutations reduce antigenicity of the receptor binding site, leading to lower antibody response.

[end paper 83]

[begin paper 84]

PMCID = 10957175

Title = Assembly of neuron- and radial glial-cell-derived extracellular matrix molecules promotes radial migration of developing cortical neurons

Radial neuronal migration is a key neurodevelopmental event for proper cortical laminar organization. The multipolar-to-bipolar transition, a critical step in establishing neuronal polarity during radial migration, occurs in the subplate/intermediate zone (SP/IZ), a distinct region of the embryonic cerebral cortex. It has been known that the extracellular matrix (ECM) molecules are enriched in the SP/IZ. However, the molecular constitution and functions of the ECM formed in this region remain poorly understood. Here, we identified neurocan (NCAN) as a major chondroitin sulfate proteoglycan in the mouse SP/IZ. NCAN binds to both radial glial-cell-derived tenascin-C (TNC) and hyaluronan (HA), a large linear polysaccharide, forming a ternary complex of NCAN, TNC, and HA in the SP/IZ. Developing cortical neurons make contact with the ternary complex during migration. The enzymatic or genetic disruption of the ternary complex impairs radial migration by suppressing the multipolar-to-bipolar transition. Furthermore, both TNC and NCAN promoted the morphological maturation of cortical neurons in vitro. The present results provide evidence for the cooperative role of neuron- and radial glial-cell-derived ECM molecules in cortical development.

[end paper 84]

[begin paper 85]

PMCID = 11474105

Title = ABodyBuilder3: improved and scalable antibody structure predictions

In this article, we introduce ABodyBuilder3, an improved and scalable antibody structure prediction model based on ABodyBuilder2. We achieve a new state-of-the-art accuracy in the modelling of CDR loops by leveraging language model embeddings, and show how predicted structures can be further improved through careful relaxation strategies. Finally, we incorporate a predicted Local Distance Difference Test into the model output to allow for a more accurate estimation of uncertainties.

The software package is available at https://github.com/Exscientia/ABodyBuilder3 with model weights and data at https://zenodo.org/records/11354577.

[end paper 85]

[begin paper 86]

PMCID = 9631452

Title = Advances in antibody discovery from human BCR repertoires

Antibodies make up an important and growing class of compounds used for the diagnosis or treatment of disease. While traditional antibody discovery utilized immunization of animals to generate lead compounds, technological innovations have made it possible to search for antibodies targeting a given antigen within the repertoires of B cells in humans. Here we group these innovations into four broad categories: cell sorting allows the collection of cells enriched in specificity to one or more antigens; BCR sequencing can be performed on bulk mRNA, genomic DNA or on paired (heavy-light) mRNA; BCR repertoire analysis generally involves clustering BCRs into specificity groups or more in-depth modeling of antibody-antigen interactions, such as antibody-specific epitope predictions; validation of antibody-antigen interactions requires expression of antibodies, followed by antigen binding assays or epitope mapping. Together with innovations in Deep learning these technologies will contribute to the future discovery of diagnostic and therapeutic antibodies directly from humans.

[end paper 86]

[begin paper 87]

PMCID = 11578886

Title = MassiveFold: unveiling AlphaFold’s hidden potential with optimized and parallelized massive sampling

Massive sampling in AlphaFold enables access to increased structural diversity. In combination with its efficient confidence ranking, this unlocks elevated modeling capabilities for monomeric structures and foremost for protein assemblies. However, the approach struggles with GPU cost and data storage. Here we introduce MassiveFold, an optimized and customizable version of AlphaFold that runs predictions in parallel, reducing the computing time from several months to hours. MassiveFold is scalable and able to run on anything from a single computer to a large GPU infrastructure, where it can fully benefit from all the computing nodes.

Although AlphaFold is very efficient for protein structure prediction, massive sampling is a very GPU demanding task. MassiveFold overcomes this limitation, being capable of parallelizing structure prediction computation.

[end paper 87]

[begin paper 88]

PMCID = 11185555

Title = Distinct Perception Mechanisms of BACH1 Quaternary Structure Degrons by Two F-box Proteins under Oxidative Stress

The transcription factor BACH1 regulates heme homeostasis and oxidative stress responses and promotes cancer metastasis upon aberrant accumulation. Its stability is controlled by two F-box protein ubiquitin ligases, FBXO22 and FBXL17. Here we show that the homodimeric BTB domain of BACH1 functions as a previously undescribed quaternary structure degron, which is deciphered by the two F-box proteins via distinct mechanisms. After BACH1 is released from chromatin by heme, FBXO22 asymmetrically recognizes a cross-protomer interface of the intact BACH1 BTB dimer, which is otherwise masked by the co-repressor NCOR1. If the BACH1 BTB dimer escapes the surveillance by FBXO22 due to oxidative modifications, its quaternary structure integrity is probed by a pair of FBXL17, which simultaneously engage and remodel the two BTB protomers into E3-bound monomers for ubiquitination. By unveiling the multifaceted regulatory mechanisms of BACH1 stability, our studies highlight the abilities of ubiquitin ligases to decode high-order protein assemblies and reveal therapeutic opportunities to block cancer invasion via compound-induced BACH1 destabilization.

[end paper 88]

[begin paper 89]

PMCID = 11213319

Title = Zinc-finger (ZiF) fold secreted effectors form a functionally diverse family across lineages of the blast fungus Magnaporthe oryzae

Filamentous plant pathogens deliver effector proteins into host cells to suppress host defence responses and manipulate metabolic processes to support colonization. Understanding the evolution and molecular function of these effectors provides knowledge about pathogenesis and can suggest novel strategies to reduce damage caused by pathogens. However, effector proteins are highly variable, share weak sequence similarity and, although they can be grouped according to their structure, only a few structurally conserved effector families have been functionally characterized to date. Here, we demonstrate that Zinc-finger fold (ZiF) secreted proteins form a functionally diverse effector family in the blast fungus Magnaporthe oryzae. This family relies on the Zinc-finger motif for protein stability and is ubiquitously present in blast fungus lineages infecting 13 different host species, forming different effector tribes. Homologs of the canonical ZiF effector, AVR-Pii, from rice infecting isolates are present in multiple M. oryzae lineages. Wheat infecting strains of the fungus also possess an AVR-Pii like allele that binds host Exo70 proteins and activates the immune receptor Pii. Furthermore, ZiF tribes may vary in the proteins they bind to, indicating functional diversification and an intricate effector/host interactome. Altogether, we uncovered a new effector family with a common protein fold that has functionally diversified in lineages of M. oryzae. This work expands our understanding of the diversity of M. oryzae effectors, the molecular basis of plant pathogenesis and may ultimately facilitate the development of new sources for pathogen resistance.

Diseases caused by filamentous plant pathogens impact global food production, leading to severe economic and humanitarian consequences. These pathogens secrete hundreds of effectors inside the host to alter cellular processes and to promote infection and disease. Effector proteins have weak or no sequence similarity but can be grouped in structural families based on conserved protein folds. However, very few conserved effector families have been functionally characterized. We have identified a family of effectors with a shared Zinc-finger protein fold (ZiF) that is present in lineages of the blast fungus Magnaporthe oryzae that can, collectively, infect 13 different grasses. We characterized the binding of a sub-set of these proteins to putative Exo70 host targets and showed they can be recognized by the plant immune system. Furthermore, we show that other ZiF effectors do not bind Exo70 targets, suggesting functional specialization within this effector family for alternative interactors. These findings shed light on the diversity of effectors and their molecular functions, as well as potentially leading to the development of new sources of blast disease resistance.

[end paper 89]

[begin paper 90]

PMCID = 10561695

Title = Altering Specificity and Autoactivity of Plant Immune Receptors Sr33 and Sr50 Via a Rational Engineering Approach

Many resistance genes deployed against pathogens in crops are intracellular nucleotide-binding (NB) leucine-rich repeat (LRR) receptors (NLRs). The ability to rationally engineer the specificity of NLRs will be crucial in the response to newly emerging crop diseases. Successful attempts to modify NLR recognition have been limited to untargeted approaches or depended on previously available structural information or knowledge of pathogen-effector targets. However, this information is not available for most NLR-effector pairs. Here, we demonstrate the precise prediction and subsequent transfer of residues involved in effector recognition between two closely related NLRs without their experimentally determined structure or detailed knowledge about their pathogen effector targets. By combining phylogenetics, allele diversity analysis, and structural modeling, we successfully predicted residues mediating interaction of Sr50 with its cognate effector AvrSr50 and transferred recognition specificity of Sr50 to the closely related NLR Sr33. We created synthetic versions of Sr33 that contain amino acids from Sr50, including Sr33syn, which gained the ability to recognize AvrSr50 with 12 amino-acid substitutions. Furthermore, we discovered that sites in the LRR domain needed to transfer recognition specificity to Sr33 also influence autoactivity in Sr50. Structural modeling suggests these residues interact with a part of the NB-ARC domain, which we named the NB-ARC latch, to possibly maintain the inactive state of the receptor. Our approach demonstrates rational modifications of NLRs, which could be useful to enhance existing elite crop germplasm.

[end paper 90]

[begin paper 91]

PMCID = 10187307

Title = Molecular basis for Gβγ-mediated activation of phosphoinositide 3-kinase γ

The conversion of PIP2 to PIP3 by phosphoinositide 3-kinase γ (PI3Kγ) is a critical step in neutrophil chemotaxis and is essential for metastasis in many types of cancer. PI3Kγ is activated via directed interaction with Gβγ heterodimers released from cell-surface G protein-coupled receptors (GPCRs) responding to extracellular signals. To resolve how Gβγ activates PI3Kγ, we determined cryo-EM reconstructions of PI3Kγ–Gβγ complexes in the presence of various substrates/analogs, revealing two distinct Gβγ binding sites, one on the p110γ helical domain and one on the C-terminal domain of the p101 subunit. Comparison of these complexes with structures of PI3Kγ alone demonstrates conformational changes in the kinase domain upon Gβγ binding similar to those induced by Ras·GTP. Assays of variants perturbing the two Gβγ binding sites and interdomain contacts that change upon Gβγ binding suggest that Gβγ not only recruits the enzyme to membranes but also allosterically controls activity via both sites. Studies in a zebrafish model examining neutrophil migration are consistent with these results. These findings set the stage for future detailed investigation of Gβγ-mediated activation mechanisms in this enzyme family and will aid in developing drugs selective for PI3Kγ.

[end paper 91]

[begin paper 92]

PMCID = 9481589

Title = A multi-adenylate cyclase regulator at the flagellar tip controls African trypanosome transmission

Signaling from ciliary microdomains controls developmental processes in metazoans. Trypanosome transmission requires development and migration in the tsetse vector alimentary tract. Flagellar cAMP signaling has been linked to parasite social motility (SoMo) in vitro, yet uncovering control of directed migration in fly organs is challenging. Here we show that the composition of an adenylate cyclase (AC) complex in the flagellar tip microdomain is essential for tsetse salivary gland (SG) colonization and SoMo. Cyclic AMP response protein 3 (CARP3) binds and regulates multiple AC isoforms. CARP3 tip localization depends on the cytoskeletal protein FLAM8. Re-localization of CARP3 away from the tip microdomain is sufficient to abolish SoMo and fly SG colonization. Since intrinsic development is normal in carp3 and flam8 knock-out parasites, AC complex-mediated tip signaling specifically controls parasite migration and thereby transmission. Participation of several developmentally regulated receptor-type AC isoforms may indicate the complexity of the in vivo signals perceived.

Trypanosomes can sense signal molecules and coordinate their movement in response to such signals, a phenomenon termed social motility (SoMo). Here, Bachmaier et al show that cyclic AMP response protein 3 (CARP3) localization to the flagellar tip and its interaction with a number of different adenylate cyclases is essential for migration to tsetse fly salivary glands and for SoMo, therewith linking SoMo and cAMP signaling to trypanosome transmission.

[end paper 92]



