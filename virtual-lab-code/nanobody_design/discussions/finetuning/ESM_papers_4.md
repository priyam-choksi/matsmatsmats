## User

This is the beginning of an individual meeting with Machine Learning Specialist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "ESM" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on ESM to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on ESM using the query "Machine learning ESM nanobody design SARS-CoV-2" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to ESM. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Machine Learning Specialist, please provide your response to the agenda.

## Machine Learning Specialist

Based on the abstracts provided, here are the papers that are directly relevant to ESM (Evolutionary Scale Modeling) in the context of designing nanobody binders for SARS-CoV-2, and which I would like to fine-tune myself on:

```python
{
    "10220908": "Structural Modeling of Nanobodies: A Benchmark of State-of-the-Art Artificial Intelligence Programs",
    "11128484": "Accurate prediction of antibody function and structure using bio-inspired antibody language model",
    "11208048": "Accurate prediction of CDR-H3 loop structures of antibodies with deep learning",
    "10886287": "Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions",
    "10925456": "EvoAI enables extreme compression and reconstruction of the protein sequence space",
    "10698008": "Improving antibody optimization ability of generative adversarial network through large language model"
}
```

These papers focus on the use of machine learning models, including language models and deep learning techniques, for antibody and nanobody design, which are relevant to the application of ESM in this context.

## User

Tool Output:

Here are the top 19 articles on PubMed Central for the query "Machine learning ESM nanobody design SARS-CoV-2":

[begin paper 1]

PMCID = 10886287

Title = Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions

This perspective sheds light on the transformative impact of recent computational advancements in the field of protein therapeutics, with a particular focus on the design and development of antibodies. Cutting-edge computational methods have revolutionized our understanding of protein–protein interactions (PPIs), enhancing the efficacy of protein therapeutics in preclinical and clinical settings. Central to these advancements is the application of machine learning and deep learning, which offers unprecedented insights into the intricate mechanisms of PPIs and facilitates precise control over protein functions. Despite these advancements, the complex structural nuances of antibodies pose ongoing challenges in their design and optimization. Our review provides a comprehensive exploration of the latest deep learning approaches, including language models and diffusion techniques, and their role in surmounting these challenges. We also present a critical analysis of these methods, offering insights to drive further progress in this rapidly evolving field. The paper includes practical recommendations for the application of these computational techniques, supplemented with independent benchmark studies. These studies focus on key performance metrics such as accuracy and the ease of program execution, providing a valuable resource for researchers engaged in antibody design and development. Through this detailed perspective, we aim to contribute to the advancement of antibody design, equipping researchers with the tools and knowledge to navigate the complexities of this field.

[end paper 1]

[begin paper 2]

PMCID = 9720949

Title = Prediction of polyreactive and nonspecific single-chain fragment variables through structural biochemical features and protein language-based descriptors

Monoclonal antibodies (mAbs) have been used as therapeutic agents, which must overcome many developability issues after the discovery from in vitro display libraries. Especially, polyreactive mAbs can strongly bind to a specific target and weakly bind to off-target proteins, which leads to poor antibody pharmacokinetics in clinical development. Although early assessment of polyreactive mAbs is important in the early discovery stage, experimental assessments are usually time-consuming and expensive. Therefore, computational approaches for predicting the polyreactivity of single-chain fragment variables (scFvs) in the early discovery stage would be promising for reducing experimental efforts. 

Here, we made prediction models for the polyreactivity of scFvs with the known polyreactive antibody features and natural language model descriptors. We predicted 19,426 protein structures of scFvs with trRosetta to calculate the polyreactive antibody features and investigated the classifying performance of each factor for polyreactivity. In the known polyreactive features, the net charge of the CDR2 loop, the tryptophan and glycine residues in CDR-H3, and the lengths of the CDR1 and CDR2 loops, importantly contributed to the performance of the models. Additionally, the hydrodynamic features, such as partial specific volume, gyration radius, and isoelectric points of CDR loops and scFvs, were newly added to improve model performance. Finally, we made the prediction model with a robust performance () with an ensemble learning of the top 3 best models.

The prediction models for polyreactivity would help assess polyreactive scFvs in the early discovery stage and our approaches would be promising to develop machine learning models with quantitative data from high throughput assays for antibody screening. 

The online version contains supplementary material available at 10.1186/s12859-022-05010-4.

[end paper 2]

[begin paper 3]

PMCID = 9294429

Title = Machine-designed biotherapeutics: opportunities, feasibility and advantages of deep learning in computational antibody discovery

Antibodies are versatile molecular binders with an established and growing role as therapeutics. Computational approaches to developing and designing these molecules are being increasingly used to complement traditional lab-based processes. Nowadays, in silico methods fill multiple elements of the discovery stage, such as characterizing antibody–antigen interactions and identifying developability liabilities. Recently, computational methods tackling such problems have begun to follow machine learning paradigms, in many cases deep learning specifically. This paradigm shift offers improvements in established areas such as structure or binding prediction and opens up new possibilities such as language-based modeling of antibody repertoires or machine-learning-based generation of novel sequences. In this review, we critically examine the recent developments in (deep) machine learning approaches to therapeutic antibody design with implications for fully computational antibody design.

[end paper 3]

[begin paper 4]

PMCID = 11221890

Title = Antibody design using deep learning: from sequence and structure design to affinity maturation

Deep learning has achieved impressive results in various fields such as computer vision and natural language processing, making it a powerful tool in biology. Its applications now encompass cellular image classification, genomic studies and drug discovery. While drug development traditionally focused deep learning applications on small molecules, recent innovations have incorporated it in the discovery and development of biological molecules, particularly antibodies. Researchers have devised novel techniques to streamline antibody development, combining in vitro and in silico methods. In particular, computational power expedites lead candidate generation, scaling and potential antibody development against complex antigens. This survey highlights significant advancements in protein design and optimization, specifically focusing on antibodies. This includes various aspects such as design, folding, antibody–antigen interactions docking and affinity maturation.

[end paper 4]

[begin paper 5]

PMCID = 11245638

Title = ISEV2024 Abstract Book



[end paper 5]

[begin paper 6]

PMCID = 11788748

Title = Structure‐based computational design of antibody mimetics: challenges and perspectives

The design of antibody mimetics holds great promise for revolutionizing therapeutic interventions by offering alternatives to conventional antibody therapies. Structure‐based computational approaches have emerged as indispensable tools in the rational design of those molecules, enabling the precise manipulation of their structural and functional properties. This review covers the main classes of designed antigen‐binding motifs, as well as alternative strategies to develop tailored ones. We discuss the intricacies of different computational protein–protein interaction design strategies, showcased by selected successful cases in the literature. Subsequently, we explore the latest advancements in the computational techniques including the integration of machine and deep learning methodologies into the design framework, which has led to an augmented design pipeline. Finally, we verse onto the current challenges that stand in the way between high‐throughput computer design of antibody mimetics and experimental realization, offering a forward‐looking perspective into the field and the promises it holds to biotechnology.

Computer graphics representation of an artificial intelligence–driven protein design pipeline, where building blocks are fed into the birth of a novel protein structure. 

[end paper 6]

[begin paper 7]

PMCID = 11128484

Title = Accurate prediction of antibody function and structure using bio-inspired antibody language model

In recent decades, antibodies have emerged as indispensable therapeutics for combating diseases, particularly viral infections. However, their development has been hindered by limited structural information and labor-intensive engineering processes. Fortunately, significant advancements in deep learning methods have facilitated the precise prediction of protein structure and function by leveraging co-evolution information from homologous proteins. Despite these advances, predicting the conformation of antibodies remains challenging due to their unique evolution and the high flexibility of their antigen-binding regions. Here, to address this challenge, we present the Bio-inspired Antibody Language Model (BALM). This model is trained on a vast dataset comprising 336 million 40% nonredundant unlabeled antibody sequences, capturing both unique and conserved properties specific to antibodies. Notably, BALM showcases exceptional performance across four antigen-binding prediction tasks. Moreover, we introduce BALMFold, an end-to-end method derived from BALM, capable of swiftly predicting full atomic antibody structures from individual sequences. Remarkably, BALMFold outperforms those well-established methods like AlphaFold2, IgFold, ESMFold and OmegaFold in the antibody benchmark, demonstrating significant potential to advance innovative engineering and streamline therapeutic antibody development by reducing the need for unnecessary trials. The BALMFold structure prediction server is freely available at https://beamlab-sh.com/models/BALMFold.

[end paper 7]

[begin paper 8]

PMCID = 10925456

Title = EvoAI enables extreme compression and reconstruction of the protein sequence space

Designing proteins with improved functions requires a deep understanding of how sequence and function are related, a vast space that is hard to explore. The ability to efficiently compress this space by identifying functionally important features is extremely valuable. Here, we first establish a method called EvoScan to comprehensively segment and scan the high-fitness sequence space to obtain anchor points that capture its essential features, especially in high dimensions. Our approach is compatible with any biomolecular function that can be coupled to a transcriptional output. We then develop deep learning and large language models to accurately reconstruct the space from these anchors, allowing computational prediction of novel, highly fit sequences without prior homology-derived or structural information. We apply this hybrid experimental-computational method, which we call EvoAI, to a repressor protein and find that only 82 anchors are sufficient to compress the high-fitness sequence space with a compression ratio of 1048. The extreme compressibility of the space informs both applied biomolecular design and understanding of natural evolution.

[end paper 8]

[begin paper 9]

PMCID = 9631452

Title = Advances in antibody discovery from human BCR repertoires

Antibodies make up an important and growing class of compounds used for the diagnosis or treatment of disease. While traditional antibody discovery utilized immunization of animals to generate lead compounds, technological innovations have made it possible to search for antibodies targeting a given antigen within the repertoires of B cells in humans. Here we group these innovations into four broad categories: cell sorting allows the collection of cells enriched in specificity to one or more antigens; BCR sequencing can be performed on bulk mRNA, genomic DNA or on paired (heavy-light) mRNA; BCR repertoire analysis generally involves clustering BCRs into specificity groups or more in-depth modeling of antibody-antigen interactions, such as antibody-specific epitope predictions; validation of antibody-antigen interactions requires expression of antibodies, followed by antigen binding assays or epitope mapping. Together with innovations in Deep learning these technologies will contribute to the future discovery of diagnostic and therapeutic antibodies directly from humans.

[end paper 9]

[begin paper 10]

PMCID = 11254834

Title = A comprehensive overview of recent advances in generative models for antibodies

Therapeutic antibodies are an important class of biopharmaceuticals. With the rapid development of deep learning methods and the increasing amount of antibody data, antibody generative models have made great progress recently. They aim to solve the antibody space searching problems and are widely incorporated into the antibody development process. Therefore, a comprehensive introduction to the development methods in this field is imperative. Here, we collected 34 representative antibody generative models published recently and all generative models can be divided into three categories: sequence-generating models, structure-generating models, and hybrid models, based on their principles and algorithms. We further studied their performance and contributions to antibody sequence prediction, structure optimization, and affinity enhancement. Our manuscript will provide a comprehensive overview of the status of antibody generative models and also offer guidance for selecting different approaches.

[end paper 10]

[begin paper 11]

PMCID = 10589401

Title = Generative models for protein sequence modeling: recent advances and future directions

The widespread adoption of high-throughput omics technologies has exponentially increased the amount of protein sequence data involved in many salient disease pathways and their respective therapeutics and diagnostics. Despite the availability of large-scale sequence data, the lack of experimental fitness annotations underpins the need for self-supervised and unsupervised machine learning (ML) methods. These techniques leverage the meaningful features encoded in abundant unlabeled sequences to accomplish complex protein engineering tasks. Proficiency in the rapidly evolving fields of protein engineering and generative AI is required to realize the full potential of ML models as a tool for protein fitness landscape navigation. Here, we support this work by (i) providing an overview of the architecture and mathematical details of the most successful ML models applicable to sequence data (e.g. variational autoencoders, autoregressive models, generative adversarial neural networks, and diffusion models), (ii) guiding how to effectively implement these models on protein sequence data to predict fitness or generate high-fitness sequences and (iii) highlighting several successful studies that implement these techniques in protein engineering (from paratope regions and subcellular localization prediction to high-fitness sequences and protein design rules generation). By providing a comprehensive survey of model details, novel architecture developments, comparisons of model applications, and current challenges, this study intends to provide structured guidance and robust framework for delivering a prospective outlook in the ML-driven protein engineering field.

[end paper 11]

[begin paper 12]

PMCID = 10220908

Title = Structural Modeling of Nanobodies: A Benchmark of State-of-the-Art Artificial Intelligence Programs

The number of applications for nanobodies is steadily expanding, positioning these molecules as fast-growing biologic products in the biotechnology market. Several of their applications require protein engineering, which in turn would greatly benefit from having a reliable structural model of the nanobody of interest. However, as with antibodies, the structural modeling of nanobodies is still a challenge. With the rise of artificial intelligence (AI), several methods have been developed in recent years that attempt to solve the problem of protein modeling. In this study, we have compared the performance in nanobody modeling of several state-of-the-art AI-based programs, either designed for general protein modeling, such as AlphaFold2, OmegaFold, ESMFold, and Yang-Server, or specifically designed for antibody modeling, such as IgFold, and Nanonet. While all these programs performed rather well in constructing the nanobody framework and CDRs 1 and 2, modeling CDR3 still represents a big challenge. Interestingly, tailoring an AI method for antibody modeling does not necessarily translate into better results for nanobodies.

[end paper 12]

[begin paper 13]

PMCID = 10236260

Title = The International BNA 2023 Festival of Neuroscience



[end paper 13]

[begin paper 14]

PMCID = 10698008

Title = Improving antibody optimization ability of generative adversarial network through large language model

Generative adversarial networks (GANs) have successfully generated functional protein sequences. However, traditional GANs often suffer from inherent randomness, resulting in a lower probability of obtaining desirable sequences. Due to the high cost of wet-lab experiments, the main goal of computer-aided antibody optimization is to identify high-quality candidate antibodies from a large range of possibilities, yet improving the ability of GANs to generate these desired antibodies is a challenge. In this study, we propose and evaluate a new GAN called the Language Model Guided Antibody Generative Adversarial Network (AbGAN-LMG). This GAN uses a language model as an input, harnessing such models’ powerful representational capabilities to improve the GAN’s generation of high-quality antibodies. We conducted a comprehensive evaluation of the antibody libraries and sequences generated by AbGAN-LMG for COVID-19 (SARS-CoV-2) and Middle East Respiratory Syndrome (MERS-CoV). Results indicate that AbGAN-LMG has learned the fundamental characteristics of antibodies and that it improved the diversity of the generated libraries. Additionally, when generating sequences using AZD-8895 as the target antibody for optimization, over 50% of the generated sequences exhibited better developability than AZD-8895 itself. Through molecular docking, we identified 70 antibodies that demonstrated higher affinity for the wild-type receptor-binding domain (RBD) of SARS-CoV-2 compared to AZD-8895. In conclusion, AbGAN-LMG demonstrates that language models used in conjunction with GANs can enable the generation of higher-quality libraries and candidate sequences, thereby improving the efficiency of antibody optimization. AbGAN-LMG is available at http://39.102.71.224:88/.

[end paper 14]

[begin paper 15]

PMCID = 10443277

Title = Recent Progress in Antibody Epitope Prediction

Recent progress in epitope prediction has shown promising results in the development of vaccines and therapeutics against various diseases. However, the overall accuracy and success rate need to be improved greatly to gain practical application significance, especially conformational epitope prediction. In this review, we examined the general features of antibody–antigen recognition, highlighting the conformation selection mechanism in flexible antibody–antigen binding. We recently highlighted the success and warning signs of antibody epitope predictions, including linear and conformation epitope predictions. While deep learning-based models gradually outperform traditional feature-based machine learning, sequence and structure features still provide insight into antibody–antigen recognition problems.

[end paper 15]

[begin paper 16]

PMCID = 11745815

Title = Recent Advances in Mass Spectrometry-Based Protein Interactome Studies

The foundation of all biological processes is the network of diverse and dynamic protein interactions with other molecules in cells known as the interactome. Understanding the interactome is crucial for elucidating molecular mechanisms but has been a longstanding challenge. Recent developments in mass spectrometry (MS)-based techniques, including affinity purification, proximity labeling, cross-linking, and co-fractionation mass spectrometry (MS), have significantly enhanced our abilities to study the interactome. They do so by identifying and quantifying protein interactions yielding profound insights into protein organizations and functions. This review summarizes recent advances in MS-based interactomics, focusing on the development of techniques that capture protein-protein, protein-metabolite, and protein-nucleic acid interactions. Additionally, we discuss how integrated MS-based approaches have been applied to diverse biological samples, focusing on significant discoveries that have leveraged our understanding of cellular functions. Finally, we highlight state-of-the-art bioinformatic approaches for predictions of interactome and complex modeling, as well as strategies for combining experimental interactome data with computation methods, thereby enhancing the ability of MS-based techniques to identify protein interactomes. Indeed, advances in MS technologies and their integrations with computational biology provide new directions and avenues for interactome research, leveraging new insights into mechanisms that govern the molecular architecture of living cells and, thereby, our comprehension of biological processes.

Recent advances in MS-based techniques for comprehensive protein interactome mapping.

Integration of experimental MS methods with cutting-edge computational tools.

Emerging strategies for high-resolution interactome analysis in various biological contexts.

This review highlights recent advancements in mass spectrometry-based techniques for mapping protein interactomes, including affinity purification, proximity labeling, cross-linking, and co-fractionation approaches. It discusses the integration of these methods with cutting-edge computational tools, emphasizing their synergistic potential in elucidating complex cellular networks. The review explores how these technologies are revolutionizing our understanding of protein interactions, providing unprecedented insights into cellular processes and opening new avenues for biological research, drug discovery, and disease diagnostics.

[end paper 16]

[begin paper 17]

PMCID = 11471263

Title = VCAb: a web-tool for structure-guided exploration of antibodies

Effective responses against immune challenges require antibodies of different isotypes performing specific effector functions. Structural information on these isotypes is essential to engineer antibodies with desired physico-chemical features of their antigen-binding properties, and optimal developability as potential therapeutics. In silico mutational scanning profiles on antibody structures would further pinpoint candidate mutations for enhancing antibody stability and function. Current antibody structure databases lack consistent annotations of isotypes and structural coverage of 3D antibody structures, as well as computed deep mutation profiles.

The V and C region bearing antibody (VCAb) web-tool is established to clarify these annotations and provides an accessible resource to facilitate antibody engineering and design. VCAb currently provides data on 7,166 experimentally determined antibody structures including both V and C regions from different species. Additionally, VCAb provides annotations of species and isotypes with numbering schemes applied. These information can be interactively queried or downloaded in batch.

VCAb is implemented as a R shiny application to enable interactive data interrogation. The online application is freely accessible https://fraternalilab.cs.ucl.ac.uk/VCAb/. The source code to generate the database and the online application is available open-source at https://github.com/Fraternalilab/VCAb.

[end paper 17]

[begin paper 18]

PMCID = 10644518

Title = DG-Affinity: predicting antigen–antibody affinity with language models from sequences

Antibody-mediated immune responses play a crucial role in the immune defense of human body. The evolution of bioengineering has led the progress of antibody-derived drugs, showing promising efficacy in cancer and autoimmune disease therapy. A critical step of this development process is obtaining the affinity between antibodies and their binding antigens.

In this study, we introduce a novel sequence-based antigen–antibody affinity prediction method, named DG-Affinity. DG-Affinity uses deep neural networks to efficiently and accurately predict the affinity between antibodies and antigens from sequences, without the need for structural information. The sequences of both the antigen and the antibody are first transformed into embedding vectors by two pre-trained language models, then these embeddings are concatenated into an ConvNeXt framework with a regression task. The results demonstrate the superiority of DG-Affinity over the existing structure-based prediction methods and the sequence-based tools, achieving a Pearson’s correlation of over 0.65 on an independent test dataset.

Compared to the baseline methods, DG-Affinity achieves the best performance and can advance the development of antibody design. It is freely available as an easy-to-use web server at https://www.digitalgeneai.tech/solution/affinity.

The online version contains supplementary material available at 10.1186/s12859-023-05562-z.

[end paper 18]

[begin paper 19]

PMCID = 11208048

Title = Accurate prediction of CDR-H3 loop structures of antibodies with deep learning

Accurate prediction of the structurally diverse complementarity determining region heavy chain 3 (CDR-H3) loop structure remains a primary and long-standing challenge for antibody modeling. Here, we present the H3-OPT toolkit for predicting the 3D structures of monoclonal antibodies and nanobodies. H3-OPT combines the strengths of AlphaFold2 with a pre-trained protein language model and provides a 2.24 Å average RMSDCα between predicted and experimentally determined CDR-H3 loops, thus outperforming other current computational methods in our non-redundant high-quality dataset. The model was validated by experimentally solving three structures of anti-VEGF nanobodies predicted by H3-OPT. We examined the potential applications of H3-OPT through analyzing antibody surface properties and antibody–antigen interactions. This structural prediction tool can be used to optimize antibody–antigen binding and engineer therapeutic antibodies with biophysical properties for specialized drug administration route.

[end paper 19]



