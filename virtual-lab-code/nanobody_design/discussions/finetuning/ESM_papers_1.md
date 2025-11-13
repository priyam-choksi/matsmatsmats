## User

This is the beginning of an individual meeting with Machine Learning Specialist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "ESM" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on ESM to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on ESM using the query "Evolutionary Scale Modeling in protein design" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to ESM. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Machine Learning Specialist, please provide your response to the agenda.

## Machine Learning Specialist

Based on the abstracts provided, here are the papers that are directly relevant to Evolutionary Scale Modeling (ESM) and protein design, which I would like to fine-tune myself on:

```python
{
    "11036486": "Combining Rosetta Sequence Design with Protein Language Model Predictions Using Evolutionary Scale Modeling (ESM) as Restraint",
    "11275918": "Modeling Alternative Conformational States of Pseudo-Symmetric Solute Carrier Transporters using Methods from Deep Learning",
    "10942313": "An integrative approach to protein sequence design through multiobjective optimization"
}
```

These papers specifically mention the use of Evolutionary Scale Modeling (ESM) in the context of protein design, which is directly relevant to the task of designing SARS-CoV-2 nanobody binders.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "Evolutionary Scale Modeling in protein design":

[begin paper 1]

PMCID = 11477718

Title = Revolutionizing Molecular Design for Innovative Therapeutic Applications through Artificial Intelligence

The field of computational protein engineering has been transformed by recent advancements in machine learning, artificial intelligence, and molecular modeling, enabling the design of proteins with unprecedented precision and functionality. Computational methods now play a crucial role in enhancing the stability, activity, and specificity of proteins for diverse applications in biotechnology and medicine. Techniques such as deep learning, reinforcement learning, and transfer learning have dramatically improved protein structure prediction, optimization of binding affinities, and enzyme design. These innovations have streamlined the process of protein engineering by allowing the rapid generation of targeted libraries, reducing experimental sampling, and enabling the rational design of proteins with tailored properties. Furthermore, the integration of computational approaches with high-throughput experimental techniques has facilitated the development of multifunctional proteins and novel therapeutics. However, challenges remain in bridging the gap between computational predictions and experimental validation and in addressing ethical concerns related to AI-driven protein design. This review provides a comprehensive overview of the current state and future directions of computational methods in protein engineering, emphasizing their transformative potential in creating next-generation biologics and advancing synthetic biology.

[end paper 1]

[begin paper 2]

PMCID = 6436896

Title = Learning protein constitutive motifs from sequence data

Statistical analysis of evolutionary-related protein sequences provides information about their structure, function, and history. We show that Restricted Boltzmann Machines (RBM), designed to learn complex high-dimensional data and their statistical features, can efficiently model protein families from sequence information. We here apply RBM to 20 protein families, and present detailed results for two short protein domains (Kunitz and WW), one long chaperone protein (Hsp70), and synthetic lattice proteins for benchmarking. The features inferred by the RBM are biologically interpretable: they are related to structure (residue-residue tertiary contacts, extended secondary motifs (α-helixes and β-sheets) and intrinsically disordered regions), to function (activity and ligand specificity), or to phylogenetic identity. In addition, we use RBM to design new protein sequences with putative properties by composing and 'turning up' or 'turning down' the different modes at will. Our work therefore shows that RBM are versatile and practical tools that can be used to unveil and exploit the genotype–phenotype relationship for protein families.

Almost every process that keeps a cell alive depends on the activities of several proteins. All proteins are made from chains of smaller molecules called amino acids, and the specific sequence of amino acids determines the protein’s overall shape, which in turn controls what the protein can do. Yet, the relationships between a protein’s structure and its function are complex, and it remains unclear how the sequence of amino acids in a protein actually determine its features and properties.

Machine learning is a computational approach that is often applied to understand complex issues in biology. It uses computer algorithms to spot statistical patterns in large amounts of data and, after 'learning' from the data, the algorithms can then provide new insights, make predictions or even generate new data.

Tubiana et al. have now used a relatively simple form of machine learning to study the amino acid sequences of 20 different families of proteins. First, frameworks of algorithms –known as Restricted Boltzmann Machines, RBM for short – were trained to read some amino-acid sequence data that coded for similar proteins. After ‘learning’ from the data, the RBM could then infer statistical patterns that were common to the sequences. Tubiana et al. saw that many of these inferred patterns could be interpreted in a meaningful way and related to properties of the proteins. For example, some were related to known twists and loops that are commonly found in proteins; others could be linked to specific activities. This level of interpretability is somewhat at odds with the results of other common methods used in machine learning, which tend to behave more like a ‘black box’.

Using their RBM, Tubiana et al. then proposed how to design new proteins that may prove to have interesting features. In the future, similar methods could be applied across computational biology as a way to make sense of complex data in an understandable way.

[end paper 2]

[begin paper 3]

PMCID = 9523718

Title = Protein Design: From the Aspect of Water Solubility and Stability

Water solubility and structural stability are key merits for proteins defined by the primary sequence and 3D-conformation. Their manipulation represents important aspects of the protein design field that relies on the accurate placement of amino acids and molecular interactions, guided by underlying physiochemical principles. Emulated designer proteins with well-defined properties both fuel the knowledge-base for more precise computational design models and are used in various biomedical and nanotechnological applications. The continuous developments in protein science, increasing computing power, new algorithms, and characterization techniques provide sophisticated toolkits for solubility design beyond guess work. In this review, we summarize recent advances in the protein design field with respect to water solubility and structural stability. After introducing fundamental design rules, we discuss the transmembrane protein solubilization and de novo transmembrane protein design. Traditional strategies to enhance protein solubility and structural stability are introduced. The designs of stable protein complexes and high-order assemblies are covered. Computational methodologies behind these endeavors, including structure prediction programs, machine learning algorithms, and specialty software dedicated to the evaluation of protein solubility and aggregation, are discussed. The findings and opportunities for Cryo-EM are presented. This review provides an overview of significant progress and prospects in accurate protein design for solubility and stability.

[end paper 3]

[begin paper 4]

PMCID = 11265717

Title = An integrative approach to protein sequence design through multiobjective optimization

With recent methodological advances in the field of computational protein design, in particular those based on deep learning, there is an increasing need for frameworks that allow for coherent, direct integration of different models and objective functions into the generative design process. Here we demonstrate how evolutionary multiobjective optimization techniques can be adapted to provide such an approach. With the established Non-dominated Sorting Genetic Algorithm II (NSGA-II) as the optimization framework, we use AlphaFold2 and ProteinMPNN confidence metrics to define the objective space, and a mutation operator composed of ESM-1v and ProteinMPNN to rank and then redesign the least favorable positions. Using the two-state design problem of the foldswitching protein RfaH as an in-depth case study, and PapD and calmodulin as examples of higher-dimensional design problems, we show that the evolutionary multiobjective optimization approach leads to significant reduction in the bias and variance in RfaH native sequence recovery, compared to a direct application of ProteinMPNN. We suggest that this improvement is due to three factors: (i) the use of an informative mutation operator that accelerates the sequence space exploration, (ii) the parallel, iterative design process inherent to the genetic algorithm that improves upon the ProteinMPNN autoregressive sequence decoding scheme, and (iii) the explicit approximation of the Pareto front that leads to optimal design candidates representing diverse tradeoff conditions. We anticipate this approach to be readily adaptable to different models and broadly relevant for protein design tasks with complex specifications.

Proteins are the fundamental building blocks of life, and engineering them has broad applications in medicine and biotechnology. Computational methods that seek to model and predict the protein sequence-structure-function relationship have seen significant advancement from the recent development in deep learning. As more models become available, it remains an open question how to effectively combine them into a coherent computational design approach. One approach is to perform computational design with one model, and filter the design candidates with the others. In this work, we demonstrate how an optimization algorithm inspired by evolution can be adapted to provide an alternative framework that outperforms the post hoc filtering approach, especially for problems with multiple competing design specifications. Such a framework may enable researchers to more effectively integrate the strengths of different modeling approaches to tackle more complex design problems.

[end paper 4]

[begin paper 5]

PMCID = 7032036

Title = Advances in protein structure prediction and design

The prediction of protein three-dimensional structure from amino acid sequence has been a grand challenge problem in computational biophysics for decades, owing to its intrinsic scientific interest and also to the many potential applications for robust protein structure prediction algorithms, from genome interpretation to protein function prediction. More recently, the inverse problem —designing an amino acid sequence that will fold into a specified three-dimensional structure — has attracted growing attention as a potential route to the rational engineering of proteins with functions useful in biotechnology and medicine. Methods for the prediction and design of protein structures have advanced dramatically in the past decade. Increases in computing power and rapid growth in the protein sequence and structure databases have fuelled the development of new data-intensive and computationally-demanding approaches for structure prediction. New algorithms for designing protein folds and protein–protein interfaces have been used to engineer novel high-order assemblies and to design from scratch fluorescent proteins with novel or enhanced properties as well as signalling proteins with therapeutic potential. In this Review, we describe current approaches for protein structure prediction and design and highlight a selection of the successful applications they have enabled.

Predicting how proteins fold enables to infer their function. Conversely, rational protein design allows engineering novel protein functionalities. Recent improvements in computational algorithms and technological advancements dramatically increased the accuracy and speed of protein structure modelling, providing novel opportunities for controlling protein function with potential applications in biomedicine, industry and research.

[end paper 5]

[begin paper 6]

PMCID = 6918593

Title = Modeling aspects of the language of life through transfer-learning protein sequences

Predicting protein function and structure from sequence is one important challenge for computational biology. For 26 years, most state-of-the-art approaches combined machine learning and evolutionary information. However, for some applications retrieving related proteins is becoming too time-consuming. Additionally, evolutionary information is less powerful for small families, e.g. for proteins from the Dark Proteome. Both these problems are addressed by the new methodology introduced here.

We introduced a novel way to represent protein sequences as continuous vectors (embeddings) by using the language model ELMo taken from natural language processing. By modeling protein sequences, ELMo effectively captured the biophysical properties of the language of life from unlabeled big data (UniRef50). We refer to these new embeddings as SeqVec (Sequence-to-Vector) and demonstrate their effectiveness by training simple neural networks for two different tasks. At the per-residue level, secondary structure (Q3 = 79% ± 1, Q8 = 68% ± 1) and regions with intrinsic disorder (MCC = 0.59 ± 0.03) were predicted significantly better than through one-hot encoding or through Word2vec-like approaches. At the per-protein level, subcellular localization was predicted in ten classes (Q10 = 68% ± 1) and membrane-bound were distinguished from water-soluble proteins (Q2 = 87% ± 1). Although SeqVec embeddings generated the best predictions from single sequences, no solution improved over the best existing method using evolutionary information. Nevertheless, our approach improved over some popular methods using evolutionary information and for some proteins even did beat the best. Thus, they prove to condense the underlying principles of protein sequences. Overall, the important novelty is speed: where the lightning-fast HHblits needed on average about two minutes to generate the evolutionary information for a target protein, SeqVec created embeddings on average in 0.03 s. As this speed-up is independent of the size of growing sequence databases, SeqVec provides a highly scalable approach for the analysis of big data in proteomics, i.e. microbiome or metaproteome analysis.

Transfer-learning succeeded to extract information from unlabeled sequence databases relevant for various protein prediction tasks. SeqVec modeled the language of life, namely the principles underlying protein sequences better than any features suggested by textbooks and prediction methods. The exception is evolutionary information, however, that information is not available on the level of a single sequence.

[end paper 6]

[begin paper 7]

PMCID = 9148388

Title = Protein Assembly by Design

Proteins are nature’s primary building blocks for the construction of sophisticated molecular machines and dynamic materials, ranging from protein complexes such as photosystem II and nitrogenase that drive biogeochemical cycles to cytoskeletal assemblies and muscle fibers for motion. Such natural systems have inspired extensive efforts in the rational design of artificial protein assemblies in the last two decades. As molecular building blocks, proteins are highly complex, both in terms of their three-dimensional structures and chemical compositions. To enable control over the self-assembly of such complex molecules, scientists have devised many creative strategies by combining tools and principles of experimental and computational biophysics, supramolecular chemistry, inorganic chemistry, materials science, and polymer chemistry, among others. Owing to these innovative strategies, what started as a purely structure-building exercise two decades ago has, in short order, led to artificial protein assemblies with unprecedented structures and functions and protein-based materials with unusual properties. Our goal in the Review is to give an overview of this exciting and highly interdisciplinary area of research, first outlining the design strategies and tools that have been devised for controlling protein self-assembly, then describing the diverse structures of artificial protein assemblies, and finally highlighting the emergent properties and functions of these assemblies.

[end paper 7]

[begin paper 8]

PMCID = 11009965

Title = Computational Approaches to Predict Protein–Protein Interactions in Crowded Cellular Environments

Investigating protein–protein interactions is crucial for understanding cellular biological processes because proteins often function within molecular complexes rather than in isolation. While experimental and computational methods have provided valuable insights into these interactions, they often overlook a critical factor: the crowded cellular environment. This environment significantly impacts protein behavior, including structural stability, diffusion, and ultimately the nature of binding. In this review, we discuss theoretical and computational approaches that allow the modeling of biological systems to guide and complement experiments and can thus significantly advance the investigation, and possibly the predictions, of protein–protein interactions in the crowded environment of cell cytoplasm. We explore topics such as statistical mechanics for lattice simulations, hydrodynamic interactions, diffusion processes in high-viscosity environments, and several methods based on molecular dynamics simulations. By synergistically leveraging methods from biophysics and computational biology, we review the state of the art of computational methods to study the impact of molecular crowding on protein–protein interactions and discuss its potential revolutionizing effects on the characterization of the human interactome.

[end paper 8]

[begin paper 9]

PMCID = 10482863

Title = Tracing the birth of structural domains from loops during protein evolution

The structures and functions of proteins are embedded into the loop scaffolds of structural domains. Their origin and evolution remain mysterious. Here, we use a novel graph-theoretical approach to describe how modular and non-modular loop prototypes combine to form folded structures in protein domain evolution. Phylogenomic data-driven chronologies reoriented a bipartite network of loops and domains (and its projections) into ‘waterfalls’ depicting an evolving ‘elementary functionome’ (EF). Two primordial waves of functional innovation involving founder ‘p-loop’ and ‘winged-helix’ domains were accompanied by an ongoing emergence and reuse of structural and functional novelty. Metabolic pathways expanded before translation functionalities. A dual hourglass recruitment pattern transferred scale-free properties from loop to domain components of the EF network in generative cycles of hierarchical modularity. Modeling the evolutionary emergence of the oldest P-loop and winged-helix domains with AlphFold2 uncovered rapid convergence towards folded structure, suggesting that a folding vocabulary exists in loops for protein fold repurposing and design.

[end paper 9]

[begin paper 10]

PMCID = 7603796

Title = Macromolecular modeling and design in Rosetta: recent methods and frameworks

The Rosetta software for macromolecular modeling, docking, and design is extensively used in laboratories worldwide. During two decades of development by a community of laboratories at more than 60 institutions, Rosetta has been continuously refactored and extended. Here we review tools developed in the last five years, including over 80 methods. We discuss improvements to the score function, user interfaces, and usability. Rosetta is available at www.rosettacommons.org.

Tools developed over the past five years in the macromolecular modeling, docking and design software Rosetta are reviewed in this Perspective.

[end paper 10]

[begin paper 11]

PMCID = 7815116

Title = Rosetta design with co-evolutionary information retains protein function

Computational protein design has the ambitious goal of crafting novel proteins that address challenges in biology and medicine. To overcome these challenges, the computational protein modeling suite Rosetta has been tailored to address various protein design tasks. Recently, statistical methods have been developed that identify correlated mutations between residues in a multiple sequence alignment of homologous proteins. These subtle inter-dependencies in the occupancy of residue positions throughout evolution are crucial for protein function, but we found that three current Rosetta design approaches fail to recover these co-evolutionary couplings. Thus, we developed the Rosetta method ResCue (residue-coupling enhanced) that leverages co-evolutionary information to favor sequences which recapitulate correlated mutations, as observed in nature. To assess the protocols via recapitulation designs, we compiled a benchmark of ten proteins each represented by two, structurally diverse states. We could demonstrate that ResCue designed sequences with an average sequence recovery rate of 70%, whereas three other protocols reached not more than 50%, on average. Our approach had higher recovery rates also for functionally important residues, which were studied in detail. This improvement has only a minor negative effect on the fitness of the designed sequences as assessed by Rosetta energy. In conclusion, our findings support the idea that informing protocols with co-evolutionary signals helps to design stable and native-like proteins that are compatible with the different conformational states required for a complex function.

In homologous proteins, functionally or structurally important residues are strongly conserved. Thus, the consideration of conservation signals during protein design protocols can help to create sequences that are more native-like. However, the number of conserved residues is small in many proteins and not all important residues can be captured by conservation analysis. Residues are forming networks whose composition is dictated by protein structure function and thus is visible through the co-evolutionary analysis. Nowadays, advanced methods allow us to deduce these networks from multiple sequence alignments. Thus, we have implemented the novel Rosetta method termed ‘ResCue’ that informs the design protocol with co-evolutionary signals. Recapitulation designs based on ten difficult benchmarks made clear that this protocol creates sequences that are more native-like than three other, state-of-the-art design protocols.

[end paper 11]

[begin paper 12]

PMCID = 11036486

Title = Combining Rosetta Sequence Design with Protein Language Model Predictions Using Evolutionary Scale Modeling (ESM) as Restraint

Computational protein sequence design has the ambitious goal of modifying existing or creating new proteins; however, designing stable and functional proteins is challenging without predictability of protein dynamics and allostery. Informing protein design methods with evolutionary information limits the mutational space to more native-like sequences and results in increased stability while maintaining functions. Recently, language models, trained on millions of protein sequences, have shown impressive performance in predicting the effects of mutations. Assessing Rosetta-designed sequences with a language model showed scores that were worse than those of their original sequence. To inform Rosetta design protocols with language model predictions, we added a new metric to restrain the energy function during design using the Evolutionary Scale Modeling (ESM) model. The resulting sequences have better language model scores and similar sequence recovery, with only a minor decrease in the fitness as assessed by Rosetta energy. In conclusion, our work combines the strength of recent machine learning approaches with the Rosetta protein design toolbox.

[end paper 12]

[begin paper 13]

PMCID = 8238390

Title = Learning the Protein Language: Evolution, Structure and Function

Language models have recently emerged as a powerful machine learning approach for distilling information from massive protein sequence databases. From readily available sequence data alone, these models discover evolutionary, structural, and functional organization across protein space. Using language models, we can encode amino acid sequences into distributed vector representations that capture their structural and functional properties as well as evaluate the evolutionary fitness of sequence variants. We discuss recent advances in protein language modeling and their applications to downstream protein property prediction problems. We then consider how these models can be enriched with prior biological knowledge and introduce an approach for encoding protein structural knowledge into the learned representations. The knowledge distilled by these models allows us to improve downstream function prediction through transfer learning. Deep protein language models are revolutionizing protein biology. They suggest new ways to approach protein and therapeutic design. However, further developments are needed to encode strong biological priors into protein language models and to increase their accessibility to the broader community.

[end paper 13]

[begin paper 14]

PMCID = 7733882

Title = Deep Learning in Protein Structural Modeling and Design

Deep learning is catalyzing a scientific revolution fueled by big data, accessible toolkits, and powerful computational resources, impacting many fields, including protein structural modeling. Protein structural modeling, such as predicting structure from amino acid sequence and evolutionary information, designing proteins toward desirable functionality, or predicting properties or behavior of a protein, is critical to understand and engineer biological systems at the molecular level. In this review, we summarize the recent advances in applying deep learning techniques to tackle problems in protein structural modeling and design. We dissect the emerging approaches using deep learning techniques for protein structural modeling and discuss advances and challenges that must be addressed. We argue for the central importance of structure, following the “sequence  structure  function” paradigm. This review is directed to help both computational biologists to gain familiarity with the deep learning methods applied in protein modeling, and computer scientists to gain perspective on the biologically meaningful problems that may benefit from deep learning techniques.

Proteins are linear polymers that fold into an incredible variety of three-dimensional structures that enable sophisticated functionality for biology. Computational modeling allows scientists to predict the three-dimensional structure of proteins from genomes, predict properties or behavior of a protein, and even modify or design new proteins for a desired function. Advances in machine learning, especially deep learning, are catalyzing a revolution in the paradigm of scientific research. In this review, we summarize recent work in applying deep learning techniques to tackle problems in protein structural modeling and design. Some deep learning-based approaches, especially in structure prediction, now outperform conventional methods, often in combination with higher-resolution physical modeling. Challenges remain in experimental validation, benchmarking, leveraging known physics and interpreting models, and extending to other biomolecules and contexts.

Proteins fold into an incredible variety of three-dimensional structures to enable sophisticated functionality in biology. Advances in machine learning, especially in deep learning-related techniques, have opened up new avenues in many areas of protein modeling and design. This review dissects the emerging approaches and discusses advances and challenges that must be addressed.

[end paper 14]

[begin paper 15]

PMCID = 6904478

Title = Deciphering protein evolution and fitness landscapes with latent space models

Protein sequences contain rich information about protein evolution, fitness landscapes, and stability. Here we investigate how latent space models trained using variational auto-encoders can infer these properties from sequences. Using both simulated and real sequences, we show that the low dimensional latent space representation of sequences, calculated using the encoder model, captures both evolutionary and ancestral relationships between sequences. Together with experimental fitness data and Gaussian process regression, the latent space representation also enables learning the protein fitness landscape in a continuous low dimensional space. Moreover, the model is also useful in predicting protein mutational stability landscapes and quantifying the importance of stability in shaping protein evolution. Overall, we illustrate that the latent space models learned using variational auto-encoders provide a mechanism for exploration of the rich data contained in protein sequences regarding evolution, fitness and stability and hence are well-suited to help guide protein engineering efforts.

Multiple sequence alignments of proteins carry information about evolution, the protein’s fitness landscape and its stability in the face of mutations. Here, the authors demonstrate the utility of latent space models learned using variational autoencoders to infer these properties from sequences.

[end paper 15]

[begin paper 16]

PMCID = 10662474

Title = A sequence-based evolutionary distance method for Phylogenetic analysis of highly divergent proteins

Because of the limited effectiveness of prevailing phylogenetic methods when applied to highly divergent protein sequences, the phylogenetic analysis problem remains challenging. Here, we propose a sequence-based evolutionary distance algorithm termed sequence distance (SD), which innovatively incorporates site-to-site correlation within protein sequences into the distance estimation. In protein superfamilies, SD can effectively distinguish evolutionary relationships both within and between protein families, producing phylogenetic trees that closely align with those based on structural information, even with sequence identity less than 20%. SD is highly correlated with the similarity of the protein structure, and can calculate evolutionary distances for thousands of protein pairs within seconds using a single CPU, which is significantly faster than most protein structure prediction methods that demand high computational resources and long run times. The development of SD will significantly advance phylogenetics, providing researchers with a more accurate and reliable tool for exploring evolutionary relationships.

[end paper 16]

[begin paper 17]

PMCID = 9159519

Title = Methodology-Centered Review of Molecular Modeling, Simulation, and Prediction of SARS-CoV-2

Despite tremendous efforts in the past two years, our understanding of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), virus–host interactions, immune response, virulence, transmission, and evolution is still very limited. This limitation calls for further in-depth investigation. Computational studies have become an indispensable component in combating coronavirus disease 2019 (COVID-19) due to their low cost, their efficiency, and the fact that they are free from safety and ethical constraints. Additionally, the mechanism that governs the global evolution and transmission of SARS-CoV-2 cannot be revealed from individual experiments and was discovered by integrating genotyping of massive viral sequences, biophysical modeling of protein–protein interactions, deep mutational data, deep learning, and advanced mathematics. There exists a tsunami of literature on the molecular modeling, simulations, and predictions of SARS-CoV-2 and related developments of drugs, vaccines, antibodies, and diagnostics. To provide readers with a quick update about this literature, we present a comprehensive and systematic methodology-centered review. Aspects such as molecular biophysics, bioinformatics, cheminformatics, machine learning, and mathematics are discussed. This review will be beneficial to researchers who are looking for ways to contribute to SARS-CoV-2 studies and those who are interested in the status of the field.

[end paper 17]

[begin paper 18]

PMCID = 10412457

Title = Mega-scale experimental analysis of protein folding stability in biology and design

Advances in DNA sequencing and machine learning are providing insights into protein sequences and structures on an enormous scale. However, the energetics driving folding are invisible in these structures and remain largely unknown. The hidden thermodynamics of folding can drive disease, shape protein evolution and guide protein engineering, and new approaches are needed to reveal these thermodynamics for every sequence and structure. Here we present cDNA display proteolysis, a method for measuring thermodynamic folding stability for up to 900,000 protein domains in a one-week experiment. From 1.8 million measurements in total, we curated a set of around 776,000 high-quality folding stabilities covering all single amino acid variants and selected double mutants of 331 natural and 148 de novo designed protein domains 40–72 amino acids in length. Using this extensive dataset, we quantified (1) environmental factors influencing amino acid fitness, (2) thermodynamic couplings (including unexpected interactions) between protein sites, and (3) the global divergence between evolutionary amino acid usage and protein folding stability. We also examined how our approach could identify stability determinants in designed proteins and evaluate design methods. The cDNA display proteolysis method is fast, accurate and uniquely scalable, and promises to reveal the quantitative rules for how amino acid sequences encode folding stability.

Large-scale assays using cDNA display proteolysis are used to measure the folding stabilities of protein domains, providing a method to quantify the effects of mutations on protein folding, with applications in protein design.

[end paper 18]

[begin paper 19]

PMCID = 7531764

Title = Mass Spectrometry-Based Protein Footprinting for High Order Structure Analysis: Fundamentals and Applications

Proteins adopt different higher order structures (HOS) to enable their unique biological functions. Understanding the complexities of protein higher order structures and dynamics requires integrated approaches, where mass spectrometry (MS) is now positioned to play a key role. One of those approaches is protein footprinting. Although the initial demonstration of footprinting was for the HOS determination of protein/nucleic acid bonding, the concept was later adapted to MS-based protein HOS analysis, through which different covalent labeling approaches “mark” the solvent accessible surface area (SASA) of proteins to reflect protein HOS. Hydrogen deuterium exchange (HDX), where deuterium in D2O replaces hydrogen of the backbone amides, is the most common example of footprinting. Its advantage is the footprint reflects SASA and hydrogen bonding, whereas one method drawback is the labeling is reversible. Another is slow irreversible labeling of functional groups on amino-acid side chains by targeted reagents with high specificity, probing structural changes at selected sites. A third footprinting approach is by reactions with fast, irreversible labeling species that are highly reactive and footprint broadly several amino-acid chains on the time scale of sub-milliseconds. All these covalent labeling approaches combine to constitute a problem-solving toolbox that takes mass spectrometry as the measurement tool for HOS elucidation. As there has been a growing need for MS-based protein footprinting in both academia and industry owing to its high throughput capability, prompt availability, and high spatial resolution, we present a summary of the history, descriptions, principles, mechanisms, and applications of these covalent labeling approaches. Moreover, their applications are highlighted according to the biological questions they can answer. This review is intended as a tutorial for MS-based protein HOS elucidation and as a reference for investigators seeking a MS-based tool to address questions in protein science.

[end paper 19]

[begin paper 20]

PMCID = 10841881

Title = Impact of AlphaFold on structure prediction of protein complexes: The CASP15‐CAPRI experiment

We present the results for CAPRI Round 54, the 5th joint CASP‐CAPRI protein assembly prediction challenge. The Round offered 37 targets, including 14 homodimers, 3 homo‐trimers, 13 heterodimers including 3 antibody–antigen complexes, and 7 large assemblies. On average ~70 CASP and CAPRI predictor groups, including more than 20 automatics servers, submitted models for each target. A total of 21 941 models submitted by these groups and by 15 CAPRI scorer groups were evaluated using the CAPRI model quality measures and the DockQ score consolidating these measures. The prediction performance was quantified by a weighted score based on the number of models of acceptable quality or higher submitted by each group among their five best models. Results show substantial progress achieved across a significant fraction of the 60+ participating groups. High‐quality models were produced for about 40% of the targets compared to 8% two years earlier. This remarkable improvement is due to the wide use of the AlphaFold2 and AlphaFold2‐Multimer software and the confidence metrics they provide. Notably, expanded sampling of candidate solutions by manipulating these deep learning inference engines, enriching multiple sequence alignments, or integration of advanced modeling tools, enabled top performing groups to exceed the performance of a standard AlphaFold2‐Multimer version used as a yard stick. This notwithstanding, performance remained poor for complexes with antibodies and nanobodies, where evolutionary relationships between the binding partners are lacking, and for complexes featuring conformational flexibility, clearly indicating that the prediction of protein complexes remains a challenging problem.

[end paper 20]

[begin paper 21]

PMCID = 8794168

Title = Heuristic algorithms in evolutionary computation and modular organization of biological macromolecules: Applications to in vitro evolution

Evolutionary computing (EC) is an area of computer sciences and applied mathematics covering heuristic optimization algorithms inspired by evolution in Nature. EC extensively study all the variety of methods which were originally based on the principles of selectionism. As a result, many new algorithms and approaches, significantly more efficient than classical selectionist schemes, were found. This is especially true for some families of special problems. There are strong arguments to believe that EC approaches are quite suitable for modeling and numerical analysis of those methods of synthetic biology and biotechnology that are known as in vitro evolution. Therefore, it is natural to expect that the new algorithms and approaches developed in EC can be effectively applied in experiments on the directed evolution of biological macromolecules. According to the John Holland’s Schema theorem, the effective evolutionary search in genetic algorithms (GA) is provided by identifying short schemata of high fitness which in the further search recombine into the larger building blocks (BBs) with higher and higher fitness. The multimodularity of functional biological macromolecules and the preservation of already found modules in the evolutionary search have a clear analogy with the BBs in EC. It seems reasonable to try to transfer and introduce the methods of EC, preserving BBs and essentially accelerating the search, into experiments on in vitro evolution. We extend the key instrument of the Holland’s theory, the Royal Roads fitness function, to problems of the in vitro evolution (Biological Royal Staircase, BioRS, functions). The specific version of BioRS developed in this publication arises from the realities of experimental evolutionary search for (DNA-) RNA-devices (aptazymes). Our numerical tests showed that for problems with the BioRS functions, simple heuristic algorithms, which turned out to be very effective for preserving BBs in GA, can be very effective in in vitro evolution approaches. We are convinced that such algorithms can be implemented in modern methods of in vitro evolution to achieve significant savings in time and resources and a significant increase in the efficiency of evolutionary search.

[end paper 21]

[begin paper 22]

PMCID = 10768222

Title = Prop3D: A flexible, Python-based platform for machine learning with protein structural properties and biophysical data

Machine learning (ML) has a rich history in structural bioinformatics, and modern approaches, such as deep learning, are revolutionizing our knowledge of the subtle relationships between biomolecular sequence, structure, function, dynamics and evolution. As with any advance that rests upon statistical learning approaches, the recent progress in biomolecular sciences is enabled by the availability of vast volumes of sufficiently-variable data. To be useful, such data must be well-structured, machine-readable, intelligible and manipulable. These and related requirements pose challenges that become especially acute at the computational scales typical in ML. Furthermore, in structural bioinformatics such data generally relate to protein three-dimensional (3D) structures, which are inherently more complex than sequence-based data. A significant and recurring challenge concerns the creation of large, high-quality, openly-accessible datasets that can be used for specific training and benchmarking tasks in ML pipelines for predictive modeling projects, along with reproducible splits for training and testing.

Here, we report ‘Prop3D’, a platform that allows for the creation, sharing and extensible reuse of libraries of protein domains, featurized with biophysical and evolutionary properties that can range from detailed, atomically-resolved physicochemical quantities (e.g., electrostatics) to coarser, residue-level features (e.g., phylogenetic conservation). As a community resource, we also supply a ‘Prop3D-20sf’ protein dataset, obtained by applying our approach to CATH. We have developed and deployed the Prop3D framework, both in the cloud and on local HPC resources, to systematically and reproducibly create comprehensive datasets via the Highly Scalable Data Service (HSDS). Our datasets are freely accessible via a public HSDS instance, or they can be used with accompanying Python wrappers for popular ML frameworks.

Prop3D and its associated Prop3D-20sf dataset can be of broad utility in at least three ways. Firstly, the Prop3D workflow code can be customized and deployed on various cloud-based compute platforms, with scalability achieved largely by saving the results to distributed HDF5 files via HSDS. Secondly, the linked Prop3D-20sf dataset provides a hand-crafted, already-featurized dataset of protein domains for 20 highly-populated CATH families; importantly, provision of this pre-computed resource can aid the more efficient development (and reproducible deployment) of ML pipelines. Thirdly, Prop3D-20sf’s construction explicitly takes into account (in creating datasets and data-splits) the enigma of ‘data leakage’, stemming from the evolutionary relationships between proteins.

[end paper 22]

[begin paper 23]

PMCID = 8920274

Title = Large-scale design and refinement of stable proteins using sequence-only models

Engineered proteins generally must possess a stable structure in order to achieve their designed function. Stable designs, however, are astronomically rare within the space of all possible amino acid sequences. As a consequence, many designs must be tested computationally and experimentally in order to find stable ones, which is expensive in terms of time and resources. Here we use a high-throughput, low-fidelity assay to experimentally evaluate the stability of approximately 200,000 novel proteins. These include a wide range of sequence perturbations, providing a baseline for future work in the field. We build a neural network model that predicts protein stability given only sequences of amino acids, and compare its performance to the assayed values. We also report another network model that is able to generate the amino acid sequences of novel stable proteins given requested secondary sequences. Finally, we show that the predictive model—despite weaknesses including a noisy data set—can be used to substantially increase the stability of both expert-designed and model-generated proteins.

[end paper 23]

[begin paper 24]

PMCID = 7616297

Title = Opportunities and challenges in design and optimization of protein function

Protein design methodology has made remarkable progress over the past decade. Historically, the low reliability of purely structure-based design methods limited their application. But recent strategies that combine structure- and sequence-based calculations as well as machine-learning approaches, have dramatically improved success. These approaches have led to design of increasingly complex structures and therapeutically relevant activities. Additionally, protein optimization methods have improved the stability and activity of complex eukaryotic proteins. Thanks to their heightened reliability, design methods have been applied to improve therapeutics, enzymes for green chemistry, and generated vaccines, antivirals and drug-delivery nanostructures. These exciting developments bring protein design closer to becoming a mainstream approach in protein science and engineering. Moreover, they reflect an increased understanding of basic rules that govern the relationship between protein sequence, structure, and function. We note, however, that de novo design is still limited mostly to α-helix bundles, restricting its potential to generate sophisticated enzymes and diverse protein and small-molecule binders. Designing complex protein structures is a challenging next step if the field is to realize its objective of generating new-to-nature activities. Here, we review developments in computational protein optimization and de novo design of function and highlight pressing areas for future research.

[end paper 24]

[begin paper 25]

PMCID = 11721782

Title = Exploring Evolution to Uncover Insights Into Protein Mutational Stability

Determining the impact of mutations on the thermodynamic stability of proteins is essential for a wide range of applications such as rational protein design and genetic variant interpretation. Since protein stability is a major driver of evolution, evolutionary data are often used to guide stability predictions. Many state-of-the-art stability predictors extract evolutionary information from multiple sequence alignments of proteins homologous to a query protein, and leverage it to predict the effects of mutations on protein stability. To evaluate the power and the limitations of such methods, we used the massive amount of stability data recently obtained by deep mutational scanning to study how best to construct multiple sequence alignments and optimally extract evolutionary information from them. We tested different evolutionary models and found that, unexpectedly, independent-site models achieve similar accuracy to more complex epistatic models. A detailed analysis of the latter models suggests that their inference often results in noisy couplings, which do not appear to add predictive power over the independent-site contribution, at least in the context of stability prediction. Interestingly, by combining any of the evolutionary features with a simple structural feature, the relative solvent accessibility of the mutated residue, we achieved similar prediction accuracy to supervised, machine learning-based, protein stability change predictors. Our results provide new insights into the relationship between protein evolution and stability, and show how evolutionary information can be exploited to improve the performance of mutational stability prediction.

[end paper 25]

[begin paper 26]

PMCID = 9496392

Title = Protein Function Analysis through Machine Learning

Machine learning (ML) has been an important arsenal in computational biology used to elucidate protein function for decades. With the recent burgeoning of novel ML methods and applications, new ML approaches have been incorporated into many areas of computational biology dealing with protein function. We examine how ML has been integrated into a wide range of computational models to improve prediction accuracy and gain a better understanding of protein function. The applications discussed are protein structure prediction, protein engineering using sequence modifications to achieve stability and druggability characteristics, molecular docking in terms of protein–ligand binding, including allosteric effects, protein–protein interactions and protein-centric drug discovery. To quantify the mechanisms underlying protein function, a holistic approach that takes structure, flexibility, stability, and dynamics into account is required, as these aspects become inseparable through their interdependence. Another key component of protein function is conformational dynamics, which often manifest as protein kinetics. Computational methods that use ML to generate representative conformational ensembles and quantify differences in conformational ensembles important for function are included in this review. Future opportunities are highlighted for each of these topics.

[end paper 26]

[begin paper 27]

PMCID = 10893003

Title = Recent Progress of Protein Tertiary Structure Prediction

The prediction of three-dimensional (3D) protein structure from amino acid sequences has stood as a significant challenge in computational and structural bioinformatics for decades. Recently, the widespread integration of artificial intelligence (AI) algorithms has substantially expedited advancements in protein structure prediction, yielding numerous significant milestones. In particular, the end-to-end deep learning method AlphaFold2 has facilitated the rise of structure prediction performance to new heights, regularly competitive with experimental structures in the 14th Critical Assessment of Protein Structure Prediction (CASP14). To provide a comprehensive understanding and guide future research in the field of protein structure prediction for researchers, this review describes various methodologies, assessments, and databases in protein structure prediction, including traditionally used protein structure prediction methods, such as template-based modeling (TBM) and template-free modeling (FM) approaches; recently developed deep learning-based methods, such as contact/distance-guided methods, end-to-end folding methods, and protein language model (PLM)-based methods; multi-domain protein structure prediction methods; the CASP experiments and related assessments; and the recently released AlphaFold Protein Structure Database (AlphaFold DB). We discuss their advantages, disadvantages, and application scopes, aiming to provide researchers with insights through which to understand the limitations, contexts, and effective selections of protein structure prediction methods in protein-related fields.

[end paper 27]

[begin paper 28]

PMCID = 8250049

Title = Evolution, folding, and design of TIM barrels and related proteins

In-depth sequence analysis reveals that the protein fold universe is more evolutionarily connected than previously assumed.

Short ancestral fragments are observed to have propagated to many modern proteins and hint at possible evolutionary pathways.

Experimental reconstruction of such events by chimeragenesis and directed evolution allows to test evolutionary relationships.

Detailed knowledge of folding landscapes helps to understand evolutionary history and improve protein engineering.

The ubiquitous and versatile TIM-barrel fold is a model system to explore evolution, folding, and design.

Proteins are chief actors in life that perform a myriad of exquisite functions. This diversity has been enabled through the evolution and diversification of protein folds. Analysis of sequences and structures strongly suggest that numerous protein pieces have been reused as building blocks and propagated to many modern folds. This information can be traced to understand how the protein world has diversified. In this review, we discuss the latest advances in the analysis of protein evolutionary units, and we use as a model system one of the most abundant and versatile topologies, the TIM-barrel fold, to highlight the existing common principles that interconnect protein evolution, structure, folding, function, and design.

[end paper 28]

[begin paper 29]

PMCID = 8586911

Title = D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions

We combine advances in neural language modeling and structurally-motivated design to develop D-SCRIPT, an interpretable and generalizable deep learning model which predicts interaction between two proteins using only their sequence and maintains high accuracy with limited training data and across species. We show that a D-SCRIPT model trained on 38,345 human PPIs enables significantly improved functional characterization of fly proteins compared to the state-of-the-art approach. Evaluating the same D-SCRIPT model on protein complexes with known 3-D structure, we find that the inter-protein contact map output by D-SCRIPT has significant overlap with the ground truth. We apply D-SCRIPT to screen for PPIs in cow (Bos taurus) at a genome-wide scale, and focusing on rumen physiology, identify functional gene modules related to metabolism and immune response. The predicted interactions can then be leveraged for function prediction at scale, addressing the genome to phenome challenge, especially in species where little data is available.

With the explosive growth of available gene sequences, the ability to predict the interactome of an organism from its genome would help address the pressing challenge of genome-to-phenome mapping. Our method, D-SCRIPT, leverages advances in deep language models to map protein sequences to implicit structure representations and predict interaction between two proteins based on their structural compatibility. D-SCRIPT’s structure-aware approach generalizes to unseen species better than current approaches, and its efficiency allows for rapid genome-scale investigation of protein function.

[end paper 29]

[begin paper 30]

PMCID = 11089422

Title = A comprehensive review of protein-centric predictors for biomolecular interactions: from proteins to nucleic acids and beyond

Proteins interact with diverse ligands to perform a large number of biological functions, such as gene expression and signal transduction. Accurate identification of these protein–ligand interactions is crucial to the understanding of molecular mechanisms and the development of new drugs. However, traditional biological experiments are time-consuming and expensive. With the development of high-throughput technologies, an increasing amount of protein data is available. In the past decades, many computational methods have been developed to predict protein–ligand interactions. Here, we review a comprehensive set of over 160 protein–ligand interaction predictors, which cover protein–protein, protein−nucleic acid, protein−peptide and protein−other ligands (nucleotide, heme, ion) interactions. We have carried out a comprehensive analysis of the above four types of predictors from several significant perspectives, including their inputs, feature profiles, models, availability, etc. The current methods primarily rely on protein sequences, especially utilizing evolutionary information. The significant improvement in predictions is attributed to deep learning methods. Additionally, sequence-based pretrained models and structure-based approaches are emerging as new trends.

[end paper 30]

[begin paper 31]

PMCID = 11070111

Title = Myofilament-associated proteins with intrinsic disorder (MAPIDs) and their resolution by computational modeling

The cardiac sarcomere is a cellular structure in the heart that enables muscle cells to contract. Dozens of proteins belong to the cardiac sarcomere, which work in tandem to generate force and adapt to demands on cardiac output. Intriguingly, the majority of these proteins have significant intrinsic disorder that contributes to their functions, yet the biophysics of these intrinsically disordered regions (IDRs) have been characterized in limited detail. In this review, we first enumerate these myofilament-associated proteins with intrinsic disorder (MAPIDs) and recent biophysical studies to characterize their IDRs. We secondly summarize the biophysics governing IDR properties and the state-of-the-art in computational tools toward MAPID identification and characterization of their conformation ensembles. We conclude with an overview of future computational approaches toward broadening the understanding of intrinsic disorder in the cardiac sarcomere.

[end paper 31]

[begin paper 32]

PMCID = 8336924

Title = Folding non-homologous proteins by coupling deep-learning contact maps with I-TASSER assembly simulations

Structure prediction for proteins lacking homologous templates in the Protein Data Bank (PDB) remains a significant unsolved problem. We developed a protocol, C-I-TASSER, to integrate interresidue contact maps from deep neural-network learning with the cutting-edge I-TASSER fragment assembly simulations. Large-scale benchmark tests showed that C-I-TASSER can fold more than twice the number of non-homologous proteins than the I-TASSER, which does not use contacts. When applied to a folding experiment on 8,266 unsolved Pfam families, C-I-TASSER successfully folded 4,162 domain families, including 504 folds that are not found in the PDB. Furthermore, it created correct folds for 85% of proteins in the SARS-CoV-2 genome, despite the quick mutation rate of the virus and sparse sequence profiles. The results demonstrated the critical importance of coupling whole-genome and metagenome-based evolutionary information with optimal structure assembly simulations for solving the problem of non-homologous protein structure prediction.

C-I-TASSER adds deep-learning contact prediction to fragment assembly simulations

C-I-TASSER enables ab initio folding of proteins lacking homology in the PDB

The inherent force field is critical for proteins with poor templates and sparse MSAs

Half of unsolved Pfam families are foldable by C-I-TASSER

Taking advantage of the rapid progress in deep-learning technologies, residue-residue contact-map prediction recently achieved impressive breakthroughs. However, how to efficiently convert the binary contact maps into atomic-level structure models remains an important unsolved problem in ab initio protein structure prediction. In this work, we integrated the deep-learning contact-map predictions with cutting-edge threading assembly simulations and found that the inherent force field of the structural folding simulations is essential to maximize the potential of contact-assisted protein structure prediction, especially for the targets and regions that lack spatial restraints and sufficient evolutionary data.

Zheng et al. develop C-I-TASSER, which integrates interresidue contact maps from deep neural-network learning with the cutting-edge I-TASSER fragment assembly simulations, for high-accuracy protein structure prediction. C-I-TASSER folds more than twice the number of proteins without homology than I-TASSER and has successfully folded 50% of Pfam families without solved experimental structures.

[end paper 32]

[begin paper 33]

PMCID = 9017885

Title = Dynamic coupling of residues within proteins as a mechanistic foundation of many enigmatic pathogenic missense variants

Many pathogenic missense mutations are found in protein positions that are neither well-conserved nor fall in any known functional domains. Consequently, we lack any mechanistic underpinning of dysfunction caused by such mutations. We explored the disruption of allosteric dynamic coupling between these positions and the known functional sites as a possible mechanism for pathogenesis. In this study, we present an analysis of 591 pathogenic missense variants in 144 human enzymes that suggests that allosteric dynamic coupling of mutated positions with known active sites is a plausible biophysical mechanism and evidence of their functional importance. We illustrate this mechanism in a case study of β-Glucocerebrosidase (GCase) in which a vast majority of 94 sites harboring Gaucher disease-associated missense variants are located some distance away from the active site. An analysis of the conformational dynamics of GCase suggests that mutations on these distal sites cause changes in the flexibility of active site residues despite their distance, indicating a dynamic communication network throughout the protein. The disruption of the long-distance dynamic coupling caused by missense mutations may provide a plausible general mechanistic explanation for biological dysfunction and disease.

Genetic diseases often occur when mutations in proteins cause gain/loss of functions. Although several methods based on conservation and protein biochemistry exist to predict genetic mutations that may impact function, many disease-associated mutations remain unexplained by these metrics. In this study, we sought a mechanistic explanation for such disease-associated mutations. In order to function, important regions of a protein must be able to exhibit collective motion. Through computer simulations, we observed that mutation of even a single amino acid position within a protein can change the protein motion. We found that disease-associated mutations tend to alter the motion of regions critical to protein function, even though these mutations occur far from these critical regions. In addition, we examined the degree to which two amino acid positions within a protein may be “coupled,” i.e., the extent to which motion in one position affects the other. We found that positions highly coupled to the active site of a protein are more likely to result in disease when mutated, thereby offering a new tool for predicting pathogenesis of new mutations by incorporating internal protein dynamics.

[end paper 33]

[begin paper 34]

PMCID = 6650774

Title = Biomolecular Assemblies: Moving from Observation to Predictive Design

Biomolecular assembly is a key driving force in nearly all life processes, providing structure, information storage, and communication within cells and at the whole organism level. These assembly processes rely on precise interactions between functional groups on nucleic acids, proteins, carbohydrates, and small molecules, and can be fine tuned to span a range of time, length, and complexity scales. Recognizing the power of these motifs, researchers have sought to emulate and engineer biomolecular assemblies in the laboratory, with goals ranging from modulating cellular function to the creation of new polymeric materials. In most cases, engineering efforts are inspired or informed by understanding the structure and properties of naturally occurring assemblies, which has in turn fueled the development of predictive models that enable computational design of novel assemblies. This Review will focus on selected examples of protein assemblies, highlighting the story arc from initial discovery of an assembly, through initial engineering attempts, toward the ultimate goal of predictive design. The aim of this Review is to highlight areas where significant progress has been made, as well as to outline remaining challenges, as solving these challenges will be the key that unlocks the full power of biomolecules for advances in technology and medicine.

[end paper 34]

[begin paper 35]

PMCID = 11350595

Title = Exploring protein functions from structural flexibility using CABS‐flex modeling

Understanding protein function often necessitates characterizing the flexibility of protein structures. However, simulating protein flexibility poses significant challenges due to the complex dynamics of protein systems, requiring extensive computational resources and accurate modeling techniques. In response to these challenges, the CABS‐flex method has been developed as an efficient modeling tool that combines coarse‐grained simulations with all‐atom detail. Available both as a web server and a standalone package, CABS‐flex is dedicated to a wide range of users. The web server version offers an accessible interface for straightforward tasks, while the standalone command‐line program is designed for advanced users, providing additional features, analytical tools, and support for handling large systems. This paper examines the application of CABS‐flex across various structure–function studies, facilitating investigations into the interplay among protein structure, dynamics, and function in diverse research fields. We present an overview of the current status of the CABS‐flex methodology, highlighting its recent advancements, practical applications, and forthcoming challenges.

[end paper 35]

[begin paper 36]

PMCID = 11395565

Title = Utilizing Molecular Dynamics Simulations, Machine Learning, Cryo-EM, and NMR Spectroscopy to Predict and Validate Protein Dynamics

Protein dynamics play a crucial role in biological function, encompassing motions ranging from atomic vibrations to large-scale conformational changes. Recent advancements in experimental techniques, computational methods, and artificial intelligence have revolutionized our understanding of protein dynamics. Nuclear magnetic resonance spectroscopy provides atomic-resolution insights, while molecular dynamics simulations offer detailed trajectories of protein motions. Computational methods applied to X-ray crystallography and cryo-electron microscopy (cryo-EM) have enabled the exploration of protein dynamics, capturing conformational ensembles that were previously unattainable. The integration of machine learning, exemplified by AlphaFold2, has accelerated structure prediction and dynamics analysis. These approaches have revealed the importance of protein dynamics in allosteric regulation, enzyme catalysis, and intrinsically disordered proteins. The shift towards ensemble representations of protein structures and the application of single-molecule techniques have further enhanced our ability to capture the dynamic nature of proteins. Understanding protein dynamics is essential for elucidating biological mechanisms, designing drugs, and developing novel biocatalysts, marking a significant paradigm shift in structural biology and drug discovery.

[end paper 36]

[begin paper 37]

PMCID = 11275831

Title = Evolutionary Implications of Self-Assembling Cybernetic Materials with Collective Problem-Solving Intelligence at Multiple Scales

In recent years, the scientific community has increasingly recognized the complex multi-scale competency architecture (MCA) of biology, comprising nested layers of active homeostatic agents, each forming the self-orchestrated substrate for the layer above, and, in turn, relying on the structural and functional plasticity of the layer(s) below. The question of how natural selection could give rise to this MCA has been the focus of intense research. Here, we instead investigate the effects of such decision-making competencies of MCA agential components on the process of evolution itself, using in silico neuroevolution experiments of simulated, minimal developmental biology. We specifically model the process of morphogenesis with neural cellular automata (NCAs) and utilize an evolutionary algorithm to optimize the corresponding model parameters with the objective of collectively self-assembling a two-dimensional spatial target pattern (reliable morphogenesis). Furthermore, we systematically vary the accuracy with which the uni-cellular agents of an NCA can regulate their cell states (simulating stochastic processes and noise during development). This allows us to continuously scale the agents’ competency levels from a direct encoding scheme (no competency) to an MCA (with perfect reliability in cell decision executions). We demonstrate that an evolutionary process proceeds much more rapidly when evolving the functional parameters of an MCA compared to evolving the target pattern directly. Moreover, the evolved MCAs generalize well toward system parameter changes and even modified objective functions of the evolutionary process. Thus, the adaptive problem-solving competencies of the agential parts in our NCA-based in silico morphogenesis model strongly affect the evolutionary process, suggesting significant functional implications of the near-ubiquitous competency seen in living matter.

[end paper 37]

[begin paper 38]

PMCID = 5800964

Title = Biophysical models of protein evolution: Understanding the patterns of evolutionary sequence divergence

For decades, rates of protein evolution have been interpreted in terms of the vague concept of “functional importance”. Slowly evolving proteins or sites within proteins were assumed to be more functionally important and thus subject to stronger selection pressure. More recently, biophysical models of protein evolution, which combine evolutionary theory with protein biophysics, have completely revolutionized our view of the forces that shape sequence divergence. Slowly evolving proteins have been found to evolve slowly because of selection against toxic misfolding and misinteractions, linking their rate of evolution primarily to their abundance. Similarly, most slowly evolving sites in proteins are not directly involved in function, but mutating them has large impacts on protein structure and stability. Here, we review the studies of the emergent field of biophysical protein evolution that have shaped our current understanding of sequence divergence patterns. We also propose future research directions to develop this nascent field.

[end paper 38]

[begin paper 39]

PMCID = 11228553

Title = The evolution of mammalian Rem2: unraveling the impact of purifying selection and coevolution on protein function, and implications for human disorders

Rad And Gem-Like GTP-Binding Protein 2 (Rem2), a member of the RGK family of Ras-like GTPases, is implicated in Huntington’s disease and Long QT Syndrome and is highly expressed in the brain and endocrine cells. We examine the evolutionary history of Rem2 identified in various mammalian species, focusing on the role of purifying selection and coevolution in shaping its sequence and protein structural constraints. Our analysis of Rem2 sequences across 175 mammalian species found evidence for strong purifying selection in 70% of non-invariant codon sites which is characteristic of essential proteins that play critical roles in biological processes and is consistent with Rem2’s role in the regulation of neuronal development and function. We inferred epistatic effects in 50 pairs of codon sites in Rem2, some of which are predicted to have deleterious effects on human health. Additionally, we reconstructed the ancestral evolutionary history of mammalian Rem2 using protein structure prediction of extinct and extant sequences which revealed the dynamics of how substitutions that change the gene sequence of Rem2 can impact protein structure in variable regions while maintaining core functional mechanisms. By understanding the selective pressures, protein- and gene - interactions that have shaped the sequence and structure of the Rem2 protein, we gain a stronger understanding of its biological and functional constraints.

[end paper 39]

[begin paper 40]

PMCID = 10684548

Title = Ongoing shuffling of protein fragments diversifies core viral functions linked to interactions with bacterial hosts

Biological modularity enhances evolutionary adaptability. This principle is vividly exemplified by bacterial viruses (phages), which display extensive genomic modularity. Phage genomes are composed of independent functional modules that evolve separately and recombine in various configurations. While genomic modularity in phages has been extensively studied, less attention has been paid to protein modularity—proteins consisting of distinct building blocks that can evolve and recombine, enhancing functional and genetic diversity. Here, we use a set of 133,574 representative phage proteins and highly sensitive homology detection to capture instances of domain mosaicism, defined as fragment sharing between two otherwise unrelated proteins, and to understand its relationship with functional diversity in phage genomes. We discover that unrelated proteins from diverse functional classes frequently share homologous domains. This phenomenon is particularly pronounced within receptor-binding proteins, endolysins, and DNA polymerases. We also identify multiple instances of recent diversification via domain shuffling in receptor-binding proteins, neck passage structures, endolysins and some members of the core replication machinery, often transcending distant taxonomic and ecological boundaries. Our findings suggest that ongoing diversification via domain shuffling is reflective of a co-evolutionary arms race, driven by the need to overcome various bacterial resistance mechanisms against phages.

Proteins are composed of distinct functional domains, each serving a specific role. Here, Smug et al. show that phages are able to shuffle fragments of their proteins and this predominantly occurs in proteins involved in bacterial host interactions.

[end paper 40]

[begin paper 41]

PMCID = 7259980

Title = Predicting the viability of beta-lactamase: How folding and binding free energies correlate with beta-lactamase fitness

One of the long-standing holy grails of molecular evolution has been the ability to predict an organism’s fitness directly from its genotype. With such predictive abilities in hand, researchers would be able to more accurately forecast how organisms will evolve and how proteins with novel functions could be engineered, leading to revolutionary advances in medicine and biotechnology. In this work, we assemble the largest reported set of experimental TEM-1 β-lactamase folding free energies and use this data in conjunction with previously acquired fitness data and computational free energy predictions to determine how much of the fitness of β-lactamase can be directly predicted by thermodynamic folding and binding free energies. We focus upon β-lactamase because of its long history as a model enzyme and its central role in antibiotic resistance. Based upon a set of 21 β-lactamase single and double mutants expressly designed to influence protein folding, we first demonstrate that modeling software designed to compute folding free energies such as FoldX and PyRosetta can meaningfully, although not perfectly, predict the experimental folding free energies of single mutants. Interestingly, while these techniques also yield sensible double mutant free energies, we show that they do so for the wrong physical reasons. We then go on to assess how well both experimental and computational folding free energies explain single mutant fitness. We find that folding free energies account for, at most, 24% of the variance in β-lactamase fitness values according to linear models and, somewhat surprisingly, complementing folding free energies with computationally-predicted binding free energies of residues near the active site only increases the folding-only figure by a few percent. This strongly suggests that the majority of β-lactamase’s fitness is controlled by factors other than free energies. Overall, our results shed a bright light on to what extent the community is justified in using thermodynamic measures to infer protein fitness as well as how applicable modern computational techniques for predicting free energies will be to the large data sets of multiply-mutated proteins forthcoming.

[end paper 41]

[begin paper 42]

PMCID = 10861886

Title = Learning the shape of protein microenvironments with a holographic convolutional neural network

Proteins are the machinery of life facilitating the key processes that drive living organisms. Recent advances have increased the number of experimentally resolved or computationally predicted tertiary structures of proteins. However, we still lack an understanding of how tertiary structure determines the function of a protein. M. Pun et al. introduce a physically motivated machine learning approach to learn interpretable models for protein structures, reflecting the underlying biophysics. This model accurately predicts the impact of mutations on protein stability and binding of protein complexes. The flexibility and efficiency of this approach also show promise for building generative models to design novel protein structures with desired stability and binding reactivity.

Proteins play a central role in biology from immune recognition to brain activity. While major advances in machine learning have improved our ability to predict protein structure from sequence, determining protein function from its sequence or structure remains a major challenge. Here, we introduce holographic convolutional neural network (H-CNN) for proteins, which is a physically motivated machine learning approach to model amino acid preferences in protein structures. H-CNN reflects physical interactions in a protein structure and recapitulates the functional information stored in evolutionary data. H-CNN accurately predicts the impact of mutations on protein stability and binding of protein complexes. Our interpretable computational model for protein structure–function maps could guide design of novel proteins with desired function.

[end paper 42]

[begin paper 43]

PMCID = 6394115

Title = The Evolutionary Traceability of a Protein

Orthologs document the evolution of genes and metabolic capacities encoded in extant and ancient genomes. However, the similarity between orthologs decays with time, and ultimately it becomes insufficient to infer common ancestry. This leaves ancient gene set reconstructions incomplete and distorted to an unknown extent. Here we introduce the “evolutionary traceability” as a measure that quantifies, for each protein, the evolutionary distance beyond which the sensitivity of the ortholog search becomes limiting. Using yeast, we show that genes that were thought to date back to the last universal common ancestor are of high traceability. Their functions mostly involve catalysis, ion transport, and ribonucleoprotein complex assembly. In turn, the fraction of yeast genes whose traceability is not sufficient to infer their presence in last universal common ancestor is enriched for regulatory functions. Computing the traceabilities of genes that have been experimentally characterized as being essential for a self-replicating cell reveals that many of the genes that lack orthologs outside bacteria have low traceability. This leaves open whether their orthologs in the eukaryotic and archaeal domains have been overlooked. Looking at the example of REC8, a protein essential for chromosome cohesion, we demonstrate how a traceability-informed adjustment of the search sensitivity identifies hitherto missed orthologs in the fast-evolving microsporidia. Taken together, the evolutionary traceability helps to differentiate between true absence and nondetection of orthologs, and thus improves our understanding about the evolutionary conservation of functional protein networks. “protTrace,” a software tool for computing evolutionary traceability, is freely available at https://github.com/BIONF/protTrace.git; last accessed February 10, 2019.

[end paper 43]

[begin paper 44]

PMCID = 6728967

Title = Sibe: a computation tool to apply protein sequence statistics to predict folding and design in silico

Evolutionary information contained in the amino acid sequences of proteins specifies the biological function and fold, but exactly what information contained in the protein sequence drives both of these processes? Considerable progress has been made to answer this fundamental question, but it remains challenging to explore the potential space of cooperative interactions between amino acids. Statistical analysis plays a significant role in studying such interactions and its use has expanded in recent years to studies ranging from coevolution-guided rational protein design to protein folding in silico.

Here we describe a computational tool named Sibe for use in studies of protein sequence, folding, and design using evolutionary coupling between amino acids as a driving factor. In this study, Sibe is used to identify positionally conserved couplings between pairwise amino acids and aid rational protein design. In this process, pairwise couplings are filtered according to the relative entropy computed from the positional conservations and grouped into several ’blocks’, which could contribute to driving protein folding and design. A human β2-adrenergic receptor (β2AR) was used to demonstrate that those ’blocks’ contribute the rational design for specifying functional residues. Sibe also provides folding modules based on both the positionally conserved couplings and well-established statistical potentials for simulating protein folding in silico and predicting tertiary structure. Our results show that statistically inferences of basic evolutionary principles, such as conservations and coupled-mutations, can be used to rapidly design a diverse set of proteins and study protein folding.

The developed software Sibe provides a computational tool for systematical analysis from protein primary to its tertiary structure using the evolutionary couplings as a driving factor. Sibe, written in C++, accounts for compatibility with the ’big data’ era in biological science, and it primarily focuses on protein sequence analysis, but it is also applicable to extend to other modeling and predictions of experimental measurements

The online version of this article (10.1186/s12859-019-2984-1) contains supplementary material, which is available to authorized users.

[end paper 44]

[begin paper 45]

PMCID = 8222070

Title = Deep learning techniques have significantly impacted protein structure prediction and protein design

Protein structure prediction and design can be regarded as two inverse processes governed by the same folding principle. Although progress remained stagnant over the last two decades, the recent application of deep neural networks to spatial constraint prediction and end-to-end model training has significantly improved the accuracy of protein structure prediction, largely solving the problem at the fold level for single-domain proteins. The field of protein design has also witnessed dramatic improvement, where noticeable examples have shown that information stored in neural-network models can be used to advance functional protein design. Thus, incorporation of deep learning techniques into different steps of protein folding and design approaches represents an exciting future direction and should continue to have a transformative impact on both fields.

[end paper 45]

[begin paper 46]

PMCID = 7243446

Title = De novo protein design, a retrospective

Proteins are molecular machines whose function depends on their ability to achieve complex folds with precisely defined structural and dynamic properties. The rational design of proteins from first-principles, or de novo, was once considered to be impossible, but today proteins with a variety of folds and functions have been realized. We review the evolution of the field from its earliest days, placing particular emphasis on how this endeavor has illuminated our understanding of the principles underlying the folding and function of natural proteins, and is informing the design of macromolecules with unprecedented structures and properties. An initial set of milestones in de novo protein design focused on the construction of sequences that folded in water and membranes to adopt folded conformations. The first proteins were designed from first-principles using very simple physical models. As computers became more powerful, the use of the rotamer approximation allowed one to discover amino acid sequences that stabilize the desired fold. As the crystallographic database of protein structures expanded in subsequent years, it became possible to construct proteins by assembling short backbone fragments that frequently recur in Nature. The second set of milestones in de novo design involves the discovery of complex functions. Proteins have been designed to bind a variety of metals, porphyrins, and other cofactors. The design of proteins that catalyze hydrolysis and oxygen-dependent reactions has progressed significantly. However, de novo design of catalysts for energetically demanding reactions, or even proteins that bind with high affinity and specificity to highly functionalized complex polar molecules remains an importnant challenge that is now being achieved. Finally, the protein design contributed significantly to our understanding of membrane protein folding and transport of ions across membranes. The area of membrane protein design, or more generally of biomimetic polymers that function in mixed or non-aqueous environments, is now becoming increasingly possible.

[end paper 46]

[begin paper 47]

PMCID = 10075473

Title = Atomistic simulation of protein evolution reveals sequence covariation and time-dependent fluctuations of site-specific substitution rates

Thermodynamic stability is a crucial fitness constraint in protein evolution and is a central factor in shaping the sequence landscapes of proteins. The correlation between stability and molecular fitness depends on the mechanism that relates the biophysical property with biological function. In the simplest case, stability and fitness are related by the amount of folded protein. However, when proteins are toxic in the unfolded state, the fitness function shifts, resulting in higher stability under mutation-selection balance. Likewise, a higher population size results in a similar change in protein stability, as it magnifies the effect of the selection pressure in evolutionary dynamics. This study investigates how such factors affect the evolution of protein stability, site-specific mutation rates, and residue-residue covariation. To simulate evolutionary trajectories with realistic modeling of protein energetics, we develop an all-atom simulator of protein evolution, RosettaEvolve. By evolving proteins under different fitness functions, we can study how the fitness function affects the distribution of proposed and accepted mutations, site-specific rates, and the prevalence of correlated amino acid substitutions. We demonstrate that fitness pressure affects the proposal distribution of mutational effects, that changes in stability can largely explain variations in site-specific substitution rates in evolutionary trajectories, and that increased fitness pressure results in a stronger covariation signal. Our results give mechanistic insight into the evolutionary consequences of variation in protein stability and provide a basis to rationalize the strong covariation signal observed in natural sequence alignments.

Modern-day proteins are the result of the process of evolution. The fate of random substitutions at the nucleotide level is dependent on the fitness of the new gene variant. One of the strongest fitness pressures shaping the sequences of protein is thermodynamic stability; proteins must typically be stable to carry out its function and misfolded proteins can be toxic. To understand the importance of thermodynamic stability in protein evolution and to what extent it can explain natural sequence variation we have developed a method for simulating protein evolution using a three-dimensional structure and structure-based stability calculations. In the simulations, the strength of selection can be varied, and complete phylogenetic trees of a protein family can be generated. Using these simulations, we demonstrate how mutation rates at individual sites in a protein are coupled to the overall stability of the protein, and how the spectrum of accepted mutations is shaped by stability, and how strong interactions between residues in a protein can result in sequence covariation.

[end paper 47]

[begin paper 48]

PMCID = 10813169

Title = Predictive Modeling of Proteins Encoded by a Plant Virus Sheds a New Light on Their Structure and Inherent Multifunctionality

Plant virus genomes encode proteins that are involved in replication, encapsidation, cell-to-cell, and long-distance movement, avoidance of host detection, counter-defense, and transmission from host to host, among other functions. Even though the multifunctionality of plant viral proteins is well documented, contemporary functional repertoires of individual proteins are incomplete. However, these can be enhanced by modeling tools. Here, predictive modeling of proteins encoded by the two genomic RNAs, i.e., RNA1 and RNA2, of grapevine fanleaf virus (GFLV) and their satellite RNAs by a suite of protein prediction software confirmed not only previously validated functions (suppressor of RNA silencing [VSR], viral genome-linked protein [VPg], protease [Pro], symptom determinant [Sd], homing protein [HP], movement protein [MP], coat protein [CP], and transmission determinant [Td]) and previously identified putative functions (helicase [Hel] and RNA-dependent RNA polymerase [Pol]), but also predicted novel functions with varying levels of confidence. These include a T3/T7-like RNA polymerase domain for protein 1AVSR, a short-chain reductase for protein 1BHel/VSR, a parathyroid hormone family domain for protein 1EPol/Sd, overlapping domains of unknown function and an ABC transporter domain for protein 2BMP, and DNA topoisomerase domains, transcription factor FBXO25 domain, or DNA Pol subunit cdc27 domain for the satellite RNA protein. Structural predictions for proteins 2AHP/Sd, 2BMP, and 3A? had low confidence, while predictions for proteins 1AVSR, 1BHel*/VSR, 1CVPg, 1DPro, 1EPol*/Sd, and 2CCP/Td retained higher confidence in at least one prediction. This research provided new insights into the structure and functions of GFLV proteins and their satellite protein. Future work is needed to validate these findings.

[end paper 48]

[begin paper 49]

PMCID = 5909693

Title = Enhancing structure prediction and design of soluble and membrane proteins with explicit solvent-protein interactions

Solvent molecules interact intimately with proteins and can profoundly regulate their structure and function. However, accurately and efficiently modeling protein solvation effects at the molecular level has been challenging. Here, we present a method that improves the atomic-level modeling of soluble and membrane protein structures and binding by efficiently predicting de novo protein-solvent molecule interactions. The method predicted with unprecedented accuracy buried water molecule positions, solvated protein conformations, and challenging mutational effects on protein binding. When applied to homology modeling, solvent-bound membrane protein structures, pockets, and cavities were recapitulated with near-atomic precision even from distant homologs. Blindly refined atomic-level structures of evolutionary distant G protein-coupled receptors imply strikingly different functional roles of buried solvent between receptor classes. The method should prove useful for refining low-resolution protein structures, accurately modeling drug binding sites in structurally-uncharacterized receptors, and designing solvent-mediated protein catalysis, recognition, ligand binding, and membrane protein signaling.

Lai et al present a method to improve atomic-level modeling of protein structures and binding by efficiently predicting protein-solvent molecule interactions. The approach recapitulated mutational effects on protein binding, modeled solvated cavities from distant homologs, and buried solvent networks in membrane proteins with unprecedented accuracy.

[end paper 49]

[begin paper 50]

PMCID = 10400306

Title = Large language models generate functional protein sequences across diverse families

Generating artificial protein sequences using artificial intelligence could enable breakthrough solutions for biomedical and environmental challenges. Viewing amino acid sequences as a language, we demonstrate that a deep learning-based language model can generate functional artificial protein sequences across large protein families, akin to generating grammatically and semantically correct natural language sentences on diverse topics. Our protein language model is trained by simply learning to predict the next amino acid for over 280 million protein sequences from thousands of protein families, without explicit biophysical modeling. We experimentally evaluate model-generated artificial proteins fine-tuned to five distinct antibacterial lysozyme families. Artificial proteins show similar activities and catalytic efficiencies as representative natural lysozymes, including hen egg white lysozyme, while maintaining activity with as low as 31.4% identity to any known naturally-evolved protein. The X-ray crystal structure of an enzymatically active artificial protein recapitulates the conserved fold and positioning of active site residues found in natural proteins. We show our language model’s ability to be adapted to different protein families by accurately predicting the functionality of artificial chorismate mutase and malate dehydrogenase proteins. These results indicate that neural language models successfully perform artificial protein generation across protein families and may prove to be a tool to shortcut evolution.

[end paper 50]

[begin paper 51]

PMCID = 10589401

Title = Generative models for protein sequence modeling: recent advances and future directions

The widespread adoption of high-throughput omics technologies has exponentially increased the amount of protein sequence data involved in many salient disease pathways and their respective therapeutics and diagnostics. Despite the availability of large-scale sequence data, the lack of experimental fitness annotations underpins the need for self-supervised and unsupervised machine learning (ML) methods. These techniques leverage the meaningful features encoded in abundant unlabeled sequences to accomplish complex protein engineering tasks. Proficiency in the rapidly evolving fields of protein engineering and generative AI is required to realize the full potential of ML models as a tool for protein fitness landscape navigation. Here, we support this work by (i) providing an overview of the architecture and mathematical details of the most successful ML models applicable to sequence data (e.g. variational autoencoders, autoregressive models, generative adversarial neural networks, and diffusion models), (ii) guiding how to effectively implement these models on protein sequence data to predict fitness or generate high-fitness sequences and (iii) highlighting several successful studies that implement these techniques in protein engineering (from paratope regions and subcellular localization prediction to high-fitness sequences and protein design rules generation). By providing a comprehensive survey of model details, novel architecture developments, comparisons of model applications, and current challenges, this study intends to provide structured guidance and robust framework for delivering a prospective outlook in the ML-driven protein engineering field.

[end paper 51]

[begin paper 52]

PMCID = 6247239

Title = Award Winners and Abstracts of the 32nd Annual Symposium of The Protein Society; Boston, MA, July 9–12, 2018



[end paper 52]

[begin paper 53]

PMCID = 5831536

Title = Electrostatic Interactions in Protein Structure, Folding, Binding, and Condensation

Charged and polar groups, through forming ion pairs, hydrogen bonds, and other less specific electrostatic interactions, impart important properties to proteins. Modulation of the charges on the amino acids, e.g., by pH and by phosphorylation and dephosphorylation, have significant effects such as protein denaturation and switch-like response of signal transduction networks. This review aims to present a unifying theme among the various effects of protein charges and polar groups. Simple models will be used to illustrate basic ideas about electrostatic interactions in proteins, and these ideas in turn will be used to elucidate the roles of electrostatic interactions in protein structure, folding, binding, condensation, and related biological functions. In particular, we will examine how charged side chains are spatially distributed in various types of proteins and how electrostatic interactions affect thermodynamic and kinetic properties of proteins. Our hope is to capture both important historical developments and recent experimental and theoretical advances in quantifying electrostatic contributions of proteins.

[end paper 53]

[begin paper 54]

PMCID = 8366979

Title = Deep geometric representations for modeling effects of mutations on protein-protein binding affinity

Modeling the impact of amino acid mutations on protein-protein interaction plays a crucial role in protein engineering and drug design. In this study, we develop GeoPPI, a novel structure-based deep-learning framework to predict the change of binding affinity upon mutations. Based on the three-dimensional structure of a protein, GeoPPI first learns a geometric representation that encodes topology features of the protein structure via a self-supervised learning scheme. These representations are then used as features for training gradient-boosting trees to predict the changes of protein-protein binding affinity upon mutations. We find that GeoPPI is able to learn meaningful features that characterize interactions between atoms in protein structures. In addition, through extensive experiments, we show that GeoPPI achieves new state-of-the-art performance in predicting the binding affinity changes upon both single- and multi-point mutations on six benchmark datasets. Moreover, we show that GeoPPI can accurately estimate the difference of binding affinities between a few recently identified SARS-CoV-2 antibodies and the receptor-binding domain (RBD) of the S protein. These results demonstrate the potential of GeoPPI as a powerful and useful computational tool in protein design and engineering. Our code and datasets are available at: https://github.com/Liuxg16/GeoPPI.

Estimating the binding affinities of protein-protein interactions (PPIs) is crucial to understand protein function and design new functional proteins. Since the experimental measurement in wet-labs is labor-intensive and time-consuming, fast and accurate in silico approaches have received much attention. Although considerable efforts have been made in this direction, predicting the effects of mutations on the protein-protein binding affinity is still a challenging research problem. In this work, we introduce GeoPPI, a novel computational approach that uses deep geometric representations of protein complexes to predict the effects of mutations on the binding affinity. The geometric representations are first learned via a self-supervised learning scheme and then integrated with gradient-boosting trees to accomplish the prediction. We find that the learned representations encode meaningful patterns underlying the interactions between atoms in protein structures. Also, extensive tests on major benchmark datasets show that GeoPPI has made an important improvement over the existing methods in predicting the effects of mutations on the binding affinity.

[end paper 54]

[begin paper 55]

PMCID = 11023698

Title = Genome-scale annotation of protein binding sites via language model and geometric deep learning

Revealing protein binding sites with other molecules, such as nucleic acids, peptides, or small ligands, sheds light on disease mechanism elucidation and novel drug design. With the explosive growth of proteins in sequence databases, how to accurately and efficiently identify these binding sites from sequences becomes essential. However, current methods mostly rely on expensive multiple sequence alignments or experimental protein structures, limiting their genome-scale applications. Besides, these methods haven’t fully explored the geometry of the protein structures. Here, we propose GPSite, a multi-task network for simultaneously predicting binding residues of DNA, RNA, peptide, protein, ATP, HEM, and metal ions on proteins. GPSite was trained on informative sequence embeddings and predicted structures from protein language models, while comprehensively extracting residual and relational geometric contexts in an end-to-end manner. Experiments demonstrate that GPSite substantially surpasses state-of-the-art sequence-based and structure-based approaches on various benchmark datasets, even when the structures are not well-predicted. The low computational cost of GPSite enables rapid genome-scale binding residue annotations for over 568,000 sequences, providing opportunities to unveil unexplored associations of binding sites with molecular functions, biological processes, and genetic variants. The GPSite webserver and annotation database can be freely accessed at https://bio-web1.nscc-gz.cn/app/GPSite.

[end paper 55]

[begin paper 56]

PMCID = 10089152

Title = De novo design of small beta barrel proteins

De novo design of mini-proteins for engineering new functions has primarily focused on all-alpha and alpha-beta folds. Despite the enormous functional diversity observed in naturally occurring beta barrels less than 70 residues in length, the de novo design of these structures has not been achieved. Here, we describe the de novo design and characterization of four different classes of small beta barrels. The results provide insight into the determinants of folding and design of this important class of proteins, and provide routes to the design of high-affinity binding proteins.

Small beta barrel proteins are attractive targets for computational design because of their considerable functional diversity despite their very small size (<70 amino acids). However, there are considerable challenges to designing such structures, and there has been little success thus far. Because of the small size, the hydrophobic core stabilizing the fold is necessarily very small, and the conformational strain of barrel closure can oppose folding; also intermolecular aggregation through free beta strand edges can compete with proper monomer folding. Here, we explore the de novo design of small beta barrel topologies using both Rosetta energy–based methods and deep learning approaches to design four small beta barrel folds: Src homology 3 (SH3) and oligonucleotide/oligosaccharide-binding (OB) topologies found in nature and five and six up-and-down-stranded barrels rarely if ever seen in nature. Both approaches yielded successful designs with high thermal stability and experimentally determined structures with less than 2.4 Å rmsd from the designed models. Using deep learning for backbone generation and Rosetta for sequence design yielded higher design success rates and increased structural diversity than Rosetta alone. The ability to design a large and structurally diverse set of small beta barrel proteins greatly increases the protein shape space available for designing binders to protein targets of interest.

[end paper 56]

[begin paper 57]

PMCID = 10729993

Title = Computational design of novel Cas9 PAM-interacting domains using evolution-based modelling and structural quality assessment

We present here an approach to protein design that combines (i) scarce functional information such as experimental data (ii) evolutionary information learned from a natural sequence variants and (iii) physics-grounded modeling. Using a Restricted Boltzmann Machine (RBM), we learn a sequence model of a protein family. We use semi-supervision to leverage available functional information during the RBM training. We then propose a strategy to explore the protein representation space that can be informed by external models such as an empirical force-field method (FoldX). Our approach is applied to a domain of the Cas9 protein responsible for recognition of a short DNA motif. We experimentally assess the functionality of 71 variants generated to explore a range of RBM and FoldX energies. Sequences with as many as 50 differences (20% of the protein domain) to the wild-type retained functionality. Overall, 21/71 sequences designed with our method were functional. Interestingly, 6/71 sequences showed an improved activity in comparison with the original wild-type protein sequence. These results demonstrate the interest in further exploring the synergies between machine-learning of protein sequence representations and physics grounded modeling strategies informed by structural information.

Proteins are essential molecules in all living organisms, with their function largely determined by their sequence. Modifying a protein’s sequence to achieve a desired function remains a challenging endeavor, requiring careful consideration of factors such as the stability of the structure and interactions with molecular partners. In this study, we devised a protein design method that combines insights from experimental data, natural variation in protein sequences, and physics-based predictions. This approach provides a reliable and interpretable means of altering a protein’s sequence while maintaining its functionality. We applied our technique to a domain of the Cas9 protein, a key component in the CRISPR gene editing system. Our results demonstrate the possibility of generating functional protein domains with over 20% of their sequence modified. These findings underscore how the integration of diverse sources of information in a unified design process enhances the quality of engineered proteins. This advancement holds promise for creating valuable protein variants for applications in drug development and various industries.

[end paper 57]

[begin paper 58]

PMCID = 11275918

Title = Modeling Alternative Conformational States of Pseudo-Symmetric Solute Carrier Transporters using Methods from Deep Learning

The Solute Carrier (SLC) superfamily of integral membrane proteins function to transport a wide array of small molecules across plasma and organelle membranes. SLC proteins also function as important drug transporters and as viral receptors. Despite being classified as a single superfamily, SLC proteins do not share a single common fold classification; however, most belong to multi-pass transmembrane helical protein fold families. SLC proteins populate different conformational states during the solute transport process, including outward-open, intermediate (occluded), and inward-open conformational states. For some SLC fold families this structural “flipping” corresponds to swapping between conformations of their N-terminal and C-terminal symmetry-related sub-structures. Conventional AlphaFold2, AlphaFold3, or Evolutionary Scale Modeling methods typically generate models for only one of these multiple conformational states of SLC proteins. Several modifications of these AI-based protocols for modeling multiple conformational states of proteins have been described recently. These methods are often impacted by “memorization” of one of the alternative conformational states, and do not always provide both the inward and outward facing conformations of SLC proteins. Here we describe a combined ESM – template-based-modeling process, based on a previously described template-based modeling method that relies on the internal pseudo-symmetry of many SLC proteins, to consistently model alternate conformational states of SLC proteins. We further demonstrate how the resulting multi-state models can be validated experimentally by comparison with sequence-based evolutionary co-variance data (ECs) that encode information about contacts present in the various conformational states adopted by the protein. This simple, rapid, and robust approach for modeling conformational landscapes of pseudo-symmetric SLC proteins is demonstrated for several integral membrane protein transporters, including SLC35F2 the receptor of a feline leukemia virus envelope protein required for viral entry into eukaryotic cells.

[end paper 58]

[begin paper 59]

PMCID = 10942313

Title = An integrative approach to protein sequence design through multiobjective optimization

With recent methodological advances in the field of computational protein design, in particular those based on deep learning, there is an increasing need for frameworks that allow for coherent, direct integration of different models and objective functions into the generative design process. Here we demonstrate how evolutionary multiobjective optimization techniques can be adapted to provide such an approach. With the established Non-dominated Sorting Genetic Algorithm II (NSGA-II) as the optimization framework, we use AlphaFold2 and ProteinMPNN confidence metrics to define the objective space, and a mutation operator composed of ESM-1v and ProteinMPNN to rank and then redesign the least favorable positions. Using the multistate design problem of the foldswitching protein RfaH as an in-depth case study, we show that the evolutionary multiobjective optimization approach leads to significant reduction in the bias and variance in RfaH native sequence recovery, compared to a direct application of ProteinMPNN. We suggest that this improvement is due to three factors: (i) the use of an informative mutation operator that accelerates the sequence space exploration, (ii) the parallel, iterative design process inherent to the genetic algorithm that improves upon the ProteinMPNN autoregressive sequence decoding scheme, and (iii) the explicit approximation of the Pareto front that leads to optimal design candidates representing diverse tradeoff conditions. We anticipate this approach to be readily adaptable to different models and broadly relevant for protein design tasks with complex specifications.

Proteins are the fundamental building blocks of life, and engineering them has broad applications in medicine and biotechnology. Computational methods that seek to model and predict the protein sequence-structure-function relationship have seen significant advancement from the recent development in deep learning. As more models become available, it remains an open question how to effectively combine them into a coherent computational design approach. One approach is to perform computational design with one model, and filter the design candidates with the others. In this work, we demonstrate how an optimization algorithm inspired by evolution can be adapted to provide an alternative framework that outperforms the post hoc filtering approach, especially for problems with multiple competing design specifications. Such a framework may enable researchers to more effectively integrate the strengths of different modeling approaches to tackle more complex design problems.

[end paper 59]

[begin paper 60]

PMCID = 8647222

Title = The Power of Universal Contextualized Protein Embeddings in Cross-species Protein Function Prediction

Computationally annotating proteins with a molecular function is a difficult problem that is made even harder due to the limited amount of available labeled protein training data. Unsupervised protein embeddings partly circumvent this limitation by learning a universal protein representation from many unlabeled sequences. Such embeddings incorporate contextual information of amino acids, thereby modeling the underlying principles of protein sequences insensitive to the context of species. We used an existing pre-trained protein embedding method and subjected its molecular function prediction performance to detailed characterization, first to advance the understanding of protein language models, and second to determine areas of improvement. Then, we applied the model in a transfer learning task by training a function predictor based on the embeddings of annotated protein sequences of one training species and making predictions on the proteins of several test species with varying evolutionary distance. We show that this approach successfully generalizes knowledge about protein function from one eukaryotic species to various other species, outperforming both an alignment-based and a supervised-learning-based baseline. This implies that such a method could be effective for molecular function prediction in inadequately annotated species from understudied taxonomic kingdoms.

[end paper 60]

[begin paper 61]

PMCID = 11651203

Title = Applicability of AlphaFold2 in the modeling of dimeric, trimeric, and tetrameric coiled‐coil domains

Coiled coils are a common protein structural motif involved in cellular functions ranging from mediating protein–protein interactions to facilitating processes such as signal transduction or regulation of gene expression. They are formed by two or more alpha helices that wind around a central axis to form a buried hydrophobic core. Various forms of coiled‐coil bundles have been reported, each characterized by the number, orientation, and degree of winding of the constituent helices. This variability is underpinned by short sequence repeats that form coiled coils and whose properties determine both their overall topology and the local geometry of the hydrophobic core. The strikingly repetitive sequence has enabled the development of accurate sequence‐based coiled‐coil prediction methods; however, the modeling of coiled‐coil domains remains a challenging task. In this work, we evaluated the accuracy of AlphaFold2 in modeling coiled‐coil domains, both in modeling local geometry and in predicting global topological properties. Furthermore, we show that the prediction of the oligomeric state of coiled‐coil bundles can be achieved by using the internal representations of AlphaFold2, with a performance better than any previous state‐of‐the‐art method (code available at https://github.com/labstructbioinf/dc2_oligo).

[end paper 61]

[begin paper 62]

PMCID = 11313475

Title = AI-Driven Deep Learning Techniques in Protein Structure Prediction

Protein structure prediction is important for understanding their function and behavior. This review study presents a comprehensive review of the computational models used in predicting protein structure. It covers the progression from established protein modeling to state-of-the-art artificial intelligence (AI) frameworks. The paper will start with a brief introduction to protein structures, protein modeling, and AI. The section on established protein modeling will discuss homology modeling, ab initio modeling, and threading. The next section is deep learning-based models. It introduces some state-of-the-art AI models, such as AlphaFold (AlphaFold, AlphaFold2, AlphaFold3), RoseTTAFold, ProteinBERT, etc. This section also discusses how AI techniques have been integrated into established frameworks like Swiss-Model, Rosetta, and I-TASSER. The model performance is compared using the rankings of CASP14 (Critical Assessment of Structure Prediction) and CASP15. CASP16 is ongoing, and its results are not included in this review. Continuous Automated Model EvaluatiOn (CAMEO) complements the biennial CASP experiment. Template modeling score (TM-score), global distance test total score (GDT_TS), and Local Distance Difference Test (lDDT) score are discussed too. This paper then acknowledges the ongoing difficulties in predicting protein structure and emphasizes the necessity of additional searches like dynamic protein behavior, conformational changes, and protein–protein interactions. In the application section, this paper introduces some applications in various fields like drug design, industry, education, and novel protein development. In summary, this paper provides a comprehensive overview of the latest advancements in established protein modeling and deep learning-based models for protein structure predictions. It emphasizes the significant advancements achieved by AI and identifies potential areas for further investigation.

[end paper 62]

[begin paper 63]

PMCID = 9850016

Title = Towards evolutionary predictions: Current promises and challenges

Evolution has traditionally been a historical and descriptive science, and predicting future evolutionary processes has long been considered impossible. However, evolutionary predictions are increasingly being developed and used in medicine, agriculture, biotechnology and conservation biology. Evolutionary predictions may be used for different purposes, such as to prepare for the future, to try and change the course of evolution or to determine how well we understand evolutionary processes. Similarly, the exact aspect of the evolved population that we want to predict may also differ. For example, we could try to predict which genotype will dominate, the fitness of the population or the extinction probability of a population. In addition, there are many uses of evolutionary predictions that may not always be recognized as such. The main goal of this review is to increase awareness of methods and data in different research fields by showing the breadth of situations in which evolutionary predictions are made. We describe how diverse evolutionary predictions share a common structure described by the predictive scope, time scale and precision. Then, by using examples ranging from SARS‐CoV2 and influenza to CRISPR‐based gene drives and sustainable product formation in biotechnology, we discuss the methods for predicting evolution, the factors that affect predictability and how predictions can be used to prevent evolution in undesirable directions or to promote beneficial evolution (i.e. evolutionary control). We hope that this review will stimulate collaboration between fields by establishing a common language for evolutionary predictions.

[end paper 63]

[begin paper 64]

PMCID = 10570143

Title = Novel machine learning approaches revolutionize protein knowledge

Two artificial intelligence (AI)-based methods for protein structure prediction, AlphaFold 2 and RoseTTAFold, increase dramatically the quality of structural modeling from sequence, nearing experimental accuracy.

Protein language models encode the written language of proteins, allowing for more accurate annotations and predictions than homology-based methods.

Most model organisms, neglected disease pathogens, and proteins with curated annotations have models available with varying quality, aiding wet-laboratory experiments targeting single-question issues.

Ultrafast alignment tools can traverse the protein space by both sequence and structure to identify remote evolutionary relations previously precluded to older and slower methods.

Preliminary analyses of predicted AlphaFold 2 3D-models from 21 model organisms suggest that the majority (>90%) of globular domains in proteins can be assigned to currently characterized domain evolutionary superfamilies.

Breakthrough methods in machine learning (ML), protein structure prediction, and novel ultrafast structural aligners are revolutionizing structural biology. Obtaining accurate models of proteins and annotating their functions on a large scale is no longer limited by time and resources. The most recent method to be top ranked by the Critical Assessment of Structure Prediction (CASP) assessment, AlphaFold 2 (AF2), is capable of building structural models with an accuracy comparable to that of experimental structures. Annotations of 3D models are keeping pace with the deposition of the structures due to advancements in protein language models (pLMs) and structural aligners that help validate these transferred annotations. In this review we describe how recent developments in ML for protein science are making large-scale structural bioinformatics available to the general scientific community.

[end paper 64]

[begin paper 65]

PMCID = 7020559

Title = High-throughput quantification of protein structural change reveals potential mechanisms of temperature adaptation in Mytilus mussels

Temperature exerts a strong influence on protein evolution: species living in thermally distinct environments often exhibit adaptive differences in protein structure and function. However, previous research on protein temperature adaptation has focused on small numbers of proteins and on proteins adapted to extreme temperatures. Consequently, less is known about the types and quantity of evolutionary change that occurs to proteins when organisms adapt to small shifts in environmental temperature. In this study, these uncertainties were addressed by developing software that enabled comparison of structural changes associated with temperature adaptation (hydrogen bonding, salt bridge formation, and amino acid use) among large numbers of proteins from warm- and cold-adapted species of marine mussels, Mytilus galloprovincialis and Mytilus trossulus, respectively.

Small differences in habitat temperature that characterize the evolutionary history of Mytilus mussels were sufficient to cause protein structural changes consistent with temperature adaptation. Hydrogen bonds and salt bridges that increase stability and protect against heat-induced denaturation were more abundant in proteins from warm-adapted M. galloprovincialis compared with proteins from cold-adapted M. trossulus. These structural changes were related to deviations in the use of polar and charged amino acids that facilitate formation of hydrogen bonds and salt bridges within proteins, respectively. Enzymes, in particular those within antioxidant and cell death pathways, were over-represented among proteins with the most hydrogen bonds and salt bridges in warm-adapted M. galloprovincialis. Unlike extremophile proteins, temperature adaptation in Mytilus proteins did not involve substantial changes in the number of hydrophobic or large volume amino acids, nor in the content of glycine or proline.

Small shifts in organism temperature tolerance, such as that needed to cope with climate warming, may result from structural and functional changes to a small percentage of the proteome. Proteins in which function is dependent on large conformational change, notably enzymes, may be particularly sensitive to temperature perturbation and represent foci for natural selection. Protein temperature adaptation can occur through different types and frequencies of structural change, and adaptive mechanisms used to cope with small shifts in habitat temperature appear different from mechanisms used to retain protein function at temperature extremes.

[end paper 65]

[begin paper 66]

PMCID = 11281659

Title = Are Viruses Taxonomic Units? A Protein Domain and Loop-Centric Phylogenomic Assessment

Virus taxonomy uses a Linnaean-like subsumption hierarchy to classify viruses into taxonomic units at species and higher rank levels. Virus species are considered monophyletic groups of mobile genetic elements (MGEs) often delimited by the phylogenetic analysis of aligned genomic or metagenomic sequences. Taxonomic units are assumed to be independent organizational, functional and evolutionary units that follow a ‘natural history’ rationale. Here, I use phylogenomic and other arguments to show that viruses are not self-standing genetically-driven systems acting as evolutionary units. Instead, they are crucial components of holobionts, which are units of biological organization that dynamically integrate the genetics, epigenetic, physiological and functional properties of their co-evolving members. Remarkably, phylogenomic analyses show that viruses share protein domains and loops with cells throughout history via massive processes of reticulate evolution, helping spread evolutionary innovations across a wider taxonomic spectrum. Thus, viruses are not merely MGEs or microbes. Instead, their genomes and proteomes conduct cellularly integrated processes akin to those cataloged by the GO Consortium. This prompts the generation of compositional hierarchies that replace the ‘is-a-kind-of’ by a ‘is-a-part-of’ logic to better describe the mereology of integrated cellular and viral makeup. My analysis demands a new paradigm that integrates virus taxonomy into a modern evolutionarily centered taxonomy of organisms.

[end paper 66]

[begin paper 67]

PMCID = 10991518

Title = Genomic language model predicts protein co-regulation and function

Deciphering the relationship between a gene and its genomic context is fundamental to understanding and engineering biological systems. Machine learning has shown promise in learning latent relationships underlying the sequence-structure-function paradigm from massive protein sequence datasets. However, to date, limited attempts have been made in extending this continuum to include higher order genomic context information. Evolutionary processes dictate the specificity of genomic contexts in which a gene is found across phylogenetic distances, and these emergent genomic patterns can be leveraged to uncover functional relationships between gene products. Here, we train a genomic language model (gLM) on millions of metagenomic scaffolds to learn the latent functional and regulatory relationships between genes. gLM learns contextualized protein embeddings that capture the genomic context as well as the protein sequence itself, and encode biologically meaningful and functionally relevant information (e.g. enzymatic function, taxonomy). Our analysis of the attention patterns demonstrates that gLM is learning co-regulated functional modules (i.e. operons). Our findings illustrate that gLM’s unsupervised deep learning of the metagenomic corpus is an effective and promising approach to encode functional semantics and regulatory syntax of genes in their genomic contexts and uncover complex relationships between genes in a genomic region.

A gene’s function is governed by its sequence, structure and context. Here, the authors develop a genomic language model that learns contextualized functional representations from diverse and large-scale metagenomic datasets.

[end paper 67]

[begin paper 68]

PMCID = 11057625

Title = Agent-based modeling in cancer biomedicine: applications and tools for calibration and validation

Computational models are not just appealing because they can simulate and predict the development of biological phenomena across multiple spatial and temporal scales, but also because they can integrate information from well-established in vitro and in vivo models and test new hypotheses in cancer biomedicine. Agent-based models and simulations are especially interesting candidates among computational modeling procedures in cancer research due to the capability to, for instance, recapitulate the dynamics of neoplasia and tumor – host interactions. Yet, the absence of methods to validate the consistency of the results across scales can hinder adoption by turning fine-tuned models into black boxes. This review compiles relevant literature that explores strategies to leverage high-fidelity simulations of multi-scale, or multi-level, cancer models with a focus on verification approached as simulation calibration. We consolidate our review with an outline of modern approaches for agent-based models’ validation and provide an ambitious outlook toward rigorous and reliable calibration.

[end paper 68]

[begin paper 69]

PMCID = 8396596

Title = Modeling and Structure Determination of Homo-Oligomeric Proteins: An Overview of Challenges and Current Approaches

Protein homo-oligomerization is a very common phenomenon, and approximately half of proteins form homo-oligomeric assemblies composed of identical subunits. The vast majority of such assemblies possess internal symmetry which can be either exploited to help or poses challenges during structure determination. Moreover, aspects of symmetry are critical in the modeling of protein homo-oligomers either by docking or by homology-based approaches. Here, we first provide a brief overview of the nature of protein homo-oligomerization. Next, we describe how the symmetry of homo-oligomers is addressed by crystallographic and non-crystallographic symmetry operations, and how biologically relevant intermolecular interactions can be deciphered from the ordered array of molecules within protein crystals. Additionally, we describe the most important aspects of protein homo-oligomerization in structure determination by NMR. Finally, we give an overview of approaches aimed at modeling homo-oligomers using computational methods that specifically address their internal symmetry and allow the incorporation of other experimental data as spatial restraints to achieve higher model reliability.

[end paper 69]

[begin paper 70]

PMCID = 3749098

Title = Structural Phylogenomics Retrodicts the Origin of the Genetic Code and Uncovers the Evolutionary Impact of Protein Flexibility

The genetic code shapes the genetic repository. Its origin has puzzled molecular scientists for over half a century and remains a long-standing mystery. Here we show that the origin of the genetic code is tightly coupled to the history of aminoacyl-tRNA synthetase enzymes and their interactions with tRNA. A timeline of evolutionary appearance of protein domain families derived from a structural census in hundreds of genomes reveals the early emergence of the ‘operational’ RNA code and the late implementation of the standard genetic code. The emergence of codon specificities and amino acid charging involved tight coevolution of aminoacyl-tRNA synthetases and tRNA structures as well as episodes of structural recruitment. Remarkably, amino acid and dipeptide compositions of single-domain proteins appearing before the standard code suggest archaic synthetases with structures homologous to catalytic domains of tyrosyl-tRNA and seryl-tRNA synthetases were capable of peptide bond formation and aminoacylation. Results reveal that genetics arose through coevolutionary interactions between polypeptides and nucleic acid cofactors as an exacting mechanism that favored flexibility and folding of the emergent proteins. These enhancements of phenotypic robustness were likely internalized into the emerging genetic system with the early rise of modern protein structure.

[end paper 70]

[begin paper 71]

PMCID = 9119679

Title = Heterogeneity of the GFP fitness landscape and data-driven protein design

Studies of protein fitness landscapes reveal biophysical constraints guiding protein evolution and empower prediction of functional proteins. However, generalisation of these findings is limited due to scarceness of systematic data on fitness landscapes of proteins with a defined evolutionary relationship. We characterized the fitness peaks of four orthologous fluorescent proteins with a broad range of sequence divergence. While two of the four studied fitness peaks were sharp, the other two were considerably flatter, being almost entirely free of epistatic interactions. Mutationally robust proteins, characterized by a flat fitness peak, were not optimal templates for machine-learning-driven protein design – instead, predictions were more accurate for fragile proteins with epistatic landscapes. Our work paves insights for practical application of fitness landscape heterogeneity in protein engineering.

[end paper 71]

[begin paper 72]

PMCID = 10089191

Title = Improving the secretion of designed protein assemblies through negative design of cryptic transmembrane domains

Most biopharmaceuticals traverse the eukaryotic secretory pathway during their production as either recombinant proteins or as mRNA-delivered products. As designed proteins more commonly form the basis of new medicines, methods for optimizing their secretion will become increasingly important. Here we present a general computational method for improving protein secretion that focuses on the elimination of cryptic transmembrane domains while maintaining overall protein stability. We show that the method can retrospectively and prospectively improve the secretion of computationally designed protein nanoparticles, a promising platform for the development of new vaccines and therapeutics.

Computationally designed protein nanoparticles have recently emerged as a promising platform for the development of new vaccines and biologics. For many applications, secretion of designed nanoparticles from eukaryotic cells would be advantageous, but in practice, they often secrete poorly. Here we show that designed hydrophobic interfaces that drive nanoparticle assembly are often predicted to form cryptic transmembrane domains, suggesting that interaction with the membrane insertion machinery could limit efficient secretion. We develop a general computational protocol, the Degreaser, to design away cryptic transmembrane domains without sacrificing protein stability. The retroactive application of the Degreaser to previously designed nanoparticle components and nanoparticles considerably improves secretion, and modular integration of the Degreaser into design pipelines results in new nanoparticles that secrete as robustly as naturally occurring protein assemblies. Both the Degreaser protocol and the nanoparticles we describe may be broadly useful in biotechnological applications.

[end paper 72]

[begin paper 73]

PMCID = 11662154

Title = How the technologies behind self‐driving cars, social networks, ChatGPT, and DALL‐E2 are changing structural biology How the technologies behind self‐driving cars, social networks, CHATGPT, and DALL‐E2 are changing structural biology

The performance of deep Neural Networks (NNs) in the text (ChatGPT) and image (DALL‐E2) domains has attracted worldwide attention. Convolutional NNs (CNNs), Large Language Models (LLMs), Denoising Diffusion Probabilistic Models (DDPMs)/Noise Conditional Score Networks (NCSNs), and Graph NNs (GNNs) have impacted computer vision, language editing and translation, automated conversation, image generation, and social network management. Proteins can be viewed as texts written with the alphabet of amino acids, as images, or as graphs of interacting residues. Each of these perspectives suggests the use of tools from a different area of deep learning for protein structural biology. Here, I review how CNNs, LLMs, DDPMs/NCSNs, and GNNs have led to major advances in protein structure prediction, inverse folding, protein design, and small molecule design. This review is primarily intended as a deep learning primer for practicing experimental structural biologists. However, extensive references to the deep learning literature should also make it relevant to readers who have a background in machine learning, physics or statistics, and an interest in protein structural biology.

Deep learning, a brain inspired field of computer science, revolutionizes structural biology. I summarize how convolutional neural networks (CNNs), large language models (LLMs), denoising diffusion probabilistic models (DDPMs)/Noise conditional score networks (NCSNs), and graph neural networks (GNNs) have impacted protein structure prediction, inverse folding, protein design, and small molecule design.  

[end paper 73]

[begin paper 74]

PMCID = 7799268

Title = The impact of structural bioinformatics tools and resources on SARS-CoV-2 research and therapeutic strategies

SARS-CoV-2 is the causative agent of COVID-19, the ongoing global pandemic. It has posed a worldwide challenge to human health as no effective treatment is currently available to combat the disease. Its severity has led to unprecedented collaborative initiatives for therapeutic solutions against COVID-19. Studies resorting to structure-based drug design for COVID-19 are plethoric and show good promise. Structural biology provides key insights into 3D structures, critical residues/mutations in SARS-CoV-2 proteins, implicated in infectivity, molecular recognition and susceptibility to a broad range of host species. The detailed understanding of viral proteins and their complexes with host receptors and candidate epitope/lead compounds is the key to developing a structure-guided therapeutic design.

Since the discovery of SARS-CoV-2, several structures of its proteins have been determined experimentally at an unprecedented speed and deposited in the Protein Data Bank. Further, specialized structural bioinformatics tools and resources have been developed for theoretical models, data on protein dynamics from computer simulations, impact of variants/mutations and molecular therapeutics.

Here, we provide an overview of ongoing efforts on developing structural bioinformatics tools and resources for COVID-19 research. We also discuss the impact of these resources and structure-based studies, to understand various aspects of SARS-CoV-2 infection and therapeutic development. These include (i) understanding differences between SARS-CoV-2 and SARS-CoV, leading to increased infectivity of SARS-CoV-2, (ii) deciphering key residues in the SARS-CoV-2 involved in receptor–antibody recognition, (iii) analysis of variants in host proteins that affect host susceptibility to infection and (iv) analyses facilitating structure-based drug and vaccine design against SARS-CoV-2.

[end paper 74]

[begin paper 75]

PMCID = 7240986

Title = Gene expression of functionally-related genes coevolves across fungal species: detecting coevolution of gene expression using phylogenetic comparative methods

Researchers often measure changes in gene expression across conditions to better understand the shared functional roles and regulatory mechanisms of different genes. Analogous to this is comparing gene expression across species, which can improve our understanding of the evolutionary processes shaping the evolution of both individual genes and functional pathways. One area of interest is determining genes showing signals of coevolution, which can also indicate potential functional similarity, analogous to co-expression analysis often performed across conditions for a single species. However, as with any trait, comparing gene expression across species can be confounded by the non-independence of species due to shared ancestry, making standard hypothesis testing inappropriate.

We compared RNA-Seq data across 18 fungal species using a multivariate Brownian Motion phylogenetic comparative method (PCM), which allowed us to quantify coevolution between protein pairs while directly accounting for the shared ancestry of the species. Our work indicates proteins which physically-interact show stronger signals of coevolution than randomly-generated pairs. Interactions with stronger empirical and computational evidence also showing stronger signals of coevolution. We examined the effects of number of protein interactions and gene expression levels on coevolution, finding both factors are overall poor predictors of the strength of coevolution between a protein pair. Simulations further demonstrate the potential issues of analyzing gene expression coevolution without accounting for shared ancestry in a standard hypothesis testing framework. Furthermore, our simulations indicate the use of a randomly-generated null distribution as a means of determining statistical significance for detecting coevolving genes with phylogenetically-uncorrected correlations, as has previously been done, is less accurate than PCMs, although is a significant improvement over standard hypothesis testing. These methods are further improved by using a phylogenetically-corrected correlation metric.

Our work highlights potential benefits of using PCMs to detect gene expression coevolution from high-throughput omics scale data. This framework can be built upon to investigate other evolutionary hypotheses, such as changes in transcription regulatory mechanisms across species.

[end paper 75]

[begin paper 76]

PMCID = 11247410

Title = A survey of generative AI for de novo drug design: new frontiers in molecule and protein generation

Artificial intelligence (AI)-driven methods can vastly improve the historically costly drug design process, with various generative models already in widespread use. Generative models for de novo drug design, in particular, focus on the creation of novel biological compounds entirely from scratch, representing a promising future direction. Rapid development in the field, combined with the inherent complexity of the drug design process, creates a difficult landscape for new researchers to enter. In this survey, we organize de novo drug design into two overarching themes: small molecule and protein generation. Within each theme, we identify a variety of subtasks and applications, highlighting important datasets, benchmarks, and model architectures and comparing the performance of top models. We take a broad approach to AI-driven drug design, allowing for both micro-level comparisons of various methods within each subtask and macro-level observations across different fields. We discuss parallel challenges and approaches between the two applications and highlight future directions for AI-driven de novo drug design as a whole. An organized repository of all covered sources is available at https://github.com/gersteinlab/GenAI4Drug.

[end paper 76]

[begin paper 77]

PMCID = 9886305

Title = End-to-end sequence-structure-function meta-learning predicts genome-wide chemical-protein interactions for dark proteins

Systematically discovering protein-ligand interactions across the entire human and pathogen genomes is critical in chemical genomics, protein function prediction, drug discovery, and many other areas. However, more than 90% of gene families remain “dark”—i.e., their small-molecule ligands are undiscovered due to experimental limitations or human/historical biases. Existing computational approaches typically fail when the dark protein differs from those with known ligands. To address this challenge, we have developed a deep learning framework, called PortalCG, which consists of four novel components: (i) a 3-dimensional ligand binding site enhanced sequence pre-training strategy to encode the evolutionary links between ligand-binding sites across gene families; (ii) an end-to-end pretraining-fine-tuning strategy to reduce the impact of inaccuracy of predicted structures on function predictions by recognizing the sequence-structure-function paradigm; (iii) a new out-of-cluster meta-learning algorithm that extracts and accumulates information learned from predicting ligands of distinct gene families (meta-data) and applies the meta-data to a dark gene family; and (iv) a stress model selection step, using different gene families in the test data from those in the training and development data sets to facilitate model deployment in a real-world scenario. In extensive and rigorous benchmark experiments, PortalCG considerably outperformed state-of-the-art techniques of machine learning and protein-ligand docking when applied to dark gene families, and demonstrated its generalization power for target identifications and compound screenings under out-of-distribution (OOD) scenarios. Furthermore, in an external validation for the multi-target compound screening, the performance of PortalCG surpassed the rational design from medicinal chemists. Our results also suggest that a differentiable sequence-structure-function deep learning framework, where protein structural information serves as an intermediate layer, could be superior to conventional methodology where predicted protein structures were used for the compound screening. We applied PortalCG to two case studies to exemplify its potential in drug discovery: designing selective dual-antagonists of dopamine receptors for the treatment of opioid use disorder (OUD), and illuminating the understudied human genome for target diseases that do not yet have effective and safe therapeutics. Our results suggested that PortalCG is a viable solution to the OOD problem in exploring understudied regions of protein functional space.

Many complex diseases, such as Alzheimer’s disease, mental disorders, and substance use disorders, do not have safe and effective therapeutics because of the polygenic nature of the diseases and a lack of thoroughly validated drug targets (and their corresponding ligands). Identifying small-molecule ligands for all proteins encoded in the human genome would provide powerful new opportunities for drug discovery of currently untreatable diseases. However, the small-molecule ligand of more than 90% of gene families is completely unknown. Existing protein-ligand docking and machine learning methods often fail when the protein of interest is dissimilar to those with known functions or structures. We have developed a new deep learning framework, PortalCG, for efficiently and accurately predicting ligands of understudied proteins which are out of reach of existing methods. Our method achieves unprecedented accuracy versus state-of-the-art approaches, and it achieves this by incorporating ligand binding site information and the sequence-to-structure-to-function paradigm into a novel deep meta-learning algorithm. In a case study, the performance of PortalCG surpassed the rational design from medicinal chemists. The proposed computational framework can shed new light on how chemicals modulate biological systems, which is indispensable in drug repurposing and rational design of polypharmacology. This approach could offer a new way to develop safe and effective therapeutics for currently incurable diseases. PortalCG can be extended to other types of tasks, such as predicting protein-protein interactions and protein-nucleic acid recognition.

[end paper 77]

[begin paper 78]

PMCID = 6788782

Title = Chemical shift-based methods in NMR structure determination

Chemical shifts are highly sensitive probes harnessed by NMR spectroscopists and structural biologists as conformational parameters to characterize a range of biological molecules. Traditionally, assignment of chemical shifts has been a labor-intensive process requiring numerous samples and a suite of multidimensional experiments. Over the past two decades, the development of complementary computational approaches has bolstered the analysis, interpretation and utilization of chemical shifts for elucidation of high resolution protein and nucleic acid structures. Here, we review the development and application of chemical shift-based methods for structure determination with a focus on ab initio fragment assembly, comparative modeling, oligomeric systems, and automated assignment methods. Throughout our discussion, we point out practical uses, as well as advantages and caveats, of using chemical shifts in structure modeling. We additionally highlight (i) hybrid methods that employ chemical shifts with other types of NMR restraints (residual dipolar couplings, paramagnetic relaxation enhancements and pseudocontact shifts) that allow for improved accuracy and resolution of generated 3D structures, (ii) the utilization of chemical shifts to model the structures of sparsely populated excited states, and (iii) modeling of side-chain conformations. Finally, we briefly discuss the advantages of contemporary methods that employ sparse NMR data recorded using site-specific isotope labeling schemes for chemical shift-driven structure determination of larger molecules. With this review, we aim to emphasize the accessibility and versatility of chemical shifts for structure determination of challenging biological systems, and to point out emerging areas of development that lead us towards the next generation of tools.

[end paper 78]

[begin paper 79]

PMCID = 7032724

Title = Multi-state design of flexible proteins predicts sequences optimal for conformational change

Computational protein design of an ensemble of conformations for one protein–i.e., multi-state design–determines the side chain identity by optimizing the energetic contributions of that side chain in each of the backbone conformations. Sampling the resulting large sequence-structure search space limits the number of conformations and the size of proteins in multi-state design algorithms. Here, we demonstrated that the REstrained CONvergence (RECON) algorithm can simultaneously evaluate the sequence of large proteins that undergo substantial conformational changes. Simultaneous optimization of side chain conformations across all conformations increased sequence conservation when compared to single-state designs in all cases. More importantly, the sequence space sampled by RECON MSD resembled the evolutionary sequence space of flexible proteins, particularly when confined to predicting the mutational preferences of limited common ancestral descent, such as in the case of influenza type A hemagglutinin. Additionally, we found that sequence positions which require substantial changes in their local environment across an ensemble of conformations are more likely to be conserved. These increased conservation rates are better captured by RECON MSD over multiple conformations and thus multiple local residue environments during design. To quantify this rewiring of contacts at a certain position in sequence and structure, we introduced a new metric designated ‘contact proximity deviation’ that enumerates contact map changes. This measure allows mapping of global conformational changes into local side chain proximity adjustments, a property not captured by traditional global similarity metrics such as RMSD or local similarity metrics such as changes in φ and ψ angles.

Multi-state design can be used to engineer proteins that need to exist in multiple conformations or that bind to multiple partner molecules. In essence, multi-state design selects a compromise of protein sequences that allow for an ensemble of protein conformations, or states, associated with a particular biological function. In this paper, we used the REstrained CONvergence (RECON) algorithm with Rosetta to show that multi-state design of flexible proteins predicts sequences optimal for conformational change, mimicking mutation preferences sampled in evolution. Modeling optimal local side chain physicochemical environments within an ensemble selected significantly more native-like sequences than selections performed when all conformations states are designed independently. This outcome was particularly true for amino acids whose local side chain environment change between conformations. To quantify such contact map changes, we introduced a novel metric to show that sequence conservation is dependent on protein flexibility, i.e., changes in local side chain environments between stated limit the space of tolerated mutations. Additionally, such positions in sequence and structure are more likely to be energetically frustrated, at least in some states. Importantly, we showed that multi-state design over an ensemble of conformations (space) can explore evolutionary tolerated sequence space (time), thus enabling RECON to not only design proteins that require multiple states for function but also predict mutations that might be tolerated in native proteins but have not yet been explored by evolution. The latter aspect can be important to anticipate escape mutations, for example in pathogens or oncoproteins.

[end paper 79]

[begin paper 80]

PMCID = 7489204

Title = The ranging of amino acids substitution matrices of various types in accordance with the alignment accuracy criterion

The alignment of character sequences is important in bioinformatics. The quality of this procedure is determined by the substitution matrix and parameters of the insertion-deletion penalty function. These matrices are derived from sequence alignment and thus reflect the evolutionary process. Currently, in addition to evolutionary matrices, a large number of different background matrices have been obtained. To make an optimal choice of the substitution matrix and the penalty parameters, we conducted a numerical experiment using a representative sample of existing matrices of various types and origins.

We tested both the classical evolutionary matrix series (PAM, Blosum, VTML, Pfasum); structural alignment based matrices, contact energy matrix, and matrix based on the properties of the genetic code. This study presents results for two test set types: first, we simulated sequences that reflect the divergent evolution; second, we performed tests on Balibase sequences. In both cases, we obtained the dependences of the alignment quality (Accuracy, Confidence) on the evolutionary distance between sequences and the evolutionary distance to which the substitution matrices correspond. Optimization of a combination of matrices and the penalty parameters was carried out for local and global alignment on the values of penalty function parameters.

Consequently, we found that the best alignment quality is achieved with matrices corresponding to the largest evolutionary distance. These matrices prove to be universal, i.e. suitable for aligning sequences separated by both large and small evolutionary distances. We analysed the correspondence of the correlation coefficients of matrices to the alignment quality. It was found that matrices showing high quality alignment have an above average correlation value, but the converse is not true.

This study showed that the best alignment quality is achieved with evolutionary matrices designed for long distances: Gonnet, VTML250, PAM250, MIQS, and Pfasum050. The same property is inherent in matrices not only of evolutionary origin, but also of another background corresponding to a large evolutionary distance. Therefore, matrices based on structural data show alignment quality close enough to its value for evolutionary matrices. This agrees with the idea that the spatial structure is more conservative than the protein sequence.

[end paper 80]

[begin paper 81]

PMCID = 9429973

Title = Abstract Book for the 27th Congress of the European Hematology Association



[end paper 81]

[begin paper 82]

PMCID = 8065224

Title = Recent advances in de novo protein design: Principles, methods, and applications

The computational de novo protein design is increasingly applied to address a number of key challenges in biomedicine and biological engineering. Successes in expanding applications are driven by advances in design principles and methods over several decades. Here, we review recent innovations in major aspects of the de novo protein design and include how these advances were informed by principles of protein architecture and interactions derived from the wealth of structures in the Protein Data Bank. We describe developments in de novo generation of designable backbone structures, optimization of sequences, design scoring functions, and the design of the function. The advances not only highlight design goals reachable now but also point to the challenges and opportunities for the future of the field.

[end paper 82]

[begin paper 83]

PMCID = 10096462

Title = chiLife: An open-source Python package for in silico spin labeling and integrative protein modeling

Here we introduce chiLife, a Python package for site-directed spin label (SDSL) modeling for electron paramagnetic resonance (EPR) spectroscopy, in particular double electron–electron resonance (DEER). It is based on in silico attachment of rotamer ensemble representations of spin labels to protein structures. chiLife enables the development of custom protein analysis and modeling pipelines using SDSL EPR experimental data. It allows the user to add custom spin labels, scoring functions and spin label modeling methods. chiLife is designed with integration into third-party software in mind, to take advantage of the diverse and rapidly expanding set of molecular modeling tools available with a Python interface. This article describes the main design principles of chiLife and presents a series of examples.

Thanks to modern modeling methods like AlphaFold2, RosettaFold, and ESMFold, high-resolution structural models of proteins are widely available. While these models offer insight into the structure and function of biomedically and technologically significant proteins, most of them are not experimentally validated. Furthermore, many proteins exhibit functionally important conformational flexibility that is not captured by these models. Site-directed spin labeling (SDSL) electron paramagnetic resonance (EPR) spectroscopy is a powerful tool for probing protein structure and conformational heterogeneity, making it ideal for validating, refining, and expanding protein models. To extract quantitative protein backbone information from experimental SDSL EPR data, accurate modeling methods are needed. For this purpose, we introduce chiLife, an open-source Python package for SDSL modeling designed to be extensible and integrable with other Python-based protein modeling packages. With chiLife, appropriate SDSL EPR experiments for protein model validation can be designed, and protein models can be refined using experimental SDSL EPR data as constraints.

[end paper 83]

[begin paper 84]

PMCID = 4777540

Title = Selection on Coding and Regulatory Variation Maintains Individuality in Major Urinary Protein Scent Marks in Wild Mice

Recognition of individuals by scent is widespread across animal taxa. Though animals can often discriminate chemical blends based on many compounds, recent work shows that specific protein pheromones are necessary and sufficient for individual recognition via scent marks in mice. The genetic nature of individuality in scent marks (e.g. coding versus regulatory variation) and the evolutionary processes that maintain diversity are poorly understood. The individual signatures in scent marks of house mice are the protein products of a group of highly similar paralogs in the major urinary protein (Mup) gene family. Using the offspring of wild-caught mice, we examine individuality in the major urinary protein (MUP) scent marks at the DNA, RNA and protein levels. We show that individuality arises through a combination of variation at amino acid coding sites and differential transcription of central Mup genes across individuals, and we identify eSNPs in promoters. There is no evidence of post-transcriptional processes influencing phenotypic diversity as transcripts accurately predict the relative abundance of proteins in urine samples. The match between transcripts and urine samples taken six months earlier also emphasizes that the proportional relationships across central MUP isoforms in urine is stable. Balancing selection maintains coding variants at moderate frequencies, though pheromone diversity appears limited by interactions with vomeronasal receptors. We find that differential transcription of the central Mup paralogs within and between individuals significantly increases the individuality of pheromone blends. Balancing selection on gene regulation allows for increased individuality via combinatorial diversity in a limited number of pheromones.

Individual recognition via scent is critical for many aspects of behavior including parental care, competition, cooperation and mate choice. While animal scents can differ in a huge number of dimensions, recent work has shown that only some specialized semiochemicals in scent marks are behaviorally relevant for individual recognition. How is individuality in specialized semiochemical blends produced and maintained in populations? At the extremes, individuality may depend on either a plethora of semiochemical isoforms or on combinatorial variation in a small number of shared isoforms across individuals. Analyzing the major urinary protein (MUP) pheromone blends of a wild population of house mice, we find evidence in favor of a combinatorial diversity model for the production and maintenance of individuality. Balancing selection maintains MUP proteins at moderate frequencies in the population, though interactions with the pheromone receptors appear to limit the extent of pheromone diversity in the system. By contrast, differential transcription of proteins greatly increases individuality in pheromone blends with balancing selection maintaining diversity in promoter regions associated with gene expression patterns. Selection maintaining combinatorial diversity in a limited set of behaviorally important semiochemicals may be a widespread mechanism generating and maintaining individuality in scent across taxa.

[end paper 84]

[begin paper 85]

PMCID = 10821953

Title = The origin and structural evolution of de novo genes in Drosophila

Recent studies reveal that de novo gene origination from previously non-genic sequences is a common mechanism for gene innovation. These young genes provide an opportunity to study the structural and functional origins of proteins. Here, we combine high-quality base-level whole-genome alignments and computational structural modeling to study the origination, evolution, and protein structures of lineage-specific de novo genes. We identify 555 de novo gene candidates in D. melanogaster that originated within the Drosophilinae lineage. Sequence composition, evolutionary rates, and expression patterns indicate possible gradual functional or adaptive shifts with their gene ages. Surprisingly, we find little overall protein structural changes in candidates from the Drosophilinae lineage. We identify several candidates with potentially well-folded protein structures. Ancestral sequence reconstruction analysis reveals that most potentially well-folded candidates are often born well-folded. Single-cell RNA-seq analysis in testis shows that although most de novo gene candidates are enriched in spermatocytes, several young candidates are biased towards the early spermatogenesis stage, indicating potentially important but less emphasized roles of early germline cells in the de novo gene origination in testis. This study provides a systematic overview of the origin, evolution, and protein structural changes of Drosophilinae-specific de novo genes.

It is unclear whether naturally evolved de novo proteins have stable, folded structures. Here, through systematic identification and structural modeling of de novo genes, this study reveals that a small subset of these proteins may have well-folded structures, and were likely born with these structures.

[end paper 85]

[begin paper 86]

PMCID = 11297950

Title = Accurate prediction of protein function using statistics-informed graph networks

Understanding protein function is pivotal in comprehending the intricate mechanisms that underlie many crucial biological activities, with far-reaching implications in the fields of medicine, biotechnology, and drug development. However, more than 200 million proteins remain uncharacterized, and computational efforts heavily rely on protein structural information to predict annotations of varying quality. Here, we present a method that utilizes statistics-informed graph networks to predict protein functions solely from its sequence. Our method inherently characterizes evolutionary signatures, allowing for a quantitative assessment of the significance of residues that carry out specific functions. PhiGnet not only demonstrates superior performance compared to alternative approaches but also narrows the sequence-function gap, even in the absence of structural information. Our findings indicate that applying deep learning to evolutionary data can highlight functional sites at the residue level, providing valuable support for interpreting both existing properties and new functionalities of proteins in research and biomedicine.

Understanding protein function is vital for biomedicine. Here, authors develop a method using statistics-informed graph networks to predict functions from sequences. The method integrates evolutionary couplings and residue communities to improve the accuracy of function annotations for proteins.

[end paper 86]

[begin paper 87]

PMCID = 11149777

Title = The Rapid Evolution of De Novo Proteins in Structure and Complex

Recent studies in the rice genome-wide have established that de novo genes, evolving from noncoding sequences, enhance protein diversity through a stepwise process. However, the pattern and rate of their evolution in protein structure over time remain unclear. Here, we addressed these issues within a surprisingly short evolutionary timescale (<1 million years for 97% of Oryza de novo genes) with comparative approaches to gene duplicates. We found that de novo genes evolve faster than gene duplicates in the intrinsically disordered regions (such as random coils), secondary structure elements (such as α helix and β strand), hydrophobicity, and molecular recognition features. In de novo proteins, specifically, we observed an 8% to 14% decay in random coils and intrinsically disordered region lengths and a 2.3% to 6.5% increase in structured elements, hydrophobicity, and molecular recognition features, per million years on average. These patterns of structural evolution align with changes in amino acid composition over time as well. We also revealed higher positive charges but smaller molecular weights for de novo proteins than duplicates. Tertiary structure predictions showed that most de novo proteins, though not typically well folded on their own, readily form low-energy and compact complexes with other proteins facilitated by extensive residue contacts and conformational flexibility, suggesting a faster-binding scenario in de novo proteins to promote interaction. These analyses illuminate a rapid evolution of protein structure in de novo genes in rice genomes, originating from noncoding sequences, highlighting their quick transformation into active, protein complex-forming components within a remarkably short evolutionary timeframe.

[end paper 87]

[begin paper 88]

PMCID = 11353971

Title = Advances in the Application of Protein Language Modeling for Nucleic Acid Protein Binding Site Prediction

Protein and nucleic acid binding site prediction is a critical computational task that benefits a wide range of biological processes. Previous studies have shown that feature selection holds particular significance for this prediction task, making the generation of more discriminative features a key area of interest for many researchers. Recent progress has shown the power of protein language models in handling protein sequences, in leveraging the strengths of attention networks, and in successful applications to tasks such as protein structure prediction. This naturally raises the question of the applicability of protein language models in predicting protein and nucleic acid binding sites. Various approaches have explored this potential. This paper first describes the development of protein language models. Then, a systematic review of the latest methods for predicting protein and nucleic acid binding sites is conducted by covering benchmark sets, feature generation methods, performance comparisons, and feature ablation studies. These comparisons demonstrate the importance of protein language models for the prediction task. Finally, the paper discusses the challenges of protein and nucleic acid binding site prediction and proposes possible research directions and future trends. The purpose of this survey is to furnish researchers with actionable suggestions for comprehending the methodologies used in predicting protein–nucleic acid binding sites, fostering the creation of protein-centric language models, and tackling real-world obstacles encountered in this field.

[end paper 88]

[begin paper 89]

PMCID = 8616814

Title = Prediction of protein assemblies, the next frontier: The CASP14-CAPRI experiment

We present the results for CAPRI Round 50, the 4th joint CASP-CAPRI protein assembly prediction challenge. The Round comprised a total of 12 targets, including 6 dimers, 3 trimers, and 3 higher-order oligomers. Four of these were easy targets, for which good structural templates were available either for the full assembly, or for the main interfaces (of the higher-order oligomers). Eight were difficult targets for which only distantly related templates were found for the individual subunits. Twenty-five CAPRI groups including 8 automatic servers submitted ~1250 models per target. Twenty groups including 6 servers participated in the CAPRI scoring challenge submitted ~190 models per target. The accuracy of the predicted models was evaluated using the classical CAPRI criteria. The prediction performance was measured by a weighted scoring scheme that takes into account the number of models of acceptable quality or higher submitted by each group as part of their 5 top-ranking models. Compared to the previous CASP-CAPRI challenge, top performing groups submitted such models for a larger fraction (70–75%) of the targets in this Round, but fewer of these models were of high accuracy. Scorer groups achieved stronger performance with more groups submitting correct models for 70–80% of the targets or achieving high accuracy predictions. Servers performed less well in general, except for the MDOCKPP and LZERD servers, who performed on par with human groups. In addition to these results, major advances in methodology are discussed, providing an informative overview of where the prediction of protein assemblies currently stands.

[end paper 89]

[begin paper 90]

PMCID = 5030136

Title = Prediction of homoprotein and heteroprotein complexes by protein docking and template‐based modeling: A CASP‐CAPRI experiment

We present the results for CAPRI Round 30, the first joint CASP‐CAPRI experiment, which brought together experts from the protein structure prediction and protein–protein docking communities. The Round comprised 25 targets from amongst those submitted for the CASP11 prediction experiment of 2014. The targets included mostly homodimers, a few homotetramers, and two heterodimers, and comprised protein chains that could readily be modeled using templates from the Protein Data Bank. On average 24 CAPRI groups and 7 CASP groups submitted docking predictions for each target, and 12 CAPRI groups per target participated in the CAPRI scoring experiment. In total more than 9500 models were assessed against the 3D structures of the corresponding target complexes. Results show that the prediction of homodimer assemblies by homology modeling techniques and docking calculations is quite successful for targets featuring large enough subunit interfaces to represent stable associations. Targets with ambiguous or inaccurate oligomeric state assignments, often featuring crystal contact‐sized interfaces, represented a confounding factor. For those, a much poorer prediction performance was achieved, while nonetheless often providing helpful clues on the correct oligomeric state of the protein. The prediction performance was very poor for genuine tetrameric targets, where the inaccuracy of the homology‐built subunit models and the smaller pair‐wise interfaces severely limited the ability to derive the correct assembly mode. Our analysis also shows that docking procedures tend to perform better than standard homology modeling techniques and that highly accurate models of the protein components are not always required to identify their association modes with acceptable accuracy. Proteins 2016; 84(Suppl 1):323–348. © 2016 The Authors Proteins: Structure, Function, and Bioinformatics Published by Wiley Periodicals, Inc.

[end paper 90]

[begin paper 91]

PMCID = 11127006

Title = The physical and evolutionary energy landscapes of devolved protein sequences corresponding to pseudogenes

Pseudogenes are DNA sequences that previously encoded protein sequences but are no longer translated due to sequence degradation. As pseudogenes no longer experience selection pressure to fold into functional proteins stably, they serve as unique examples of naturally occurring protein devolution. We surveyed pseudogenes belonging to protein families of varying length, architecture, and biological function using coevolutionary and optimized physical models. We found the mutations found in pseudogenes typically destabilize their former protein structure. Some mutations that would stabilize pseudogenes in their former structure are found to inhibit or alter their previous biological function.

Protein evolution is guided by structural, functional, and dynamical constraints ensuring organismal viability. Pseudogenes are genomic sequences identified in many eukaryotes that lack translational activity due to sequence degradation and thus over time have undergone “devolution.” Previously pseudogenized genes sometimes regain their protein-coding function, suggesting they may still encode robust folding energy landscapes despite multiple mutations. We study both the physical folding landscapes of protein sequences corresponding to human pseudogenes using the Associative Memory, Water Mediated, Structure and Energy Model, and the evolutionary energy landscapes obtained using direct coupling analysis (DCA) on their parent protein families. We found that generally mutations that have occurred in pseudogene sequences have disrupted their native global network of stabilizing residue interactions, making it harder for them to fold if they were translated. In some cases, however, energetic frustration has apparently decreased when the functional constraints were removed. We analyzed this unexpected situation for Cyclophilin A, Profilin-1, and Small Ubiquitin-like Modifier 2 Protein. Our analysis reveals that when such mutations in the pseudogene ultimately stabilize folding, at the same time, they likely alter the pseudogenes’ former biological activity, as estimated by DCA. We localize most of these stabilizing mutations generally to normally frustrated regions required for binding to other partners.

[end paper 91]

[begin paper 92]

PMCID = 8745595

Title = Is Protein Folding a Thermodynamically Unfavorable, Active, Energy-Dependent Process?

The prevailing current view of protein folding is the thermodynamic hypothesis, under which the native folded conformation of a protein corresponds to the global minimum of Gibbs free energy G. We question this concept and show that the empirical evidence behind the thermodynamic hypothesis of folding is far from strong. Furthermore, physical theory-based approaches to the prediction of protein folds and their folding pathways so far have invariably failed except for some very small proteins, despite decades of intensive theory development and the enormous increase of computer power. The recent spectacular successes in protein structure prediction owe to evolutionary modeling of amino acid sequence substitutions enhanced by deep learning methods, but even these breakthroughs provide no information on the protein folding mechanisms and pathways. We discuss an alternative view of protein folding, under which the native state of most proteins does not occupy the global free energy minimum, but rather, a local minimum on a fluctuating free energy landscape. We further argue that ΔG of folding is likely to be positive for the majority of proteins, which therefore fold into their native conformations only through interactions with the energy-dependent molecular machinery of living cells, in particular, the translation system and chaperones. Accordingly, protein folding should be modeled as it occurs in vivo, that is, as a non-equilibrium, active, energy-dependent process.

[end paper 92]

[begin paper 93]

PMCID = 7108332

Title = FEDERATING STRUCTURAL MODELS AND DATA: OUTCOMES FROM A WORKSHOP ON ARCHIVING INTEGRATIVE STRUCTURES

Structures of biomolecular systems are increasingly computed by integrative modeling. In this approach, a structural model is constructed by combining information from multiple sources, including varied experimental methods and prior models. In 2019, a Workshop was held as a Biophysical Society Satellite Meeting to assess progress and discuss further requirements for archiving integrative structures. The primary goal of the Workshop was to build consensus for addressing the challenges involved in creating common data standards, building methods for federated data exchange, and developing mechanisms for validating integrative structures. The summary of the Workshop and the recommendations that emerged are presented here.

[end paper 93]

[begin paper 94]

PMCID = 6812854

Title = MANET 3.0: Hierarchy and modularity in evolving metabolic networks

Enzyme recruitment is a fundamental evolutionary driver of modern metabolism. We see evidence of recruitment at work in the metabolic Molecular Ancestry Networks (MANET) database, an online resource that integrates data from KEGG, SCOP and structural phylogenomic reconstruction. The database, which was introduced in 2006, traces the deep history of the structural domains of enzymes in metabolic pathways. Here we release version 3.0 of MANET, which updates data from KEGG and SCOP, links enzyme and PDB information with PDBsum, and traces evolutionary information of domains defined at fold family level of SCOP classification in metabolic subnetwork diagrams. Compared to SCOP folds used in the previous versions, fold families are cohesive units of functional similarity that are highly conserved at sequence level and offer a 10-fold increase of data entries. We surveyed enzymatic, functional and catalytic site distributions among superkingdoms showing that ancient enzymatic innovations followed a biphasic temporal pattern of diversification typical of module innovation. We grouped enzymatic activities of MANET into a hierarchical system of subnetworks and mesonetworks matching KEGG classification. The evolutionary growth of these modules of metabolic activity was studied using bipartite networks and their one-mode projections at enzyme, subnetwork and mesonetwork levels of organization. Evolving metabolic networks revealed patterns of enzyme sharing that transcended mesonetwork boundaries and supported the patchwork model of metabolic evolution. We also explored the scale-freeness, randomness and small-world properties of evolving networks as possible organizing principles of network growth and diversification. The network structure shows an increase in hierarchical modularity and scale-free behavior as metabolic networks unfold in evolutionary time. Remarkably, this evolutionary constraint on structure was stronger at lower levels of metabolic organization. Evolving metabolic structure reveals a ‘principle of granularity’, an evolutionary increase of the cohesiveness of lower-level parts of a hierarchical system. MANET is available at http://manet.illinois.edu.

[end paper 94]

[begin paper 95]

PMCID = 9399266

Title = Prediction and validation of host-pathogen interactions by a versatile inference approach using Aspergillus fumigatus as a case study

Biological networks are characterized by diverse interactions and dynamics in time and space. Many regulatory modules operate in parallel and are interconnected with each other. Some pathways are functionally known and annotated accordingly, e.g., endocytosis, migration, or cytoskeletal rearrangement. However, many interactions are not so well characterized. For reconstructing the biological complexity in cellular networks, we combine here existing experimentally confirmed and analyzed interactions with a protein-interaction inference framework using as basis experimentally confirmed interactions from other organisms. Prediction scoring includes sequence similarity, evolutionary conservation of interactions, the coexistence of interactions in the same pathway, orthology as well as structure similarity to rank and compare inferred interactions. We exemplify our inference method by studying host-pathogen interactions during infection of Mus musculus (phagolysosomes in alveolar macrophages) with Aspergillus fumigatus (conidia, airborne, asexual spores). Three of nine predicted critical host-pathogen interactions could even be confirmed by direct experiments. Moreover, we suggest drugs that manipulate the host-pathogen interaction.

[end paper 95]

[begin paper 96]

PMCID = 10010530

Title = Inferring protein fitness landscapes from laboratory evolution experiments

Directed laboratory evolution applies iterative rounds of mutation and selection to explore the protein fitness landscape and provides rich information regarding the underlying relationships between protein sequence, structure, and function. Laboratory evolution data consist of protein sequences sampled from evolving populations over multiple generations and this data type does not fit into established supervised and unsupervised machine learning approaches. We develop a statistical learning framework that models the evolutionary process and can infer the protein fitness landscape from multiple snapshots along an evolutionary trajectory. We apply our modeling approach to dihydrofolate reductase (DHFR) laboratory evolution data and the resulting landscape parameters capture important aspects of DHFR structure and function. We use the resulting model to understand the structure of the fitness landscape and find numerous examples of epistasis but an overall global peak that is evolutionarily accessible from most starting sequences. Finally, we use the model to perform an in silico extrapolation of the DHFR laboratory evolution trajectory and computationally design proteins from future evolutionary rounds.

Laboratory evolution has revolutionized our understanding of protein structure, function, and evolution, and has generated countless useful proteins broad applications in medicine, biocatalysis, and biotechnology. These experiments explore protein sequence space through iterative rounds of mutation and selection and can provide rich data of populations traversing the fitness landscape. In this paper, we present a statistical learning framework that models the evolutionary process and can infer the structure of the underlying protein fitness landscape from multiple snapshots along a laboratory evolution trajectory. We generate a dihydrofolate reductase (DHFR) laboratory evolution data set and apply our modeling approach to infer the landscape parameters. The estimated parameters pinpoint key residues that dictate DHFR structure and function. We use the resulting model to understand the local and global structure of the fitness landscape and to perform in silico directed evolution for protein engineering.

[end paper 96]

[begin paper 97]

PMCID = 10928435

Title = Protein Structure Prediction: Challenges, Advances, and the Shift of Research Paradigms

Protein structure prediction is an interdisciplinary research topic that has attracted researchers from multiple fields, including biochemistry, medicine, physics, mathematics, and computer science. These researchers adopt various research paradigms to attack the same structure prediction problem: biochemists and physicists attempt to reveal the principles governing protein folding; mathematicians, especially statisticians, usually start from assuming a probability distribution of protein structures given a target sequence and then find the most likely structure, while computer scientists formulate protein structure prediction as an optimization problem — finding the structural conformation with the lowest energy or minimizing the difference between predicted structure and native structure. These research paradigms fall into the two statistical modeling cultures proposed by Leo Breiman, namely, data modeling and algorithmic modeling. Recently, we have also witnessed the great success of deep learning in protein structure prediction. In this review, we present a survey of the efforts for protein structure prediction. We compare the research paradigms adopted by researchers from different fields, with an emphasis on the shift of research paradigms in the era of deep learning. In short, the algorithmic modeling techniques, especially deep neural networks, have considerably improved the accuracy of protein structure prediction; however, theories interpreting the neural networks and knowledge on protein folding are still highly desired.

[end paper 97]

[begin paper 98]

PMCID = 5576689

Title = Elastic network model of learned maintained contacts to predict protein motion

We present a novel elastic network model, lmcENM, to determine protein motion even for localized functional motions that involve substantial changes in the protein’s contact topology. Existing elastic network models assume that the contact topology remains unchanged throughout the motion and are thus most appropriate to simulate highly collective function-related movements. lmcENM uses machine learning to differentiate breaking from maintained contacts. We show that lmcENM accurately captures functional transitions unexplained by the classical ENM and three reference ENM variants, while preserving the simplicity of classical ENM. We demonstrate the effectiveness of our approach on a large set of proteins covering different motion types. Our results suggest that accurately predicting a “deformation-invariant” contact topology offers a promising route to increase the general applicability of ENMs. We also find that to correctly predict this contact topology a combination of several features seems to be relevant which may vary slightly depending on the protein. Additionally, we present case studies of two biologically interesting systems, Ferric Citrate membrane transporter FecA and Arachidonate 15-Lipoxygenase.

[end paper 98]

[begin paper 99]

PMCID = 5956850

Title = Functional insights from proteome-wide structural modeling of Treponema pallidum subspecies pallidum, the causative agent of syphilis

Syphilis continues to be a major global health threat with 11 million new infections each year, and a global burden of 36 million cases. The causative agent of syphilis, Treponema pallidum subspecies pallidum, is a highly virulent bacterium, however the molecular mechanisms underlying T. pallidum pathogenesis remain to be definitively identified. This is due to the fact that T. pallidum is currently uncultivatable, inherently fragile and thus difficult to work with, and phylogenetically distinct with no conventional virulence factor homologs found in other pathogens. In fact, approximately 30% of its predicted protein-coding genes have no known orthologs or assigned functions. Here we employed a structural bioinformatics approach using Phyre2-based tertiary structure modeling to improve our understanding of T. pallidum protein function on a proteome-wide scale.

Phyre2-based tertiary structure modeling generated high-confidence predictions for 80% of the T. pallidum proteome (780/978 predicted proteins). Tertiary structure modeling also inferred the same function as primary structure-based annotations from genome sequencing pipelines for 525/605 proteins (87%), which represents 54% (525/978) of all T. pallidum proteins. Of the 175 T. pallidum proteins modeled with high confidence that were not assigned functions in the previously annotated published proteome, 167 (95%) were able to be assigned predicted functions. Twenty-one of the 175 hypothetical proteins modeled with high confidence were also predicted to exhibit significant structural similarity with proteins experimentally confirmed to be required for virulence in other pathogens.

Phyre2-based structural modeling is a powerful bioinformatics tool that has provided insight into the potential structure and function of the majority of T. pallidum proteins and helped validate the primary structure-based annotation of more than 50% of all T. pallidum proteins with high confidence. This work represents the first T. pallidum proteome-wide structural modeling study and is one of few studies to apply this approach for the functional annotation of a whole proteome.

The online version of this article (10.1186/s12900-018-0086-3) contains supplementary material, which is available to authorized users.

[end paper 99]

[begin paper 100]

PMCID = 7613228

Title = Low protein expression enhances phenotypic evolvability by intensifying selection on folding stability

Protein abundance affects the evolution of protein genotypes, but we do not know how it affects the evolution of protein phenotypes. Here we investigate the role of protein abundance on the evolvability of green fluorescent protein (GFP) towards the novel phenotype of cyan fluorescence. We evolve GFP in E.coli through multiple cycles of mutation and selection, and show that low GFP expression facilitates the evolution of cyan fluorescence. A computational model whose predictions we test experimentally helps explain why: Lowly expressed proteins are under stronger selection for proper folding, which facilitates their evolvability on short evolutionary time scales. The reason is that high fluorescence can be achieved by either few proteins that fold well, or by many proteins that fold less well. In other words, we observe a synergy between a protein’s scarcity and its stability. Because many proteins meet the essential requirements for this scarcity-stability synergy, it may be a widespread mechanism by which low expression helps proteins evolve new phenotypes and functions.

[end paper 100]



