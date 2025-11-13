## User

This is the beginning of an individual meeting with Computational Biologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "Rosetta" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on Rosetta to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on Rosetta using the query "Rosetta spike protein binding prediction" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to Rosetta. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Computational Biologist, please provide your response to the agenda.

## Computational Biologist

Based on the abstracts provided, I have identified the following papers that are directly relevant to the use of Rosetta in the context of designing nanobody binders for SARS-CoV-2. Here are the PMCIDs and titles of the papers that I wish to fine-tune myself on:

```python
{
    "7995276": "Mutations in spike protein and allele variations in ACE2 impact targeted therapy strategies against SARS-CoV-2",
    "11031966": "Computational analysis of affinity dynamics between the variants of SARS-CoV-2 spike protein (RBD) and human ACE-2 receptor",
    "8967028": "Structure-based dual affinity optimization of a SARS-CoV-1/2 cross-reactive single-domain antibody",
    "11055898": "The hinge-engineered IgG1-IgG3 hybrid subclass IgGh47 potently enhances Fc-mediated function of anti-streptococcal and SARS-CoV-2 antibodies",
    "9936628": "Computational pipeline provides mechanistic understanding of Omicron variant of concern neutralizing engineered ACE2 receptor traps",
    "7500280": "Computational biophysical characterization of the SARS-CoV-2 spike protein binding with the ACE2 receptor and implications for infectivity",
    "9934699": "Massively parallel protein-protein interaction measurement by sequencing (MP3-seq) enables rapid screening of protein heterodimers",
    "11040732": "ALDELE: All-Purpose Deep Learning Toolkits for Predicting the Biocatalytic Activities of Enzymes",
    "10975102": "Production and Immunogenicity Assessment of LTp50: An Escherichia coli-Made Chimeric Antigen Targeting S1- and S2-Epitopes from the SARS-CoV-2/BA.5 Spike Protein",
    "10965422": "AbFlex: designing antibody complementarity determining regions with flexible CDR definition"
}
```

These papers were selected based on their relevance to Rosetta and its application in protein design, particularly in the context of SARS-CoV-2 spike protein interactions and antibody/nanobody design.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "Rosetta spike protein binding prediction":

[begin paper 1]

PMCID = 7334636

Title = Protein structure analysis of the interactions between SARS-CoV-2 spike protein and the human ACE2 receptor: from conformational changes to novel neutralizing antibodies

The recent severe acute respiratory syndrome, known as Coronavirus Disease 2019 (COVID-19) has spread so much rapidly and severely to induce World Health Organization (WHO) to declare a state of emergency over the new coronavirus SARS-CoV-2 pandemic. While several countries have chosen the almost complete lock-down for slowing down SARS-CoV-2 spread, the scientific community is called to respond to the devastating outbreak by identifying new tools for diagnosis and treatment of the dangerous COVID-19. With this aim, we performed an in silico comparative modeling analysis, which allows gaining new insights into the main conformational changes occurring in the SARS-CoV-2 spike protein, at the level of the receptor-binding domain (RBD), along interactions with human cells angiotensin-converting enzyme 2 (ACE2) receptor, that favor human cell invasion. Furthermore, our analysis provides (1) an ideal pipeline to identify already characterized antibodies that might target SARS-CoV-2 spike RBD, aiming to prevent interactions with the human ACE2, and (2) instructions for building new possible neutralizing antibodies, according to chemical/physical space restraints and complementary determining regions (CDR) mutagenesis of the identified existing antibodies. The proposed antibodies show in silico high affinity for SARS-CoV-2 spike RBD and can be used as reference antibodies also for building new high-affinity antibodies against present and future coronaviruses able to invade human cells through interactions of their spike proteins with the human ACE2. More in general, our analysis provides indications for the set-up of the right biological molecular context for investigating spike RBD–ACE2 interactions for the development of new vaccines, diagnostic kits, and other treatments based on the targeting of SARS-CoV-2 spike protein.

The online version of this article (10.1007/s00018-020-03580-1) contains supplementary material, which is available to authorized users.

[end paper 1]

[begin paper 2]

PMCID = 8796240

Title = Spike-Dependent Opsonization Indicates Both Dose-Dependent Inhibition of Phagocytosis and That Non-Neutralizing Antibodies Can Confer Protection to SARS-CoV-2

Spike-specific antibodies are central to effective COVID19 immunity. Research efforts have focused on antibodies that neutralize the ACE2-Spike interaction but not on non-neutralizing antibodies. Antibody-dependent phagocytosis is an immune mechanism enhanced by opsonization, where typically, more bound antibodies trigger a stronger phagocyte response. Here, we show that Spike-specific antibodies, dependent on concentration, can either enhance or reduce Spike-bead phagocytosis by monocytes independently of the antibody neutralization potential. Surprisingly, we find that both convalescent patient plasma and patient-derived monoclonal antibodies lead to maximum opsonization already at low levels of bound antibodies and is reduced as antibody binding to Spike protein increases. Moreover, we show that this Spike-dependent modulation of opsonization correlate with the outcome in an experimental SARS-CoV-2 infection model. These results suggest that the levels of anti-Spike antibodies could influence monocyte-mediated immune functions and propose that non-neutralizing antibodies could confer protection to SARS-CoV-2 infection by mediating phagocytosis.

[end paper 2]

[begin paper 3]

PMCID = 7995276

Title = Mutations in spike protein and allele variations in ACE2 impact targeted therapy strategies against SARS-CoV-2

Coronavirus disease 2019 (COVID-19), which is caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has spread rapidly worldwide with high rates of transmission and substantial mortality. To date, however, no effective treatments or enough vaccines for COVID-19 are available. The roles of angiotensin converting enzyme 2 (ACE2) and spike protein in the treatment of COVID-19 are major areas of research. In this study, we explored the potential of ACE2 and spike protein as targets for the development of antiviral agents against SARS-CoV-2. We analyzed clinical data, genetic data, and receptor binding capability. Clinical data revealed that COVID-19 patients with comorbidities related to an abnormal renin-angiotensin system exhibited more early symptoms and poorer prognoses. However, the relationship between ACE2 expression and COVID-19 progression is still not clear. Furthermore, if ACE2 is not a good targetable protein, it would not be applicable across a wide range of populations. The spike-S1 receptor-binding domain that interacts with ACE2 showed various amino acid mutations based on sequence analysis. We identified two spike-S1 point mutations (V354F and V470A) by receptor-ligand docking and binding enzyme-linked immunosorbent assays. These variants enhanced the binding of the spike protein to ACE2 receptors and were potentially associated with increased infectivity. Importantly, the number of patients infected with the V354F and V470A mutants has increased with the development of the SARS-CoV-2 pandemic. These results suggest that ACE2 and spike-S1 are likely not ideal targets for the design of peptide drugs to treat COVID-19 in different populations.

A: Comorbidities in COVID-19 deaths. B: Age distribution in three groups, i.e., G1 (COVID-19 patients with comorbidities associated with abnormal RAS, 75.00±2.57 (mean±SEM (standard error of the mean))), G2 (COVID-19 patients with comorbidities but normal RAS, 70.50±4.44), and G3 (COVID-19 patients without comorbidities, 65.29±3.18). For age distribution, P-values between G1 and G2, G1 and G3, and G2 and G3 were 0.79, 0.80, and 0.70, respectively. C: Days of symptoms until death (time from early symptoms to death in G1 (15.47±1.71), G2 (16.50±2.06), and G3 (15.43±2.20) groups). For days of symptoms until death, P-values between G1 and G2, G1 and G3, and G2 and G3 were 0.36, 0.67, and 0.28, respectively. D: Sex ratio in three groups (G1-male (0.68), G2-male (0.75), G3-male (0.79); G1-female (0.32), G2-female (0.25), G3-female (0.21)). E: Early symptoms for patients in three groups (fever (G1: 0.74, G2: 1, G3: 0.79); cough (G1: 0.53, G2: 0.75, G3: 0.57); dyspnea (G1: 0.53, G2: 0.50, G3: 0.29); fatigue (G1: 0.26, G2: 0.50, G3: 0.14); diarrhea (G1: 0.05); chills (G1: 0.05); and headache (G1: 0.11). 

A: Binding sites in ACE2 and spike-S1. Green and cyan schematic represents 3D structure of ACE2 and spike-S1, respectively. Yellow letters and blue spheres show spatial positions for ACE2-binding sites in spike-S1. Black letters and red spheres show spatial positions for spike-binding sites in ACE2. B: Spike-binding domain in ACE2. C: Nineteen amino acid substitutions in ACE2. D: Allele frequency for K26R in ACE2 for different populations. AFR, AMR, EAS, EUR, SAS, NFE, ALSPAC, TWINSUK, and OTH represent African/African American, Admixed American, East Asian, Europeans, South Asian, Non-Finnish European, Avon Longitudinal Study of Parents and Children, UK Adult Twin registry, and Other (population not assigned), respectively. E: ACE2-binding domain in spike-S1. F: Amino acid substitutions in ACE2-binding domain for spike-S1 and their corresponding frequencies.

A: Genomic organization of SARS-CoV-2. Red letters represent proteins that are potential drug targets. B: Compared to SARS-CoV-2-Wuhan01, percentages of synonymous/nonsynonymous substitution sites in 10 CDS regions for bat-CoV-RaTG13, pangolin-CoV-Guangxi, and pangolin-CoV-Guangdong. C, D: Show number and percentage of synonymous/nonsynonymous substitution sites in 10 CDS regions and five potential drug targets for SARS-CoV-2, respectively. E: Amino acid substitutions in SARS-CoV-2 mutants and corresponding residues in bat-CoV-RaTG13, pangolin-CoV, and SARS viruses. Red indicates that amino acid substitution rates are greater than 2. F: Phylogenetic tree of SARS-CoV-2 mutants (L84S in ORF8).

A: RMD domain in spike-S1. B: Distribution of three spike mutations, i.e., N341D, D351Y, and V354F, in 3D structure. C: Physical and chemical parameters for spike-S1 and its mutants. D: Energy and interaction scores for complex structures of ACE2-spike-S1 and its mutants. E: Number of COVID-19 patients in Europe and USA.

A: Number of amino acid mutants in spike-S1 at different time points. B: Distribution of spike-S1 mutants worldwide. C: Number of V354F and V470A mutants at different times. D: Optical density at 450 nm (OD450) for wild-type spike-S1 and its mutants, i.e., V354F and V470A mutants.

A: Sequence similarity between different ACE2 sequences. B: Phylogenetic tree for different ACE2 sequences. C: Sequence logo for ACE2. Red, yellow, and blue triangles represent ACE2 residues that could potentially form a hydrogen bond with the spike-S1 protein of SARS-CoV-2-Wuhan01 and an ionic or PiPistack interaction with the corresponding residues in spike-S1. D: Binding interaction energy scores for ACE2 and spike-S1.

[end paper 3]

[begin paper 4]

PMCID = 11009965

Title = Computational Approaches to Predict Protein–Protein Interactions in Crowded Cellular Environments

Investigating protein–protein interactions is crucial for understanding cellular biological processes because proteins often function within molecular complexes rather than in isolation. While experimental and computational methods have provided valuable insights into these interactions, they often overlook a critical factor: the crowded cellular environment. This environment significantly impacts protein behavior, including structural stability, diffusion, and ultimately the nature of binding. In this review, we discuss theoretical and computational approaches that allow the modeling of biological systems to guide and complement experiments and can thus significantly advance the investigation, and possibly the predictions, of protein–protein interactions in the crowded environment of cell cytoplasm. We explore topics such as statistical mechanics for lattice simulations, hydrodynamic interactions, diffusion processes in high-viscosity environments, and several methods based on molecular dynamics simulations. By synergistically leveraging methods from biophysics and computational biology, we review the state of the art of computational methods to study the impact of molecular crowding on protein–protein interactions and discuss its potential revolutionizing effects on the characterization of the human interactome.

[end paper 4]

[begin paper 5]

PMCID = 8165686

Title = Plausible blockers of Spike RBD in SARS-CoV2—molecular design and underlying interaction dynamics from high-level structural descriptors

COVID-19 is characterized by an unprecedented abrupt increase in the viral transmission rate (SARS-CoV-2) relative to its pandemic evolutionary ancestor, SARS-CoV (2003). The complex molecular cascade of events related to the viral pathogenicity is triggered by the Spike protein upon interacting with the ACE2 receptor on human lung cells through its receptor binding domain (RBDSpike). One potential therapeutic strategy to combat COVID-19 could thus be limiting the infection by blocking this key interaction. In this current study, we adopt a protein design approach to predict and propose non-virulent structural mimics of the RBDSpike which can potentially serve as its competitive inhibitors in binding to ACE2. The RBDSpike is an independently foldable protein domain, resilient to conformational changes upon mutations and therefore an attractive target for strategic re-design. Interestingly, in spite of displaying an optimal shape fit between their interacting surfaces (attributed to a consequently high mutual affinity), the RBDSpike–ACE2 interaction appears to have a quasi-stable character due to a poor electrostatic match at their interface. Structural analyses of homologous protein complexes reveal that the ACE2 binding site of RBDSpike has an unusually high degree of solvent-exposed hydrophobic residues, attributed to key evolutionary changes, making it inherently “reaction-prone.” The designed mimics aimed to block the viral entry by occupying the available binding sites on ACE2, are tested to have signatures of stable high-affinity binding with ACE2 (cross-validated by appropriate free energy estimates), overriding the native quasi-stable feature. The results show the apt of directly adapting natural examples in rational protein design, wherein, homology-based threading coupled with strategic “hydrophobic ↔ polar” mutations serve as a potential breakthrough.

The online version contains supplementary material available at 10.1007/s00894-021-04779-0.

[end paper 5]

[begin paper 6]

PMCID = 9112565

Title = An Electrostatically-steered Conformational Selection Mechanism Promotes SARS-CoV-2 Spike Protein Variation

After two years since the outbreak, the COVID-19 pandemic remains a global public health emergency. SARS-CoV-2 variants with substitutions on the spike (S) protein emerge increasing the risk of immune evasion and cross-species transmission. Here, we analyzed the evolution of the S protein as recorded in 276,712 samples collected before the start of vaccination efforts. Our analysis shows that most variants destabilize the S protein trimer, increase its conformational heterogeneity and improve the odds of the recognition by the host cell receptor. Most frequent substitutions promote overall hydrophobicity by replacing charged amino acids, reducing stabilizing local interactions in the unbound S protein trimer. Moreover, our results identify “forbidden” regions that rarely show any sequence variation, and which are related to conformational changes occurring upon fusion. These results are significant for understanding the structure and function of SARS-CoV-2 related proteins which is a critical step in vaccine development and epidemiological surveillance.

[end paper 6]

[begin paper 7]

PMCID = 11031966

Title = Computational analysis of affinity dynamics between the variants of SARS-CoV-2 spike protein (RBD) and human ACE-2 receptor

The novel coronavirus SARS-CoV-2 resulted in a significant worldwide health emergency known as the COVID-19 pandemic. This crisis has been marked by the widespread of various variants, with certain ones causing notable apprehension. In this study, we harnessed computational techniques to scrutinize these Variants of Concern (VOCs), including various Omicron subvariants. Our approach involved the use of protein structure prediction algorithms and molecular docking techniques, we have investigated the effects of mutations within the Receptor Binding Domain (RBD) of SARS-CoV-2 and how these mutations influence its interactions with the human angiotensin-converting enzyme 2 (hACE-2) receptor. Further we have predicted the structural alterations in the RBD of naturally occurring SARS-CoV-2 variants using the tr-Rosetta algorithm. Subsequent docking and binding analysis employing HADDOCK and PRODIGY illuminated crucial interactions occurring at the Receptor-Binding Motif (RBM). Our findings revealed a hierarchy of increased binding affinity between the human ACE2 receptor and the various RBDs, in the order of wild type (Wuhan-strain) < Beta < Alpha < Gamma < Omicron-B.1.1.529 < Delta < Omicron-BA.2.12.1 < Omicron-BA.5.2.1 < Omicron-BA.1.1. Notably, Omicron-BA.1.1 demonstrated the highest binding affinity of -17.4 kcal mol−1 to the hACE2 receptor when compared to all the mutant complexes. Additionally, our examination indicated that mutations occurring in active residues of the Receptor Binding Domain (RBD) consistently improved the binding affinity and intermolecular interactions in all mutant complexes. Analysis of the differences among variants has laid a foundation for the structure-based drug design targeting the RBD region of SARS-CoV-2.

The online version contains supplementary material available at 10.1186/s12985-024-02365-3.

[end paper 7]

[begin paper 8]

PMCID = 8967028

Title = Structure-based dual affinity optimization of a SARS-CoV-1/2 cross-reactive single-domain antibody

The SARS coronavirus 2 (SARS-CoV-2) spike (S) protein binding to the human ACE2 receptor is the molecular event that initiates viral entry into host cells and leads to infection and virus replication. There is a need for agents blocking viral entry into host cells that are cross-reactive with emerging virus variants. VHH-72 is an anti-SARS-CoV-1 single-domain antibody that also exhibits cross-specificity with SARS-CoV-2 but with decreased binding affinity. Here we applied a structure-based approach to affinity-mature VHH-72 for the SARS-CoV-2 spike protein while retaining the original affinity for SARS-CoV-1. This was achieved by employing the computational platform ADAPT in a constrained dual-affinity optimization mode as a means of broadening specificity. Select mutants designed by ADAPT were formatted as fusions with a human IgG1-Fc fragment. These mutants demonstrated improved binding to the SARS-CoV-2 spike protein due to decreased dissociation rates. Functional testing for virus neutralization revealed improvements relative to the parental VHH72-Fc up to 10-fold using a SARS-CoV-2 pseudotyped lentivirus and 20-fold against the SARS-CoV-2 authentic live virus (Wuhan variant). Binding and neutralization improvements were maintained for some other SARS-CoV-2 variants currently in circulation. These improved VHH-72 mutants are predicted to establish novel interactions with the S antigen. They will be useful, alone or as fusions with other functional modules, in the global quest for treatments of COVID-19 infections.

[end paper 8]

[begin paper 9]

PMCID = 8665054

Title = A 3D Structural SARS-CoV-2-Human Interactome to Explore Genetic and Drug Perturbations

Emergence of new viral agents is driven by evolution of interactions between viral proteins and host targets. For instance, increased infectivity of SARS-CoV-2 compared to SARS-CoV-1 arose in part through rapid evolution along the interface between the Spike protein and its human receptor ACE2, leading to increased binding affinity. To facilitate broader exploration of how pathogen-host interactions might impact transmission and virulence in the ongoing COVID-19 pandemic, we performed state-of-the-art interface prediction followed by molecular docking to construct a 3D structural interactome between SARS-CoV-2 and human. We additionally carried out downstream meta-analyses to investigate enrichment of sequence divergence between SARS-CoV-1 and SARS-CoV-2 or human population variants along viral-human protein interaction interfaces, predict changes in binding affinity by these mutations/variants, and further prioritize drug repurposing candidates predicted to competitively bind human targets. We believe this resource (http://3D-SARS2.yulab.org) will aid in development and testing of informed hypotheses for SARS-CoV-2 etiology and treatments.

[end paper 9]

[begin paper 10]

PMCID = 8618472

Title = Mutational Hotspot in the SARS-CoV-2 Spike Protein N-Terminal Domain Conferring Immune Escape Potential

Global efforts are being made to monitor the evolution of SARS-CoV-2, aiming for early identification of genotypes providing increased infectivity or virulence. However, viral lineage-focused tracking might fail in early detection of advantageous mutations emerging independently across phylogenies. Here, the emergence patterns of Spike mutations were investigated in sequences deposited in local and global databases to identify mutational hotspots across phylogenies and we evaluated their impact on SARS-CoV-2 evolution. We found a striking increase in the frequency of recruitment of diverse substitutions at a critical residue (W152), positioned in the N-terminal domain (NTD) of the Spike protein, observed repeatedly across independent phylogenetic and geographical contexts. These mutations might have an impact on the evasion of neutralizing antibodies. Finally, we found that NTD is a region exhibiting particularly high frequency of mutation recruitments, suggesting an evolutionary path in which the virus maintains optimal efficiency of ACE2 binding combined with the flexibility facilitating the immune escape. We conclude that adaptive mutations, frequently present outside of the receptor-binding domain, can emerge in virtually any SARS-CoV-2 lineage and at any geographical location. Therefore, surveillance should not be restricted to monitoring defined lineages alone.

[end paper 10]

[begin paper 11]

PMCID = 9144871

Title = Structure-Based Development of SARS-CoV-2 Spike Interactors

Coronaviruses, including SARS-CoV-2 (the etiological agent of the current COVID-19 pandemic), rely on the surface spike glycoprotein to access the host cells, mainly through the interaction of their receptor-binding domain (RBD) with the human angiotensin-converting enzyme 2 (ACE2). Therefore, molecular entities able to interfere with the binding of the SARS-CoV-2 spike protein to ACE2 have great potential to inhibit viral entry. Starting from the available structural data on the interaction between SARS-CoV-2 spike protein and the host ACE2 receptor, we engineered a set of soluble and stable spike interactors, here denoted as S-plugs. Starting from the prototype S-plug, we adopted a computational approach by combining stability prediction, associated to single-point mutations, with molecular dynamics to enhance both S-plug thermostability and binding affinity to the spike protein. The best developed molecule, S-plug3, possesses a highly stable α-helical con-formation (with melting temperature Tm of 54 °C) and can interact with the spike RBD and S1 domains with similar low nanomolar affinities. Importantly, S-plug3 exposes the spike RBD to almost the same interface as the human ACE2 receptor, aimed at the recognition of all ACE2-accessing coronaviruses. Consistently, S-plug3 preserves a low nanomolar dissociation constant with the delta B.1.617.2 variant of SARS-CoV-2 spike protein (KD = 29.2 ± 0.6 nM). Taken together, we provide valid starting data for the development of therapeutical and diagnostic tools against coronaviruses accessing through ACE2.

[end paper 11]

[begin paper 12]

PMCID = 7382329

Title = Understanding COVID-19 via comparative analysis of dark proteomes of SARS-CoV-2, human SARS and bat SARS-like coronaviruses

The recently emerged coronavirus designated as SARS-CoV-2 (also known as 2019 novel coronavirus (2019-nCoV) or Wuhan coronavirus) is a causative agent of coronavirus disease 2019 (COVID-19), which is rapidly spreading throughout the world now. More than 1.21 million cases of SARS-CoV-2 infection and more than 67,000 COVID-19-associated mortalities have been reported worldwide till the writing of this article, and these numbers are increasing every passing hour. The World Health Organization (WHO) has declared the SARS-CoV-2 spread as a global public health emergency and admitted COVID-19 as a pandemic now. Multiple sequence alignment data correlated with the already published reports on SARS-CoV-2 evolution indicated that this virus is closely related to the bat severe acute respiratory syndrome-like coronavirus (bat SARS-like CoV) and the well-studied human SARS coronavirus (SARS-CoV). The disordered regions in viral proteins are associated with the viral infectivity and pathogenicity. Therefore, in this study, we have exploited a set of complementary computational approaches to examine the dark proteomes of SARS-CoV-2, bat SARS-like, and human SARS CoVs by analysing the prevalence of intrinsic disorder in their proteins. According to our findings, SARS-CoV-2 proteome contains very significant levels of structural order. In fact, except for nucleocapsid, Nsp8, and ORF6, the vast majority of SARS-CoV-2 proteins are mostly ordered proteins containing less intrinsically disordered protein regions (IDPRs). However, IDPRs found in SARS-CoV-2 proteins are functionally important. For example, cleavage sites in its replicase 1ab polyprotein are found to be highly disordered, and almost all SARS-CoV-2 proteins contains molecular recognition features (MoRFs), which are intrinsic disorder-based protein–protein interaction sites that are commonly utilized by proteins for interaction with specific partners. The results of our extensive investigation of the dark side of SARS-CoV-2 proteome will have important implications in understanding the structural and non-structural biology of SARS or SARS-like coronaviruses.

The online version of this article (10.1007/s00018-020-03603-x) contains supplementary material, which is available to authorized users.

[end paper 12]

[begin paper 13]

PMCID = 10928100

Title = Quantifying how single dose Ad26.COV2.S vaccine efficacy depends on Spike sequence features

In the ENSEMBLE randomized, placebo-controlled phase 3 trial (NCT04505722), estimated single-dose Ad26.COV2.S vaccine efficacy (VE) was 56% against moderate to severe–critical COVID-19. SARS-CoV-2 Spike sequences were determined from 484 vaccine and 1,067 placebo recipients who acquired COVID-19. In this set of prespecified analyses, we show that in Latin America, VE was significantly lower against Lambda vs. Reference and against Lambda vs. non-Lambda [family-wise error rate (FWER) p < 0.05]. VE differed by residue match vs. mismatch to the vaccine-insert at 16 amino acid positions (4 FWER p < 0.05; 12 q-value ≤ 0.20); significantly decreased with physicochemical-weighted Hamming distance to the vaccine-strain sequence for Spike, receptor-binding domain, N-terminal domain, and S1 (FWER p < 0.001); differed (FWER ≤ 0.05) by distance to the vaccine strain measured by 9 antibody-epitope escape scores and 4 NTD neutralization-impacting features; and decreased (p = 0.011) with neutralization resistance level to vaccinee sera. VE against severe–critical COVID-19 was stable across most sequence features but lower against the most distant viruses.

SARS-CoV-2 variants with mutations in spike have emerged during the pandemic. Magaret et al. show that in Latin America, efficacy of the Ad26.COV2.S vaccine against moderate to severe–critical COVID-19 varied by sequence features, antibody escape scores, and neutralization impacting features of the SARS-CoV-2 variant.

[end paper 13]

[begin paper 14]

PMCID = 8178184

Title = An Expanded Benchmark for Antibody-Antigen Docking and Affinity Prediction Reveals Insights into Antibody Recognition Determinants

Accurate predictive modeling of antibody-antigen complex structures and structure-based antibody design remain major challenges in computational biology, with implications for biotherapeutics, immunity, and vaccines. Through a systematic search for high resolution structures of antibody-antigen complexes and unbound antibody and antigen structures, in conjunction with identification of experimentally determined binding affinities, we have assembled a non-redundant set of test cases for antibody-antigen docking and affinity prediction. This benchmark more than doubles the number of antibody-antigen complexes and corresponding affinities available in our previous benchmarks, providing an unprecedented view of the determinants of antibody recognition and insights into molecular flexibility. Initial assessments of docking and affinity prediction tools highlight the challenges posed by this diverse set of cases, which includes camelid nanobodies, therapeutic monoclonal antibodies, and broadly neutralizing antibodies targeting viral glycoproteins. This dataset will enable development of advanced predictive modeling and design methods for this therapeutically relevant class of protein-protein interactions.

Predictive antibody-antigen docking and structure-based design represent longstanding and therapeutically important challenges in computational biology. Guest et al. assembled a large and nonredundant set of antibody-antigen test cases, with high resolution bound and unbound structures and binding affinities, to enable advanced algorithm developments and benchmarking in this area.

[end paper 14]

[begin paper 15]

PMCID = 10990136

Title = Rapid and automated design of two-component protein nanomaterials using ProteinMPNN

We address the current limitations in design of protein–protein interfaces by employing ProteinMPNN, a deep learning method, to create tetrahedral two-component protein nanomaterials that outperform the established physics-based Rosetta design method in terms of computational efficiency and manual refinement. Importantly, the interfaces designed by ProteinMPNN exhibit enhanced polarity, facilitating the seamless assembly of nanomaterials in vitro, which is crucial for efficient biotechnological manufacturing. Our findings demonstrate the potential of deep learning in democratizing protein interface design and showcases the potential of advanced AI methods in speeding up the development of the next generation of protein-based technologies.

The design of protein–protein interfaces using physics-based design methods such as Rosetta requires substantial computational resources and manual refinement by expert structural biologists. Deep learning methods promise to simplify protein–protein interface design and enable its application to a wide variety of problems by researchers from various scientific disciplines. Here, we test the ability of a deep learning method for protein sequence design, ProteinMPNN, to design two-component tetrahedral protein nanomaterials and benchmark its performance against Rosetta. ProteinMPNN had a similar success rate to Rosetta, yielding 13 new experimentally confirmed assemblies, but required orders of magnitude less computation and no manual refinement. The interfaces designed by ProteinMPNN were substantially more polar than those designed by Rosetta, which facilitated in vitro assembly of the designed nanomaterials from independently purified components. Crystal structures of several of the assemblies confirmed the accuracy of the design method at high resolution. Our results showcase the potential of deep learning–based methods to unlock the widespread application of designed protein–protein interfaces and self-assembling protein nanomaterials in biotechnology.

[end paper 15]

[begin paper 16]

PMCID = 9936628

Title = Computational pipeline provides mechanistic understanding of Omicron variant of concern neutralizing engineered ACE2 receptor traps

The SARS-CoV-2 Omicron variant, with 15 mutations in Spike receptor-binding domain (Spike-RBD), renders virtually all clinical monoclonal antibodies against WT SARS-CoV-2 ineffective. We recently engineered the SARS-CoV-2 host entry receptor, ACE2, to tightly bind WT-RBD and prevent viral entry into host cells (“receptor traps”). Here we determine cryo-EM structures of our receptor traps in complex with stabilized Spike ectodomain. We develop a multi-model pipeline combining Rosetta protein modeling software and cryo-EM to allow interface energy calculations even at limited resolution and identify interface side chains that allow for high-affinity interactions between our ACE2 receptor traps and Spike-RBD. Our structural analysis provides a mechanistic rationale for the high-affinity (0.53–4.2 nM) binding of our ACE2 receptor traps to Omicron-RBD confirmed with biolayer interferometry measurements. Finally, we show that ACE2 receptor traps potently neutralize Omicron and Delta pseudotyped viruses, providing alternative therapeutic routes to combat this evolving virus.

With the emergence of new SARS-CoV-2 variants, there is a need for pan-variant Spike binders to inhibit viral entry. Using a novel pipeline combining cryo-EM and Rosetta, Remesh et al. characterize the mechanism of high-affinity interactions of designed ACE2 mimetic binders and show that they neutralize the Omicron variant.

[end paper 16]

[begin paper 17]

PMCID = 8230804

Title = Full-Length Computational Model of the SARS-CoV-2 Spike Protein and Its Implications for a Viral Membrane Fusion Mechanism

The SARS-CoV-2 virus has now become one of the greatest causes of infectious death and morbidity since the 1918 flu pandemic. Substantial and unprecedented progress has been made in the elucidation of the viral infection process in a short time; however, our understanding of the structure–function dynamics of the spike protein during the membrane fusion process and viral uptake remains incomplete. Employing computational approaches, we use full-length structural models of the SARS-CoV-2 spike protein integrating Cryo-EM images and biophysical properties, which fill the gaps in our understanding. We propose a membrane fusion model incorporating structural transitions associated with the proteolytic processing of the spike protein, which initiates and regulates a series of events to facilitate membrane fusion and viral genome uptake. The membrane fusion mechanism highlights the notable role of the S1 subunit and eventual mature spike protein uptake through the host membrane. Our comprehensive view accounts for distinct neutralizing antibody binding effects targeting the spike protein and the enhanced infectivity of the SARS-CoV-2 variant.

[end paper 17]

[begin paper 18]

PMCID = 8929045

Title = A Comparative Study between Spanish and British SARS-CoV-2 Variants

The study of the interaction between the SARS-CoV-2 spike protein and the angiotensin-converting enzyme 2 (ACE2) receptor is key to understanding binding affinity and stability. In the present report, we sought to investigate the differences between two already sequenced genome variants (Spanish and British) of SARS-CoV-2. Methods: In silico model evaluating the homology, identity and similarity in the genome sequence and the structure and alignment of the predictive spike by computational docking methods. Results: The identity results between the Spanish and British variants of the Spike protein were 28.67%. This close correspondence in the results between the Spanish and British SARS-CoV-2 variants shows that they are very similar (99.99%). The alignment obtained results in four deletions. There were 23 nucleotide substitutions also predicted which could affect the functionality of the proteins produced from this sequence. The interaction between the binding receptor domain from the spike protein and the ACE2 receptor produces some of the mutations found and, therefore, the energy of this ligand varies. However, the estimated antigenicity of the British variant is higher than its Spanish counterpart. Conclusions: Our results indicate that minimal mutations could interfere in the infectivity of the virus due to changes in the fitness between host cell recognition and interaction proteins. In particular, the N501Y substitution, situated in the RBD of the spike of the British variant, might be the reason for its extraordinary infective potential.

[end paper 18]

[begin paper 19]

PMCID = 7775788

Title = Conformational dynamics of SARS-CoV-2 trimeric spike glycoprotein in complex with receptor ACE2 revealed by cryo-EM

Structure of SARS-CoV-2 spike in complex with ACE2 reveals the mechanism of ACE2 induced spike conformational transitions.

The recent outbreaks of SARS-CoV-2 pose a global health emergency. The SARS-CoV-2 trimeric spike (S) glycoprotein interacts with the human ACE2 receptor to mediate viral entry into host cells. We report the cryo-EM structures of a tightly closed SARS-CoV-2 S trimer with packed fusion peptide and an ACE2-bound S trimer at 2.7- and 3.8-Å resolution, respectively. Accompanying ACE2 binding to the up receptor-binding domain (RBD), the associated ACE2-RBD exhibits continuous swing motions. Notably, the SARS-CoV-2 S trimer appears much more sensitive to the ACE2 receptor than the SARS-CoV S trimer regarding receptor-triggered transformation from the closed prefusion state to the fusion-prone open state, potentially contributing to the superior infectivity of SARS-CoV-2. We defined the RBD T470-T478 loop and Y505 as viral determinants for specific recognition of SARS-CoV-2 RBD by ACE2. Our findings depict the mechanism of ACE2-induced S trimer conformational transitions from the ground prefusion state toward the postfusion state, facilitating development of anti–SARS-CoV-2 vaccines and therapeutics.

[end paper 19]

[begin paper 20]

PMCID = 8931700

Title = Systematic Tracing of Susceptible Animals to SARS-CoV-2 by a Bioinformatics Framework

Since the outbreak of SARS-CoV-2 in 2019, the Chinese horseshoe bats were considered as a potential original host of SARS-CoV-2. In addition, cats, tigers, lions, mints, and ferrets were naturally or experimentally infected with SARS-CoV-2. For the surveillance and control of this highly infectious disease, it is critical to trace susceptible animals and predict the consequence of potential mutations at the binding region of viral spike protein and host ACE2 protein. This study proposed a novel bioinformatics framework to systematically trace susceptible animals to SARS-CoV-2 and predict the binding affinity between susceptible animals’ mutated/un-mutated ACE2 receptors. As a result, we identified a few animals posing a potential risk of infection with SARS-CoV-2 using the docking analysis of ACE2 protein and viral spike protein. The binding affinity of some of these species is weaker than that of humans but more potent than that of Chinese horseshoe bats. We also found that a few point mutations in human ACE2 protein or viral spike protein could significantly enhance their binding affinity, posing an enormous potential threat to public health. The ancestors of the Omicron may evolve rapidly through the accumulation of mutations in infecting the host and jumped into human beings. These findings indicate that if the epidemic expands, there may be a human-animal-human transmission route, which will increase the difficulty of disease prevention and control.

[end paper 20]

[begin paper 21]

PMCID = 9515332

Title = Reduced B cell antigenicity of Omicron lowers host serologic response

The SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. However, the mechanism remains unclear. Here, we find using a geometric deep-learning model that Omicron’s extensively mutated receptor binding site (RBS) features reduced antigenicity compared with previous variants. Mice immunization experiments with different recombinant receptor binding domain (RBD) variants confirm that the serological response to Omicron is drastically attenuated and less potent. Analyses of serum cross-reactivity and competitive ELISA reveal a reduction in antibody response across both variable and conserved RBD epitopes. Computational modeling confirms that the RBS has a potential for further antigenicity reduction while retaining efficient receptor binding. Finally, we find a similar trend of antigenicity reduction over decades for hCoV229E, a common cold coronavirus. Thus, our study explains the reduced antibody titers associated with Omicron infection and reveals a possible trajectory of future viral evolution.

SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. Tubiana et al. investigate the underlying mechanism using geometric deep learning, mice immunization experiments, and biochemical assays. Mutations reduce antigenicity of the receptor binding site, leading to lower antibody response.

[end paper 21]

[begin paper 22]

PMCID = 8283670

Title = Seq12, Seq12m, and Seq13m, peptide analogues of the spike glycoprotein shows antiviral properties against SARS-CoV-2: An in silico study through molecular docking, molecular dynamics simulation, and MM-PB/GBSA calculations

At the very beginning of the new decade, the COVID-19 pandemic has badly hit modern human societies. SARS-CoV-2, the causative agent of COVID-19 acquiring mutations and circulating as new variants. Herein, we have found three new antiviral peptides (AVPs) against the SARS-CoV-2. These AVPs are analogous to the spike glycoprotein of the SARS-CoV-2. Antiviral peptides, i.e., Seq12, Seq12m, and Seq13m, can block the receptor-binding domain (RBD) of the SARS-CoV-2, which is necessary for communicating with the angiotensin-converting enzyme 2 (ACE2). Also, these AVPs sustain their antiviral properties, even after the insertion of 25 mutations in the RBD (Rosetta and FoldX based). Further, Seq12 and Seq12m showed negligible cytotoxicity. Besides, the binding free energies calculated using MM-PB/GBSA method are also in agreement with the molecular docking studies. The molecular interactions between AVPs and the viral membrane protein (M) also showed a favorable interaction suggesting it could inhibit the viral re-packaging process. In conclusion, this study suggests Seq12, Seq12m, and Seq13m could be helpful to fight against SARS-CoV-2. These AVPs could also aid virus diagnostic tools and nasal spray against SARS-CoV-2 in the future.

[end paper 22]

[begin paper 23]

PMCID = 8998791

Title = VHH Structural Modelling Approaches: A Critical Review

VHH, i.e., VH domains of camelid single-chain antibodies, are very promising therapeutic agents due to their significant physicochemical advantages compared to classical mammalian antibodies. The number of experimentally solved VHH structures has significantly improved recently, which is of great help, because it offers the ability to directly work on 3D structures to humanise or improve them. Unfortunately, most VHHs do not have 3D structures. Thus, it is essential to find alternative ways to get structural information. The methods of structure prediction from the primary amino acid sequence appear essential to bypass this limitation. This review presents the most extensive overview of structure prediction methods applied for the 3D modelling of a given VHH sequence (a total of 21). Besides the historical overview, it aims at showing how model software programs have been shaping the structural predictions of VHHs. A brief explanation of each methodology is supplied, and pertinent examples of their usage are provided. Finally, we present a structure prediction case study of a recently solved VHH structure. According to some recent studies and the present analysis, AlphaFold 2 and NanoNet appear to be the best tools to predict a structural model of VHH from its sequence.

[end paper 23]

[begin paper 24]

PMCID = 7430568

Title = An ultra-potent synthetic nanobody neutralizes SARS-CoV-2 by locking Spike into an inactive conformation

Without an effective prophylactic solution, infections from SARS-CoV-2 continue to rise worldwide with devastating health and economic costs. SARS-CoV-2 gains entry into host cells via an interaction between its Spike protein and the host cell receptor angiotensin converting enzyme 2 (ACE2). Disruption of this interaction confers potent neutralization of viral entry, providing an avenue for vaccine design and for therapeutic antibodies. Here, we develop single-domain antibodies (nanobodies) that potently disrupt the interaction between the SARS-CoV-2 Spike and ACE2. By screening a yeast surface-displayed library of synthetic nanobody sequences, we identified a panel of nanobodies that bind to multiple epitopes on Spike and block ACE2 interaction via two distinct mechanisms. Cryogenic electron microscopy (cryo-EM) revealed that one exceptionally stable nanobody, Nb6, binds Spike in a fully inactive conformation with its receptor binding domains (RBDs) locked into their inaccessible down-state, incapable of binding ACE2. Affinity maturation and structure-guided design of multivalency yielded a trivalent nanobody, mNb6-tri, with femtomolar affinity for SARS-CoV-2 Spike and picomolar neutralization of SARS-CoV-2 infection. mNb6-tri retains stability and function after aerosolization, lyophilization, and heat treatment. These properties may enable aerosol-mediated delivery of this potent neutralizer directly to the airway epithelia, promising to yield a widely deployable, patient-friendly prophylactic and/or early infection therapeutic agent to stem the worst pandemic in a century.

[end paper 24]

[begin paper 25]

PMCID = 8322212

Title = Designing a multi-epitope vaccine candidate to combat MERS-CoV by employing an immunoinformatics approach

Currently, no approved vaccine is available against the Middle East respiratory syndrome coronavirus (MERS-CoV), which causes severe respiratory disease. The spike glycoprotein is typically considered a suitable target for MERS-CoV vaccine candidates. A computational strategy can be used to design an antigenic vaccine against a pathogen. Therefore, we used immunoinformatics and computational approaches to design a multi-epitope vaccine that targets the spike glycoprotein of MERS-CoV. After using numerous immunoinformatics tools and applying several immune filters, a poly-epitope vaccine was constructed comprising cytotoxic T-cell lymphocyte (CTL)-, helper T-cell lymphocyte (HTL)-, and interferon-gamma (IFN-γ)-inducing epitopes. In addition, various physicochemical, allergenic, and antigenic profiles were evaluated to confirm the immunogenicity and safety of the vaccine. Molecular interactions, binding affinities, and the thermodynamic stability of the vaccine were examined through molecular docking and dynamic simulation approaches, during which we identified a stable and strong interaction with Toll-like receptors (TLRs). In silico immune simulations were performed to assess the immune-response triggering capabilities of the vaccine. This computational analysis suggested that the proposed vaccine candidate would be structurally stable and capable of generating an effective immune response to combat viral infections; however, experimental evaluations remain necessary to verify the exact safety and immunogenicity profile of this vaccine.

[end paper 25]

[begin paper 26]

PMCID = 8748687

Title = Structural assessment of HLA-A2-restricted SARS-CoV-2 spike epitopes recognized by public and private T-cell receptors

T cells play a vital role in combatting SARS-CoV-2 and forming long-term memory responses. Whereas extensive structural information is available on neutralizing antibodies against SARS-CoV-2, such information on SARS-CoV-2-specific T-cell receptors (TCRs) bound to their peptide–MHC targets is lacking. Here we determine the structures of a public and a private TCR from COVID-19 convalescent patients in complex with HLA-A2 and two SARS-CoV-2 spike protein epitopes (YLQ and RLQ). The structures reveal the basis for selection of particular TRAV and TRBV germline genes by the public but not the private TCR, and for the ability of the TCRs to recognize natural variants of RLQ but not YLQ. Neither TCR recognizes homologous epitopes from human seasonal coronaviruses. By elucidating the mechanism for TCR recognition of an immunodominant yet variable epitope (YLQ) and a conserved but less commonly targeted epitope (RLQ), this study can inform prospective efforts to design vaccines to elicit pan-coronavirus immunity.

Structural immunology is critical in understanding the interplay between the immune response and the infective agent but such studies in T cells and SARS-CoV-2 lag behind those of antibodies and B-cell receptors. Here the authors assess recognition of SARS-CoV-2 spike epitopes and their natural variants by public and private T cell receptors.

[end paper 26]

[begin paper 27]

PMCID = 7546226

Title = Exploring the out of sight antigens of SARS-CoV-2 to design a candidate multi-epitope vaccine by utilizing immunoinformatics approaches

The vaccine is composed of immunodominant regions of SARS-CoV-2 non-structural proteins.

Also, the functional region of the spike protein is incorporated in the vaccine construct.

The final vaccine construct contains multiple CD8+ and CD4+ overlapping epitopes

Also, it contains multiple IFN-γ inducing, linear and conformational B cell epitopes.

It forms significant interactions and stable complex with TLR-4/MD.

The DNA vaccine is designed by reverse translation of the final vaccine construct.

SARS-CoV-2 causes a severe respiratory disease called COVID-19. Currently, global health is facing its devastating outbreak. However, there is no vaccine available against this virus up to now. In this study, a novel multi-epitope vaccine against SARS-CoV-2 was designed to provoke both innate and adaptive immune responses. The immunodominant regions of six non-structural proteins (nsp7, nsp8, nsp9, nsp10, nsp12 and nsp14) of SARS-CoV-2 were selected by multiple immunoinformatic tools to provoke T cell immune response. Also, immunodominant fragment of the functional region of SARS-CoV-2 spike (400–510 residues) protein was selected for inducing neutralizing antibodies production. The selected regions’ sequences were connected to each other by furin-sensitive linker (RVRR). Moreover, the functional region of β-defensin as a well-known agonist for the TLR-4/MD complex was added at the N-terminus of the vaccine using (EAAAK)3 linker. Also, a CD4 + T-helper epitope, PADRE, was used at the C-terminal of the vaccine by GPGPG and A(EAAAK)2A linkers to form the final vaccine construct. The physicochemical properties, allergenicity, antigenicity, functionality and population coverage of the final vaccine construct were analyzed. The final vaccine construct was an immunogenic, non-allergen and unfunctional protein which contained multiple CD8 + and CD4 + overlapping epitopes, IFN-γ inducing epitopes, linear and conformational B cell epitopes. It could form stable and significant interactions with TLR-4/MD according to molecular docking and dynamics simulations. Global population coverage of the vaccine for HLA-I and II were estimated 96.2% and 97.1%, respectively. At last, the final vaccine construct was reverse translated to design the DNA vaccine. Although the designed vaccine exhibited high efficacy in silico, further experimental validation is necessary.

[end paper 27]

[begin paper 28]

PMCID = 8810837

Title = Computational discovery of binding mode of anti-TRBC1 antibody and predicted key amino acids of TRBC1

Peripheral T-cell lymphoma (PTCL) is a type of non-Hodgkin lymphoma that progresses aggressively with poor survival rate. CAR T cell targeting T-cell receptor β-chain constant domains 1 (TRBC1) of malignant T cells has been developed recently by using JOVI.1 monoclonal antibody as a template. However, the mode of JOVI.1 binding is still unknown. This study aimed to investigate the molecular interaction between JOVI.1 antibody and TRBC1 by using computational methods and molecular docking. Therefore, the TRBC protein crystal structures (TRBC1 and TRBC2) as well as the sequences of JOVI.1 CDR were chosen as the starting materials. TRBC1 and TRBC2 epitopes were predicted, and molecular dynamic (MD) simulation was used to visualize the protein dynamic behavior. The structure of JOVI.1 antibody was also generated before the binding mode was predicted using molecular docking with an antibody mode. Epitope prediction suggested that the N3K4 region of TRBC1 may be a key to distinguish TRBC1 from TCBC2. MD simulation showed the major different surface conformation in this area between two TRBCs. The JOVI.1-TRBC1 structures with three binding modes demonstrated JOVI.1 interacted TRBC1 at N3K4 residues, with the predicted dissociation constant (Kd) ranging from 1.5 × 108 to 1.1 × 1010 M. The analysis demonstrated JOVI.1 needed D1 residues of TRBC1 for the interaction formation to N3K4 in all binding modes. In conclusion, we proposed the three binding modes of the JOVI.1 antibody to TRBC1 with the new key residue (D1) necessary for N3K4 interaction. This data was useful for JOVI.1 redesign to improve the PTCL-targeting CAR T cell.

[end paper 28]

[begin paper 29]

PMCID = 6939872

Title = Atomic Structure of Francisella T6SS Central Spike Reveals Unique α-Helical Lid and a Putative Cargo

Francisella bacteria rely on a phylogenetically distinct Type VI Secretion System (T6SS) to escape host phagosomes and cause the fatal disease tularemia, but the structural and molecular mechanisms involved are unknown. Here, we report the atomic structure of the Francisella T6SS central spike complex, obtained by cryo-electron microscopy. Our structural and functional studies demonstrate that, unlike the single-protein spike composition of other T6SS subtypes, Francisella T6SS’s central spike is formed by two proteins, PdpA and VgrG, akin to T4-bacteriophage gp27 and gp5, respectively, and that PdpA has unique characteristics, including a putative cargo within its cavity and an N-terminal helical lid. Structure-guided mutagenesis demonstrates that the PdpA N-terminal lid and C-terminal spike are essential to Francisella T6SS function. PdpA is thus both an adaptor, connecting VgrG to the tube, and a likely carrier of secreted cargo. These findings are important to understanding Francisella pathogenicity and designing therapeutics to combat tularemia.

Type 6 secretion systems, critical to virulence of many bacteria, including Francisella, drive a central spike into target cells. Xue et al. used cryo-EM to determine the Francisella central spike atomic structure, revealing features absent from canonical T6SS, including an α-helical lid and putative cargo within its head cavity.

[end paper 29]

[begin paper 30]

PMCID = 8863144

Title = Reduced antigenicity of Omicron lowers host serologic response

SARS-CoV-2 Omicron variant of concern (VOC) contains fifteen mutations on the receptor binding domain (RBD), evading most neutralizing antibodies from vaccinated sera. Emerging evidence suggests that Omicron breakthrough cases are associated with substantially lower antibody titers than other VOC cases. However, the mechanism remains unclear. Here, using a novel geometric deep-learning model, we discovered that the antigenic profile of Omicron RBD is distinct from the prior VOCs, featuring reduced antigenicity in its remodeled receptor binding sites (RBS). To substantiate our deep-learning prediction, we immunized mice with different recombinant RBD variants and found that the Omicron’s extensive mutations can lead to a drastically attenuated serologic response with limited neutralizing activity in vivo, while the T cell response remains potent. Analyses of serum cross-reactivity and competitive ELISA with epitope-specific nanobodies revealed that the antibody response to Omicron was reduced across RBD epitopes, including both the variable RBS and epitopes without any known VOC mutations. Moreover, computational modeling confirmed that the RBS is highly versatile with a capacity to further decrease antigenicity while retaining efficient receptor binding. Longitudinal analysis showed that this evolutionary trend of decrease in antigenicity was also found in hCoV229E, a common cold coronavirus that has been circulating in humans for decades. Thus, our study provided unprecedented insights into the reduced antibody titers associated with Omicron infection, revealed a possible trajectory of future viral evolution and may inform the vaccine development against future outbreaks.

[end paper 30]

[begin paper 31]

PMCID = 8222070

Title = Deep learning techniques have significantly impacted protein structure prediction and protein design

Protein structure prediction and design can be regarded as two inverse processes governed by the same folding principle. Although progress remained stagnant over the last two decades, the recent application of deep neural networks to spatial constraint prediction and end-to-end model training has significantly improved the accuracy of protein structure prediction, largely solving the problem at the fold level for single-domain proteins. The field of protein design has also witnessed dramatic improvement, where noticeable examples have shown that information stored in neural-network models can be used to advance functional protein design. Thus, incorporation of deep learning techniques into different steps of protein folding and design approaches represents an exciting future direction and should continue to have a transformative impact on both fields.

[end paper 31]

[begin paper 32]

PMCID = 7543886

Title = SARS-CoV-2 Disrupts Splicing, Translation, and Protein Trafficking to Suppress Host Defenses

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is a recently identified coronavirus that causes the respiratory disease known as coronavirus disease 2019 (COVID-19). Despite the urgent need, we still do not fully understand the molecular basis of SARS-CoV-2 pathogenesis. Here, we comprehensively define the interactions between SARS-CoV-2 proteins and human RNAs. NSP16 binds to the mRNA recognition domains of the U1 and U2 splicing RNAs and acts to suppress global mRNA splicing upon SARS-CoV-2 infection. NSP1 binds to 18S ribosomal RNA in the mRNA entry channel of the ribosome and leads to global inhibition of mRNA translation upon infection. Finally, NSP8 and NSP9 bind to the 7SL RNA in the signal recognition particle and interfere with protein trafficking to the cell membrane upon infection. Disruption of each of these essential cellular functions acts to suppress the interferon response to viral infection. Our results uncover a multipronged strategy utilized by SARS-CoV-2 to antagonize essential cellular processes to suppress host defenses.

NSP16 binds mRNA recognition domains of U1/U2 snRNAs and disrupts mRNA splicing

NSP1 binds in the mRNA entry channel of the ribosome to disrupt protein translation

NSP8 and NSP9 bind the signal recognition particle and disrupt protein trafficking

These disruptions of protein production suppress the interferon response to infection

SARS-CoV-2 proteins directly engage host RNAs to dysregulate essential steps of protein production and suppress the interferon response.

[end paper 32]

[begin paper 33]

PMCID = 9683856

Title = Structural evolution of Delta lineage of SARS-CoV-2

One of the main obstacles in prevention and treatment of COVID-19 is the rapid evolution of the SARS-CoV-2 Spike protein. Given that Spike is the main target of common treatments of COVID-19, mutations occurring at this virulent factor can affect the effectiveness of treatments. The B.1.617.2 lineage of SARS-CoV-2, being characterized by many Spike mutations inside and outside of its receptor-binding domain (RBD), shows high infectivity and relative resistance to existing cures. Here, utilizing a wide range of computational biology approaches, such as immunoinformatics, molecular dynamics (MD), analysis of intrinsically disordered regions (IDRs), protein-protein interaction analyses, residue scanning, and free energy calculations, we examine the structural and biological attributes of the B.1.617.2 Spike protein. Furthermore, the antibody design protocol of Rosetta was implemented for evaluation the stability and affinity improvement of the Bamlanivimab (LY-CoV55) antibody, which is not capable of interactions with the B.1.617.2 Spike. We observed that the detected mutations in the Spike of the B1.617.2 variant of concern can cause extensive structural changes compatible with the described variation in immunogenicity, secondary and tertiary structure, oligomerization potency, Furin cleavability, and drug targetability. Compared to the Spike of Wuhan lineage, the B.1.617.2 Spike is more stable and binds to the Angiotensin-converting enzyme 2 (ACE2) with higher affinity.

[end paper 33]

[begin paper 34]

PMCID = 9628197

Title = Computationally restoring the potency of a clinical antibody against SARS-CoV-2 Omicron subvariants

The COVID-19 pandemic underscored the promise of monoclonal antibody-based prophylactic and therapeutic drugs, but also revealed how quickly viral escape can curtail effective options. With the emergence of the SARS-CoV-2 Omicron variant in late 2021, many clinically used antibody drug products lost potency, including Evusheld™ and its constituent, cilgavimab. Cilgavimab, like its progenitor COV2-2130, is a class 3 antibody that is compatible with other antibodies in combination and is challenging to replace with existing approaches. Rapidly modifying such high-value antibodies with a known clinical profile to restore efficacy against emerging variants is a compelling mitigation strategy. We sought to redesign COV2-2130 to rescue in vivo efficacy against Omicron BA.1 and BA.1.1 strains while maintaining efficacy against the contemporaneously dominant Delta variant. Here we show that our computationally redesigned antibody, 2130-1-0114-112, achieves this objective, simultaneously increases neutralization potency against Delta and many variants of concern that subsequently emerged, and provides protection in vivo against the strains tested, WA1/2020, BA.1.1, and BA.5. Deep mutational scanning of tens of thousands pseudovirus variants reveals 2130-1-0114-112 improves broad potency without incurring additional escape liabilities. Our results suggest that computational approaches can optimize an antibody to target multiple escape variants, while simultaneously enriching potency. Because our approach is computationally driven, not requiring experimental iterations or pre-existing binding data, it could enable rapid response strategies to address escape variants or pre-emptively mitigate escape vulnerabilities.

[end paper 34]

[begin paper 35]

PMCID = 8920274

Title = Large-scale design and refinement of stable proteins using sequence-only models

Engineered proteins generally must possess a stable structure in order to achieve their designed function. Stable designs, however, are astronomically rare within the space of all possible amino acid sequences. As a consequence, many designs must be tested computationally and experimentally in order to find stable ones, which is expensive in terms of time and resources. Here we use a high-throughput, low-fidelity assay to experimentally evaluate the stability of approximately 200,000 novel proteins. These include a wide range of sequence perturbations, providing a baseline for future work in the field. We build a neural network model that predicts protein stability given only sequences of amino acids, and compare its performance to the assayed values. We also report another network model that is able to generate the amino acid sequences of novel stable proteins given requested secondary sequences. Finally, we show that the predictive model—despite weaknesses including a noisy data set—can be used to substantially increase the stability of both expert-designed and model-generated proteins.

[end paper 35]

[begin paper 36]

PMCID = 9889117

Title = Rational approaches to discover SARS-CoV-2/ACE2 interaction inhibitors: Pharmacophore-based virtual screening, molecular docking, molecular dynamics and binding free energy studies

The lack of effective treatment remains a bottleneck in combating the current coronavirus family pandemic, particularly coronavirus 2019 (COVID-19), which is caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The infection of host cells by SARS-CoV-2 is mediated by the binding of its receptor-binding domain (RBD) on the spike (S) glycoprotein to the host angiotensin-converting enzyme (ACE2) receptor. As all developed and available vaccines against COVID-19 do not provide long-term immunity, the creation of an effective drug for the treatment of COVID-19 is necessary and cannot be ignored. Therefore, the aim of this study is to present a computational screening method to identify potential inhibitor candidates with a high probability of blocking the binding of RBD to the ACE2 receptor. Pharmacophore mapping, molecular docking, molecular dynamics (MD) simulations, and binding free-energy analyses were performed to identify potential inhibitor candidates against ACE2/SARS-CoV-2. In conclusion, we propose the compound PubChem-84280085 as a potential inhibitor of protein–protein interactions to disrupt the binding of the SARS-CoV-2-RBD to the ACE2 receptor.

[end paper 36]

[begin paper 37]

PMCID = 7500280

Title = Computational biophysical characterization of the SARS-CoV-2 spike protein binding with the ACE2 receptor and implications for infectivity

SARS-CoV-2 is a novel highly virulent pathogen which gains entry to human cells by binding with the cell surface receptor – angiotensin converting enzyme (ACE2). We computationally contrasted the binding interactions between human ACE2 and coronavirus spike protein receptor binding domain (RBD) of the 2002 epidemic-causing SARS-CoV-1, SARS-CoV-2, and bat coronavirus RaTG13 using the Rosetta energy function. We find that the RBD of the spike protein of SARS-CoV-2 is highly optimized to achieve very strong binding with human ACE2 (hACE2) which is consistent with its enhanced infectivity. SARS-CoV-2 forms the most stable complex with hACE2 compared to SARS-CoV-1 (23% less stable) or RaTG13 (11% less stable). Notably, we calculate that the SARS-CoV-2 RBD lowers the binding strength of angiotensin 2 receptor type I (ATR1) which is the native binding partner of ACE2 by 44.2%. Strong binding is mediated through strong electrostatic attachments with every fourth residue on the N-terminus alpha-helix (starting from Ser19 to Asn53) as the turn of the helix makes these residues solvent accessible. By contrasting the spike protein SARS-CoV-2 Rosetta binding energy with ACE2 of different livestock and pet species we find strongest binding with bat ACE2 followed by human, feline, equine, canine and finally chicken. This is consistent with the hypothesis that bats are the viral origin and reservoir species. These results offer a computational explanation for the increased infection susceptibility by SARS-CoV-2 and allude to therapeutic modalities by identifying and rank-ordering the ACE2 residues involved in binding with the virus.

[end paper 37]

[begin paper 38]

PMCID = 6970540

Title = The human coronavirus HCoV-229E S-protein structure and receptor binding

The coronavirus S-protein mediates receptor binding and fusion of the viral and host cell membranes. In HCoV-229E, its receptor binding domain (RBD) shows extensive sequence variation but how S-protein function is maintained is not understood. Reported are the X-ray crystal structures of Class III-V RBDs in complex with human aminopeptidase N (hAPN), as well as the electron cryomicroscopy structure of the 229E S-protein. The structures show that common core interactions define the specificity for hAPN and that the peripheral RBD sequence variation is accommodated by loop plasticity. The results provide insight into immune evasion and the cross-species transmission of 229E and related coronaviruses. We also find that the 229E S-protein can expose a portion of its helical core to solvent. This is undoubtedly facilitated by hydrophilic subunit interfaces that we show are conserved among coronaviruses. These interfaces likely play a role in the S-protein conformational changes associated with membrane fusion.

[end paper 38]

[begin paper 39]

PMCID = 8379094

Title = The neutralization potency of anti-SARS-CoV-2 therapeutic human monoclonal antibodies is retained against viral variants

A wide range of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) neutralizing monoclonal antibodies (mAbs) have been reported, most of which target the spike glycoprotein. Therapeutic implementation of these antibodies has been challenged by emerging SARS-CoV-2 variants harboring mutated spike versions. Consequently, re-assessment of previously identified mAbs is of high priority. Four previously selected mAbs targeting non-overlapping epitopes are now evaluated for binding potency to mutated RBD versions, reported to mediate escape from antibody neutralization. In vitro neutralization potencies of these mAbs, and two NTD-specific mAbs, are evaluated against two frequent SARS-CoV-2 variants of concern, the B.1.1.7 Alpha and the B.1.351 Beta. Furthermore, we demonstrate therapeutic potential of three selected mAbs by treatment of K18-human angiotensin-converting enzyme 2 (hACE2) transgenic mice 2 days post-infection with each virus variant. Thus, despite the accumulation of spike mutations, the highly potent MD65 and BL6 mAbs retain their ability to bind the prevalent viral mutants, effectively protecting against B.1.1.7 and B.1.351 variants.

Novel SARS-CoV-2 antigenic variants jeopardize the efficacy of immunotherapies. Makdasi et al. re-evaluate anti-SARS-CoV-2 Abs previously shown to be highly effective against the original version of the virus. Some of the inspected antibodies retain their neutralization ability and in vivo protective efficacy against various viral variants.

[end paper 39]

[begin paper 40]

PMCID = 7383237

Title = Diversification of DNA-Binding Specificity by Permissive and Specificity-Switching Mutations in the ParB/Noc Protein Family

Specific interactions between proteins and DNA are essential to many biological processes. Yet, it remains unclear how the diversification in DNA-binding specificity was brought about, and the mutational paths that led to changes in specificity are unknown. Using a pair of evolutionarily related DNA-binding proteins, each with a different DNA preference (ParB [Partitioning Protein B] and Noc [Nucleoid Occlusion Factor], which both play roles in bacterial chromosome maintenance), we show that specificity is encoded by a set of four residues at the protein-DNA interface. Combining X-ray crystallography and deep mutational scanning of the interface, we suggest that permissive mutations must be introduced before specificity-switching mutations to reprogram specificity and that mutational paths to new specificity do not necessarily involve dual-specificity intermediates. Overall, our results provide insight into the possible evolutionary history of ParB and Noc and, in a broader context, might be useful for understanding the evolution of other classes of DNA-binding proteins.

DNA-binding specificity for parS and NBS is conserved within ParB and Noc family

Specificity is encoded by a set of four residues at the protein-DNA interface

Mutations must be introduced in a defined order to reprogram specificity

Jalal et al. elucidate the molecular basis for how specific protein-DNA interactions can evolve, using ParB and Noc as models. Using X-ray crystallography and deep mutational scanning, they define protein-DNA interfaces and suggest that permissive mutations must be introduced before specificity-switching mutations to reprogram specificity.

[end paper 40]

[begin paper 41]

PMCID = 7065964

Title = HIV-1 gp120-CD4i Antibody Complex Elicits CD4 Binding Site Specific Antibody Response in Mice

Elicitation of broadly neutralizing antibody (bNAb) responses toward the conserved HIV-1 Envelope (Env) CD4 binding site (CD4bs) by vaccination is an important goal for vaccine development and yet to be achieved. The outcome of previous immunogenicity studies suggests that the limited accessibility of the CD4bs and the presence of predominant non-neutralizing determinants (nND) on Env may impede the elicitation of bNAbs and their precursors by vaccination. Here we designed a panel of novel immunogens that 1) preferentially expose the CD4bs by selective elimination of glycosylation sites flanking the CD4bs, and 2) minimize the nND immune response by engineering fusion proteins consisting of gp120 core and one or two CD4-induced (CD4i) monoclonal antibodies (mAbs) for masking nND epitopes, referred to as gp120-CD4i fusion proteins. As expected, the fusion proteins possess improved antigenicity with retained affinity for VRC01-class CD4bs-directed bNAbs and dampened affinity for non-neutralizing antibodies. We immunized C57BL/6 mice with these fusion proteins and found that overall the fusion proteins elicit more focused CD4bs antibody response than prototypical gp120 core by serological analysis. Consistently, we found that mice immunized with selected gp120-CD4i fusion proteins have higher frequencies of germinal center activated B cells and CD4bs-directed memory B cells than those inoculated with parental immunogens. We isolated three mAbs from mice immunized with selected gp120-CD4i fusion proteins and found that their footprints on Env are similar to VRC01-class bNAbs. Thus, utilizing gp120-CD4i fusion proteins with selective glycan deletion as immunogens could focus antibody response toward CD4bs epitope.

[end paper 41]

[begin paper 42]

PMCID = 10013278

Title = SARS-CoV-2 receptor-binding domain deep mutational AlphaFold2 structures

Leveraging recent advances in computational modeling of proteins with AlphaFold2 (AF2) we provide a complete curated data set of all single mutations from each of the 7 main SARS-CoV-2 lineages spike protein receptor binding domain (RBD) resulting in 3819X7 = 26733 PDB structures. We visualize the generated structures and show that AF2 pLDDT values are correlated with state-of-the-art disorder approximations, implying some internal protein dynamics are also captured by the model. Joint increasing mutational coverage of both structural and phenotype data coupled with advances in machine learning can be leveraged to accelerate virology research, specifically future variant prediction. We hope this data release can offer assistance into further understanding of the local and global mutational landscape of SARS-CoV-2 as well as provide insight into the biological understanding that 3D structure acts as a bridge between protein genotype and phenotype.

[end paper 42]

[begin paper 43]

PMCID = 10086000

Title = High-throughput identification of prefusion-stabilizing mutations in SARS-CoV-2 spike

Designing prefusion-stabilized SARS-CoV-2 spike is critical for the effectiveness of COVID-19 vaccines. All COVID-19 vaccines in the US encode spike with K986P/V987P mutations to stabilize its prefusion conformation. However, contemporary methods on engineering prefusion-stabilized spike immunogens involve tedious experimental work and heavily rely on structural information. Here, we establish a systematic and unbiased method of identifying mutations that concomitantly improve expression and stabilize the prefusion conformation of the SARS-CoV-2 spike. Our method integrates a fluorescence-based fusion assay, mammalian cell display technology, and deep mutational scanning. As a proof-of-concept, we apply this method to a region in the S2 domain that includes the first heptad repeat and central helix. Our results reveal that besides K986P and V987P, several mutations simultaneously improve expression and significantly lower the fusogenicity of the spike. As prefusion stabilization is a common challenge for viral immunogen design, this work will help accelerate vaccine development against different viruses.

Designing vaccine immunogens is often a tedious process. Here the authors develop a deep mutational scanning-based method to rapidly and comprehensively identify prefusion stabilizing mutations of SARS-CoV-2 spike as a vaccine immunogen.

[end paper 43]

[begin paper 44]

PMCID = 8746363

Title = The Structural Basis of Peptide Binding at Class A G Protein-Coupled Receptors

G protein-coupled receptors (GPCRs) represent the largest membrane protein family and a significant target class for therapeutics. Receptors from GPCRs’ largest class, class A, influence virtually every aspect of human physiology. About 45% of the members of this family endogenously bind flexible peptides or peptides segments within larger protein ligands. While many of these peptides have been structurally characterized in their solution state, the few studies of peptides in their receptor-bound state suggest that these peptides interact with a shared set of residues and undergo significant conformational changes. For the purpose of understanding binding dynamics and the development of peptidomimetic drug compounds, further studies should investigate the peptide ligands that are complexed to their cognate receptor.

[end paper 44]

[begin paper 45]

PMCID = 7900118

Title = In silico detection of SARS-CoV-2 specific B-cell epitopes and validation in ELISA for serological diagnosis of COVID-19

Rapid generation of diagnostics is paramount to understand epidemiology and to control the spread of emerging infectious diseases such as COVID-19. Computational methods to predict serodiagnostic epitopes that are specific for the pathogen could help accelerate the development of new diagnostics. A systematic survey of 27 SARS-CoV-2 proteins was conducted to assess whether existing B-cell epitope prediction methods, combined with comprehensive mining of sequence databases and structural data, could predict whether a particular protein would be suitable for serodiagnosis. Nine of the predictions were validated with recombinant SARS-CoV-2 proteins in the ELISA format using plasma and sera from patients with SARS-CoV-2 infection, and a further 11 predictions were compared to the recent literature. Results appeared to be in agreement with 12 of the predictions, in disagreement with 3, while a further 5 were deemed inconclusive. We showed that two of our top five candidates, the N-terminal fragment of the nucleoprotein and the receptor-binding domain of the spike protein, have the highest sensitivity and specificity and signal-to-noise ratio for detecting COVID-19 sera/plasma by ELISA. Mixing the two antigens together for coating ELISA plates led to a sensitivity of 94% (N = 80 samples from persons with RT-PCR confirmed SARS-CoV-2 infection), and a specificity of 97.2% (N = 106 control samples).

[end paper 45]

[begin paper 46]

PMCID = 8835783

Title = A Multi-Disulfide Receptor-Binding Domain (RBD) of the SARS-CoV-2 Spike Protein Expressed in E. coli Using a SEP-Tag Produces Antisera Interacting with the Mammalian Cell Expressed Spike (S1) Protein

An Escherichia coli (E. coli) production of the receptor-binding domain (RBD) of the SARS-CoV-2 (isolate Wuhan-Hu-1) spike protein would significantly accelerate the search for anti-COVID-19 therapeutics because of its versatility and low cost. However, RBD contains four disulfide bonds and its expression in E. coli is limited by the formation of aberrant disulfide bonds resulting in inclusion bodies. Here, we show that a solubility-enhancing peptide (SEP) tag containing nine arginine residues (RBD-C9R) attached at the C-terminus can overcome this problem. The SEP-tag increased the expression in the soluble fraction and the final yield by five times (2 mg/L). The folding properties of the E. coli expressed RBD-C9R were demonstrated with biophysical characterization using RP-HPLC, circular dichroism, thermal denaturation, fluorescence, and light scattering. A quartz crystal microbalance (QCM) analysis confirmed the binding activity of RBD-C9R with ACE2, the host cell’s receptor. In addition, RBD-C9R elicited a Th-2 immune response with a high IgG titer in Jcl: ICR mice. The RBD-C9R antisera interacted with both itself and the mammalian-cell expressed spike protein (S1), as demonstrated by ELISA, indicating that the E. coli expressed RBD-C9R harbors native-like epitopes. Overall, these results emphasize the potential of our SEP-tag for the E. coli production of active multi-disulfide-bonded RBD.

[end paper 46]

[begin paper 47]

PMCID = 9854172

Title = Neural network‐derived Potts models for structure‐based protein design using backbone atomic coordinates and tertiary motifs

Designing novel proteins to perform desired functions, such as binding or catalysis, is a major goal in synthetic biology. A variety of computational approaches can aid in this task. An energy‐based framework rooted in the sequence‐structure statistics of tertiary motifs (TERMs) can be used for sequence design on predefined backbones. Neural network models that use backbone coordinate‐derived features provide another way to design new proteins. In this work, we combine the two methods to make neural structure‐based models more suitable for protein design. Specifically, we supplement backbone‐coordinate features with TERM‐derived data, as inputs, and we generate energy functions as outputs. We present two architectures that generate Potts models over the sequence space: TERMinator, which uses both TERM‐based and coordinate‐based information, and COORDinator, which uses only coordinate‐based information. Using these two models, we demonstrate that TERMs can be utilized to improve native sequence recovery performance of neural models. Furthermore, we demonstrate that sequences designed by TERMinator are predicted to fold to their target structures by AlphaFold. Finally, we show that both TERMinator and COORDinator learn notions of energetics, and these methods can be fine‐tuned on experimental data to improve predictions. Our results suggest that using TERM‐based and coordinate‐based features together may be beneficial for protein design and that structure‐based neural models that produce Potts energy tables have utility for flexible applications in protein science.

[end paper 47]

[begin paper 48]

PMCID = 10592684

Title = A Suite of Designed Protein Cages Using Machine Learning Algorithms and Protein Fragment-Based Protocols

Designed protein cages and related materials provide unique opportunities for applications in biotechnology and medicine, while methods for their creation remain challenging and unpredictable. In the present study, we apply new computational approaches to design a suite of new tetrahedrally symmetric, self-assembling protein cages. For the generation of docked poses, we emphasize a protein fragment-based approach, while for de novo interface design, a comparison of computational protocols highlights the power and increased experimental success achieved using the machine learning program ProteinMPNN. In relating information from docking and design, we observe that agreement between fragment-based sequence preferences and ProteinMPNN sequence inference correlates with experimental success. Additional insights for designing polar interactions are highlighted by experimentally testing larger and more polar interfaces. In all, using X-ray crystallography and cryo-EM, we report five structures for seven protein cages, with atomic resolution in the best case reaching 2.0 Å. We also report structures of two incompletely assembled protein cages, providing unique insights into one type of assembly failure. The new set of designed cages and their structures add substantially to the body of available protein nanoparticles, and to methodologies for their creation.

[end paper 48]

[begin paper 49]

PMCID = 7999464

Title = Recent Progress Using De Novo Design to Study Protein Structure, Design and Binding Interactions

De novo protein design is a powerful methodology used to study natural functions in an artificial-protein context. Since its inception, it has been used to reproduce a plethora of reactions and uncover biophysical principles that are often difficult to extract from direct studies of natural proteins. Natural proteins are capable of assuming a variety of different structures and subsequently binding ligands at impressively high levels of both specificity and affinity. Here, we will review recent examples of de novo design studies on binding reactions for small molecules, nucleic acids, and the formation of protein-protein interactions. We will then discuss some new structural advances in the field. Finally, we will discuss some advancements in computational modeling and design approaches and provide an overview of some modern algorithmic tools being used to design these proteins.

[end paper 49]

[begin paper 50]

PMCID = 10507807

Title =  De novo design of a stapled peptide targeting SARS-CoV-2 spike protein receptor-binding domain†

Although effective vaccines have been developed against SARS-CoV-2, many regions in the world still have low rates of vaccination and new variants with mutations in the viral spike protein have reduced the effectiveness of most available vaccines and treatments. There is an urgent need for a drug to cure this disease and prevent infection. The SARS-CoV-2 virus enters the host cell through protein–protein interaction between the virus's spike protein and the host's angiotensin converting enzyme (ACE2). Using protein design software and molecular dynamics simulations, we have designed a 17-residue peptide (pep39), that binds to the spike protein receptor-binding domain (RBD) and blocks interaction of spike protein with ACE2. We have confirmed the binding activity of the designed peptide for the original spike protein and the delta variant spike protein using micro-cantilever and bio-layer interferometry (BLI) based methods. We also confirmed that pep39 strongly inhibits SARS-CoV-2 virus replication in Vero E6 cells. Taken together these data suggest that a newly designed spike protein RBD blocking peptide pep39 has a potential as a SARS-CoV-2 virus inhibitor.

Binding configuration of a de novo stapled peptide on SARS-CoV-2 spike protein, as predicted by molecular simulation. Stapled residues enhance peptide stability while interacting residues engage key amino acids on the protein receptor-binding domain.

[end paper 50]

[begin paper 51]

PMCID = 9939524

Title = Quantitative comparison of protein-protein interaction interface using physicochemical feature-based descriptors of surface patches

Driving mechanisms of many biological functions in a cell include physical interactions of proteins. As protein-protein interactions (PPIs) are also important in disease development, protein-protein interactions are highlighted in the pharmaceutical industry as possible therapeutic targets in recent years. To understand the variety of protein-protein interactions in a proteome, it is essential to establish a method that can identify similarity and dissimilarity between protein-protein interactions for inferring the binding of similar molecules, including drugs and other proteins. In this study, we developed a novel method, protein-protein interaction-Surfer, which compares and quantifies similarity of local surface regions of protein-protein interactions. protein-protein interaction-Surfer represents a protein-protein interaction surface with overlapping surface patches, each of which is described with a three-dimensional Zernike descriptor (3DZD), a compact mathematical representation of 3D function. 3DZD captures both the 3D shape and physicochemical properties of the protein surface. The performance of protein-protein interaction-Surfer was benchmarked on datasets of protein-protein interactions, where we were able to show that protein-protein interaction-Surfer finds similar potential drug binding regions that do not share sequence and structure similarity. protein-protein interaction-Surfer is available at https://kiharalab.org/ppi-surfer.

[end paper 51]

[begin paper 52]

PMCID = 11055898

Title = The hinge-engineered IgG1-IgG3 hybrid subclass IgGh47 potently enhances Fc-mediated function of anti-streptococcal and SARS-CoV-2 antibodies

Streptococcus pyogenes can cause invasive disease with high mortality despite adequate antibiotic treatments. To address this unmet need, we have previously generated an opsonic IgG1 monoclonal antibody, Ab25, targeting the bacterial M protein. Here, we engineer the IgG2-4 subclasses of Ab25. Despite having reduced binding, the IgG3 version promotes stronger phagocytosis of bacteria. Using atomic simulations, we show that IgG3’s Fc tail has extensive movement in 3D space due to its extended hinge region, possibly facilitating interactions with immune cells. We replaced the hinge of IgG1 with four different IgG3-hinge segment subclasses, IgGhxx. Hinge-engineering does not diminish binding as with IgG3 but enhances opsonic function, where a 47 amino acid hinge is comparable to IgG3 in function. IgGh47 shows improved protection against S. pyogenes in a systemic infection mouse model, suggesting that IgGh47 has promise as a preclinical therapeutic candidate. Importantly, the enhanced opsonic function of IgGh47 is generalizable to diverse S. pyogenes strains from clinical isolates. We generated IgGh47 versions of anti-SARS-CoV-2 mAbs to broaden the biological applicability, and these also exhibit strongly enhanced opsonic function compared to the IgG1 subclass. The improved function of the IgGh47 subclass in two distant biological systems provides new insights into antibody function.

Here, the authors elongated the hinge structure of IgG1 monoclonal antibodies. The modified IgG1-IgG3 hybrid subclass showed enhanced Fc-mediated function compared to IgG1 in two distinct biological systems, Streptococcus pyogenes and SARS-CoV-2.

[end paper 52]

[begin paper 53]

PMCID = 8697837

Title = Single point mutations can potentially enhance infectivity of SARS-CoV-2 revealed by in silico affinity maturation and SPR assay†

The RBD (receptor binding domain) of the SARS-CoV-2 virus S (spike) protein mediates viral cell attachment and serves as a promising target for therapeutics development. Mutations on the S-RBD may alter its affinity to the cell receptor and affect the potency of vaccines and antibodies. Here we used an in silico approach to predict how mutations on RBD affect its binding affinity to hACE2 (human angiotensin-converting enzyme2). The effect of all single point mutations on the interface was predicted. SPR assay results show that 6 out of 9 selected mutations can strengthen binding affinity. Our prediction has reasonable agreement with the previous deep mutational scan results and recently reported mutants. Our work demonstrated the in silico method as a powerful tool to forecast more powerful virus mutants, which will significantly benefit the development of broadly neutralizing vaccine and antibody.

The RBD (receptor binding domain) of the SARS-CoV-2 virus S (spike) protein mediates viral cell attachment and serves as a promising target for therapeutics development.

[end paper 53]

[begin paper 54]

PMCID = 9294430

Title = Characterizing and explaining the impact of disease-associated mutations in proteins without known structures or structural homologs

Mutations in human proteins lead to diseases. The structure of these proteins can help understand the mechanism of such diseases and develop therapeutics against them. With improved deep learning techniques, such as RoseTTAFold and AlphaFold, we can predict the structure of proteins even in the absence of structural homologs. We modeled and extracted the domains from 553 disease-associated human proteins without known protein structures or close homologs in the Protein Databank. We noticed that the model quality was higher and the Root mean square deviation (RMSD) lower between AlphaFold and RoseTTAFold models for domains that could be assigned to CATH families as compared to those which could only be assigned to Pfam families of unknown structure or could not be assigned to either. We predicted ligand-binding sites, protein–protein interfaces and conserved residues in these predicted structures. We then explored whether the disease-associated missense mutations were in the proximity of these predicted functional sites, whether they destabilized the protein structure based on ddG calculations or whether they were predicted to be pathogenic. We could explain 80% of these disease-associated mutations based on proximity to functional sites, structural destabilization or pathogenicity. When compared to polymorphisms, a larger percentage of disease-associated missense mutations were buried, closer to predicted functional sites, predicted as destabilizing and pathogenic. Usage of models from the two state-of-the-art techniques provide better confidence in our predictions, and we explain 93 additional mutations based on RoseTTAFold models which could not be explained based solely on AlphaFold models.

[end paper 54]

[begin paper 55]

PMCID = 8238390

Title = Learning the Protein Language: Evolution, Structure and Function

Language models have recently emerged as a powerful machine learning approach for distilling information from massive protein sequence databases. From readily available sequence data alone, these models discover evolutionary, structural, and functional organization across protein space. Using language models, we can encode amino acid sequences into distributed vector representations that capture their structural and functional properties as well as evaluate the evolutionary fitness of sequence variants. We discuss recent advances in protein language modeling and their applications to downstream protein property prediction problems. We then consider how these models can be enriched with prior biological knowledge and introduce an approach for encoding protein structural knowledge into the learned representations. The knowledge distilled by these models allows us to improve downstream function prediction through transfer learning. Deep protein language models are revolutionizing protein biology. They suggest new ways to approach protein and therapeutic design. However, further developments are needed to encode strong biological priors into protein language models and to increase their accessibility to the broader community.

[end paper 55]

[begin paper 56]

PMCID = 11437503

Title = AlphaFold2 Modeling and Molecular Dynamics Simulations of the Conformational Ensembles for the SARS-CoV-2 Spike Omicron JN.1, KP.2 and KP.3 Variants: Mutational Profiling of Binding Energetics Reveals Epistatic Drivers of the ACE2 Affinity and Escape Hotspots of Antibody Resistance

The most recent wave of SARS-CoV-2 Omicron variants descending from BA.2 and BA.2.86 exhibited improved viral growth and fitness due to convergent evolution of functional hotspots. These hotspots operate in tandem to optimize both receptor binding for effective infection and immune evasion efficiency, thereby maintaining overall viral fitness. The lack of molecular details on structure, dynamics and binding energetics of the latest FLiRT and FLuQE variants with the ACE2 receptor and antibodies provides a considerable challenge that is explored in this study. We combined AlphaFold2-based atomistic predictions of structures and conformational ensembles of the SARS-CoV-2 spike complexes with the host receptor ACE2 for the most dominant Omicron variants JN.1, KP.1, KP.2 and KP.3 to examine the mechanisms underlying the role of convergent evolution hotspots in balancing ACE2 binding and antibody evasion. Using the ensemble-based mutational scanning of the spike protein residues and computations of binding affinities, we identified binding energy hotspots and characterized the molecular basis underlying epistatic couplings between convergent mutational hotspots. The results suggested the existence of epistatic interactions between convergent mutational sites at L455, F456, Q493 positions that protect and restore ACE2-binding affinity while conferring beneficial immune escape. To examine immune escape mechanisms, we performed structure-based mutational profiling of the spike protein binding with several classes of antibodies that displayed impaired neutralization against BA.2.86, JN.1, KP.2 and KP.3. The results confirmed the experimental data that JN.1, KP.2 and KP.3 harboring the L455S and F456L mutations can significantly impair the neutralizing activity of class 1 monoclonal antibodies, while the epistatic effects mediated by F456L can facilitate the subsequent convergence of Q493E changes to rescue ACE2 binding. Structural and energetic analysis provided a rationale to the experimental results showing that BD55-5840 and BD55-5514 antibodies that bind to different binding epitopes can retain neutralizing efficacy against all examined variants BA.2.86, JN.1, KP.2 and KP.3. The results support the notion that evolution of Omicron variants may favor emergence of lineages with beneficial combinations of mutations involving mediators of epistatic couplings that control balance of high ACE2 affinity and immune evasion.

[end paper 56]

[begin paper 57]

PMCID = 9387132

Title = Computational pipeline provides mechanistic understanding of Omicron variant of concern neutralizing engineered ACE2 receptor traps

The SARS-CoV-2 Omicron variant, with 15 mutations in Spike receptor binding domain (Spike-RBD), renders virtually all clinical monoclonal antibodies against WT SARS-CoV-2 ineffective. We recently engineered the SARS-CoV-2 host entry receptor, ACE2, to tightly bind WT-Spike-RBD and prevent viral entry into host cells (“receptor traps”). Here we determine cryo-EM structures of our receptor traps in complex with full length Spike. We develop a multi-model pipeline combining Rosetta protein modeling software and cryo-EM to allow interface energy calculations even at limited resolution and identify interface side chains that allow for high affinity interactions between our ACE2 receptor traps and Spike-RBD. Our structural analysis provides a mechanistic rationale for the high affinity (0.53 – 4.2nM) binding of our ACE2 receptor traps to Omicron-RBD confirmed with biolayer interferometry measurements. Finally, we show that ACE2 receptor traps potently neutralize Omicron- and Delta- pseudotyped viruses, providing alternative therapeutic routes to combat this evolving virus.

[end paper 57]

[begin paper 58]

PMCID = 10975102

Title = Production and Immunogenicity Assessment of LTp50: An Escherichia coli-Made Chimeric Antigen Targeting S1- and S2-Epitopes from the SARS-CoV-2/BA.5 Spike Protein

Subunit vaccines stand as a leading approach to expanding the current portfolio of vaccines to fight against COVID-19, seeking not only to lower costs but to achieve long-term immunity against variants of concern and have the main attributes that could overcome the limitations of the current vaccines. Herein a chimeric protein targeting S1 and S2 epitopes, called LTp50, was designed as a convenient approach to induce humoral responses against SARS-CoV-2. LTp50 was produced in recombinant Escherichia coli using a conventional pET vector, recovering the expected antigen in the insoluble fraction. LTp50 was purified by chromatography (purity > 90%). The solubilization and refolding stages helped to obtain a stable protein amenable for vaccine formulation. LTp50 was adsorbed onto alum, resulting in a stable formulation whose immunogenic properties were assessed in BALB/c mice. Significant humoral responses against the S protein (BA.5 variant) were detected in mice subjected to three subcutaneous doses (10 µg) of the LTp50/alum formulation. This study opens the path for the vaccine formulation optimization using additional adjuvants to advance in the development of a highly effective anti-COVID-19 vaccine directed against the antigenic regions of the S protein, which are less prone to mutations.

[end paper 58]

[begin paper 59]

PMCID = 8629191

Title = Stabilization of the SARS-CoV-2 Receptor Binding Domain by Protein Core Redesign and Deep Mutational Scanning

Stabilizing antigenic proteins as vaccine immunogens or diagnostic reagents is a stringent case of protein engineering and design as the exterior surface must maintain recognition by receptor(s) and antigen—specific antibodies at multiple distinct epitopes. This is a challenge, as stability-enhancing mutations must be focused on the protein core, whereas successful computational stabilization algorithms typically select mutations at solvent-facing positions. In this study we report the stabilization of SARS-CoV-2 Wuhan Hu-1 Spike receptor binding domain (S RBD) using a combination of deep mutational scanning and computational design, including the FuncLib algorithm. Our most successful design encodes I358F, Y365W, T430I, and I513L RBD mutations, maintains recognition by the receptor ACE2 and a panel of different anti-RBD monoclonal antibodies, is between 1–2°C more thermally stable than the original RBD using a thermal shift assay, and is less proteolytically sensitive to chymotrypsin and thermolysin than the original RBD. Our approach could be applied to the computational stabilization of a wide range of proteins without requiring detailed knowledge of active sites or binding epitopes, particularly powerful for cases when there are multiple or unknown binding sites.

[end paper 59]

[begin paper 60]

PMCID = 7820290

Title = The SARS-CoV-2 nucleocapsid phosphoprotein forms mutually exclusive condensates with RNA and the membrane-associated M protein

The multifunctional nucleocapsid (N) protein in SARS-CoV-2 binds the ~30 kb viral RNA genome to aid its packaging into the 80–90 nm membrane-enveloped virion. The N protein is composed of N-terminal RNA-binding and C-terminal dimerization domains that are flanked by three intrinsically disordered regions. Here we demonstrate that the N protein’s central disordered domain drives phase separation with RNA, and that phosphorylation of an adjacent serine/arginine rich region modulates the physical properties of the resulting condensates. In cells, N forms condensates that recruit the stress granule protein G3BP1, highlighting a potential role for N in G3BP1 sequestration and stress granule inhibition. The SARS-CoV-2 membrane (M) protein independently induces N protein phase separation, and three-component mixtures of N + M + RNA form condensates with mutually exclusive compartments containing N + M or N + RNA, including annular structures in which the M protein coats the outside of an N + RNA condensate. These findings support a model in which phase separation of the SARS-CoV-2 N protein contributes both to suppression of the G3BP1-dependent host immune response and to packaging genomic RNA during virion assembly.

The SARS-CoV-2 nucleocapsid (N) protein binds the viral RNA genome and contains two ordered domains flanked by three intrinsically-disordered regions. Here, the authors show that RNA binding induces liquid-liquid phase separation of N, which is driven by its central intrinsically-disordered region and is modulated by phosphorylation. The SARS-CoV-2 Membrane (M) protein also phase-separates with N, and three-component mixtures of N + M + RNA form mutually exclusive compartments containing N + M or N + RNA.

[end paper 60]

[begin paper 61]

PMCID = 8765073

Title = Computational design of a neutralizing antibody with picomolar binding affinity for all concerning SARS-CoV-2 variants

Coronavirus disease 2019, caused by SARS-CoV-2, remains an on-going pandemic, partly due to the emergence of variant viruses that can “break-through” the protection of the current vaccines and neutralizing antibodies (nAbs), highlighting the needs for broadly nAbs and next-generation vaccines. We report an antibody that exhibits breadth and potency in binding the receptor-binding domain (RBD) of the virus spike glycoprotein across SARS coronaviruses. Initially, a lead antibody was computationally discovered and crystallographically validated that binds to a highly conserved surface of the RBD of wild-type SARS-CoV-2. Subsequently, through experimental affinity enhancement and computational affinity maturation, it was further developed to bind the RBD of all concerning SARS-CoV-2 variants, SARS-CoV-1 and pangolin coronavirus with pico-molar binding affinities, consistently exhibited strong neutralization activity against wild-type SARS-CoV-2 and the Alpha and Delta variants. These results identify a vulnerable target site on coronaviruses for development of pan-sarbecovirus nAbs and vaccines.

[end paper 61]

[begin paper 62]

PMCID = 8007013

Title = Gene variants of coagulation related proteins that interact with SARS-CoV-2

Thrombosis is a recognized complication of Coronavirus disease of 2019 (COVID-19) and is often associated with poor prognosis. There is a well-recognized link between coagulation and inflammation, however, the extent of thrombotic events associated with COVID-19 warrants further investigation. Poly(A) Binding Protein Cytoplasmic 4 (PABPC4), Serine/Cysteine Proteinase Inhibitor Clade G Member 1 (SERPING1) and Vitamin K epOxide Reductase Complex subunit 1 (VKORC1), which are all proteins linked to coagulation, have been shown to interact with SARS proteins. We computationally examined the interaction of these with SARS-CoV-2 proteins and, in the case of VKORC1, we describe its binding to ORF7a in detail. We examined the occurrence of variants of each of these proteins across populations and interrogated their potential contribution to COVID-19 severity. Potential mechanisms, by which some of these variants may contribute to disease, are proposed. Some of these variants are prevalent in minority groups that are disproportionally affected by severe COVID-19. Therefore, we are proposing that further investigation around these variants may lead to better understanding of disease pathogenesis in minority groups and more informed therapeutic approaches.

Increased blood clotting, especially in the lungs, is a common complication of COVID-19. Infectious diseases cause inflammation, which in turn can contribute to increased blood clotting. However, the extent of clot formation that is seen in the lungs of COVID-19 patients suggests that there may be a more direct link. We identified three human proteins that are involved indirectly in the blood clotting cascade and have been shown to interact with proteins of SARS virus, which is closely related to the novel coronavirus. We examined computationally the interaction of these human proteins with the viral proteins. We looked for genetic variants of these proteins and examined how they are distributed across populations. We investigated whether variants of these genes could impact severity of COVID-19. Further investigation around these variants may provide clues for the pathogenesis of COVID-19, particularly in minority groups.

[end paper 62]

[begin paper 63]

PMCID = 7773544

Title = Computational design of SARS-CoV-2 spike glycoproteins to increase immunogenicity by T cell epitope engineering

The development of effective and safe vaccines is the ultimate way to efficiently stop the ongoing COVID-19 pandemic, which is caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Built on the fact that SARS-CoV-2 utilizes the association of its Spike (S) protein with the human angiotensin-converting enzyme 2 (ACE2) receptor to invade host cells, we computationally redesigned the S protein sequence to improve its immunogenicity and antigenicity. Toward this purpose, we extended an evolutionary protein design algorithm, EvoDesign, to create thousands of stable S protein variants that perturb the core protein sequence but keep the surface conformation and B cell epitopes. The T cell epitope content and similarity scores of the perturbed sequences were calculated and evaluated. Out of 22,914 designs with favorable stability energy, 301 candidates contained at least two pre-existing immunity-related epitopes and had promising immunogenic potential. The benchmark tests showed that, although the epitope restraints were not included in the scoring function of EvoDesign, the top S protein design successfully recovered 31 out of the 32 major histocompatibility complex (MHC)-II T cell promiscuous epitopes in the native S protein, where two epitopes were present in all seven human coronaviruses. Moreover, the newly designed S protein introduced nine new MHC-II T cell promiscuous epitopes that do not exist in the wildtype SARS-CoV-2. These results demonstrated a new and effective avenue to enhance a target protein’s immunogenicity using rational protein design, which could be applied for new vaccine design against COVID-19 and other pathogens.

[end paper 63]

[begin paper 64]

PMCID = 8852347

Title = CR-I-TASSER: Assemble Protein Structures from Cryo-EM Density Maps using Deep Convolutional Neural Networks

Cryo-electron microscopy (cryo-EM) has become a leading approach for protein structure determination, but it remains challenging to accurately model atomic structures with cryo-EM density maps. We propose a hybrid method, CR-I-TASSER, which integrates deep neural-network learning with I-TASSER assembly simulations for automated cryo-EM structure determination. The method is benchmarked on 778 proteins with simulated and experimental density maps, where CR-I-TASSER constructs models with a correct fold (TM-score>0.5) for 643 targets that is 64% higher than the best of other de novo and refinement-based approaches on high-resolution data samples. Detailed data analyses showed that the major advantage of CR-I-TASSER lies in the deep-learning based  position prediction, which significantly improves the threading template quality and therefore boosts the accuracy of final models through optimized fragment assembly simulations. These results demonstrate a new avenue to determine cryo-EM protein structures with high accuracy and robustness covering various target types and density-map resolutions.

[end paper 64]

[begin paper 65]

PMCID = 9117152

Title = Design of protein-binding proteins from the target structure alone

The design of proteins that bind to a specific site on the surface of a target protein using no information other than the three-dimensional structure of the target remains a challenge. Here we describe a general solution to this problem that starts with a broad exploration of the vast space of possible binding modes to a selected region of a protein surface, and then intensifies the search in the vicinity of the most promising binding modes. We demonstrate the broad applicability of this approach through the de novo design of binding proteins to 12 diverse protein targets with different shapes and surface properties. Biophysical characterization shows that the binders, which are all smaller than 65 amino acids, are hyperstable and, following experimental optimization, bind their targets with nanomolar to picomolar affinities. We succeeded in solving crystal structures of five of the binder–target complexes, and all five closely match the corresponding computational design models. Experimental data on nearly half a million computational designs and hundreds of thousands of point mutants provide detailed feedback on the strengths and limitations of the method and of our current understanding of protein–protein interactions, and should guide improvements of both. Our approach enables the targeted design of binders to sites of interest on a wide variety of proteins for therapeutic and diagnostic applications.

A design pipeline is presented whereby binding proteins can be designed de novo without the need for prior information on binding hotspots or fragments from structures of complexes with binding partners.

[end paper 65]

[begin paper 66]

PMCID = 7430581

Title = Rational Design of SARS-CoV-2 Spike Glycoproteins To Increase Immunogenicity By T Cell Epitope Engineering

The current COVID-19 pandemic caused by SARS-CoV-2 has resulted in millions of confirmed cases and thousands of deaths globally. Extensive efforts and progress have been made to develop effective and safe vaccines against COVID-19. A primary target of these vaccines is the SARS-CoV-2 spike (S) protein, and many studies utilized structural vaccinology techniques to either stabilize the protein or fix the receptor-binding domain at certain states. In this study, we extended an evolutionary protein design algorithm, EvoDesign, to create thousands of stable S protein variants without perturbing the surface conformation and B cell epitopes of the S protein. We then evaluated the mutated S protein candidates based on predicted MHC-II T cell promiscuous epitopes as well as the epitopes’ similarity to human peptides. The presented strategy aims to improve the S protein’s immunogenicity and antigenicity by inducing stronger CD4 T cell response while maintaining the protein’s native structure and function. The top EvoDesign S protein candidate (Design-10705) recovered 31 out of 32 MHC-II T cell promiscuous epitopes in the native S protein, in which two epitopes were present in all seven human coronaviruses. This newly designed S protein also introduced nine new MHC-II T cell promiscuous epitopes and showed high structural similarity to its native conformation. The proposed structural vaccinology method provides an avenue to rationally design the antigen’s structure with increased immunogenicity, which could be applied to the rational design of new COVID-19 vaccine candidates.

[end paper 66]

[begin paper 67]

PMCID = 9523718

Title = Protein Design: From the Aspect of Water Solubility and Stability

Water solubility and structural stability are key merits for proteins defined by the primary sequence and 3D-conformation. Their manipulation represents important aspects of the protein design field that relies on the accurate placement of amino acids and molecular interactions, guided by underlying physiochemical principles. Emulated designer proteins with well-defined properties both fuel the knowledge-base for more precise computational design models and are used in various biomedical and nanotechnological applications. The continuous developments in protein science, increasing computing power, new algorithms, and characterization techniques provide sophisticated toolkits for solubility design beyond guess work. In this review, we summarize recent advances in the protein design field with respect to water solubility and structural stability. After introducing fundamental design rules, we discuss the transmembrane protein solubilization and de novo transmembrane protein design. Traditional strategies to enhance protein solubility and structural stability are introduced. The designs of stable protein complexes and high-order assemblies are covered. Computational methodologies behind these endeavors, including structure prediction programs, machine learning algorithms, and specialty software dedicated to the evaluation of protein solubility and aggregation, are discussed. The findings and opportunities for Cryo-EM are presented. This review provides an overview of significant progress and prospects in accurate protein design for solubility and stability.

[end paper 67]

[begin paper 68]

PMCID = 9269947

Title = Ig-VAE: Generative modeling of protein structure by direct 3D coordinate generation

While deep learning models have seen increasing applications in protein science, few have been implemented for protein backbone generation—an important task in structure-based problems such as active site and interface design. We present a new approach to building class-specific backbones, using a variational auto-encoder to directly generate the 3D coordinates of immunoglobulins. Our model is torsion- and distance-aware, learns a high-resolution embedding of the dataset, and generates novel, high-quality structures compatible with existing design tools. We show that the Ig-VAE can be used with Rosetta to create a computational model of a SARS-CoV2-RBD binder via latent space sampling. We further demonstrate that the model’s generative prior is a powerful tool for guiding computational protein design, motivating a new paradigm under which backbone design is solved as constrained optimization problem in the latent space of a generative model.

Many essential biochemical processes are governed by protein-protein interactions (PPIs), and our ability to make binding proteins that modulate PPIs is crucial to the creation of therapeutics and the study of cell-signaling. One critical aspect of PPI design is to capture protein conformational flexibility. Deep generative models are a class of mathematical models that are able to synthesize novel data from a finite set of training examples. Here, we make advances in computational protein design methodology by developing a deep generative model that creates protein backbones adopting the immunoglobulin fold, which is found in natural binding proteins such as antibodies. While generative models have been powerful in tasks such as image generation, using them to create proteins has remained a challenge. We solve this problem with a new model that allows for the direct generation of novel 3D molecules and show that they are of high chemical accuracy. Generated structures work well with existing protein design methods such as Rosetta, providing access to a large collection of novel immunoglobulin structures. Finally, we present a new protein design framework, called “generative design,” that shows how deep generative models such as ours can be applied to virtually any protein design problem.

[end paper 68]

[begin paper 69]

PMCID = 9934699

Title = Massively parallel protein-protein interaction measurement by sequencing (MP3-seq) enables rapid screening of protein heterodimers

Protein-protein interactions (PPIs) regulate many cellular processes, and engineered PPIs have cell and gene therapy applications. Here we introduce massively parallel protein-protein interaction measurement by sequencing (MP3-seq), an easy-to-use and highly scalable yeast-two-hybrid approach for measuring PPIs. In MP3-seq, DNA barcodes are associated with specific protein pairs, and barcode enrichment can be read by sequencing to provide a direct measure of interaction strength. We show that MP3-seq is highly quantitative and scales to over 100,000 interactions. We apply MP3-seq to characterize interactions between families of rationally designed heterodimers and to investigate elements conferring specificity to coiled-coil interactions. Finally, we predict coiled heterodimer structures using AlphaFold-Multimer (AF-M) and train linear models on physics simulation energy terms to predict MP3-seq values. We find that AF-M and AF-M complex prediction-based models could be valuable for pre-screening interactions, but that measuring interactions experimentally remains necessary to rank their strengths quantitatively.

[end paper 69]

[begin paper 70]

PMCID = 7698023

Title = Methods and Applications of In Silico Aptamer Design and Modeling

Aptamers are nucleic acid analogues of antibodies with high affinity to different targets, such as cells, viruses, proteins, inorganic materials, and coenzymes. Empirical approaches allow the design of in vitro aptamers that bind particularly to a target molecule with high affinity and selectivity. Theoretical methods allow significant expansion of the possibilities of aptamer design. In this study, we review theoretical and joint theoretical-experimental studies dedicated to aptamer design and modeling. We consider aptamers with different targets, such as proteins, antibiotics, organophosphates, nucleobases, amino acids, and drugs. During nucleic acid modeling and in silico design, a full set of in silico methods can be applied, such as docking, molecular dynamics (MD), and statistical analysis. The typical modeling workflow starts with structure prediction. Then, docking of target and aptamer is performed. Next, MD simulations are performed, which allows for an evaluation of the stability of aptamer/ligand complexes and determination of the binding energies with higher accuracy. Then, aptamer/ligand interactions are analyzed, and mutations of studied aptamers made. Subsequently, the whole procedure of molecular modeling can be reiterated. Thus, the interactions between aptamers and their ligands are complex and difficult to understand using only experimental approaches. Docking and MD are irreplaceable when aptamers are studied in silico.

[end paper 70]

[begin paper 71]

PMCID = 3108954

Title = SNPs Occur in Regions with Less Genomic Sequence Conservation

Rates of SNPs (single nucleotide polymorphisms) and cross-species genomic sequence conservation reflect intra- and inter-species variation, respectively. Here, I report SNP rates and genomic sequence conservation adjacent to mRNA processing regions and show that, as expected, more SNPs occur in less conserved regions and that functional regions have fewer SNPs. Results are confirmed using both mouse and human data. Regions include protein start codons, 3′ splice sites, 5′ splice sites, protein stop codons, predicted miRNA binding sites, and polyadenylation sites. Throughout, SNP rates are lower and conservation is higher at regulatory sites. Within coding regions, SNP rates are highest and conservation is lowest at codon position three and the fewest SNPs are found at codon position two, reflecting codon degeneracy for amino acid encoding. Exon splice sites show high conservation and very low SNP rates, reflecting both splicing signals and protein coding. Relaxed constraint on the codon third position is dramatically seen when separating exonic SNP rates based on intron phase. At polyadenylation sites, a peak of conservation and low SNP rate occurs from 30 to 17 nt preceding the site. This region is highly enriched for the sequence AAUAAA, reflecting the location of the conserved polyA signal. miRNA 3′ UTR target sites are predicted incorporating interspecies genomic sequence conservation; SNP rates are low in these sites, again showing fewer SNPs in conserved regions. Together, these results confirm that SNPs, reflecting recent genetic variation, occur more frequently in regions with less evolutionarily conservation.

[end paper 71]

[begin paper 72]

PMCID = 10965422

Title = AbFlex: designing antibody complementarity determining regions with flexible CDR definition

Antibodies are proteins that the immune system produces in response to foreign pathogens. Designing antibodies that specifically bind to antigens is a key step in developing antibody therapeutics. The complementarity determining regions (CDRs) of the antibody are mainly responsible for binding to the target antigen, and therefore must be designed to recognize the antigen.

We develop an antibody design model, AbFlex, that exhibits state-of-the-art performance in terms of structure prediction accuracy and amino acid recovery rate. Furthermore, >38% of newly designed antibody models are estimated to have better binding energies for their antigens than wild types. The effectiveness of the model is attributed to two different strategies that are developed to overcome the difficulty associated with the scarcity of antibody–antigen complex structure data. One strategy is to use an equivariant graph neural network model that is more data-efficient. More importantly, a new data augmentation strategy based on the flexible definition of CDRs significantly increases the performance of the CDR prediction model.

The source code and implementation are available at https://github.com/wsjeon92/AbFlex.

[end paper 72]

[begin paper 73]

PMCID = 7945008

Title = Protein storytelling through physics

Understanding biology, particularly at the level of actionable drug discovery, is often a matter of developing accurate stories about how proteins work. This requires understanding the physics of the system, and physics-based computer modeling is a prime tool for that. However, the computational molecular physics (CMP) of proteins has previously been much too expensive and slow. A large fraction of public supercomputing resources worldwide is currently running CMP simulations of biologically relevant systems. We review here the history and status of this large and diverse scientific enterprise. Among other things, protein modeling has driven major computer hardware advances, such as IBM’s Blue Gene and DE Shaw’s Anton computers. Further, protein modeling has advanced rapidly over 50 years, even slightly faster than Moore’s law. We also review an interesting scientific social construct that has arisen around protein modeling: community-wide blind competitions. They have transformed how we test, validate, and improve our computational models of proteins.

For 50 years, two approaches to computer modeling have been mainstays for developing stories about protein molecules and their biological actions. (i) Inferences from structure-property relations: Based on the principle that a protein’s action depends on its shape, it is possible to use databases of known proteins to learn about unknown proteins. (ii) Computational molecular physics uses force fields of atom-atom interactions, sampled by molecular dynamics (MD), to develop biological action stories that satisfy principles of chemistry and thermodynamics. CMP has traditionally been computationally costly, limited to studying only simple actions of small proteins. But CMP has recently advanced enormously. (i) Force fields and their corresponding solvent models are now sufficiently accurate at capturing the molecular interactions, and conformational searching and sampling methods are sufficiently fast, that CMP is able to model, fairly accurately, protein actions on time scales longer than microseconds, and sometimes milliseconds. So, we are now accessing important biological events, such as protein folding, unbinding, allosteric change, and assembly. (ii) Just as car races do for auto manufacturers, communal blind tests such as protein structure-prediction events are giving protein modelers a shared evaluation venue for improving our methods. CMP methods are now competing and often doing quite well. (iii) New methods are harnessing external information—like experimental structural data—to accelerate CMP, notably, while preserving proper physics.

What are we learning? For one thing, a long-standing hypothesis is that proteins fold by multiple different microscopic routes, a story that is too granular to learn from experiments alone. CMP recently affirmed this principle while giving accurate and testable microscopic details, protein by protein. In addition, CMP is now contributing to physico-chemical drug design. Structure-based methods of drug discovery have long been able to discern what small-molecule drug candidates might bind to a given target protein and where on the protein they might bind. However, such methods don’t reveal some all-important physical properties needed for drug discovery campaigns—the affinities and the on- and off-rates of the ligand binding to the protein. CMP is beginning to compute these properties accurately. A third example is shown in the figure. It shows the spike protein of severe acute respiratory syndrome coronavirus 2(SARS-CoV-2), the causative agent of today’s coronavirus disease 2019 (COVID-19) pandemic. A large, hinge-like movement of this sizable protein is the critical action needed for the virus to enter and infect the human cell. The only way to see the details of this motion—to attempt to block it with drugs—is by CMP. The figure shows CMP simulation results of three dynamical states of this motion.

A cell’s behavior is due to the actions of its thousands of different proteins. Every protein has its own story to tell. CMP is a granular and principled tool that is able to discover those stories. CMP is now being tested and improved through blind communal validations. It is attacking ever larger proteins, exploring increasingly bigger and slower motions, and with ever more accurate physics. We are reaching a physical understanding of biology at the microscopic level as CMP reveals causations and forces, step-by-step actions in space and time, conformational distributions along the way, and important physical quantities such as free energies, rates, and equilibrium constants.■

CMP modeling of COVID-19 infecting the human cell. SARS-CoV-2 spike glycoprotein (green, with its glycan shield in yellow) attaching to the human angiotensin-converting enzyme 2 (ACE2) receptor protein (purple) through its spike receptor-binding domain (red). (Left) The receptor binding domain (RBD) is hidden. (Middle) The RBD is open and accessible. (Right) The RBD binds human ACE2 receptor. This is followed by a cascade of larger conformational changes in the spike protein, leading to viral fusion to the human host cell.

Every protein has a story—how it folds, what it binds, its biological actions, and how it misbehaves in aging or disease. Stories are often inferred from a protein’s shape (i.e., its structure). But increasingly, stories are told using computational molecular physics (CMP). CMP is rooted in the principled physics of driving forces and reveals granular detail of conformational populations in space and time. Recent advances are accessing longer time scales, larger actions, and blind testing, enabling more of biology’s stories to be told in the language of atomistic physics.

[end paper 73]

[begin paper 74]

PMCID = 6302666

Title = Cryo-EM Structures of Eastern Equine Encephalitis Virus Reveal Mechanisms of Virus Disassembly and Antibody Neutralization

Alphaviruses are enveloped pathogens that cause arthritis and encephalitis. Here, we report a 4.4-Å cryoelectron microscopy (cryo-EM) structure of eastern equine encephalitis virus (EEEV), an alphavirus that causes fatal encephalitis in humans. Our analysis provides insights into viral entry into host cells. The envelope protein E2 showed a binding site for the cellular attachment factor heparan sulfate. The presence of a cryptic E2 glycan suggests how EEEV escapes surveillance by lectin-expressing myeloid lineage cells, which are sentinels of the immune system. A mechanism for nucleocapsid core release and disassembly upon viral entry was inferred based on pH changes and capsid dissociation from envelope proteins. The EEEV capsid structure showed a viral RNA genome binding site adjacent to a ribosome binding site for viral genome translation following genome release. Using five Fab-EEEV complexes derived from neutralizing antibodies, our investigation provides insights into EEEV host cell interactions and protective epitopes relevant to vaccine design.

EEEV cryo-EM structure shows the basis of receptor binding and pH-triggered disassembly

Cryptic envelope protein glycosylation interferes with immune detection

EEEV RNA genome binding site on capsid protein has an extended conformation

Antibody inhibition of EEEV entry involves cross-linking of viral envelope proteins

Hasan et al. use single-particle cryoelectron microscopy to elucidate the molecular basis of host cell entry of neurovirulent EEEV. They show that the EEEV envelope is primed for intracellular pH sensing and subsequent disassembly. Monoclonal antibodies effectively inhibit EEEV entry by cross-linking the viral envelope.

[end paper 74]

[begin paper 75]

PMCID = 9302003

Title = Structural Features of Antibody-Peptide Recognition

Antibody recognition of antigens is a critical element of adaptive immunity. One key class of antibody-antigen complexes is comprised of antibodies targeting linear epitopes of proteins, which in some cases are conserved elements of viruses and pathogens of relevance for vaccine design and immunotherapy. Here we report a detailed analysis of the structural and interface features of this class of complexes, based on a set of nearly 200 nonredundant high resolution antibody-peptide complex structures that were assembled from the Protein Data Bank. We found that antibody-bound peptides adopt a broad range of conformations, often displaying limited secondary structure, and that the same peptide sequence bound by different antibodies can in many cases exhibit varying conformations. Propensities of contacts with antibody loops and extent of antibody binding conformational changes were found to be broadly similar to those for antibodies in complex with larger protein antigens. However, antibody-peptide interfaces showed lower buried surface areas and fewer hydrogen bonds than antibody-protein antigen complexes, while calculated binding energy per buried interface area was found to be higher on average for antibody-peptide interfaces, likely due in part to a greater proportion of buried hydrophobic residues and higher shape complementarity. This dataset and these observations can be of use for future studies focused on this class of interactions, including predictive computational modeling efforts and the design of antibodies or epitope-based vaccine immunogens.

[end paper 75]

[begin paper 76]

PMCID = 9126319

Title = Advances to tackle backbone flexibility in protein docking

Computational docking methods can provide structural models of protein–protein complexes, but protein backbone flexibility upon association often thwarts accurate predictions. In recent blind challenges, medium or high accuracy models were submitted in less than 20% of the ‘difficult’ targets (with significant backbone change or uncertainty). Here, we describe recent developments in protein–protein docking and highlight advances that tackle backbone flexibility. In molecular dynamics and Monte Carlo approaches, enhanced sampling techniques have reduced time-scale limitations. Internal coordinate formulations can now capture realistic motions of monomers and complexes using harmonic dynamics. And machine learning approaches adaptively guide docking trajectories or generate novel binding site predictions from deep neural networks trained on protein interfaces. These tools poise the field to break through the longstanding challenge of correctly predicting complex structures with significant conformational change.

[end paper 76]

[begin paper 77]

PMCID = 7756312

Title = Potentially adaptive SARS-CoV-2 mutations discovered with novel spatiotemporal and explainable AI models

A mechanistic understanding of the spread of SARS-CoV-2 and diligent tracking of ongoing mutagenesis are of key importance to plan robust strategies for confining its transmission. Large numbers of available sequences and their dates of transmission provide an unprecedented opportunity to analyze evolutionary adaptation in novel ways. Addition of high-resolution structural information can reveal the functional basis of these processes at the molecular level. Integrated systems biology-directed analyses of these data layers afford valuable insights to build a global understanding of the COVID-19 pandemic.

Here we identify globally distributed haplotypes from 15,789 SARS-CoV-2 genomes and model their success based on their duration, dispersal, and frequency in the host population. Our models identify mutations that are likely compensatory adaptive changes that allowed for rapid expansion of the virus. Functional predictions from structural analyses indicate that, contrary to previous reports, the Asp614Gly mutation in the spike glycoprotein (S) likely reduced transmission and the subsequent Pro323Leu mutation in the RNA-dependent RNA polymerase led to the precipitous spread of the virus. Our model also suggests that two mutations in the nsp13 helicase allowed for the adaptation of the virus to the Pacific Northwest of the USA. Finally, our explainable artificial intelligence algorithm identified a mutational hotspot in the sequence of S that also displays a signature of positive selection and may have implications for tissue or cell-specific expression of the virus.

These results provide valuable insights for the development of drugs and surveillance strategies to combat the current and future pandemics.

[end paper 77]

[begin paper 78]

PMCID = 10163288

Title = Improving de novo protein binder design with deep learning

Recently it has become possible to de novo design high affinity protein binding proteins from target structural information alone. There is, however, considerable room for improvement as the overall design success rate is low. Here, we explore the augmentation of energy-based protein binder design using deep learning. We find that using AlphaFold2 or RoseTTAFold to assess the probability that a designed sequence adopts the designed monomer structure, and the probability that this structure binds the target as designed, increases design success rates nearly 10-fold. We find further that sequence design using ProteinMPNN rather than Rosetta considerably increases computational efficiency.

Recently, a pipeline for the design of protein-binding proteins using only the structure of the target protein was reported. Here, the authors report that the incorporation of deep learning methods into the original pipeline increases experimental success rate by ten-fold.

[end paper 78]

[begin paper 79]

PMCID = 7418955

Title = Structural and functional evaluation of de novo-designed, two-component nanoparticle carriers for HIV Env trimer immunogens

Two-component, self-assembling nanoparticles represent a versatile platform for multivalent presentation of viral antigens. Computational design of protein nanoparticles with differing sizes and geometries enables combination with antigens of choice to test novel multimerization concepts in immunization strategies where the goal is to improve the induction and maturation of neutralizing antibody lineages. Here, we describe detailed antigenic, structural, and functional characterization of computationally designed tetrahedral, octahedral, and icosahedral nanoparticle immunogens displaying trimeric HIV envelope glycoprotein (Env) ectodomains. Env trimers, based on subtype A (BG505) or consensus group M (ConM) sequences and engineered with SOSIP stabilizing mutations, were fused to an underlying trimeric building block of each nanoparticle. Initial screening yielded one icosahedral and two tetrahedral nanoparticle candidates, capable of presenting twenty or four copies of the Env trimer. A number of analyses, including detailed structural characterization by cryo-EM, demonstrated that the nanoparticle immunogens possessed the intended structural and antigenic properties. When the immunogenicity of ConM-SOSIP trimers presented on a two-component tetrahedral nanoparticle or as soluble proteins were compared in rabbits, the two immunogens elicited similar serum antibody binding titers against the trimer component. Neutralizing antibody titers were slightly elevated in the animals given the nanoparticle immunogen and were initially more focused to the trimer apex. Altogether, our findings indicate that tetrahedral nanoparticles can be successfully applied for presentation of HIV Env trimer immunogens; however, the optimal implementation to different immunization strategies remains to be determined.

Protein constructs based on soluble ectodomains of HIV glycoprotein (Env) trimers are the basis of many current HIV vaccine platforms. Multivalent antigen display is one strategy applied to improve the immunogenicity of various subunit vaccine candidates. Here, we describe and comprehensively evaluate a library of de novo designed protein nanoparticles of different geometries for their ability to present trimeric Env antigens. We found three nanoparticle candidates that can stably incorporate model Env trimers on their surfaces while maintaining structure and antigenicity. The designed nanoparticle immunogens had an increased capacity to stimulate B-cells expressing antigen-specific receptors. The immunogenicity of one nanoparticle candidate was assessed in rabbits. Nanoparticle presentation geometry appeared to alter the distribution of antibody responses against different epitopes while inducing similar serum binding titers and only slightly elevated neutralizing titers. In addition to introducing a novel set of reagents for multivalent display of Env trimers, this work provides both guiding principles and a detailed experimental roadmap for the generation, characterization, and optimization of Env-presenting, self-assembling nanoparticle immunogens.

[end paper 79]

[begin paper 80]

PMCID = 11040732

Title = ALDELE: All-Purpose Deep Learning Toolkits for Predicting the Biocatalytic Activities of Enzymes

Rapidly predicting enzyme properties for catalyzing specific substrates is essential for identifying potential enzymes for industrial transformations. The demand for sustainable production of valuable industry chemicals utilizing biological resources raised a pressing need to speed up biocatalyst screening using machine learning techniques. In this research, we developed an all-purpose deep-learning-based multiple-toolkit (ALDELE) workflow for screening enzyme catalysts. ALDELE incorporates both structural and sequence representations of proteins, alongside representations of ligands by subgraphs and overall physicochemical properties. Comprehensive evaluation demonstrated that ALDELE can predict the catalytic activities of enzymes, and particularly, it identifies residue-based hotspots to guide enzyme engineering and generates substrate heat maps to explore the substrate scope for a given biocatalyst. Moreover, our models notably match empirical data, reinforcing the practicality and reliability of our approach through the alignment with confirmed mutation sites. ALDELE offers a facile and comprehensive solution by integrating different toolkits tailored for different purposes at affordable computational cost and therefore would be valuable to speed up the discovery of new functional enzymes for their exploitation by the industry.

[end paper 80]

[begin paper 81]

PMCID = 8642831

Title = SARS-CoV-2 nucleocapsid protein binds host mRNAs and attenuates stress granules to impair host stress response

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) nucleocapsid (N) protein is essential for viral replication, making it a promising target for antiviral drug and vaccine development. SARS-CoV-2 infected patients exhibit an uncoordinated immune response; however, the underlying mechanistic details of this imbalance remain obscure. Here, starting from a functional proteomics workflow, we cataloged the protein–protein interactions of SARS-CoV-2 proteins, including an evolutionarily conserved specific interaction of N with the stress granule resident proteins G3BP1 and G3BP2. N localizes to stress granules and sequesters G3BPs away from their typical interaction partners, thus attenuating stress granule formation. We found that N binds directly to host mRNAs in cells, with a preference for 3′ UTRs, and modulates target mRNA stability. We show that the N protein rewires the G3BP1 mRNA-binding profile and suppresses the physiological stress response of host cells, which may explain the imbalanced immune response observed in SARS-CoV-2 infected patients.

AP-MS identifies 753 viral-host protein–protein interactions for 27 SARS-CoV-2 proteins

SARS-CoV-2 N protein sequesters G3BP1/2 and attenuates stress granule formation

N binds directly to mRNAs, rewires G3BP1 mRNA targets, and modulates mRNA stability

SARS-CoV-2 N dampens host stress response via altering posttranscriptional programs

Molecular biology; Virology; Cell biology; Proteomics;

[end paper 81]

[begin paper 82]

PMCID = 9320768

Title = Determining the International Spread of B.1.1.523 SARS-CoV-2 Lineage with a Set of Mutations Highly Associated with Reduced Immune Neutralization

Here, we report the emergence of the variant lineage B.1.1.523 that contains a set of mutations including 156_158del, E484K and S494P in the spike protein. E484K and S494P are known to significantly reduce SARS-CoV-2 neutralization by convalescent and vaccinated sera and are considered as mutations of concern. Lineage B.1.1.523 presumably originated in the Russian Federation and spread across European countries with the peak of transmission in April–May 2021. The B.1.1.523 lineage has now been reported from 31 countries. In this article, we analyze the possible origin of this mutation subset and its immune response using in silico methods.

[end paper 82]

[begin paper 83]

PMCID = 9281400

Title = From Protein Design to the Energy Landscape of a Cold Unfolding Protein

Understanding protein folding is crucial for protein sciences. The conformational spaces and energy landscapes of cold (unfolded) protein states, as well as the associated transitions, are hardly explored. Furthermore, it is not known how structure relates to the cooperativity of cold transitions, if cold and heat unfolded states are thermodynamically similar, and if cold states play important roles for protein function. We created the cold unfolding 4-helix bundle DCUB1 with a de novo designed bipartite hydrophilic/hydrophobic core featuring a hydrogen bond network which extends across the bundle in order to study the relative importance of hydrophobic versus hydrophilic protein–water interactions for cold unfolding. Structural and thermodynamic characterization resulted in the discovery of a complex energy landscape for cold transitions, while the heat unfolded state is a random coil. Below ∼0 °C, the core of DCUB1 disintegrates in a largely cooperative manner, while a near-native helical content is retained. The resulting cold core-unfolded state is compact and features extensive internal dynamics. Below −5 °C, two additional cold transitions are seen, that is, (i) the formation of a water-mediated, compact, and highly dynamic dimer, and (ii) the onset of cold helix unfolding decoupled from cold core unfolding. Our results suggest that cold unfolding is initiated by the intrusion of water into the hydrophilic core network and that cooperativity can be tuned by varying the number of core hydrogen bond networks. Protein design has proven to be invaluable to explore the energy landscapes of cold states and to robustly test related theories.

[end paper 83]

[begin paper 84]

PMCID = 10131520

Title = De novo design of protein interactions with learned surface fingerprints

Physical interactions between proteins are essential for most biological processes governing life. However, the molecular determinants of such interactions have been challenging to understand, even as genomic, proteomic and structural data increase. This knowledge gap has been a major obstacle for the comprehensive understanding of cellular protein–protein interaction networks and for the de novo design of protein binders that are crucial for synthetic biology and translational applications. Here we use a geometric deep-learning framework operating on protein surfaces that generates fingerprints to describe geometric and chemical features that are critical to drive protein–protein interactions. We hypothesized that these fingerprints capture the key aspects of molecular recognition that represent a new paradigm in the computational design of novel protein interactions. As a proof of principle, we computationally designed several de novo protein binders to engage four protein targets: SARS-CoV-2 spike, PD-1, PD-L1 and CTLA-4. Several designs were experimentally optimized, whereas others were generated purely in silico, reaching nanomolar affinity with structural and mutational characterization showing highly accurate predictions. Overall, our surface-centric approach captures the physical and chemical determinants of molecular recognition, enabling an approach for the de novo design of protein interactions and, more broadly, of artificial proteins with function.

A surface-centric approach captures the physical and chemical determinants of molecular recognition, enabling the de novo design of protein interactions and of artificial proteins with function.

[end paper 84]

[begin paper 85]

PMCID = 9221694

Title = Evaluation of a Maternal Plasma RNA Panel Predicting Spontaneous Preterm Birth and Its Expansion to the Prediction of Preeclampsia

Preterm birth is the principal contributor to neonatal death and morbidity worldwide. We previously described a plasma cell-free RNA panel that between 16 and 20 weeks of pregnancy had potential to predict spontaneous preterm birth (sPTB) ≤ 32 weeks caused by preterm labor (PTL) or preterm premature rupture of membranes (PPROM). The present study had three objectives: (1) estimate the RNA panel prognostic accuracy for PTL/PPROM ≤ 32 weeks in a larger series; (2) improve accuracy by adding clinical characteristics to the predictive model; and (3) examine the association of the RNA panel with preeclampsia. We studied 289 women from Memphis TN prospectively sampled 16.0–20.7 weeks and found: (1) PSME2 and Hsa-Let 7g were differentially expressed in cases of PTL/PPROM ≤ 32 weeks and together provided fair predictive accuracy with AUC of 0.76; (2) combining the two RNAs with clinical characteristics improved good predictive accuracy for PTL/PPROM ≤ 32 weeks (AUC 0.83); (3) NAMPT and APOA1 were differentially expressed in women with ‘early-onset preeclampsia’ (EOP) and together provided good predictive accuracy with AUC of 0.89; and (4) combining the two RNAs with clinical characteristics provided excellent predictive accuracy (AUC 0.96). Our findings suggest an underlying common pathophysiological relationship between PTL/PPROM ≤ 32 weeks and EOP and open inroads for the prognostication of high-risk pregnancies.

[end paper 85]

[begin paper 86]

PMCID = 9999731

Title = Integration of immunoinformatics and cheminformatics to design and evaluate a multitope vaccine against Klebsiella pneumoniae and Pseudomonas aeruginosa coinfection

 Introduction: Klebsiella pneumoniae (K. pneumoniae) and Pseudomonas aeruginosa (P. aeruginosa) are the most common Gram-negative bacteria associated with pneumonia and coinfecting the same patient. Despite their high virulence, there is no effective vaccine against them.

 Methods: In the current study, the screening of several proteins from both pathogens highlighted FepA and OmpK35 for K. pneumonia in addition to HasR and OprF from P. aeruginosa as promising candidates for epitope mapping. Those four proteins were linked to form a multitope vaccine, that was formulated with a suitable adjuvant, and PADRE peptides to finalize the multitope vaccine construct. The final vaccine’s physicochemical features, antigenicity, toxicity, allergenicity, and solubility were evaluated for use in humans.

 Results: The output of the computational analysis revealed that the designed multitope construct has passed these assessments with satisfactory scores where, as the last stage, we performed a molecular docking study between the potential vaccine construct and K. pneumonia associated immune receptors, TLR4 and TLR2, showing affinitive to both targets with preferentiality for the TLR4 receptor protein. Validation of the docking studies has proceeded through molecular dynamics simulation, which estimated a strong binding and supported the nomination of the designed vaccine as a putative solution for K. pneumoniae and P. aeruginosa coinfection. Here, we describe the approach for the design and assessment of our potential vaccine.

[end paper 86]

[begin paper 87]

PMCID = 8594574

Title = Computational prediction of the effect of amino acid changes on the binding affinity between SARS-CoV-2 spike RBD and human ACE2

SARS-CoV-2 infection proceeds through the binding of viral surface spike protein to the human ACE2 protein. The global spread of the infection has led to the emergence of fitter and more transmissible variants with increased adaptation both in human and nonhuman hosts. Molecular simulations of the binding event between the spike and ACE2 proteins offer a route to assess potential increase or decrease in infectivity by measuring the change in binding strength. We trained a neural network model that accurately maps simulated binding energies to experimental changes in binding strength upon amino acid changes in the spike protein. This computational workflow can be used to a priori assess currently circulating and prospectively future viral variants for their affinity for hACE2.

The association of the receptor binding domain (RBD) of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein with human angiotensin-converting enzyme 2 (hACE2) represents the first required step for cellular entry. SARS-CoV-2 has continued to evolve with the emergence of several novel variants, and amino acid changes in the RBD have been implicated with increased fitness and potential for immune evasion. Reliably predicting the effect of amino acid changes on the ability of the RBD to interact more strongly with the hACE2 can help assess the implications for public health and the potential for spillover and adaptation into other animals. Here, we introduce a two-step framework that first relies on 48 independent 4-ns molecular dynamics (MD) trajectories of RBD−hACE2 variants to collect binding energy terms decomposed into Coulombic, covalent, van der Waals, lipophilic, generalized Born solvation, hydrogen bonding, π−π packing, and self-contact correction terms. The second step implements a neural network to classify and quantitatively predict binding affinity changes using the decomposed energy terms as descriptors. The computational base achieves a validation accuracy of 82.8% for classifying single–amino acid substitution variants of the RBD as worsening or improving binding affinity for hACE2 and a correlation coefficient of 0.73 between predicted and experimentally calculated changes in binding affinities. Both metrics are calculated using a fivefold cross-validation test. Our method thus sets up a framework for screening binding affinity changes caused by unknown single– and multiple–amino acid changes offering a valuable tool to predict host adaptation of SARS-CoV-2 variants toward tighter hACE2 binding.

[end paper 87]

[begin paper 88]

PMCID = 9401610

Title = Femtomolar detection of SARS-CoV-2 via peptide beacons integrated on a miniaturized TIRF microscope

The novel coronavirus SARS-CoV-2 (severe acute respiratory syndrome coronavirus 2) continues to pose a substantial global health threat. Along with vaccines and targeted therapeutics, there is a critical need for rapid diagnostic solutions. In this work, we use computational protein modeling tools to suggest molecular beacon architectures that function as conformational switches for high-sensitivity detection of the SARS-CoV-2 spike protein receptor binding domain (S-RBD). Integrating these beacons on a miniaturized total internal reflection fluorescence (mini-TIRF) microscope, we detect the S-RBD and pseudotyped SARS-CoV-2 with limits of detection in the femtomolar range. We envision that our designed mini-TIRF platform will serve as a robust platform for point-of-care diagnostics for SARS-CoV-2 and future emergent viral threats.

Computationally validated peptide beacons integrated on mini-TIRF microscopes detect SARS-CoV-2 with femtomolar sensitivity.

[end paper 88]

[begin paper 89]

PMCID = 11657118

Title = Analysis of how antigen mutations disrupt antibody binding interactions toward enabling rapid and reliable antibody repurposing

Antibody repurposing is the process of changing a known antibody so that it binds to a mutated antigen. One of the findings to emerge from the Coronavirus Disease 2019 (COVID-19) pandemic was that it was possible to repurpose neutralizing antibodies for Severe Acute Respiratory Syndrome, a related disease, to work for COVID-19. Thus, antibody repurposing is a possible pathway to prepare for and respond to future pandemics, as well as personalizing cancer therapies. For antibodies to be successfully repurposed, it is necessary to know both how antigen mutations disrupt their binding and how they should be mutated to recover binding, with this work describing an analysis to address the first of these topics. Every possible antigen point mutation in the interface of 246 antibody–protein complexes were analyzed using the Rosetta molecular mechanics force field. The results highlight a number of features of how antigen mutations affect antibody binding, including the effects of mutating critical hotspot residues versus other positions, how many mutations are necessary to be likely to disrupt binding, the prevalence of indirect effects of mutations on binding, and the relative importance of changing attractive versus repulsive energies. These data are expected to be useful in guiding future antibody repurposing experiments.

[end paper 89]

[begin paper 90]

PMCID = 8931377

Title = Deep learning guided optimization of human antibody against SARS-CoV-2 variants with broad neutralization

SARS-CoV-2 continues to evolve through emerging variants, more frequently observed with higher transmissibility. Despite the wide application of vaccines and antibodies, the selection pressure on the Spike protein may lead to further evolution of variants that include mutations that can evade immune response. To catch up with the virus’s evolution, we introduced a deep learning approach to redesign the complementarity-determining regions (CDRs) to target multiple virus variants and obtained an antibody that broadly neutralizes SARS-CoV-2 variants.

The ability of viruses to mutate and evade the human immune system and neutralizing antibodies remains an obstacle to antiviral and vaccine development. Many neutralizing antibodies, including some approved for emergency use authorization (EUA), reduced or lost activity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants. Here, we introduce a geometric deep learning algorithm that efficiently enhances antibody affinity to achieve broader and more potent neutralizing activity against such variants. We demonstrate the utility of our approach on a human antibody P36-5D2, which is effective against SARS-CoV-2 Alpha, Beta, and Gamma but not Delta. We show that our geometric neural network model optimizes this antibody’s complementarity-determining region (CDR) sequences to improve its binding affinity against multiple SARS-CoV-2 variants. Through iterative optimization of the CDR regions and experimental measurements, we enable expanded antibody breadth and improved potency by ∼10- to 600-fold against SARS-CoV-2 variants, including Delta. We have also demonstrated that our approach can identify CDR changes that alleviate the impact of two Omicron mutations on the epitope. These results highlight the power of our deep learning approach in antibody optimization and its potential application to engineering other protein molecules. Our optimized antibodies can potentially be developed into antibody drug candidates for current and emerging SARS-CoV-2 variants.

[end paper 90]

[begin paper 91]

PMCID = 8828479

Title = Broadly neutralizing antibodies target a haemagglutinin anchor epitope

Broadly neutralizing antibodies that target epitopes of haemagglutinin on the influenza virus have the potential to provide near universal protection against influenza virus infection. However, viral mutants that escape broadly neutralizing antibodies have been reported. The identification of broadly neutralizing antibody classes that can neutralize viral escape mutants is critical for universal influenza virus vaccine design. Here we report a distinct class of broadly neutralizing antibodies that target a discrete membrane-proximal anchor epitope of the haemagglutinin stalk domain. Anchor epitope-targeting antibodies are broadly neutralizing across H1 viruses and can cross-react with H2 and H5 viruses that are a pandemic threat. Antibodies that target this anchor epitope utilize a highly restricted repertoire, which encodes two public binding motifs that make extensive contacts with conserved residues in the fusion peptide. Moreover, anchor epitope-targeting B cells are common in the human memory B cell repertoire and were recalled in humans by an oil-in-water adjuvanted chimeric haemagglutinin vaccine, which is a potential universal influenza virus vaccine. To maximize protection against seasonal and pandemic influenza viruses, vaccines should aim to boost this previously untapped source of broadly neutralizing antibodies that are widespread in the human memory B cell pool.

A distinct class of broadly neutralizing antibodies to the influenza virus target a membrane-proximal anchor epitope of the haemagglutinin stalk domain.

[end paper 91]

[begin paper 92]

PMCID = 7939140

Title = One-step sequence and structure-guided optimization of HIV-1 envelope gp140

Stabilization of the metastable envelope glycoprotein (Env) of HIV-1 is hypothesized to improve induction of broadly neutralizing antibodies. We improved the expression yield and stability of the HIV-1 envelope glycoprotein BG505SOSIP.664 gp140 by means of a previously described automated sequence and structure-guided computational thermostabilization approach, PROSS. This combines sequence conservation information with computational assessment of mutant stabilization, thus taking advantage of the extensive natural sequence variation present in HIV-1 Env. PROSS is used to design three gp140 variants with 17–45 mutations relative to the parental construct. One of the designs is experimentally observed to have a fourfold improvement in yield and a 4 °C increment in thermostability. In addition, the designed immunogens have similar antigenicity profiles to the native flexible linker version of wild type, BG505SOSIP.664 gp140 (NFL Wt) to major epitopes targeted by broadly neutralizing antibodies. PROSS eliminates the laborious process of screening many variants for stability and functionality, providing a proof of principle of the method for stabilization and improvement of yield without compromising antigenicity for next generation complex, highly glycosylated vaccine candidates.

One-step stabilization of HIV-1 Env gp140.

One-step yield improvement of HIV-1 Env gp140.

Native-like oligomeric conformation of designed vaccine candidates.

Unaltered antigenicity of designed vaccine candidates.

[end paper 92]

[begin paper 93]

PMCID = 8661510

Title = Computational study, synthesis and evaluation of active peptides derived from Parasporin-2 and spike protein from Alphacoronavirus against colorectal cancer cells

Parasporin-2Aa1 (PS2Aa1) is a toxic protein of 37 KDa (30 kDa, activated form produced by proteolysis) that was shown to be cytotoxic against specific human cancer cells, although its mechanism of action has not been elucidated yet. In order to study the role of some native peptide fragments of proteins on anticancer activity, here we investigated the cytotoxic effect of peptide fragments from domain-1 of PS2Aa1 and one of the loops present in the binding region of the virus spike protein from Alphacoronavirus (HCoV-229E), the latter according to scientific reports, who showed interaction with the human APN (h-APN) receptor, evidence corroborated through computational simulations, and thus being possible active against colon cancer cells. Peptides namely P264-G274, Loop1-PS2Aa, and Loop2-PS2Aa were synthesized using the Fmoc solid-phase synthesis and characterized by mass spectrometry (MS). Additionally, one region from loop 1 of HCoV-229E, Loop1-HCoV-229E, was also synthesized and characterized. The A4W-GGN5 anticancer peptide and 5-fluorouracil (5-FU) were taken as a control in all experiments. Circular dichroism revealed an α-helix structure for the peptides derived from PS2Aa1 (P264-G274, Loop1-PS2Aa, and Loop2-PS2Aa) and β-laminar structure for the peptide derived from Alphacoronavirus spike protein Loop1-HCoV-229E. Peptides showed a hemolysis percentage of less than 20% at 100 µM concentration. Besides, peptides exhibited stronger anticancer activity against SW480 and SW620 cells after exposure for 48 h. Likewise, these compounds showed significantly lower toxicity against normal cells CHO-K1. The results suggest that native peptide fragments from Ps2Aa1 may be optimized as a novel potential cancer-therapeutic agents.

[end paper 93]

[begin paper 94]

PMCID = 9952452

Title = What Can De Novo Protein Design Bring to the Treatment of Hematological Disorders?

Because of the seriousness and complexity of hematological disorders, it is particularly critical to develop new methods for treating them. A protein engineering technique has been used to further enhance therapeutic effects and minimize side effects of protein-based therapeutics. However, the essence of the protein engineering technique is to modify and/or ameliorate natural existing proteins. In recent years, de novo proteins have been developed at a high speed, and their applications in the biomedical field are increasing, including in developing novel diagnostic and therapeutic drugs, novel vaccine techniques, and novel biological materials. At the same time, de novo proteins have also been applied to improve the efficacy of treatment methods in hematological disorders, such as designing the novel structures of chimeric antigen receptors, new inhibitors for treating chronic myeloid leukemia, and the novel type of interleukin-2. Therefore, the purpose of our review is to summarize the recent development of de novo protein design and its application in biomedicine, especially in exploring new treatment methods for hematological disorders.

Protein therapeutics have been widely used to treat hematological disorders. With the advent of de novo protein design, protein therapeutics are not limited to ameliorating natural proteins but also produce novel protein sequences, folds, and functions with shapes and functions customized to bind to the therapeutic targets. De novo protein techniques have been widely used biomedically to design novel diagnostic and therapeutic drugs, novel vaccines, and novel biological materials. In addition, de novo protein design has provided new options for treating hematological disorders. Scientists have designed protein switches called Colocalization-dependent Latching Orthogonal Cage–Key pRoteins (Co-LOCKR) that perform computations on the surface of cells. De novo designed molecules exhibit a better capacity than the currently available tyrosine kinase inhibitors in chronic myeloid leukemia therapy. De novo designed protein neoleukin-2/15 enhances chimeric antigen receptor T-cell activity. This new technique has great biomedical potential, especially in exploring new treatment methods for hematological disorders. This review discusses the development of de novo protein design and its biological applications, with emphasis on the treatment of hematological disorders.

[end paper 94]

[begin paper 95]

PMCID = 8826494

Title = The Next Frontier for Designing Switchable Proteins: Rational Enhancement of Kinetics

Designing proteins that can switch between active (ON) and inactive (OFF) conformations in response to signals such as ligand binding and incident light has been a tantalizing endeavor in protein engineering for over a decade. While such designs have yielded novel biosensors, therapeutic agents, and smart biomaterials, the response times (times for switching ON and OFF) of many switches have been too slow to be of practical use. Among the defining properties of such switches, the kinetics of switching has been the most challenging to optimize. This is largely due to the difficulty of characterizing the structures of transient states, which are required for manipulating the height of the effective free energy barrier between the ON and OFF states. We share our perspective of the most promising new experimental and computational strategies over the past several years for tackling this next frontier for designing switchable proteins.

[end paper 95]

[begin paper 96]

PMCID = 8906867

Title = The endoplasmic reticulum proteostasis network profoundly shapes the protein sequence space accessible to HIV envelope

The sequence space accessible to evolving proteins can be enhanced by cellular chaperones that assist biophysically defective clients in navigating complex folding landscapes. It is also possible, at least in theory, for proteostasis mechanisms that promote strict quality control to greatly constrain accessible protein sequence space. Unfortunately, most efforts to understand how proteostasis mechanisms influence evolution rely on artificial inhibition or genetic knockdown of specific chaperones. The few experiments that perturb quality control pathways also generally modulate the levels of only individual quality control factors. Here, we use chemical genetic strategies to tune proteostasis networks via natural stress response pathways that regulate the levels of entire suites of chaperones and quality control mechanisms. Specifically, we upregulate the unfolded protein response (UPR) to test the hypothesis that the host endoplasmic reticulum (ER) proteostasis network shapes the sequence space accessible to human immunodeficiency virus-1 (HIV-1) envelope (Env) protein. Elucidating factors that enhance or constrain Env sequence space is critical because Env evolves extremely rapidly, yielding HIV strains with antibody- and drug-escape mutations. We find that UPR-mediated upregulation of ER proteostasis factors, particularly those controlled by the IRE1-XBP1s UPR arm, globally reduces Env mutational tolerance. Conserved, functionally important Env regions exhibit the largest decreases in mutational tolerance upon XBP1s induction. Our data indicate that this phenomenon likely reflects strict quality control endowed by XBP1s-mediated remodeling of the ER proteostasis environment. Intriguingly, and in contrast, specific regions of Env, including regions targeted by broadly neutralizing antibodies, display enhanced mutational tolerance when XBP1s is induced, hinting at a role for host proteostasis network hijacking in potentiating antibody escape. These observations reveal a key function for proteostasis networks in decreasing instead of expanding the sequence space accessible to client proteins, while also demonstrating that the host ER proteostasis network profoundly shapes the mutational tolerance of Env in ways that could have important consequences for HIV adaptation.

The host cell’s endoplasmic reticulum proteostasis network has a profound, constraining impact on the protein sequence space accessible to HIV’s envelope protein, which is a major target of the host’s adaptive immune system; in particular, upregulation of stringent quality control pathways appears to restrict the viability of destabilizing envelope variants.

[end paper 96]

[begin paper 97]

PMCID = 7808408

Title = Comparative host-coronavirus protein interaction networks reveal pan-viral disease mechanisms

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is closely related to the deadly coronaviruses SARS-CoV-1 and Middle East respiratory syndrome coronavirus (MERS-CoV). Considerable efforts are focused on developing treatments, and therapies that work across coronaviruses would be particularly valuable. Shedding light on the host factors hijacked by the viruses, Gordon et al. mapped the interactions between viral and human proteins for SARS-CoV-2, SARS-CoV-1, and MERS-CoV; analyzed the localization of viral proteins in human cells; and used genetic screening to identify host factors that either enhance or inhibit viral infection. For a subset of the interactions essential for the virus life cycle, the authors determined the cryo–electron microscopy structures and mined patient data to understand how targeting host factors may be relevant to clinical outcomes.

Science, this issue p. eabe9403

Comparison of host interactions of three lethal coronaviruses identifies commonly hijacked pathways and potential drug targets.

The emergence of three lethal coronaviruses in <20 years and the urgency of the COVID-19 pandemic have prompted efforts to develop new therapeutic strategies, including by repurposing existing agents. After performing a comparative analysis of the three pathogenic human coronaviruses severe acute respiratory syndrome coronavirus 1 (SARS-CoV-1), SARS-CoV-2, and Middle East respiratory syndrome coronavirus (MERS-CoV), we identified shared biology and host-directed drug targets to prioritize therapeutics with potential for rapid deployment against current and future coronavirus outbreaks.

Expanding on our recent SARS-CoV-2 interactome, we mapped the virus-host protein-protein interactions for SARS-CoV-1 and MERS-CoV and assessed the cellular localization of each viral protein across the three strains. We conducted two genetic screens of SARS-CoV-2 interactors to prioritize functionally-relevant host factors and structurally characterized one virus-host interaction. We then tested the clinical relevance of three more host factors by assessing risk in genetic cohorts or observing effectiveness of host factor–targeting drugs in real-world evidence.

Quantitative comparison of the 389 interactors of SARS-CoV-2, 366 of SARS-CoV-1, and 296 of MERS-CoV highlighted interactions with host processes that are conserved across all three viruses, including where nonorthologous proteins from different virus strains seem to fill similar roles. We also localized each individually-expressed viral protein by microscopy and then raised and validated antisera against 14 SARS-CoV-2 proteins to determine their localization during infection.

On the basis of two independent genetic perturbation screens, we identified 73 host factors that, when depleted, caused significant changes in SARS-CoV-2 replication. From this list of potential drug targets, we validated the biological and clinical relevance of Tom70, IL17RA, PGES-2, and SigmaR1.

A 3-Å cryo–electron microscopy structure of Tom70, a mitochondrial import receptor, in complex with SARS-CoV-2 ORF9b, provides insight into how ORF9b may modulate the host immune response. Using curated genome-wide association study data, we found that individuals with genotypes corresponding to higher soluble IL17RA levels in plasma are at decreased risk of COVID-19 hospitalization.

To demonstrate the value of our data for drug repurposing, we identified SARS-CoV-2 patients who were prescribed drugs against prioritized targets and asked how they fared compared with carefully matched patients treated with clinically similar drugs that do not inhibit SARS-CoV-2. Both indomethacin, an inhibitor of host factor PGES-2, and typical antipsychotics, selected for their interaction with sigma receptors, showed effectiveness against COVID-19 compared with celecoxib and atypical antipsychotics, respectively.

By employing an integrative and collaborative approach, we identified conserved mechanisms across three pathogenic coronavirus strains and further investigated potential drug targets. This versatile approach is broadly applicable to other infectious agents and disease areas.

(Left) Viral-human protein-protein interaction network mapping, viral protein localization studies, and functional genetic screens provide key insights into the shared and individual characteristics of each virus. (Right) Structural studies and hypothesis testing in clinical datasets demonstrate the utility of this approach for prioritizing therapeutic strategies. Nsp, nonstructural protein; ORF, open reading frame; ER, endoplasmic reticulum.

The COVID-19 pandemic, caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), is a grave threat to public health and the global economy. SARS-CoV-2 is closely related to the more lethal but less transmissible coronaviruses SARS-CoV-1 and Middle East respiratory syndrome coronavirus (MERS-CoV). Here, we have carried out comparative viral-human protein-protein interaction and viral protein localization analyses for all three viruses. Subsequent functional genetic screening identified host factors that functionally impinge on coronavirus proliferation, including Tom70, a mitochondrial chaperone protein that interacts with both SARS-CoV-1 and SARS-CoV-2 ORF9b, an interaction we structurally characterized using cryo–electron microscopy. Combining genetically validated host factors with both COVID-19 patient genetic data and medical billing records identified molecular mechanisms and potential drug treatments that merit further molecular and clinical study.

[end paper 97]

[begin paper 98]

PMCID = 7402048

Title = The SARS-CoV-2 Nucleocapsid phosphoprotein forms mutually exclusive condensates with RNA and the membrane-associated M protein

The multifunctional nucleocapsid (N) protein in SARS-CoV-2 binds the ~30 kb viral RNA genome to aid its packaging into the 80–90nm membrane-enveloped virion. The N protein is composed of N-terminal RNA-binding and C-terminal dimerization domains that are flanked by three intrinsically disordered regions. Here we demonstrate that a centrally located 40 amino acid intrinsically disordered domain drives phase separation of N protein when bound to RNA, with the morphology of the resulting condensates affected by inclusion in the RNA of the putative SARS-CoV-2 packaging signal. The SARS-CoV-2 M protein, normally embedded in the virion membrane with its C-terminus extending into the virion core, independently induces N protein phase separation that is dependent on the N protein’s C-terminal dimerization domain and disordered region. Three-component mixtures of N+M+RNA form condensates with mutually exclusive compartments containing N+M or N+RNA, including spherical annular structures in which the M protein coats the outside of an N+RNA condensate. These findings support a model in which phase separation of the N protein with both the viral genomic RNA and the SARS-CoV-2 M protein facilitates RNA packaging and virion assembly.

[end paper 98]

[begin paper 99]

PMCID = 6205781

Title = 27th Annual Computational Neuroscience Meeting (CNS*2018): Part One



[end paper 99]

[begin paper 100]

PMCID = 9976687

Title = Automatic and accurate ligand structure determination guided by cryo-electron microscopy maps

Advances in cryo-electron microscopy (cryoEM) and deep-learning guided protein structure prediction have expedited structural studies of protein complexes. However, methods for accurately determining ligand conformations are lacking. In this manuscript, we develop EMERALD, a tool for automatically determining ligand structures guided by medium-resolution cryoEM density. We show this method is robust at predicting ligands along with surrounding side chains in maps as low as 4.5 Å local resolution. Combining this with a measure of placement confidence and running on all protein/ligand structures in the EMDB, we show that 57% of ligands replicate the deposited model, 16% confidently find alternate conformations, 22% have ambiguous density where multiple conformations might be present, and 5% are incorrectly placed. For five cases where our approach finds an alternate conformation with high confidence, high-resolution crystal structures validate our placement. EMERALD and the resulting analysis should prove critical in using cryoEM to solve protein-ligand complexes.

As cryo-EM becomes commonplace in drug discovery, tools for automating small molecule structure determination are needed. Here, authors show a map-guided ligand modeling approach to building ligand structures at resolutions common in cryo-EM.

[end paper 100]



