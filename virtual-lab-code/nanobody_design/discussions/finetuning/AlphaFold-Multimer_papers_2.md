## User

This is the beginning of an individual meeting with Computational Biologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "AlphaFold-Multimer" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on AlphaFold-Multimer to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on AlphaFold-Multimer using the query "AlphaFold-Multimer SARS-CoV-2 spike protein" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to AlphaFold-Multimer. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Computational Biologist, please provide your response to the agenda.

## Computational Biologist

Based on the abstracts provided, here are the papers that are directly relevant to AlphaFold-Multimer and its application to SARS-CoV-2 spike protein, which I would like to fine-tune myself on:

```python
{
    "9406418": "SARS-CoV-2 Spike Does Not Possess Intrinsic Superantigen-like Inflammatory Activity",
    "10136805": "Using AlphaFold Predictions in Viral Research",
    "10312674": "AI-guided pipeline for protein-protein interaction drug discovery identifies a SARS-CoV-2 inhibitor",
    "11157076": "Human cytokine and coronavirus nucleocapsid protein interactivity using large-scale virtual screens",
    "11319189": "AlphaFold2 in biomedical research: facilitating the development of diagnostic strategies for disease",
    "11468914": "Structures of the Foamy virus fusion protein reveal an unexpected link with the F protein of paramyxo- and pneumoviruses",
    "11543627": "Prediction of conformational states in a coronavirus channel using Alphafold-2 and DeepMSA2: Strengths and limitations",
    "11788748": "Structure‐based computational design of antibody mimetics: challenges and perspectives"
}
```

These papers are selected based on their relevance to AlphaFold-Multimer, its applications, and its implications for SARS-CoV-2 research.

## User

Tool Output:

Here are the top 96 articles on PubMed Central for the query "AlphaFold-Multimer SARS-CoV-2 spike protein":

[begin paper 1]

PMCID = 10987651

Title = AI-guided pipeline for protein–protein interaction drug discovery identifies a SARS-CoV-2 inhibitor

Protein–protein interactions (PPIs) offer great opportunities to expand the druggable proteome and therapeutically tackle various diseases, but remain challenging targets for drug discovery. Here, we provide a comprehensive pipeline that combines experimental and computational tools to identify and validate PPI targets and perform early-stage drug discovery. We have developed a machine learning approach that prioritizes interactions by analyzing quantitative data from binary PPI assays or AlphaFold-Multimer predictions. Using the quantitative assay LuTHy together with our machine learning algorithm, we identified high-confidence interactions among SARS-CoV-2 proteins for which we predicted three-dimensional structures using AlphaFold-Multimer. We employed VirtualFlow to target the contact interface of the NSP10-NSP16 SARS-CoV-2 methyltransferase complex by ultra-large virtual drug screening. Thereby, we identified a compound that binds to NSP10 and inhibits its interaction with NSP16, while also disrupting the methyltransferase activity of the complex, and SARS-CoV-2 replication. Overall, this pipeline will help to prioritize PPI targets to accelerate the discovery of early-stage drug candidates targeting protein complexes and pathways.

A new pipeline for prioritizing protein-protein interactions (PPIs) for drug discovery, combines machine learning-based scoring of quantitative PPI data, protein complex structure prediction and virtual drug screening.

A multi-adaptive support vector machine (maSVM) classifier is used for scoring PPIs from quantitative interaction and structure prediction data.

The machine learning-based classifier is applicable to PPI datasets from various assays and AlphaFold-Multimer predictions improving comparability between different methods.

Interaction mapping with LuTHy and maSVM-based scoring identified high-confidence SARS-CoV-2 PPIs. Subsequent AlphaFold-Multimer predictions revealed key interaction residues within the NSP10-NSP16 methyltransferase complex.

Targeting the complex with virtual compound screening identified an early-stage small molecule inhibitor that disrupts the NSP10-NSP16 interaction and SARS-CoV-2 replication.

A new pipeline for prioritizing protein-protein interactions (PPIs) for drug discovery, combines machine learning-based scoring of quantitative PPI data, protein complex structure prediction and virtual drug screening.

[end paper 1]

[begin paper 2]

PMCID = 10312674

Title = AI-guided pipeline for protein-protein interaction drug discovery identifies a SARS-CoV-2 inhibitor

Protein-protein interactions (PPIs) offer great opportunities to expand the druggable proteome and therapeutically tackle various diseases, but remain challenging targets for drug discovery. Here, we provide a comprehensive pipeline that combines experimental and computational tools to identify and validate PPI targets and perform early-stage drug discovery. We have developed a machine learning approach that prioritizes interactions by analyzing quantitative data from binary PPI assays and AlphaFold-Multimer predictions. Using the quantitative assay LuTHy together with our machine learning algorithm, we identified high-confidence interactions among SARS-CoV-2 proteins for which we predicted three-dimensional structures using AlphaFold Multimer. We employed VirtualFlow to target the contact interface of the NSP10-NSP16 SARS-CoV-2 methyltransferase complex by ultra-large virtual drug screening. Thereby, we identified a compound that binds to NSP10 and inhibits its interaction with NSP16, while also disrupting the methyltransferase activity of the complex, and SARS-CoV-2 replication. Overall, this pipeline will help to prioritize PPI targets to accelerate the discovery of early-stage drug candidates targeting protein complexes and pathways.

[end paper 2]

[begin paper 3]

PMCID = 10588992

Title = Antibodies targeting a quaternary site on SARS-CoV-2 spike glycoprotein prevent viral receptor engagement by conformational locking

SARS-CoV-2 continues to evolve, with many variants evading clinically authorized antibodies. To isolate monoclonal antibodies (mAbs) with broadly neutralizing capacities against the virus, we screened serum samples from convalescing COVID-19 patients. We isolated two mAbs, 12-16 and 12-19, which neutralized all SARS-CoV-2 variants tested, including the XBB subvariants, and prevented infection in hamsters challenged with Omicron BA.1 intranasally. Structurally, both antibodies targeted a conserved quaternary epitope located at the interface between the N-terminal domain and subdomain 1, uncovering a site of vulnerability on SARS-CoV-2 spike. These antibodies prevented viral receptor engagement by locking the receptor-binding domain (RBD) of spike in the down conformation, revealing a mechanism of virus neutralization for non-RBD antibodies. Deep mutational scanning showed that SARS-CoV-2 could mutate to escape 12-19, but such mutations are rarely found in circulating viruses. Antibodies 12-16 and 12-19 hold promise as prophylactic agents for immunocompromised persons who do not respond robustly to COVID-19 vaccines.

Isolated bnAbs 12-16 and 12-19 from a SARS-CoV-2 recovered/vaccinated individual

These mAbs target a conserved quaternary epitope at the interface between NTD-SD1

The mAbs neutralize all current SARS-CoV-2 VOCs by locking RBD in down conformation

12-19 escape mutations are rarely found in circulating SARS-CoV-2 viruses

Current variants of SARS-CoV-2 can evade clinically authorized antibodies. Liu et al. demonstrate that two monoclonal antibodies isolated from convalescing COVID-19 patients neutralize all current SARS-CoV-2 variants of concern via interaction with a mechanism that locks the RBD in the down conformation. Mutations in the epitope targeted by these mAbs are rarely found in circulating SARS-CoV-2 viruses, suggesting clinical applicability.

[end paper 3]

[begin paper 4]

PMCID = 10470497

Title = SARS-CoV-2 main protease cleaves MAGED2 to antagonize host antiviral defense

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is the agent causing the global pandemic of COVID-19. SARS-CoV-2 genome encodes a main protease (nsp5, also called Mpro) and a papain-like protease (nsp3, also called PLpro), which are responsible for processing viral polyproteins to assemble a functional replicase complex. In this study, we found that Mpro of SARS-CoV-2 can cleave human MAGED2 and other mammalian orthologs at Gln-263. Moreover, SARS-CoV and MERS-CoV Mpro can also cleave human MAGED2, suggesting MAGED2 cleavage by Mpro is an evolutionarily conserved mechanism of coronavirus infection in mammals. Intriguingly, Mpro from Beta variant cleaves MAGED2 more efficiently than wild type, but Omicron Mpro is opposite. Further studies show that MAGED2 inhibits SARS-CoV-2 infection at viral replication step. Mechanistically, MAGED2 is associated with SARS-CoV-2 nucleocapsid protein through its N-terminal region in an RNA-dependent manner, and this disrupts the interaction between SARS-CoV-2 nucleocapsid protein and viral genome, thus inhibiting viral replication. When MAGED2 is cleaved by Mpro, the N-terminal of MAGED2 will translocate into the nucleus, and the truncated MAGED2 is unable to suppress SARS-CoV-2 replication. This work not only discovers the antiviral function of MAGED2 but also provides new insights into how SARS-CoV-2 Mpro antagonizes host antiviral response.

Host factors that restrict severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection remain elusive. Here, we found that MAGED2 can be cleaved by SARS-CoV-2 main protease (Mpro) at Gln-263. SARS-CoV and MERS-CoV Mpro can also cleave MAGED2, and MAGED2 from multiple species can be cleaved by SARS-CoV-2 Mpro. Mpro from Beta variant cleaves MAGED2 more efficiently efficiently than wild type, but Omicron is the opposite. MAGED2 depletion enhances SARS-CoV-2 infection, suggesting its inhibitory role in SARS-CoV-2 infection. Mechanistically, MAGED2 restricts SARS-CoV-2 replication by disrupting the interaction between nucleocapsid and viral genomes. When MAGED2 is cleaved, its N-terminal will translocate into the nucleus. In this way, Mpro relieves MAGED2' inhibition on viral replication. This study improves our understanding of complex viral-host interaction and provides novel targets to treat SARS-CoV-2 infection.

[end paper 4]

[begin paper 5]

PMCID = 9406418

Title = SARS-CoV-2 Spike Does Not Possess Intrinsic Superantigen-like Inflammatory Activity

Multisystem inflammatory syndrome in children (MIS-C) is a rare hyperinflammatory disease occurring several weeks after SARS-CoV-2 infection. The clinical similarities between MIS-C and the toxic shock syndrome, together with the preferential expansion of T cells with a T-cell receptor variable β chain (TCRVβ) skewing, suggested a superantigen theory of MIS-C. For instance, recent in silico modelling evidenced the presence of a highly conserved motif within SARS-CoV-2 spike protein similar in structure to the superantigenic fragment of staphylococcal enterotoxin B (SEB). However, experimental data on the superantigenic activity of the SARS-CoV-2 spike have not yet been provided. Here, we assessed the superantigenic activity of the SARS-CoV-2 spike by analysing inflammatory cytokine production in both Jurkat cells and the peripheral blood CD4+ T cells stimulated with the SARS-CoV-2 spike or SEB as a control. We found that, unlike SEB, the SARS-CoV-2 spike does not exhibit an intrinsic superantigen-like activity.

[end paper 5]

[begin paper 6]

PMCID = 9906597

Title = Targeted photodynamic neutralization of SARS-CoV-2 mediated by singlet oxygen

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) virus has been on a rampage for more than two years. Vaccines in combination with neutralizing antibodies (NAbs) against SARS-CoV-2 carry great hope in the treatment and final elimination of coronavirus disease 2019 (COVID-19). However, the relentless emergence of variants of concern (VOC), including the most recent Omicron variants, presses for novel measures to counter these variants that often show immune evasion. Hereby we developed a targeted photodynamic approach to neutralize SARS-CoV-2 by engineering a genetically encoded photosensitizer (SOPP3) to a diverse list of antibodies targeting the wild-type (WT) spike protein, including human antibodies isolated from a 2003 Severe acute respiratory syndrome (SARS) patient, potent monomeric and multimeric nanobodies targeting receptor-binding domain (RBD), and non-neutralizing antibodies (non-NAbs) targeting the more conserved N-terminal domain (NTD). As confirmed by pseudovirus neutralization assay, this targeted photodynamic approach significantly increased the efficacy of these antibodies, especially that of non-NAbs, against not only the WT but also the Delta strain and the heavily immune escape Omicron strain (BA.1). Subsequent measurement of infrared phosphorescence at 1270 nm confirmed the generation of singlet oxygen (1O2) in the photodynamic process. Mass spectroscopy assay uncovered amino acids in the spike protein targeted by 1O2. Impressively, Y145 and H146 form an oxidization “hotspot”, which overlaps with the antigenic “supersite” in NTD. Taken together, our study established a targeted photodynamic approach against the SARS-CoV-2 virus and provided mechanistic insights into the photodynamic modification of protein molecules mediated by 1O2.

The online version contains supplementary material available at 10.1007/s43630-023-00381-w.

[end paper 6]

[begin paper 7]

PMCID = 11190248

Title = Predicting human and viral protein variants affecting COVID-19 susceptibility and repurposing therapeutics

The COVID-19 disease is an ongoing global health concern. Although vaccination provides some protection, people are still susceptible to re-infection. Ostensibly, certain populations or clinical groups may be more vulnerable. Factors causing these differences are unclear and whilst socioeconomic and cultural differences are likely to be important, human genetic factors could influence susceptibility. Experimental studies indicate SARS-CoV-2 uses innate immune suppression as a strategy to speed-up entry and replication into the host cell. Therefore, it is necessary to understand the impact of variants in immunity-associated human proteins on susceptibility to COVID-19. In this work, we analysed missense coding variants in several SARS-CoV-2 proteins and their human protein interactors that could enhance binding affinity to SARS-CoV-2. We curated a dataset of 19 SARS-CoV-2: human protein 3D-complexes, from the experimentally determined structures in the Protein Data Bank and models built using AlphaFold2-multimer, and analysed the impact of missense variants occurring in the protein–protein interface region. We analysed 468 missense variants from human proteins and 212 variants from SARS-CoV-2 proteins and computationally predicted their impacts on binding affinities for the human viral protein complexes. We predicted a total of 26 affinity-enhancing variants from 13 human proteins implicated in increased binding affinity to SARS-CoV-2. These include key-immunity associated genes (TOMM70, ISG15, IFIH1, IFIT2, RPS3, PALS1, NUP98, AXL, ARF6, TRIMM, TRIM25) as well as important spike receptors (KREMEN1, AXL and ACE2). We report both common (e.g., Y13N in IFIH1) and rare variants in these proteins and discuss their likely structural and functional impact, using information on known and predicted functional sites. Potential mechanisms associated with immune suppression implicated by these variants are discussed. Occurrence of certain predicted affinity-enhancing variants should be monitored as they could lead to increased susceptibility and reduced immune response to SARS-CoV-2 infection in individuals/populations carrying them. Our analyses aid in understanding the potential impact of genetic variation in immunity-associated proteins on COVID-19 susceptibility and help guide drug-repurposing strategies.

[end paper 7]

[begin paper 8]

PMCID = 10507161

Title = N4-hydroxycytidine, the active compound of Molnupiravir, promotes SARS-CoV-2 mutagenesis and escape from a neutralizing nanobody

N4-hydroxycytidine (NHC), the active compound of the drug Molnupiravir, is incorporated into SARS-CoV-2 RNA, causing false base pairing. The desired result is an “error catastrophe,” but this bears the risk of mutated virus progeny. To address this experimentally, we propagated the initial SARS-CoV-2 strain in the presence of NHC. Deep sequencing revealed numerous NHC-induced mutations and host-cell-adapted virus variants. The presence of the neutralizing nanobody Re5D06 selected for immune escape mutations, in particular p.E484K and p.F490S, which are key mutations of the Beta/Gamma and Omicron-XBB strains, respectively. With NHC treatment, nanobody resistance occurred two passages earlier than without. Thus, within the limitations of this purely in vitro study, we conclude that the combined action of Molnupiravir and a spike-neutralizing antagonist leads to the rapid emergence of escape mutants. We propose caution use and supervision when using Molnupiravir, especially when patients are still at risk of spreading virus.

Molnupiravir might give rise to new SARS-CoV-2 variants when used without precautions

In vitro, the active compound of Molnupiravir gives rise to nanobody-resistant variants

We provide an integrated view on the selection of nanobody-escape mutants

Pharmaceutical science; Immunology; Virology; Structural biology

[end paper 8]

[begin paper 9]

PMCID = 10169150

Title = SARS-CoV-2 N protein enhances the anti-apoptotic activity of MCL-1 to promote viral replication

Viral infection in respiratory tract usually leads to cell death, impairing respiratory function to cause severe disease. However, the diversity of clinical manifestations of SARS-CoV-2 infection increases the complexity and difficulty of viral infection prevention, and especially the high-frequency asymptomatic infection increases the risk of virus transmission. Studying how SARS-CoV-2 affects apoptotic pathway may help to understand the pathological process of its infection. Here, we uncovered SARS-CoV-2 imployed a distinct anti-apoptotic mechanism via its N protein. We found SARS-CoV-2 virus-like particles (trVLP) suppressed cell apoptosis, but the trVLP lacking N protein didn’t. Further study verified that N protein repressed cell apoptosis in cultured cells, human lung organoids and mice. Mechanistically, N protein specifically interacted with anti-apoptotic protein MCL-1, and recruited a deubiquitinating enzyme USP15 to remove the K63-linked ubiquitination of MCL-1, which stabilized this protein and promoted it to hijack Bak in mitochondria. Importantly, N protein promoted the replications of IAV, DENV and ZIKV, and exacerbated death of IAV-infected mice, all of which could be blocked by a MCL-1 specific inhibitor, S63845. Altogether, we identifed a distinct anti-apoptotic function of the N protein, through which it promoted viral replication. These may explain how SARS-CoV-2 effectively replicates in asymptomatic individuals without cuasing respiratory dysfunction, and indicate a risk of enhanced coinfection with other viruses. We anticipate that abrogating the N/MCL-1-dominated apoptosis repression is conducive to the treatments of SARS-CoV-2 infection as well as coinfections with other viruses.

[end paper 9]

[begin paper 10]

PMCID = 9456519

Title = Effect of Delta and Omicron Mutations on the RBD-SD1 Domain of the Spike Protein in SARS-CoV-2 and the Omicron Mutations on RBD-ACE2 Interface Complex

The receptor-binding domain (RBD) is the essential part in the Spike-protein (S-protein) of SARS-CoV-2 virus that directly binds to the human ACE2 receptor, making it a key target for many vaccines and therapies. Therefore, any mutations at this domain could affect the efficacy of these treatments as well as the viral-cell entry mechanism. We introduce ab initio DFT-based computational study that mainly focuses on two parts: (1) Mutations effects of both Delta and Omicron variants in the RBD-SD1 domain. (2) Impact of Omicron RBD mutations on the structure and properties of the RBD-ACE2 interface system. The in-depth analysis is based on the novel concept of amino acid-amino acid bond pair units (AABPU) that reveal the differences between the Delta and/or Omicron mutations and its corresponding wild-type strain in terms of the role played by non-local amino acid interactions, their 3D shapes and sizes, as well as contribution to hydrogen bonding and partial charge distributions. Our results also show that the interaction of Omicron RBD with ACE2 significantly increased its bonding between amino acids at the interface providing information on the implications of penetration of S-protein into ACE2, and thus offering a possible explanation for its high infectivity. Our findings enable us to present, in more conspicuous atomic level detail, the effect of specific mutations that may help in predicting and/or mitigating the next variant of concern. 

[end paper 10]

[begin paper 11]

PMCID = 9448712

Title = Deep learning-based molecular dynamics simulation for structure-based drug design against SARS-CoV-2

Coronavirus disease 2019 (COVID-19), caused by severe acute respiratory syndrome coronavirus type 2 (SARS-CoV-2), has led to a global pandemic. Deep learning (DL) technology and molecular dynamics (MD) simulation are two mainstream computational approaches to investigate the geometric, chemical and structural features of protein and guide the relevant drug design. Despite a large amount of research papers focusing on drug design for SARS-COV-2 using DL architectures, it remains unclear how the binding energy of the protein-protein/ligand complex dynamically evolves which is also vital for drug development. In addition, traditional deep neural networks usually have obvious deficiencies in predicting the interaction sites as protein conformation changes. In this review, we introduce the latest progresses of the DL and DL-based MD simulation approaches in structure-based drug design (SBDD) for SARS-CoV-2 which could address the problems of protein structure and binding prediction, drug virtual screening, molecular docking and complex evolution. Furthermore, the current challenges and future directions of DL-based MD simulation for SBDD are also discussed.

[end paper 11]

[begin paper 12]

PMCID = 10580870

Title = SARS-CoV-2 omicron spike simulations: broad antibody escape, weakened ACE2 binding, and modest furin cleavage

The recent emergence of the omicron variant of the SARS-CoV-2 virus with large numbers of mutations has raised concern about a potential new surge in infections. Here we use molecular dynamics to study the biophysics of the interface of the BA1 and BA2 omicron spike protein binding to (i) the ACE2 receptor protein, (ii) antibodies from all known binding regions, and (iii) the furin binding domain. Our simulations suggest that while there is a significant reduction of antibody (Ab) binding strength corresponding to escape, the omicron spikes pay a cost in terms of weaker receptor binding as measured by interfacial hydrogen bonds (H-bond). The furin cleavage domain (FCD) is the same or weaker binding than the delta variant, suggesting lower fusogenicity resulting in less viral load and disease intensity than the delta variant.

The BA1 and BA2 and closely related BA2.12.2 and BA.5 omicron variants of SARS-CoV-2 dominate the current global infection landscape. Given the high number of mutations, particularly those which will lead to antibody escape, it is important to establish accurate methods that can guide developing health policy responses that identify at a fundamental level whether omicron and its variants are more threatening than its predecessors, especially delta. The importance of our work is to demonstrate that simple in silico simulations can predict biochemical binding details of the omicron spike protein that have epidemiological consequences, especially for binding to the cells and for fusing the viral membrane with the cells. In each case, we predicted weaker binding of the omicron spike, which agreed with subsequent experimental results. Future virology experiments will be needed to test these predictions further.

[end paper 12]

[begin paper 13]

PMCID = 10131520

Title = De novo design of protein interactions with learned surface fingerprints

Physical interactions between proteins are essential for most biological processes governing life. However, the molecular determinants of such interactions have been challenging to understand, even as genomic, proteomic and structural data increase. This knowledge gap has been a major obstacle for the comprehensive understanding of cellular protein–protein interaction networks and for the de novo design of protein binders that are crucial for synthetic biology and translational applications. Here we use a geometric deep-learning framework operating on protein surfaces that generates fingerprints to describe geometric and chemical features that are critical to drive protein–protein interactions. We hypothesized that these fingerprints capture the key aspects of molecular recognition that represent a new paradigm in the computational design of novel protein interactions. As a proof of principle, we computationally designed several de novo protein binders to engage four protein targets: SARS-CoV-2 spike, PD-1, PD-L1 and CTLA-4. Several designs were experimentally optimized, whereas others were generated purely in silico, reaching nanomolar affinity with structural and mutational characterization showing highly accurate predictions. Overall, our surface-centric approach captures the physical and chemical determinants of molecular recognition, enabling an approach for the de novo design of protein interactions and, more broadly, of artificial proteins with function.

A surface-centric approach captures the physical and chemical determinants of molecular recognition, enabling the de novo design of protein interactions and of artificial proteins with function.

[end paper 13]

[begin paper 14]

PMCID = 9915606

Title = Structural prediction of chimeric immunogens to elicit targeted antibodies against betacoronaviruses

Betacoronaviruses pose an ongoing pandemic threat. Antigenic evolution of the SARS-CoV-2 virus has shown that much of the spontaneous antibody response is narrowly focused rather than broadly neutralizing against even SARS-CoV-2 variants, let alone future threats. One way to overcome this is by focusing the antibody response against better-conserved regions of the viral spike protein. Here, we present a design approach to predict stable chimeras between SARS-CoV-2 and other coronaviruses, creating synthetic spike proteins that display a desired conserved region and vary other regions. We leverage AlphaFold to predict chimeric structures and create a new metric for scoring chimera stability based on AlphaFold outputs. We evaluated 114 candidate spike chimeras using this approach. Top chimeras were further evaluated using molecular dynamics simulation as an intermediate validation technique, showing good stability compared to low-scoring controls. Experimental testing of five predicted-stable and two predicted-unstable chimeras confirmed 5/7 predictions, with one intermediate result. This demonstrates the feasibility of the underlying approach, which can be used to design custom immunogens to focus the immune response against a desired viral glycoprotein epitope.

[end paper 14]

[begin paper 15]

PMCID = 11433606

Title = HLA-C Peptide Repertoires as Predictors of Clinical Response during Early SARS-CoV-2 Infection

The human leukocyte antigen (HLA) system plays a pivotal role in the immune response to viral infections, mediating the presentation of viral peptides to T cells and influencing both the strength and specificity of the host immune response. Variations in HLA genotypes across individuals lead to differences in susceptibility to viral infection and severity of illness. This study uses observations from the early phase of the COVID-19 pandemic to explore how specific HLA class I molecules affect clinical responses to SARS-CoV-2 infection. By analyzing paired high-resolution HLA types and viral genomic sequences from 60 patients, we assess the relationship between predicted HLA class I peptide binding repertoires and infection severity as measured by the sequential organ failure assessment score. This approach leverages functional convergence across HLA-C alleles to identify relationships that may otherwise be inaccessible due to allelic diversity and limitations in sample size. Surprisingly, our findings show that severely symptomatic infection in this cohort is associated with disproportionately abundant binding of SARS-CoV-2 structural and non-structural protein epitopes by patient HLA-C molecules. In addition, the extent of overlap between a given patient’s predicted HLA-C and HLA-A peptide binding repertoires correlates with worse prognoses in this cohort. The findings highlight immunologic mechanisms linking HLA-C molecules with the human response to viral pathogens that warrant further investigation.

[end paper 15]

[begin paper 16]

PMCID = 10075959

Title = A conserved oligomerization domain in the disordered linker of coronavirus nucleocapsid proteins

The nucleocapsid (N-)protein of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has a key role in viral assembly and scaffolding of the viral RNA. It promotes liquid-liquid phase separation (LLPS), forming dense droplets that support the assembly of ribonucleoprotein particles with as-of-yet unknown macromolecular architecture. Combining biophysical experiments, molecular dynamics simulations, and analysis of the mutational landscape, we describe a heretofore unknown oligomerization site that contributes to LLPS, is required for the assembly of higher-order protein-nucleic acid complexes, and is coupled to large-scale conformational changes of N-protein upon nucleic acid binding. The self-association interface is located in a leucine-rich sequence of the intrinsically disordered linker between N-protein folded domains and formed by transient helices assembling into trimeric coiled-coils. Critical residues stabilizing hydrophobic and electrostatic interactions between adjacent helices are highly protected against mutations in viable SARS-CoV-2 genomes, and the oligomerization motif is conserved across related coronaviruses, thus presenting a target for antiviral therapeutics.

The oligomerization site is key for ribonucleoprotein assembly and presents a new antiviral target for SARS-CoV-2.

[end paper 16]

[begin paper 17]

PMCID = 11157076

Title = Human cytokine and coronavirus nucleocapsid protein interactivity using large-scale virtual screens

Understanding the interactions between SARS-CoV-2 and the human immune system is paramount to the characterization of novel variants as the virus co-evolves with the human host. In this study, we employed state-of-the-art molecular docking tools to conduct large-scale virtual screens, predicting the binding affinities between 64 human cytokines against 17 nucleocapsid proteins from six betacoronaviruses. Our comprehensive in silico analyses reveal specific changes in cytokine-nucleocapsid protein interactions, shedding light on potential modulators of the host immune response during infection. These findings offer valuable insights into the molecular mechanisms underlying viral pathogenesis and may guide the future development of targeted interventions. This manuscript serves as insight into the comparison of deep learning based AlphaFold2-Multimer and the semi-physicochemical based HADDOCK for protein-protein docking. We show the two methods are complementary in their predictive capabilities. We also introduce a novel algorithm for rapidly assessing the binding interface of protein-protein docks using graph edit distance: graph-based interface residue assessment function (GIRAF). The high-performance computational framework presented here will not only aid in accelerating the discovery of effective interventions against emerging viral threats, but extend to other applications of high throughput protein-protein screens.

[end paper 17]

[begin paper 18]

PMCID = 10886287

Title = Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions

This perspective sheds light on the transformative impact of recent computational advancements in the field of protein therapeutics, with a particular focus on the design and development of antibodies. Cutting-edge computational methods have revolutionized our understanding of protein–protein interactions (PPIs), enhancing the efficacy of protein therapeutics in preclinical and clinical settings. Central to these advancements is the application of machine learning and deep learning, which offers unprecedented insights into the intricate mechanisms of PPIs and facilitates precise control over protein functions. Despite these advancements, the complex structural nuances of antibodies pose ongoing challenges in their design and optimization. Our review provides a comprehensive exploration of the latest deep learning approaches, including language models and diffusion techniques, and their role in surmounting these challenges. We also present a critical analysis of these methods, offering insights to drive further progress in this rapidly evolving field. The paper includes practical recommendations for the application of these computational techniques, supplemented with independent benchmark studies. These studies focus on key performance metrics such as accuracy and the ease of program execution, providing a valuable resource for researchers engaged in antibody design and development. Through this detailed perspective, we aim to contribute to the advancement of antibody design, equipping researchers with the tools and knowledge to navigate the complexities of this field.

[end paper 18]

[begin paper 19]

PMCID = 9515332

Title = Reduced B cell antigenicity of Omicron lowers host serologic response

The SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. However, the mechanism remains unclear. Here, we find using a geometric deep-learning model that Omicron’s extensively mutated receptor binding site (RBS) features reduced antigenicity compared with previous variants. Mice immunization experiments with different recombinant receptor binding domain (RBD) variants confirm that the serological response to Omicron is drastically attenuated and less potent. Analyses of serum cross-reactivity and competitive ELISA reveal a reduction in antibody response across both variable and conserved RBD epitopes. Computational modeling confirms that the RBS has a potential for further antigenicity reduction while retaining efficient receptor binding. Finally, we find a similar trend of antigenicity reduction over decades for hCoV229E, a common cold coronavirus. Thus, our study explains the reduced antibody titers associated with Omicron infection and reveals a possible trajectory of future viral evolution.

SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. Tubiana et al. investigate the underlying mechanism using geometric deep learning, mice immunization experiments, and biochemical assays. Mutations reduce antigenicity of the receptor binding site, leading to lower antibody response.

[end paper 19]

[begin paper 20]

PMCID = 10136805

Title = Using AlphaFold Predictions in Viral Research

Elucidation of the tertiary structure of proteins is an important task for biological and medical studies. AlphaFold, a modern deep-learning algorithm, enables the prediction of protein structure to a high level of accuracy. It has been applied in numerous studies in various areas of biology and medicine. Viruses are biological entities infecting eukaryotic and procaryotic organisms. They can pose a danger for humans and economically significant animals and plants, but they can also be useful for biological control, suppressing populations of pests and pathogens. AlphaFold can be used for studies of molecular mechanisms of viral infection to facilitate several activities, including drug design. Computational prediction and analysis of the structure of bacteriophage receptor-binding proteins can contribute to more efficient phage therapy. In addition, AlphaFold predictions can be used for the discovery of enzymes of bacteriophage origin that are able to degrade the cell wall of bacterial pathogens. The use of AlphaFold can assist fundamental viral research, including evolutionary studies. The ongoing development and improvement of AlphaFold can ensure that its contribution to the study of viral proteins will be significant in the future.

[end paper 20]

[begin paper 21]

PMCID = 11743207

Title = Reverse mutational scanning of SARS-CoV-2 spike BA.2.86 identifies epitopes contributing to immune escape from polyclonal sera

The recently detected Omicron BA.2.86 lineage contains more than 30 amino acid mutations relative to BA.2. BA.2.86 and its JN.1 derivative evade neutralization by serum antibodies of fully vaccinated individuals. In this study, we elucidate epitopes driving the immune escape of BA.2.86 and JN.1 via pseudovirus neutralization. Here we generate 33 BA.2.86 mutants, each reverting a single mutation back to BA.2. We use this library in an approach that we call reverse mutational scanning to define distinct neutralization titers against each epitope. Mutations within the receptor binding domain at K356T, V483Δ, and to a lesser extent N460K, A484K, and F486P enhance immune escape. Interestingly, 16insMPLF within the spike N-terminal domain and P621S within S1/S2 also significantly contribute to antibody escape of BA.2.86. Upon XBB.1.5 booster vaccination, neutralization titers against JN.1 and BA.2.86 improve considerably, and residual immune escape is driven by 16insMPLF, N460K, E554K, and to a lesser extent P621S, and A484K.

SARS-CoV-2 Omicron lineage BA.2.86 has over 30 mutations compared to the parental BA.2 lineage. Here Bdeir and colleagues apply reverse mutational scanning to determine which among these mutations present in Omicron BA.2.86 are epitopes linked to immune escape from antibody recognition.

[end paper 21]

[begin paper 22]

PMCID = 11474075

Title = Improved deep learning prediction of antigen–antibody interactions

Accurately predicting antibody–antigen interactions, which are central to the adaptive immune response, is a daunting task. This study explores the potential of a deep learning approach for computationally predicting these interactions. Using the spike protein of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) as a test case, we demonstrate the capability of this computational approach to predict interactions between antibodies and various epitopes on the surface of the antigen. One particularly promising strategy involves using antibody sequences collected from B cell sequencing. The encouraging findings from this approach have significant implications for practical applications to antibody development.

Identifying antibodies that neutralize specific antigens is crucial for developing effective immunotherapies, but this task remains challenging for many target antigens. The rise of deep learning–based computational approaches presents a promising avenue to address this challenge. Here, we assess the performance of a deep learning approach through two benchmark tests aimed at predicting antibodies for the receptor-binding domain of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein. Three different strategies for constructing input sequence alignments are employed for predicting structural models of antigen–antibody complexes. In our initial testing set, which comprises known experimental structures, these strategies collectively yield a significant top-ranked prediction for 61% of cases and a success rate of 47%. Notably, one strategy that utilizes the sequences of known antigen binders outperforms the other two, achieving a precision of 90% in a subsequent test set of ~1,000 antibodies, balanced between true and control antibodies for the antigen, albeit with a lower recall of 25%. Our results underscore the potential of integrating deep learning methods with single B cell sequencing techniques to enhance the prediction accuracy of antigen–antibody interactions.

[end paper 22]

[begin paper 23]

PMCID = 10928099

Title = Impact of mutations on the stability of SARS-CoV-2 nucleocapsid protein structure

The nucleocapsid (N) protein of SARS-CoV-2 is known to participate in various host cellular processes, including interferon inhibition, RNA interference, apoptosis, and regulation of virus life cycles. Additionally, it has potential as a diagnostic antigen and/or immunogen. Our research focuses on examining structural changes caused by mutations in the N protein. We have modeled the complete tertiary structure of native and mutated forms of the N protein using Alphafold2. Notably, the N protein contains 3 disordered regions. The focus was on investigating the impact of mutations on the stability of the protein's dimeric structure based on binding free energy calculations (MM-PB/GB-SA) and RMSD fluctuations after MD simulations. The results demonstrated that 28 mutations out of 37 selected mutations analyzed, compared with wild-type N protein, resulted in a stable dimeric structure, while 9 mutations led to destabilization. Our results are important to understand the tertiary structure of the N protein dimer of SARS-CoV-2 and the effect of mutations on it, their behavior in the host cell, as well as for the research of other viruses belonging to the same genus additionally, to anticipate potential strategies for addressing this viral illness․

[end paper 23]

[begin paper 24]

PMCID = 9653971

Title = Using Alphafold2 to Predict the Structure of the Gp5/M Dimer of Porcine Respiratory and Reproductive Syndrome Virus

Porcine reproductive and respiratory syndrome virus is a positive-stranded RNA virus of the family Arteriviridae. The Gp5/M dimer, the major component of the viral envelope, is required for virus budding and is an antibody target. We used alphafold2, an artificial-intelligence-based system, to predict a credible structure of Gp5/M. The short disulfide-linked ectodomains lie flat on the membrane, with the exception of the erected N-terminal helix of Gp5, which contains the antibody epitopes and a hypervariable region with a changing number of carbohydrates. The core of the dimer consists of six curved and tilted transmembrane helices, and three are from each protein. The third transmembrane regions extend into the cytoplasm as amphiphilic helices containing the acylation sites. The endodomains of Gp5 and M are composed of seven β-strands from each protein, which interact via β-strand seven. The area under the membrane forms an open cavity with a positive surface charge. The M and Orf3a proteins of coronaviruses have a similar structure, suggesting that all four proteins are derived from the same ancestral gene. Orf3a, like Gp5/M, is acylated at membrane-proximal cysteines. The role of Gp5/M during virus replication is discussed, in particular the mechanisms of virus budding and models of antibody-dependent virus neutralization.

[end paper 24]

[begin paper 25]

PMCID = 9848278

Title = Design of a pan-betacoronavirus vaccine candidate through a phylogenetically informed approach

Coronaviruses are a diverse family of viruses that crossed over into humans at least seven times, precipitating mild to catastrophic outcomes. The severe acute respiratory syndrome coronavirus 2 pandemic renewed efforts to identify strains with zoonotic potential and to develop pan-coronavirus vaccines. The analysis of 2181 coronavirus genomes (from 102 host species) confirmed the limited sequence conservation across genera (alpha-, beta-, delta-, and gammacoronavirus) and proteins. A phylogenetically informed pan-coronavirus vaccine was not feasible because of high genetic heterogeneity across genera. We focused on betacoronaviruses and identified nonhuman-infecting receptor binding domain (RBD) sequences that were more genetically similar to human coronaviruses than expected given their phylogenetic divergence. These human-like RBDs defined three phylogenetic clusters. A vaccine candidate based on a representative sequence for each cluster covers the diversity estimated to protect against existing and future human-infecting betacoronaviruses. Our findings emphasize the potential value of conceptualizing prophylaxis against zoonoses in terms of genetic, rather than species, diversity.

High genetic heterogeneity invalidates pan-coronavirus vaccine strategies but pan-betacoronavirus vaccines may be achievable.

[end paper 25]

[begin paper 26]

PMCID = 9859041

Title = Structure-based prediction of T cell receptor:peptide-MHC interactions

The regulatory and effector functions of T cells are initiated by the binding of their cell-surface T cell receptor (TCR) to peptides presented by major histocompatibility complex (MHC) proteins on other cells. The specificity of TCR:peptide-MHC interactions, thus, underlies nearly all adaptive immune responses. Despite intense interest, generalizable predictive models of TCR:peptide-MHC specificity remain out of reach; two key barriers are the diversity of TCR recognition modes and the paucity of training data. Inspired by recent breakthroughs in protein structure prediction achieved by deep neural networks, we evaluated structural modeling as a potential avenue for prediction of TCR epitope specificity. We show that a specialized version of the neural network predictor AlphaFold can generate models of TCR:peptide-MHC interactions that can be used to discriminate correct from incorrect peptide epitopes with substantial accuracy. Although much work remains to be done for these predictions to have widespread practical utility, we are optimistic that deep learning-based structural modeling represents a path to generalizable prediction of TCR:peptide-MHC interaction specificity.

[end paper 26]

[begin paper 27]

PMCID = 10011802

Title = AlphaFold2 and its applications in the fields of biology and medicine

AlphaFold2 (AF2) is an artificial intelligence (AI) system developed by DeepMind that can predict three-dimensional (3D) structures of proteins from amino acid sequences with atomic-level accuracy. Protein structure prediction is one of the most challenging problems in computational biology and chemistry, and has puzzled scientists for 50 years. The advent of AF2 presents an unprecedented progress in protein structure prediction and has attracted much attention. Subsequent release of structures of more than 200 million proteins predicted by AF2 further aroused great enthusiasm in the science community, especially in the fields of biology and medicine. AF2 is thought to have a significant impact on structural biology and research areas that need protein structure information, such as drug discovery, protein design, prediction of protein function, et al. Though the time is not long since AF2 was developed, there are already quite a few application studies of AF2 in the fields of biology and medicine, with many of them having preliminarily proved the potential of AF2. To better understand AF2 and promote its applications, we will in this article summarize the principle and system architecture of AF2 as well as the recipe of its success, and particularly focus on reviewing its applications in the fields of biology and medicine. Limitations of current AF2 prediction will also be discussed.

[end paper 27]

[begin paper 28]

PMCID = 10082545

Title = Natural selection differences detected in key protein domains between non-pathogenic and pathogenic feline coronavirus phenotypes

Feline coronaviruses (FCoVs) commonly cause mild enteric infections in felines worldwide (termed feline enteric coronavirus [FECV]), with around 12 per cent developing into deadly feline infectious peritonitis (FIP; feline infectious peritonitis virus [FIPV]). Genomic differences between FECV and FIPV have been reported, yet the putative genotypic basis of the highly pathogenic phenotype remains unclear. Here, we used state-of-the-art molecular evolutionary genetic statistical techniques to identify and compare differences in natural selection pressure between FECV and FIPV sequences, as well as to identify FIPV- and FECV-specific signals of positive selection. We analyzed full-length FCoV protein coding genes thought to contain mutations associated with FIPV (Spike, ORF3abc, and ORF7ab). We identified two sites exhibiting differences in natural selection pressure between FECV and FIPV: one within the S1/S2 furin cleavage site (FCS) and the other within the fusion domain of Spike. We also found fifteen sites subject to positive selection associated with FIPV within Spike, eleven of which have not previously been suggested as possibly relevant to FIP development. These sites fall within Spike protein subdomains that participate in host cell receptor interaction, immune evasion, tropism shifts, host cellular entry, and viral escape. There were fourteen sites (twelve novel sites) within Spike under positive selection associated with the FECV phenotype, almost exclusively within the S1/S2 FCS and adjacent to C domain, along with a signal of relaxed selection in FIPV relative to FECV, suggesting that furin cleavage functionality may not be needed for FIPV. Positive selection inferred in ORF7b was associated with the FECV phenotype and included twenty-four positively selected sites, while ORF7b had signals of relaxed selection in FIPV. We found evidence of positive selection in ORF3c in FCoV-wide analyses, but no specific association with the FIPV or FECV phenotype. We hypothesize that some combination of mutations in FECV may contribute to FIP development, and that it is unlikely to be one singular ‘switch’ mutational event. This work expands our understanding of the complexities of FIP development and provides insights into how evolutionary forces may alter pathogenesis in coronavirus genomes.

[end paper 28]

[begin paper 29]

PMCID = 11009965

Title = Computational Approaches to Predict Protein–Protein Interactions in Crowded Cellular Environments

Investigating protein–protein interactions is crucial for understanding cellular biological processes because proteins often function within molecular complexes rather than in isolation. While experimental and computational methods have provided valuable insights into these interactions, they often overlook a critical factor: the crowded cellular environment. This environment significantly impacts protein behavior, including structural stability, diffusion, and ultimately the nature of binding. In this review, we discuss theoretical and computational approaches that allow the modeling of biological systems to guide and complement experiments and can thus significantly advance the investigation, and possibly the predictions, of protein–protein interactions in the crowded environment of cell cytoplasm. We explore topics such as statistical mechanics for lattice simulations, hydrodynamic interactions, diffusion processes in high-viscosity environments, and several methods based on molecular dynamics simulations. By synergistically leveraging methods from biophysics and computational biology, we review the state of the art of computational methods to study the impact of molecular crowding on protein–protein interactions and discuss its potential revolutionizing effects on the characterization of the human interactome.

[end paper 29]

[begin paper 30]

PMCID = 11468914

Title = Structures of the Foamy virus fusion protein reveal an unexpected link with the F protein of paramyxo- and pneumoviruses

Foamy viruses (FVs) constitute a subfamily of retroviruses. Their envelope (Env) glycoprotein drives the merger of viral and cellular membranes during entry into cells. The only available structures of retroviral Envs are those from human and simian immunodeficiency viruses from the subfamily of orthoretroviruses, which are only distantly related to the FVs. We report the cryo–electron microscopy structures of the FV Env ectodomain in the pre- and post-fusion states, which unexpectedly demonstrate structural similarity with the fusion protein (F) of paramyxo- and pneumoviruses, implying an evolutionary link between the viral fusogens. We describe the structural features that are unique to the FV Env and propose a mechanistic model for its conformational change, highlighting how the interplay of its structural elements could drive membrane fusion and viral entry. The structural knowledge on the FV Env now provides a framework for functional investigations, which can benefit the design of FV Env variants with improved features for use as gene therapy vectors.

Structural similarity of the proteins mediating viral entry reveals an evolutionary link between retro- and paramyxoviruses.

[end paper 30]

[begin paper 31]

PMCID = 11783119

Title = Epitope mapping via in vitro deep mutational scanning methods and its applications

Epitope mapping is a technique employed to define the region of an antigen that elicits an immune response, providing crucial insight into the structural architecture of the antigen as well as epitope-paratope interactions. With this breadth of knowledge, immunotherapies, diagnostics, and vaccines are being developed with a rational and data-supported design. Traditional epitope mapping methods are laborious, time-intensive, and often lack the ability to screen proteins in a high-throughput manner or provide high resolution. Deep mutational scanning (DMS), however, is revolutionizing the field as it can screen all possible single amino acid mutations and provide an efficient and high-throughput way to infer the structures of both linear and three-dimensional epitopes with high resolution. Currently, more than 50 publications take this approach to efficiently identify enhancing or escaping mutations, with many then employing this information to rapidly develop broadly neutralizing antibodies, T-cell immunotherapies, vaccine platforms, or diagnostics. We provide a comprehensive review of the approaches to accomplish epitope mapping while also providing a summation of the development of DMS technology and its impactful applications.

[end paper 31]

[begin paper 32]

PMCID = 9789376

Title = Recent developments of sequence-based prediction of protein–protein interactions

The identification of protein–protein interactions (PPIs) can lead to a better understanding of cellular functions and biological processes of proteins and contribute to the design of drugs to target disease-causing PPIs. In addition, targeting host–pathogen PPIs is useful for elucidating infection mechanisms. Although several experimental methods have been used to identify PPIs, these methods can yet to draw complete PPI networks. Hence, computational techniques are increasingly required for the prediction of potential PPIs, which have never been seen experimentally. Recent high-performance sequence-based methods have contributed to the construction of PPI networks and the elucidation of pathogenetic mechanisms in specific diseases. However, the usefulness of these methods depends on the quality and quantity of training data of PPIs. In this brief review, we introduce currently available PPI databases and recent sequence-based methods for predicting PPIs. Also, we discuss key issues in this field and present future perspectives of the sequence-based PPI predictions.

[end paper 32]

[begin paper 33]

PMCID = 9882035

Title = Natural selection differences detected in key protein domains between non-pathogenic and pathogenic Feline Coronavirus phenotypes

Feline Coronaviruses (FCoVs) commonly cause mild enteric infections in felines worldwide (termed Feline Enteric Coronavirus [FECV]), with around 12% developing into deadly Feline Infectious Peritonitis (FIP; Feline Infectious Peritonitis Virus [FIPV]). Genomic differences between FECV and FIPV have been reported, yet the putative genotypic basis of the highly pathogenic phenotype remains unclear. Here, we used state-of-the-art molecular evolutionary genetic statistical techniques to identify and compare differences in natural selection pressure between FECV and FIPV sequences, as well as to identify FIPV and FECV specific signals of positive selection. We analyzed full length FCoV protein coding genes thought to contain mutations associated with FIPV (Spike, ORF3abc, and ORF7ab). We identified two sites exhibiting differences in natural selection pressure between FECV and FIPV: one within the S1/S2 furin cleavage site, and the other within the fusion domain of Spike. We also found 15 sites subject to positive selection associated with FIPV within Spike, 11 of which have not previously been suggested as possibly relevant to FIP development. These sites fall within Spike protein subdomains that participate in host cell receptor interaction, immune evasion, tropism shifts, host cellular entry, and viral escape. There were 14 sites (12 novel) within Spike under positive selection associated with the FECV phenotype, almost exclusively within the S1/S2 furin cleavage site and adjacent C domain, along with a signal of relaxed selection in FIPV relative to FECV, suggesting that furin cleavage functionality may not be needed for FIPV. Positive selection inferred in ORF7b was associated with the FECV phenotype, and included 24 positively selected sites, while ORF7b had signals of relaxed selection in FIPV. We found evidence of positive selection in ORF3c in FCoV wide analyses, but no specific association with the FIPV or FECV phenotype. We hypothesize that some combination of mutations in FECV may contribute to FIP development, and that is unlikely to be one singular “switch” mutational event. This work expands our understanding of the complexities of FIP development and provides insights into how evolutionary forces may alter pathogenesis in coronavirus genomes.

[end paper 33]

[begin paper 34]

PMCID = 11319189

Title = AlphaFold2 in biomedical research: facilitating the development of diagnostic strategies for disease

Proteins, as the primary executors of physiological activity, serve as a key factor in disease diagnosis and treatment. Research into their structures, functions, and interactions is essential to better understand disease mechanisms and potential therapies. DeepMind’s AlphaFold2, a deep-learning protein structure prediction model, has proven to be remarkably accurate, and it is widely employed in various aspects of diagnostic research, such as the study of disease biomarkers, microorganism pathogenicity, antigen-antibody structures, and missense mutations. Thus, AlphaFold2 serves as an exceptional tool to bridge fundamental protein research with breakthroughs in disease diagnosis, developments in diagnostic strategies, and the design of novel therapeutic approaches and enhancements in precision medicine. This review outlines the architecture, highlights, and limitations of AlphaFold2, placing particular emphasis on its applications within diagnostic research grounded in disciplines such as immunology, biochemistry, molecular biology, and microbiology.

[end paper 34]

[begin paper 35]

PMCID = 8744059

Title = From systems to structure — using genetic data to model protein structures

Understanding the effects of genetic variation is a fundamental problem in biology that requires methods to analyse both physical and functional consequences of sequence changes at systems-wide and mechanistic scales. To achieve a systems view, protein interaction networks map which proteins physically interact, while genetic interaction networks inform on the phenotypic consequences of perturbing these protein interactions. Until recently, understanding the molecular mechanisms that underlie these interactions often required biophysical methods to determine the structures of the proteins involved. The past decade has seen the emergence of new approaches based on coevolution, deep mutational scanning and genome-scale genetic or chemical–genetic interaction mapping that enable modelling of the structures of individual proteins or protein complexes. Here, we review the emerging use of large-scale genetic datasets and deep learning approaches to model protein structures and their interactions, and discuss the integration of structural data from different sources.

Large-scale genetic datasets and deep learning approaches are being used to model the structures of proteins or protein complexes. This Review describes approaches based on coevolution, deep mutational scanning and genome-scale genetic or chemical–genetic interaction mapping and their application and integration to inform structural modelling.

[end paper 35]

[begin paper 36]

PMCID = 11221890

Title = Antibody design using deep learning: from sequence and structure design to affinity maturation

Deep learning has achieved impressive results in various fields such as computer vision and natural language processing, making it a powerful tool in biology. Its applications now encompass cellular image classification, genomic studies and drug discovery. While drug development traditionally focused deep learning applications on small molecules, recent innovations have incorporated it in the discovery and development of biological molecules, particularly antibodies. Researchers have devised novel techniques to streamline antibody development, combining in vitro and in silico methods. In particular, computational power expedites lead candidate generation, scaling and potential antibody development against complex antigens. This survey highlights significant advancements in protein design and optimization, specifically focusing on antibodies. This includes various aspects such as design, folding, antibody–antigen interactions docking and affinity maturation.

[end paper 36]

[begin paper 37]

PMCID = 10893003

Title = Recent Progress of Protein Tertiary Structure Prediction

The prediction of three-dimensional (3D) protein structure from amino acid sequences has stood as a significant challenge in computational and structural bioinformatics for decades. Recently, the widespread integration of artificial intelligence (AI) algorithms has substantially expedited advancements in protein structure prediction, yielding numerous significant milestones. In particular, the end-to-end deep learning method AlphaFold2 has facilitated the rise of structure prediction performance to new heights, regularly competitive with experimental structures in the 14th Critical Assessment of Protein Structure Prediction (CASP14). To provide a comprehensive understanding and guide future research in the field of protein structure prediction for researchers, this review describes various methodologies, assessments, and databases in protein structure prediction, including traditionally used protein structure prediction methods, such as template-based modeling (TBM) and template-free modeling (FM) approaches; recently developed deep learning-based methods, such as contact/distance-guided methods, end-to-end folding methods, and protein language model (PLM)-based methods; multi-domain protein structure prediction methods; the CASP experiments and related assessments; and the recently released AlphaFold Protein Structure Database (AlphaFold DB). We discuss their advantages, disadvantages, and application scopes, aiming to provide researchers with insights through which to understand the limitations, contexts, and effective selections of protein structure prediction methods in protein-related fields.

[end paper 37]

[begin paper 38]

PMCID = 10608922

Title = Exploring the World of Membrane Proteins: Techniques and Methods for Understanding Structure, Function, and Dynamics

In eukaryotic cells, membrane proteins play a crucial role. They fall into three categories: intrinsic proteins, extrinsic proteins, and proteins that are essential to the human genome (30% of which is devoted to encoding them). Hydrophobic interactions inside the membrane serve to stabilize integral proteins, which span the lipid bilayer. This review investigates a number of computational and experimental methods used to study membrane proteins. It encompasses a variety of technologies, including electrophoresis, X-ray crystallography, cryogenic electron microscopy (cryo-EM), nuclear magnetic resonance spectroscopy (NMR), biophysical methods, computational methods, and artificial intelligence. The link between structure and function of membrane proteins has been better understood thanks to these approaches, which also hold great promise for future study in the field. The significance of fusing artificial intelligence with experimental data to improve our comprehension of membrane protein biology is also covered in this paper. This effort aims to shed light on the complexity of membrane protein biology by investigating a variety of experimental and computational methods. Overall, the goal of this review is to emphasize how crucial it is to understand the functions of membrane proteins in eukaryotic cells. It gives a general review of the numerous methods used to look into these crucial elements and highlights the demand for multidisciplinary approaches to advance our understanding.

[end paper 38]

[begin paper 39]

PMCID = 9631452

Title = Advances in antibody discovery from human BCR repertoires

Antibodies make up an important and growing class of compounds used for the diagnosis or treatment of disease. While traditional antibody discovery utilized immunization of animals to generate lead compounds, technological innovations have made it possible to search for antibodies targeting a given antigen within the repertoires of B cells in humans. Here we group these innovations into four broad categories: cell sorting allows the collection of cells enriched in specificity to one or more antigens; BCR sequencing can be performed on bulk mRNA, genomic DNA or on paired (heavy-light) mRNA; BCR repertoire analysis generally involves clustering BCRs into specificity groups or more in-depth modeling of antibody-antigen interactions, such as antibody-specific epitope predictions; validation of antibody-antigen interactions requires expression of antibodies, followed by antigen binding assays or epitope mapping. Together with innovations in Deep learning these technologies will contribute to the future discovery of diagnostic and therapeutic antibodies directly from humans.

[end paper 39]

[begin paper 40]

PMCID = 9934699

Title = Massively parallel protein-protein interaction measurement by sequencing (MP3-seq) enables rapid screening of protein heterodimers

Protein-protein interactions (PPIs) regulate many cellular processes, and engineered PPIs have cell and gene therapy applications. Here we introduce massively parallel protein-protein interaction measurement by sequencing (MP3-seq), an easy-to-use and highly scalable yeast-two-hybrid approach for measuring PPIs. In MP3-seq, DNA barcodes are associated with specific protein pairs, and barcode enrichment can be read by sequencing to provide a direct measure of interaction strength. We show that MP3-seq is highly quantitative and scales to over 100,000 interactions. We apply MP3-seq to characterize interactions between families of rationally designed heterodimers and to investigate elements conferring specificity to coiled-coil interactions. Finally, we predict coiled heterodimer structures using AlphaFold-Multimer (AF-M) and train linear models on physics simulation energy terms to predict MP3-seq values. We find that AF-M and AF-M complex prediction-based models could be valuable for pre-screening interactions, but that measuring interactions experimentally remains necessary to rank their strengths quantitatively.

[end paper 40]

[begin paper 41]

PMCID = 8998612

Title = Assortment of Frontiers in Protein Science



[end paper 41]

[begin paper 42]

PMCID = 11788755

Title = Nanobody engineering: computational modelling and design for biomedical and therapeutic applications

Nanobodies, the smallest functional antibody fragment derived from camelid heavy‐chain‐only antibodies, have emerged as powerful tools for diverse biomedical applications. In this comprehensive review, we discuss the structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We explore their unique antigen‐binding domains, highlighting the critical role of complementarity‐determining regions in target recognition and specificity. This review further underscores the advantages of nanobodies over conventional antibodies from a biosynthesis perspective, including their small size, stability, and solubility, which make them ideal candidates for economical antigen capture in diagnostics, therapeutics, and biosensing. We discuss the recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. Finally, we examine a direct example of how computational design strategies were implemented for improving a nanobody‐based immunosensor, known as a Quenchbody. Through combining experimental findings and computational insights, this review elucidates the transformative impact of nanobodies in biotechnology and biomedical research, offering a roadmap for future advancements and applications in healthcare and diagnostics.

This review focuses on structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We discuss recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. It underscores their advantages over conventional antibodies from a biosynthesis perspective in diagnostics, therapeutics, and biosensing. 

[end paper 42]

[begin paper 43]

PMCID = 10148629

Title = A structural model of the iRhom–ADAM17 sheddase complex reveals functional insights into its trafficking and activity

Several membrane-anchored signal mediators such as cytokines (e.g. TNFα) and growth factors are proteolytically shed from the cell surface by the metalloproteinase ADAM17, which, thus, has an essential role in inflammatory and developmental processes. The membrane proteins iRhom1 and iRhom2 are instrumental for the transport of ADAM17 to the cell surface and its regulation. However, the structure–function determinants of the iRhom–ADAM17 complex are poorly understood. We used AI-based modelling to gain insights into the structure–function relationship of this complex. We identified different regions in the iRhom homology domain (IRHD) that are differentially responsible for iRhom functions. We have supported the validity of the predicted structure–function determinants with several in vitro, ex vivo and in vivo approaches and demonstrated the regulatory role of the IRHD for iRhom–ADAM17 complex cohesion and forward trafficking. Overall, we provide mechanistic insights into the iRhom–ADAM17-mediated shedding event, which is at the centre of several important cytokine and growth factor pathways.

The online version contains supplementary material available at 10.1007/s00018-023-04783-y.

[end paper 43]

[begin paper 44]

PMCID = 11788748

Title = Structure‐based computational design of antibody mimetics: challenges and perspectives

The design of antibody mimetics holds great promise for revolutionizing therapeutic interventions by offering alternatives to conventional antibody therapies. Structure‐based computational approaches have emerged as indispensable tools in the rational design of those molecules, enabling the precise manipulation of their structural and functional properties. This review covers the main classes of designed antigen‐binding motifs, as well as alternative strategies to develop tailored ones. We discuss the intricacies of different computational protein–protein interaction design strategies, showcased by selected successful cases in the literature. Subsequently, we explore the latest advancements in the computational techniques including the integration of machine and deep learning methodologies into the design framework, which has led to an augmented design pipeline. Finally, we verse onto the current challenges that stand in the way between high‐throughput computer design of antibody mimetics and experimental realization, offering a forward‐looking perspective into the field and the promises it holds to biotechnology.

Computer graphics representation of an artificial intelligence–driven protein design pipeline, where building blocks are fed into the birth of a novel protein structure. 

[end paper 44]

[begin paper 45]

PMCID = 11226102

Title = Analysis of AlphaFold and molecular dynamics structure predictions of mutations in serpins

Serine protease inhibitors (serpins) include thousands of structurally conserved proteins playing key roles in many organisms. Mutations affecting serpins may disturb their conformation, leading to inactive forms. Unfortunately, conformational consequences of serpin mutations are difficult to predict. In this study, we integrate experimental data of patients with mutations affecting one serpin with the predictions obtained by AlphaFold and molecular dynamics. Five SERPINC1 mutations causing antithrombin deficiency, the strongest congenital thrombophilia were selected from a cohort of 350 unrelated patients based on functional, biochemical, and crystallographic evidence supporting a folding defect. AlphaFold gave an accurate prediction for the wild-type structure. However, it also produced native structures for all variants, regardless of complexity or conformational consequences in vivo. Similarly, molecular dynamics of up to 1000 ns at temperatures causing conformational transitions did not show significant changes in the native structure of wild-type and variants. In conclusion, AlphaFold and molecular dynamics force predictions into the native conformation at conditions with experimental evidence supporting a conformational change to other structures. It is necessary to improve predictive strategies for serpins that consider the conformational sensitivity of these molecules.

[end paper 45]

[begin paper 46]

PMCID = 11221889

Title = AttABseq: an attention-based deep learning prediction method for antigen–antibody binding affinity changes based on protein sequences

The optimization of therapeutic antibodies through traditional techniques, such as candidate screening via hybridoma or phage display, is resource-intensive and time-consuming. In recent years, computational and artificial intelligence-based methods have been actively developed to accelerate and improve the development of therapeutic antibodies. In this study, we developed an end-to-end sequence-based deep learning model, termed AttABseq, for the predictions of the antigen–antibody binding affinity changes connected with antibody mutations. AttABseq is a highly efficient and generic attention-based model by utilizing diverse antigen–antibody complex sequences as the input to predict the binding affinity changes of residue mutations. The assessment on the three benchmark datasets illustrates that AttABseq is 120% more accurate than other sequence-based models in terms of the Pearson correlation coefficient between the predicted and experimental binding affinity changes. Moreover, AttABseq also either outperforms or competes favorably with the structure-based approaches. Furthermore, AttABseq consistently demonstrates robust predictive capabilities across a diverse array of conditions, underscoring its remarkable capacity for generalization across a wide spectrum of antigen-antibody complexes. It imposes no constraints on the quantity of altered residues, rendering it particularly applicable in scenarios where crystallographic structures remain unavailable. The attention-based interpretability analysis indicates that the causal effects of point mutations on antibody–antigen binding affinity changes can be visualized at the residue level, which might assist automated antibody sequence optimization. We believe that AttABseq provides a fiercely competitive answer to therapeutic antibody optimization.

[end paper 46]

[begin paper 47]

PMCID = 8950966

Title = Modeling the Dynamics of Protein–Protein Interfaces, How and Why?

Protein–protein assemblies act as a key component in numerous cellular processes. Their accurate modeling at the atomic level remains a challenge for structural biology. To address this challenge, several docking and a handful of deep learning methodologies focus on modeling protein–protein interfaces. Although the outcome of these methods has been assessed using static reference structures, more and more data point to the fact that the interaction stability and specificity is encoded in the dynamics of these interfaces. Therefore, this dynamics information must be taken into account when modeling and assessing protein interactions at the atomistic scale. Expanding on this, our review initially focuses on the recent computational strategies aiming at investigating protein–protein interfaces in a dynamic fashion using enhanced sampling, multi-scale modeling, and experimental data integration. Then, we discuss how interface dynamics report on the function of protein assemblies in globular complexes, in fuzzy complexes containing intrinsically disordered proteins, as well as in active complexes, where chemical reactions take place across the protein–protein interface.

[end paper 47]

[begin paper 48]

PMCID = 10220908

Title = Structural Modeling of Nanobodies: A Benchmark of State-of-the-Art Artificial Intelligence Programs

The number of applications for nanobodies is steadily expanding, positioning these molecules as fast-growing biologic products in the biotechnology market. Several of their applications require protein engineering, which in turn would greatly benefit from having a reliable structural model of the nanobody of interest. However, as with antibodies, the structural modeling of nanobodies is still a challenge. With the rise of artificial intelligence (AI), several methods have been developed in recent years that attempt to solve the problem of protein modeling. In this study, we have compared the performance in nanobody modeling of several state-of-the-art AI-based programs, either designed for general protein modeling, such as AlphaFold2, OmegaFold, ESMFold, and Yang-Server, or specifically designed for antibody modeling, such as IgFold, and Nanonet. While all these programs performed rather well in constructing the nanobody framework and CDRs 1 and 2, modeling CDR3 still represents a big challenge. Interestingly, tailoring an AI method for antibody modeling does not necessarily translate into better results for nanobodies.

[end paper 48]

[begin paper 49]

PMCID = 11543627

Title = Prediction of conformational states in a coronavirus channel using Alphafold-2 and DeepMSA2: Strengths and limitations

The envelope (E) protein is present in all coronavirus genera. This protein can form pentameric oligomers with ion channel activity which have been proposed as a possible therapeutic target. However, high resolution structures of E channels are limited to those of the severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2), responsible for the recent COVID-19 pandemic. In the present work, we used Alphafold-2 (AF2), in ColabFold without templates, to predict the transmembrane domain (TMD) structure of six E-channels representative of genera alpha-, beta- and gamma-coronaviruses in the Coronaviridae family. High-confidence models were produced in all cases when combining multiple sequence alignments (MSAs) obtained from DeepMSA2. Overall, AF2 predicted at least two possible orientations of the α-helices in E-TMD channels: one where a conserved polar residue (Asn-15 in the SARS sequence) is oriented towards the center of the channel, ‘polar-in’, and one where this residue is in an interhelical orientation ‘polar-inter’. For the SARS models, the comparison with the two experimental models ‘closed’ (PDB: 7K3G) and ‘open’ (PDB: 8SUZ) is described, and suggests a ∼60˚ α-helix rotation mechanism involving either the full TMD or only its N-terminal half, to allow the passage of ions. While the results obtained are not identical to the two high resolution models available, they suggest various conformational states with striking similarities to those models. We believe these results can be further optimized by means of MSA subsampling, and guide future high resolution structural studies in these and other viral channels.

[end paper 49]

[begin paper 50]

PMCID = 10459022

Title = Just Keep Rolling?—An Encompassing Review towards Accelerated Vaccine Product Life Cycles

In light of the recent pandemic, several COVID-19 vaccines were developed, tested and approved in a very short time, a process that otherwise takes many years. Above all, these efforts have also unmistakably revealed the capacity limits and potential for improvement in vaccine production. This review aims to emphasize recent approaches for the targeted rapid adaptation and production of vaccines from an interdisciplinary, multifaceted perspective. Using research from the literature, stakeholder analysis and a value proposition canvas, we reviewed technological innovations on the pharmacological level, formulation, validation and resilient vaccine production to supply bottlenecks and logistic networks. We identified four main drivers to accelerate the vaccine product life cycle: computerized candidate screening, modular production, digitized quality management and a resilient business model with corresponding transparent supply chains. In summary, the results presented here can serve as a guide and implementation tool for flexible, scalable vaccine production to swiftly respond to pandemic situations in the future.

[end paper 50]

[begin paper 51]

PMCID = 10231047

Title = The impact of AlphaFold2 on experimental structure solution

AlphaFold2 is a machine-learning based program that predicts a protein structure based on the amino acid sequence. In this article, we report on the current usages of this new tool and give examples from our work in the Coronavirus Structural Task Force. With its unprecedented accuracy, it can be utilized for the design of expression constructs, de novo protein design and the interpretation of Cryo-EM data with an atomic model. However, these methods are limited by their training data and are of limited use to predict conformational variability and fold flexibility; they also lack co-factors, post-translational modifications and multimeric complexes with oligonucleotides. They also are not always perfect in terms of chemical geometry. Nevertheless, machine learning-based fold prediction is a game changer for structural bioinformatics and experimentalists alike, with exciting developments ahead.

AlphaFold2 predicts protein folds from sequence, which can be used for experimental structural biology, in construction and de novo protein design, prediction of complexes and perhaps even effects of mutations and conformational space exploration.

[end paper 51]

[begin paper 52]

PMCID = 11251186

Title = Higher-order structure and proteoforms of co-occurring C4b-binding protein assemblies in human serum

The complement is a conserved cascade that plays a central role in the innate immune system. To maintain a delicate equilibrium preventing excessive complement activation, complement inhibitors are essential. One of the major fluid-phase complement inhibitors is C4b-binding protein (C4BP). Human C4BP is a macromolecular glycoprotein composed of two distinct subunits, C4BPα and C4BPβ. These associate with vitamin K-dependent protein S (ProS) forming an ensemble of co-occurring higher-order structures. Here, we characterize these C4BP assemblies. We resolve and quantify isoforms of purified human serum C4BP using distinct single-particle detection techniques: charge detection mass spectrometry, and mass photometry accompanied by high-speed atomic force microscopy. Combining cross-linking mass spectrometry, glycoproteomics, and structural modeling, we report comprehensive glycoproteoform profiles and full-length structural models of the endogenous C4BP assemblies, expanding knowledge of this key complement inhibitor’s structure and composition. Finally, we reveal that an increased C4BPα to C4BPβ ratio coincides with elevated C-reactive protein levels in patient plasma samples. This observation highlights C4BP isoform variation and affirms a distinct role of co-occurring C4BP assemblies upon acute phase inflammation.

C4b-binding protein complement inhibitor forms co-occurring assemblies in human serum. Here, by using an integrative structural biology approach, structural and compositional details of C4BP are exposed, resulting in structural models of full-length glycosylated C4BP assemblies.

Integrating high-speed atomic force microscopy and cross-linking mass spectrometry we present full-length structural models of C4BP, a spider-like assembly of C4BPα and C4BPβ, non-covalently interacting with vitamin K-dependent protein S.

Dominant C4BP variants in healthy human serum are C4BP(β+) captured and analyzed by single molecule mass photometry and charge detection mass spectrometry.

Quantitative serum proteomics reveals that C4BP variant composition changes during acute phase inflammation, with an increase in α7 variant abundance.

Structual analysis of complement inhibitor C4BP uncovers full-length spider-like assemblies of C4BPα and C4BPβ subunits and composition changes during acute phase inflammation.

[end paper 52]

[begin paper 53]

PMCID = 11251597

Title = Mapping mutational fitness effects across the coxsackievirus B3 proteome reveals distinct profiles of mutation tolerability

RNA viruses have notoriously high mutation rates due to error-prone replication by their RNA polymerase. However, natural selection concentrates variability in a few key viral proteins. To test whether this stems from different mutation tolerance profiles among viral proteins, we measured the effect of >40,000 non-synonymous mutations across the full proteome of coxsackievirus B3 as well as >97% of all possible codon deletions in the nonstructural proteins. We find significant variation in mutational tolerance within and between individual viral proteins, which correlated with both general and protein-specific structural and functional attributes. Furthermore, mutational fitness effects remained stable across cell lines, suggesting selection pressures are mostly conserved across environments. In addition to providing a rich dataset for understanding virus biology and evolution, our results illustrate that incorporation of mutational tolerance data into druggable pocket discovery can aid in selecting targets with high barriers to drug resistance.

RNA viruses replicate with high mutation rates, but natural selection concentrates variability in a few proteins. This study measures the effect of >40,000 non-synonymous mutations across the full proteome of coxsackievirus B3, revealing significant variation in mutational tolerance within and between viral proteins, with implications for drug target selection.

[end paper 53]

[begin paper 54]

PMCID = 10302489

Title = Recent Advances in Studying Toll-like Receptors with the Use of Computational Methods

Toll-like receptors (TLRs) are transmembrane proteins that recognize various molecular patterns and activate signaling that triggers the immune response. In this review, our goal is to summarize how, in recent years, various computational solutions have contributed to a better understanding of TLRs, regarding both their function and mechanism of action. We update the recent information about small-molecule modulators and expanded the topic toward next-generation vaccine design, as well as studies of the dynamic nature of TLRs. Also, we underline problems that remain unsolved.

[end paper 54]

[begin paper 55]

PMCID = 11128484

Title = Accurate prediction of antibody function and structure using bio-inspired antibody language model

In recent decades, antibodies have emerged as indispensable therapeutics for combating diseases, particularly viral infections. However, their development has been hindered by limited structural information and labor-intensive engineering processes. Fortunately, significant advancements in deep learning methods have facilitated the precise prediction of protein structure and function by leveraging co-evolution information from homologous proteins. Despite these advances, predicting the conformation of antibodies remains challenging due to their unique evolution and the high flexibility of their antigen-binding regions. Here, to address this challenge, we present the Bio-inspired Antibody Language Model (BALM). This model is trained on a vast dataset comprising 336 million 40% nonredundant unlabeled antibody sequences, capturing both unique and conserved properties specific to antibodies. Notably, BALM showcases exceptional performance across four antigen-binding prediction tasks. Moreover, we introduce BALMFold, an end-to-end method derived from BALM, capable of swiftly predicting full atomic antibody structures from individual sequences. Remarkably, BALMFold outperforms those well-established methods like AlphaFold2, IgFold, ESMFold and OmegaFold in the antibody benchmark, demonstrating significant potential to advance innovative engineering and streamline therapeutic antibody development by reducing the need for unnecessary trials. The BALMFold structure prediction server is freely available at https://beamlab-sh.com/models/BALMFold.

[end paper 55]

[begin paper 56]

PMCID = 9302003

Title = Structural Features of Antibody-Peptide Recognition

Antibody recognition of antigens is a critical element of adaptive immunity. One key class of antibody-antigen complexes is comprised of antibodies targeting linear epitopes of proteins, which in some cases are conserved elements of viruses and pathogens of relevance for vaccine design and immunotherapy. Here we report a detailed analysis of the structural and interface features of this class of complexes, based on a set of nearly 200 nonredundant high resolution antibody-peptide complex structures that were assembled from the Protein Data Bank. We found that antibody-bound peptides adopt a broad range of conformations, often displaying limited secondary structure, and that the same peptide sequence bound by different antibodies can in many cases exhibit varying conformations. Propensities of contacts with antibody loops and extent of antibody binding conformational changes were found to be broadly similar to those for antibodies in complex with larger protein antigens. However, antibody-peptide interfaces showed lower buried surface areas and fewer hydrogen bonds than antibody-protein antigen complexes, while calculated binding energy per buried interface area was found to be higher on average for antibody-peptide interfaces, likely due in part to a greater proportion of buried hydrophobic residues and higher shape complementarity. This dataset and these observations can be of use for future studies focused on this class of interactions, including predictive computational modeling efforts and the design of antibodies or epitope-based vaccine immunogens.

[end paper 56]

[begin paper 57]

PMCID = 11471525

Title = Integrating machine learning to advance epitope mapping

Identifying epitopes, or the segments of a protein that bind to antibodies, is critical for the development of a variety of immunotherapeutics and diagnostics. In vaccine design, the intent is to identify the minimal epitope of an antigen that can elicit an immune response and avoid off-target effects. For prognostics and diagnostics, the epitope-antibody interaction is exploited to measure antigens associated with disease outcomes. Experimental methods such as X-ray crystallography, cryo-electron microscopy, and peptide arrays are used widely to map epitopes but vary in accuracy, throughput, cost, and feasibility. By comparing machine learning epitope mapping tools, we discuss the importance of data selection, feature design, and algorithm choice in determining the specificity and prediction accuracy of an algorithm. This review discusses limitations of current methods and the potential for machine learning to deepen interpretation and increase feasibility of these methods. We also propose how machine learning can be employed to refine epitope prediction to address the apparent promiscuity of polyreactive antibodies and the challenge of defining conformational epitopes. We highlight the impact of machine learning on our current understanding of epitopes and its potential to guide the design of therapeutic interventions with more predictable outcomes.

[end paper 57]

[begin paper 58]

PMCID = 9994104

Title = Protein–protein interfaces in molecular glue-induced ternary complexes: classification, characterization, and prediction

Molecular glues are a class of small molecules that stabilize the interactions between proteins. Naturally occurring molecular glues are present in many areas of biology where they serve as central regulators of signaling pathways. Importantly, several clinical compounds act as molecular glue degraders that stabilize interactions between E3 ubiquitin ligases and target proteins, leading to their degradation. Molecular glues hold promise as a new generation of therapeutic agents, including those molecular glue degraders that can redirect the protein degradation machinery in a precise way. However, rational discovery of molecular glues is difficult in part due to the lack of understanding of the protein–protein interactions they stabilize. In this review, we summarize the structures of known molecular glue-induced ternary complexes and the interface properties. Detailed analysis shows different mechanisms of ternary structure formation. Additionally, we also review computational approaches for predicting protein–protein interfaces and highlight the promises and challenges. This information will ultimately help inform future approaches for rational molecular glue discovery.

This review surveys molecular glue-induced ternary complexes in the PDB and provides an overview of computational methods that can be utilized to predict them.

[end paper 58]

[begin paper 59]

PMCID = 11695057

Title = The co-receptor Tetraspanin12 directly captures Norrin to promote ligand-specific β-catenin signaling

Wnt/β-catenin signaling directs animal development and tissue renewal in a tightly controlled, cell- and tissue-specific manner. In the mammalian central nervous system, the atypical ligand Norrin controls angiogenesis and maintenance of the blood-brain barrier and blood-retina barrier through the Wnt/β-catenin pathway. Like Wnt, Norrin activates signaling by binding and heterodimerizing the receptors Frizzled (Fzd) and low-density lipoprotein receptor-related protein 5 or 6 (LRP5/6), leading to membrane recruitment of the intracellular transducer Dishevelled (Dvl) and ultimately stabilizing the transcriptional coactivator β-catenin. Unlike Wnt, the cystine knot ligand Norrin only signals through Fzd4 and additionally requires the co-receptor Tetraspanin12 (Tspan12); however, the mechanism underlying Tspan12-mediated signal enhancement is unclear. It has been proposed that Tspan12 integrates into the Norrin-Fzd4 complex to enhance Norrin-Fzd4 affinity or otherwise allosterically modulate Fzd4 signaling. Here, we measure direct, high-affinity binding between purified Norrin and Tspan12 in a lipid environment and use AlphaFold models to interrogate this interaction interface. We find that Tspan12 and Fzd4 can simultaneously bind Norrin and that a pre-formed Tspan12/Fzd4 heterodimer, as well as cells co-expressing Tspan12 and Fzd4, more efficiently capture low concentrations of Norrin than Fzd4 alone. We also show that Tspan12 competes with both heparan sulfate proteoglycans and LRP6 for Norrin binding and that Tspan12 does not impact Fzd4-Dvl affinity in the presence or absence of Norrin. Our findings suggest that Tspan12 does not allosterically enhance Fzd4 binding to Norrin or Dvl, but instead functions to directly capture Norrin upstream of signaling.

[end paper 59]

[begin paper 60]

PMCID = 9753067

Title = The evolution of structural genomics

Structural genomics began as a global effort in the 1990s to determine the tertiary structures of all protein families as a response to large-scale genome sequencing projects. The immediate outcome was an influx of tens of thousands of protein structures, many of which had unknown functions. At the time, the value of structural genomics was controversial. However, the structures themselves were only the most obvious output. In addition, these newly solved structures motivated the emergence of huge data science and infrastructure efforts, which, together with advances in Deep Learning, have brought about a revolution in computational molecular biology. Here, we review some of the computational research carried out at the Protein Data Bank Japan (PDBj) during the Protein 3000 project under the leadership of Haruki Nakamura, much of which continues to flourish today. 

[end paper 60]

[begin paper 61]

PMCID = 9607061

Title = Recent Emergence of Bovine Coronavirus Variants with Mutations in the Hemagglutinin-Esterase Receptor Binding Domain in U.S. Cattle

Bovine coronavirus (BCoV) has spilled over to many species, including humans, where the host range variant coronavirus OC43 is endemic. The balance of the opposing activities of the surface spike (S) and hemagglutinin-esterase (HE) glycoproteins controls BCoV avidity, which is critical for interspecies transmission and host adaptation. Here, 78 genomes were sequenced directly from clinical samples collected between 2013 and 2022 from cattle in 12 states, primarily in the Midwestern U.S. Relatively little genetic diversity was observed, with genomes having >98% nucleotide identity. Eleven isolates collected between 2020 and 2022 from four states (Nebraska, Colorado, California, and Wisconsin) contained a 12 nucleotide insertion in the receptor-binding domain (RBD) of the HE gene similar to one recently reported in China, and a single genome from Nebraska collected in 2020 contained a novel 12 nucleotide deletion in the HE gene RBD. Isogenic HE proteins containing either the insertion or deletion in the HE RBD maintained esterase activity and could bind bovine submaxillary mucin, a substrate enriched in the receptor 9-O-acetylated-sialic acid, despite modeling that predicted structural changes in the HE R3 loop critical for receptor binding. The emergence of BCoV with structural variants in the RBD raises the possibility of further interspecies transmission.

[end paper 61]

[begin paper 62]

PMCID = 11763998

Title = Proximity Labeling: Precise Proteomics Technology for Mapping Receptor Protein Neighborhoods at the Cancer Cell Surface

Cell surface proteins and receptors on cancer cells, important as disease markers and drug targets, are highly dynamic, with their biological functions controlled by constantly changing protein–protein interaction networks. Studying these transient macromolecular neighborhoods is technically challenging, and traditional methods often fall short. Proximity labeling, a cutting-edge proteomics technology, offers a more precise approach that overcomes the limitations of existing methods. This review highlights the strategies, applications, benefits, and limitations of different proximity labeling platforms used to map the receptor protein microenvironment on the surface of cancer cells.

Cell surface receptors are pivotal to cancer cell transformation, disease progression, metastasis, early detection, targeted therapy, drug responses, and clinical outcomes. Since they coordinate complex signaling communication networks in the tumor microenvironment, mapping the physical interaction partners of cell surface receptors in vivo is vital for understanding their roles, functional states, and suitability as therapeutic targets. Yet traditional methods like immunoprecipitation and affinity purification–mass spectrometry often fail to detect key but weak or transient receptor–protein interactions. Proximity labeling, a cutting-edge proteomics technology, addresses these technical challenges by enabling precise mapping of protein neighborhoods around a receptor target on the cell surface of cancer cells. This technique has been successfully applied in vitro and in vivo for proteomic mapping across various model systems. This review explores the fundamental principles, technologies, advantages, limitations, and applications of proximity labeling in cancer biology, focusing on mapping receptor microenvironments. By advancing mechanistic insights into cancer cell receptor signaling mechanisms, proximity labeling is poised to transform cancer research, improve targeted therapies, and illuminate avenues to overcome drug resistance.

[end paper 62]

[begin paper 63]

PMCID = 9908307

Title = Can we predict T cell specificity with digital biology and machine learning?

Recent advances in machine learning and experimental biology have offered breakthrough solutions to problems such as protein structure prediction that were long thought to be intractable. However, despite the pivotal role of the T cell receptor (TCR) in orchestrating cellular immunity in health and disease, computational reconstruction of a reliable map from a TCR to its cognate antigens remains a holy grail of systems immunology. Current data sets are limited to a negligible fraction of the universe of possible TCR–ligand pairs, and performance of state-of-the-art predictive models wanes when applied beyond these known binders. In this Perspective article, we make the case for renewed and coordinated interdisciplinary effort to tackle the problem of predicting TCR–antigen specificity. We set out the general requirements of predictive models of antigen binding, highlight critical challenges and discuss how recent advances in digital biology such as single-cell technology and machine learning may provide possible solutions. Finally, we describe how predicting TCR specificity might contribute to our understanding of the broader puzzle of antigen immunogenicity.

Koohy and co-workers discuss how we must turn to machine-learning approaches to define the antigen specificity of the many millions of possible T cell receptors. They review the models and methods currently being used to predict cognate antigens for orphan T cell receptors.

[end paper 63]

[begin paper 64]

PMCID = 8897956

Title = Urgency of COVID-19 vaccination in adolescents: Androgen and estrogen receptors view



[end paper 64]

[begin paper 65]

PMCID = 10896983

Title = Pathological mutations reveal the key role of the cytosolic iRhom2 N-terminus for phosphorylation-independent 14-3-3 interaction and ADAM17 binding, stability, and activity

The protease ADAM17 plays an important role in inflammation and cancer and is regulated by iRhom2. Mutations in the cytosolic N-terminus of human iRhom2 cause tylosis with oesophageal cancer (TOC). In mice, partial deletion of the N-terminus results in a curly hair phenotype (cub). These pathological consequences are consistent with our findings that iRhom2 is highly expressed in keratinocytes and in oesophageal cancer. Cub and TOC are associated with hyperactivation of ADAM17-dependent EGFR signalling. However, the underlying molecular mechanisms are not understood. We have identified a non-canonical, phosphorylation-independent 14-3-3 interaction site that encompasses all known TOC mutations. Disruption of this site dysregulates ADAM17 activity. The larger cub deletion also includes the TOC site and thus also dysregulated ADAM17 activity. The cub deletion, but not the TOC mutation, also causes severe reductions in stimulated shedding, binding, and stability of ADAM17, demonstrating the presence of additional regulatory sites in the N-terminus of iRhom2. Overall, this study contrasts the TOC and cub mutations, illustrates their different molecular consequences, and reveals important key functions of the iRhom2 N-terminus in regulating ADAM17.

The online version contains supplementary material available at 10.1007/s00018-024-05132-3.

[end paper 65]

[begin paper 66]

PMCID = 11708891

Title = Adaptive sampling–based structural prediction reveals opening of a GABAA receptor through the αβ interface

γ-Aminobutyric acid type A (GABAA) receptors are ligand-gated ion channels in the central nervous system with largely inhibitory function. Despite being a target for drugs including general anesthetics and benzodiazepines, experimental structures have yet to capture an open state of classical synaptic α1β2γ2 GABAA receptors. Here, we use a goal-oriented adaptive sampling strategy in molecular dynamics simulations followed by Markov state modeling to capture an energetically stable putative open state of the receptor. The model conducts chloride ions with comparable conductance as in electrophysiology measurements. Relative to experimental structures, our open model is relatively expanded at both the cytoplasmic (−2′) and central (9′) gates, coordinated with distinctive rearrangements at the transmembrane αβ subunit interface. Consistent with previous experiments, targeted substitutions disrupting interactions at this interface slowed the open-to-desensitized transition rate. This work demonstrates the capacity of advanced simulation techniques to investigate a computationally and experimentally plausible functionally critical of a complex membrane protein yet to be resolved by experimental methods.

Computational methods allow structure prediction of an open-state GABAA receptor, currently inaccessible in the laboratory.

[end paper 66]

[begin paper 67]

PMCID = 11257942

Title = The palisade layer of the poxvirus core is composed of flexible A10 trimers

Due to its asymmetric shape, size and compactness, the structure of the infectious mature virus (MV) of vaccinia virus (VACV), the best-studied poxvirus, remains poorly understood. Instead, subviral particles, in particular membrane-free viral cores, have been studied with cryo-electron microscopy. Here, we compared viral cores obtained by detergent stripping of MVs with cores in the cellular cytoplasm, early in infection. We focused on the prominent palisade layer on the core surface, combining cryo-electron tomography, subtomogram averaging and AlphaFold2 structure prediction. We showed that the palisade is composed of densely packed trimers of the major core protein A10. Trimers display a random order and their classification indicates structural flexibility. A10 on cytoplasmic cores is organized in a similar manner, indicating that the structures obtained in vitro are physiologically relevant. We discuss our results in the context of the VACV replicative cycle, and the assembly and disassembly of the infectious MV.

The authors employ cryo-ET to study the structure of the palisade layer of the mature vaccinia virus core in isolation, as well as inside infected cells, revealing that it is composed of A10 protein trimers.

[end paper 67]

[begin paper 68]

PMCID = 8916015

Title = Researchers turn to deep learning to decode protein structures



[end paper 68]

[begin paper 69]

PMCID = 10798987

Title = Development and use of machine learning algorithms in vaccine target selection

Computer-aided discovery of vaccine targets has become a cornerstone of rational vaccine design. In this article, I discuss how Machine Learning (ML) can inform and guide key computational steps in rational vaccine design concerned with the identification of B and T cell epitopes and correlates of protection. I provide examples of ML models, as well as types of data and predictions for which they are built. I argue that interpretable ML has the potential to improve the identification of immunogens also as a tool for scientific discovery, by helping elucidate the molecular processes underlying vaccine-induced immune responses. I outline the limitations and challenges in terms of data availability and method development that need to be addressed to bridge the gap between advances in ML predictions and their translational application to vaccine design.

[end paper 69]

[begin paper 70]

PMCID = 9795047

Title = A new dawn for monoclonal antibodies against antimicrobial resistant bacteria

Antimicrobial resistance (AMR) is a quickly advancing threat for human health worldwide and almost 5 million deaths are already attributable to this phenomenon every year. Since antibiotics are failing to treat AMR-bacteria, new tools are needed, and human monoclonal antibodies (mAbs) can fill this role. In almost 50 years since the introduction of the first technology that led to mAb discovery, enormous leaps forward have been made to identify and develop extremely potent human mAbs. While their usefulness has been extensively proved against viral pathogens, human mAbs have yet to find their space in treating and preventing infections from AMR-bacteria and fully conquer the field of infectious diseases. The novel and most innovative technologies herein reviewed can support this goal and add powerful tools in the arsenal of weapons against AMR.

[end paper 70]

[begin paper 71]

PMCID = 10299731

Title = Challenges and frontiers of computational modelling of biomolecular recognition

Biomolecular recognition including binding of small molecules, peptides and proteins to their target receptors plays a key role in cellular function and has been targeted for therapeutic drug design. However, the high flexibility of biomolecules and slow binding and dissociation processes have presented challenges for computational modelling. Here, we review the challenges and computational approaches developed to characterise biomolecular binding, including molecular docking, molecular dynamics simulations (especially enhanced sampling) and machine learning. Further improvements are still needed in order to accurately and efficiently characterise binding structures, mechanisms, thermodynamics and kinetics of biomolecules in the future.

[end paper 71]

[begin paper 72]

PMCID = 11492285

Title = CHARMM at 45: Enhancements in Accessibility, Functionality, and Speed

Since its inception nearly a half century ago, CHARMM has been playing a central role in computational biochemistry and biophysics. Commensurate with the developments in experimental research and advances in computer hardware, the range of methods and applicability of CHARMM have also grown. This review summarizes major developments that occurred after 2009 when the last review of CHARMM was published. They include the following: new faster simulation engines, accessible user interfaces for convenient workflows, and a vast array of simulation and analysis methods that encompass quantum mechanical, atomistic, and coarse-grained levels, as well as extensive coverage of force fields. In addition to providing the current snapshot of the CHARMM development, this review may serve as a starting point for exploring relevant theories and computational methods for tackling contemporary and emerging problems in biomolecular systems. CHARMM is freely available for academic and nonprofit research at .

[end paper 72]

[begin paper 73]

PMCID = 11364091

Title = Reversibly photoswitchable protein assemblies with collagen affinity for in vivo photoacoustic imaging of tumors

Recent advancements in photoacoustic (PA) imaging have leveraged reversibly photoswitchable chromophores, known for their dual absorbance states, to enhance imaging sensitivity through differential techniques. Yet, their deployment in tumor imaging has faced obstacles in achieving targeted delivery with high efficiency and specificity. Addressing this challenge, we introduce innovative protein assemblies, DrBphP-CBD, by genetically fusing a photosensory module from Deinococcus radiodurans bacterial phytochrome (DrBphP) with a collagen-binding domain (CBD). These protein assemblies form sub-100-nanometer structures composed of 24 DrBphP dimers and 12 CBD trimers, presenting 24 protein subunits. Their affinity for collagens, combined with impressive photoswitching contrast, markedly improves PA imaging precision. In various tumor models, intravenous administration of DrBphP-CBD has demonstrated enhanced tumor targeting and retention, augmenting contrast in PA imaging by minimizing background noise. This strategy underscores the clinical potential of DrBphP-CBD as PA contrast agents, propelling photoswitchable chromoproteins to the forefront of precise cancer diagnosis.

DrBphP-CBD assemblies with collagen affinity and photoswitching capability enhance tumor PA imaging specificity and contrast.

[end paper 73]

[begin paper 74]

PMCID = 11432724

Title = Predicting Mutation-Induced Allosteric Changes in Structures and Conformational Ensembles of the ABL Kinase Using AlphaFold2 Adaptations with Alanine Sequence Scanning

Despite the success of AlphaFold2 approaches in predicting single protein structures, these methods showed intrinsic limitations in predicting multiple functional conformations of allosteric proteins and have been challenged to accurately capture the effects of single point mutations that induced significant structural changes. We examined several implementations of AlphaFold2 methods to predict conformational ensembles for state-switching mutants of the ABL kinase. The results revealed that a combination of randomized alanine sequence masking with shallow multiple sequence alignment subsampling can significantly expand the conformational diversity of the predicted structural ensembles and capture shifts in populations of the active and inactive ABL states. Consistent with the NMR experiments, the predicted conformational ensembles for M309L/L320I and M309L/H415P ABL mutants that perturb the regulatory spine networks featured the increased population of the fully closed inactive state. The proposed adaptation of AlphaFold can reproduce the experimentally observed mutation-induced redistributions in the relative populations of the active and inactive ABL states and capture the effects of regulatory mutations on allosteric structural rearrangements of the kinase domain. The ensemble-based network analysis complemented AlphaFold predictions by revealing allosteric hotspots that correspond to state-switching mutational sites which may explain the global effect of regulatory mutations on structural changes between the ABL states. This study suggested that attention-based learning of long-range dependencies between sequence positions in homologous folds and deciphering patterns of allosteric interactions may further augment the predictive abilities of AlphaFold methods for modeling of alternative protein sates, conformational ensembles and mutation-induced structural transformations.

[end paper 74]

[begin paper 75]

PMCID = 10312457

Title = Unveiling Liquid-Crystalline Lipids in the Urothelial Membrane through Cryo-EM

The urothelium, a distinct epithelial tissue lining the urinary tract, serves as an essential component in preserving urinary tract integrity and thwarting infections. The asymmetric unit membrane (AUM), primarily composed of the uroplakin complex, constitutes a critical permeability barrier in fulfilling this role. However, the molecular architectures of both the AUM and the uroplakin complex have remained enigmatic due to the paucity of high-resolution structural data. In this study, we utilized cryo-electron microscopy to elucidate the three-dimensional structure of the uroplakin complex within the porcine AUM. While the global resolution achieved was 3.5 Å, we acknowledge that due to orientation bias, the resolution in the vertical direction was determined to be 6.3 Å. Our findings unveiled that the uroplakin complexes are situated within hexagonally arranged crystalline lipid membrane domains, rich in hexosylceramides. Moreover, our research rectifies a misconception in a previous model by confirming the existence of a domain initially believed to be absent, and pinpointing the accurate location of a crucial Escherichia coli binding site implicated in urinary tract infections. These discoveries offer valuable insights into the molecular underpinnings governing the permeability barrier function of the urothelium and the orchestrated lipid phase formation within the plasma membrane.

[end paper 75]

[begin paper 76]

PMCID = 10371089

Title = Unveiling Liquid-Crystalline Lipids in the Urothelial Membrane through Cryo-EM

The urothelium, a distinct epithelial tissue lining the urinary tract, serves as an essential component in preserving urinary tract integrity and thwarting infections. The asymmetric unit membrane (AUM), primarily composed of the uroplakin complex, constitutes a critical permeability barrier in fulfilling this role. However, the molecular architectures of both the AUM and the uroplakin complex have remained enigmatic due to the paucity of high-resolution structural data. In this investigation, we employed cryoelectron microscopy to elucidate the three-dimensional structure of the uroplakin complex embedded within the porcine AUM at a resolution of 3.5 Å. Our findings unveiled that the uroplakin complexes are situated within hexagonally arranged crystalline lipid membrane domains, rich in hexosylceramides. Moreover, our research rectifies a misconception in a previous model by confirming the existence of a domain initially believed to be absent, and pinpointing the accurate location of a crucial Escherichia coli binding site implicated in urinary tract infections. These discoveries offer valuable insights into the molecular underpinnings governing the permeability barrier function of the urothelium and the orchestrated lipid phase formation within the plasma membrane.

[end paper 76]

[begin paper 77]

PMCID = 8842845

Title = Label-free Visual Proteomics: Coupling MS- and EM-based approaches in structural biology

Combining diverse experimental structural and interactomic methods allows for the construction of comprehensible molecular encyclopedias of biological systems. Typically, this involves merging several independent approaches that provide complementary structural and functional information from multiple perspectives and at different resolution ranges. A particularly potent combination lies in coupling structural information from Cryogenic Electron Microscopy or Tomography (cryo-EM or cryo-ET) with interactomic and structural information from Mass Spectrometry (MS)-based Structural Proteomics. Cryo-EM/ET allows for sub-nanometer visualization of biological specimens in purified and near-native states, while MS provides bioanalytical information for proteins and protein complexes without introducing additional labels. Here we highlight recent achievements in protein structure and interactome determination using cryo-EM/ET which benefit from additional MS analysis. We also give our perspective on how combining cryo-EM/ET and MS will continue bridging gaps between molecular and cellular studies through capturing and describing 3D-snapshots of proteomes and interactomes.

[end paper 77]

[begin paper 78]

PMCID = 11074592

Title = Predicting TCR sequences for unseen antigen epitopes using structural and sequence features

T-cell receptor (TCR) recognition of antigens is fundamental to the adaptive immune response. With the expansion of experimental techniques, a substantial database of matched TCR–antigen pairs has emerged, presenting opportunities for computational prediction models. However, accurately forecasting the binding affinities of unseen antigen–TCR pairs remains a major challenge. Here, we present convolutional-self-attention TCR (CATCR), a novel framework tailored to enhance the prediction of epitope and TCR interactions. Our approach utilizes convolutional neural networks to extract peptide features from residue contact matrices, as generated by OpenFold, and a transformer to encode segment-based coded sequences. We introduce CATCR-D, a discriminator that can assess binding by analyzing the structural and sequence features of epitopes and CDR3-β regions. Additionally, the framework comprises CATCR-G, a generative module designed for CDR3-β sequences, which applies the pretrained encoder to deduce epitope characteristics and a transformer decoder for predicting matching CDR3-β sequences. CATCR-D achieved an AUROC of 0.89 on previously unseen epitope–TCR pairs and outperformed four benchmark models by a margin of 17.4%. CATCR-G has demonstrated high precision, recall and F1 scores, surpassing 95% in bidirectional encoder representations from transformers score assessments. Our results indicate that CATCR is an effective tool for predicting unseen epitope–TCR interactions. Incorporating structural insights enhances our understanding of the general rules governing TCR–epitope recognition significantly. The ability to predict TCRs for novel epitopes using structural and sequence information is promising, and broadening the repository of experimental TCR–epitope data could further improve the precision of epitope–TCR binding predictions.

[end paper 78]

[begin paper 79]

PMCID = 11500303

Title = Generating Protein Structures for Pathway Discovery Using Deep Learning

Resolving the intricate details of biological phenomena at the molecular level is fundamentally limited by both length- and time scales that can be probed experimentally. Molecular dynamics (MD) simulations at various scales are powerful tools frequently employed to offer valuable biological insights beyond experimental resolution. However, while it is relatively simple to observe long-lived, stable configurations of, for example, proteins, at the required spatial resolution, simulating the more interesting rare transitions between such states often takes orders of magnitude longer than what is feasible even on the largest supercomputers available today. One common aspect of this challenge is pathway discovery, where the start and end states of a scientific phenomenon are known or can be approximated, but the mechanistic details in between are unknown. Here, we propose a representation-learning-based solution that uses interpolation and extrapolation in an abstract representation space to synthesize potential transition states, which are automatically validated using MD simulations. The new simulations of the synthesized transition states are subsequently incorporated into the representation learning, leading to an iterative framework for targeted path sampling. Our approach is demonstrated by recovering the transition of a RAS-RAF protein domain (CRD) from membrane-free to interacting with the membrane using coarse-grain MD simulations.

[end paper 79]

[begin paper 80]

PMCID = 11292590

Title = Review of AlphaFold 3: Transformative Advances in Drug Design and Therapeutics

Google DeepMind Technologies Limited (London, United Kingdom) recently released its new version of the biomolecular structure predictor artificial intelligence (AI) model named AlphaFold 3. Superior in accuracy and more powerful than its predecessor AlphaFold 2, this innovation has astonished the world with its capacity and speed. It takes humans years to determine the structure of various proteins and how the shape works with the receptors but AlphaFold 3 predicts the same structure in seconds. The version's utility is unimaginable in the field of drug discoveries, vaccines, enzymatic processes, and determining the rate and effect of different biological processes. AlphaFold 3 uses similar machine learning and deep learning models such as Gemini (Google DeepMind Technologies Limited). AlphaFold 3 has already established itself as a turning point in the field of computational biochemistry and drug development along with receptor modulation and biomolecular development. With the help of AlphaFold 3 and models similar to this, researchers will gain unparalleled insights into the structural dynamics of proteins and their interactions, opening up new avenues for scientists and doctors to exploit for the benefit of the patient. The integration of AI models like AlphaFold 3, bolstered by rigorous validation against high-standard research publications, is set to catalyze further innovations and offer a glimpse into the future of biomedicine.

[end paper 80]

[begin paper 81]

PMCID = 11381518

Title = Human cell surface-AAV interactomes identify LRP6 as blood-brain barrier transcytosis receptor and immune cytokine IL3 as AAV9 binder

Adeno-associated viruses (AAVs) are foundational gene delivery tools for basic science and clinical therapeutics. However, lack of mechanistic insight, especially for engineered vectors created by directed evolution, can hamper their application. Here, we adapt an unbiased human cell microarray platform to determine the extracellular and cell surface interactomes of natural and engineered AAVs. We identify a naturally-evolved and serotype-specific interaction between the AAV9 capsid and human interleukin 3 (IL3), with possible roles in host immune modulation, as well as lab-evolved low-density lipoprotein receptor-related protein 6 (LRP6) interactions specific to engineered capsids with enhanced blood-brain barrier crossing in non-human primates after intravenous administration. The unbiased cell microarray screening approach also allows us to identify off-target tissue binding interactions of engineered brain-enriched AAV capsids that may inform vectors’ peripheral organ tropism and side effects. Our cryo-electron tomography and AlphaFold modeling of capsid-interactor complexes reveal LRP6 and IL3 binding sites. These results allow confident application of engineered AAVs in diverse organisms and unlock future target-informed engineering of improved viral and non-viral vectors for non-invasive therapeutic delivery to the brain.

Engineered adeno-associated viruses (AAVs) aim to improve safety and potency for use in gene therapy but mechanisms underlying these features are poorly understood. Here, authors use unbiased screens to identify an interaction with the human immune system and a determinant of enhanced brain potency.

[end paper 81]

[begin paper 82]

PMCID = 11099071

Title = Amino acid insertion in Bat MHC-I enhances complex stability and augments peptide presentation

Bats serve as reservoirs for numerous zoonotic viruses, yet they typically remain asymptomatic owing to their unique immune system. Of particular significance is the MHC-I in bats, which plays crucial role in anti-viral response and exhibits polymorphic amino acid (AA) insertions. This study demonstrated that both 5AA and 3AA insertions enhance the thermal stability of the bat MHC-I complex and enrich the diversity of bound peptides in terms of quantity and length distribution, by stabilizing the 310 helix, a region prone to conformational changes during peptide loading. However, the mismatched insertion could diminish the stability of bat pMHC-I. We proposed that a suitable insertion may help bat MHC-I adapt to high body temperatures during flight while enhancing antiviral responses. Moreover, this site-specific insertions may represent a strategy of evolutionary adaptation of MHC-I molecules to fluctuations in body temperature, as similar insertions have been found in other lower vertebrates.

Bat MHC-I’s AA insertions boost thermal stability and peptide diversity, potentially adapting to flight-induced high temperatures and improving antiviral defenses, suggesting an evolutionary strategy to body temperature changes.

[end paper 82]

[begin paper 83]

PMCID = 9178029

Title = The structure of EXTL3 helps to explain the different roles of bi-domain exostosins in heparan sulfate synthesis

Heparan sulfate is a highly modified O-linked glycan that performs diverse physiological roles in animal tissues. Though quickly modified, it is initially synthesised as a polysaccharide of alternating β-d-glucuronosyl and N-acetyl-α-d-glucosaminyl residues by exostosins. These enzymes generally possess two glycosyltransferase domains (GT47 and GT64)—each thought to add one type of monosaccharide unit to the backbone. Although previous structures of murine exostosin-like 2 (EXTL2) provide insight into the GT64 domain, the rest of the bi-domain architecture is yet to be characterised; hence, how the two domains co-operate is unknown. Here, we report the structure of human exostosin-like 3 (EXTL3) in apo and UDP-bound forms. We explain the ineffectiveness of EXTL3’s GT47 domain to transfer β-d-glucuronosyl units, and we observe that, in general, the bi-domain architecture would preclude a processive mechanism of backbone extension. We therefore propose that heparan sulfate backbone polymerisation occurs by a simple dissociative mechanism.

Heparan sulphate (HS)—a common cell surface decoration—is a carbohydrate of alternating sugars assembled by bi-domain enzymes such as EXTL3. Here, authors present the structure of EXTL3, explain EXTL3 lost activity, and propose that HS extension is distributive.

[end paper 83]

[begin paper 84]

PMCID = 11604545

Title = Intra-channel bi-epitopic crosslinking unleashes ultrapotent antibodies targeting NaV1.7 for pain alleviation

Crucial for cell activities, ion channels are key drug discovery targets. Although small-molecule and peptide modulators dominate ion channel drug discovery, antibodies are emerging as an alternative modality. However, challenges persist in generating potent antibodies, especially for channels with limited extracellular epitopes. We herein present a bi-epitopic crosslinking strategy to overcome these challenges, focusing on NaV1.7, a potential analgesic target. Aiming to crosslink two non-overlapping epitopes on voltage-sensing domains II and IV, we construct bispecific antibodies and ligand-antibody conjugates. Enhanced affinity and potency are observed in comparison to the monospecific controls. Among them, a ligand-antibody conjugate (1080-PEG7-ACDTB) displays a two-orders-of-magnitude improvement in potency (IC50 of 0.06 ± 0.01 nM) and over 1,000-fold selectivity for NaV1.7. Additionally, this conjugate demonstrates robust analgesic effects in mouse pain models. Our study introduces an approach to developing effective antibodies against NaV1.7, thereby initiating a promising direction for the advancement of pain therapeutics.

Bi-epitopic crosslinking enhances the potency of antibodies targeting ion channels

A ligand-antibody conjugate shows ultrapotent and highly selective NaV1.7 inhibition

The ligand-antibody conjugate exhibits potent analgesic effects in mouse pain models

Zhang et al. introduce a bi-epitopic crosslinking strategy to enhance antibody potency against ion channels, particularly for NaV1.7, a pain relief target. This approach, using bispecific antibodies and ligand-antibody conjugates, significantly improves affinity and potency, with one conjugate showing a 1,000-fold selectivity for NaV1.7 and strong analgesic effects in mice, offering a promising direction for pain therapeutics.

[end paper 84]

[begin paper 85]

PMCID = 10606070

Title = Novel Genetic and Phenotypic Expansion in GOSR2-Related Progressive Myoclonus Epilepsy

Biallelic variants in the Golgi SNAP receptor complex member 2 gene (GOSR2) have been reported in progressive myoclonus epilepsy with neurodegeneration. Typical clinical features include ataxia and areflexia during early childhood, followed by seizures, scoliosis, dysarthria, and myoclonus. Here, we report two novel patients from unrelated families with a GOSR2-related disorder and novel genetic and clinical findings. The first patient, a male compound heterozygous for the GOSR2 splice site variant c.336+1G>A and the novel c.364G>A,p.Glu122Lys missense variant showed global developmental delay and seizures at the age of 2 years, followed by myoclonus at the age of 8 years with partial response to clonazepam. The second patient, a female homozygous for the GOSR2 founder variant p.Gly144Trp, showed only mild fine motor developmental delay and generalized tonic–clonic seizures triggered by infections during adolescence, with seizure remission on levetiracetam. The associated movement disorder progressed atypically slowly during adolescence compared to its usual speed, from initial intention tremor and myoclonus to ataxia, hyporeflexia, dysmetria, and dystonia. These findings expand the genotype–phenotype spectrum of GOSR2-related disorders and suggest that GOSR2 should be included in the consideration of monogenetic causes of dystonia, global developmental delay, and seizures.

[end paper 85]

[begin paper 86]

PMCID = 10862866

Title = The co-receptor Tetraspanin12 directly captures Norrin to promote ligand-specific β-catenin signaling

Wnt/β-catenin signaling directs animal development and tissue renewal in a tightly controlled, cell- and tissue-specific manner. In the mammalian central nervous system, the atypical ligand Norrin controls angiogenesis and maintenance of the blood-brain barrier and blood-retina barrier through the Wnt/β-catenin pathway. Like Wnt, Norrin activates signaling by binding and heterodimerizing the receptors Frizzled (Fzd) and Low-density lipoprotein receptor-related protein 5 or 6 (LRP5/6), leading to membrane recruitment of the intracellular transducer Dishevelled (Dvl) and ultimately stabilizing the transcriptional coactivator β-catenin. Unlike Wnt, the cystine-knot ligand Norrin only signals through Fzd4 and additionally requires the co-receptor Tetraspanin12 (Tspan12); however, the mechanism underlying Tspan12-mediated signal enhancement is unclear. It has been proposed that Tspan12 integrates into the Norrin-Fzd4 complex to enhance Norrin-Fzd4 affinity or otherwise allosterically modulate Fzd4 signaling. Here, we measure direct, high-affinity binding between purified Norrin and Tspan12 in a lipid environment and use AlphaFold models to interrogate this interaction interface. We find that Tspan12 and Fzd4 can simultaneously bind Norrin and that a pre-formed Tspan12/Fzd4 heterodimer, as well as cells co-expressing Tspan12 and Fzd4, more efficiently capture low concentrations of Norrin than Fzd4 alone. We also show that Tspan12 competes with both heparan sulfate proteoglycans and LRP6 for Norrin binding and that Tspan12 does not impact Fzd4-Dvl affinity in the presence or absence of Norrin. Our findings suggest that Tspan12 does not allosterically enhance Fzd4 binding to Norrin or Dvl, but instead functions to directly capture Norrin upstream of signaling.

[end paper 86]

[begin paper 87]

PMCID = 10560298

Title = Cryo-EM elucidates the uroplakin complex structure within liquid-crystalline lipids in the porcine urothelial membrane

The urothelium, a distinct epithelial tissue lining the urinary tract, serves as an essential component in preserving urinary tract integrity and thwarting infections. The asymmetric unit membrane (AUM), primarily composed of the uroplakin complex, constitutes a critical permeability barrier in fulfilling this role. However, the molecular architectures of both the AUM and the uroplakin complex have remained enigmatic due to the paucity of high-resolution structural data. In this study, we utilized cryo-electron microscopy to elucidate the three-dimensional structure of the uroplakin complex within the porcine AUM. While the global resolution achieved was 3.5 Å, we acknowledge that due to orientation bias, the resolution in the vertical direction was determined to be 6.3 Å. Our findings unveiled that the uroplakin complexes are situated within hexagonally arranged crystalline lipid membrane domains, rich in hexosylceramides. Moreover, our research rectifies a misconception in a previous model by confirming the existence of a domain initially believed to be absent, and pinpointing the accurate location of a crucial Escherichia coli binding site implicated in urinary tract infections. These discoveries offer valuable insights into the molecular underpinnings governing the permeability barrier function of the urothelium and the orchestrated lipid phase formation within the plasma membrane.

Cryo-EM reveals the molecular architecture of the uroplakin complex within the asymmetric unit membrane (AUM) of the urothelium, helping to correctly identify the crucial E.coli binding site involved in urinary tract infections.

[end paper 87]

[begin paper 88]

PMCID = 9272110

Title = Editorial: Influence of Protein-Protein Interactions (PPIs) on the Outcome of Viral Infections



[end paper 88]

[begin paper 89]

PMCID = 10369898

Title = The Impact of AI-Based Modeling on the Accuracy of Protein Assembly Prediction: Insights from CASP15

In CASP15, 87 predictors submitted around 11,000 models on 41 assembly targets. The community demonstrated exceptional performance in overall fold and interface contact prediction, achieving an impressive success rate of 90% (compared to 31% in CASP14). This remarkable accomplishment is largely due to the incorporation of DeepMind’s AF2-Multimer approach into custom-built prediction pipelines. To evaluate the added value of participating methods, we compared the community models to the baseline AF2-Multimer predictor. In over 1/3 of cases the community models were superior to the baseline predictor. The main reasons for this improved performance were the use of custom-built multiple sequence alignments, optimized AF2-Multimer sampling, and the manual assembly of AF2-Multimer-built subcomplexes. The best three groups, in order, are Zheng, Venclovas and Wallner. Zheng and Venclovas reached a 73.2% success rate over all (41) cases, while Wallner attained 69.4% success rate over 36 cases. Nonetheless, challenges remain in predicting structures with weak evolutionary signals, such as nanobody-antigen, antibody-antigen, and viral complexes. Expectedly, modeling large complexes remains also challenging due to their high memory compute demands.

In addition to the assembly category, we assessed the accuracy of modeling interdomain interfaces in the tertiary structure prediction targets. Models on seven targets featuring 17 unique interfaces were analyzed. Best predictors achieved the 76.5% success rate, with the UM-TBM group being the leader. In the interdomain category, we observed that the predictors faced challenges, as in the case of the assembly category, when the evolutionary signal for a given domain pair was weak or the structure was large. Overall, CASP15 witnessed unprecedented improvement in interface modeling, reflecting the AI revolution seen in CASP14.

[end paper 89]

[begin paper 90]

PMCID = 11230163

Title = ProBASS – a language model with sequence and structural features for predicting the effect of mutations on binding affinity

Protein-protein interactions (PPIs) govern virtually all cellular processes. Even a single mutation within PPI can significantly influence overall protein functionality and potentially lead to various types of diseases. To date, numerous approaches have emerged for predicting the change in free energy of binding (ΔΔGbind) resulting from mutations, yet the majority of these methods lack precision. In recent years, protein language models (PLMs) have been developed and shown powerful predictive capabilities by leveraging both sequence and structural data from protein-protein complexes. Yet, PLMs have not been optimized specifically for predicting ΔΔGbind. We developed an approach to predict effects of mutations on PPI binding affinity based on two most advanced protein language models ESM2 and ESM-IF1 that incorporate PPI sequence and structural features, respectively. We used the two models to generate embeddings for each PPI mutant and subsequently fine-tuned our model by training on a large dataset of experimental ΔΔGbind values. Our model, ProBASS (Protein Binding Affinity from Structure and Sequence) achieved a correlation with experimental ΔΔGbind values of 0.83 ± 0.05 for single mutations and 0.69 ± 0.04 for double mutations when model training and testing was done on the same PDB. Moreover, ProBASS exhibited very high correlation (0.81 ± 0.02) between prediction and experiment when training and testing was performed on a dataset containing 2325 single mutations in 132 PPIs. ProBASS surpasses the state-of-the-art methods in correlation with experimental data and could be further trained as more experimental data becomes available. Our results demonstrate that the integration of extensive datasets containing ΔΔGbind values across multiple PPIs to refine the pre-trained PLMs represents a successful approach for achieving a precise and broadly applicable model for ΔΔGbind prediction, greatly facilitating future protein engineering and design studies.

[end paper 90]

[begin paper 91]

PMCID = 10268964

Title = All-Atom Simulations Reveal the Intricacies of Signal Transduction upon Binding of the HLA-E Ligand to the Transmembrane Inhibitory CD94/NKG2A Receptor

Natural killer (NK) cells play an important role in the innate immune response against tumors and various pathogens such as viruses and bacteria. Their function is controlled by a wide array of activating and inhibitory receptors, which are expressed on their cell surface. Among them is a dimeric NKG2A/CD94 inhibitory transmembrane (TM) receptor which specifically binds to the non-classical MHC I molecule HLA-E, which is often overexpressed on the surface of senescent and tumor cells. Using the Alphafold 2 artificial intelligence system, we constructed the missing segments of the NKG2A/CD94 receptor and generated its complete 3D structure comprising extracellular (EC), TM, and intracellular regions, which served as a starting point for the multi-microsecond all-atom molecular dynamics simulations of the receptor with and without the bound HLA-E ligand and its nonameric peptide. The simulated models revealed that an intricate interplay of events is taking place between the EC and TM regions ultimately affecting the intracellular immunoreceptor tyrosine-based inhibition motif (ITIM) regions that host the point at which the signal is transmitted further down the inhibitory signaling cascade. Signal transduction through the lipid bilayer was also coupled with the changes in the relative orientation of the NKG2A/CD94 TM helices in response to linker reorganization, mediated by fine-tuned interactions in the EC region of the receptor, taking place after HLA-E binding. This research provides atomistic details of the cells’ protection mechanism against NK cells and broadens the knowledge regarding the TM signaling of ITIM-bearing receptors.

[end paper 91]

[begin paper 92]

PMCID = 11375674

Title = PD-1 Targeted Antibody Discovery Using AI Protein Diffusion

The programmed cell death protein 1 (PD-1, CD279) is an important therapeutic target in many oncological diseases. This checkpoint protein inhibits T lymphocytes from attacking other cells in the body and thus blocking it improves the clearance of tumor cells by the immune system. While there are already multiple FDA-approved anti-PD-1 antibodies, including nivolumab (Opdivo® from Bristol-Myers Squibb) and pembrolizumab (Keytruda® from Merck), there are ongoing efforts to discover new and improved checkpoint inhibitor therapeutics. In this study, we present multiple anti-PD-1 antibody fragments that were derived computationally using protein diffusion and evaluated through our scalable, in silico pipeline. Here we present nine synthetic Fv structures that are suitable for further empirical testing of their anti-PD-1 activity due to desirable predicted binding performance.

[end paper 92]

[begin paper 93]

PMCID = 11361660

Title = A charged tail on anti-α-Synuclein antibodies does not enhance their affinity to α-Synuclein fibrils

The aggregation of α-Synuclein (αSyn) is strongly linked to neuronal death in Parkinson’s disease and other synucleinopathies. The spreading of aggregated αSyn between neurons is at least partly dependent on electrostatic interactions between positively charged stretches on αSyn fibrils and the negatively charged heparan sulphate proteoglycans on the cell surface. To date there is still no therapeutic option available that could halt the progression of Parkinson’s disease and one of the major limitations is likely the relatively low proportion of αSyn aggregates accessible to drugs in the extracellular space. Here, we investigated whether a negatively charged peptide tail fused to the αSyn aggregate-specific antibodies SynO2 and 9E4 could enhance the antibodies’ avidity to αSyn aggregates in order to improve their potential therapeutic effect through inhibiting cell-to-cell spreading and enhancing the clearance of extracellular aggregates. We performed ELISAs to test the avidity to αSyn aggregates of both monovalent and bivalent antibody formats with and without the peptide tail. Our results show that the addition of the negatively charged peptide tail decreased the binding strength of both antibodies to αSyn aggregates at physiological salt conditions, which can likely be explained by intermolecular repulsions between the tail and the negatively charged C-terminus of αSyn. Additionally, the tail might interact with the paratopes of the SynO2 antibody abolishing its binding to αSyn aggregates. Conclusively, our peptide tail did not fulfil the required characteristics to improve the antibodies’ binding to αSyn aggregates. Fine-tuning the design of the peptide tail to avoid its interaction with the antibodies’ CDR and to better mimic relevant characteristics of heparan sulphates for αSyn aggregate binding may help overcome the limitations observed in this study.

[end paper 93]

[begin paper 94]

PMCID = 10871235

Title = Preservation of memory B cell homeostasis in an individual producing broadly neutralising antibodies against HIV-1

Immunological determinants favouring emergence of broadly neutralising antibodies are crucial to the development of HIV-1 vaccination strategies. Here, we combined RNAseq and B cell cloning approaches to isolate a broadly neutralising antibody (bnAb) ELC07 from an individual living with untreated HIV-1. Using single particle cryogenic electron microscopy (cryo-EM), we show that the antibody recognises a conformational epitope at the gp120-gp41 interface. ELC07 binds the closed state of the viral glycoprotein causing considerable perturbations to the gp41 trimer core structure. Phenotypic analysis of memory B cell subsets from the ELC07 bnAb donor revealed a lack of expected HIV-1-associated dysfunction, specifically no increase in CD21−/CD27− cells was observed whilst the resting memory (CD21+/CD27+) population appeared preserved despite uncontrolled HIV-1 viraemia. Moreover, single cell transcriptomes of memory B cells from this bnAb donor showed a resting memory phenotype irrespective of the epitope they targeted or their ability to neutralise diverse strains of HIV-1. Strikingly, single memory B cells from the ELC07 bnAb donor were transcriptionally similar to memory B cells from HIV-negative individuals. Our results demonstrate that potent bnAbs can arise without the HIV-1-induced dysregulation of the memory B cell compartment and suggest that sufficient levels of antigenic stimulation with a strategically designed immunogen could be effective in HIV-negative vaccine recipients.

[end paper 94]

[begin paper 95]

PMCID = 11786912

Title = Plasticity of BioPhi-driven humanness optimization in ScFv-CD99 binding affinity validated through AlphaFold, HADDOCK, and MD simulations

BioPhi-guided humanization was utilized to enhance the humanness of a humanized single-chain variable fragment targeting CD99, leading to the development of two variants: HuScFvMT99/3BP and HuScFvMT99/3HY. The HuScFvMT99/3BP variant incorporated framework region modifications, leading to modest improvements in humanness, particularly in the VH domain, although the VL domain remained suboptimal. To address this limitation, HuScFvMT99/3HY was designed by combining the VL domain of wild-type with the VH domain of HuScFvMT99/3BP. Molecular dynamics simulations employing AlphaFold2, AlphaFold3, and HADDOCK were performed to evaluate the HuScFv-CD99 peptide complexes. AF2-based simulations demonstrated enhanced binding free energy (ΔGbinding) for both variants compared to HuScFvMT99/3WT. However, ΔGbinding values obtained from AF3 and HD simulations were inconsistent, with HuScFvMT99/3BP exhibiting the weakest binding affinity. While ΔGbinding patterns derived from AlphaFold3 and HADDOCK simulations aligned, amino acid decomposition analysis revealed variations in the interaction coordinates of the predicted complexes. Root-mean-square deviation analysis indicated improved structural stability for HuScFvMT99/3BP (0.975 Å) and HuScFvMT99/3HY (1.075 Å) relative to HuScFvMT99/3WT (1.225 Å). Biolayer interferometry further confirmed that HuScFvMT99/3WT exhibited the highest binding affinity (KD = 1.35 × 10⁻⁷ M) compared to HuScFvMT99/3BP (KD = 2.64 × 10⁻⁷ M) and HuScFvMT99/3HY (KD = 3.95 × 10⁻⁷ M). Supporting evidence was provided by ELISA and flow cytometry experiments. PITHA analysis revealed a high immunogenicity risk for all variants, despite HuScFvMT99/3HY displaying improved humanness, a larger complementarity-determining region (CDR) cavity, and a more hydrophobic CDR-H3 loop. These findings highlight the delicate balance between enhancing humanness and preserving the structural and functional integrity critical for therapeutic antibody development.

[end paper 95]

[begin paper 96]

PMCID = 10709145

Title = Chiral and nematic phases of flexible active filaments

The emergence of large-scale order in self-organized systems relies on local interactions between individual components. During bacterial cell division, FtsZ—a prokaryotic homologue of the eukaryotic protein tubulin—polymerizes into treadmilling filaments that further organize into a cytoskeletal ring. In vitro, FtsZ filaments can form dynamic chiral assemblies. However, how the active and passive properties of individual filaments relate to these large-scale self-organized structures remains poorly understood. Here we connect single-filament properties with the mesoscopic scale by combining minimal active matter simulations and biochemical reconstitution experiments. We show that the density and flexibility of active chiral filaments define their global order. At intermediate densities, curved, flexible filaments organize into chiral rings and polar bands. An effectively nematic organization dominates for high densities and for straight, mutant filaments with increased rigidity. Our predicted phase diagram quantitatively captures these features, demonstrating how the flexibility, density and chirality of the active filaments affect their collective behaviour. Our findings shed light on the fundamental properties of active chiral matter and explain how treadmilling FtsZ filaments organize during bacterial cell division.

Filaments of the FtsZ protein can form chiral assemblies. Now, active matter tools link the microscopic structure of active filaments to the large-scale collective phase of these assemblies.

[end paper 96]



