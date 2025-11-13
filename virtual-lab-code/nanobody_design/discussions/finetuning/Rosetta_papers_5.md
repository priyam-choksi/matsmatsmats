## User

This is the beginning of an individual meeting with Computational Biologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "Rosetta" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on Rosetta to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on Rosetta using the query "Rosetta computational antibody design SARS-CoV-2" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to Rosetta. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Computational Biologist, please provide your response to the agenda.

## Computational Biologist

Based on the abstracts provided, I will select papers that are directly relevant to the use of Rosetta in the context of designing SARS-CoV-2 nanobody binders. Here is the Python dictionary mapping PMCID to the title of the papers that I wish to fine-tune myself on:

```python
{
    "8765073": "Computational design of a neutralizing antibody with picomolar binding affinity for all concerning SARS-CoV-2 variants",
    "8590000": "Computational redesign of Fab CC12.3 with substantially better predicted binding affinity to SARS-CoV-2 than human ACE2 receptor",
    "9628197": "Computationally restoring the potency of a clinical antibody against SARS-CoV-2 Omicron subvariants",
    "10069166": "Computational design of nanomolar-binding antibodies specific to multiple SARS-CoV-2 variants by engineering a specificity switch of antibody 80R using RosettaAntibodyDesign (RAbD) results in potential generalizable therapeutic antibodies for novel SARS-CoV-2 virus",
    "9936628": "Computational pipeline provides mechanistic understanding of Omicron variant of concern neutralizing engineered ACE2 receptor traps",
    "9683856": "Structural evolution of Delta lineage of SARS-CoV-2",
    "10509195": "Computational redesign of Beta-27 Fab with substantially better predicted binding affinity to the SARS-CoV-2 Omicron variant than human ACE2 receptor"
}
```

These papers were chosen because they specifically mention the use of Rosetta or computational design strategies that align with the goals of designing SARS-CoV-2 nanobody binders.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "Rosetta computational antibody design SARS-CoV-2":

[begin paper 1]

PMCID = 8765073

Title = Computational design of a neutralizing antibody with picomolar binding affinity for all concerning SARS-CoV-2 variants

Coronavirus disease 2019, caused by SARS-CoV-2, remains an on-going pandemic, partly due to the emergence of variant viruses that can “break-through” the protection of the current vaccines and neutralizing antibodies (nAbs), highlighting the needs for broadly nAbs and next-generation vaccines. We report an antibody that exhibits breadth and potency in binding the receptor-binding domain (RBD) of the virus spike glycoprotein across SARS coronaviruses. Initially, a lead antibody was computationally discovered and crystallographically validated that binds to a highly conserved surface of the RBD of wild-type SARS-CoV-2. Subsequently, through experimental affinity enhancement and computational affinity maturation, it was further developed to bind the RBD of all concerning SARS-CoV-2 variants, SARS-CoV-1 and pangolin coronavirus with pico-molar binding affinities, consistently exhibited strong neutralization activity against wild-type SARS-CoV-2 and the Alpha and Delta variants. These results identify a vulnerable target site on coronaviruses for development of pan-sarbecovirus nAbs and vaccines.

[end paper 1]

[begin paper 2]

PMCID = 8796240

Title = Spike-Dependent Opsonization Indicates Both Dose-Dependent Inhibition of Phagocytosis and That Non-Neutralizing Antibodies Can Confer Protection to SARS-CoV-2

Spike-specific antibodies are central to effective COVID19 immunity. Research efforts have focused on antibodies that neutralize the ACE2-Spike interaction but not on non-neutralizing antibodies. Antibody-dependent phagocytosis is an immune mechanism enhanced by opsonization, where typically, more bound antibodies trigger a stronger phagocyte response. Here, we show that Spike-specific antibodies, dependent on concentration, can either enhance or reduce Spike-bead phagocytosis by monocytes independently of the antibody neutralization potential. Surprisingly, we find that both convalescent patient plasma and patient-derived monoclonal antibodies lead to maximum opsonization already at low levels of bound antibodies and is reduced as antibody binding to Spike protein increases. Moreover, we show that this Spike-dependent modulation of opsonization correlate with the outcome in an experimental SARS-CoV-2 infection model. These results suggest that the levels of anti-Spike antibodies could influence monocyte-mediated immune functions and propose that non-neutralizing antibodies could confer protection to SARS-CoV-2 infection by mediating phagocytosis.

[end paper 2]

[begin paper 3]

PMCID = 7334636

Title = Protein structure analysis of the interactions between SARS-CoV-2 spike protein and the human ACE2 receptor: from conformational changes to novel neutralizing antibodies

The recent severe acute respiratory syndrome, known as Coronavirus Disease 2019 (COVID-19) has spread so much rapidly and severely to induce World Health Organization (WHO) to declare a state of emergency over the new coronavirus SARS-CoV-2 pandemic. While several countries have chosen the almost complete lock-down for slowing down SARS-CoV-2 spread, the scientific community is called to respond to the devastating outbreak by identifying new tools for diagnosis and treatment of the dangerous COVID-19. With this aim, we performed an in silico comparative modeling analysis, which allows gaining new insights into the main conformational changes occurring in the SARS-CoV-2 spike protein, at the level of the receptor-binding domain (RBD), along interactions with human cells angiotensin-converting enzyme 2 (ACE2) receptor, that favor human cell invasion. Furthermore, our analysis provides (1) an ideal pipeline to identify already characterized antibodies that might target SARS-CoV-2 spike RBD, aiming to prevent interactions with the human ACE2, and (2) instructions for building new possible neutralizing antibodies, according to chemical/physical space restraints and complementary determining regions (CDR) mutagenesis of the identified existing antibodies. The proposed antibodies show in silico high affinity for SARS-CoV-2 spike RBD and can be used as reference antibodies also for building new high-affinity antibodies against present and future coronaviruses able to invade human cells through interactions of their spike proteins with the human ACE2. More in general, our analysis provides indications for the set-up of the right biological molecular context for investigating spike RBD–ACE2 interactions for the development of new vaccines, diagnostic kits, and other treatments based on the targeting of SARS-CoV-2 spike protein.

The online version of this article (10.1007/s00018-020-03580-1) contains supplementary material, which is available to authorized users.

[end paper 3]

[begin paper 4]

PMCID = 8590000

Title = Computational redesign of Fab CC12.3 with substantially better predicted binding affinity to SARS-CoV-2 than human ACE2 receptor

SARS-CoV-2 is responsible for COVID-19 pandemic, causing large numbers of cases and deaths. It initiates entry into human cells by binding to the peptidase domain of angiotensin-converting enzyme 2 (ACE2) receptor via its receptor binding domain of S1 subunit of spike protein (SARS-CoV-2-RBD). Employing neutralizing antibodies to prevent binding between SARS-CoV-2-RBD and ACE2 is an effective COVID-19 therapeutic solution. Previous studies found that CC12.3 is a highly potent neutralizing antibody that was isolated from a SARS-CoV-2 infected patient, and its Fab fragment (Fab CC12.3) bound to SARS-CoV-2-RBD with comparable binding affinity to ACE2. To enhance its binding affinity, we employed computational protein design to redesign all CDRs of Fab CC12.3 and molecular dynamics (MD) to validate their predicted binding affinities by the MM-GBSA method. MD results show that the predicted binding affinities of the three best designed Fabs CC12.3 (CC12.3-D02, CC12.3-D05, and CC12.3-D08) are better than those of Fab CC12.3 and ACE2. Additionally, our results suggest that enhanced binding affinities of CC12.3-D02, CC12.3-D05, and CC12.3-D08 are caused by increased SARS-CoV-2-RBD binding interactions of CDRs L1 and L3. This study redesigned neutralizing antibodies with better predicted binding affinities to SARS-CoV-2-RBD than Fab CC12.3 and ACE2. They are promising candidates as neutralizing antibodies against SARS-CoV-2.

[end paper 4]

[begin paper 5]

PMCID = 9341430

Title = Broad sarbecovirus neutralization by a human monoclonal antibody

The recent emergence of SARS-CoV-2 variants of concern (VOC) and the recurrent spillovers of coronaviruses in the human population highlight the need for broadly neutralizing antibodies that are not affected by the ongoing antigenic drift and that can prevent or treat future zoonotic infections. Here, we describe a human monoclonal antibody (mAb), designated S2X259, recognizing a highly conserved cryptic receptor-binding domain (RBD) epitope and cross-reacting with spikes from all sarbecovirus clades. S2X259 broadly neutralizes spike-mediated entry of SARS-CoV-2 including the B.1.1.7, B.1.351, P.1, B.1.427/B.1.429 VOC, as well as a wide spectrum of human and potentially zoonotic sarbecoviruses through inhibition of ACE2 binding to the RBD. Furthermore, deep-mutational scanning and in vitro escape selection experiments demonstrate that S2X259 possesses an escape profile limited to the single substitution G504D. We show that prophylactic and therapeutic administration of S2X259 protects Syrian hamsters against challenge with the prototypic SARS-CoV-2 and the B.1.351 VOC, suggesting this mAb is a promising candidate for the prevention and treatment of emergent variants and zoonotic infections. Our data unveil a key antigenic site targeted by broadly-neutralizing antibodies and will guide the design of pan-sarbecovirus vaccines.

[end paper 5]

[begin paper 6]

PMCID = 8747942

Title = Nucleic acid delivery of immune-focused SARS-CoV-2 nanoparticles drives rapid and potent immunogenicity capable of single-dose protection

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) vaccines may target epitopes that reduce durability or increase the potential for escape from vaccine-induced immunity. Using synthetic vaccinology, we have developed rationally immune-focused SARS-CoV-2 Spike-based vaccines. Glycans can be employed to alter antibody responses to infection and vaccines. Utilizing computational modeling and in vitro screening, we have incorporated glycans into the receptor-binding domain (RBD) and assessed antigenic profiles. We demonstrate that glycan-coated RBD immunogens elicit stronger neutralizing antibodies and have engineered seven multivalent configurations. Advanced DNA delivery of engineered nanoparticle vaccines rapidly elicits potent neutralizing antibodies in guinea pigs, hamsters, and multiple mouse models, including human ACE2 and human antibody repertoire transgenics. RBD nanoparticles induce high levels of cross-neutralizing antibodies against variants of concern with durable titers beyond 6 months. Single, low-dose immunization protects against a lethal SARS-CoV-2 challenge. Single-dose coronavirus vaccines via DNA-launched nanoparticles provide a platform for rapid clinical translation of potent and durable coronavirus vaccines.

Glycan engineering of immunogens is a key vaccine design strategy to focus antibodies to select epitopes. Konrath et al. create an automated pipeline to screen SARS-CoV-2 RBD for PNGS. PNGS-modified RBDs elicit enhanced ACE2-directed responses and higher neutralization. Multivalent display of glycan-engineered RBDs delivered via nucleic acid provides single-dose protection from challenge.

[end paper 6]

[begin paper 7]

PMCID = 8931377

Title = Deep learning guided optimization of human antibody against SARS-CoV-2 variants with broad neutralization

SARS-CoV-2 continues to evolve through emerging variants, more frequently observed with higher transmissibility. Despite the wide application of vaccines and antibodies, the selection pressure on the Spike protein may lead to further evolution of variants that include mutations that can evade immune response. To catch up with the virus’s evolution, we introduced a deep learning approach to redesign the complementarity-determining regions (CDRs) to target multiple virus variants and obtained an antibody that broadly neutralizes SARS-CoV-2 variants.

The ability of viruses to mutate and evade the human immune system and neutralizing antibodies remains an obstacle to antiviral and vaccine development. Many neutralizing antibodies, including some approved for emergency use authorization (EUA), reduced or lost activity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants. Here, we introduce a geometric deep learning algorithm that efficiently enhances antibody affinity to achieve broader and more potent neutralizing activity against such variants. We demonstrate the utility of our approach on a human antibody P36-5D2, which is effective against SARS-CoV-2 Alpha, Beta, and Gamma but not Delta. We show that our geometric neural network model optimizes this antibody’s complementarity-determining region (CDR) sequences to improve its binding affinity against multiple SARS-CoV-2 variants. Through iterative optimization of the CDR regions and experimental measurements, we enable expanded antibody breadth and improved potency by ∼10- to 600-fold against SARS-CoV-2 variants, including Delta. We have also demonstrated that our approach can identify CDR changes that alleviate the impact of two Omicron mutations on the epitope. These results highlight the power of our deep learning approach in antibody optimization and its potential application to engineering other protein molecules. Our optimized antibodies can potentially be developed into antibody drug candidates for current and emerging SARS-CoV-2 variants.

[end paper 7]

[begin paper 8]

PMCID = 9367177

Title = Engineering SARS-CoV-2 neutralizing antibodies for increased potency and reduced viral escape pathways

The rapid spread of SARS-CoV-2 variants poses a constant threat of escape from monoclonal antibody and vaccine countermeasures. Mutations in the ACE2 receptor binding site on the surface S protein have been shown to disrupt antibody binding and prevent viral neutralization. Here, we used a directed evolution-based approach to engineer three neutralizing antibodies for enhanced binding to S protein. The engineered antibodies showed increased in vitro functional activity in terms of neutralization potency and/or breadth of neutralization against viral variants. Deep mutational scanning revealed that higher binding affinity reduces the total number of viral escape mutations. Studies in the Syrian hamster model showed two examples where the affinity-matured antibody provided superior protection compared to the parental antibody. These data suggest that monoclonal antibodies for antiviral indications would benefit from affinity maturation to reduce viral escape pathways and appropriate affinity maturation in vaccine immunization could help resist viral variation.

Cryo-EM analysis of Class 2 and Class 3 SARS-CoV-2 neutralizing monoclonal antibodies

Engineered antibodies with higher affinity neutralize emerging SARS-CoV-2 variants

The higher affinity antibodies reduce the pathways for viral escape

Engineered antibodies improve protection in hamster model

Immunology; Virology; Structural biology.

[end paper 8]

[begin paper 9]

PMCID = 9282883

Title = SARS-CoV-2 RBD antibodies that maximize breadth and resistance to escape

An ideal anti-SARS-CoV-2 antibody would resist viral escape, have activity against diverse SARS-related coronaviruses (sarbecoviruses), and be highly protective through viral neutralization and effector functions. Understanding how these properties relate to each other and vary across epitopes would aid development of antibody therapeutics and guide vaccine design. Here, we comprehensively characterize escape, breadth, and potency across a panel of SARS-CoV-2 antibodies targeting the receptor-binding domain (RBD). Despite a tradeoff between in vitro neutralization potency and breadth of sarbecovirus binding, we identify neutralizing antibodies with exceptional sarbecovirus breadth and a corresponding resistance to SARS-CoV-2 escape. One of these antibodies, S2H97, binds with high affinity across all sarbecovirus clades to a previously undescribed cryptic epitope and prophylactically protects hamsters from viral challenge. Antibodies targeting the ACE2 receptor binding motif (RBM) typically have poor breadth and are readily escaped by mutations despite high neutralization potency. Nevertheless, we characterize one potent RBM antibody (S2E12) with breadth across sarbecoviruses related to SARS-CoV-2 and a high barrier to viral escape. These data highlight principles underlying variation in escape, breadth, and potency among antibodies targeting the RBD, and identify epitopes and features to prioritize for therapeutic development against the current and potential future pandemics.

[end paper 9]

[begin paper 10]

PMCID = 9628197

Title = Computationally restoring the potency of a clinical antibody against SARS-CoV-2 Omicron subvariants

The COVID-19 pandemic underscored the promise of monoclonal antibody-based prophylactic and therapeutic drugs, but also revealed how quickly viral escape can curtail effective options. With the emergence of the SARS-CoV-2 Omicron variant in late 2021, many clinically used antibody drug products lost potency, including Evusheld™ and its constituent, cilgavimab. Cilgavimab, like its progenitor COV2-2130, is a class 3 antibody that is compatible with other antibodies in combination and is challenging to replace with existing approaches. Rapidly modifying such high-value antibodies with a known clinical profile to restore efficacy against emerging variants is a compelling mitigation strategy. We sought to redesign COV2-2130 to rescue in vivo efficacy against Omicron BA.1 and BA.1.1 strains while maintaining efficacy against the contemporaneously dominant Delta variant. Here we show that our computationally redesigned antibody, 2130-1-0114-112, achieves this objective, simultaneously increases neutralization potency against Delta and many variants of concern that subsequently emerged, and provides protection in vivo against the strains tested, WA1/2020, BA.1.1, and BA.5. Deep mutational scanning of tens of thousands pseudovirus variants reveals 2130-1-0114-112 improves broad potency without incurring additional escape liabilities. Our results suggest that computational approaches can optimize an antibody to target multiple escape variants, while simultaneously enriching potency. Because our approach is computationally driven, not requiring experimental iterations or pre-existing binding data, it could enable rapid response strategies to address escape variants or pre-emptively mitigate escape vulnerabilities.

[end paper 10]

[begin paper 11]

PMCID = 10986514

Title = Nanobody engineering for SARS-CoV-2 neutralization and detection

In response to the ongoing COVID-19 pandemic, the quest for coronavirus inhibitors has inspired research on a variety of small proteins beyond conventional antibodies, including robust single-domain antibody fragments, i.e., “nanobodies.” Here, we explore the potential of nanobody engineering in the development of antivirals and diagnostic tools. Through fusion of nanobody domains that target distinct binding sites, we engineered multimodular nanobody constructs that neutralize wild-type SARS-CoV-2 and the Alpha and Delta variants at high potency, with IC50 values as low as 50 pM. Despite simultaneous binding to distinct epitopes, Beta and Omicron variants were more resistant to neutralization by the multimodular nanobodies, which highlights the importance of accounting for antigenic drift in the design of biologics. To further explore the applications of nanobody engineering in outbreak management, we present an assay based on fusions of nanobodies with fragments of NanoLuc luciferase that can detect sub-nanomolar quantities of the SARS-CoV-2 spike protein in a single step. Our work showcases the potential of nanobody engineering to combat emerging infectious diseases.

Nanobodies, small protein binders derived from the camelid antibody, are highly potent inhibitors of respiratory viruses that offer several advantages over conventional antibodies as candidates for specific therapies, including high stability and low production costs. In this work, we leverage the unique properties of nanobodies and apply them as building blocks for new therapeutic and diagnostic tools. We report ultra-potent SARS-CoV-2 inhibition by engineered nanobodies comprising multiple modules in structure-guided combinations and develop nanobodies that carry signal molecules, allowing rapid detection of the SARS-CoV-2 spike protein. Our results highlight the potential of engineered nanobodies in the development of effective countermeasures, both therapeutic and diagnostic, to manage outbreaks of emerging viruses.

[end paper 11]

[begin paper 12]

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

[end paper 12]

[begin paper 13]

PMCID = 10104900

Title = Site of vulnerability on SARS-CoV-2 spike induces broadly protective antibody against antigenically distinct Omicron subvariants

The rapid evolution of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) Omicron variants has emphasized the need to identify antibodies with broad neutralizing capabilities to inform future monoclonal therapies and vaccination strategies. Herein, we identified S728-1157, a broadly neutralizing antibody (bnAb) targeting the receptor-binding site (RBS) that was derived from an individual previously infected with WT SARS-CoV-2 prior to the spread of variants of concern (VOCs). S728-1157 demonstrated broad cross-neutralization of all dominant variants, including D614G, Beta, Delta, Kappa, Mu, and Omicron (BA.1/BA.2/BA.2.75/BA.4/BA.5/BL.1/XBB). Furthermore, S728-1157 protected hamsters against in vivo challenges with WT, Delta, and BA.1 viruses. Structural analysis showed that this antibody targets a class 1/RBS-A epitope in the receptor binding domain via multiple hydrophobic and polar interactions with its heavy chain complementarity determining region 3 (CDR-H3), in addition to common motifs in CDR-H1/CDR-H2 of class 1/RBS-A antibodies. Importantly, this epitope was more readily accessible in the open and prefusion state, or in the hexaproline (6P)-stabilized spike constructs, as compared with diproline (2P) constructs. Overall, S728-1157 demonstrates broad therapeutic potential and may inform target-driven vaccine designs against future SARS-CoV-2 variants.

[end paper 13]

[begin paper 14]

PMCID = 8448325

Title = Structural and energetic profiling of SARS-CoV-2 receptor binding domain antibody recognition and the impact of circulating variants

The SARS-CoV-2 pandemic highlights the need for a detailed molecular understanding of protective antibody responses. This is underscored by the emergence and spread of SARS-CoV-2 variants, including Alpha (B.1.1.7) and Delta (B.1.617.2), some of which appear to be less effectively targeted by current monoclonal antibodies and vaccines. Here we report a high resolution and comprehensive map of antibody recognition of the SARS-CoV-2 spike receptor binding domain (RBD), which is the target of most neutralizing antibodies, using computational structural analysis. With a dataset of nonredundant experimentally determined antibody-RBD structures, we classified antibodies by RBD residue binding determinants using unsupervised clustering. We also identified the energetic and conservation features of epitope residues and assessed the capacity of viral variant mutations to disrupt antibody recognition, revealing sets of antibodies predicted to effectively target recently described viral variants. This detailed structure-based reference of antibody RBD recognition signatures can inform therapeutic and vaccine design strategies.

The ongoing COVID-19 pandemic, and the emergence of SARS-CoV-2 variants that evade antibodies induced by vaccines and natural infection, highlights the need for assessment of key molecular and structural features of immune responses against the SARS-CoV-2 virus. Using a large nonredundant set of structures of monoclonal antibodies in complex with the SARS-CoV-2 spike receptor binding domain, we performed analysis of molecular determinants of antibody recognition of the spike glycoprotein, mapping key residues through analysis of atomic contacts and computational modeling to identify molecular hotspots. Clustering was used to identify four major groups of antibodies based on target residues, and we compared epitope conservation and impact of SARS-CoV-2 variant mutations, showing that certain sets of antibodies predicted to be affected by those variants, while others are capable of targeting escape variants. This analysis can serve as a useful reference for vaccine and immunotherapeutic studies, and we provide updated classifications of antibodies to the research community on our CoV3D site.

[end paper 14]

[begin paper 15]

PMCID = 11746035

Title = An allelic atlas of immunoglobulin heavy chain variable regions reveals antibody binding epitope preference resilient to SARS-CoV-2 mutation escape

Although immunoglobulin (Ig) alleles play a pivotal role in the antibody response to pathogens, research to understand their role in the humoral immune response is still limited.

We retrieved the germline sequences for the IGHV from the IMGT database to illustrate the amino acid polymorphism present within germline sequences of IGHV genes. We aassembled the sequences of IgM and IgD repertoire from 130 people to investigate the genetic variations in the population. A dataset comprising 10,643 SARS-CoV-2 spike-specific antibodies, obtained from COV-AbDab, was compiled to assess the impact of SARS-CoV-2 infection on allelic gene utilization. Binding affinity and neutralizing activity were determined using bio-layer interferometry and pseudovirus neutralization assays. Primary docking was performed using ZDOCK (3.0.2) to generate the initial conformation of the antigen-antibody complex, followed by simulations of the complete conformations using Rosetta SnugDock software. The original and simulated structural conformations were visualized and presented using ChimeraX (v1.5).

We present an allelic atlas of immunoglobulin heavy chain (IgH) variable regions, illustrating the diversity of allelic variants across 33 IGHV family germline sequences by sequencing the IgH repertoire of in the population. Our comprehensive analysis of SARS-CoV-2 spike-specific antibodies revealed the preferential use of specific Ig alleles among these antibodies. We observed an association between Ig alleles and antibody binding epitopes. Different allelic genotypes binding to the same RBD epitope on the spike show different neutralizing potency and breadth. We found that antibodies carrying the IGHV1-69*02 allele tended to bind to the RBD E2.2 epitope. The antibodies carrying G50 and L55 amino acid residues exhibit potential enhancements in binding affinity and neutralizing potency to SARS-CoV-2 variants containing the L452R mutation on RBD, whereas R50 and F55 amino acid residues tend to have reduced binding affinity and neutralizing potency. IGHV2-5*02 antibodies using the D56 allele bind to the RBD D2 epitope with greater binding and neutralizing potency due to the interaction between D56 on HCDR2 and K444 on RBD of most Omicron subvariants. In contrast, IGHV2-5*01 antibodies using the N56 allele show increased binding resistance to the K444T mutation on RBD.

This study provides valuable insights into humoral immune responses from the perspective of Ig alleles and population genetics. These findings underscore the importance of Ig alleles in vaccine design and therapeutic antibody development.

[end paper 15]

[begin paper 16]

PMCID = 9910834

Title = The SARS-CoV-2 accessory protein Orf3a is not an ion channel, but does interact with trafficking proteins

The severe acute respiratory syndrome associated coronavirus 2 (SARS-CoV-2) and SARS-CoV-1 accessory protein Orf3a colocalizes with markers of the plasma membrane, endocytic pathway, and Golgi apparatus. Some reports have led to annotation of both Orf3a proteins as viroporins. Here, we show that neither SARS-CoV-2 nor SARS-CoV-1 Orf3a form functional ion conducting pores and that the conductances measured are common contaminants in overexpression and with high levels of protein in reconstitution studies. Cryo-EM structures of both SARS-CoV-2 and SARS-CoV-1 Orf3a display a narrow constriction and the presence of a positively charged aqueous vestibule, which would not favor cation permeation. We observe enrichment of the late endosomal marker Rab7 upon SARS-CoV-2 Orf3a overexpression, and co-immunoprecipitation with VPS39. Interestingly, SARS-CoV-1 Orf3a does not cause the same cellular phenotype as SARS-CoV-2 Orf3a and does not interact with VPS39. To explain this difference, we find that a divergent, unstructured loop of SARS-CoV-2 Orf3a facilitates its binding with VPS39, a HOPS complex tethering protein involved in late endosome and autophagosome fusion with lysosomes. We suggest that the added loop enhances SARS-CoV-2 Orf3a’s ability to co-opt host cellular trafficking mechanisms for viral exit or host immune evasion.

[end paper 16]

[begin paper 17]

PMCID = 8742224

Title = Nasal delivery of an IgM offers broad protection from SARS-CoV-2 variants

Resistance represents a major challenge for antibody-based therapy for COVID-19. Here we engineered an immunoglobulin M (IgM) neutralizing antibody (IgM-14) to overcome the resistance encountered by immunoglobulin G (IgG)-based therapeutics. IgM-14 is over 230-fold more potent than its parental IgG-14 in neutralizing SARS-CoV-2. IgM-14 potently neutralizes the resistant virus raised by its corresponding IgG-14, three variants of concern—B.1.1.7 (Alpha, which first emerged in the UK), P.1 (Gamma, which first emerged in Brazil) and B.1.351 (Beta, which first emerged in South Africa)—and 21 other receptor-binding domain mutants, many of which are resistant to the IgG antibodies that have been authorized for emergency use. Although engineering IgG into IgM enhances antibody potency in general, selection of an optimal epitope is critical for identifying the most effective IgM that can overcome resistance. In mice, a single intranasal dose of IgM-14 at 0.044 mg per kg body weight confers prophylactic efficacy and a single dose at 0.4 mg per kg confers therapeutic efficacy against SARS-CoV-2. IgM-14, but not IgG-14, also confers potent therapeutic protection against the P.1 and B.1.351 variants. IgM-14 exhibits desirable pharmacokinetics and safety profiles when administered intranasally in rodents. Our results show that intranasal administration of an engineered IgM can improve efficacy, reduce resistance and simplify the prophylactic and therapeutic treatment of COVID-19.

[end paper 17]

[begin paper 18]

PMCID = 10702856

Title =  In vitro generated antibodies guide thermostable ADDomer nanoparticle design for nasal vaccination and passive immunization against SARS-CoV-2

Due to COVID-19, pandemic preparedness emerges as a key imperative, necessitating new approaches to accelerate development of reagents against infectious pathogens.

Here, we developed an integrated approach combining synthetic, computational and structural methods with in vitro antibody selection and in vivo immunization to design, produce and validate nature-inspired nanoparticle-based reagents against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).

Our approach resulted in two innovations: (i) a thermostable nasal vaccine called ADDoCoV, displaying multiple copies of a SARS-CoV-2 receptor binding motif derived epitope and (ii) a multivalent nanoparticle superbinder, called Gigabody, against SARS-CoV-2 including immune-evasive variants of concern (VOCs). In vitro generated neutralizing nanobodies and electron cryo-microscopy established authenticity and accessibility of epitopes displayed by ADDoCoV. Gigabody comprising multimerized nanobodies prevented SARS-CoV-2 virion attachment with picomolar EC50. Vaccinating mice resulted in antibodies cross-reacting with VOCs including Delta and Omicron.

Our study elucidates Adenovirus-derived dodecamer (ADDomer)-based nanoparticles for use in active and passive immunization and provides a blueprint for crafting reagents to combat respiratory viral infections.

Statement of Significance: COVID-19 reminds us of the critical importance of pandemic preparedness, requiring sophisticated approaches to facilitate and accelerate reagent development against emerging threats. We described an integrated pipeline approach to rapidly craft in parallel ADDomer-based nanoparticles for vaccination and for passive immunization, applicable to many infectious diseases.

[end paper 18]

[begin paper 19]

PMCID = 10086000

Title = High-throughput identification of prefusion-stabilizing mutations in SARS-CoV-2 spike

Designing prefusion-stabilized SARS-CoV-2 spike is critical for the effectiveness of COVID-19 vaccines. All COVID-19 vaccines in the US encode spike with K986P/V987P mutations to stabilize its prefusion conformation. However, contemporary methods on engineering prefusion-stabilized spike immunogens involve tedious experimental work and heavily rely on structural information. Here, we establish a systematic and unbiased method of identifying mutations that concomitantly improve expression and stabilize the prefusion conformation of the SARS-CoV-2 spike. Our method integrates a fluorescence-based fusion assay, mammalian cell display technology, and deep mutational scanning. As a proof-of-concept, we apply this method to a region in the S2 domain that includes the first heptad repeat and central helix. Our results reveal that besides K986P and V987P, several mutations simultaneously improve expression and significantly lower the fusogenicity of the spike. As prefusion stabilization is a common challenge for viral immunogen design, this work will help accelerate vaccine development against different viruses.

Designing vaccine immunogens is often a tedious process. Here the authors develop a deep mutational scanning-based method to rapidly and comprehensively identify prefusion stabilizing mutations of SARS-CoV-2 spike as a vaccine immunogen.

[end paper 19]

[begin paper 20]

PMCID = 8194341

Title = Prospects of Neutralizing Nanobodies Against SARS-CoV-2

Since December 2019, the SARS-CoV-2 has erupted on a large scale worldwide and spread rapidly. Passive immunization of antibody-related molecules provides opportunities for prevention and treatment of high-risk patients and children. Nanobodies (Nbs) have many strong physical and chemical properties. They can be atomized, administered by inhalation, and can be directly applied to the infected site, with fast onset, high local drug concentration/high bioavailability, and high patient compliance (no needles). It has very attractive potential in the treatment of respiratory viruses. Rapid and low-cost development of Nbs targeting SARS-CoV-2 can quickly be achieved. Nbs against SARS-CoV-2 mutant strains also can be utilized quickly to prevent the virus from escaping. It provides important technical supports for the treatment of the SARS-CoV-2 and has the potential to become an essential medicine in the toolbox against the SARS-CoV-2.

[end paper 20]

[begin paper 21]

PMCID = 11645210

Title = A potent pan-sarbecovirus neutralizing antibody resilient to epitope diversification

SARS-CoV-2 evolution has resulted in viral escape from clinically authorized monoclonal antibodies (mAbs), creating a need for mAbs that are resilient to epitope diversification. Broadly neutralizing coronavirus mAbs that are sufficiently potent for clinical development and retain activity despite viral evolution remain elusive. We identified a human mAb, designated VIR-7229, which targets the viral receptor-binding motif (RBM) with unprecedented cross-reactivity to all sarbecovirus clades, including non-ACE2-utilizing bat sarbecoviruses, while potently neutralizing SARS-CoV-2 variants since 2019, including the recent EG.5, BA.2.86, and JN.1. VIR-7229 tolerates extraordinary epitope variability, partly attributed to its high binding affinity, receptor molecular mimicry, and interactions with RBM backbone atoms. Consequently, VIR-7229 features a high barrier for selection of escape mutants, which are rare and associated with reduced viral fitness, underscoring its potential to be resilient to future viral evolution. VIR-7229 is a strong candidate to become a next-generation COVID-19 medicine.

A monoclonal antibody VIR-7229 uniquely combines potent SARS-CoV-2 inhibition via ACE2 competition, with pan-sarbecovirus and pan-variant cross-reactivity and a high barrier to viral escape. This antibody may be resilient to SARS-CoV-2 evolution, with potential to be a next-generation COVID-19 medicine and a key component of pandemic preparedness in the event of a novel emerging sarbecovirus.

[end paper 21]

[begin paper 22]

PMCID = 7668070

Title = Engineered ACE2 receptor traps potently neutralize SARS-CoV-2

During the ongoing COVID-19 pandemic, protein engineering offers a rapid and powerful approach for building therapeutics to treat severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infections. We use computational design, affinity maturation, and fusion to dimerization domains to engineer “receptor traps” based on wild-type angiotensin-converting enzyme II (ACE2), the target for viral spike-mediated SARS-CoV-2 entry into cells. The optimized ACE2 receptor traps neutralize authentic SARS-CoV-2 infections as effectively as high-affinity antibodies isolated from convalescent patients and also bind viral spike proteins from other coronaviruses known to cause respiratory diseases. ACE2 receptor traps have large binding interfaces and block the entire receptor binding interface, limiting the potential impact of viral escape mutations.

An essential mechanism for severe acute respiratory syndrome coronavirus 1 (SARS-CoV-1) and severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection begins with the viral spike protein binding to the human receptor protein angiotensin-converting enzyme II (ACE2). Here, we describe a stepwise engineering approach to generate a set of affinity optimized, enzymatically inactivated ACE2 variants that potently block SARS-CoV-2 infection of cells. These optimized receptor traps tightly bind the receptor binding domain (RBD) of the viral spike protein and prevent entry into host cells. We first computationally designed the ACE2–RBD interface using a two-stage flexible protein backbone design process that improved affinity for the RBD by up to 12-fold. These designed receptor variants were affinity matured an additional 14-fold by random mutagenesis and selection using yeast surface display. The highest-affinity variant contained seven amino acid changes and bound to the RBD 170-fold more tightly than wild-type ACE2. With the addition of the natural ACE2 collectrin domain and fusion to a human immunoglobulin crystallizable fragment (Fc) domain for increased stabilization and avidity, the most optimal ACE2 receptor traps neutralized SARS-CoV-2–pseudotyped lentivirus and authentic SARS-CoV-2 virus with half-maximal inhibitory concentrations (IC50s) in the 10- to 100-ng/mL range. Engineered ACE2 receptor traps offer a promising route to fighting infections by SARS-CoV-2 and other ACE2-using coronaviruses, with the key advantage that viral resistance would also likely impair viral entry. Moreover, such traps can be predesigned for viruses with known entry receptors for faster therapeutic response without the need for neutralizing antibodies isolated from convalescent patients.

[end paper 22]

[begin paper 23]

PMCID = 8748687

Title = Structural assessment of HLA-A2-restricted SARS-CoV-2 spike epitopes recognized by public and private T-cell receptors

T cells play a vital role in combatting SARS-CoV-2 and forming long-term memory responses. Whereas extensive structural information is available on neutralizing antibodies against SARS-CoV-2, such information on SARS-CoV-2-specific T-cell receptors (TCRs) bound to their peptide–MHC targets is lacking. Here we determine the structures of a public and a private TCR from COVID-19 convalescent patients in complex with HLA-A2 and two SARS-CoV-2 spike protein epitopes (YLQ and RLQ). The structures reveal the basis for selection of particular TRAV and TRBV germline genes by the public but not the private TCR, and for the ability of the TCRs to recognize natural variants of RLQ but not YLQ. Neither TCR recognizes homologous epitopes from human seasonal coronaviruses. By elucidating the mechanism for TCR recognition of an immunodominant yet variable epitope (YLQ) and a conserved but less commonly targeted epitope (RLQ), this study can inform prospective efforts to design vaccines to elicit pan-coronavirus immunity.

Structural immunology is critical in understanding the interplay between the immune response and the infective agent but such studies in T cells and SARS-CoV-2 lag behind those of antibodies and B-cell receptors. Here the authors assess recognition of SARS-CoV-2 spike epitopes and their natural variants by public and private T cell receptors.

[end paper 23]

[begin paper 24]

PMCID = 8043444

Title = Antibodies to the SARS-CoV-2 receptor-binding domain that maximize breadth and resistance to viral escape

An ideal anti-SARS-CoV-2 antibody would resist viral escape, have activity against diverse SARS-related coronaviruses, and be highly protective through viral neutralization and effector functions. Understanding how these properties relate to each other and vary across epitopes would aid development of antibody therapeutics and guide vaccine design. Here, we comprehensively characterize escape, breadth, and potency across a panel of SARS-CoV-2 antibodies targeting the receptor-binding domain (RBD), including S309, the parental antibody of the late-stage clinical antibody VIR-7831. We observe a tradeoff between SARS-CoV-2 in vitro neutralization potency and breadth of binding across SARS-related coronaviruses. Nevertheless, we identify several neutralizing antibodies with exceptional breadth and resistance to escape, including a new antibody (S2H97) that binds with high affinity to all SARS-related coronavirus clades via a unique RBD epitope centered on residue E516. S2H97 and other escape-resistant antibodies have high binding affinity and target functionally constrained RBD residues. We find that antibodies targeting the ACE2 receptor binding motif (RBM) typically have poor breadth and are readily escaped by mutations despite high neutralization potency, but we identify one potent RBM antibody (S2E12) with breadth across sarbecoviruses closely related to SARS-CoV-2 and with a high barrier to viral escape. These data highlight functional diversity among antibodies targeting the RBD and identify epitopes and features to prioritize for antibody and vaccine development against the current and potential future pandemics.

[end paper 24]

[begin paper 25]

PMCID = 8967028

Title = Structure-based dual affinity optimization of a SARS-CoV-1/2 cross-reactive single-domain antibody

The SARS coronavirus 2 (SARS-CoV-2) spike (S) protein binding to the human ACE2 receptor is the molecular event that initiates viral entry into host cells and leads to infection and virus replication. There is a need for agents blocking viral entry into host cells that are cross-reactive with emerging virus variants. VHH-72 is an anti-SARS-CoV-1 single-domain antibody that also exhibits cross-specificity with SARS-CoV-2 but with decreased binding affinity. Here we applied a structure-based approach to affinity-mature VHH-72 for the SARS-CoV-2 spike protein while retaining the original affinity for SARS-CoV-1. This was achieved by employing the computational platform ADAPT in a constrained dual-affinity optimization mode as a means of broadening specificity. Select mutants designed by ADAPT were formatted as fusions with a human IgG1-Fc fragment. These mutants demonstrated improved binding to the SARS-CoV-2 spike protein due to decreased dissociation rates. Functional testing for virus neutralization revealed improvements relative to the parental VHH72-Fc up to 10-fold using a SARS-CoV-2 pseudotyped lentivirus and 20-fold against the SARS-CoV-2 authentic live virus (Wuhan variant). Binding and neutralization improvements were maintained for some other SARS-CoV-2 variants currently in circulation. These improved VHH-72 mutants are predicted to establish novel interactions with the S antigen. They will be useful, alone or as fusions with other functional modules, in the global quest for treatments of COVID-19 infections.

[end paper 25]

[begin paper 26]

PMCID = 9112565

Title = An Electrostatically-steered Conformational Selection Mechanism Promotes SARS-CoV-2 Spike Protein Variation

After two years since the outbreak, the COVID-19 pandemic remains a global public health emergency. SARS-CoV-2 variants with substitutions on the spike (S) protein emerge increasing the risk of immune evasion and cross-species transmission. Here, we analyzed the evolution of the S protein as recorded in 276,712 samples collected before the start of vaccination efforts. Our analysis shows that most variants destabilize the S protein trimer, increase its conformational heterogeneity and improve the odds of the recognition by the host cell receptor. Most frequent substitutions promote overall hydrophobicity by replacing charged amino acids, reducing stabilizing local interactions in the unbound S protein trimer. Moreover, our results identify “forbidden” regions that rarely show any sequence variation, and which are related to conformational changes occurring upon fusion. These results are significant for understanding the structure and function of SARS-CoV-2 related proteins which is a critical step in vaccine development and epidemiological surveillance.

[end paper 26]

[begin paper 27]

PMCID = 8986284

Title = David versus goliath: ACE2-Fc receptor traps as potential SARS-CoV-2 inhibitors

Anti-SARS-CoV-2 monoclonal antibodies and vaccines have shown improvement in lowering viral burden and hospitalization. However, emerging SARS-CoV-2 variants contain neutralizing antibody-escape mutations. Therefore, several reports have suggested the administration of recombinant angiotensin-converting enzyme 2 (rACE2) as a soluble receptor trap to block SARS-CoV-2 infection and limit viral escape potential. Several strategies have been implemented to enhance the efficacy of rACE2 as a therapeutic agent. Fc fusions have been used to improve pharmacokinetics and boost the affinity and avidity of ACE2 decoys for the virus spike protein. Furthermore, the intrinsic catalytic activity of ACE2 can be eliminated by introducing point mutations on the catalytic site of ACE2 to obtain an exclusive antiviral activity. This review summarizes different evolution platforms that have been used to enhance ACE2-Fc (i.e., immunoadhesins) as potential therapeutics for the current pandemic or future outbreaks of SARS-associated betacoronaviruses.

[end paper 27]

[begin paper 28]

PMCID = 7773544

Title = Computational design of SARS-CoV-2 spike glycoproteins to increase immunogenicity by T cell epitope engineering

The development of effective and safe vaccines is the ultimate way to efficiently stop the ongoing COVID-19 pandemic, which is caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Built on the fact that SARS-CoV-2 utilizes the association of its Spike (S) protein with the human angiotensin-converting enzyme 2 (ACE2) receptor to invade host cells, we computationally redesigned the S protein sequence to improve its immunogenicity and antigenicity. Toward this purpose, we extended an evolutionary protein design algorithm, EvoDesign, to create thousands of stable S protein variants that perturb the core protein sequence but keep the surface conformation and B cell epitopes. The T cell epitope content and similarity scores of the perturbed sequences were calculated and evaluated. Out of 22,914 designs with favorable stability energy, 301 candidates contained at least two pre-existing immunity-related epitopes and had promising immunogenic potential. The benchmark tests showed that, although the epitope restraints were not included in the scoring function of EvoDesign, the top S protein design successfully recovered 31 out of the 32 major histocompatibility complex (MHC)-II T cell promiscuous epitopes in the native S protein, where two epitopes were present in all seven human coronaviruses. Moreover, the newly designed S protein introduced nine new MHC-II T cell promiscuous epitopes that do not exist in the wildtype SARS-CoV-2. These results demonstrated a new and effective avenue to enhance a target protein’s immunogenicity using rational protein design, which could be applied for new vaccine design against COVID-19 and other pathogens.

[end paper 28]

[begin paper 29]

PMCID = 10177734

Title = A computationally designed ACE2 decoy has broad efficacy against SARS-CoV-2 omicron variants and related viruses in vitro and in vivo

SARS-CoV-2, especially B.1.1.529/omicron and its sublineages, continues to mutate to evade monoclonal antibodies and antibodies elicited by vaccination. Affinity-enhanced soluble ACE2 (sACE2) is an alternative strategy that works by binding the SARS-CoV-2 S protein, acting as a ‘decoy’ to block the interaction between the S and human ACE2. Using a computational design strategy, we designed an affinity-enhanced ACE2 decoy, FLIF, that exhibited tight binding to SARS-CoV-2 delta and omicron variants. Our computationally calculated absolute binding free energies (ABFE) between sACE2:SARS-CoV-2 S proteins and their variants showed excellent agreement to binding experiments. FLIF displayed robust therapeutic utility against a broad range of SARS-CoV-2 variants and sarbecoviruses, and neutralized omicron BA.5 in vitro and in vivo. Furthermore, we directly compared the in vivo therapeutic efficacy of wild-type ACE2 (non-affinity enhanced ACE2) against FLIF. A few wild-type sACE2 decoys have shown to be effective against early circulating variants such as Wuhan in vivo. Our data suggest that moving forward, affinity-enhanced ACE2 decoys like FLIF may be required to combat evolving SARS-CoV-2 variants. The approach described herein emphasizes how computational methods have become sufficiently accurate for the design of therapeutics against viral protein targets. Affinity-enhanced ACE2 decoys remain highly effective at neutralizing omicron subvariants.

A computational design strategy is used to develop an affinity-enhanced ACE2 decoy, which is shown to be effective at neutralizing omicron subvariants in vitro and in vivo.

[end paper 29]

[begin paper 30]

PMCID = 8826850

Title = Parallel profiling of antigenicity alteration and immune escape of SARS-CoV-2 Omicron and other variants

SARS-CoV-2 variants have evolved a variety of critical mutations, leading to antigenicity changes and immune escape. The recent emerging SARS-CoV-2 Omicron variant attracted global attention due to its significant resistance to current antibody therapies and vaccines. Here, we profiled the mutations of Omicron and other various circulating SARS-CoV-2 variants in parallel by computational interface analysis and in vitro experimental assays. We identified critical mutations that lead to antigenicity changes and diminished neutralization efficiency of a panel of 14 antibodies due to diverse molecular mechanisms influencing the antigen-antibody interaction. Our study identified that Omicron exhibited extraordinary potency in immune escape compared to the other variants of concern, and explores the application of computational interface analysis in SARS-CoV-2 mutation surveillance and demonstrates its potential for the early identification of concerning variants, providing preliminary guidance for neutralizing antibody therapy.

[end paper 30]

[begin paper 31]

PMCID = 11055898

Title = The hinge-engineered IgG1-IgG3 hybrid subclass IgGh47 potently enhances Fc-mediated function of anti-streptococcal and SARS-CoV-2 antibodies

Streptococcus pyogenes can cause invasive disease with high mortality despite adequate antibiotic treatments. To address this unmet need, we have previously generated an opsonic IgG1 monoclonal antibody, Ab25, targeting the bacterial M protein. Here, we engineer the IgG2-4 subclasses of Ab25. Despite having reduced binding, the IgG3 version promotes stronger phagocytosis of bacteria. Using atomic simulations, we show that IgG3’s Fc tail has extensive movement in 3D space due to its extended hinge region, possibly facilitating interactions with immune cells. We replaced the hinge of IgG1 with four different IgG3-hinge segment subclasses, IgGhxx. Hinge-engineering does not diminish binding as with IgG3 but enhances opsonic function, where a 47 amino acid hinge is comparable to IgG3 in function. IgGh47 shows improved protection against S. pyogenes in a systemic infection mouse model, suggesting that IgGh47 has promise as a preclinical therapeutic candidate. Importantly, the enhanced opsonic function of IgGh47 is generalizable to diverse S. pyogenes strains from clinical isolates. We generated IgGh47 versions of anti-SARS-CoV-2 mAbs to broaden the biological applicability, and these also exhibit strongly enhanced opsonic function compared to the IgG1 subclass. The improved function of the IgGh47 subclass in two distant biological systems provides new insights into antibody function.

Here, the authors elongated the hinge structure of IgG1 monoclonal antibodies. The modified IgG1-IgG3 hybrid subclass showed enhanced Fc-mediated function compared to IgG1 in two distinct biological systems, Streptococcus pyogenes and SARS-CoV-2.

[end paper 31]

[begin paper 32]

PMCID = 10069166

Title = Computational design of nanomolar-binding antibodies specific to multiple SARS-CoV-2 variants by engineering a specificity switch of antibody 80R using RosettaAntibodyDesign (RAbD) results in potential generalizable therapeutic antibodies for novel SARS-CoV-2 virus

The human infectious disease COVID-19 caused by the SARS-CoV-2 virus has become a major threat to global public health. Developing a vaccine is the preferred prophylactic response to epidemics and pandemics. However, for individuals who have contracted the disease, the rapid design of antibodies that can target the SARS-CoV-2 virus fulfils a critical need. Further, discovering antibodies that bind multiple variants of SARS-CoV-2 can aid in the development of rapid antigen tests (RATs) which are critical for the identification and isolation of individuals currently carrying COVID-19. Here we provide a proof-of-concept study for the computational design of high-affinity antibodies that bind to multiple variants of the SARS-CoV-2 spike protein using RosettaAntibodyDesign (RAbD). Well characterized antibodies that bind with high affinity to the SARS-CoV-1 (but not SARS-CoV-2) spike protein were used as templates and re-designed to bind the SARS-CoV-2 spike protein with high affinity, resulting in a specificity switch. A panel of designed antibodies were experimentally validated. One design bound to a broad range of variants of concern including the Omicron, Delta, Wuhan, and South African spike protein variants.

· Identification of well characterized antibodies that bind to SARS-CoV-1 but not to SARS-CoV-2.

The use of RosettaAntibodyDesign to design variants of the anti-SARS-CoV-1 antibodies bind to SARS-CoV-2 with high affinity.

Experimental validation of the antibodies designed in silico.

[end paper 32]

[begin paper 33]

PMCID = 9396458

Title = Engineered multivalent self-assembled binder protein against SARS-CoV-2 RBD

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has caused a global pandemic since December 2019, and with it, a push for innovations in rapid testing and neutralizing antibody treatments in an effort to solve the spread and fatality of the disease. One such solution to both of these prevailing issues is targeting the interaction of SARS-CoV-2 spike receptor binding domain (RBD) with the human angiotensin-converting enzyme 2 (ACE2) receptor protein. Structural studies have shown that the N-terminal alpha-helix comprised of the first 23 residues of ACE2 plays an important role in this interaction. Where it is typical to design a binding domain to fit a target, we have engineered a protein that relies on multivalency rather than the sensitivity of a monomeric ligand to provide avidity to its target by fusing the N-terminal helix of ACE2 to the coiled-coil domain of the cartilage oligomeric matrix protein. The resulting ACE-MAP is able to bind to the SARS-CoV-2 RBD with improved binding affinity, is expressible in E. coli, and is thermally stable and relatively small (62 kDa). These properties suggest ACE-MAP and the MAP scaffold to be a promising route towards developing future diagnostics and therapeutics to SARS-CoV-2.

[end paper 33]

[begin paper 34]

PMCID = 8326117

Title = Inhibition of SARS-CoV-2 pseudovirus invasion by ACE2 protecting and Spike neutralizing peptides: An alternative approach to COVID19 prevention and therapy

SARS-CoV-2 invades host cells mainly through the interaction of its spike-protein with host cell membrane ACE2. Various antibodies targeting S-protein have been developed to combat COVID-19 pandemic; however, the potential risk of antibody-dependent enhancement and novel spike mutants-induced neutralization loss or antibody resistance still remain. Alternative preventative agents or therapeutics are still urgently needed. In this study, we designed series of peptides with either ACE2 protecting or Spike-protein neutralizing activities. Molecular docking predicted that, among these peptides, ACE2 protecting peptide AYp28 and Spike-protein neutralizing peptide AYn1 showed strongest intermolecular interaction to ACE2 and Spike-protein, respectively, which were further confirmed by both cell- and non-cell-based in vitro assays. In addition, both peptides inhibited the invasion of pseudotype SARS-CoV-2 into HEK293T/hACE2 cells, either alone or in combination. Moreover, the intranasal administration of AYp28 could partially block pseudovirus invasion in hACE2 transgenic mice. Much more importantly, no significant toxicity was observed in peptides-treated cells. AYp28 showed no impacts on ACE2 function. Taken together, the data from our present study predicted promising preventative and therapeutic values of peptides against COVID-19, and may prove the concept that cocktail containing ACE2 protecting peptides and spike neutralizing peptides could serve as a safe and effective approach for SARS-CoV-2 prevention and therapy.

[end paper 34]

[begin paper 35]

PMCID = 9683733

Title = Probing the biophysical constraints of SARS-CoV-2 spike N-terminal domain using deep mutational scanning

Increasing the expression level of the SARS-CoV-2 spike (S) protein has been critical for COVID-19 vaccine development. While previous efforts largely focused on engineering the receptor-binding domain (RBD) and the S2 subunit, the amino-terminal domain (NTD) has been long overlooked because of the limited understanding of its biophysical constraints. In this study, the effects of thousands of NTD single mutations on S protein expression were quantified by deep mutational scanning. Our results revealed that in terms of S protein expression, the mutational tolerability of NTD residues was inversely correlated with their proximity to the RBD and S2. We also identified NTD mutations at the interdomain interface that increased S protein expression without altering its antigenicity. Overall, this study not only advances the understanding of the biophysical constraints of the NTD but also provides invaluable insights into S-based immunogen design.

A survey on thousands of SARS-CoV-2 S protein NTD mutations reveals a complementary strategy for immunogen optimization.

[end paper 35]

[begin paper 36]

PMCID = 8642831

Title = SARS-CoV-2 nucleocapsid protein binds host mRNAs and attenuates stress granules to impair host stress response

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) nucleocapsid (N) protein is essential for viral replication, making it a promising target for antiviral drug and vaccine development. SARS-CoV-2 infected patients exhibit an uncoordinated immune response; however, the underlying mechanistic details of this imbalance remain obscure. Here, starting from a functional proteomics workflow, we cataloged the protein–protein interactions of SARS-CoV-2 proteins, including an evolutionarily conserved specific interaction of N with the stress granule resident proteins G3BP1 and G3BP2. N localizes to stress granules and sequesters G3BPs away from their typical interaction partners, thus attenuating stress granule formation. We found that N binds directly to host mRNAs in cells, with a preference for 3′ UTRs, and modulates target mRNA stability. We show that the N protein rewires the G3BP1 mRNA-binding profile and suppresses the physiological stress response of host cells, which may explain the imbalanced immune response observed in SARS-CoV-2 infected patients.

AP-MS identifies 753 viral-host protein–protein interactions for 27 SARS-CoV-2 proteins

SARS-CoV-2 N protein sequesters G3BP1/2 and attenuates stress granule formation

N binds directly to mRNAs, rewires G3BP1 mRNA targets, and modulates mRNA stability

SARS-CoV-2 N dampens host stress response via altering posttranscriptional programs

Molecular biology; Virology; Cell biology; Proteomics;

[end paper 36]

[begin paper 37]

PMCID = 9304886

Title = Vascular Endothelial Growth Factor Receptor, fms-Like Tyrosine Kinase-1 (Flt-1), as a Novel Binding Partner for SARS-CoV-2 Spike Receptor-Binding Domain

Angiotensin-converting enzyme 2 (ACE2) and neuropilin 1, a vascular endothelial growth factor (VEGF) receptor, were identified to bind to the SARS-CoV-2 spike receptor-binding domain (spike RBD). In silico analysis based on 3D structure, multiple sequence alignment, and molecular docking of second domain of soluble Flt-1 (sFlt-1) and spike RBD revealed structural similarities, sequence homology, and protein-protein interaction. Interaction and binding of recombinant spike RBD (rspike RBD) and recombinant sFlt-1 (rsFlt-1) in vitro induced a conformational change, as revealed by spectrofluorimetric data, with increased fluorescence intensity in emission spectra as compared to either of the proteins alone. Results on ELISA confirmed the binding and cross-reactivity of rspike-RBD and rsFlt-1 as determined by using either specific antibodies towards each protein or immunized human serum. We found that polyclonal or monoclonal anti-spike RBD antibodies can recognize either rsFlt-1 or rspike RBD, showing cross-reactivity for the two proteins in a dose-dependent binding response. Recognition of bound rspike RBD or rsFlt-1 by anti-Flt-1 or anti-spike RBD antibodies, respectively, as observed by immunoblotting, further confirmed interaction between the two proteins. Immunoprecipitation and immunoblot analysis demonstrated the identification of rspike RBD binding to the Flt-1 receptor on A549 cells. Further, the binding of rspike RBD to Flt-1 receptor was shown using immunofluorescence on 2D-culture or 3D-spheroid of MDA-MB-231 cells, which over-express Flt-1 receptor. Together, our study concludes that the Flt-1 receptor is a novel binding partner for SARS-CoV-2 spike RBD.

[end paper 37]

[begin paper 38]

PMCID = 9207876

Title = Antibodies to combat viral infections: development strategies and progress

Monoclonal antibodies (mAbs) are appealing as potential therapeutics and prophylactics for viral infections owing to characteristics such as their high specificity and their ability to enhance immune responses. Furthermore, antibody engineering can be used to strengthen effector function and prolong mAb half-life, and advances in structural biology have enabled the selection and optimization of potent neutralizing mAbs through identification of vulnerable regions in viral proteins, which can also be relevant for vaccine design. The COVID-19 pandemic has stimulated extensive efforts to develop neutralizing mAbs against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), with several mAbs now having received authorization for emergency use, providing not just an important component of strategies to combat COVID-19 but also a boost to efforts to harness mAbs in therapeutic and preventive settings for other infectious diseases. Here, we describe advances in antibody discovery and engineering that have led to the development of mAbs for use against infections caused by viruses including SARS-CoV-2, respiratory syncytial virus (RSV), Ebola virus (EBOV), human cytomegalovirus (HCMV) and influenza. We also discuss the rationale for moving from empirical to structure-guided strategies in vaccine development, based on identifying optimal candidate antigens and vulnerable regions within them that can be targeted by antibodies to result in a strong protective immune response.

Monoclonal antibodies (mAbs) are appealing as potential therapeutics and prophylactics for viral infections. This Review describes advances in antibody discovery and engineering that have led to the development of mAbs that target viruses such as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), respiratory syncytial virus and Ebola virus, and also considers the implications for vaccine development.

[end paper 38]

[begin paper 39]

PMCID = 7546226

Title = Exploring the out of sight antigens of SARS-CoV-2 to design a candidate multi-epitope vaccine by utilizing immunoinformatics approaches

The vaccine is composed of immunodominant regions of SARS-CoV-2 non-structural proteins.

Also, the functional region of the spike protein is incorporated in the vaccine construct.

The final vaccine construct contains multiple CD8+ and CD4+ overlapping epitopes

Also, it contains multiple IFN-γ inducing, linear and conformational B cell epitopes.

It forms significant interactions and stable complex with TLR-4/MD.

The DNA vaccine is designed by reverse translation of the final vaccine construct.

SARS-CoV-2 causes a severe respiratory disease called COVID-19. Currently, global health is facing its devastating outbreak. However, there is no vaccine available against this virus up to now. In this study, a novel multi-epitope vaccine against SARS-CoV-2 was designed to provoke both innate and adaptive immune responses. The immunodominant regions of six non-structural proteins (nsp7, nsp8, nsp9, nsp10, nsp12 and nsp14) of SARS-CoV-2 were selected by multiple immunoinformatic tools to provoke T cell immune response. Also, immunodominant fragment of the functional region of SARS-CoV-2 spike (400–510 residues) protein was selected for inducing neutralizing antibodies production. The selected regions’ sequences were connected to each other by furin-sensitive linker (RVRR). Moreover, the functional region of β-defensin as a well-known agonist for the TLR-4/MD complex was added at the N-terminus of the vaccine using (EAAAK)3 linker. Also, a CD4 + T-helper epitope, PADRE, was used at the C-terminal of the vaccine by GPGPG and A(EAAAK)2A linkers to form the final vaccine construct. The physicochemical properties, allergenicity, antigenicity, functionality and population coverage of the final vaccine construct were analyzed. The final vaccine construct was an immunogenic, non-allergen and unfunctional protein which contained multiple CD8 + and CD4 + overlapping epitopes, IFN-γ inducing epitopes, linear and conformational B cell epitopes. It could form stable and significant interactions with TLR-4/MD according to molecular docking and dynamics simulations. Global population coverage of the vaccine for HLA-I and II were estimated 96.2% and 97.1%, respectively. At last, the final vaccine construct was reverse translated to design the DNA vaccine. Although the designed vaccine exhibited high efficacy in silico, further experimental validation is necessary.

[end paper 39]

[begin paper 40]

PMCID = 7683566

Title = Targeted intracellular degradation of SARS-CoV-2 via computationally optimized peptide fusions

The COVID-19 pandemic, caused by the novel coronavirus SARS-CoV-2, has elicited a global health crisis of catastrophic proportions. With only a few vaccines approved for early or limited use, there is a critical need for effective antiviral strategies. In this study, we report a unique antiviral platform, through computational design of ACE2-derived peptides which both target the viral spike protein receptor binding domain (RBD) and recruit E3 ubiquitin ligases for subsequent intracellular degradation of SARS-CoV-2 in the proteasome. Our engineered peptide fusions demonstrate robust RBD degradation capabilities in human cells and are capable of inhibiting infection-competent viral production, thus prompting their further experimental characterization and therapeutic development.

Pranam Chatterjee et al. present a novel computational platform for engineering peptide fusions that bind to the SARS-CoV-2 spike protein and tag it for proteasomal degradation. They experimentally validate an optimal variant in human cells, showing that it inhibits production of infection-competent virus.

[end paper 40]

[begin paper 41]

PMCID = 8230804

Title = Full-Length Computational Model of the SARS-CoV-2 Spike Protein and Its Implications for a Viral Membrane Fusion Mechanism

The SARS-CoV-2 virus has now become one of the greatest causes of infectious death and morbidity since the 1918 flu pandemic. Substantial and unprecedented progress has been made in the elucidation of the viral infection process in a short time; however, our understanding of the structure–function dynamics of the spike protein during the membrane fusion process and viral uptake remains incomplete. Employing computational approaches, we use full-length structural models of the SARS-CoV-2 spike protein integrating Cryo-EM images and biophysical properties, which fill the gaps in our understanding. We propose a membrane fusion model incorporating structural transitions associated with the proteolytic processing of the spike protein, which initiates and regulates a series of events to facilitate membrane fusion and viral genome uptake. The membrane fusion mechanism highlights the notable role of the S1 subunit and eventual mature spike protein uptake through the host membrane. Our comprehensive view accounts for distinct neutralizing antibody binding effects targeting the spike protein and the enhanced infectivity of the SARS-CoV-2 variant.

[end paper 41]

[begin paper 42]

PMCID = 9980103

Title = Recognition and Cleavage of Human tRNA Methyltransferase TRMT1 by the SARS-CoV-2 Main Protease

The SARS-CoV-2 main protease (Mpro, or Nsp5) is critical for the production of functional viral proteins during infection and, like many viral proteases, can also target host proteins to subvert their cellular functions. Here, we show that the human tRNA methyltransferase TRMT1 can be recognized and cleaved by SARS-CoV-2 Mpro. TRMT1 installs the N2,N2-dimethylguanosine (m2,2G) modification on mammalian tRNAs, which promotes global protein synthesis and cellular redox homeostasis. We find that Mpro can cleave endogenous TRMT1 in human cell lysate, resulting in removal of the TRMT1 zinc finger domain. TRMT1 proteolysis results in elimination of TRMT1 tRNA methyltransferase activity and reduced tRNA binding affinity. Evolutionary analysis shows that the TRMT1 cleavage site is highly conserved in mammals, except in Muroidea, where TRMT1 is likely resistant to cleavage. In primates, regions outside the cleavage site with rapid evolution could indicate adaptation to ancient viral pathogens. Furthermore, we determined the structure of a TRMT1 peptide in complex with Mpro, revealing a substrate binding conformation distinct from the majority of available Mpro-peptide complexes. Kinetic parameters for peptide cleavage show that the TRMT1(526–536) sequence is cleaved with comparable efficiency to the Mpro-targeted nsp8/9 viral cleavage site. Mutagenesis studies and molecular dynamics simulations together indicate that kinetic discrimination occurs during a later step of Mpro-mediated proteolysis that follows substrate binding. Our results provide new information about the structural basis for Mpro substrate recognition and cleavage, the functional roles of the TRMT1 zinc finger domain in tRNA binding and modification, and the regulation of TRMT1 activity by SARS-CoV-2 Mpro. These studies could inform future therapeutic design targeting Mpro and raise the possibility that proteolysis of human TRMT1 during SARS-CoV-2 infection suppresses protein translation and oxidative stress response to impact viral pathogenesis.

Viral proteases can strategically target human proteins to manipulate host biochemistry during infection. Here, we show that the SARS-CoV-2 main protease (Mpro) can specifically recognize and cleave the human tRNA methyltransferase enzyme TRMT1, and that cleavage of TRMT1 cripples its ability to install a key modification on human tRNAs that is critical for protein translation. Our structural and functional analysis of the Mpro-TRMT1 interaction shows how the flexible Mpro active site engages a conserved sequence in TRMT1 in an uncommon binding mode to catalyze its cleavage and inactivation. These studies provide new insights into substrate recognition by SARS-CoV-2 Mpro that could help guide future antiviral therapeutic development and show how proteolysis of TRMT1 during SARS-CoV-2 infection impairs both TRMT1 tRNA binding and tRNA modification activity to disrupt host translation and potentially impact COVID-19 pathogenesis or phenotypes.

[end paper 42]

[begin paper 43]

PMCID = 7952403

Title = SARS-CoV-2 neutralizing human recombinant antibodies selected from pre-pandemic healthy donors binding at RBD-ACE2 interface

COVID-19 is a severe acute respiratory disease caused by SARS-CoV-2, a new recently emerged sarbecovirus. This virus uses the human ACE2 enzyme as receptor for cell entry, recognizing it with the receptor binding domain (RBD) of the S1 subunit of the viral spike protein. We present the use of phage display to select anti-SARS-CoV-2 spike antibodies from the human naïve antibody gene libraries HAL9/10 and subsequent identification of 309 unique fully human antibodies against S1. 17 antibodies are binding to the RBD, showing inhibition of spike binding to cells expressing ACE2 as scFv-Fc and neutralize active SARS-CoV-2 virus infection of VeroE6 cells. The antibody STE73-2E9 is showing neutralization of active SARS-CoV-2 as IgG and is binding to the ACE2-RBD interface. Thus, universal libraries from healthy human donors offer the advantage that antibodies can be generated quickly and independent from the availability of material from recovering patients in a pandemic situation.

Antibodies targeting the spike protein of coronaviruses are potential candidates for therapeutic development. Here, Bertoglio et al. use phage display to select anti-SARS-CoV-2 spike antibodies from the human naïve universal antibody gene libraries HAL9/10 that block interaction with ACE2 receptor to inhibit infection.

[end paper 43]

[begin paper 44]

PMCID = 8344454

Title = Application of an integrated computational antibody engineering platform to design SARS-CoV-2 neutralizers

As the COVID-19 pandemic continues to spread, hundreds of new initiatives including studies on existing medicines are running to fight the disease. To deliver a potentially immediate and lasting treatment to current and emerging SARS-CoV-2 variants, new collaborations and ways of sharing are required to create as many paths forward as possible. Here, we leverage our expertise in computational antibody engineering to rationally design/engineer three previously reported SARS-CoV neutralizing antibodies and share our proposal towards anti-SARS-CoV-2 biologics therapeutics. SARS-CoV neutralizing antibodies, m396, 80R and CR-3022 were chosen as templates due to their diversified epitopes and confirmed neutralization potency against SARS-CoV (but not SARS-CoV-2 except for CR3022). Structures of variable fragment (Fv) in complex with receptor binding domain (RBD) from SARS-CoV or SARS-CoV-2 were subjected to our established in silico antibody engineering platform to improve their binding affinity to SARS-CoV-2 and developability profiles. The selected top mutations were ensembled into a focused library for each antibody for further screening. In addition, we convert the selected binders with different epitopes into the trispecific format, aiming to increase potency and to prevent mutational escape. Lastly, to avoid antibody-induced virus activation or enhancement, we suggest application of NNAS and DQ mutations to the Fc region to eliminate effector functions and extend half-life.

[end paper 44]

[begin paper 45]

PMCID = 10042144

Title = Computational Protein Design for COVID-19 Research and Emerging Therapeutics

As the world struggles with the ongoing COVID-19 pandemic, unprecedented obstacles have continuously been traversed as new SARS-CoV-2 variants continually emerge. Infectious disease outbreaks are unavoidable, but the knowledge gained from the successes and failures will help create a robust health management system to deal with such pandemics. Previously, scientists required years to develop diagnostics, therapeutics, or vaccines; however, we have seen that, with the rapid deployment of high-throughput technologies and unprecedented scientific collaboration worldwide, breakthrough discoveries can be accelerated and insights broadened. Computational protein design (CPD) is a game-changing new technology that has provided alternative therapeutic strategies for pandemic management. In addition to the development of peptide-based inhibitors, miniprotein binders, decoys, biosensors, nanobodies, and monoclonal antibodies, CPD has also been used to redesign native SARS-CoV-2 proteins and human ACE2 receptors. We discuss how novel CPD strategies have been exploited to develop rationally designed and robust COVID-19 treatment strategies.

This outlook summarizes CPD approaches utilized for the development of SARS-CoV-2 therapeutics and highlights the current challenges and future directions of CPD to tackle various infectious diseases.

[end paper 45]

[begin paper 46]

PMCID = 8423221

Title = Epistasis at the SARS-CoV-2 RBD Interface and the Propitiously Boring Implications for Vaccine Escape

At the time of this writing, December 2021, potential emergence of vaccine escape variants of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is a grave global concern. The interface between the receptor-binding domain (RBD) of SARS-CoV-2 spike (S) protein and the host receptor (ACE2) overlap with the binding site of principal neutralizing antibodies (NAb), limiting the repertoire of viable mutations. Nonetheless, variants with multiple mutations in the RBD have rose to dominance. Non-additive, epistatic relationships among RBD mutations are apparent, and assessing the impact of such epistasis on the mutational landscape is crucial. Epistasis can substantially increase the risk of vaccine escape and cannot be completely characterized through the study of the wild type (WT) alone. We employed protein structure modeling using Rosetta to compare the effects of all single mutants at the RBD-NAb and RBD-ACE2 interfaces for the WT, Delta, Gamma, and Omicron variants. Overall, epistasis at the RBD interface appears to be limited and the effects of most multiple mutations are additive. Epistasis at the Delta variant interface weakly stabilizes NAb interaction relative to ACE2 interaction, whereas in the Gamma variant, epistasis more substantially destabilizes NAb interaction. Although a small, systematic trend towards NAb destabilization not observed for Delta or Gamma was detected for Omicron, and despite bearing significantly more RBD mutations, the epistatic landscape of the Omicron variant closely resembles that of Gamma. These results suggest that, although Omicron poses new risks not observed with Delta, structural constraints on the RBD hamper continued evolution towards more complete vaccine escape. The modest ensemble of mutations relative to the WT that are currently known to reduce vaccine efficacy is likely to comprise the majority of all possible escape mutations for future variants, predicting continued efficacy of the existing vaccines.

[end paper 46]

[begin paper 47]

PMCID = 7900118

Title = In silico detection of SARS-CoV-2 specific B-cell epitopes and validation in ELISA for serological diagnosis of COVID-19

Rapid generation of diagnostics is paramount to understand epidemiology and to control the spread of emerging infectious diseases such as COVID-19. Computational methods to predict serodiagnostic epitopes that are specific for the pathogen could help accelerate the development of new diagnostics. A systematic survey of 27 SARS-CoV-2 proteins was conducted to assess whether existing B-cell epitope prediction methods, combined with comprehensive mining of sequence databases and structural data, could predict whether a particular protein would be suitable for serodiagnosis. Nine of the predictions were validated with recombinant SARS-CoV-2 proteins in the ELISA format using plasma and sera from patients with SARS-CoV-2 infection, and a further 11 predictions were compared to the recent literature. Results appeared to be in agreement with 12 of the predictions, in disagreement with 3, while a further 5 were deemed inconclusive. We showed that two of our top five candidates, the N-terminal fragment of the nucleoprotein and the receptor-binding domain of the spike protein, have the highest sensitivity and specificity and signal-to-noise ratio for detecting COVID-19 sera/plasma by ELISA. Mixing the two antigens together for coating ELISA plates led to a sensitivity of 94% (N = 80 samples from persons with RT-PCR confirmed SARS-CoV-2 infection), and a specificity of 97.2% (N = 106 control samples).

[end paper 47]

[begin paper 48]

PMCID = 11672753

Title = Plant‐produced SARS‐CoV‐2 antibody engineered towards enhanced potency and in vivo efficacy

Prevention of severe COVID‐19 disease by SARS‐CoV‐2 in high‐risk patients, such as immuno‐compromised individuals, can be achieved by administration of antibody prophylaxis, but producing antibodies can be costly. Plant expression platforms allow substantial lower production costs compared to traditional bio‐manufacturing platforms depending on mammalian cells in bioreactors. In this study, we describe the expression, production and purification of the originally human COVA2‐15 antibody in plants. Our plant‐produced mAbs demonstrated comparable neutralizing activity with COVA2‐15 produced in mammalian cells. Furthermore, they exhibited similar capacity to prevent SARS‐CoV‐2 infection in a hamster model. To further enhance these biosimilars, we performed three glyco‐ and protein engineering techniques. First, to increase antibody half‐life, we introduced YTE‐mutation in the Fc tail; second, optimization of N‐linked glycosylation by the addition of a C‐terminal ER‐retention motif (HDEL), and finally; production of mAb in plant production lines lacking β‐1,2‐xylosyltransferase and α‐1,3‐fucosyltransferase activities (FX‐KO). These engineered biosimilars exhibited optimized glycosylation, enhanced phagocytosis and NK cell activation capacity compared to conventional plant‐produced S15 and M15 biosimilars, in some cases outperforming mammalian cell produced COVA2‐15. These engineered antibodies hold great potential for enhancing in vivo efficacy of mAb treatment against COVID‐19 and provide a platform for the development of antibodies against other emerging viruses in a cost‐effective manner.

[end paper 48]

[begin paper 49]

PMCID = 8811534

Title = SARS-CoV-2 RNA elements share human sequence identity and upregulate hyaluronan via NamiRNA-enhancer network

Since late 2019, SARS-CoV-2 infection has resulted in COVID-19 accompanied by diverse clinical manifestations. However, the underlying mechanism of how SARS-CoV-2 interacts with host and develops multiple symptoms is largely unexplored.

Bioinformatics analysis determined the sequence similarity between SARS-CoV-2 and human genomes. Diverse fragments of SARS-CoV-2 genome containing Human Identical Sequences (HIS) were cloned into the lentiviral vector. HEK293T, MRC5 and HUVEC were infected with laboratory-packaged lentivirus or transfected with plasmids or antagomirs for HIS. Quantitative RT-PCR and chromatin immunoprecipitation assay detected gene expression and H3K27ac enrichment, respectively. UV-Vis spectroscopy assessed the interaction between HIS and their target locus. Enzyme-linked immunosorbent assay evaluated the hyaluronan (HA) levels of culture supernatant and plasma of COVID-19 patients.

Five short sequences (24–27 nt length) sharing identity between SARS-CoV-2 and human genome were identified. These RNA elements were highly conserved in primates. The genomic fragments containing HIS were predicted to form hairpin structures in silico similar to miRNA precursors. HIS may function through direct genomic interaction leading to activation of host enhancers, and upregulation of adjacent and distant genes, including cytokine genes and hyaluronan synthase 2 (HAS2). HIS antagomirs and Cas13d-mediated HIS degradation reduced HAS2 expression. Severe COVID-19 patients displayed decreased lymphocytes and elevated D-dimer, and C-reactive proteins, as well as increased plasma hyaluronan. Hymecromone inhibited hyaluronan production in vitro, and thus could be further investigated as a therapeutic option for preventing severe outcome in COVID-19 patients.

HIS of SARS-CoV-2 could promote COVID-19 progression by upregulating hyaluronan, providing novel targets for treatment.

The National Key R&D Program of China (2018YFC1005004), Major Special Projects of Basic Research of Shanghai Science and Technology Commission (18JC1411101), and the National Natural Science Foundation of China (31872814, 32000505).

[end paper 49]

[begin paper 50]

PMCID = 11316817

Title = De novo generation of SARS-CoV-2 antibody CDRH3 with a pre-trained generative large language model

Artificial Intelligence (AI) techniques have made great advances in assisting antibody design. However, antibody design still heavily relies on isolating antigen-specific antibodies from serum, which is a resource-intensive and time-consuming process. To address this issue, we propose a Pre-trained Antibody generative large Language Model (PALM-H3) for the de novo generation of artificial antibodies heavy chain complementarity-determining region 3 (CDRH3) with desired antigen-binding specificity, reducing the reliance on natural antibodies. We also build a high-precision model antigen-antibody binder (A2binder) that pairs antigen epitope sequences with antibody sequences to predict binding specificity and affinity. PALM-H3-generated antibodies exhibit binding ability to SARS-CoV-2 antigens, including the emerging XBB variant, as confirmed through in-silico analysis and in-vitro assays. The in-vitro assays validate that PALM-H3-generated antibodies achieve high binding affinity and potent neutralization capability against spike proteins of SARS-CoV-2 wild-type, Alpha, Delta, and the emerging XBB variant. Meanwhile, A2binder demonstrates exceptional predictive performance on binding specificity for various epitopes and variants. Furthermore, by incorporating the attention mechanism inherent in the Roformer architecture into the PALM-H3 model, we improve its interpretability, providing crucial insights into the fundamental principles of antibody design.

Antibody design still heavily relies on isolating antigen-specific antibodies from serum. Here the authors report a Pre-trained Antibody generative large Language Model (PALM-H3) for the de novo generation of artificial antibodies heavy chain complementarity-determining region 3 with desired antigen-binding specificity.

[end paper 50]

[begin paper 51]

PMCID = 9514956

Title = Anti-SARS-CoV-2 immunoadhesin remains effective against Omicron and other emerging variants of concern

Blocking the interaction of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) with its angiotensin-converting enzyme 2 (ACE2) receptor was proved to be an effective therapeutic option. Various protein binders as well as monoclonal antibodies that effectively target the receptor-binding domain (RBD) of SARS-CoV-2 to prevent interaction with ACE2 were developed. The emergence of SARS-CoV-2 variants that accumulate alterations in the RBD can severely affect the efficacy of such immunotherapeutic agents, as is indeed the case with Omicron that resists many of the previously isolated monoclonal antibodies. Here, we evaluate an ACE2-based immunoadhesin that we have developed early in the pandemic against some of the recent variants of concern (VoCs), including the Delta and the Omicron variants. We show that our ACE2-immunoadhesin remains effective in neutralizing these variants, suggesting that immunoadhesin-based immunotherapy is less prone to escape by the virus and has a potential to remain effective against future VoCs.

ACE2-based immunoadhesins are a promising therapeutic tool against SARS-CoV-2

We employed a computational approach to design a superior ACE2 binder

An order of magnitude increase in neutralization capacity compared with human ACE2

The modified ACE2 retains efficacy against emerging variants of concern

Health sciences; Virology; Immunology; Biological sciences.

[end paper 51]

[begin paper 52]

PMCID = 9515332

Title = Reduced B cell antigenicity of Omicron lowers host serologic response

The SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. However, the mechanism remains unclear. Here, we find using a geometric deep-learning model that Omicron’s extensively mutated receptor binding site (RBS) features reduced antigenicity compared with previous variants. Mice immunization experiments with different recombinant receptor binding domain (RBD) variants confirm that the serological response to Omicron is drastically attenuated and less potent. Analyses of serum cross-reactivity and competitive ELISA reveal a reduction in antibody response across both variable and conserved RBD epitopes. Computational modeling confirms that the RBS has a potential for further antigenicity reduction while retaining efficient receptor binding. Finally, we find a similar trend of antigenicity reduction over decades for hCoV229E, a common cold coronavirus. Thus, our study explains the reduced antibody titers associated with Omicron infection and reveals a possible trajectory of future viral evolution.

SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. Tubiana et al. investigate the underlying mechanism using geometric deep learning, mice immunization experiments, and biochemical assays. Mutations reduce antigenicity of the receptor binding site, leading to lower antibody response.

[end paper 52]

[begin paper 53]

PMCID = 8329052

Title = Computational design of SARS-CoV-2 peptide binders with better predicted binding affinities than human ACE2 receptor

SARS-CoV-2 is coronavirus causing COVID-19 pandemic. To enter human cells, receptor binding domain of S1 subunit of SARS-CoV-2 (SARS-CoV-2-RBD) binds to peptidase domain (PD) of angiotensin-converting enzyme 2 (ACE2) receptor. Employing peptides to inhibit binding between SARS-CoV-2-RBD and ACE2-PD is a therapeutic solution for COVID-19. Previous experimental study found that 23-mer peptide (SBP1) bound to SARS-CoV-2-RBD with lower affinity than ACE2. To increase SBP1 affinity, our previous study used residues 21–45 of α1 helix of ACE2-PD (SPB25) to design peptides with predicted affinity better than SBP1 and SPB25 by increasing interactions of residues that do not form favorable interactions with SARS-CoV-2-RBD. To design SPB25 with better affinity than ACE2, we employed computational protein design to increase interactions of residues reported to form favorable interactions with SARS-CoV-2-RBD and combine newly designed mutations with the best single mutations from our previous study. Molecular dynamics show that predicted binding affinities of three peptides (SPB25Q22R, SPB25F8R/K11W/L25R and SPB25F8R/K11F/Q22R/L25R) are better than ACE2. Moreover, their predicted stabilities may be slightly higher than SBP1 as suggested by their helicities. This study developed an approach to design SARS-CoV-2 peptide binders with predicted binding affinities better than ACE2. These designed peptides are promising candidates as SARS-CoV-2 inhibitors.

[end paper 53]

[begin paper 54]

PMCID = 11437503

Title = AlphaFold2 Modeling and Molecular Dynamics Simulations of the Conformational Ensembles for the SARS-CoV-2 Spike Omicron JN.1, KP.2 and KP.3 Variants: Mutational Profiling of Binding Energetics Reveals Epistatic Drivers of the ACE2 Affinity and Escape Hotspots of Antibody Resistance

The most recent wave of SARS-CoV-2 Omicron variants descending from BA.2 and BA.2.86 exhibited improved viral growth and fitness due to convergent evolution of functional hotspots. These hotspots operate in tandem to optimize both receptor binding for effective infection and immune evasion efficiency, thereby maintaining overall viral fitness. The lack of molecular details on structure, dynamics and binding energetics of the latest FLiRT and FLuQE variants with the ACE2 receptor and antibodies provides a considerable challenge that is explored in this study. We combined AlphaFold2-based atomistic predictions of structures and conformational ensembles of the SARS-CoV-2 spike complexes with the host receptor ACE2 for the most dominant Omicron variants JN.1, KP.1, KP.2 and KP.3 to examine the mechanisms underlying the role of convergent evolution hotspots in balancing ACE2 binding and antibody evasion. Using the ensemble-based mutational scanning of the spike protein residues and computations of binding affinities, we identified binding energy hotspots and characterized the molecular basis underlying epistatic couplings between convergent mutational hotspots. The results suggested the existence of epistatic interactions between convergent mutational sites at L455, F456, Q493 positions that protect and restore ACE2-binding affinity while conferring beneficial immune escape. To examine immune escape mechanisms, we performed structure-based mutational profiling of the spike protein binding with several classes of antibodies that displayed impaired neutralization against BA.2.86, JN.1, KP.2 and KP.3. The results confirmed the experimental data that JN.1, KP.2 and KP.3 harboring the L455S and F456L mutations can significantly impair the neutralizing activity of class 1 monoclonal antibodies, while the epistatic effects mediated by F456L can facilitate the subsequent convergence of Q493E changes to rescue ACE2 binding. Structural and energetic analysis provided a rationale to the experimental results showing that BD55-5840 and BD55-5514 antibodies that bind to different binding epitopes can retain neutralizing efficacy against all examined variants BA.2.86, JN.1, KP.2 and KP.3. The results support the notion that evolution of Omicron variants may favor emergence of lineages with beneficial combinations of mutations involving mediators of epistatic couplings that control balance of high ACE2 affinity and immune evasion.

[end paper 54]

[begin paper 55]

PMCID = 10693169

Title = Development of a neutralization monoclonal antibody with a broad neutralizing effect against SARS-CoV-2 variants

The emergence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants has challenged the effectiveness of current therapeutic regimens. Here, we aimed to develop a potent SARS-CoV-2 antibody with broad neutralizing effect by screening a scFv library with the spike protein receptor-binding domain (RBD) via phage display.

SKAI-DS84 was identified through phage display, and we performed pseudovirus neutralization assays, authentic virus neutralization assays, and in vivo neutralization efficacy evaluations. Furthermore, surface plasmon resonance (SPR) analysis was conducted to assess the physical characteristics of the antibody, including binding kinetics and measure its affinity for variant RBDs.

The selected clones were converted to human IgG, and among them, SKAI-DS84 was selected for further analyses based on its binding affinity with the variant RBDs. Using pseudoviruses, we confirmed that SKAI-DS84 was strongly neutralizing against wild-type, B.1.617.2, B.1.1.529, and subvariants of SARS-CoV-2. We also tested the neutralizing effect of SKAI-DS84 on authentic viruses, in vivo and observed a reduction in viral replication and improved lung pathology. We performed binding and epitope mapping experiments to understand the mechanisms underlying neutralization and identified quaternary epitopes formed by the interaction between RBDs as the target of SKAI-DS84.

We identified, produced, and tested the neutralizing effect of SKAI-DS84 antibody. Our results highlight that SKAI-DS84 could be a potential neutralizing antibody against SARS-CoV-2 and its variants.

The online version contains supplementary material available at 10.1186/s12985-023-02230-9.

[end paper 55]

[begin paper 56]

PMCID = 7891125

Title = The antigenic anatomy of SARS-CoV-2 receptor binding domain

Antibodies are crucial to immune protection against SARS-CoV-2, with some in emergency use as therapeutics. Here, we identify 377 human monoclonal antibodies (mAbs) recognizing the virus spike and focus mainly on 80 that bind the receptor binding domain (RBD). We devise a competition data-driven method to map RBD binding sites. We find that although antibody binding sites are widely dispersed, neutralizing antibody binding is focused, with nearly all highly inhibitory mAbs (IC50 < 0.1 μg/mL) blocking receptor interaction, except for one that binds a unique epitope in the N-terminal domain. Many of these neutralizing mAbs use public V-genes and are close to germline. We dissect the structural basis of recognition for this large panel of antibodies through X-ray crystallography and cryoelectron microscopy of 19 Fab-antigen structures. We find novel binding modes for some potently inhibitory antibodies and demonstrate that strongly neutralizing mAbs protect, prophylactically or therapeutically, in animal models.

Map 377 mAbs: 19 of 80 recognizing the RBD are potent neutralizers; 1 potent NTD binder

19 Fab-antigen complex structures; 80 mAbs mapped on RBD and clustered into 5 epitopes

Most potent mAbs are ACE2 blockers, neutralize with few ACE2s, some Fabs glycosylated

mAbs reveal unique examples of NTD binding, RBD binding mode, and LC optimization

Dejnirattisai et al. present an in-depth study of the human antibody response to SARS-CoV-2 infection. By characterizing 377 human mAbs from recovered COVID-19 patients, and determining 19 protein structures, they construct a map of antibody footprints on the RBD that describes in great detail its antigenic anatomy.

[end paper 56]

[begin paper 57]

PMCID = 7871111

Title = Current and prospective computational approaches and challenges for developing COVID-19 vaccines

SARS-CoV-2, which causes COVID-19, was first identified in humans in late 2019 and is a coronavirus which is zoonotic in origin. As it spread around the world there has been an unprecedented effort in developing effective vaccines. Computational methods can be used to speed up the long and costly process of vaccine development. Antigen selection, epitope prediction, and toxicity and allergenicity prediction are areas in which computational tools have already been applied as part of reverse vaccinology for SARS-CoV-2 vaccine development. However, there is potential for computational methods to assist further. We review approaches which have been used and highlight additional bioinformatic approaches and PK modelling as in silico methods which may be useful for SARS-CoV-2 vaccine design but remain currently unexplored. As more novel viruses with pandemic potential are expected to arise in future, these techniques are not limited to application to SARS-CoV-2 but also useful to rapidly respond to novel emerging viruses.

[end paper 57]

[begin paper 58]

PMCID = 7663370

Title = Epitope-Based Immunoinformatics Approach on Nucleocapsid Protein of Severe Acute Respiratory Syndrome-Coronavirus-2

With an increasing fatality rate, severe acute respiratory syndrome-coronavirus-2 (SARS-CoV-2) has emerged as a promising threat to human health worldwide. Recently, the World Health Organization (WHO) has announced the infectious disease caused by SARS-CoV-2, which is known as coronavirus disease-2019 (COVID-2019), as a global pandemic. Additionally, the positive cases are still following an upward trend worldwide and as a corollary, there is a need for a potential vaccine to impede the progression of the disease. Lately, it has been documented that the nucleocapsid (N) protein of SARS-CoV-2 is responsible for viral replication and interferes with host immune responses. We comparatively analyzed the sequences of N protein of SARS-CoV-2 for the identification of core attributes and analyzed the ancestry through phylogenetic analysis. Subsequently, we predicted the most immunogenic epitope for the T-cell and B-cell. Importantly, our investigation mainly focused on major histocompatibility complex (MHC) class I potential peptides and NTASWFTAL interacted with most human leukocyte antigen (HLA) that are encoded by MHC class I molecules. Further, molecular docking analysis unveiled that NTASWFTAL possessed a greater affinity towards HLA and also available in a greater range of the population. Our study provides a consolidated base for vaccine design and we hope that this computational analysis will pave the way for designing novel vaccine candidates.

[end paper 58]

[begin paper 59]

PMCID = 11031966

Title = Computational analysis of affinity dynamics between the variants of SARS-CoV-2 spike protein (RBD) and human ACE-2 receptor

The novel coronavirus SARS-CoV-2 resulted in a significant worldwide health emergency known as the COVID-19 pandemic. This crisis has been marked by the widespread of various variants, with certain ones causing notable apprehension. In this study, we harnessed computational techniques to scrutinize these Variants of Concern (VOCs), including various Omicron subvariants. Our approach involved the use of protein structure prediction algorithms and molecular docking techniques, we have investigated the effects of mutations within the Receptor Binding Domain (RBD) of SARS-CoV-2 and how these mutations influence its interactions with the human angiotensin-converting enzyme 2 (hACE-2) receptor. Further we have predicted the structural alterations in the RBD of naturally occurring SARS-CoV-2 variants using the tr-Rosetta algorithm. Subsequent docking and binding analysis employing HADDOCK and PRODIGY illuminated crucial interactions occurring at the Receptor-Binding Motif (RBM). Our findings revealed a hierarchy of increased binding affinity between the human ACE2 receptor and the various RBDs, in the order of wild type (Wuhan-strain) < Beta < Alpha < Gamma < Omicron-B.1.1.529 < Delta < Omicron-BA.2.12.1 < Omicron-BA.5.2.1 < Omicron-BA.1.1. Notably, Omicron-BA.1.1 demonstrated the highest binding affinity of -17.4 kcal mol−1 to the hACE2 receptor when compared to all the mutant complexes. Additionally, our examination indicated that mutations occurring in active residues of the Receptor Binding Domain (RBD) consistently improved the binding affinity and intermolecular interactions in all mutant complexes. Analysis of the differences among variants has laid a foundation for the structure-based drug design targeting the RBD region of SARS-CoV-2.

The online version contains supplementary material available at 10.1186/s12985-024-02365-3.

[end paper 59]

[begin paper 60]

PMCID = 8165686

Title = Plausible blockers of Spike RBD in SARS-CoV2—molecular design and underlying interaction dynamics from high-level structural descriptors

COVID-19 is characterized by an unprecedented abrupt increase in the viral transmission rate (SARS-CoV-2) relative to its pandemic evolutionary ancestor, SARS-CoV (2003). The complex molecular cascade of events related to the viral pathogenicity is triggered by the Spike protein upon interacting with the ACE2 receptor on human lung cells through its receptor binding domain (RBDSpike). One potential therapeutic strategy to combat COVID-19 could thus be limiting the infection by blocking this key interaction. In this current study, we adopt a protein design approach to predict and propose non-virulent structural mimics of the RBDSpike which can potentially serve as its competitive inhibitors in binding to ACE2. The RBDSpike is an independently foldable protein domain, resilient to conformational changes upon mutations and therefore an attractive target for strategic re-design. Interestingly, in spite of displaying an optimal shape fit between their interacting surfaces (attributed to a consequently high mutual affinity), the RBDSpike–ACE2 interaction appears to have a quasi-stable character due to a poor electrostatic match at their interface. Structural analyses of homologous protein complexes reveal that the ACE2 binding site of RBDSpike has an unusually high degree of solvent-exposed hydrophobic residues, attributed to key evolutionary changes, making it inherently “reaction-prone.” The designed mimics aimed to block the viral entry by occupying the available binding sites on ACE2, are tested to have signatures of stable high-affinity binding with ACE2 (cross-validated by appropriate free energy estimates), overriding the native quasi-stable feature. The results show the apt of directly adapting natural examples in rational protein design, wherein, homology-based threading coupled with strategic “hydrophobic ↔ polar” mutations serve as a potential breakthrough.

The online version contains supplementary material available at 10.1007/s00894-021-04779-0.

[end paper 60]

[begin paper 61]

PMCID = 8217500

Title = Sequence signatures of two public antibody clonotypes that bind SARS-CoV-2 receptor binding domain

Since the COVID-19 pandemic onset, the antibody response to SARS-CoV-2 has been extensively characterized. Antibodies to the receptor binding domain (RBD) on the spike protein are frequently encoded by IGHV3-53/3-66 with a short complementarity-determining region (CDR) H3. Germline-encoded sequence motifs in heavy chain CDRs H1 and H2 have a major function, but whether any common motifs are present in CDR H3, which is often critical for binding specificity, is not clear. Here, we identify two public clonotypes of IGHV3-53/3-66 RBD antibodies with a 9-residue CDR H3 that pair with different light chains. Distinct sequence motifs on CDR H3 are present in the two public clonotypes that seem to be related to differential light chain pairing. Additionally, we show that Y58F is a common somatic hypermutation that results in increased binding affinity of IGHV3-53/3-66 RBD antibodies with a short CDR H3. These results advance understanding of the antibody response to SARS-CoV-2.

Public antibody clonotypes that recognize SARS-CoV-2 spike protein are important for protection against COVID-19. Here, the authors characterize sequence motifs in the heavy chain complementarity-determining region (CDR) H3s of two public clonotypes and their association with light chain identity.

[end paper 61]

[begin paper 62]

PMCID = 9258422

Title = Multivalent designed proteins neutralize SARS-CoV-2 variants of concern and confer protection against infection in mice

New variants of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) continue to arise and prolong the coronavirus disease 2019 (COVID-19) pandemic. Here we used a cell-free expression workflow to rapidly screen and optimize constructs containing multiple computationally designed miniprotein inhibitors of SARS-CoV-2. We found the broadest efficacy with a homo-trimeric version of the 75-residue angiotensin converting enzyme 2 (ACE2) mimic AHB2 (TRI2–2) designed to geometrically match the trimeric spike architecture. In the cryo-electron microscopy structure, TRI2 formed a tripod on top of the spike protein which engaged all three receptor binding domains (RBDs) simultaneously as in the design model. TRI2–2 neutralized Omicron (B.1.1.529), Delta (B.1.617.2), and all other variants tested with greater potency than that of monoclonal antibodies used clinically for the treatment of COVID-19. TRI2–2 also conferred prophylactic and therapeutic protection against SARS-CoV-2 challenge when administered intranasally in mice. Designed miniprotein receptor mimics geometrically arrayed to match pathogen receptor binding sites could be a widely applicable antiviral therapeutic strategy with advantages over antibodies and native receptor traps. By comparison, the designed proteins have resistance to viral escape and antigenic drift by construction, precisely tuned avidity, and greatly reduced chance of autoimmune responses.

Computationally designed trivalent minibinders provide therapeutic protection in mice against emerging SARS-CoV-2 variants of concern.

[end paper 62]

[begin paper 63]

PMCID = 7587204

Title = Biomedical Science to Tackle the COVID-19 Pandemic: Current Status and Future Perspectives

The coronavirus infectious disease (COVID-19) pandemic emerged at the end of 2019, and was caused by the Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2), which has resulted in an unprecedented health and economic crisis worldwide. One key aspect, compared to other recent pandemics, is the level of urgency, which has started a race for finding adequate answers. Solutions for efficient prevention approaches, rapid, reliable, and high throughput diagnostics, monitoring, and safe therapies are needed. Research across the world has been directed to fight against COVID-19. Biomedical science has been presented as a possible area for combating the SARS-CoV-2 virus due to the unique challenges raised by the pandemic, as reported by epidemiologists, immunologists, and medical doctors, including COVID-19’s survival, symptoms, protein surface composition, and infection mechanisms. While the current knowledge about the SARS-CoV-2 virus is still limited, various (old and new) biomedical approaches have been developed and tested. Here, we review the current status and future perspectives of biomedical science in the context of COVID-19, including nanotechnology, prevention through vaccine engineering, diagnostic, monitoring, and therapy. This review is aimed at discussing the current impact of biomedical science in healthcare for the management of COVID-19, as well as some challenges to be addressed.

[end paper 63]

[begin paper 64]

PMCID = 7430581

Title = Rational Design of SARS-CoV-2 Spike Glycoproteins To Increase Immunogenicity By T Cell Epitope Engineering

The current COVID-19 pandemic caused by SARS-CoV-2 has resulted in millions of confirmed cases and thousands of deaths globally. Extensive efforts and progress have been made to develop effective and safe vaccines against COVID-19. A primary target of these vaccines is the SARS-CoV-2 spike (S) protein, and many studies utilized structural vaccinology techniques to either stabilize the protein or fix the receptor-binding domain at certain states. In this study, we extended an evolutionary protein design algorithm, EvoDesign, to create thousands of stable S protein variants without perturbing the surface conformation and B cell epitopes of the S protein. We then evaluated the mutated S protein candidates based on predicted MHC-II T cell promiscuous epitopes as well as the epitopes’ similarity to human peptides. The presented strategy aims to improve the S protein’s immunogenicity and antigenicity by inducing stronger CD4 T cell response while maintaining the protein’s native structure and function. The top EvoDesign S protein candidate (Design-10705) recovered 31 out of 32 MHC-II T cell promiscuous epitopes in the native S protein, in which two epitopes were present in all seven human coronaviruses. This newly designed S protein also introduced nine new MHC-II T cell promiscuous epitopes and showed high structural similarity to its native conformation. The proposed structural vaccinology method provides an avenue to rationally design the antigen’s structure with increased immunogenicity, which could be applied to the rational design of new COVID-19 vaccine candidates.

[end paper 64]

[begin paper 65]

PMCID = 10349087

Title = Structural delineation and computational design of SARS-CoV-2-neutralizing antibodies against Omicron subvariants

SARS-CoV-2 Omicron subvariants have evolved to evade receptor-binding site (RBS) antibodies that exist in diverse individuals as public antibody clones. We rationally selected RBS antibodies resilient to mutations in emerging Omicron subvariants. Y489 was identified as a site of virus vulnerability and a common footprint of broadly neutralizing antibodies against the subvariants. Multiple Y489-binding antibodies were encoded by public clonotypes and additionally recognized F486, potentially accounting for the emergence of Omicron subvariants harboring the F486V mutation. However, a subclass of antibodies broadly neutralized BA.4/BA.5 variants via hydrophobic binding sites of rare clonotypes along with high mutation-resilience under escape mutation screening. A computationally designed antibody based on one of the Y489-binding antibodies, NIV-10/FD03, was able to bind XBB with any 486 mutation and neutralized XBB.1.5. The structural basis for the mutation-resilience of this Y489-binding antibody group may provide important insights into the design of therapeutics resistant to viral escape.

In this study, the authors isolated SARS-CoV-2 receptor binding site monoclonal antibodies resistant to Omicron mutations. An amino acid in the receptor binding domain, tyrosine-489, is a virus-vulnerable site and a common footprint of broadly neutralizing antibodies.

[end paper 65]

[begin paper 66]

PMCID = 9936628

Title = Computational pipeline provides mechanistic understanding of Omicron variant of concern neutralizing engineered ACE2 receptor traps

The SARS-CoV-2 Omicron variant, with 15 mutations in Spike receptor-binding domain (Spike-RBD), renders virtually all clinical monoclonal antibodies against WT SARS-CoV-2 ineffective. We recently engineered the SARS-CoV-2 host entry receptor, ACE2, to tightly bind WT-RBD and prevent viral entry into host cells (“receptor traps”). Here we determine cryo-EM structures of our receptor traps in complex with stabilized Spike ectodomain. We develop a multi-model pipeline combining Rosetta protein modeling software and cryo-EM to allow interface energy calculations even at limited resolution and identify interface side chains that allow for high-affinity interactions between our ACE2 receptor traps and Spike-RBD. Our structural analysis provides a mechanistic rationale for the high-affinity (0.53–4.2 nM) binding of our ACE2 receptor traps to Omicron-RBD confirmed with biolayer interferometry measurements. Finally, we show that ACE2 receptor traps potently neutralize Omicron and Delta pseudotyped viruses, providing alternative therapeutic routes to combat this evolving virus.

With the emergence of new SARS-CoV-2 variants, there is a need for pan-variant Spike binders to inhibit viral entry. Using a novel pipeline combining cryo-EM and Rosetta, Remesh et al. characterize the mechanism of high-affinity interactions of designed ACE2 mimetic binders and show that they neutralize the Omicron variant.

[end paper 66]

[begin paper 67]

PMCID = 8778790

Title = A Comparative and Comprehensive Review of Antibody Applications in the Treatment of Lung Disease

Antibodies are a type of protein produced by active B cells in response to antigen stimulation. A series of monoclonal antibodies and neutralizing antibodies have been invented and put into clinical use because of their high therapeutic effect and bright developing insight. Patients with cancer, infectious diseases, and autoimmune diseases can all benefit from antibody therapy. However, the targeting aspects and potential mechanisms for treating these diseases differ. In the treatment of patients with infectious diseases such as COVID-19, neutralizing antibodies have been proposed as reliable vaccines against COVID-19, which target the ACE2 protein by preventing virus entry into somatic cells. Monoclonal antibodies can target immune checkpoints (e.g., PD-L1 and CTLA-4), tyrosine kinase and subsequent signaling pathways (e.g., VEGF), and cytokines in cancer patients (e.g. IL-6 and IL-1β). It is debatable whether there is any connection between the use of antibodies in these diseases. It would be fantastic to discover the related points and explain the burden for the limitation of cross-use of these techniques. In this review, we provided a comprehensive overview of the use of antibodies in the treatment of infectious disease and cancer patients. There are also discussions of their mechanisms and history. In addition, we discussed our future outlook on the use of antibodies.

[end paper 67]

[begin paper 68]

PMCID = 7820290

Title = The SARS-CoV-2 nucleocapsid phosphoprotein forms mutually exclusive condensates with RNA and the membrane-associated M protein

The multifunctional nucleocapsid (N) protein in SARS-CoV-2 binds the ~30 kb viral RNA genome to aid its packaging into the 80–90 nm membrane-enveloped virion. The N protein is composed of N-terminal RNA-binding and C-terminal dimerization domains that are flanked by three intrinsically disordered regions. Here we demonstrate that the N protein’s central disordered domain drives phase separation with RNA, and that phosphorylation of an adjacent serine/arginine rich region modulates the physical properties of the resulting condensates. In cells, N forms condensates that recruit the stress granule protein G3BP1, highlighting a potential role for N in G3BP1 sequestration and stress granule inhibition. The SARS-CoV-2 membrane (M) protein independently induces N protein phase separation, and three-component mixtures of N + M + RNA form condensates with mutually exclusive compartments containing N + M or N + RNA, including annular structures in which the M protein coats the outside of an N + RNA condensate. These findings support a model in which phase separation of the SARS-CoV-2 N protein contributes both to suppression of the G3BP1-dependent host immune response and to packaging genomic RNA during virion assembly.

The SARS-CoV-2 nucleocapsid (N) protein binds the viral RNA genome and contains two ordered domains flanked by three intrinsically-disordered regions. Here, the authors show that RNA binding induces liquid-liquid phase separation of N, which is driven by its central intrinsically-disordered region and is modulated by phosphorylation. The SARS-CoV-2 Membrane (M) protein also phase-separates with N, and three-component mixtures of N + M + RNA form mutually exclusive compartments containing N + M or N + RNA.

[end paper 68]

[begin paper 69]

PMCID = 7871767

Title = In vivo structural characterization of the SARS-CoV-2 RNA genome identifies host proteins vulnerable to repurposed drugs

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is the cause of the ongoing coronavirus disease 2019 (COVID-19) pandemic. Understanding of the RNA virus and its interactions with host proteins could improve therapeutic interventions for COVID-19. By using icSHAPE, we determined the structural landscape of SARS-CoV-2 RNA in infected human cells and from refolded RNAs, as well as the regulatory untranslated regions of SARS-CoV-2 and six other coronaviruses. We validated several structural elements predicted in silico and discovered structural features that affect the translation and abundance of subgenomic viral RNAs in cells. The structural data informed a deep-learning tool to predict 42 host proteins that bind to SARS-CoV-2 RNA. Strikingly, antisense oligonucleotides targeting the structural elements and FDA-approved drugs inhibiting the SARS-CoV-2 RNA binding proteins dramatically reduced SARS-CoV-2 infection in cells derived from human liver and lung tumors. Our findings thus shed light on coronavirus and reveal multiple candidate therapeutics for COVID-19 treatment.

Sun et al. determined the SARS-CoV-2 RNA genome structure in infected cells and from refolded RNAs, which enabled prediction of 42 host proteins that bind to viral RNA, using a deep-learning tool and identification of FDA-approved drugs for repurposing to reduce SARS-CoV-2 infection in cells.

[end paper 69]

[begin paper 70]

PMCID = 9144871

Title = Structure-Based Development of SARS-CoV-2 Spike Interactors

Coronaviruses, including SARS-CoV-2 (the etiological agent of the current COVID-19 pandemic), rely on the surface spike glycoprotein to access the host cells, mainly through the interaction of their receptor-binding domain (RBD) with the human angiotensin-converting enzyme 2 (ACE2). Therefore, molecular entities able to interfere with the binding of the SARS-CoV-2 spike protein to ACE2 have great potential to inhibit viral entry. Starting from the available structural data on the interaction between SARS-CoV-2 spike protein and the host ACE2 receptor, we engineered a set of soluble and stable spike interactors, here denoted as S-plugs. Starting from the prototype S-plug, we adopted a computational approach by combining stability prediction, associated to single-point mutations, with molecular dynamics to enhance both S-plug thermostability and binding affinity to the spike protein. The best developed molecule, S-plug3, possesses a highly stable α-helical con-formation (with melting temperature Tm of 54 °C) and can interact with the spike RBD and S1 domains with similar low nanomolar affinities. Importantly, S-plug3 exposes the spike RBD to almost the same interface as the human ACE2 receptor, aimed at the recognition of all ACE2-accessing coronaviruses. Consistently, S-plug3 preserves a low nanomolar dissociation constant with the delta B.1.617.2 variant of SARS-CoV-2 spike protein (KD = 29.2 ± 0.6 nM). Taken together, we provide valid starting data for the development of therapeutical and diagnostic tools against coronaviruses accessing through ACE2.

[end paper 70]

[begin paper 71]

PMCID = 8999638

Title = Identification of Entry Inhibitors against Delta and Omicron Variants of SARS-CoV-2

Entry inhibitors against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) are urgently needed to control the outbreak of coronavirus disease 2019 (COVID-19). This study developed a robust and straightforward assay that detected the molecular interaction between the receptor-binding domain (RBD) of viral spike protein and the angiotensin-converting enzyme 2 (ACE2) receptor in just 10 min. A drug library of 1068 approved compounds was used to screen for SARS-CoV2 entry inhibition, and 9 active drugs were identified as specific pseudovirus entry inhibitors. A plaque reduction neutralization test using authentic SARS-CoV-2 virus in Vero E6 cells confirmed that 2 of these drugs (Etravirine and Dolutegravir) significantly inhibited the infection of SARS-CoV-2. With molecular docking, we showed that both Etravirine and Dolutegravir are preferentially bound to primary ACE2-interacting residues on the RBD domain, implying that these two drug blocks may prohibit the viral attachment of SARS-CoV-2. We compared the neutralizing activities of these entry inhibitors against different pseudoviruses carrying spike proteins from alpha, beta, gamma, and delta variants. Both Etravirine and Dolutegravir showed similar neutralizing activities against different variants, with EC50 values between 4.5 to 5.8 nM for Etravirine and 10.2 to 22.9 nM for Dolutegravir. These data implied that Etravirine and Dolutegravir may serve as general spike inhibitors against dominant viral variants of SARS-CoV-2.

[end paper 71]

[begin paper 72]

PMCID = 7799268

Title = The impact of structural bioinformatics tools and resources on SARS-CoV-2 research and therapeutic strategies

SARS-CoV-2 is the causative agent of COVID-19, the ongoing global pandemic. It has posed a worldwide challenge to human health as no effective treatment is currently available to combat the disease. Its severity has led to unprecedented collaborative initiatives for therapeutic solutions against COVID-19. Studies resorting to structure-based drug design for COVID-19 are plethoric and show good promise. Structural biology provides key insights into 3D structures, critical residues/mutations in SARS-CoV-2 proteins, implicated in infectivity, molecular recognition and susceptibility to a broad range of host species. The detailed understanding of viral proteins and their complexes with host receptors and candidate epitope/lead compounds is the key to developing a structure-guided therapeutic design.

Since the discovery of SARS-CoV-2, several structures of its proteins have been determined experimentally at an unprecedented speed and deposited in the Protein Data Bank. Further, specialized structural bioinformatics tools and resources have been developed for theoretical models, data on protein dynamics from computer simulations, impact of variants/mutations and molecular therapeutics.

Here, we provide an overview of ongoing efforts on developing structural bioinformatics tools and resources for COVID-19 research. We also discuss the impact of these resources and structure-based studies, to understand various aspects of SARS-CoV-2 infection and therapeutic development. These include (i) understanding differences between SARS-CoV-2 and SARS-CoV, leading to increased infectivity of SARS-CoV-2, (ii) deciphering key residues in the SARS-CoV-2 involved in receptor–antibody recognition, (iii) analysis of variants in host proteins that affect host susceptibility to infection and (iv) analyses facilitating structure-based drug and vaccine design against SARS-CoV-2.

[end paper 72]

[begin paper 73]

PMCID = 8016453

Title = The UCSC SARS-CoV-2 Genome Browser

Researchers are generating molecular data pertaining to the SARS-CoV-2 RNA genome and its proteins at an unprecedented rate during the COVID-19 pandemic. As a result, there is a critical need for rapid and continuously updated access to the latest molecular data in a format in which all data can be quickly cross-referenced and compared. We adapted our genome browser visualization tool to the viral genome for this purpose. Molecular data, curated from published studies or from database submissions, are mapped to the viral genome and grouped together into “annotation tracks” where they can be visualized along the linear map of the viral genome sequence and programmatically downloaded in standard format for analysis.

The UCSC Genome Browser for SARS-CoV-2 (https://genome.ucsc.edu/covid19.html) provides continuously updated access to the mutations in the many thousands of SARS-CoV-2 genomes deposited in GISAID and the international nucleotide sequencing databases, displayed alongside phylogenetic trees. These data are augmented with alignments of bat, pangolin, and other animal and human coronavirus genomes, including per-base evolutionary rate analysis. All available annotations are cross-referenced on the virus genome, including those from major databases (PDB, RFAM, IEDB, UniProt) as well as up-to-date individual results from preprints. Annotated data include predicted and validated immune epitopes, promising antibodies, RT-PCR and sequencing primers, CRISPR guides (from research, diagnostics, vaccines, and therapies), and points of interaction between human and viral genes. As a community resource, any user can add manual annotations which are quality checked and shared publicly on the browser the next day.

We invite all investigators to contribute additional data and annotations to this resource to accelerate research and development activities globally. Contact us at genome-www@soe.ucsc.edu with data suggestions or requests for support for adding data. Rapid sharing of data will accelerate SARS-CoV-2 research, especially when researchers take time to integrate their data with those from other labs on a widely-used community browser platform with standardized machine-readable data formats, such as the SARS-CoV-2 Genome Browser.

[end paper 73]

[begin paper 74]

PMCID = 10689493

Title = Engineered immunogens to elicit antibodies against conserved coronavirus epitopes

Immune responses to SARS-CoV-2 primarily target the receptor binding domain of the spike protein, which continually mutates to escape acquired immunity. Other regions in the spike S2 subunit, such as the stem helix and the segment encompassing residues 815-823 adjacent to the fusion peptide, are highly conserved across sarbecoviruses and are recognized by broadly reactive antibodies, providing hope that vaccines targeting these epitopes could offer protection against both current and emergent viruses. Here we employ computational modeling to design scaffolded immunogens that display the spike 815-823 peptide and the stem helix epitopes without the distracting and immunodominant receptor binding domain. These engineered proteins bind with high affinity and specificity to the mature and germline versions of previously identified broadly protective human antibodies. Epitope scaffolds interact with both sera and isolated monoclonal antibodies with broadly reactivity from individuals with pre-existing SARS-CoV-2 immunity. When used as immunogens, epitope scaffolds elicit sera with broad betacoronavirus reactivity and protect as “boosts” against live virus challenge in mice, illustrating their potential as components of a future pancoronavirus vaccine.

A pan-betacoronavirus vaccine will likely require the elicitation of antibodies against spike regions conserved across diverse coronaviruses. Here, authors computationally engineer and experimentally validate immunogens to elicit antibodies against two such spike regions.

[end paper 74]

[begin paper 75]

PMCID = 7402059

Title = Structure and inhibition of the SARS-CoV-2 main protease reveals strategy for developing dual inhibitors against Mpro and cathepsin L

The main protease (Mpro) of SARS-CoV-2, the pathogen responsible for the COVID-19 pandemic, is a key antiviral drug target. While most SARS-CoV-2 Mpro inhibitors have a γ-lactam glutamine surrogate at the P1 position, we recently discovered several Mpro inhibitors have hydrophobic moieties at the P1 site, including calpain inhibitors II/XII, which are also active against human cathepsin L, a host-protease that is important for viral entry. To determine the binding mode of these calpain inhibitors and establish a structure-activity relationship, we solved X-ray crystal structures of Mpro in complex with calpain inhibitors II and XII, and three analogues of GC-376, one of the most potent Mpro inhibitors in vitro. The structure of Mpro with calpain inhibitor II confirmed the S1 pocket of Mpro can accommodate a hydrophobic methionine side chain, challenging the idea that a hydrophilic residue is necessary at this position. Interestingly, the structure of calpain inhibitor XII revealed an unexpected, inverted binding pose where the P1’ pyridine inserts in the S1 pocket and the P1 norvaline is positioned in the S1’ pocket. The overall conformation is semi-helical, wrapping around the catalytic core, in contrast to the extended conformation of other peptidomimetic inhibitors. Additionally, the structures of three GC-376 analogues UAWJ246, UAWJ247, and UAWJ248 provide insight to the sidechain preference of the S1’, S2, S3 and S4 pockets, and the superior cell-based activity of the aldehyde warhead compared with the α-ketoamide. Taken together, the biochemical, computational, structural, and cellular data presented herein provide new directions for the development of Mpro inhibitors as SARS-CoV-2 antivirals.

[end paper 75]

[begin paper 76]

PMCID = 7576348

Title = Sequence Analysis and Structure Prediction of SARS-CoV-2 Accessory Proteins 9b and ORF14: Evolutionary Analysis Indicates Close Relatedness to Bat Coronavirus

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has a single-stranded RNA genome that encodes 14 open reading frames (ORFs), eight of which encode accessory proteins that allow the virus to infect the host and promote virulence. The genome expresses around 29 structural and nonstructural protein products. The accessory proteins of SARS-CoV-2 are not essential for virus replication but do affect viral release, stability, and pathogenesis and finally contribute to virulence. This paper has attempted the structure prediction and functional analysis of two such accessory proteins, 9b and ORF14, in the absence of experimental structures. Sequence analysis, structure prediction, functional characterization, and evolutionary analysis based on the UniProtKB reviewed the amino acid sequences of SARS-CoV-2 9b (P0DTD2) and ORF14 (P0DTD3) proteins. Modeling has been presented with the introduction of hybrid comparative and ab initio modeling. QMEANDisCo 4.0.0 and ProQ3 for global and local (per residue) quality estimates verified the structures as high quality, which may be attributed to structure-based drug design targets. Tunnel analysis revealed the presence of 1-2 highly active tunneling sites, perhaps which will able to provide certain inputs for advanced structure-based drug design or to formulate potential vaccines in the absence of a complete experimental structure. The evolutionary analysis of both proteins of human SARS-CoV-2 indicates close relatedness to the bat coronavirus. The whole-genome phylogeny indicates that only the new bat coronavirus followed by pangolin coronaviruses has a close evolutionary relationship with the novel SARS-CoV-2.

[end paper 76]

[begin paper 77]

PMCID = 10371346

Title = Modeling of ACE2 and antibodies bound to SARS-CoV-2 provides insights into infectivity and immune evasion

Given the COVID-19 pandemic, there is interest in understanding ligand-receptor features and targeted antibody-binding attributes against emerging SARS-CoV-2 variants. Here, we developed a large-scale structure-based pipeline for analysis of protein-protein interactions regulating SARS-CoV-2 immune evasion. First, we generated computed structural models of the Spike protein of 3 SARS-CoV-2 variants (B.1.1.529, BA.2.12.1, and BA.5) bound either to a native receptor (ACE2) or to a large panel of targeted ligands (n = 282), which included neutralizing or therapeutic monoclonal antibodies. Moreover, by using the Barnes classification, we noted an overall loss of interfacial interactions (with gain of new interactions in certain cases) at the receptor-binding domain (RBD) mediated by substituted residues for neutralizing complexes in classes 1 and 2, whereas less destabilization was observed for classes 3 and 4. Finally, an experimental validation of predicted weakened therapeutic antibody binding was performed in a cell-based assay. Compared with the original Omicron variant (B.1.1.529), derivative variants featured progressive destabilization of antibody-RBD interfaces mediated by a larger set of substituted residues, thereby providing a molecular basis for immune evasion. This approach and findings provide a framework for rapidly and efficiently generating structural models for SARS-CoV-2 variants bound to ligands of mechanistic and therapeutic value.

[end paper 77]

[begin paper 78]

PMCID = 8286823

Title = Identifying SARS-CoV-2 antiviral compounds by screening for small molecule inhibitors of nsp15 endoribonuclease

SARS-CoV-2 is responsible for COVID-19, a human disease that has caused over 2 million deaths, stretched health systems to near-breaking point and endangered economies of countries and families around the world. Antiviral treatments to combat COVID-19 are currently lacking. Remdesivir, the only antiviral drug approved for the treatment of COVID-19, can affect disease severity, but better treatments are needed. SARS-CoV-2 encodes 16 non-structural proteins (nsp) that possess different enzymatic activities with important roles in viral genome replication, transcription and host immune evasion. One key aspect of host immune evasion is performed by the uridine-directed endoribonuclease activity of nsp15. Here we describe the expression and purification of nsp15 recombinant protein. We have developed biochemical assays to follow its activity, and we have found evidence for allosteric behaviour. We screened a custom chemical library of over 5000 compounds to identify nsp15 endoribonuclease inhibitors, and we identified and validated NSC95397 as an inhibitor of nsp15 endoribonuclease in vitro. Although NSC95397 did not inhibit SARS-CoV-2 growth in VERO E6 cells, further studies will be required to determine the effect of nsp15 inhibition on host immune evasion.

[end paper 78]

[begin paper 79]

PMCID = 11222146

Title = Human SARS-CoV-2 challenge uncovers local and systemic response dynamics

The COVID-19 pandemic is an ongoing global health threat, yet our understanding of the dynamics of early cellular responses to this disease remains limited. Here in our SARS-CoV-2 human challenge study, we used single-cell multi-omics profiling of nasopharyngeal swabs and blood to temporally resolve abortive, transient and sustained infections in seronegative individuals challenged with pre-Alpha SARS-CoV-2. Our analyses revealed rapid changes in cell-type proportions and dozens of highly dynamic cellular response states in epithelial and immune cells associated with specific time points and infection status. We observed that the interferon response in blood preceded the nasopharyngeal response. Moreover, nasopharyngeal immune infiltration occurred early in samples from individuals with only transient infection and later in samples from individuals with sustained infection. High expression of HLA-DQA2 before inoculation was associated with preventing sustained infection. Ciliated cells showed multiple immune responses and were most permissive for viral replication, whereas nasopharyngeal T cells and macrophages were infected non-productively. We resolved 54 T cell states, including acutely activated T cells that clonally expanded while carrying convergent SARS-CoV-2 motifs. Our new computational pipeline Cell2TCR identifies activated antigen-responding T cells based on a gene expression signature and clusters these into clonotype groups and motifs. Overall, our detailed time series data can serve as a Rosetta stone for epithelial and immune cell responses and reveals early dynamic responses associated with protection against infection.

A human SARS-CoV-2 challenge study in individuals without previous exposure to the virus or vaccines provides detailed profiles of local and systemic epithelial and immune cell response dynamics over time and infection status.

[end paper 79]

[begin paper 80]

PMCID = 8276696

Title = Stabilization of the SARS-CoV-2 Spike Receptor-Binding Domain Using Deep Mutational Scanning and Structure-Based Design

The unprecedented global demand for SARS-CoV-2 vaccines has demonstrated the need for highly effective vaccine candidates that are thermostable and amenable to large-scale manufacturing. Nanoparticle immunogens presenting the receptor-binding domain (RBD) of the SARS-CoV-2 Spike protein (S) in repetitive arrays are being advanced as second-generation vaccine candidates, as they feature robust manufacturing characteristics and have shown promising immunogenicity in preclinical models. Here, we used previously reported deep mutational scanning (DMS) data to guide the design of stabilized variants of the RBD. The selected mutations fill a cavity in the RBD that has been identified as a linoleic acid binding pocket. Screening of several designs led to the selection of two lead candidates that expressed at higher yields than the wild-type RBD. These stabilized RBDs possess enhanced thermal stability and resistance to aggregation, particularly when incorporated into an icosahedral nanoparticle immunogen that maintained its integrity and antigenicity for 28 days at 35-40°C, while corresponding immunogens displaying the wild-type RBD experienced aggregation and loss of antigenicity. The stabilized immunogens preserved the potent immunogenicity of the original nanoparticle immunogen, which is currently being evaluated in a Phase I/II clinical trial. Our findings may improve the scalability and stability of RBD-based coronavirus vaccines in any format and more generally highlight the utility of comprehensive DMS data in guiding vaccine design.

[end paper 80]

[begin paper 81]

PMCID = 8929045

Title = A Comparative Study between Spanish and British SARS-CoV-2 Variants

The study of the interaction between the SARS-CoV-2 spike protein and the angiotensin-converting enzyme 2 (ACE2) receptor is key to understanding binding affinity and stability. In the present report, we sought to investigate the differences between two already sequenced genome variants (Spanish and British) of SARS-CoV-2. Methods: In silico model evaluating the homology, identity and similarity in the genome sequence and the structure and alignment of the predictive spike by computational docking methods. Results: The identity results between the Spanish and British variants of the Spike protein were 28.67%. This close correspondence in the results between the Spanish and British SARS-CoV-2 variants shows that they are very similar (99.99%). The alignment obtained results in four deletions. There were 23 nucleotide substitutions also predicted which could affect the functionality of the proteins produced from this sequence. The interaction between the binding receptor domain from the spike protein and the ACE2 receptor produces some of the mutations found and, therefore, the energy of this ligand varies. However, the estimated antigenicity of the British variant is higher than its Spanish counterpart. Conclusions: Our results indicate that minimal mutations could interfere in the infectivity of the virus due to changes in the fitness between host cell recognition and interaction proteins. In particular, the N501Y substitution, situated in the RBD of the spike of the British variant, might be the reason for its extraordinary infective potential.

[end paper 81]

[begin paper 82]

PMCID = 10246912

Title = Structural and computational design of a SARS-CoV-2 spike antigen with improved expression and immunogenicity

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants of concern challenge the efficacy of approved vaccines, emphasizing the need for updated spike antigens. Here, we use an evolutionary-based design aimed at boosting protein expression levels of S-2P and improving immunogenic outcomes in mice. Thirty-six prototype antigens were generated in silico and 15 were produced for biochemical analysis. S2D14, which contains 20 computationally designed mutations within the S2 domain and a rationally engineered D614G mutation in the SD2 domain, has an ~11-fold increase in protein yield and retains RBD antigenicity. Cryo–electron microscopy structures reveal a mixture of populations in various RBD conformational states. Vaccination of mice with adjuvanted S2D14 elicited higher cross-neutralizing antibody titers than adjuvanted S-2P against the SARS-CoV-2 Wuhan strain and four variants of concern. S2D14 may be a useful scaffold or tool for the design of future coronavirus vaccines, and the approaches used for the design of S2D14 may be broadly applicable to streamline vaccine discovery.

A spike antigen is designed with high expression that elicits a nAb response in mice effective against the Wuhan strain and VoCs.

[end paper 82]

[begin paper 83]

PMCID = 7940615

Title = The in-vitro effect of famotidine on sars-cov-2 proteases and virus replication

The lack of coronavirus-specific antiviral drugs has instigated multiple drug repurposing studies to redirect previously approved medicines for the treatment of SARS-CoV-2, the coronavirus behind the ongoing COVID-19 pandemic. A recent, large-scale, retrospective clinical study showed that famotidine, when administered at a high dose to hospitalized COVID-19 patients, reduced the rates of intubation and mortality. A separate, patient-reported study associated famotidine use with improvements in mild to moderate symptoms such as cough and shortness of breath. While a prospective, multi-center clinical study is ongoing, two parallel in silico studies have proposed one of the two SARS-CoV-2 proteases, 3CLpro or PLpro, as potential molecular targets of famotidine activity; however, this remains to be experimentally validated. In this report, we systematically analyzed the effect of famotidine on viral proteases and virus replication. Leveraging a series of biophysical and enzymatic assays, we show that famotidine neither binds with nor inhibits the functions of 3CLpro and PLpro. Similarly, no direct antiviral activity of famotidine was observed at concentrations of up to 200 µM, when tested against SARS-CoV-2 in two different cell lines, including a human cell line originating from lungs, a primary target of COVID-19. These results rule out famotidine as a direct-acting inhibitor of SARS-CoV-2 replication and warrant further investigation of its molecular mechanism of action in the context of COVID-19.

[end paper 83]

[begin paper 84]

PMCID = 9269947

Title = Ig-VAE: Generative modeling of protein structure by direct 3D coordinate generation

While deep learning models have seen increasing applications in protein science, few have been implemented for protein backbone generation—an important task in structure-based problems such as active site and interface design. We present a new approach to building class-specific backbones, using a variational auto-encoder to directly generate the 3D coordinates of immunoglobulins. Our model is torsion- and distance-aware, learns a high-resolution embedding of the dataset, and generates novel, high-quality structures compatible with existing design tools. We show that the Ig-VAE can be used with Rosetta to create a computational model of a SARS-CoV2-RBD binder via latent space sampling. We further demonstrate that the model’s generative prior is a powerful tool for guiding computational protein design, motivating a new paradigm under which backbone design is solved as constrained optimization problem in the latent space of a generative model.

Many essential biochemical processes are governed by protein-protein interactions (PPIs), and our ability to make binding proteins that modulate PPIs is crucial to the creation of therapeutics and the study of cell-signaling. One critical aspect of PPI design is to capture protein conformational flexibility. Deep generative models are a class of mathematical models that are able to synthesize novel data from a finite set of training examples. Here, we make advances in computational protein design methodology by developing a deep generative model that creates protein backbones adopting the immunoglobulin fold, which is found in natural binding proteins such as antibodies. While generative models have been powerful in tasks such as image generation, using them to create proteins has remained a challenge. We solve this problem with a new model that allows for the direct generation of novel 3D molecules and show that they are of high chemical accuracy. Generated structures work well with existing protein design methods such as Rosetta, providing access to a large collection of novel immunoglobulin structures. Finally, we present a new protein design framework, called “generative design,” that shows how deep generative models such as ours can be applied to virtually any protein design problem.

[end paper 84]

[begin paper 85]

PMCID = 9333103

Title = Computer-aided discovery, design, and investigation of COVID-19 therapeutics

Coronavirus disease 2019 (COVID-19) pandemic is currently the most serious public health threat faced by mankind. Thus, the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), which causes COVID-19, is being intensively investigated. Several vaccines are now available for clinical use. However, owing to the highly mutated nature of RNA viruses, the SARS-CoV-2 is changing at a rapid speed. Breakthrough infections by SARS-CoV-2 variants have been seen in vaccinated individuals. As a result, effective therapeutics for treating COVID-19 patients is urgently required. With the advance of computer technology, computational methods have become increasingly powerful in the biomedical research and pharmaceutical drug discovery. The applications of these techniques have largely reduced the costs and simplified processes of pharmaceutical drug developments. Intensive and extensive studies on SARS-CoV-2 proteins have been carried out and three-dimensional structures of the major SARS-CoV-2 proteins have been resolved and deposited in the Protein Data Bank. These structures provide the foundations for drug discovery and design using the structure-based computations, such as molecular docking and molecular dynamics simulations. In this review, introduction to the applications of computational methods in the discovery and design of novel drugs and repurposing of existing drugs for the treatments of COVID-19 is given. The examples of computer-aided investigations and screening of COVID-19 effective therapeutic compounds, functional peptides, as well as effective molecules from the herb medicines are discussed.

[end paper 85]

[begin paper 86]

PMCID = 9889117

Title = Rational approaches to discover SARS-CoV-2/ACE2 interaction inhibitors: Pharmacophore-based virtual screening, molecular docking, molecular dynamics and binding free energy studies

The lack of effective treatment remains a bottleneck in combating the current coronavirus family pandemic, particularly coronavirus 2019 (COVID-19), which is caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The infection of host cells by SARS-CoV-2 is mediated by the binding of its receptor-binding domain (RBD) on the spike (S) glycoprotein to the host angiotensin-converting enzyme (ACE2) receptor. As all developed and available vaccines against COVID-19 do not provide long-term immunity, the creation of an effective drug for the treatment of COVID-19 is necessary and cannot be ignored. Therefore, the aim of this study is to present a computational screening method to identify potential inhibitor candidates with a high probability of blocking the binding of RBD to the ACE2 receptor. Pharmacophore mapping, molecular docking, molecular dynamics (MD) simulations, and binding free-energy analyses were performed to identify potential inhibitor candidates against ACE2/SARS-CoV-2. In conclusion, we propose the compound PubChem-84280085 as a potential inhibitor of protein–protein interactions to disrupt the binding of the SARS-CoV-2-RBD to the ACE2 receptor.

[end paper 86]

[begin paper 87]

PMCID = 7665879

Title = Peptide and peptide-based inhibitors of SARS-CoV-2 entry

To date, no effective vaccines or therapies are available against the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative pandemic agent of the coronavirus disease 2019 (COVID-19). Due to their safety, efficacy and specificity, peptide inhibitors hold great promise for the treatment of newly emerging viral pathogens. Based on the known structures of viral proteins and their cellular targets, antiviral peptides can be rationally designed and optimized. The resulting peptides may be highly specific for their respective targets and particular viral pathogens or exert broad antiviral activity. Here, we summarize the current status of peptides inhibiting SARS-CoV-2 entry and outline the strategies used to design peptides targeting the ACE2 receptor or the viral spike protein and its activating proteases furin, transmembrane serine protease 2 (TMPRSS2), or cathepsin L. In addition, we present approaches used against related viruses such as SARS-CoV-1 that might be implemented for inhibition of SARS-CoV-2 infection.

[end paper 87]

[begin paper 88]

PMCID = 8953562

Title = E-Volve: understanding the impact of mutations in SARS-CoV-2 variants spike protein on antibodies and ACE2 affinity through patterns of chemical interactions at protein interfaces

The SARS-CoV-2 pandemic reverberated, posing health and social hygiene obstacles throughout the globe. Mutant lineages of the virus have concerned scientists because of convergent amino acid alterations, mainly on the viral spike protein. Studies have shown that mutants have diminished activity of neutralizing antibodies and enhanced affinity with its human cell receptor, the ACE2 protein.

Hence, for real-time measuring of the impacts caused by variant strains in such complexes, we implemented E-Volve, a tool designed to model a structure with a list of mutations requested by users and return analyses of the variant protein. As a proof of concept, we scrutinized the spike-antibody and spike-ACE2 complexes formed in the variants of concern, B.1.1.7 (Alpha), B.1.351 (Beta), and P.1 (Gamma), by using contact maps depicting the interactions made amid them, along with heat maps to quantify these major interactions.

The results found in this study depict the highly frequent interface changes made by the entire set of mutations, mainly conducted by N501Y and E484K. In the spike-Antibody complex, we have noticed alterations concerning electrostatic surface complementarity, breaching essential sites in the P17 and BD-368-2 antibodies. Alongside, the spike-ACE2 complex has presented new hydrophobic bonds.

Molecular dynamics simulations followed by Poisson-Boltzmann calculations corroborate the higher complementarity to the receptor and lower to the antibodies for the K417T/E484K/N501Y (Gamma) mutant compared to the wild-type strain, as pointed by E-Volve, as well as an intensification of this effect by changes at the protein conformational equilibrium in solution. A local disorder of the loop α1′/β1′, as well its possible effects on the affinity to the BD-368-2 antibody were also incorporated to the final conclusions after this analysis. Moreover, E-Volve can depict the main alterations in important biological structures, as shown in the SARS-CoV-2 complexes, marking a major step in the real-time tracking of the virus mutant lineages. E-Volve is available at .

[end paper 88]

[begin paper 89]

PMCID = 8931700

Title = Systematic Tracing of Susceptible Animals to SARS-CoV-2 by a Bioinformatics Framework

Since the outbreak of SARS-CoV-2 in 2019, the Chinese horseshoe bats were considered as a potential original host of SARS-CoV-2. In addition, cats, tigers, lions, mints, and ferrets were naturally or experimentally infected with SARS-CoV-2. For the surveillance and control of this highly infectious disease, it is critical to trace susceptible animals and predict the consequence of potential mutations at the binding region of viral spike protein and host ACE2 protein. This study proposed a novel bioinformatics framework to systematically trace susceptible animals to SARS-CoV-2 and predict the binding affinity between susceptible animals’ mutated/un-mutated ACE2 receptors. As a result, we identified a few animals posing a potential risk of infection with SARS-CoV-2 using the docking analysis of ACE2 protein and viral spike protein. The binding affinity of some of these species is weaker than that of humans but more potent than that of Chinese horseshoe bats. We also found that a few point mutations in human ACE2 protein or viral spike protein could significantly enhance their binding affinity, posing an enormous potential threat to public health. The ancestors of the Omicron may evolve rapidly through the accumulation of mutations in infecting the host and jumped into human beings. These findings indicate that if the epidemic expands, there may be a human-animal-human transmission route, which will increase the difficulty of disease prevention and control.

[end paper 89]

[begin paper 90]

PMCID = 8998791

Title = VHH Structural Modelling Approaches: A Critical Review

VHH, i.e., VH domains of camelid single-chain antibodies, are very promising therapeutic agents due to their significant physicochemical advantages compared to classical mammalian antibodies. The number of experimentally solved VHH structures has significantly improved recently, which is of great help, because it offers the ability to directly work on 3D structures to humanise or improve them. Unfortunately, most VHHs do not have 3D structures. Thus, it is essential to find alternative ways to get structural information. The methods of structure prediction from the primary amino acid sequence appear essential to bypass this limitation. This review presents the most extensive overview of structure prediction methods applied for the 3D modelling of a given VHH sequence (a total of 21). Besides the historical overview, it aims at showing how model software programs have been shaping the structural predictions of VHHs. A brief explanation of each methodology is supplied, and pertinent examples of their usage are provided. Finally, we present a structure prediction case study of a recently solved VHH structure. According to some recent studies and the present analysis, AlphaFold 2 and NanoNet appear to be the best tools to predict a structural model of VHH from its sequence.

[end paper 90]

[begin paper 91]

PMCID = 9958328

Title = Global landscape of SARS-CoV-2 mutations and conserved regions

At the end of December 2019, a novel strain of Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) disease (COVID-19) has been identified in Wuhan, a central city in China, and then spread to every corner of the globe. As of October 8, 2022, the total number of COVID-19 cases had reached over 621 million worldwide, with more than 6.56 million confirmed deaths. Since SARS-CoV-2 genome sequences change due to mutation and recombination, it is pivotal to surveil emerging variants and monitor changes for improving pandemic management.

10,287,271 SARS-CoV-2 genome sequence samples were downloaded in FASTA format from the GISAID databases from February 24, 2020, to April 2022. Python programming language (version 3.8.0) software was utilized to process FASTA files to identify variants and sequence conservation. The NCBI RefSeq SARS-CoV-2 genome (accession no. NC_045512.2) was considered as the reference sequence.

Six mutations had more than 50% frequency in global SARS-CoV-2. These mutations include the P323L (99.3%) in NSP12, D614G (97.6) in S, the T492I (70.4) in NSP4, R203M (62.8%) in N, T60A (61.4%) in Orf9b, and P1228L (50.0%) in NSP3. In the SARS-CoV-2 genome, no mutation was observed in more than 90% of nsp11, nsp7, nsp10, nsp9, nsp8, and nsp16 regions. On the other hand, N, nsp3, S, nsp4, nsp12, and M had the maximum rate of mutations. In the S protein, the highest mutation frequency was observed in aa 508–635(0.77%) and aa 381–508 (0.43%). The highest frequency of mutation was observed in aa 66–88 (2.19%), aa 7–14, and aa 164–246 (2.92%) in M, E, and N proteins, respectively.

Therefore, monitoring SARS-CoV-2 proteomic changes and detecting hot spots mutations and conserved regions could be applied to improve the SARS‐CoV‐2 diagnostic efficiency and design safe and effective vaccines against emerging variants.

The online version contains supplementary material available at 10.1186/s12967-023-03996-w.

[end paper 91]

[begin paper 92]

PMCID = 10373554

Title = Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening

Members of the Sarbecovirus subgenus of Coronaviridae have twice caused deadly threats to humans. There is increasing concern about the rapid mutation of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), which has evolved into multiple generations of epidemic variants in 3 years. Broad neutralizing antibodies are of great importance for pandemic preparedness against SARS-CoV-2 variants and divergent zoonotic sarbecoviruses. Here, we analyzed the structural conservation of the receptor-binding domain (RBD) from representative sarbecoviruses and chose S2H97, a previously reported RBD antibody with ideal breadth and resistance to escape, as a template for computational design to enhance the neutralization activity and spectrum. A total of 35 designs were purified for evaluation. The neutralizing activity of a large proportion of these designs against multiple variants was increased from several to hundreds of times. Molecular dynamics simulation suggested that extra interface contacts and enhanced intermolecular interactions between the RBD and the designed antibodies are established. After light and heavy chain reconstitution, AI-1028, with five complementarity determining regions optimized, showed the best neutralizing activity across all tested sarbecoviruses, including SARS-CoV, multiple SARS-CoV-2 variants, and bat-derived viruses. AI-1028 recognized the same cryptic RBD epitope as the parental prototype antibody. In addition to computational design, chemically synthesized nanobody libraries are also a precious resource for rapid antibody development. By applying distinct RBDs as baits for reciprocal screening, we identified two novel nanobodies with broad activities. These findings provide potential pan-sarbecovirus neutralizing drugs and highlight new pathways to rapidly optimize therapeutic candidates when novel SARS-CoV-2 escape variants or new zoonotic coronaviruses emerge.

IMPORTANCE The subgenus Sarbecovirus includes human SARS-CoV, SARS-CoV-2, and hundreds of genetically related bat viruses. The continuous evolution of SARS-CoV-2 has led to the striking evasion of neutralizing antibody (NAb) drugs and convalescent plasma. Antibodies with broad activity across sarbecoviruses would be helpful to combat current SARS-CoV-2 mutations and longer term animal virus spillovers. The study of pan-sarbecovirus NAbs described here is significant for the following reasons. First, we established a structure-based computational pipeline to design and optimize NAbs to obtain more potent and broader neutralizing activity across multiple sarbecoviruses. Second, we screened and identified nanobodies from a highly diversified synthetic library with a broad neutralizing spectrum using an elaborate screening strategy. These methodologies provide guidance for the rapid development of antibody therapeutics against emerging pathogens with highly variable characteristics.

[end paper 92]

[begin paper 93]

PMCID = 9683856

Title = Structural evolution of Delta lineage of SARS-CoV-2

One of the main obstacles in prevention and treatment of COVID-19 is the rapid evolution of the SARS-CoV-2 Spike protein. Given that Spike is the main target of common treatments of COVID-19, mutations occurring at this virulent factor can affect the effectiveness of treatments. The B.1.617.2 lineage of SARS-CoV-2, being characterized by many Spike mutations inside and outside of its receptor-binding domain (RBD), shows high infectivity and relative resistance to existing cures. Here, utilizing a wide range of computational biology approaches, such as immunoinformatics, molecular dynamics (MD), analysis of intrinsically disordered regions (IDRs), protein-protein interaction analyses, residue scanning, and free energy calculations, we examine the structural and biological attributes of the B.1.617.2 Spike protein. Furthermore, the antibody design protocol of Rosetta was implemented for evaluation the stability and affinity improvement of the Bamlanivimab (LY-CoV55) antibody, which is not capable of interactions with the B.1.617.2 Spike. We observed that the detected mutations in the Spike of the B1.617.2 variant of concern can cause extensive structural changes compatible with the described variation in immunogenicity, secondary and tertiary structure, oligomerization potency, Furin cleavability, and drug targetability. Compared to the Spike of Wuhan lineage, the B.1.617.2 Spike is more stable and binds to the Angiotensin-converting enzyme 2 (ACE2) with higher affinity.

[end paper 93]

[begin paper 94]

PMCID = 8661935

Title = Evolution of the SARS-CoV-2 proteome in three dimensions (3D) during the first six months of the COVID-19 pandemic

Understanding the molecular evolution of the SARS-CoV-2 virus as it continues to spread in communities around the globe is important for mitigation and future pandemic preparedness. Three-dimensional structures of SARS-CoV-2 and other coronaviral proteins archived in the Protein Data Bank were used to analyze viral proteome evolution during the first six months of the COVID-19 pandemic. Analyses of spatial locations, chemical properties, and structural and energetic impacts of the observed amino acid changes in >48,000 viral isolates showed how each one of the 29 viral proteins have undergone amino acid changes. Catalytic residues in active sites and binding residues in protein-protein interfaces showed modest, but significant, numbers of substitutions, highlighting the mutational robustness of the viral proteome. Energetics calculations showed that the impact of substitutions on the thermodynamic stability of the proteome follows a universal bi-Gaussian distribution. Detailed results are presented for potential drug discovery targets and the four structural proteins that comprise the virion, highlighting substitutions with the potential to impact protein structure, enzyme activity, and protein-protein and protein-nucleic acid interfaces. Characterizing the evolution of the virus in three dimensions provides testable insights into viral protein function and should aid in structure-based drug discovery efforts as well as the prospective identification of amino acid substitutions with potential for drug resistance.

[end paper 94]

[begin paper 95]

PMCID = 9750863

Title = The trispecific DARPin ensovibep inhibits diverse SARS-CoV-2 variants

The emergence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants with potential resistance to existing drugs emphasizes the need for new therapeutic modalities with broad variant activity. Here we show that ensovibep, a trispecific DARPin (designed ankyrin repeat protein) clinical candidate, can engage the three units of the spike protein trimer of SARS-CoV-2 and inhibit ACE2 binding with high potency, as revealed by cryo-electron microscopy analysis. The cooperative binding together with the complementarity of the three DARPin modules enable ensovibep to inhibit frequent SARS-CoV-2 variants, including Omicron sublineages BA.1 and BA.2. In Roborovski dwarf hamsters infected with SARS-CoV-2, ensovibep reduced fatality similarly to a standard-of-care monoclonal antibody (mAb) cocktail. When used as a single agent in viral passaging experiments in vitro, ensovibep reduced the emergence of escape mutations in a similar fashion to the same mAb cocktail. These results support further clinical evaluation of ensovibep as a broad variant alternative to existing targeted therapies for Coronavirus Disease 2019 (COVID-19).

Multiple variants of SARS-CoV-2 are inhibited by a trispecific DARPin.

[end paper 95]

[begin paper 96]

PMCID = 8283670

Title = Seq12, Seq12m, and Seq13m, peptide analogues of the spike glycoprotein shows antiviral properties against SARS-CoV-2: An in silico study through molecular docking, molecular dynamics simulation, and MM-PB/GBSA calculations

At the very beginning of the new decade, the COVID-19 pandemic has badly hit modern human societies. SARS-CoV-2, the causative agent of COVID-19 acquiring mutations and circulating as new variants. Herein, we have found three new antiviral peptides (AVPs) against the SARS-CoV-2. These AVPs are analogous to the spike glycoprotein of the SARS-CoV-2. Antiviral peptides, i.e., Seq12, Seq12m, and Seq13m, can block the receptor-binding domain (RBD) of the SARS-CoV-2, which is necessary for communicating with the angiotensin-converting enzyme 2 (ACE2). Also, these AVPs sustain their antiviral properties, even after the insertion of 25 mutations in the RBD (Rosetta and FoldX based). Further, Seq12 and Seq12m showed negligible cytotoxicity. Besides, the binding free energies calculated using MM-PB/GBSA method are also in agreement with the molecular docking studies. The molecular interactions between AVPs and the viral membrane protein (M) also showed a favorable interaction suggesting it could inhibit the viral re-packaging process. In conclusion, this study suggests Seq12, Seq12m, and Seq13m could be helpful to fight against SARS-CoV-2. These AVPs could also aid virus diagnostic tools and nasal spray against SARS-CoV-2 in the future.

[end paper 96]

[begin paper 97]

PMCID = 11193732

Title = Development of a single-chain variable antibody fragment against a conserved region of the SARS-CoV-2 spike protein

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has prolonged the duration of the pandemic because of the continuous emergence of new variant strains. The emergence of these mutant strains makes it difficult to detect the virus with the existing antibodies; thus, the development of novel antibodies that can target both the variants as well as the original strain is necessary. In this study, we generated a high-affinity monoclonal antibody (5G2) against the highly conserved region of the SARS-CoV-2 spike protein to detect the protein variants. Moreover, we generated its single-chain variable antibody fragment (sc5G2). The sc5G2 expressed in mammalian and bacterial cells detected the spike protein of the original SARS-CoV-2 and variant strains. The resulting sc5G2 will be a useful tool to detect the original SARS-CoV-2 and variant strains.

[end paper 97]

[begin paper 98]

PMCID = 11785922

Title = Structural characterization of human monoclonal antibodies targeting uncommon antigenic sites on spike glycoprotein of SARS-CoV

The function of the spike protein N terminal domain (NTD) in coronavirus (CoV) infections is poorly understood. However, some rare antibodies that target the SARS-CoV-2 NTD potently neutralize the virus. This finding suggests the NTD may contribute, in part, to protective immunity. Pansarbecovirus antibodies are desirable for broad protection, but the NTD region of SARS-CoV and SARS-CoV-2 exhibit a high level of sequence divergence; therefore, cross-reactive NTD-specific antibodies are unexpected, and there is no structure of a SARS-CoV NTD-specific antibody in complex with NTD. Here, we report a monoclonal antibody COV1-65, encoded by the IGHV1-69 gene, that recognizes the NTD of SARS-CoV S protein. A prophylaxis study showed the mAb COV1-65 prevented disease when administered before SARS-CoV challenge of BALB/c mice, an effect that requires intact fragment crystallizable region (Fc) effector functions for optimal protection in vivo. The footprint on the S protein of COV1-65 is near to functional components of the S2 fusion machinery, and the selection of COV1-65 escape mutant viruses identified critical residues Y886H and Q974H, which likely affect the epitope through allosteric effects. Structural features of the mAb COV1-65–SARS-CoV antigen interaction suggest critical antigenic determinants that should be considered in the rational design of sarbecovirus vaccine candidates.

We describe a naturally occurring human monoclonal antibody that recognizes the N terminal domain of SARS-CoV S protein that protects mice when administered prophylactically.

[end paper 98]

[begin paper 99]

PMCID = 8594574

Title = Computational prediction of the effect of amino acid changes on the binding affinity between SARS-CoV-2 spike RBD and human ACE2

SARS-CoV-2 infection proceeds through the binding of viral surface spike protein to the human ACE2 protein. The global spread of the infection has led to the emergence of fitter and more transmissible variants with increased adaptation both in human and nonhuman hosts. Molecular simulations of the binding event between the spike and ACE2 proteins offer a route to assess potential increase or decrease in infectivity by measuring the change in binding strength. We trained a neural network model that accurately maps simulated binding energies to experimental changes in binding strength upon amino acid changes in the spike protein. This computational workflow can be used to a priori assess currently circulating and prospectively future viral variants for their affinity for hACE2.

The association of the receptor binding domain (RBD) of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein with human angiotensin-converting enzyme 2 (hACE2) represents the first required step for cellular entry. SARS-CoV-2 has continued to evolve with the emergence of several novel variants, and amino acid changes in the RBD have been implicated with increased fitness and potential for immune evasion. Reliably predicting the effect of amino acid changes on the ability of the RBD to interact more strongly with the hACE2 can help assess the implications for public health and the potential for spillover and adaptation into other animals. Here, we introduce a two-step framework that first relies on 48 independent 4-ns molecular dynamics (MD) trajectories of RBD−hACE2 variants to collect binding energy terms decomposed into Coulombic, covalent, van der Waals, lipophilic, generalized Born solvation, hydrogen bonding, π−π packing, and self-contact correction terms. The second step implements a neural network to classify and quantitatively predict binding affinity changes using the decomposed energy terms as descriptors. The computational base achieves a validation accuracy of 82.8% for classifying single–amino acid substitution variants of the RBD as worsening or improving binding affinity for hACE2 and a correlation coefficient of 0.73 between predicted and experimentally calculated changes in binding affinities. Both metrics are calculated using a fivefold cross-validation test. Our method thus sets up a framework for screening binding affinity changes caused by unknown single– and multiple–amino acid changes offering a valuable tool to predict host adaptation of SARS-CoV-2 variants toward tighter hACE2 binding.

[end paper 99]

[begin paper 100]

PMCID = 10509195

Title = Computational redesign of Beta-27 Fab with substantially better predicted binding affinity to the SARS-CoV-2 Omicron variant than human ACE2 receptor

During the COVID-19 pandemic, SARS-CoV-2 has caused large numbers of morbidity and mortality, and the Omicron variant (B.1.1.529) was an important variant of concern. To enter human cells, the receptor-binding domain (RBD) of the S1 subunit of SARS-CoV-2 (SARS-CoV-2-RBD) binds to the peptidase domain (PD) of Angiotensin-converting enzyme 2 (ACE2) receptor. Disrupting the binding interactions between SARS-CoV-2-RBD and ACE2-PD using neutralizing antibodies is an effective COVID-19 therapeutic solution. Previous study found that Beta-27 Fab, which was obtained by digesting the full IgG antibodies that were isolated from a patient infected with SARS-CoV-2 Beta variant, can neutralize Victoria, Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), and Delta (B.1.617.2) variants. This study employed computational protein design and molecular dynamics (MD) to investigate and enhance the binding affinity of Beta-27 Fab to SARS-CoV-2-RBD Omicron variant. MD results show that five best designed Beta-27 Fabs (Beta-27-D01 Fab, Beta-27-D03 Fab, Beta-27-D06 Fab, Beta-27-D09 Fab and Beta-27-D10 Fab) were predicted to bind to Omicron RBD in the area, where ACE2 binds, with significantly better binding affinities than Beta-27 Fab and ACE2. Their enhanced binding affinities are mostly caused by increased binding interactions of CDR L2 and L3. They are promising candidates that could potentially be employed to disrupt the binding between ACE2 and Omicron RBD.

[end paper 100]



