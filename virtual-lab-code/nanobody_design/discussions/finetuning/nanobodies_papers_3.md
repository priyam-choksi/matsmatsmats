## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "nanobodies" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on nanobodies to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on nanobodies using the query "machine learning AND nanobody design AND SARS-CoV-2" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to nanobodies. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, here are the papers that are directly relevant to nanobodies and machine learning in the context of SARS-CoV-2, which I would like to fine-tune myself on:

```python
{
    "8651292": "Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape",
    "8043454": "Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2",
    "11793827": "Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies",
    "11481066": "Evaluation of an Affinity-Enhanced Anti-SARS-CoV2 Nanobody Design Workflow Using Machine Learning and Molecular Dynamics",
    "9170159": "Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants",
    "10373554": "Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening",
    "11788755": "Nanobody engineering: computational modelling and design for biomedical and therapeutic applications",
    "10865538": "Single-domain antibodies against SARS-CoV-2 RBD from a two-stage phage screening of universal and focused synthetic libraries",
    "11515141": "Discovery of nanobodies: a comprehensive review of their applications and potential over the past five years",
    "11133524": "Prospects for the computational humanization of antibodies and nanobodies",
    "11361996": "Construction and validation of a synthetic phage-displayed nanobody library",
    "11650500": "NABP-BERT: NANOBODY®-antigen binding prediction based on bidirectional encoder representations from transformers (BERT) architecture",
    "9411858": "NanoNet: Rapid and accurate end-to-end nanobody modeling by deep learning",
    "9651861": "Fragment-based computational design of antibodies targeting structured epitopes"
}
```

These papers focus on the design, screening, and application of nanobodies against SARS-CoV-2, leveraging machine learning and computational approaches, which are directly relevant to your research project.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "machine learning AND nanobody design AND SARS-CoV-2":

[begin paper 1]

PMCID = 8651292

Title = Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape

The emergence of SARS-CoV-2 variants threatens current vaccines and therapeutic antibodies and urgently demands powerful new therapeutics that can resist viral escape. We therefore generated a large nanobody repertoire to saturate the distinct and highly conserved available epitope space of SARS-CoV-2 spike, including the S1 receptor binding domain, N-terminal domain, and the S2 subunit, to identify new nanobody binding sites that may reflect novel mechanisms of viral neutralization. Structural mapping and functional assays show that indeed these highly stable monovalent nanobodies potently inhibit SARS-CoV-2 infection, display numerous neutralization mechanisms, are effective against emerging variants of concern, and are resistant to mutational escape. Rational combinations of these nanobodies that bind to distinct sites within and between spike subunits exhibit extraordinary synergy and suggest multiple tailored therapeutic and prophylactic strategies.

[end paper 1]

[begin paper 2]

PMCID = 8043454

Title = Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2

Despite the great promise of vaccines, the COVID-19 pandemic is ongoing and future serious outbreaks are highly likely, so that multi-pronged containment strategies will be required for many years. Nanobodies are the smallest naturally occurring single domain antigen binding proteins identified to date, possessing numerous properties advantageous to their production and use. We present a large repertoire of high affinity nanobodies against SARS-CoV-2 Spike protein with excellent kinetic and viral neutralization properties, which can be strongly enhanced with oligomerization. This repertoire samples the epitope landscape of the Spike ectodomain inside and outside the receptor binding domain, recognizing a multitude of distinct epitopes and revealing multiple neutralization targets of pseudoviruses and authentic SARS-CoV-2, including in primary human airway epithelial cells. Combinatorial nanobody mixtures show highly synergistic activities, and are resistant to mutational escape and emerging viral variants of concern. These nanobodies establish an exceptional resource for superior COVID-19 prophylactics and therapeutics.

[end paper 2]

[begin paper 3]

PMCID = 11793827

Title = Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies

Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) continues to persist, demonstrating the risks posed by emerging infectious diseases to national security, public health, and the economy. Development of new vaccines and antibodies for emerging viral threats requires substantial resources and time, and traditional development platforms for vaccines and antibodies are often too slow to combat continuously evolving immunological escape variants, reducing their efficacy over time. Previously, we designed a next-generation synthetic humanized nanobody (Nb) phage display library and demonstrated that this library could be used to rapidly identify highly specific and potent neutralizing heavy chain-only antibodies (HCAbs) with prophylactic and therapeutic efficacy in vivo against the original SARS-CoV-2. In this study, we used a combination of high throughput screening and machine learning (ML) models to identify HCAbs with potent efficacy against SARS-CoV-2 viral variants of interest (VOIs) and concern (VOCs). To start, we screened our highly diverse Nb phage display library against several pre-Omicron VOI and VOC receptor binding domains (RBDs) to identify panels of cross-reactive HCAbs. Using HCAb affinity for SARS-CoV-2 VOI and VOCs (pre-Omicron variants) and model features from other published data, we were able to develop a ML model that successfully identified HCAbs with efficacy against Omicron variants, independent of our experimental biopanning workflow. This biopanning informed ML approach reduced the experimental screening burden by 78% to 90% for the Omicron BA.5 and Omicron BA.1 variants, respectively. The combined approach can be applied to other emerging viruses with pandemic potential to rapidly identify effective therapeutic antibodies against emerging variants.

We leveraged a robust experimental pipeline for heavy chain-only (HCAb) library screening to identify 59 potent HCAbs that can cross-neutralize different SARS-CoV-2 variants. Several of these HCAbs with efficacy against different variants also bind to different SARS-CoV-2 epitopes, suggesting they could be used in antibody cocktails or be engineered as bispecific antibodies that are cross-variant and resistant to viral escape. Using existing published data and data generated from our library of HCAbs against diverse pre-Omicron SARS-CoV-2 variants, we developed an ML model to rapidly predict which HCAbs would be effective against Omicron BA.1 and Omicron BA.5. Using this ML model three HCAbs with efficacy against Omicron BA.1 and BA.5 were identified without requiring additional biopanning. This integrated computational and experimental pipeline can be leveraged for future outbreaks to rapidly identify cross-variant countermeasures that are effective against potential emerging variants.

[end paper 3]

[begin paper 4]

PMCID = 11481066

Title = Evaluation of an Affinity-Enhanced Anti-SARS-CoV2 Nanobody Design Workflow Using Machine Learning and Molecular Dynamics

In silico optimization of protein binding has received a great deal of attention in the recent years. Since in silico prefiltering of strong binders is fast and cheap compared to in vitro library screening methods, the advent of powerful hardware and advanced machine learning algorithms has made this strategy more accessible and preferred. These advances have already impacted the global response to pandemic threats. In this study, we proposed and tested a workflow for designing nanobodies targeting the SARS-CoV-2 spike protein receptor binding domain (S-RBD) using machine learning techniques complemented by molecular dynamics simulations. We evaluated the feasibility of this workflow using a test set of 3 different nanobodies and 2 different S-RBD variants, from in silico design and bacterial expression to binding assays of the designed nanobody mutants. We successfully designed nanobodies that were subsequently tested against both the wild-type (Wuhan type) and the delta variant S-RBD and found 2 of them to be stronger binders compared to the wild-type nanobody. We use this case study to describe both the strengths and weaknesses of this in silico assisted nanobody design strategy.

[end paper 4]

[begin paper 5]

PMCID = 9655607

Title = In Silico Approach for the Evaluation of the Potential Antiviral Activity of Extra Virgin Olive Oil (EVOO) Bioactive Constituents Oleuropein and Oleocanthal on Spike Therapeutic Drug Target of SARS-CoV-2

Since there is an urgent need for novel treatments to combat the current coronavirus disease 2019 (COVID-19) pandemic, in silico molecular docking studies were implemented as an attempt to explore the ability of selected bioactive constituents of extra virgin olive oil (EVOO) to act as potent SARS-CoV-2 (severe acute respiratory syndrome coronavirus 2) antiviral compounds, aiming to explore their ability to interact with SARS-CoV-2 Spike key therapeutic target protein. Our results suggest that EVOO constituents display substantial capacity for binding and interfering with Spike (S) protein, both wild-type and mutant, via the receptor-binding domain (RBD) of Spike, or other binding targets such as angiotensin-converting enzyme 2 (ACE2) or the RBD-ACE2 protein complex, inhibiting the interaction of the virus with host cells. This in silico study provides useful insights for the understanding of the mechanism of action of the studied compounds at a molecular level. From the present study, it could be suggested that the studied active phytochemicals could potentially inhibit the Spike protein, contributing thus to the understanding of the role that they can play in future drug designing and the development of anti-COVID-19 therapeutics.

[end paper 5]

[begin paper 6]

PMCID = 9313406

Title = Plasmonic Approaches for the Detection of SARS-CoV-2 Viral Particles

The ongoing highly contagious Coronavirus disease 2019 (COVID-19) pandemic, caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), underlines the fundamental position of diagnostic testing in outbreak control by allowing a distinction of the infected from the non-infected people. Diagnosis of COVID-19 remains largely based on reverse transcription PCR (RT-PCR), identifying the genetic material of the virus. Molecular testing approaches have been largely proposed in addition to infectivity testing of patients via sensing the presence of viral particles of SARS-CoV-2 specific structural proteins, such as the spike glycoproteins (S1, S2) and the nucleocapsid (N) protein. While the S1 protein remains the main target for neutralizing antibody treatment upon infection and the focus of vaccine and therapeutic design, it has also become a major target for the development of point-of care testing (POCT) devices. This review will focus on the possibility of surface plasmon resonance (SPR)-based sensing platforms to convert the receptor-binding event of SARS-CoV-2 viral particles into measurable signals. The state-of-the-art SPR-based SARS-CoV-2 sensing devices will be provided, and highlights about the applicability of plasmonic sensors as POCT for virus particle as well as viral protein sensing will be discussed.

[end paper 6]

[begin paper 7]

PMCID = 9170159

Title = Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants

Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. In this report, we built dromedary camel (Camelus dromedarius) VHH phage libraries for the isolation of high-affinity nanobodies that broadly neutralize SARS-CoV-2 variants. Cryo-EM complex structures reveal that one dromedary camel VHH nanobody (8A2) binds the S1 subunit of the viral spike protein, and the other (7A3) targets a deeply buried region that uniquely extends to the S2 subunit beyond the S1 subunit. These nanobodies can protect mice from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting the therapeutic potential of these nanobodies against COVID-19. The dromedary camel VHH libraries could be helpful to isolate neutralizing nanobodies against future emerging viruses quickly.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike is a trimer of S1/S2 heterodimers with three receptor-binding domains (RBDs) at the S1 subunit for human angiotensin-converting enzyme 2 (hACE2). Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. To isolate high-affinity nanobodies, large libraries with great diversity are highly desirable. Dromedary camels (Camelus dromedarius) are natural reservoirs of coronaviruses like Middle East respiratory syndrome CoV (MERS-CoV) that are transmitted to humans. Here, we built large dromedary camel VHH phage libraries to isolate nanobodies that broadly neutralize SARS-CoV-2 variants. We isolated two VHH nanobodies, NCI-CoV-7A3 (7A3) and NCI-CoV-8A2 (8A2), which have a high affinity for the RBD via targeting nonoverlapping epitopes and show broad neutralization activity against SARS-CoV-2 and its emerging variants of concern. Cryoelectron microscopy (cryo-EM) complex structures revealed that 8A2 binds the RBD in its up mode with a long CDR3 loop directly involved in the ACE2 binding residues and that 7A3 targets a deeply buried region that uniquely extends from the S1 subunit to the apex of the S2 subunit regardless of the conformational state of the RBD. At a dose of ≥5 mg/kg, 7A3 efficiently protected transgenic mice expressing hACE2 from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting its therapeutic use against COVID-19 variants. The dromedary camel VHH phage libraries could be helpful as a unique platform ready for quickly isolating potent nanobodies against future emerging viruses.

[end paper 7]

[begin paper 8]

PMCID = 10023316

Title = Machine learning and protein allostery

The fundamental biological importance and complexity of allosterically regulated proteins stem from their central role in signal transduction and cellular processes. Recently, machine learning approaches have been developed and actively deployed to facilitate theoretical and experimental studies of protein dynamics and allosteric mechanisms. We surveyed recent developments in applications of machine learning methods for studies of allosteric mechanisms, prediction of allosteric effects and allostery-related physicochemical properties, and allosteric protein engineering. We also reviewed the applications of machine learning strategies for characterization of allosteric mechanisms and drug design targeting SARS-CoV-2 virus. Continuous development and task-specific adaptation of machine learning methods for protein allosteric mechanisms will play an increasingly important role in bridging a wide spectrum of data-intensive experimental and theoretical technologies.

[end paper 8]

[begin paper 9]

PMCID = 10702856

Title =  In vitro generated antibodies guide thermostable ADDomer nanoparticle design for nasal vaccination and passive immunization against SARS-CoV-2

Due to COVID-19, pandemic preparedness emerges as a key imperative, necessitating new approaches to accelerate development of reagents against infectious pathogens.

Here, we developed an integrated approach combining synthetic, computational and structural methods with in vitro antibody selection and in vivo immunization to design, produce and validate nature-inspired nanoparticle-based reagents against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).

Our approach resulted in two innovations: (i) a thermostable nasal vaccine called ADDoCoV, displaying multiple copies of a SARS-CoV-2 receptor binding motif derived epitope and (ii) a multivalent nanoparticle superbinder, called Gigabody, against SARS-CoV-2 including immune-evasive variants of concern (VOCs). In vitro generated neutralizing nanobodies and electron cryo-microscopy established authenticity and accessibility of epitopes displayed by ADDoCoV. Gigabody comprising multimerized nanobodies prevented SARS-CoV-2 virion attachment with picomolar EC50. Vaccinating mice resulted in antibodies cross-reacting with VOCs including Delta and Omicron.

Our study elucidates Adenovirus-derived dodecamer (ADDomer)-based nanoparticles for use in active and passive immunization and provides a blueprint for crafting reagents to combat respiratory viral infections.

Statement of Significance: COVID-19 reminds us of the critical importance of pandemic preparedness, requiring sophisticated approaches to facilitate and accelerate reagent development against emerging threats. We described an integrated pipeline approach to rapidly craft in parallel ADDomer-based nanoparticles for vaccination and for passive immunization, applicable to many infectious diseases.

[end paper 9]

[begin paper 10]

PMCID = 9941073

Title = Emerging trends in point-of-care biosensing strategies for molecular architectures and antibodies of SARS-CoV-2

COVID-19, a highly contagious viral infection caused by the occurrence of severe acute respiratory syndrome coronavirus (SARS-CoV-2), has turned out to be a viral pandemic then ravaged many countries worldwide. In the recent years, point-of-care (POC) biosensors combined with state-of-the-art bioreceptors, and transducing systems enabled the development of novel diagnostic tools for rapid and reliable detection of biomarkers associated with SARS-CoV-2. The present review thoroughly summarises and discusses various biosensing strategies developed for probing SARS-CoV-2 molecular architectures (viral genome, S Protein, M protein, E protein, N protein and non-structural proteins) and antibodies as a potential diagnostic tool for COVID-19. This review discusses the various structural components of SARS-CoV-2, their binding regions and the bioreceptors used for recognizing the structural components. The various types of clinical specimens investigated for rapid and POC detection of SARS-CoV-2 is also highlighted. The importance of nanotechnology and artificial intelligence (AI) approaches in improving the biosensor performance for real-time and reagent-free monitoring the biomarkers of SARS-CoV-2 is also summarized. This review also encompasses existing practical challenges and prospects for developing new POC biosensors for clinical monitoring of COVID-19.

[end paper 10]

[begin paper 11]

PMCID = 10076079

Title = Next-generation nanophotonic-enabled biosensors for intelligent diagnosis of SARS-CoV-2 variants

Constantly mutating SARS-CoV-2 is a global concern resulting in COVID-19 infectious waves from time to time in different regions, challenging present-day diagnostics and therapeutics. Early-stage point-of-care diagnostic (POC) biosensors are a crucial vector for the timely management of morbidity and mortalities caused due to COVID-19. The state-of-the-art SARS-CoV-2 biosensors depend upon developing a single platform for its diverse variants/biomarkers, enabling precise detection and monitoring. Nanophotonic-enabled biosensors have emerged as ‘one platform’ to diagnose COVID-19, addressing the concern of constant viral mutation. This review assesses the evolution of current and future variants of the SARS-CoV-2 and critically summarizes the current state of biosensor approaches for detecting SARS-CoV-2 variants/biomarkers employing nanophotonic-enabled diagnostics. It discusses the integration of modern-age technologies, including artificial intelligence, machine learning and 5G communication with nanophotonic biosensors for intelligent COVID-19 monitoring and management. It also highlights the challenges and potential opportunities for developing intelligent biosensors for diagnosing future SARS-CoV-2 variants. This review will guide future research and development on nano-enabled intelligent photonic-biosensor strategies for early-stage diagnosing of highly infectious diseases to prevent repeated outbreaks and save associated human mortalities.

[end paper 11]

[begin paper 12]

PMCID = 8718524

Title = A rapid and affordable point of care test for antibodies against SARS-CoV-2 based on hemagglutination and artificial intelligence interpretation

Diagnostic tests that detect antibodies (AB) against SARS-CoV-2 for evaluation of seroprevalence and guidance of health care measures are important tools for managing the COVID-19 pandemic. Current tests have certain limitations with regard to turnaround time, costs and availability, particularly in point-of-care (POC) settings. We established a hemagglutination-based AB test that is based on bi-specific proteins which contain a dromedary-derived antibody (nanobody) binding red blood cells (RBD) and a SARS-CoV-2-derived antigen, such as the receptor-binding domain of the Spike protein (Spike-RBD). While the nanobody mediates swift binding to RBC, the antigen moiety directs instantaneous, visually apparent hemagglutination in the presence of SARS-CoV-2-specific AB generated in COVID-19 patients or vaccinated individuals. Method comparison studies with assays cleared by emergency use authorization demonstrate high specificity and sensitivity. To further increase objectivity of test interpretation, we developed an image analysis tool based on digital image acquisition (via a cell phone) and a machine learning algorithm based on defined sample-training and -validation datasets. Preliminary data, including a small clinical study, provides proof of principle for test performance in a POC setting. Together, the data support the interpretation that this AB test format, which we refer to as ‘NanoSpot.ai’, is suitable for POC testing, can be manufactured at very low costs and, based on its generic mode of action, can likely be adapted to a variety of other pathogens.

[end paper 12]

[begin paper 13]

PMCID = 10373554

Title = Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening

Members of the Sarbecovirus subgenus of Coronaviridae have twice caused deadly threats to humans. There is increasing concern about the rapid mutation of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), which has evolved into multiple generations of epidemic variants in 3 years. Broad neutralizing antibodies are of great importance for pandemic preparedness against SARS-CoV-2 variants and divergent zoonotic sarbecoviruses. Here, we analyzed the structural conservation of the receptor-binding domain (RBD) from representative sarbecoviruses and chose S2H97, a previously reported RBD antibody with ideal breadth and resistance to escape, as a template for computational design to enhance the neutralization activity and spectrum. A total of 35 designs were purified for evaluation. The neutralizing activity of a large proportion of these designs against multiple variants was increased from several to hundreds of times. Molecular dynamics simulation suggested that extra interface contacts and enhanced intermolecular interactions between the RBD and the designed antibodies are established. After light and heavy chain reconstitution, AI-1028, with five complementarity determining regions optimized, showed the best neutralizing activity across all tested sarbecoviruses, including SARS-CoV, multiple SARS-CoV-2 variants, and bat-derived viruses. AI-1028 recognized the same cryptic RBD epitope as the parental prototype antibody. In addition to computational design, chemically synthesized nanobody libraries are also a precious resource for rapid antibody development. By applying distinct RBDs as baits for reciprocal screening, we identified two novel nanobodies with broad activities. These findings provide potential pan-sarbecovirus neutralizing drugs and highlight new pathways to rapidly optimize therapeutic candidates when novel SARS-CoV-2 escape variants or new zoonotic coronaviruses emerge.

IMPORTANCE The subgenus Sarbecovirus includes human SARS-CoV, SARS-CoV-2, and hundreds of genetically related bat viruses. The continuous evolution of SARS-CoV-2 has led to the striking evasion of neutralizing antibody (NAb) drugs and convalescent plasma. Antibodies with broad activity across sarbecoviruses would be helpful to combat current SARS-CoV-2 mutations and longer term animal virus spillovers. The study of pan-sarbecovirus NAbs described here is significant for the following reasons. First, we established a structure-based computational pipeline to design and optimize NAbs to obtain more potent and broader neutralizing activity across multiple sarbecoviruses. Second, we screened and identified nanobodies from a highly diversified synthetic library with a broad neutralizing spectrum using an elaborate screening strategy. These methodologies provide guidance for the rapid development of antibody therapeutics against emerging pathogens with highly variable characteristics.

[end paper 13]

[begin paper 14]

PMCID = 11788755

Title = Nanobody engineering: computational modelling and design for biomedical and therapeutic applications

Nanobodies, the smallest functional antibody fragment derived from camelid heavy‐chain‐only antibodies, have emerged as powerful tools for diverse biomedical applications. In this comprehensive review, we discuss the structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We explore their unique antigen‐binding domains, highlighting the critical role of complementarity‐determining regions in target recognition and specificity. This review further underscores the advantages of nanobodies over conventional antibodies from a biosynthesis perspective, including their small size, stability, and solubility, which make them ideal candidates for economical antigen capture in diagnostics, therapeutics, and biosensing. We discuss the recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. Finally, we examine a direct example of how computational design strategies were implemented for improving a nanobody‐based immunosensor, known as a Quenchbody. Through combining experimental findings and computational insights, this review elucidates the transformative impact of nanobodies in biotechnology and biomedical research, offering a roadmap for future advancements and applications in healthcare and diagnostics.

This review focuses on structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We discuss recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. It underscores their advantages over conventional antibodies from a biosynthesis perspective in diagnostics, therapeutics, and biosensing. 

[end paper 14]

[begin paper 15]

PMCID = 10661182

Title = Wireless, battery-free, multifunctional integrated bioelectronics for respiratory pathogens monitoring and severity evaluation

The rapid diagnosis of respiratory virus infection through breath and blow remains challenging. Here we develop a wireless, battery-free, multifunctional pathogenic infection diagnosis system (PIDS) for diagnosing SARS-CoV-2 infection and symptom severity by blow and breath within 110 s and 350 s, respectively. The accuracies reach to 100% and 92% for evaluating the infection and symptom severity of 42 participants, respectively. PIDS realizes simultaneous gaseous sample collection, biomarker identification, abnormal physical signs recording and machine learning analysis. We transform PIDS into other miniaturized wearable or portable electronic platforms that may widen the diagnostic modes at home, outdoors and public places. Collectively, we demonstrate a general-purpose technology for rapidly diagnosing respiratory pathogenic infection by breath and blow, alleviating the technical bottleneck of saliva and nasopharyngeal secretions. PIDS may serve as a complementary diagnostic tool for other point-of-care techniques and guide the symptomatic treatment of viral infections.

The SARS-CoV-2 pandemic highlighted our need for methods that allow rapid viral surveillance. Here, authors report a wireless, battery-free and wearable self-diagnosis platform that can continuously capture viral particles, diagnose infection status and evaluate symptom severity via breath and blow.

[end paper 15]

[begin paper 16]

PMCID = 11403221

Title = Discovery of a novel inhibitor of macropinocytosis with antiviral activity

Several viruses hijack various forms of endocytosis in order to infect host cells. Here, we report the discovery of a molecule with antiviral properties that we named virapinib, which limits viral entry by macropinocytosis. The identification of virapinib derives from a chemical screen using high-throughput microscopy, where we identified chemical entities capable of preventing infection with a pseudotype virus expressing the spike (S) protein from SARS-CoV-2. Subsequent experiments confirmed the capacity of virapinib to inhibit infection by SARS-CoV-2, as well as by additional viruses, such as mpox virus and TBEV. Mechanistic analyses revealed that the compound inhibited macropinocytosis, limiting this entry route for the viruses. Importantly, virapinib has no significant toxicity to host cells. In summary, we present the discovery of a molecule that inhibits macropinocytosis, thereby limiting the infectivity of viruses that use this entry route such as SARS-CoV2.

Porebski and colleagues report on a chemical screen that enables the identification of a new chemical series that inhibits viral infection by several viruses, including SARS-CoV-2. The drug, named virapinib, limits viral infection by inhibiting macropinocytosis, an endocytic route by which multiple viral particles are engulfed by host cells.

[end paper 16]

[begin paper 17]

PMCID = 9111085

Title = Aptamers as promising nanotheranostic tools in the COVID‐19 pandemic era

The emergence of SARS‐COV‐2, the causative agent of new coronavirus disease (COVID‐19) has become a pandemic threat. Early and precise detection of the virus is vital for effective diagnosis and treatment. Various testing kits and assays, including nucleic acid detection methods, antigen tests, serological tests, and enzyme‐linked immunosorbent assay (ELISA), have been implemented or are being explored to detect the virus and/or characterize cellular and antibody responses to the infection. However, these approaches have inherent drawbacks such as nonspecificity, high cost, are characterized by long turnaround times for test results, and can be labor‐intensive. Also, the circulating SARS‐COV‐2 variant of concerns, reduced antibody sensitivity and/or neutralization, and possible antibody‐dependent enhancement (ADE) have warranted the search for alternative potent therapeutics. Aptamers, which are single‐stranded oligonucleotides, generated artificially by SELEX (Evolution of Ligands by Exponential Enrichment) may offer the capacity to generate high‐affinity neutralizers and/or bioprobes for monitoring relevant SARS‐COV‐2 and COVID‐19 biomarkers. This article reviews and discusses the prospects of implementing aptamers for rapid point‐of‐care detection and treatment of SARS‐COV‐2. We highlight other SARS‐COV‐2 targets (N protein, spike protein stem‐helix), SELEX augmented with competition assays and in silico technologies for rapid discovery and isolation of theranostic aptamers against COVID‐19 and future pandemics. It further provides an overview on site‐specific bioconjugation approaches, customizable molecular scaffolding strategies, and nanotechnology platforms to engineer these aptamers into ultrapotent blockers, multivalent therapeutics, and vaccines to boost both humoral and cellular immunity against the virus.

Therapeutic Approaches and Drug Discovery > Emerging Technologies

Diagnostic Tools > Biosensing

Therapeutic Approaches and Drug Discovery > Nanomedicine for Infectious Disease

Therapeutic Approaches and Drug Discovery > Nanomedicine for Respiratory Disease

This article is categorized under: 

The emergence of the COVID‐19 pandemic caused by SARS‐COV‐2 has resulted in a global public health threat, necessitating the need for low‐cost, easy‐to‐use, rapid, ultrasensitive, accurate point‐of‐care diagnostics, and effective treatment. In this review, we discuss recent development on aptamers as alternative potential SARS‐COV‐2 diagnostics and therapeutics. We also highlight promising strategies and targets for rapidly discovering and engineering ultrapotent aptamer‐based inhibitors, drug candidates, and vaccines against SARS‐COV‐2 escape mutants. 

[end paper 17]

[begin paper 18]

PMCID = 11773156

Title = Rethinking Optimal Immunogens to Face SARS‐CoV‐2 Evolution Through Vaccination

SARS‐CoV‐2, which originated in China in late 2019, quickly fueled the global COVID‐19 pandemic, profoundly impacting health and the economy worldwide. A series of vaccines, mostly based on the full SARS‐CoV‐2 Spike protein, were rapidly developed, showing excellent humoral and cellular responses and high efficacy against both symptomatic infection and severe disease. However, viral evolution and the waning humoral neutralizing responses strongly challenged vaccine long term effectiveness, mainly against symptomatic infection, making necessary a strategy of repeated and updated booster shots. In this repeated vaccination context, antibody repertoire diversification was evidenced, although immune imprinting after booster doses or reinfection was also demonstrated and identified as a major determinant of immunological responses to repeated antigen exposures. Considering that a small domain of the SARS‐CoV‐2 Spike protein, the receptor binding domain (RBD), is the major target of neutralizing antibodies and concentrates most viral mutations, the following text aims to provide insights into the ongoing debate over the best strategies for vaccine boosters. We address the relevance of developing new booster vaccines that target the evolving RBD, thus focusing on the relevant antigenic sites of the SARS‐CoV‐2 new variants. A combination of this strategy with immunofusing and computerized approaches could minimize immune imprinting, therefore optimizing neutralizing immune responses and booster vaccine efficacy.

[end paper 18]

[begin paper 19]

PMCID = 9688389

Title = Angiotensin-Converting Enzyme 2-Based Biosensing Modalities and Devices for Coronavirus Detection

Rapid and cost-effective diagnostic tests for severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) are a critical and valuable weapon for the coronavirus disease 2019 (COVID-19) pandemic response. SARS-CoV-2 invasion is primarily mediated by human angiotensin-converting enzyme 2 (hACE2). Recent developments in ACE2-based SARS-CoV-2 detection modalities accentuate the potential of this natural host-virus interaction for developing point-of-care (POC) COVID-19 diagnostic systems. Although research on harnessing ACE2 for SARS-CoV-2 detection is in its infancy, some interesting biosensing devices have been developed, showing the commercial viability of this intriguing new approach. The exquisite performance of the reported ACE2-based COVID-19 biosensors provides opportunities for researchers to develop rapid detection tools suitable for virus detection at points of entry, workplaces, or congregate scenarios in order to effectively implement pandemic control and management plans. However, to be considered as an emerging approach, the rationale for ACE2-based biosensing needs to be critically and comprehensively surveyed and discussed. Herein, we review the recent status of ACE2-based detection methods, the signal transduction principles in ACE2 biosensors and the development trend in the future. We discuss the challenges to development of ACE2-biosensors and delineate prospects for their use, along with recommended solutions and suggestions.

[end paper 19]

[begin paper 20]

PMCID = 9312918

Title = Utilizing Electrochemical-Based Sensing Approaches for the Detection of SARS-CoV-2 in Clinical Samples: A Review

The development of precise and efficient diagnostic tools enables early treatment and proper isolation of infected individuals, hence limiting the spread of coronavirus disease 2019 (COVID-19). The standard diagnostic tests used by healthcare workers to diagnose severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) infection have some limitations, including longer detection time, the need for qualified individuals, and the use of sophisticated bench-top equipment, which limit their use for rapid SARS-CoV-2 assessment. Advances in sensor technology have renewed the interest in electrochemical biosensors miniaturization, which provide improved diagnostic qualities such as rapid response, simplicity of operation, portability, and readiness for on-site screening of infection. This review gives a condensed overview of the current electrochemical sensing platform strategies for SARS-CoV-2 detection in clinical samples. The fundamentals of fabricating electrochemical biosensors, such as the chosen electrode materials, electrochemical transducing techniques, and sensitive biorecognition molecules, are thoroughly discussed in this paper. Furthermore, we summarised electrochemical biosensors detection strategies and their analytical performance on diverse clinical samples, including saliva, blood, and nasopharyngeal swab. Finally, we address the employment of miniaturized electrochemical biosensors integrated with microfluidic technology in viral electrochemical biosensors, emphasizing its potential for on-site diagnostics applications.

[end paper 20]

[begin paper 21]

PMCID = 8155790

Title = Review of COVID-19 Antibody Therapies

In the global health emergency caused by coronavirus disease 2019 (COVID-19), efficient and specific therapies are urgently needed. Compared with traditional small-molecular drugs, antibody therapies are relatively easy to develop; they are as specific as vaccines in targeting severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2); and they have thus attracted much attention in the past few months. This article reviews seven existing antibodies for neutralizing SARS-CoV-2 with 3D structures deposited in the Protein Data Bank (PDB). Five 3D antibody structures associated with the SARS-CoV spike (S) protein are also evaluated for their potential in neutralizing SARS-CoV-2. The interactions of these antibodies with the S protein receptor-binding domain (RBD) are compared with those between angiotensin-converting enzyme 2 and RBD complexes. Due to the orders of magnitude in the discrepancies of experimental binding affinities, we introduce topological data analysis, a variety of network models, and deep learning to analyze the binding strength and therapeutic potential of the 14 antibody–antigen complexes. The current COVID-19 antibody clinical trials, which are not limited to the S protein target, are also reviewed.

[end paper 21]

[begin paper 22]

PMCID = 8585557

Title = COVID-19 Diagnosis: Current and Future Techniques

COVID-19 pandemic continues to be a global threat, affecting more than 200 countries/territories at both human and economic level. This necessitates the rapid development of highly reliable diagnostic methods in order to effectively and accurately diagnose the pathology to prevent the spread of COVID-19. Currently, RT-PCR is the most widely used method worldwide for SARS-CoV-2 detection. Serological assays are being used for sero-surveys of SARS-CoV-2 antibody prevalence in the community. Radiology imaging has been useful in the clinical diagnosis of COVID-19. These methods have their own limitations and there are continued efforts to develop easier, economic, highly sensitive and specific, point-of-care methods. Reverse transcription-loop mediated isothermal amplification (RT-LAMP), nucleic acid sequence-based amplification (NASBA), CRISPR-Cas-based detection, and digital PCR are such techniques being employed in research laboratories, with many awaiting diagnostic approval from competent authorities. This review highlights the rapidly expanding array of existing and in-development diagnostic tests/strategies that may be used to diagnose SARS-CoV-2 infection in both clinical and research settings.

[end paper 22]

[begin paper 23]

PMCID = 8851820

Title = Impact of new variants on SARS-CoV-2 infectivity and neutralization: A molecular assessment of the alterations in the spike-host protein interactions

The emergence of SARS-CoV-2 variants necessitates rational assessment of their impact on the recognition and neutralization of the virus by the host cell. We present a comparative analysis of the interactions of Alpha, Beta, Gamma, and Delta variants with cognate molecules (ACE2 and/or furin), neutralizing nanobodies (Nbs), and monoclonal antibodies (mAbs) using in silico methods, in addition to Nb-binding assays. Our study elucidates the molecular origin of the ability of Beta and Delta variants to evade selected antibodies, such as REGN10933, LY-CoV555, B38, C105, or H11-H4, while being insensitive to others including REGN10987. Experiments confirm that nanobody Nb20 retains neutralizing activity against the Delta variant. The substitutions T478K and L452R in the Delta variant enhance associations with ACE2, whereas P681R promotes recognition by proteases, thus facilitating viral entry. The Ab-specific responses of variants highlight how full-atomic structure and dynamics analyses are required for assessing the response to newly emerging variants.

Atomic models and simulations explain how SARS-CoV-2 variants evade some antibodies

All variants bind ACE2 more strongly with Delta variant optimizing electrostatics

P681H and P681R enhance recognition of Alpha and Delta variants by Furin

Some antibodies retain binding to variant RBDs including Nb20 to Delta variant

Biological sciences; Virology; Structural biology

[end paper 23]

[begin paper 24]

PMCID = 8312898

Title = A rapid and affordable point-of-care test for detection of SARS-CoV-2-specific antibodies based on hemagglutination and Artificial Intelligence-based image interpretation

Diagnostic tests that detect antibodies (AB) against SARS-CoV-2 for evaluation of seroprevalence and guidance of health care measures are important tools for managing the COVID-19 pandemic. Current tests have certain limitations with regard to turnaround time, costs and availability, particularly in point-of-care (POC) settings. We established a hemagglutination-based AB test (HAT) that is based on bi-specific proteins which contain a dromedary-derived antibody (nanobody) binding red blood cells (RBD) and a SARS-CoV-2-derived antigen, such as the receptor-binding domain of the Spike protein (Spike-RBD). While the nanobody mediates swift binding to RBC, the antigen moiety directs instantaneous, visually apparent hemagglutination in the presence of SARS-CoV-2-specific AB generated in COVID-19 patients or vaccinated individuals. Method comparison studies with assays cleared by emergency use authorization (EUA) demonstrate high specificity and sensitivity. To further increase objectivity of test interpretation, we developed an image analysis tool based on digital image acquisition (via a cell phone) and a machine learning algorithm based on defined sample-training and –validation datasets.

Preliminary data, including a small clinical study, provides proof of principle for test performance in a POC setting. Together, the data support the interpretation that this AB test format, which we refer to as ‘NanoSpot.ai’, is suitable for POC testing, can be manufactured at very low costs and, based on its generic mode of action, can likely be adapted to a variety of other pathogens.

[end paper 24]

[begin paper 25]

PMCID = 9719906

Title = Small molecules in the treatment of COVID-19

The outbreak of COVID-19 has become a global crisis, and brought severe disruptions to societies and economies. Until now, effective therapeutics against COVID-19 are in high demand. Along with our improved understanding of the structure, function, and pathogenic process of SARS-CoV-2, many small molecules with potential anti-COVID-19 effects have been developed. So far, several antiviral strategies were explored. Besides directly inhibition of viral proteins such as RdRp and Mpro, interference of host enzymes including ACE2 and proteases, and blocking relevant immunoregulatory pathways represented by JAK/STAT, BTK, NF-κB, and NLRP3 pathways, are regarded feasible in drug development. The development of small molecules to treat COVID-19 has been achieved by several strategies, including computer-aided lead compound design and screening, natural product discovery, drug repurposing, and combination therapy. Several small molecules representative by remdesivir and paxlovid have been proved or authorized emergency use in many countries. And many candidates have entered clinical-trial stage. Nevertheless, due to the epidemiological features and variability issues of SARS-CoV-2, it is necessary to continue exploring novel strategies against COVID-19. This review discusses the current findings in the development of small molecules for COVID-19 treatment. Moreover, their detailed mechanism of action, chemical structures, and preclinical and clinical efficacies are discussed.

[end paper 25]

[begin paper 26]

PMCID = 8234815

Title = Studying SARS-CoV-2 with Fluorescence Microscopy

The COVID-19 pandemic caused by SARS-CoV-2 coronavirus deeply affected the world community. It gave a strong impetus to the development of not only approaches to diagnostics and therapy, but also fundamental research of the molecular biology of this virus. Fluorescence microscopy is a powerful technology enabling detailed investigation of virus–cell interactions in fixed and live samples with high specificity. While spatial resolution of conventional fluorescence microscopy is not sufficient to resolve all virus-related structures, super-resolution fluorescence microscopy can solve this problem. In this paper, we review the use of fluorescence microscopy to study SARS-CoV-2 and related viruses. The prospects for the application of the recently developed advanced methods of fluorescence labeling and microscopy—which in our opinion can provide important information about the molecular biology of SARS-CoV-2—are discussed.

[end paper 26]

[begin paper 27]

PMCID = 10683510

Title = Structural understanding of SARS-CoV-2 virus entry to host cells

Coronavirus disease 2019 (COVID-19), caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), is a major global health concern associated with millions of fatalities worldwide. Mutant variants of the virus have further exacerbated COVID-19 mortality and infection rates, emphasizing the urgent need for effective preventive strategies. Understanding the viral infection mechanism is crucial for developing therapeutics and vaccines. The entry of SARS-CoV-2 into host cells is a key step in the infection pathway and has been targeted for drug development. Despite numerous reviews of COVID-19 and the virus, there is a lack of comprehensive reviews focusing on the structural aspects of viral entry. In this review, we analyze structural changes in Spike proteins during the entry process, dividing the entry process into prebinding, receptor binding, proteolytic cleavage, and membrane fusion steps. By understanding the atomic-scale details of viral entry, we can better target the entry step for intervention strategies. We also examine the impacts of mutations in Spike proteins, including the Omicron variant, on viral entry. Structural information provides insights into the effects of mutations and can guide the development of therapeutics and vaccines. Finally, we discuss available structure-based approaches for the development of therapeutics and vaccines. Overall, this review provides a detailed analysis of the structural aspects of SARS-CoV-2 viral entry, highlighting its significance in the development of therapeutics and vaccines against COVID-19. Therefore, our review emphasizes the importance of structural information in combating SARS-CoV-2 infection.

[end paper 27]

[begin paper 28]

PMCID = 9549062

Title = Recent review of COVID-19 management: diagnosis, treatment and vaccination

The idiopathic Coronavirus disease 2019 (COVID-19) pandemic outbreak caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has reached global proportions; the World Health Organization (WHO) declared it as a public health emergency during the month of January 30, 2020. The major causes of the rise of new variants of SARS-CoV-2 are genetic mutations and recombination. Some of the variants with high infection and transmission rates are termed as variants of concern (VOCs) like currently Omicron variants. Pregnant women, aged people, and immunosuppressed and compromised patients constitute the most susceptible human population to the SARS-CoV-2 infection, especially to the new evolving VOCs. To effectively manage the pathological condition of infection, the focus should be directed towards prevention and prophylactic approach. In this narrative review, we aimed to analyze the current scenario of COVID-19 management and discuss the treatment and prevention strategies. We also focused on the complications prevalent during the COVID-19 and post-COVID period and to discuss the novel approaches developed for mitigation of the global pandemic. We have also emphasized on the COVID-19 management approaches for the special population including children, pregnant women, aged groups, and immunocompromised patients. We conclude that the advancements in therapeutic and pharmacological domains have provided opportunities to develop and design novel diagnosis, treatment, and prevention strategies. New advanced techniques such as RT-LAMP, RT-qPCR, High-Resolution Computed Tomography, etc., efficiently diagnose patients with SARS-CoV-2 infection. In the case of treatment options, new drugs like paxlovid, combinations of β-lactum drugs and molnupiravir are found to be effective against even the new emerging variants. In addition, vaccination is an essential approach to prevent the infection or to reduce its severity. Vaccines for against COVID-19 from Comirnaty by Pfizer-BioNTech, SpikeVax by Moderna, and Vaxzevria by Oxford-AstraZeneca are approved and used widely. Similarly, numerous vaccines have been developed with different percentages of effectiveness against VOCs. New developments like nanotechnology and AI can be beneficial in providing an efficient and reliable solution for the suppression of SARS-CoV-2. Public health concerns can be efficiently treated by a unified scientific approach, public engagement, and better diagnosis.

[end paper 28]

[begin paper 29]

PMCID = 9358106

Title = Visual Detection of COVID-19 from Materials Aspect

In the recent COVID-19 pandemic, World Health Organization emphasized that early detection is an effective strategy to reduce the spread of SARS-CoV-2 viruses. Several diagnostic methods, such as reverse transcription-polymerase chain reaction (RT-PCR) and lateral flow immunoassay (LFIA), have been applied based on the mechanism of specific recognition and binding of the probes to viruses or viral antigens. Although the remarkable progress, these methods still suffer from inadequate cellular materials or errors in the detection and sampling procedure of nasopharyngeal/oropharyngeal swab collection. Therefore, developing accurate, ultrafast, and visualized detection calls for more advanced materials and technology urgently to fight against the epidemic. In this review, we first summarize the current methodologies for SARS-CoV-2 diagnosis. Then, recent representative examples are introduced based on various output signals (e.g., colorimetric, fluorometric, electronic, acoustic). Finally, we discuss the limitations of the methods and provide our perspectives on priorities for future test development.

[end paper 29]

[begin paper 30]

PMCID = 8692065

Title = SARS-CoV-2 reactive and neutralizing antibodies discovered by single-cell sequencing of plasma cells and mammalian display

Characterization of COVID-19 antibodies has largely focused on memory B cells; however, it is the antibody-secreting plasma cells that are directly responsible for the production of serum antibodies, which play a critical role in resolving SARS-CoV-2 infection. Little is known about the specificity of plasma cells, largely because plasma cells lack surface antibody expression, thereby complicating their screening. Here, we describe a technology pipeline that integrates single-cell antibody repertoire sequencing and mammalian display to interrogate the specificity of plasma cells from 16 convalescent patients. Single-cell sequencing allows us to profile antibody repertoire features and identify expanded clonal lineages. Mammalian display screening is used to reveal that 43 antibodies (of 132 candidates) derived from expanded plasma cell lineages are specific to SARS-CoV-2 antigens, including antibodies with high affinity to the SARS-CoV-2 receptor-binding domain (RBD) that exhibit potent neutralization and broad binding to the RBD of SARS-CoV-2 variants (of concern/interest).

Ehling et al. identify SARS-CoV-2 specific and neutralizing antibodies from plasma cells of patients who recovered from COVID-19 using an integrated pipeline of single-cell sequencing and mammalian display. The method described herein enables the functional profiling of expanded plasma cells.

[end paper 30]

[begin paper 31]

PMCID = 11679646

Title = The Application of Machine Learning on Antibody Discovery and Optimization

Antibodies play critical roles in modern medicine, serving as diagnostics and therapeutics for various diseases due to their ability to specifically bind to target antigens. Traditional antibody discovery and optimization methods are time-consuming and resource-intensive, though they have successfully generated antibodies for diagnosing and treating diseases. The advancements in protein data, computational hardware, and machine learning (ML) models have the opportunity to disrupt antibody discovery and optimization research. Machine learning models have demonstrated their abilities in antibody design. These machine learning models enable rapid in silico design of antibody candidates within a few days, achieving approximately a 60% reduction in time and a 50% reduction in cost compared to traditional methods. This review focuses on the latest machine learning-based antibody discovery and optimization developments. We briefly discuss the limitations of traditional methods and then explore the machine learning-based antibody discovery and optimization methodologies. We also focus on future research directions, including developing Antibody Design AI Agents and data foundries, alongside the ethical and regulatory considerations essential for successfully adopting machine learning-driven antibody designs.

[end paper 31]

[begin paper 32]

PMCID = 10627145

Title = Multi-omics for COVID-19: driving development of therapeutics and vaccines

The ongoing COVID-19 pandemic caused by SARS-CoV-2 has raised global concern for public health and economy. The development of therapeutics and vaccines to combat this virus is continuously progressing. Multi-omics approaches, including genomics, transcriptomics, proteomics, metabolomics, epigenomics and metallomics, have helped understand the structural and molecular features of the virus, thereby assisting in the design of potential therapeutics and accelerating vaccine development for COVID-19. Here, we provide an up-to-date overview of the latest applications of multi-omics technologies in strategies addressing COVID-19, in order to provide suggestions towards the development of highly effective knowledge-based therapeutics and vaccines.

Understanding the roles of multi-omics in COVID-19 facilitates the development of novel therapeutics and provides guidance for the rational design of vaccines.

[end paper 32]

[begin paper 33]

PMCID = 10042144

Title = Computational Protein Design for COVID-19 Research and Emerging Therapeutics

As the world struggles with the ongoing COVID-19 pandemic, unprecedented obstacles have continuously been traversed as new SARS-CoV-2 variants continually emerge. Infectious disease outbreaks are unavoidable, but the knowledge gained from the successes and failures will help create a robust health management system to deal with such pandemics. Previously, scientists required years to develop diagnostics, therapeutics, or vaccines; however, we have seen that, with the rapid deployment of high-throughput technologies and unprecedented scientific collaboration worldwide, breakthrough discoveries can be accelerated and insights broadened. Computational protein design (CPD) is a game-changing new technology that has provided alternative therapeutic strategies for pandemic management. In addition to the development of peptide-based inhibitors, miniprotein binders, decoys, biosensors, nanobodies, and monoclonal antibodies, CPD has also been used to redesign native SARS-CoV-2 proteins and human ACE2 receptors. We discuss how novel CPD strategies have been exploited to develop rationally designed and robust COVID-19 treatment strategies.

This outlook summarizes CPD approaches utilized for the development of SARS-CoV-2 therapeutics and highlights the current challenges and future directions of CPD to tackle various infectious diseases.

[end paper 33]

[begin paper 34]

PMCID = 7995000

Title = Advanced microscopy technologies enable rapid response to SARS‐CoV‐2 pandemic

The ongoing SARS‐CoV‐2 pandemic with over 80 million infections and more than a million deaths worldwide represents the worst global health crisis of the 21th century. Beyond the health crisis, the disruptions caused by the COVID‐19 pandemic have serious global socio‐economic consequences. It has also placed a significant pressure on the scientific community to understand the virus and its pathophysiology and rapidly provide anti‐viral treatments and procedures in order to help the society and stop the virus spread. Here, we outline how advanced microscopy technologies such as high‐throughput microscopy and electron microscopy played a major role in rapid response against SARS‐CoV‐2. General applicability of developed microscopy technologies makes them uniquely positioned to act as the first line of defence against any emerging infection in the future.

[end paper 34]

[begin paper 35]

PMCID = 11298684

Title = Immunobiology of COVID-19: Mechanistic and therapeutic insights from animal models

The distribution of the immune system throughout the body complicates in vitro assessments of coronavirus disease 2019 (COVID-19) immunobiology, often resulting in a lack of reproducibility when extrapolated to the whole organism. Consequently, developing animal models is imperative for a comprehensive understanding of the pathology and immunology of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection. This review summarizes current progress related to COVID-19 animal models, including non-human primates (NHPs), mice, and hamsters, with a focus on their roles in exploring the mechanisms of immunopathology, immune protection, and long-term effects of SARS-CoV-2 infection, as well as their application in immunoprevention and immunotherapy of SARS-CoV-2 infection. Differences among these animal models and their specific applications are also highlighted, as no single model can fully encapsulate all aspects of COVID-19. To effectively address the challenges posed by COVID-19, it is essential to select appropriate animal models that can accurately replicate both fatal and non-fatal infections with varying courses and severities. Optimizing animal model libraries and associated research tools is key to resolving the global COVID-19 pandemic, serving as a robust resource for future emerging infectious diseases.

Rhesus macaques, mice transduced with human ACE2 gene, mice infected with adaptive SARS-CoV-2, Syrian hamsters, and ferrets are widely used to study immune pathogenesis and immune protection mechanisms of COVID-19, and also play important roles in COVID-19 vaccine and immunotherapeutic drug development. Figure created using BioRender (https://biorender.com/).

Application of animal models of SARS-CoV-2 infection has promoted systematic research on COVID-19 immunopathology. Animal experiments on vaccines and adjuvants have advanced our understanding of the immunoprotective mechanisms against SARS-CoV-2 infection. Figure created using BioRender (https://biorender.com/).

Validation of immune interventions targeting immune receptors, inflammatory mediators, and immunopathological signals in SARS-CoV-2-infected animal models is crucial for the development of COVID-19 immunotherapy. Figure created using BioRender (https://biorender.com/).

[end paper 35]

[begin paper 36]

PMCID = 10886287

Title = Leveraging Artificial Intelligence to Expedite Antibody Design and Enhance Antibody–Antigen Interactions

This perspective sheds light on the transformative impact of recent computational advancements in the field of protein therapeutics, with a particular focus on the design and development of antibodies. Cutting-edge computational methods have revolutionized our understanding of protein–protein interactions (PPIs), enhancing the efficacy of protein therapeutics in preclinical and clinical settings. Central to these advancements is the application of machine learning and deep learning, which offers unprecedented insights into the intricate mechanisms of PPIs and facilitates precise control over protein functions. Despite these advancements, the complex structural nuances of antibodies pose ongoing challenges in their design and optimization. Our review provides a comprehensive exploration of the latest deep learning approaches, including language models and diffusion techniques, and their role in surmounting these challenges. We also present a critical analysis of these methods, offering insights to drive further progress in this rapidly evolving field. The paper includes practical recommendations for the application of these computational techniques, supplemented with independent benchmark studies. These studies focus on key performance metrics such as accuracy and the ease of program execution, providing a valuable resource for researchers engaged in antibody design and development. Through this detailed perspective, we aim to contribute to the advancement of antibody design, equipping researchers with the tools and knowledge to navigate the complexities of this field.

[end paper 36]

[begin paper 37]

PMCID = 10865538

Title = Single-domain antibodies against SARS-CoV-2 RBD from a two-stage phage screening of universal and focused synthetic libraries

Coronavirus disease 2019 (COVID-19) is an evolving global pandemic, and nanobodies, as well as other single-domain antibodies (sdAbs), have been recognized as a potential diagnostic and therapeutic tool for infectious diseases. High-throughput screening techniques such as phage display have been developed as an alternative to in vivo immunization for the discovery of antibody-like target-specific binders.

We designed and constructed a highly diverse synthetic phage library sdAb-U (single-domain Antibody - Universal library ) based on a human framework. The SARS-CoV-2 receptor-binding domain (RBD) was expressed and purified. The universal library sdAb-U was panned against the RBD protein target for two rounds, followed by monoclonal phage ELISA (enzyme-linked immunosorbent assay) to identify RBD-specific binders (the first stage). High-affinity binders were sequenced and the obtained CDR1 and CDR2 sequences were combined with fully randomized CDR3 to construct a targeted (focused) phage library sdAb-RBD, for subsequent second-stage phage panning (also two rounds) and screening. Then, sequences with high single-to-background ratios in phage ELISA were selected for expression. The binding affinities of sdAbs to RBD were measured by an ELISA-based method. In addition, we conducted competition ELISA (using ACE2 ectodomain S19-D615) and SARS-CoV-2 pseudovirus neutralization assays for the high-affinity RBD-binding sdAb39.

Significant enrichments were observed in both the first-stage (universal library) and the second-stage (focused library) phage panning. Five RBD-specific binders were identified in the first stage with high ELISA signal-to-background ratios. In the second stage, we observed a much higher possibility of finding RBD-specific clones in phage ELISA. Among 45 selected RBD-positive sequences, we found eight sdAbs can be well expressed, and five of them show high-affinity to RBD (EC50 < 100nM). We finally found that sdAb39 (EC50 ~ 4nM) can compete with ACE2 for binding to RBD.

Overall, this two-stage strategy of synthetic phage display libraries enables rapid selection of SARS-CoV-2 RBD sdAb with potential therapeutic activity, and this two-stage strategy can potentially be used for rapid discovery of sdAbs against other targets.

The online version contains supplementary material available at 10.1186/s12879-024-09022-8.

[end paper 37]

[begin paper 38]

PMCID = 8687476

Title = Structural models of SARS-CoV-2 Omicron variant in complex with ACE2 receptor or antibodies suggest altered binding interfaces

There is enormous ongoing interest in characterizing the binding properties of the SARS-CoV-2 Omicron Variant of Concern (VOC) (B.1.1.529), which continues to spread towards potential dominance worldwide. To aid these studies, based on the wealth of available structural information about several SARS-CoV-2 variants in the Protein Data Bank (PDB) and a modeling pipeline we have previously developed for tracking the ongoing global evolution of SARS-CoV-2 proteins, we provide a set of computed structural models (henceforth models) of the Omicron VOC receptor-binding domain (omRBD) bound to its corresponding receptor Angiotensin-Converting Enzyme (ACE2) and a variety of therapeutic entities, including neutralizing and therapeutic antibodies targeting previously-detected viral strains. We generated bound omRBD models using both experimentally-determined structures in the PDB as well as machine learning-based structure predictions as starting points. Examination of ACE2-bound omRBD models reveals an interdigitated multi-residue interaction network formed by omRBD-specific substituted residues (R493, S496, Y501, R498) and ACE2 residues at the interface, which was not present in the original Wuhan-Hu-1 RBD-ACE2 complex. Emergence of this interaction network suggests optimization of a key region of the binding interface, and positive cooperativity among various sites of residue substitutions in omRBD mediating ACE2 binding. Examination of neutralizing antibody complexes for Barnes Class 1 and Class 2 antibodies modeled with omRBD highlights an overall loss of interfacial interactions (with gain of new interactions in rare cases) mediated by substituted residues. Many of these substitutions have previously been found to independently dampen or even ablate antibody binding, and perhaps mediate antibody-mediated neutralization escape (e.g., K417N). We observe little compensation of corresponding interaction loss at interfaces when potential escape substitutions occur in combination. A few selected antibodies (e.g., Barnes Class 3 S309), however, feature largely unaltered or modestly affected protein-protein interfaces. While we stress that only qualitative insights can be obtained directly from our models at this time, we anticipate that they can provide starting points for more detailed and quantitative computational characterization, and, if needed, redesign of monoclonal antibodies for targeting the Omicron VOC Spike protein. In the broader context, the computational pipeline we developed provides a framework for rapidly and efficiently generating retrospective and prospective models for other novel variants of SARS-CoV-2 bound to entities of virological and therapeutic interest, in the setting of a global pandemic.

[end paper 38]

[begin paper 39]

PMCID = 8928824

Title = Progress and challenges for the machine learning-based design of fit-for-purpose monoclonal antibodies

Although the therapeutic efficacy and commercial success of monoclonal antibodies (mAbs) are tremendous, the design and discovery of new candidates remain a time and cost-intensive endeavor. In this regard, progress in the generation of data describing antigen binding and developability, computational methodology, and artificial intelligence may pave the way for a new era of in silico on-demand immunotherapeutics design and discovery. Here, we argue that the main necessary machine learning (ML) components for an in silico mAb sequence generator are: understanding of the rules of mAb-antigen binding, capacity to modularly combine mAb design parameters, and algorithms for unconstrained parameter-driven in silico mAb sequence synthesis. We review the current progress toward the realization of these necessary components and discuss the challenges that must be overcome to allow the on-demand ML-based discovery and design of fit-for-purpose mAb therapeutic candidates.

[end paper 39]

[begin paper 40]

PMCID = 10887370

Title = Hierarchical Nanobiosensors at the End of the SARS-CoV-2 Pandemic

Nanostructures have played a key role in the development of different techniques to attack severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Some applications include masks, vaccines, and biosensors. The latter are of great interest for detecting diseases since some of their features allowed us to find specific markers in secretion samples such as saliva, blood, and even tears. Herein, we highlight how hierarchical nanoparticles integrated into two or more low-dimensional materials present outstanding advantages that are attractive for photonic biosensing using their nanoscale functions. The potential of nanohybrids with their superlative mechanical characteristics together with their optical and optoelectronic properties is discussed. The progress in the scientific research focused on using nanoparticles for biosensing a variety of viruses has become a medical milestone in recent years, and has laid the groundwork for future disease treatments. This perspective analyzes the crucial information about the use of hierarchical nanostructures in biosensing for the prevention, treatment, and mitigation of SARS-CoV-2 effects.

[end paper 40]

[begin paper 41]

PMCID = 9425703

Title = Current trends in COVID-19 diagnosis and its new variants in physiological fluids: Surface antigens, antibodies, nucleic acids, and RNA sequencing

Rapid, highly sensitive, and accurate virus circulation monitoring techniques are critical to limit the spread of the virus and reduce the social and economic burden. Therefore, point-of-use diagnostic devices have played a critical role in addressing the outbreak of COVID-19 (SARS-CoV-2) viruses. This review provides a comprehensive overview of the current techniques developed for the detection of SARS-CoV-2 in various body fluids (e.g., blood, urine, feces, saliva, tears, and semen) and considers the mutations (i.e., Alpha, Beta, Gamma, Delta, Omicron). We classify and comprehensively discuss the detection methods depending on the biomarker measured (i.e., surface antigen, antibody, and nucleic acid) and the measurement techniques such as lateral flow immunoassay (LFIA), enzyme-linked immunosorbent assay (ELISA), reverse transcriptase-polymerase chain reaction (RT-PCR), reverse transcription loop-mediated isothermal amplification (RT-LAMP), microarray analysis, clustered regularly interspaced short palindromic repeats (CRISPR) and biosensors. Finally, we addressed the challenges of rapidly identifying emerging variants, detecting the virus in the early stages of infection, the detection sensitivity, selectivity, and specificity, and commented on how these challenges can be overcome in the future.

[end paper 41]

[begin paper 42]

PMCID = 10371346

Title = Modeling of ACE2 and antibodies bound to SARS-CoV-2 provides insights into infectivity and immune evasion

Given the COVID-19 pandemic, there is interest in understanding ligand-receptor features and targeted antibody-binding attributes against emerging SARS-CoV-2 variants. Here, we developed a large-scale structure-based pipeline for analysis of protein-protein interactions regulating SARS-CoV-2 immune evasion. First, we generated computed structural models of the Spike protein of 3 SARS-CoV-2 variants (B.1.1.529, BA.2.12.1, and BA.5) bound either to a native receptor (ACE2) or to a large panel of targeted ligands (n = 282), which included neutralizing or therapeutic monoclonal antibodies. Moreover, by using the Barnes classification, we noted an overall loss of interfacial interactions (with gain of new interactions in certain cases) at the receptor-binding domain (RBD) mediated by substituted residues for neutralizing complexes in classes 1 and 2, whereas less destabilization was observed for classes 3 and 4. Finally, an experimental validation of predicted weakened therapeutic antibody binding was performed in a cell-based assay. Compared with the original Omicron variant (B.1.1.529), derivative variants featured progressive destabilization of antibody-RBD interfaces mediated by a larger set of substituted residues, thereby providing a molecular basis for immune evasion. This approach and findings provide a framework for rapidly and efficiently generating structural models for SARS-CoV-2 variants bound to ligands of mechanistic and therapeutic value.

[end paper 42]

[begin paper 43]

PMCID = 9429973

Title = Abstract Book for the 27th Congress of the European Hematology Association



[end paper 43]

[begin paper 44]

PMCID = 8153213

Title = Prediction and mitigation of mutation threats to COVID-19 vaccines and antibody therapies†

Antibody therapeutics and vaccines are among our last resort to end the raging COVID-19 pandemic. They, however, are prone to over 5000 mutations on the spike (S) protein uncovered by a Mutation Tracker based on over 200 000 genome isolates. It is imperative to understand how mutations will impact vaccines and antibodies in development. In this work, we first study the mechanism, frequency, and ratio of mutations on the S protein which is the common target of most COVID-19 vaccines and antibody therapies. Additionally, we build a library of 56 antibody structures and analyze their 2D and 3D characteristics. Moreover, we predict the mutation-induced binding free energy (BFE) changes for the complexes of S protein and antibodies or ACE2. By integrating genetics, biophysics, deep learning, and algebraic topology, we reveal that most of the 462 mutations on the receptor-binding domain (RBD) will weaken the binding of S protein and antibodies and disrupt the efficacy and reliability of antibody therapies and vaccines. A list of 31 antibody disrupting mutants is identified, while many other disruptive mutations are detailed as well. We also unveil that about 65% of the existing RBD mutations, including those variants recently found in the United Kingdom (UK) and South Africa, will strengthen the binding between the S protein and human angiotensin-converting enzyme 2 (ACE2), resulting in more infectious COVID-19 variants. We discover the disparity between the extreme values of RBD mutation-induced BFE strengthening and weakening of the bindings with antibodies and angiotensin-converting enzyme 2 (ACE2), suggesting that SARS-CoV-2 is at an advanced stage of evolution for human infection, while the human immune system is able to produce optimized antibodies. This discovery, unfortunately, implies the vulnerability of current vaccines and antibody drugs to new mutations. Our predictions were validated by comparison with more than 1400 deep mutations on the S protein RBD. Our results show the urgent need to develop new mutation-resistant vaccines and antibodies and to prepare for seasonal vaccinations.

Antibody therapeutics and vaccines are among our last resort to end the raging COVID-19 pandemic.

[end paper 44]

[begin paper 45]

PMCID = 8537311

Title = An Antigenic Space Framework for Understanding Antibody Escape of SARS-CoV-2 Variants

The evolution of mutations in SARS-CoV-2 at antigenic sites that impact neutralizing antibody responses in humans poses a risk to immunity developed through vaccination and natural infection. The highly successful RNA-based vaccines have enabled rapid vaccine updates that incorporate mutations from current variants of concern (VOCs). It is therefore important to anticipate future antigenic mutations as the virus navigates the heterogeneous global landscape of host immunity. Toward this goal, we survey epitope-paratope interfaces of anti-SARS-CoV-2 antibodies to map an antigenic space that captures the role of each spike protein residue within the polyclonal antibody response directed against the ACE2-receptor binding domain (RBD) or the N-terminal domain (NTD). In particular, the antigenic space map builds on recently published epitope definitions by annotating epitope overlap and orthogonality at the residue level. We employ the antigenic space map as a framework to understand how mutations on nine major variants contribute to each variant’s evasion of neutralizing antibodies. Further, we identify constellations of mutations that span the orthogonal epitope regions of the RBD and NTD on the variants with the greatest antibody escape. Finally, we apply the antigenic space map to predict which regions of antigenic space—should they mutate—may be most likely to complementarily augment antibody evasion for the most evasive and transmissible VOCs.

[end paper 45]

[begin paper 46]

PMCID = 10879169

Title = Nanotechnology’s frontier in combatting infectious and inflammatory diseases: prevention and treatment

Inflammation-associated diseases encompass a range of infectious diseases and non-infectious inflammatory diseases, which continuously pose one of the most serious threats to human health, attributed to factors such as the emergence of new pathogens, increasing drug resistance, changes in living environments and lifestyles, and the aging population. Despite rapid advancements in mechanistic research and drug development for these diseases, current treatments often have limited efficacy and notable side effects, necessitating the development of more effective and targeted anti-inflammatory therapies. In recent years, the rapid development of nanotechnology has provided crucial technological support for the prevention, treatment, and detection of inflammation-associated diseases. Various types of nanoparticles (NPs) play significant roles, serving as vaccine vehicles to enhance immunogenicity and as drug carriers to improve targeting and bioavailability. NPs can also directly combat pathogens and inflammation. In addition, nanotechnology has facilitated the development of biosensors for pathogen detection and imaging techniques for inflammatory diseases. This review categorizes and characterizes different types of NPs, summarizes their applications in the prevention, treatment, and detection of infectious and inflammatory diseases. It also discusses the challenges associated with clinical translation in this field and explores the latest developments and prospects. In conclusion, nanotechnology opens up new possibilities for the comprehensive management of infectious and inflammatory diseases.

[end paper 46]

[begin paper 47]

PMCID = 7796695

Title = Recent biotechnological advances as potential intervention strategies against COVID-19

The emerging SARS-CoV-2 viral disease (COVID-19) has caused a global health alert due to its high rate of infection and mortality in individuals with chronic cardiovascular comorbidities, in addition to generating complex clinical conditions. This has forced the scientific community to explore different strategies that allow combating this viral infection as well as treating life-threatening systemic effect of the infection on the individual. In this work, we have reviewed the most recent scientific evidence to provide a comprehensive panorama regarding the biotechnological strategies that have been proposed to combat this new viral infection. We have focused our analysis on vaccine production, nanotechnology applications, repurposing of know drugs for unrelated pathologies, and the search for bioactive molecules obtained from natural products. The goals include safely use as potential prophylactic or therapeutic treatments, based on in silico and in vivo studies, including clinical trials around the world for the correct and timely diagnosis of the infection. This review aims to highlight the development of new ideas that can decrease the time lines for research output and improve research quality while at the same time, keeping in mind the efficacy and safety aspects of these potential biotechnological strategies.

[end paper 47]

[begin paper 48]

PMCID = 11413593

Title = A potential allosteric inhibitor of SARS-CoV-2 main protease (Mpro) identified through metastable state analysis

Anti-COVID19 drugs, such as nirmatrelvir, have been developed targeting the SARS-CoV-2 main protease, Mpro, based on the critical requirement of its proteolytic processing of the viral polyproteins into functional proteins essential for viral replication. However, the emergence of SARS-CoV-2 variants with Mpro mutations has raised the possibility of developing resistance against these drugs, likely due to therapeutic targeting of the Mpro catalytic site. An alternative to these drugs is the development of drugs that target an allosteric site distant from the catalytic site in the protein that may reduce the chance of the emergence of resistant mutants. Here, we combine computational analysis with in vitro assay and report the discovery of a potential allosteric site and an allosteric inhibitor of SARS-CoV-2 Mpro. Specifically, we identified an Mpro metastable state with a deformed catalytic site harboring potential allosteric sites, raising the possibility that stabilization of this metastable state through ligand binding can lead to the inhibition of Mpro activity. We then performed a computational screening of a library (∼4.2 million) of drug-like compounds from the ZINC database and identified several candidate molecules with high predicted binding affinity. MD simulations showed stable binding of the three top-ranking compounds to the putative allosteric sites in the protein. Finally, we tested the three compounds in vitro using a BRET-based Mpro biosensor and found that one of the compounds (ZINC4497834) inhibited the Mpro activity. We envisage that the identification of a potential allosteric inhibitor of Mpro will aid in developing improved anti-COVID-19 therapy.

[end paper 48]

[begin paper 49]

PMCID = 10507807

Title =  De novo design of a stapled peptide targeting SARS-CoV-2 spike protein receptor-binding domain†

Although effective vaccines have been developed against SARS-CoV-2, many regions in the world still have low rates of vaccination and new variants with mutations in the viral spike protein have reduced the effectiveness of most available vaccines and treatments. There is an urgent need for a drug to cure this disease and prevent infection. The SARS-CoV-2 virus enters the host cell through protein–protein interaction between the virus's spike protein and the host's angiotensin converting enzyme (ACE2). Using protein design software and molecular dynamics simulations, we have designed a 17-residue peptide (pep39), that binds to the spike protein receptor-binding domain (RBD) and blocks interaction of spike protein with ACE2. We have confirmed the binding activity of the designed peptide for the original spike protein and the delta variant spike protein using micro-cantilever and bio-layer interferometry (BLI) based methods. We also confirmed that pep39 strongly inhibits SARS-CoV-2 virus replication in Vero E6 cells. Taken together these data suggest that a newly designed spike protein RBD blocking peptide pep39 has a potential as a SARS-CoV-2 virus inhibitor.

Binding configuration of a de novo stapled peptide on SARS-CoV-2 spike protein, as predicted by molecular simulation. Stapled residues enhance peptide stability while interacting residues engage key amino acids on the protein receptor-binding domain.

[end paper 49]

[begin paper 50]

PMCID = 8000172

Title = Structure-Function Analyses of New SARS-CoV-2 Variants B.1.1.7, B.1.351 and B.1.1.28.1: Clinical, Diagnostic, Therapeutic and Public Health Implications

SARS-CoV-2 (Severe Acute Respiratory Syndrome-Coronavirus 2) has accumulated multiple mutations during its global circulation. Recently, three SARS-CoV-2 lineages, B.1.1.7 (501Y.V1), B.1.351 (501Y.V2) and B.1.1.28.1 (P.1), have emerged in the United Kingdom, South Africa and Brazil, respectively. Here, we have presented global viewpoint on implications of emerging SARS-CoV-2 variants based on structural–function impact of crucial mutations occurring in its spike (S), ORF8 and nucleocapsid (N) proteins. While the N501Y mutation was observed in all three lineages, the 501Y.V1 and P.1 accumulated a different set of mutations in the S protein. The missense mutational effects were predicted through a COVID-19 dedicated resource followed by atomistic molecular dynamics simulations. Current findings indicate that some mutations in the S protein might lead to higher affinity with host receptors and resistance against antibodies, but not all are due to different antibody binding (epitope) regions. Mutations may, however, result in diagnostic tests failures and possible interference with binding of newly identified anti-viral candidates against SARS-CoV-2, likely necessitating roll out of recurring “flu-like shots” annually for tackling COVID-19. The functional relevance of these mutations has been described in terms of modulation of host tropism, antibody resistance, diagnostic sensitivity and therapeutic candidates. Besides global economic losses, post-vaccine reinfections with emerging variants can have significant clinical, therapeutic and public health impacts.

[end paper 50]

[begin paper 51]

PMCID = 7676326

Title = Diamond Light Source: contributions to SARS-CoV-2 biology and therapeutics

The impact of COVID-19 on public health and the global economy has led to an unprecedented research response, with a major emphasis on the development of safe vaccines and drugs. However, effective, safe treatments typically take over a decade to develop and there are still no clinically approved therapies to treat highly pathogenic coronaviruses. Repurposing of known drugs can speed up development and this strategy, along with the use of biologicals (notably monoclonal antibody therapy) and vaccine development programmes remain the principal routes to dealing with the immediate impact of COVID-19. Nevertheless, the development of broadly-effective highly potent antivirals should be a major longer term goal. Structural biology has been applied with enormous effect, with key proteins structurally characterised only weeks after the SARS-CoV-2 sequence was released. Open-access to advanced infrastructure for structural biology techniques at synchrotrons and high-end cryo-EM and NMR centres has brought these technologies centre-stage in drug discovery. We summarise the role of Diamond Light Source in responses to the pandemic and note the impact of the immediate release of results in fuelling an open-science approach to early-stage drug discovery.

[end paper 51]

[begin paper 52]

PMCID = 8380468

Title = An update on novel approaches for diagnosis and treatment of SARS-CoV-2 infection

The ongoing pandemic of coronavirus disease 2019 (COVID-19) has made a serious public health and economic crisis worldwide which united global efforts to develop rapid, precise, and cost-efficient diagnostics, vaccines, and therapeutics. Numerous multi-disciplinary studies and techniques have been designed to investigate and develop various approaches to help frontline health workers, policymakers, and populations to overcome the disease. While these techniques have been reviewed within individual disciplines, it is now timely to provide a cross-disciplinary overview of novel diagnostic and therapeutic approaches summarizing complementary efforts across multiple fields of research and technology. Accordingly, we reviewed and summarized various advanced novel approaches used for diagnosis and treatment of COVID-19 to help researchers across diverse disciplines on their prioritization of resources for research and development and to give them better a picture of the latest techniques. These include artificial intelligence, nano-based, CRISPR-based, and mass spectrometry technologies as well as neutralizing factors and traditional medicines. We also reviewed new approaches for vaccine development and developed a dashboard to provide frequent updates on the current and future approved vaccines.

The online version contains supplementary material available at 10.1186/s13578-021-00674-6.

[end paper 52]

[begin paper 53]

PMCID = 7694563

Title = Diagnosis of COVID-19 for controlling the pandemic: A review of the state-of-the-art

To date, health organizations and countries around the world are struggling to completely control the spread of the coronavirus disease 2019 (COVID-19). Scientists and researchers are developing tests for the rapid detection of individuals who may carry the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), while striving to find a suitable vaccine to immunize healthy individuals. As there are clinically reported cases of asymptomatic carriers of SARS-CoV-2, fast and accurate diagnosis plays an important role in the control and further prevention of this disease. Herein, we present recent technologies and techniques that have been implemented for the diagnosis of COVID-19. We summarize the methods created by different research institutes as well as the commercial devices and kits developed by companies for the detection of SARS-CoV-2. The description of the existing methods is followed by highlighting their advantages and challenges. Finally, we propose some promising techniques that could potentially be applied to the detection of SARS-CoV-2, and tracing the asymptomatic carriers of COVID-19 rapidly and accurately in the early stages of infection, based on reviewing the research studies on the detection of similar infectious viruses, especially severe acute respiratory syndrome (SARS) coronavirus, and Middle East respiratory syndrome (MERS) coronavirus.

Recent technologies and techniques for the diagnosis of COVID-19 are presented.

Laboratory methods and commercial devices for SARS-CoV-2 detection are discussed.

CT, molecular, and serology tests' pros and cons are elaborated.

Potential techniques are proposed as the future means of COVID-19 diagnosis.

LFA, AI, piezoelectric, and microfluidic devices could trace COVID-19 carriers.

[end paper 53]

[begin paper 54]

PMCID = 9784564

Title = Optimizing Antibody Affinity and Developability Using a Framework–CDR Shuffling Approach—Application to an Anti-SARS-CoV-2 Antibody

The computational methods used for engineering antibodies for clinical development have undergone a transformation from three-dimensional structure-guided approaches to artificial-intelligence- and machine-learning-based approaches that leverage the large sequence data space of hundreds of millions of antibodies generated by next-generation sequencing (NGS) studies. Building on the wealth of available sequence data, we implemented a computational shuffling approach to antibody components, using the complementarity-determining region (CDR) and the framework region (FWR) to optimize an antibody for improved affinity and developability. This approach uses a set of rules to suitably combine the CDRs and FWRs derived from naturally occurring antibody sequences to engineer an antibody with high affinity and specificity. To illustrate this approach, we selected a representative SARS-CoV-2-neutralizing antibody, H4, which was identified and isolated previously based on the predominant germlines that were employed in a human host to target the SARS-CoV-2-human ACE2 receptor interaction. Compared to screening vast CDR libraries for affinity enhancements, our approach identified fewer than 100 antibody framework–CDR combinations, from which we screened and selected an antibody (CB79) that showed a reduced dissociation rate and improved affinity against the SARS-CoV-2 spike protein (7-fold) when compared to H4. The improved affinity also translated into improved neutralization (>75-fold improvement) of SARS-CoV-2. Our rapid and robust approach for optimizing antibodies from parts without the need for tedious structure-guided CDR optimization will have broad utility for biotechnological applications.

[end paper 54]

[begin paper 55]

PMCID = 9642992

Title = Structure of SARS-CoV-2 M protein in lipid nanodiscs

SARS-CoV-2 encodes four structural proteins incorporated into virions, spike (S), envelope (E), nucleocapsid (N), and membrane (M). M plays an essential role in viral assembly by organizing other structural proteins through physical interactions and directing them to sites of viral budding. As the most abundant protein in the viral envelope and a target of patient antibodies, M is a compelling target for vaccines and therapeutics. Still, the structure of M and molecular basis for its role in virion formation are unknown. Here, we present the cryo-EM structure of SARS-CoV-2 M in lipid nanodiscs to 3.5 Å resolution. M forms a 50 kDa homodimer that is structurally related to the SARS-CoV-2 ORF3a viroporin, suggesting a shared ancestral origin. Structural comparisons reveal how intersubunit gaps create a small, enclosed pocket in M and large open cavity in ORF3a, consistent with a structural role and ion channel activity, respectively. M displays a strikingly electropositive cytosolic surface that may be important for interactions with N, S, and viral RNA. Molecular dynamics simulations show a high degree of structural rigidity in a simple lipid bilayer and support a role for M homodimers in scaffolding viral assembly. Together, these results provide insight into roles for M in coronavirus assembly and structure.

[end paper 55]

[begin paper 56]

PMCID = 9300480

Title = Databases, DrugBank, and virtual screening platforms for therapeutic development

The upsurge of the severe acute respiratory syndrome-Coronavirus-2 (SARS-CoV-2) has turned into a global health disaster. Many remodeled medications were suggested for treatment in the early stages of this pandemic, but these dosages afterward came across with distinct offshoots. Thus, these consequences compelled the scientists to develop new drugs using various antiviral, antiinflammatory, antibacterial, and phytochemical compounds. A handful of drugs have been scrutinized in silico, in vitro, plus through human trials such as anti-SARS-CoV-2 agents and made available as various databases by various scientific communities. The SARS-CoV-2 pandemic databases are designed to allay difficulties associated with this scenario. Some of the popular databases are GESS (global evaluation of SARS-CoV-2/HCoV-19 sequences) which gives a thorough study of data based on tenfold of thousands of complete coverage and quality of SARS-CoV-2 genomes, CORona Drug InTERactions (CORDITE) database for SARS-CoV-2 which profoundly combines the understanding of potential drugs and make it available for scientists and medicos. SARSCOVIDB set one’s sights to merge all differential gene expression data, at mRNA and protein levels, helping to accelerate analysis and research on the molecular impact of covid-19. This chapter aims to provide a piece of complete information about the SARS-CoV-2 virus databases, potentially available drugs, and virtual screening methods. And also provides a different webserver to reach out for information related to the COVID-19 pandemic and its future.

[end paper 56]

[begin paper 57]

PMCID = 8495465

Title = Arterial Thromboembolism



[end paper 57]

[begin paper 58]

PMCID = 7803919

Title = A domain-based vaccine construct against SARS-CoV-2, the causative agent of COVID-19 pandemic: development of self-amplifying mRNA and peptide vaccines

  Introduction: Coronavirus disease 2019 (COVID-19) is undoubtedly the most challenging pandemic in the current century with more than 293,241 deaths worldwide since its emergence in late 2019 (updated May 13, 2020). COVID-19 is caused by a novel emerged coronavirus named severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Today, the world needs crucially to develop a prophylactic vaccine scheme for such emerged and emerging infectious pathogens. 

 Methods: In this study, we have targeted spike (S) glycoprotein, as an important surface antigen to identify its B- and T-cell immunodominant regions. We have conducted a multi-method B-cell epitope (BCE) prediction approach using different predictor algorithms to discover the most potential BCEs. Besides, we sought among a pool of MHC class I and II-associated peptide binders provided by the IEDB server through the strict cut-off values. To design a broad-coverage vaccine, we carried out a population coverage analysis for a set of candidate T-cell epitopes and based on the HLA allele frequency in the top most-affected countries by COVID-19 (update 02 April 2020). 

 Results: The final determined B- and T-cell epitopes were mapped on the S glycoprotein sequence, and three potential hub regions covering the largest number of overlapping epitopes were identified for the vaccine designing (I531–N711; T717–C877; and V883–E973). Here, we have designed two domain-based constructs to be produced and delivered through the recombinant protein- and gene-based approaches, including (i) an adjuvanted domain-based protein vaccine construct (DPVC), and (ii) a self-amplifying mRNA vaccine (SAMV) construct. The safety, stability, and immunogenicity of the DPVC were validated using the integrated sequential (i.e. allergenicity, autoimmunity, and physicochemical features) and structural (i.e. molecular docking between the vaccine and human Toll-like receptors (TLRs) 4 and 5) analysis. The stability of the docked complexes was evaluated using the molecular dynamics (MD) simulations. 

 Conclusion: These rigorous in silico validations supported the potential of the DPVC and SAMV to promote both innate and specific immune responses in preclinical studies. 

[end paper 58]

[begin paper 59]

PMCID = 9729196

Title = An in silico method to assess antibody fragment polyreactivity

Antibodies are essential biological research tools and important therapeutic agents, but some exhibit non-specific binding to off-target proteins and other biomolecules. Such polyreactive antibodies compromise screening pipelines, lead to incorrect and irreproducible experimental results, and are generally intractable for clinical development. Here, we design a set of experiments using a diverse naïve synthetic camelid antibody fragment (nanobody) library to enable machine learning models to accurately assess polyreactivity from protein sequence (AUC > 0.8). Moreover, our models provide quantitative scoring metrics that predict the effect of amino acid substitutions on polyreactivity. We experimentally test our models’ performance on three independent nanobody scaffolds, where over 90% of predicted substitutions successfully reduced polyreactivity. Importantly, the models allow us to diminish the polyreactivity of an angiotensin II type I receptor antagonist nanobody, without compromising its functional properties. We provide a companion web-server that offers a straightforward means of predicting polyreactivity and polyreactivity-reducing mutations for any given nanobody sequence.

Off-target binding hinders the development of therapeutic antibodies and reproducibility in basic research settings. Here the authors develop a method to quantify and reduce the polyreactivity of antibody fragments based on protein sequence alone.

[end paper 59]

[begin paper 60]

PMCID = 11268725

Title = Meeting report: 36th International Conference on Antiviral Research in Lyon, France – March 13–17, 2023

The 36th International Conference on Antiviral Research (ICAR), sponsored by the International Society for Antiviral Research (ISAR), was held March 13–17, 2023, in Lyon, France, and concurrently through an interactive remote meeting platform. Here we provide a report summarizing the presentations at the 36th ICAR, including the ISAR speaker awards. We also detail special events, sessions, and additional awards conferred at the meeting. ICAR returned to in-person meetings in 2022, convening in Seattle, WA, USA. The 36th ICAR is the first in-person meeting of the society in Europe since the beginning of the COVID-19 pandemic, which restricted most events to virtual attendance to help mitigate the spread and subsequent public health impact of SARS-CoV-2. An exceptionally high number of registrants and record attendance at this year’s ICAR, along with a vast array of demonstrable expertise in a variety of antiviral research-related fields, reflected a strong and growing antiviral research community committed to improving health outcomes from viral diseases, including SARS-CoV-2, and to future pandemic preparedness. This report highlights the breadth of expertise, quality of research, and notable advancements that were contributed by members of ISAR and other participants at the meeting. ICAR aims to continue to provide a platform for sharing information, fostering collaborations, and supporting trainees in the field of antiviral research. The 37th ICAR will be held in Gold Coast, Australia, May 20–24, 2024.

[end paper 60]

[begin paper 61]

PMCID = 11515141

Title = Discovery of nanobodies: a comprehensive review of their applications and potential over the past five years

Nanobodies (Nbs) are antibody fragments derived from heavy-chain-only IgG antibodies found in the Camelidae family as well as cartilaginous fish. Their unique structural and functional properties, such as their small size, the ability to be engineered for high antigen-binding affinity, stability under extreme conditions, and ease of production, have made them promising tools for diagnostics and therapeutics. This potential was realized in 2018 with the approval of caplacizumab, the world’s first Nb-based drug. Currently, Nbs are being investigated in clinical trials for a broad range of treatments, including targeted therapies against PDL1 and Epidermal Growth Factor Receptor (EGFR), cardiovascular diseases, inflammatory conditions, and neurodegenerative disorders such as Alzheimer’s disease, Parkinson’s disease, and amyotrophic lateral sclerosis. They are also being studied for their potential for detecting and imaging autoimmune conditions and infectious diseases such as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). A variety of methods are now available to generate target-specific Nbs quickly and efficiently at low costs, increasing their accessibility. This article examines these diverse applications of Nbs and their promising roles. Only the most recent articles published in the last five years have been used to summarize the most advanced developments in the field.

[end paper 61]

[begin paper 62]

PMCID = 11528319

Title = Protein–protein and protein–nucleic acid binding site prediction via interpretable hierarchical geometric deep learning

Identification of protein–protein and protein–nucleic acid binding sites provides insights into biological processes related to protein functions and technical guidance for disease diagnosis and drug design. However, accurate predictions by computational approaches remain highly challenging due to the limited knowledge of residue binding patterns. The binding pattern of a residue should be characterized by the spatial distribution of its neighboring residues combined with their physicochemical information interaction, which yet cannot be achieved by previous methods. Here, we design GraphRBF, a hierarchical geometric deep learning model to learn residue binding patterns from big data. To achieve it, GraphRBF describes physicochemical information interactions by designing an enhanced graph neural network and characterizes residue spatial distributions by introducing a prioritized radial basis function neural network. After training and testing, GraphRBF shows great improvements over existing state-of-the-art methods and strong interpretability of its learned representations. Applying GraphRBF to the SARS-CoV-2 omicron spike protein, it successfully identifies known epitopes of the protein. Moreover, it predicts multiple potential binding regions for new nanobodies or even new drugs with strong evidence. A user-friendly online server for GraphRBF is freely available at http://liulab.top/GraphRBF/server.

[end paper 62]

[begin paper 63]

PMCID = 10733906

Title = Simplifying complex antibody engineering using machine learning

Machine learning is transforming antibody engineering by enabling the generation of drug-like monoclonal antibodies with unprecedented efficiency. Unsupervised algorithms trained on massive and diverse protein sequence datasets facilitate the prediction of panels of antibody variants with native-like intrinsic properties (e.g., high stability), greatly reducing the amount of subsequent experimentation needed to identify specific candidates that also possess desired extrinsic properties (e.g., high affinity). Additionally, supervised algorithms, which are trained on deep sequencing datasets obtained after enrichment of in vitro antibody libraries for one or more specific extrinsic properties, enable the prediction of antibody variants with desired combinations of extrinsic properties without the need for additional screening. Here we review recent advances using both machine learning approaches and how they are impacting the field of antibody engineering as well as key outstanding challenges and opportunities for these paradigm-changing methods.

Machine learning is transforming antibody engineering by efficiently generating drug-like antibodies. Unsupervised algorithms trained on large protein datasets enable prediction of antibody variants with native-like intrinsic properties, reducing the amount of experimentation necessary for identifying candidates with desired extrinsic properties. Supervised algorithms trained on deep sequencing datasets enable prediction of antibody variants with favorable extrinsic properties without additional screening. We review how both approaches are transforming the field of antibody engineering as well as key challenges for these paradigm-changing methods.

[end paper 63]

[begin paper 64]

PMCID = 10179460

Title = Computational and Enzymatic Studies of Sartans in SARS-CoV-2 Spike RBD-ACE2 Binding: The Role of Tetrazole and Perspectives as Antihypertensive and COVID-19 Therapeutics

This study is an extension of current research into a novel class of synthetic antihypertensive drugs referred to as “bisartans”, which are bis-alkylated imidazole derivatives bearing two symmetric anionic biphenyltetrazoles. Research to date indicates that bisartans are superior to commercially available hypertension drugs, since the former undergo stronger docking to angiotensin-converting enzyme 2 (ACE2). ACE2 is the key receptor involved in SARS-CoV-2 entry, thus initiating COVID-19 infection and in regulating levels of vasoactive peptides such as angiotensin II and beneficial heptapeptides A(1-7) and Alamandine in the renin–angiotensin system (RAS). In previous studies using in vivo rabbit-iliac arterial models, we showed that Na+ or K+ salts of selected Bisartans initiate a potent dose–response inhibition of vasoconstriction. Furthermore, computational studies revealed that bisartans undergo stable binding to the vital interfacial region between ACE2 and the SARS-CoV-2 “receptor binding domain” (i.e., the viral RBD). Thus, bisartan homologs are expected to interfere with SARS-CoV-2 infection and/or suppress disease expression in humans. The primary goal of this study was to investigate the role of tetrazole in binding and the network of amino acids of SARS-CoV-2 Spike RBD-ACE2 complex involved in interactions with sartans. This study would, furthermore, allow the expansion of the synthetic space to create a diverse suite of new bisartans in conjunction with detailed computational and in vitro antiviral studies. A critical role for tetrazole was uncovered in this study, shedding light on the vital importance of this group in the binding of sartans and bisartans to the ACE2/Spike complex. The in silico data predicting an interaction of tetrazole-containing sartans with ACE2 were experimentally validated by the results of surface plasmon resonance (SPR) analyses performed with a recombinant human ACE2 protein.

[end paper 64]

[begin paper 65]

PMCID = 9294429

Title = Machine-designed biotherapeutics: opportunities, feasibility and advantages of deep learning in computational antibody discovery

Antibodies are versatile molecular binders with an established and growing role as therapeutics. Computational approaches to developing and designing these molecules are being increasingly used to complement traditional lab-based processes. Nowadays, in silico methods fill multiple elements of the discovery stage, such as characterizing antibody–antigen interactions and identifying developability liabilities. Recently, computational methods tackling such problems have begun to follow machine learning paradigms, in many cases deep learning specifically. This paradigm shift offers improvements in established areas such as structure or binding prediction and opens up new possibilities such as language-based modeling of antibody repertoires or machine-learning-based generation of novel sequences. In this review, we critically examine the recent developments in (deep) machine learning approaches to therapeutic antibody design with implications for fully computational antibody design.

[end paper 65]

[begin paper 66]

PMCID = 11140640

Title = Multi-epitope vaccine against SARS-CoV-2 targeting the spike RBD: an immunoinformatics approach

Aim: We designed a SARS-CoV-2 epitope vaccine based on the receptor-binding domain (RBD) in virus spike protein. Methods: RT-PCR performed on nasopharyngeal swab COVID-19 patients. After registering RBD region in the GenBank, physicochemical parameters, secondary structure, homology modeling, 3D structure of RBD region and antigenicity were determined using ProtParam ExPASy, PSIPRED, MolProbity, IEDB and Vaxijen online tools, respectively. Results: B and T cell epitopes were predicted in terms of non-allergenicity and antigenicity. MolProbity analysis provided a qualitative model for RBD. The homology model showed that most of the residues are in optimal district of energy. Conclusion: High immunogenicity score of epitopes indicates promising candidates for the development of multi-epitope vaccines. It may help to develop an effective vaccine.

In order to identify the sequence of RBD region of S protein in SARS-CoV-2, RT-PCR test was performed on nasopharyngeal swab samples of four COVID-19 patients referred to Imam Khomeini Hospital in Ardabil. After registering the sequence of the RBD region in the GenBank database, the physicochemical parameters, secondary structure, homology modeling, 3D structure of the RBD region and antigenicity were determined using ProtParam ExPASy, PSIPRED, MolProbity, IEDB and Vaxijen online tools, respectively.

The prediction of B cell and T cell epitopes was appropriate in terms of non-allergenic and antigenicity.

The homology model showed that most of the residues are in optimal district of energy.

MolProbity analysis provided a good quality model for the receptor-binding domain.

[end paper 66]

[begin paper 67]

PMCID = 11679499

Title = Vaccine Strategies Against RNA Viruses: Current Advances and Future Directions

The development of vaccines against RNA viruses has undergone a rapid evolution in recent years, particularly driven by the COVID-19 pandemic. This review examines the key roles that RNA viruses, with their high mutation rates and zoonotic potential, play in fostering vaccine innovation. We also discuss both traditional and modern vaccine platforms and the impact of new technologies, such as artificial intelligence, on optimizing immunization strategies. This review evaluates various vaccine platforms, ranging from traditional approaches (inactivated and live-attenuated vaccines) to modern technologies (subunit vaccines, viral and bacterial vectors, nucleic acid vaccines such as mRNA and DNA, and phage-like particle vaccines). To illustrate these platforms’ practical applications, we present case studies of vaccines developed for RNA viruses such as SARS-CoV-2, influenza, Zika, and dengue. Additionally, we assess the role of artificial intelligence in predicting viral mutations and enhancing vaccine design. The case studies underscore the successful application of RNA-based vaccines, particularly in the fight against COVID-19, which has saved millions of lives. Current clinical trials for influenza, Zika, and dengue vaccines continue to show promise, highlighting the growing efficacy and adaptability of these platforms. Furthermore, artificial intelligence is driving improvements in vaccine candidate optimization and providing predictive models for viral evolution, enhancing our ability to respond to future outbreaks. Advances in vaccine technology, such as the success of mRNA vaccines against SARS-CoV-2, highlight the potential of nucleic acid platforms in combating RNA viruses. Ongoing trials for influenza, Zika, and dengue demonstrate platform adaptability, while artificial intelligence enhances vaccine design by predicting viral mutations. Integrating these innovations with the One Health approach, which unites human, animal, and environmental health, is essential for strengthening global preparedness against future RNA virus threats.

[end paper 67]

[begin paper 68]

PMCID = 11768830

Title = Advances and Challenges in Antiviral Development for Respiratory Viruses

The development of antivirals for respiratory viruses has advanced markedly in response to the growing threat of pathogens such as Influenzavirus (IAV), respiratory syncytial virus (RSV), and SARS-CoV-2. This article reviews the advances and challenges in this field, highlighting therapeutic strategies that target critical stages of the viral replication cycle, including inhibitors of viral entry, replication, and assembly. In addition, innovative approaches such as inhibiting host cellular proteins to reduce viral resistance and repurposing existing drugs are explored, using advanced bioinformatics tools that optimize the identification of antiviral candidates. The analysis also covers emerging technologies such as nanomedicine and CRISPR gene editing, which promise to improve the stability and efficacy of treatments. While current antivirals offer valuable options, they face challenges such as viral evolution and the need for accessible treatments for vulnerable populations. This article underscores the importance of continued innovation in biotechnology to overcome these limitations and provide safe and effective treatments. Combining traditional and advanced approaches in developing antivirals is essential in order to address respiratory viral diseases that affect global health.

[end paper 68]

[begin paper 69]

PMCID = 9123296

Title = mRNA-based therapeutics: powerful and versatile tools to combat diseases

The therapeutic use of messenger RNA (mRNA) has fueled great hope to combat a wide range of incurable diseases. Recent rapid advances in biotechnology and molecular medicine have enabled the production of almost any functional protein/peptide in the human body by introducing mRNA as a vaccine or therapeutic agent. This represents a rising precision medicine field with great promise for preventing and treating many intractable or genetic diseases. In addition, in vitro transcribed mRNA has achieved programmed production, which is more effective, faster in design and production, as well as more flexible and cost-effective than conventional approaches that may offer. Based on these extraordinary advantages, mRNA vaccines have the characteristics of the swiftest response to large-scale outbreaks of infectious diseases, such as the currently devastating pandemic COVID-19. It has always been the scientists’ desire to improve the stability, immunogenicity, translation efficiency, and delivery system to achieve efficient and safe delivery of mRNA. Excitingly, these scientific dreams have gradually been realized with the rapid, amazing achievements of molecular biology, RNA technology, vaccinology, and nanotechnology. In this review, we comprehensively describe mRNA-based therapeutics, including their principles, manufacture, application, effects, and shortcomings. We also highlight the importance of mRNA optimization and delivery systems in successful mRNA therapeutics and discuss the key challenges and opportunities in developing these tools into powerful and versatile tools to combat many genetic, infectious, cancer, and other refractory diseases.

[end paper 69]

[begin paper 70]

PMCID = 9515332

Title = Reduced B cell antigenicity of Omicron lowers host serologic response

The SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. However, the mechanism remains unclear. Here, we find using a geometric deep-learning model that Omicron’s extensively mutated receptor binding site (RBS) features reduced antigenicity compared with previous variants. Mice immunization experiments with different recombinant receptor binding domain (RBD) variants confirm that the serological response to Omicron is drastically attenuated and less potent. Analyses of serum cross-reactivity and competitive ELISA reveal a reduction in antibody response across both variable and conserved RBD epitopes. Computational modeling confirms that the RBS has a potential for further antigenicity reduction while retaining efficient receptor binding. Finally, we find a similar trend of antigenicity reduction over decades for hCoV229E, a common cold coronavirus. Thus, our study explains the reduced antibody titers associated with Omicron infection and reveals a possible trajectory of future viral evolution.

SARS-CoV-2 Omicron variant evades most neutralizing vaccine-induced antibodies and is associated with lower antibody titers upon breakthrough infections than previous variants. Tubiana et al. investigate the underlying mechanism using geometric deep learning, mice immunization experiments, and biochemical assays. Mutations reduce antigenicity of the receptor binding site, leading to lower antibody response.

[end paper 70]

[begin paper 71]

PMCID = 7773313

Title = The find of COVID-19 vaccine: Challenges and opportunities

Severe acute respiratory syndrome coronavirus (SARS-CoV-2), a novel corona virus, causing COVID-19 with Flu-like symptoms is the first alarming pandemic of the third millennium. SARS-CoV-2 belongs to beta coronavirus as Middle East respiratory syndrome coronavirus (MERS-CoV). Pandemic COVID-19 owes devastating mortality and destructively exceptional consequences on Socio-Economics life around the world. Therefore, the current review is redirected to the scientific community to owe comprehensive visualization about SARS-CoV-2 to tackle the current pandemic. As systematically shown through the current review, it indexes unmet medical problem of COVID-19 in view of public health and vaccination discovery for the infectious SARS-CoV-2; it is currently under-investigational therapeutic protocols, and next possible vaccines. Furthermore, the review extensively reports the precautionary measures to achieve" COVID-19/Flatten the curve". It is concluded that vaccines formulation within exceptional no time in this pandemic is highly recommended, via following the same protocols of previous pandemics; MERS-CoV and SARS-CoV, and excluding some initial steps of vaccination development process.

[end paper 71]

[begin paper 72]

PMCID = 11380031

Title = Antiprotozoal peptide prediction using machine learning with effective feature selection techniques

Protozoal pathogens pose a considerable threat, leading to notable mortality rates and the ongoing challenge of developing resistance to drugs. This situation underscores the urgent need for alternative therapeutic approaches. Antimicrobial peptides stand out as promising candidates for drug development. However, there is a lack of published research focusing on predicting antimicrobial peptides specifically targeting protozoal pathogens. In this study, we introduce a successful machine learning-based framework designed to predict potential antiprotozoal peptides effective against protozoal pathogens.

The primary objective of this study is to classify and predict antiprotozoal peptides using diverse negative datasets.

A comprehensive literature review was conducted to gather experimentally validated antiprotozoal peptides, forming the positive dataset for our study. To construct a robust machine learning classifier, multiple negative datasets were incorporated, including (i) non-antimicrobial, (ii) antiviral, (iii) antibacterial, (iv) antifungal, and (v) antimicrobial peptides excluding those targeting protozoal pathogens. Various compositional features of the peptides were extracted using the pfeature algorithm. Two feature selection methods, SVC-L1 and mRMR, were employed to identify highly relevant features crucial for distinguishing between the positive and negative datasets. Additionally, five popular classifiers i.e. Decision Tree, Random Forest, Support Vector Machine, Logistic Regression, and XGBoost were used to build efficient decision models.

XGBoost was the most effective in classifying antiprotozoal peptides from each negative dataset based on the features selected by the mRMR feature selection method. The proposed machine learning framework efficiently differentiate the antiprotozoal peptides from (i) non-antimicrobial (ii) antiviral (iii) antibacterial (iv) antifungal and (v) antimicrobial with accuracy of 97.27 %, 93.64 %, 86.36 %, 90.91 %, and 89.09 % respectively on the validation dataset.

The models are incorporated in a user-friendly web server (www.soodlab.com/appred) to predict the antiprotozoal activity of given peptides.

[end paper 72]

[begin paper 73]

PMCID = 8077567

Title = Sentinel cells enable genetic detection of SARS-CoV-2 Spike protein

The COVID-19 pandemic has demonstrated the need for exploring different diagnostic and therapeutic modalities to tackle future viral threats. In this vein, we propose the idea of sentinel cells, cellular biosensors capable of detecting viral antigens and responding to them with customizable responses. Using SARS-CoV-2 as a test case, we developed a live cell sensor (SARSNotch) using a de novo-designed protein binder against the SARS-CoV-2 Spike protein. SARSNotch is capable of driving custom genetically-encoded payloads in immortalized cell lines or in primary T lymphocytes in response to purified SARS-CoV-2 Spike or in the presence of Spike-expressing cells. Furthermore, SARSNotch is functional in a cellular system used in directed evolution platforms for development of better binders or therapeutics. In keeping with the rapid dissemination of scientific knowledge that has characterized the incredible scientific response to the ongoing pandemic, we extend an open invitation for others to make use of and improve SARSNotch sentinel cells in the hopes of unlocking the potential of the next generation of smart antiviral therapeutics.

[end paper 73]

[begin paper 74]

PMCID = 10536364

Title = Bridging the Gap: Can COVID-19 Research Help Combat African Swine Fever?

African swine fever (ASF) is a highly contagious and economically devastating disease affecting domestic pigs and wild boar, caused by African swine fever virus (ASFV). Despite being harmless to humans, ASF poses significant challenges to the swine industry, due to sudden losses and trade restrictions. The ongoing COVID-19 pandemic has spurred an unparalleled global research effort, yielding remarkable advancements across scientific disciplines. In this review, we explore the potential technological spillover from COVID-19 research into ASF. Specifically, we assess the applicability of the diagnostic tools, vaccine development strategies, and biosecurity measures developed for COVID-19 for combating ASF. Additionally, we discuss the lessons learned from the pandemic in terms of surveillance systems and their implications for managing ASF. By bridging the gap between COVID-19 and ASF research, we highlight the potential for interdisciplinary collaboration and technological spillovers in the battle against ASF.

[end paper 74]

[begin paper 75]

PMCID = 11631148

Title = Molecular dynamics and machine learning stratify motion-dependent activity profiles of S-layer destabilizing nanobodies

Nanobody (Nb)-induced disassembly of surface array protein (Sap) S-layers, a two-dimensional paracrystalline protein lattice from Bacillus anthracis, has been presented as a therapeutic intervention for lethal anthrax infections. However, only a subset of existing Nbs with affinity to Sap exhibit depolymerization activity, suggesting that affinity and epitope recognition are not enough to explain inhibitory activity. In this study, we performed all-atom molecular dynamics simulations of each Nb bound to the Sap binding site and trained a collection of machine learning classifiers to predict whether each Nb induces depolymerization. We used feature importance analysis to filter out unnecessary features and engineered remaining features to regularize the feature landscape and encourage learning of the depolymerization mechanism. We find that, while not enforced in training, a gradient-boosting decision tree is able to reproduce the experimental activities of inhibitory Nbs while maintaining high classification accuracy, whereas neural networks were only able to discriminate between classes. Further feature analysis revealed that inhibitory Nbs restrain Sap motions toward an inhibitory conformational state described by domain–domain clamping and induced twisting of domains normal to the lattice plane. We believe these motions drive Sap lattice depolymerization and can be used as design targets for improved Sap-inhibitory Nbs. Finally, we expect our method of study to apply to S-layers that serve as virulence factors in other pathogens, paving the way forward for Nb therapeutics that target depolymerization mechanisms.

[end paper 75]

[begin paper 76]

PMCID = 8891432

Title = Probabilistic classification of anti‐SARS‐CoV‐2 antibody responses improves seroprevalence estimates

Population‐level measures of seropositivity are critical for understanding the epidemiology of an emerging pathogen, yet most antibody tests apply a strict cutoff for seropositivity that is not learnt in a data‐driven manner, leading to uncertainty when classifying low‐titer responses. To improve upon this, we evaluated cutoff‐independent methods for their ability to assign likelihood of SARS‐CoV‐2 seropositivity to individual samples.

Using robust ELISAs based on SARS‐CoV‐2 spike (S) and the receptor‐binding domain (RBD), we profiled antibody responses in a group of SARS‐CoV‐2 PCR+ individuals (n = 138). Using these data, we trained probabilistic learners to assign likelihood of seropositivity to test samples of unknown serostatus (n = 5100), identifying a support vector machines‐linear discriminant analysis learner (SVM‐LDA) suited for this purpose.

In the training data from confirmed ancestral SARS‐CoV‐2 infections, 99% of participants had detectable anti‐S and ‐RBD IgG in the circulation, with titers differing > 1000‐fold between persons. In data of otherwise healthy individuals, 7.2% (n = 367) of samples were of uncertain serostatus, with values in the range of 3‐6SD from the mean of pre‐pandemic negative controls (n = 595). In contrast, SVM‐LDA classified 6.4% (n = 328) of test samples as having a high likelihood (> 99% chance) of past infection, 4.5% (n = 230) to have a 50–99% likelihood, and 4.0% (n = 203) to have a 10–49% likelihood. As different probabilistic approaches were more consistent with each other than conventional SD‐based methods, such tools allow for more statistically‐sound seropositivity estimates in large cohorts.

Probabilistic antibody testing frameworks can improve seropositivity estimates in populations with large titer variability.

Correctly classifying low‐titer antibody responses is challenging using conventional assay cutoffs. To address this issue, we trained suitable probabilistic learners to assign likelihood of seropositivity. These more quantitative methods improve seroprevalence estimates and have potential application to the clinical setting. 

[end paper 76]

[begin paper 77]

PMCID = 11659936

Title = Single domain antibody: Development and application in biotechnology and biopharma

Heavy‐chain antibodies (HCAbs) are a unique type of antibodies devoid of light chains, and comprised of two heavy chains‐only that recognize their cognate antigen by virtue of a single variable domain also referred to as VHH, single domain antibody (sdAb), or nanobody (Nb). These functional HCAbs, serendipitous discovered about three decades ago, are exclusively found in camelids, comprising dromedaries, camels, llamas, and vicugnas. Nanobodies have become an essential tool in biomedical research and medicine, both in diagnostics and therapeutics due to their beneficial properties: small size, high stability, strong antigen‐binding affinity, low immunogenicity, low production cost, and straightforward engineering into more potent affinity reagents. The occurrence of HCAbs in camelids remains intriguing. It is believed to be an evolutionary adaptation, equipping camelids with a robust adaptive immune defense suitable to respond to the pressure from a pathogenic invasion necessitating a more profound antigen recognition and neutralization. This evolutionary innovation led to a simplified HCAb structure, possibly supported by genetic mutations and drift, allowing adaptive mutation and diversification in the heavy chain variable gene and constant gene regions. Beyond understanding their origins, the application of nanobodies has significantly advanced over the past 30 years. Alongside expanding laboratory research, there has been a rapid increase in patent application for nanobodies. The introduction of commercial nanobody drugs such as Cablivi, Nanozora, Envafolimab, and Carvykti has boosted confidence among in their potential. This review explores the evolutionary history of HCAbs, their ontogeny, and applications in biotechnology and pharmaceuticals, focusing on approved and ongoing medical research pipelines.

[end paper 77]

[begin paper 78]

PMCID = 11205781

Title = In Silico Exploration of CD200 as a Therapeutic Target for COVID-19

SARS-CoV-2, the pathogen causing COVID-19, continues to pose a significant threat to public health and has had major economic implications. Developing safe and effective vaccines and therapies offers a path forward for overcoming the COVID-19 pandemic. The presented study, performed by using the informational spectrum method (ISM), representing an electronic biology-based tool for analysis of protein–protein interactions, identified the highly conserved region of spike protein (SP) from SARS-CoV-2 virus, which is essential for recognition and targeting between the virus and its protein interactors on the target cells. This domain is suggested as a promising target for the drug therapy and vaccines, which could be effective against all currently circulating variants of SARS-CoV-2 viruses. The analysis of the virus/host interaction, performed by the ISM, also revealed OX-2 membrane glycoprotein (CD200) as a possible interactor of SP, which could serve as a novel therapeutic target for COVID-19 disease.

[end paper 78]

[begin paper 79]

PMCID = 9448752

Title = MVsim is a toolset for quantifying and designing multivalent interactions

Arising through multiple binding elements, multivalency can specify the avidity, duration, cooperativity, and selectivity of biomolecular interactions, but quantitative prediction and design of these properties has remained challenging. Here we present MVsim, an application suite built around a configurational network model of multivalency to facilitate the quantification, design, and mechanistic evaluation of multivalent binding phenomena through a simple graphical user interface. To demonstrate the utility and versatility of MVsim, we first show that both monospecific and multispecific multivalent ligand-receptor interactions, with their noncanonical binding kinetics, can be accurately simulated. Further, to illustrate the conceptual insights into multivalent systems that MVsim can provide, we apply it to quantitatively predict the ultrasensitivity and performance of multivalent-encoded protein logic gates, evaluate the inherent programmability of multispecificity for selective receptor targeting, and extract rate constants of conformational switching for the SARS-CoV-2 spike protein and model its binding to ACE2 as well as multivalent inhibitors of this interaction. MVsim and instructional tutorials are freely available at https://sarkarlab.github.io/MVsim/.

Arising through multiple binding elements, multivalency can specify the avidity, duration, cooperativity, and selectivity of biomolecular interactions, but quantitative prediction and design of these properties has remained challenging. Here the authors enable facile analysis and engineering of multivalent binding by developing MVsim, a simulator that incorporates biochemical and biophysical parameters of interacting molecules and is accessible through a graphical user interface.

[end paper 79]

[begin paper 80]

PMCID = 8202428

Title = Mapping Potential Antigenic Drift Sites (PADS) on SARS-CoV-2 Spike in Continuous Epitope-Paratope Space

SARS-CoV-2 mutations with antigenic effects pose a risk to immunity developed through vaccination and natural infection. While vaccine updates for current variants of concern (VOCs) are underway, it is likewise important to prepare for further antigenic mutations as the virus navigates the heterogeneous global landscape of host immunity. Toward this end, a wealth of data and tools exist that can augment existing genetic surveillance of VOC evolution. In this study, we integrate published datasets describing genetic, structural, and functional constraints on mutation along with computational analyses of antibody-spike co-crystal structures to identify a set of potential antigenic drift sites (PADS) within the receptor binding domain (RBD) and N-terminal domain (NTD) of SARS-CoV-2 spike protein. Further, we project the PADS set into a continuous epitope-paratope space to facilitate interpretation of the degree to which newly observed mutations might be antigenically synergistic with existing VOC mutations, and this representation suggests that functionally convergent and synergistic antigenic mutations are accruing across VOC NTDs. The PADS set and synergy visualization serve as a reference as new mutations are detected on VOCs, enable proactive investigation of potentially synergistic mutations, and offer guidance to antibody and vaccine design efforts.

[end paper 80]

[begin paper 81]

PMCID = 11133524

Title = Prospects for the computational humanization of antibodies and nanobodies

To be viable therapeutics, antibodies must be tolerated by the human immune system. Rational approaches to reduce the risk of unwanted immunogenicity involve maximizing the ‘humanness’ of the candidate drug. However, despite the emergence of new discovery technologies, many of which start from entirely human gene fragments, most antibody therapeutics continue to be derived from non-human sources with concomitant humanization to increase their human compatibility. Early experimental humanization strategies that focus on CDR loop grafting onto human frameworks have been critical to the dominance of this discovery route but do not consider the context of each antibody sequence, impacting their success rate. Other challenges include the simultaneous optimization of other drug-like properties alongside humanness and the humanization of fundamentally non-human modalities such as nanobodies. Significant efforts have been made to develop in silico methodologies able to address these issues, most recently incorporating machine learning techniques. Here, we outline these recent advancements in antibody and nanobody humanization, focusing on computational strategies that make use of the increasing volume of sequence and structural data available and the validation of these tools. We highlight that structural distinctions between antibodies and nanobodies make the application of antibody-focused in silico tools to nanobody humanization non-trivial. Furthermore, we discuss the effects of humanizing mutations on other essential drug-like properties such as binding affinity and developability, and methods that aim to tackle this multi-parameter optimization problem.

[end paper 81]

[begin paper 82]

PMCID = 11361996

Title = Construction and validation of a synthetic phage-displayed nanobody library

Nanobodies derived from camelids and sharks offer unique advantages in therapeutic applications due to their ability to bind to epitopes that were previously inaccessible. Traditional methods of nanobody development face challenges such as ethical concerns and antigen toxicity. Our study presents a synthetic, phage-displayed nanobody library using trinucleotide-directed mutagenesis technology, which allows precise amino acid composition in complementarity-determining regions (CDRs), with a focus on CDR3 diversity. This approach avoids common problems such as frameshift mutations and stop codon insertions associated with other synthetic antibody library construction methods. By analyzing FDA-approved nanobodies and Protein Data Bank sequences, we designed sub-libraries with different CDR3 lengths and introduced amino acid substitutions to improve solubility. The validation of our library through the successful isolation of nanobodies against targets such as PD-1, ATXN1 and STAT3 demonstrates a versatile and ethical platform for the development of high specificity and affinity nanobodies and represents a significant advance in biotechnology.

[end paper 82]

[begin paper 83]

PMCID = 11172335

Title = Artificial Intelligence in Point-of-Care Biosensing: Challenges and Opportunities

The integration of artificial intelligence (AI) into point-of-care (POC) biosensing has the potential to revolutionize diagnostic methodologies by offering rapid, accurate, and accessible health assessment directly at the patient level. This review paper explores the transformative impact of AI technologies on POC biosensing, emphasizing recent computational advancements, ongoing challenges, and future prospects in the field. We provide an overview of core biosensing technologies and their use at the POC, highlighting ongoing issues and challenges that may be solved with AI. We follow with an overview of AI methodologies that can be applied to biosensing, including machine learning algorithms, neural networks, and data processing frameworks that facilitate real-time analytical decision-making. We explore the applications of AI at each stage of the biosensor development process, highlighting the diverse opportunities beyond simple data analysis procedures. We include a thorough analysis of outstanding challenges in the field of AI-assisted biosensing, focusing on the technical and ethical challenges regarding the widespread adoption of these technologies, such as data security, algorithmic bias, and regulatory compliance. Through this review, we aim to emphasize the role of AI in advancing POC biosensing and inform researchers, clinicians, and policymakers about the potential of these technologies in reshaping global healthcare landscapes.

[end paper 83]

[begin paper 84]

PMCID = 9872955

Title = Selection of target-binding proteins from the information of weakly enriched phage display libraries by deep sequencing and machine learning

Despite the advances in surface-display systems for directed evolution, variants with high affinity are not always enriched due to undesirable biases that increase target-unrelated variants during biopanning. Here, our goal was to design a library containing improved variants from the information of the “weakly enriched” library where functional variants were weakly enriched. Deep sequencing for the previous biopanning result, where no functional antibody mimetics were experimentally identified, revealed that weak enrichment was partly due to undesirable biases during phage infection and amplification steps. The clustering analysis of the deep sequencing data from appropriate steps revealed no distinct sequence patterns, but a Bayesian machine learning model trained with the selected deep sequencing data supplied nine clusters with distinct sequence patterns. Phage libraries were designed on the basis of the sequence patterns identified, and four improved variants with target-specific affinity (EC50 = 80–277 nM) were identified by biopanning. The selection and use of deep sequencing data without undesirable bias enabled us to extract the information on prospective variants. In summary, the use of appropriate deep sequencing data and machine learning with the sequence data has the possibility of finding sequence space where functional variants are enriched.

[end paper 84]

[begin paper 85]

PMCID = 9684400

Title = Development of therapeutic antibodies for the treatment of diseases

Since the first monoclonal antibody drug, muromonab-CD3, was approved for marketing in 1986, 165 antibody drugs have been approved or are under regulatory review worldwide. With the approval of new drugs for treating a wide range of diseases, including cancer and autoimmune and metabolic disorders, the therapeutic antibody drug market has experienced explosive growth. Monoclonal antibodies have been sought after by many biopharmaceutical companies and scientific research institutes due to their high specificity, strong targeting abilities, low toxicity, side effects, and high development success rate. The related industries and markets are growing rapidly, and therapeutic antibodies are one of the most important research and development areas in the field of biology and medicine. In recent years, great progress has been made in the key technologies and theoretical innovations provided by therapeutic antibodies, including antibody–drug conjugates, antibody-conjugated nuclides, bispecific antibodies, nanobodies, and other antibody analogs. Additionally, therapeutic antibodies can be combined with technologies used in other fields to create new cross-fields, such as chimeric antigen receptor T cells (CAR-T), CAR-natural killer cells (CAR-NK), and other cell therapy. This review summarizes the latest approved or in regulatory review therapeutic antibodies that have been approved or that are under regulatory review worldwide, as well as clinical research on these approaches and their development, and outlines antibody discovery strategies that have emerged during the development of therapeutic antibodies, such as hybridoma technology, phage display, preparation of fully human antibody from transgenic mice, single B-cell antibody technology, and artificial intelligence-assisted antibody discovery.

The online version contains supplementary material available at 10.1186/s43556-022-00100-4.

[end paper 85]

[begin paper 86]

PMCID = 11650500

Title = NABP-BERT: NANOBODY®-antigen binding prediction based on bidirectional encoder representations from transformers (BERT) architecture

Antibody-mediated immunity is crucial in the vertebrate immune system. Nanobodies, also known as VHH or single-domain antibodies (sdAbs), are emerging as promising alternatives to full-length antibodies due to their compact size, precise target selectivity, and stability. However, the limited availability of nanobodies (Nbs) for numerous antigens (Ags) presents a significant obstacle to their widespread application. Understanding the interactions between Nbs and Ags is essential for enhancing their binding affinities and specificities. Experimental identification of these interactions is often costly and time-intensive. To address this issue, we introduce NABP-BERT, a deep-learning model based on the BERT architecture, designed to predict NANOBODY®-Ag binding solely from sequence information. Furthermore, we have developed a general pretrained model with transfer capabilities suitable for protein-related tasks, including protein-protein interaction tasks. NABP-BERT focuses on the surrounding amino acid contexts and outperforms existing methods, achieving an AUROC of 0.986 and an AUPR of 0.985.

[end paper 86]

[begin paper 87]

PMCID = 9411858

Title = NanoNet: Rapid and accurate end-to-end nanobody modeling by deep learning

Antibodies are a rapidly growing class of therapeutics. Recently, single domain camelid VHH antibodies, and their recognition nanobody domain (Nb) appeared as a cost-effective highly stable alternative to full-length antibodies. There is a growing need for high-throughput epitope mapping based on accurate structural modeling of the variable domains that share a common fold and differ in the Complementarity Determining Regions (CDRs). We develop a deep learning end-to-end model, NanoNet, that given a sequence directly produces the 3D coordinates of the backbone and Cβ atoms of the entire VH domain. For the Nb test set, NanoNet achieves 3.16Å average RMSD for the most variable CDR3 loops and 2.65Å, 1.73Å for the CDR1, CDR2 loops, respectively. The accuracy for antibody VH domains is even higher: 2.38Å RMSD for CDR3 and 0.89Å, 0.96Å for the CDR1, CDR2 loops, respectively. NanoNet run times allow generation of ∼1M nanobody structures in less than 4 hours on a standard CPU computer enabling high-throughput structure modeling. NanoNet is available at GitHub:  

[end paper 87]

[begin paper 88]

PMCID = 7659715

Title = Crippling life support for SARS-CoV-2 and other viruses through synthetic lethality

Mast et al. highlight common antiviral strategies and propose exploiting the well-established concept of synthetic lethality as an approach to combat viral infections.

With the rapid global spread of SARS-CoV-2, we have become acutely aware of the inadequacies of our ability to respond to viral epidemics. Although disrupting the viral life cycle is critical for limiting viral spread and disease, it has proven challenging to develop targeted and selective therapeutics. Synthetic lethality offers a promising but largely unexploited strategy against infectious viral disease; as viruses infect cells, they abnormally alter the cell state, unwittingly exposing new vulnerabilities in the infected cell. Therefore, we propose that effective therapies can be developed to selectively target the virally reconfigured host cell networks that accompany altered cellular states to cripple the host cell that has been converted into a virus factory, thus disrupting the viral life cycle.

[end paper 88]

[begin paper 89]

PMCID = 10636138

Title = Enhancing antibody affinity through experimental sampling of non-deleterious CDR mutations predicted by machine learning

The application of machine learning (ML) models to optimize antibody affinity to an antigen is gaining prominence. Unfortunately, the small and biased nature of the publicly available antibody-antigen interaction datasets makes it challenging to build an ML model that can accurately predict binding affinity changes due to mutations (ΔΔG). Recognizing these inherent limitations, we reformulated the problem to ask whether an ML model capable of classifying deleterious vs non-deleterious mutations can guide antibody affinity maturation in a practical setting. To test this hypothesis, we developed a Random Forest classifier (Antibody Random Forest Classifier or AbRFC) with expert-guided features and integrated it into a computational-experimental workflow. AbRFC effectively predicted non-deleterious mutations on an in-house validation dataset that is free of biases seen in the publicly available training datasets. Furthermore, experimental screening of a limited number of predictions from the model (<10^2 designs) identified affinity-enhancing mutations in two unrelated SARS-CoV-2 antibodies, resulting in constructs with up to 1000-fold increased binding to the SARS-COV-2 RBD. Our findings indicate that accurate prediction and screening of non-deleterious mutations using machine learning offers a powerful approach to improving antibody affinity.

The endeavor to harness machine learning to improve antibody-antigen interaction is not only significant but also fraught with challenges due to the limitations of the publicly available datasets. Here, the authors introduce an Antibody Random Forest Classifier that classifies mutations as either deleterious or non-deleterious based on physicochemical features, which enabled the identification of affinity-enhancing mutations in SARS-CoV-2 antibodies with an up to 1000-fold increased binding affinity.

[end paper 89]

[begin paper 90]

PMCID = 10827046

Title = De novo-designed minibinders expand the synthetic biology sensing repertoire

Synthetic and chimeric receptors capable of recognizing and responding to user-defined antigens have enabled “smart” therapeutics based on engineered cells. These cell engineering tools depend on antigen sensors which are most often derived from antibodies. Advances in the de novo design of proteins have enabled the design of protein binders with the potential to target epitopes with unique properties and faster production timelines compared to antibodies. Building upon our previous work combining a de novo-designed minibinder of the Spike protein of SARS-CoV-2 with the synthetic receptor synNotch (SARSNotch), we investigated whether minibinders can be readily adapted to a diversity of cell engineering tools. We show that the Spike minibinder LCB1 easily generalizes to a next-generation proteolytic receptor SNIPR that performs similarly to our previously reported SARSNotch. LCB1-SNIPR successfully enables the detection of live SARS-CoV-2, an improvement over SARSNotch which can only detect cell-expressed Spike. To test the generalizability of minibinders to diverse applications, we tested LCB1 as an antigen sensor for a chimeric antigen receptor (CAR). LCB1-CAR enabled CD8+ T cells to cytotoxically target Spike-expressing cells. Our findings suggest that minibinders represent a novel class of antigen sensors that have the potential to dramatically expand the sensing repertoire of cell engineering tools.

[end paper 90]

[begin paper 91]

PMCID = 8556785

Title = Advances and insights in the diagnosis of viral infections

Viral infections are the most common among diseases that globally require around 60 percent of medical care. However, in the heat of the pandemic, there was a lack of medical equipment and inpatient facilities to provide all patients with viral infections. The detection of viral infections is possible in three general ways such as (i) direct virus detection, which is performed immediately 1–3 days after the infection, (ii) determination of antibodies against some virus proteins mainly observed during/after virus incubation period, (iii) detection of virus-induced disease when specific tissue changes in the organism. This review surveys some global pandemics from 1889 to 2020, virus types, which induced these pandemics, and symptoms of some viral diseases. Non-analytical methods such as radiology and microscopy also are overviewed. This review overlooks molecular analysis methods such as nucleic acid amplification, antibody-antigen complex determination, CRISPR-Cas system-based viral genome determination methods. Methods widely used in the certificated diagnostic laboratory for SARS-CoV-2, Influenza A, B, C, HIV, and other viruses during a viral pandemic are outlined. A comprehensive overview of molecular analytical methods has shown that the assay's sensitivity, accuracy, and suitability for virus detection depends on the choice of the number of regions in the viral open reading frame (ORF) genome sequence and the validity of the selected analytical method.

[end paper 91]

[begin paper 92]

PMCID = 9585390

Title = Abstract



[end paper 92]

[begin paper 93]

PMCID = 9112195

Title = ISEV2022 Abstract Book



[end paper 93]

[begin paper 94]

PMCID = 11277559

Title = Single-Domain Antibodies—Novel Tools to Study and Treat Allergies

IgE-mediated allergies represent a major health problem in the modern world. Apart from allergen-specific immunotherapy (AIT), the only disease-modifying treatment, researchers focus on biologics that target different key molecules such as allergens, IgE, or type 2 cytokines to ameliorate allergic symptoms. Single-domain antibodies, or nanobodies, are the newcomers in biotherapeutics, and their huge potential is being investigated in various research fields since their discovery 30 years ago. While they are dominantly applied for theranostics of cancer and treatment of infectious diseases, nanobodies have become increasingly substantial in allergology over the last decade. In this review, we discuss the prerequisites that we consider to be important for generating useful nanobody-based drug candidates for treating allergies. We further summarize the available research data on nanobodies used as allergen monitoring and detection probes and for therapeutic approaches. We reflect on the limitations that have to be addressed during the development process, such as in vivo half-life and immunogenicity. Finally, we speculate about novel application formats for allergy treatment that might be available in the future.

[end paper 94]

[begin paper 95]

PMCID = 9720949

Title = Prediction of polyreactive and nonspecific single-chain fragment variables through structural biochemical features and protein language-based descriptors

Monoclonal antibodies (mAbs) have been used as therapeutic agents, which must overcome many developability issues after the discovery from in vitro display libraries. Especially, polyreactive mAbs can strongly bind to a specific target and weakly bind to off-target proteins, which leads to poor antibody pharmacokinetics in clinical development. Although early assessment of polyreactive mAbs is important in the early discovery stage, experimental assessments are usually time-consuming and expensive. Therefore, computational approaches for predicting the polyreactivity of single-chain fragment variables (scFvs) in the early discovery stage would be promising for reducing experimental efforts. 

Here, we made prediction models for the polyreactivity of scFvs with the known polyreactive antibody features and natural language model descriptors. We predicted 19,426 protein structures of scFvs with trRosetta to calculate the polyreactive antibody features and investigated the classifying performance of each factor for polyreactivity. In the known polyreactive features, the net charge of the CDR2 loop, the tryptophan and glycine residues in CDR-H3, and the lengths of the CDR1 and CDR2 loops, importantly contributed to the performance of the models. Additionally, the hydrodynamic features, such as partial specific volume, gyration radius, and isoelectric points of CDR loops and scFvs, were newly added to improve model performance. Finally, we made the prediction model with a robust performance () with an ensemble learning of the top 3 best models.

The prediction models for polyreactivity would help assess polyreactive scFvs in the early discovery stage and our approaches would be promising to develop machine learning models with quantitative data from high throughput assays for antibody screening. 

The online version contains supplementary material available at 10.1186/s12859-022-05010-4.

[end paper 95]

[begin paper 96]

PMCID = 10445546

Title = Quantitative flow cytometric selection of tau conformational nanobodies specific for pathological aggregates

Single-domain antibodies, also known as nanobodies, are broadly important for studying the structure and conformational states of several classes of proteins, including membrane proteins, enzymes, and amyloidogenic proteins. Conformational nanobodies specific for aggregated conformations of amyloidogenic proteins are particularly needed to better target and study aggregates associated with a growing class of associated diseases, especially neurodegenerative disorders such as Alzheimer’s and Parkinson’s diseases. However, there are few reported nanobodies with both conformational and sequence specificity for amyloid aggregates, especially for large and complex proteins such as the tau protein associated with Alzheimer’s disease, due to difficulties in selecting nanobodies that bind to complex aggregated proteins. Here, we report the selection of conformational nanobodies that selectively recognize aggregated (fibrillar) tau relative to soluble (monomeric) tau. Notably, we demonstrate that these nanobodies can be directly isolated from immune libraries using quantitative flow cytometric sorting of yeast-displayed libraries against tau aggregates conjugated to quantum dots, and this process eliminates the need for secondary nanobody screening. The isolated nanobodies demonstrate conformational specificity for tau aggregates in brain samples from both a transgenic mouse model and human tauopathies. We expect that our facile approach will be broadly useful for isolating conformational nanobodies against diverse amyloid aggregates and other complex antigens.

[end paper 96]

[begin paper 97]

PMCID = 9651861

Title = Fragment-based computational design of antibodies targeting structured epitopes

De novo design methods hold the promise of reducing the time and cost of antibody discovery while enabling the facile and precise targeting of predetermined epitopes. Here, we describe a fragment-based method for the combinatorial design of antibody binding loops and their grafting onto antibody scaffolds. We designed and tested six single-domain antibodies targeting different epitopes on three antigens, including the receptor-binding domain of the SARS-CoV-2 spike protein. Biophysical characterization showed that all designs are stable and bind their intended targets with affinities in the nanomolar range without in vitro affinity maturation. We further discuss how a high-resolution input antigen structure is not required, as similar predictions are obtained when the input is a crystal structure or a computer-generated model. This computational procedure, which readily runs on a laptop, provides a starting point for the rapid generation of lead antibodies binding to preselected epitopes.

A combinatorial method can rapidly design nanobodies that bind with nanomolar affinity predetermined epitopes.

[end paper 97]

[begin paper 98]

PMCID = 11221890

Title = Antibody design using deep learning: from sequence and structure design to affinity maturation

Deep learning has achieved impressive results in various fields such as computer vision and natural language processing, making it a powerful tool in biology. Its applications now encompass cellular image classification, genomic studies and drug discovery. While drug development traditionally focused deep learning applications on small molecules, recent innovations have incorporated it in the discovery and development of biological molecules, particularly antibodies. Researchers have devised novel techniques to streamline antibody development, combining in vitro and in silico methods. In particular, computational power expedites lead candidate generation, scaling and potential antibody development against complex antigens. This survey highlights significant advancements in protein design and optimization, specifically focusing on antibodies. This includes various aspects such as design, folding, antibody–antigen interactions docking and affinity maturation.

[end paper 98]

[begin paper 99]

PMCID = 10187017

Title = ISEV2023 Abstract Book



[end paper 99]

[begin paper 100]

PMCID = 11788748

Title = Structure‐based computational design of antibody mimetics: challenges and perspectives

The design of antibody mimetics holds great promise for revolutionizing therapeutic interventions by offering alternatives to conventional antibody therapies. Structure‐based computational approaches have emerged as indispensable tools in the rational design of those molecules, enabling the precise manipulation of their structural and functional properties. This review covers the main classes of designed antigen‐binding motifs, as well as alternative strategies to develop tailored ones. We discuss the intricacies of different computational protein–protein interaction design strategies, showcased by selected successful cases in the literature. Subsequently, we explore the latest advancements in the computational techniques including the integration of machine and deep learning methodologies into the design framework, which has led to an augmented design pipeline. Finally, we verse onto the current challenges that stand in the way between high‐throughput computer design of antibody mimetics and experimental realization, offering a forward‐looking perspective into the field and the promises it holds to biotechnology.

Computer graphics representation of an artificial intelligence–driven protein design pipeline, where building blocks are fed into the birth of a novel protein structure. 

[end paper 100]



