## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "SARS-CoV-2 spike protein" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on SARS-CoV-2 spike protein to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on SARS-CoV-2 spike protein using the query "Cross-reactivity of nanobodies with SARS-CoV-2 variants" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to SARS-CoV-2 spike protein. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, here is a Python dictionary mapping the PMCIDs to the titles of the papers that are directly relevant to the SARS-CoV-2 spike protein and the development of nanobody binders:

```python
{
    "11140877": "Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses",
    "8651292": "Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape",
    "9461429": "Arsenal of nanobodies shows broad-spectrum neutralization against SARS-CoV-2 variants of concern in vitro and in vivo in hamster models",
    "9170159": "Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants",
    "10444775": "Discovery and multimerization of cross-reactive single-domain antibodies against SARS-like viruses to enhance potency and address emerging SARS-CoV-2 variants",
    "9869787": "A Multivalent and Thermostable Nanobody Neutralizing SARS-CoV-2 Omicron (B.1.1.529)",
    "8951411": "Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering",
    "10715488": "Identification of nurse shark VNAR single-domain antibodies targeting the spike S2 subunit of SARS-CoV-2",
    "8514607": "The role of single-domain antibodies (or nanobodies) in SARS-CoV-2 neutralization",
    "10373554": "Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening",
    "9792944": "Broadly neutralizing and protective nanobodies against SARS-CoV-2 Omicron subvariants BA.1, BA.2, and BA.4/5 and diverse sarbecoviruses",
    "8260353": "Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants",
    "11148044": "Generation of nanobodies from transgenic ‘LamaMice’ lacking an endogenous immunoglobulin repertoire",
    "9365158": "A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike",
    "9834170": "Two pan-SARS-CoV-2 nanobodies and their multivalent derivatives effectively prevent Omicron infections in mice",
    "8908544": "Structural basis of nanobodies neutralizing SARS-CoV-2 variants",
    "9896449": "Shark nanobodies with potent SARS-CoV-2 neutralizing activity and broad sarbecovirus reactivity",
    "9356539": "Rapid and Quantitative In-vitro Evaluation of SARS-CoV-2 Neutralizing Antibodies and Nanobodies",
    "8982573": "Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance",
    "10588750": "A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern",
    "9482557": "Single domain antibodies derived from ancient animals as broadly neutralizing agents for SARS-CoV-2 and other coronaviruses",
    "10986514": "Nanobody engineering for SARS-CoV-2 neutralization and detection",
    "9174178": "Superimmunity by pan-sarbecovirus nanobodies",
    "8043454": "Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2",
    "10590698": "Nanobodies with cross-neutralizing activity provide prominent therapeutic efficacy in mild and severe COVID-19 rodent models",
    "7870875": "Potent neutralization of clinical isolates of SARS-CoV-2 D614 and G614 variants by a monomeric, sub-nanomolar affinity nanobody",
    "10858559": "Development and characterization of a novel nanobody with SRMV neutralizing activity",
    "8575140": "Camel nanobodies broadly neutralize SARS-CoV-2 variants",
    "8448731": "A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies",
    "9132424": "Deep mutational engineering of broadly-neutralizing nanobodies accommodating SARS-CoV-1 and 2 antigenic drift",
    "8928829": "Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19",
    "8726723": "Rapid discovery of diverse neutralizing SARS-CoV-2 antibodies from large-scale synthetic phage libraries",
    "9000494": "Structural Insights into the Design of Synthetic Nanobody Libraries",
    "11209230": "Comprehensive Overview of Broadly Neutralizing Antibodies against SARS-CoV-2 Variants",
    "9288967": "Accelerating PERx reaction enables covalent nanobodies for potent neutralization of SARS-CoV-2 and variants",
    "11515141": "Discovery of nanobodies: a comprehensive review of their applications and potential over the past five years",
    "10060843": "Review of therapeutic mechanisms and applications based on SARS-CoV-2 neutralizing antibodies",
    "11311977": "Enhanced Assessment of Cross-Reactive Antigenic Determinants within the Spike Protein",
    "11723632": "Discovery of Nanosota-EB1 and -EB2 as Novel Nanobody Inhibitors Against Ebola Virus Infection",
    "9529347": "Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2",
    "8458290": "A potent SARS-CoV-2 neutralising nanobody shows therapeutic efficacy in the Syrian golden hamster model of COVID-19",
    "8863326": "A potent alpaca-derived nanobody that neutralizes SARS-CoV-2 variants",
    "10054972": "Nanobodies against SARS-CoV-2 reduced virus load in the brain of challenged mice and neutralized Wuhan, Delta and Omicron Variants",
    "9512078": "Mucosal nanobody IgA as inhalable and affordable prophylactic and therapeutic treatment against SARS-CoV-2 and emerging variants",
    "8126837": "Nanobody cocktails potently neutralize SARS-CoV-2 D614G N501Y variant and protect mice",
    "10028217": "Efficient virus detection utilizing chitin-immobilized nanobodies synthesized in Ustilago maydis",
    "10878242": "A novel strategy for an anti-idiotype vaccine: nanobody mimicking neutralization epitope of porcine circovirus type 2",
    "10517151": "Camel nanobody-based B7-H3 CAR-T cells show high efficacy against large solid tumours",
    "9656363": "Screening, Expression, and Identification of Nanobody against SARS-CoV-2 Spike Protein",
    "11793827": "Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies",
    "9822811": "Biparatopic antibody BA7208/7125 effectively neutralizes SARS-CoV-2 variants including Omicron BA.1-BA.5",
    "7820838": "Perspective on therapeutic and diagnostic potential of camel nanobodies for coronavirus disease-19 (COVID-19)",
    "10275483": "Emerging Landscape of Nanobodies and Their Neutralizing Applications against SARS-CoV-2 Virus",
    "10757926": "Two birds with one stone: human SIRPα nanobodies for functional modulation and in vivo imaging of myeloid cells",
    "8748511": "A bispecific monomeric nanobody induces spike trimer dimers and neutralizes SARS-CoV-2 in vivo",
    "11324105": "Structure defining of ultrapotent neutralizing nanobodies against MERS-CoV with novel epitopes on receptor binding domain",
    "9990171": "Avidity engineering of human heavy-chain-only antibodies mitigates neutralization resistance of SARS-CoV-2 variants",
    "10154539": "Fortuitous somatic mutations during antibody evolution endow broad neutralization against SARS-CoV-2 Omicron variants",
    "11277385": "Broad sarbecovirus neutralization by combined memory B cell antibodies to ancestral SARS-CoV-2",
    "8354634": "The development of Nanosota-1 as anti-SARS-CoV-2 nanobody drug candidates",
    "8967028": "Structure-based dual affinity optimization of a SARS-CoV-1/2 cross-reactive single-domain antibody",
    "11200683": "A VHH single-domain platform enabling discovery and development of monospecific antibodies and modular neutralizing bispecifics against SARS-CoV-2 variants",
    "10051120": "Biomimetic Nanotechnology for SARS-CoV-2 Treatment",
    "8194341": "Prospects of Neutralizing Nanobodies Against SARS-CoV-2",
    "9778468": "Plasmodium infection is associated with cross-reactive antibodies to carbohydrate epitopes on the SARS-CoV-2 Spike protein",
    "8677774": "Mechanisms of SARS-CoV-2 neutralization by shark variable new antigen receptors elucidated through X-ray crystallography",
    "10518403": "Unleashing the power of shark variable single domains (VNARs): broadly neutralizing tools for combating SARS-CoV-2",
    "8537675": "Emerging SARS-CoV-2 Variants: A Review of Its Mutations, Its Implications and Vaccine Efficacy",
    "11099193": "Extracellular modulation of TREK-2 activity with nanobodies provides insight into the mechanisms of K2P channel regulation",
    "10532743": "Shark IgNAR: The Next Broad Application Antibody in Clinical Diagnoses and Tumor Therapies?",
    "10386765": "Cross protection to SARS-CoV-2 variants in hamsters with naturally-acquired immunity",
    "9181070": "Robust anti-SARS-CoV2 single domain antibodies cross neutralize multiple viruses",
    "9201380": "Structural Characterization of a Neutralizing Nanobody With Broad Activity Against SARS-CoV-2 Variants",
    "8837488": "Human inhalable antibody fragments neutralizing SARS-CoV-2 variants for COVID-19 therapy",
    "8526496": "Development of a highly specific and sensitive VHH-based sandwich immunoassay for the detection of the SARS-CoV-2 nucleoprotein",
    "10302031": "Status and Developing Strategies for Neutralizing Monoclonal Antibody Therapy in the Omicron Era of COVID-19",
    "9774204": "Fortuitous Somatic Mutations during Antibody Evolution Endow Broad Neutralization against SARS-CoV-2 Omicron Variants",
    "8564274": "A non-ACE2 competing human single-domain antibody confers broad neutralization against SARS-CoV-2 and circulating variants",
    "8700021": "Epitope profiling using computational structural modelling demonstrated on coronavirus-binding antibodies",
    "8899823": "A broadly cross-reactive antibody neutralizes and protects against sarbecovirus challenge in mice",
    "11100608": "Bispecific antibodies with broad neutralization potency against SARS-CoV-2 variants of concern",
    "10170451": "Targetable elements in SARS-CoV-2 S2 subunit for the design of pan-coronavirus fusion inhibitors and vaccines",
    "8890389": "Therapeutic antibodies for COVID-19: is a new age of IgM, IgA and bispecific antibodies coming?",
    "10562541": "A novel bispecific antibody dual-targeting approach for enhanced neutralization against fast-evolving SARS-CoV-2 variants",
    "9035363": "LY-CoV1404 (bebtelovimab) potently neutralizes SARS-CoV-2 variants",
    "10847457": "Potent and broadly neutralizing antibodies against sarbecoviruses induced by sequential COVID-19 vaccination"
}
```

These papers are selected based on their relevance to the development and characterization of nanobodies targeting the SARS-CoV-2 spike protein and their cross-reactivity with various variants.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "Cross-reactivity of nanobodies with SARS-CoV-2 variants":

[begin paper 1]

PMCID = 11140877

Title = Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses

Nanobodies, single-domain antibodies derived from variable domain of camelid or shark heavy-chain antibodies, have unique properties with small size, strong binding affinity, easy construction in versatile formats, high neutralizing activity, protective efficacy, and manufactural capacity on a large-scale. Nanobodies have been arisen as an effective research tool for development of nanobiotechnologies with a variety of applications. Three highly pathogenic coronaviruses (CoVs), SARS-CoV-2, SARS-CoV, and MERS-CoV, have caused serious outbreaks or a global pandemic, and continue to post a threat to public health worldwide. The viral spike (S) protein and its cognate receptor-binding domain (RBD), which initiate viral entry and play a critical role in virus pathogenesis, are important therapeutic targets. This review describes pathogenic human CoVs, including viral structures and proteins, and S protein-mediated viral entry process. It also summarizes recent advances in development of nanobodies targeting these CoVs, focusing on those targeting the S protein and RBD. Finally, we discuss potential strategies to improve the efficacy of nanobodies against emerging SARS-CoV-2 variants and other CoVs with pandemic potential. It will provide important information for rational design and evaluation of therapeutic agents against emerging and reemerging pathogens.

[end paper 1]

[begin paper 2]

PMCID = 8651292

Title = Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape

The emergence of SARS-CoV-2 variants threatens current vaccines and therapeutic antibodies and urgently demands powerful new therapeutics that can resist viral escape. We therefore generated a large nanobody repertoire to saturate the distinct and highly conserved available epitope space of SARS-CoV-2 spike, including the S1 receptor binding domain, N-terminal domain, and the S2 subunit, to identify new nanobody binding sites that may reflect novel mechanisms of viral neutralization. Structural mapping and functional assays show that indeed these highly stable monovalent nanobodies potently inhibit SARS-CoV-2 infection, display numerous neutralization mechanisms, are effective against emerging variants of concern, and are resistant to mutational escape. Rational combinations of these nanobodies that bind to distinct sites within and between spike subunits exhibit extraordinary synergy and suggest multiple tailored therapeutic and prophylactic strategies.

[end paper 2]

[begin paper 3]

PMCID = 9797221

Title = Therapeutic applications of nanobodies against SARS-CoV-2 and other viral infections: Current update

In the last two years, the world encountered the SARS-CoV-2 virus, which is still dominating the population due to the absence of a viable treatment. To eradicate the global pandemic, scientists, doctors, and researchers took an exceptionally significant initiative towards the development of effective therapeutics to save many lifes. This review discusses about the single-domain antibodies (sdAbs), also called nanobodies, their structure, and their types against the infections of dreadful SARS-CoV-2 virus. A precise description highlights the nanobodies and their therapeutic application against the other selected viruses. It aims to focus on the extraordinary features of these antibodies compared to the conventional therapeutics like mAbs, convalescent plasma therapy, and vaccines. The stable structure of these nanobodies along with the suitable mechanism of action also confers greater resistance to the evolving variants with numerous mutations. The nanobodies developed against SARS-CoV-2 and its mutant variants have shown the greater neutralization potential than the primitive ones. Engineering of these specialized antibodies by modern biotechnological approaches will surely be more beneficial in treating this COVID-19 pandemic along with certain other viral infections.

[end paper 3]

[begin paper 4]

PMCID = 9461429

Title = Arsenal of nanobodies shows broad-spectrum neutralization against SARS-CoV-2 variants of concern in vitro and in vivo in hamster models

Nanobodies offer several potential advantages over mAbs for the control of SARS-CoV-2. Their ability to access cryptic epitopes conserved across SARS-CoV-2 variants of concern (VoCs) and feasibility to engineer modular, multimeric designs, make these antibody fragments ideal candidates for developing broad-spectrum therapeutics against current and continually emerging SARS-CoV-2 VoCs. Here we describe a diverse collection of 37 anti-SARS-CoV-2 spike glycoprotein nanobodies extensively characterized as both monovalent and IgG Fc-fused bivalent modalities. The nanobodies were collectively shown to have high intrinsic affinity; high thermal, thermodynamic and aerosolization stability; broad subunit/domain specificity and cross-reactivity across existing VoCs; wide-ranging epitopic and mechanistic diversity and high and broad in vitro neutralization potencies. A select set of Fc-fused nanobodies showed high neutralization efficacies in hamster models of SARS-CoV-2 infection, reducing viral burden by up to six orders of magnitude to below detectable levels. In vivo protection was demonstrated with anti-RBD and previously unreported anti-NTD and anti-S2 nanobodies. This collection of nanobodies provides a potential therapeutic toolbox from which various cocktails or multi-paratopic formats could be built to combat multiple SARS-CoV-2 variants.

Isolation and extensive characterization of a collection of 37 anti-SARS-CoV-2 spike glycoprotein nanobodies show broad neutralization efficacies in vitro and in vivo in a hamster model of SARS-CoV-2 infection.

[end paper 4]

[begin paper 5]

PMCID = 9884143

Title = Expanding and improving nanobody repertoires using a yeast display method: Targeting SARS-CoV-2

COVID-19, caused by the coronavirus SARS-CoV-2, represents a serious worldwide health issue, with continually emerging new variants challenging current therapeutics. One promising alternate therapeutic avenue is represented by nanobodies, small single-chain antibodies derived from camelids with numerous advantageous properties and the potential to neutralize the virus. For identification and characterization of a broad spectrum of anti–SARS-CoV-2 Spike nanobodies, we further optimized a yeast display method, leveraging a previously published mass spectrometry-based method, using B-cell complementary DNA from the same immunized animals as a source of VHH sequences. Yeast display captured many of the sequences identified by the previous approach, as well as many additional sequences that proved to encode a large new repertoire of nanobodies with high affinities and neutralization activities against different SARS-CoV-2 variants. We evaluated DNA shuffling applied to the three complementarity-determining regions of antiviral nanobodies. The results suggested a surprising degree of modularity to complementarity-determining region function. Importantly, the yeast display approach applied to nanobody libraries from immunized animals allows parallel interrogation of a vast number of nanobodies. For example, we employed a modified yeast display to carry out massively parallel epitope binning. The current yeast display approach proved comparable in efficiency and specificity to the mass spectrometry–based approach, while requiring none of the infrastructure and expertise required for that approach, making these highly complementary approaches that together appear to comprehensively explore the paratope space. The larger repertoires produced maximize the likelihood of discovering broadly specific reagents and those that powerfully synergize in mixtures.

[end paper 5]

[begin paper 6]

PMCID = 9170159

Title = Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants

Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. In this report, we built dromedary camel (Camelus dromedarius) VHH phage libraries for the isolation of high-affinity nanobodies that broadly neutralize SARS-CoV-2 variants. Cryo-EM complex structures reveal that one dromedary camel VHH nanobody (8A2) binds the S1 subunit of the viral spike protein, and the other (7A3) targets a deeply buried region that uniquely extends to the S2 subunit beyond the S1 subunit. These nanobodies can protect mice from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting the therapeutic potential of these nanobodies against COVID-19. The dromedary camel VHH libraries could be helpful to isolate neutralizing nanobodies against future emerging viruses quickly.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike is a trimer of S1/S2 heterodimers with three receptor-binding domains (RBDs) at the S1 subunit for human angiotensin-converting enzyme 2 (hACE2). Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. To isolate high-affinity nanobodies, large libraries with great diversity are highly desirable. Dromedary camels (Camelus dromedarius) are natural reservoirs of coronaviruses like Middle East respiratory syndrome CoV (MERS-CoV) that are transmitted to humans. Here, we built large dromedary camel VHH phage libraries to isolate nanobodies that broadly neutralize SARS-CoV-2 variants. We isolated two VHH nanobodies, NCI-CoV-7A3 (7A3) and NCI-CoV-8A2 (8A2), which have a high affinity for the RBD via targeting nonoverlapping epitopes and show broad neutralization activity against SARS-CoV-2 and its emerging variants of concern. Cryoelectron microscopy (cryo-EM) complex structures revealed that 8A2 binds the RBD in its up mode with a long CDR3 loop directly involved in the ACE2 binding residues and that 7A3 targets a deeply buried region that uniquely extends from the S1 subunit to the apex of the S2 subunit regardless of the conformational state of the RBD. At a dose of ≥5 mg/kg, 7A3 efficiently protected transgenic mice expressing hACE2 from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting its therapeutic use against COVID-19 variants. The dromedary camel VHH phage libraries could be helpful as a unique platform ready for quickly isolating potent nanobodies against future emerging viruses.

[end paper 6]

[begin paper 7]

PMCID = 10444775

Title = Discovery and multimerization of cross-reactive single-domain antibodies against SARS-like viruses to enhance potency and address emerging SARS-CoV-2 variants

Coronaviruses have been the causative agent of three epidemics and pandemics in the past two decades, including the ongoing COVID-19 pandemic. A broadly-neutralizing coronavirus therapeutic is desirable not only to prevent and treat COVID-19, but also to provide protection for high-risk populations against future emergent coronaviruses. As all coronaviruses use spike proteins on the viral surface to enter the host cells, and these spike proteins share sequence and structural homology, we set out to discover cross-reactive biologic agents targeting the spike protein to block viral entry. Through llama immunization campaigns, we have identified single domain antibodies (VHHs) that are cross-reactive against multiple emergent coronaviruses (SARS-CoV, SARS-CoV-2, and MERS). Importantly, a number of these antibodies show sub-nanomolar potency towards all SARS-like viruses including emergent CoV-2 variants. We identified nine distinct epitopes on the spike protein targeted by these VHHs. Further, by engineering VHHs targeting distinct, conserved epitopes into multi-valent formats, we significantly enhanced their neutralization potencies compared to the corresponding VHH cocktails. We believe this approach is ideally suited to address both emerging SARS-CoV-2 variants during the current pandemic as well as potential future pandemics caused by SARS-like coronaviruses.

[end paper 7]

[begin paper 8]

PMCID = 9869787

Title = A Multivalent and Thermostable Nanobody Neutralizing SARS-CoV-2 Omicron (B.1.1.529)

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) Omicron variants have risen to dominance, which contains far more mutations in the spike protein in comparison to previously reported variants, compromising the efficacy of most existing vaccines or therapeutic monoclonal antibodies. Nanobody screened from high-throughput naïve libraries is a potential candidate for developing preventive and therapeutic antibodies.

Four nanobodies specific to the SARS-CoV-2 wild-type receptor-binding domain (RBD) were screened from a naïve phage display library. Their affinity and neutralizing activity were evaluated by surface plasmon resonance assays, surrogate virus neutralization tests, and pseudovirus neutralization assays. Preliminary identification of the binding epitopes of nanobodies by peptide-based ELISA and competition assay. Then four multivalent nanobodies were engineered by attaching the monovalent nanobodies to an antibody-binding nanoplatform constructed based on the lumazine synthase protein cage nanoparticles isolated from the Aquifex aeolicus (AaLS). Finally, the differences in potency between the monovalent and multivalent nanobodies were compared using the same methods.

Three of the four specific nanobodies could maintain substantial inhibitory activity against the Omicron (B.1.1.529), of them, B-B2 had the best neutralizing activity against the Omicron (B.1.1.529) pseudovirus (IC50 = 1.658 μg/mL). The antiviral ability of multivalent nanobody LS-B-B2 was improved in the Omicron (B.1.1.529) pseudovirus assays (IC50 = 0.653 μg/mL). The results of peptide-based ELISA indicated that LS-B-B2 might react with the linear epitopes in the SARS-CoV-2 RBD conserved regions, which would clarify the mechanisms for the maintenance of potent neutralization of Omicron (B.1.1.529) preliminary.

Our study indicated that the AaLS could be used as an antibody-binding nanoplatform to present nanobodies on its surface and improve the potency of nanobodies. The multivalent nanobody LS-B-B2 may serve as a potential agent for the neutralization of SARS-CoV-2 variants.

[end paper 8]

[begin paper 9]

PMCID = 8951411

Title = Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering

Nanobodies provide important advantages over traditional antibodies, including their smaller size and robust biochemical properties such as high thermal stability, high solubility, and the ability to be bioengineered into novel multivalent, multi-specific, and high-affinity molecules, making them a class of emerging powerful therapies against SARS-CoV-2. Recent research efforts on the design, protein engineering, and structure-functional characterization of nanobodies and their binding with SARS-CoV-2 S proteins reflected a growing realization that nanobody combinations can exploit distinct binding epitopes and leverage the intrinsic plasticity of the conformational landscape for the SARS-CoV-2 S protein to produce efficient neutralizing and mutation resistant characteristics. Structural and computational studies have also been instrumental in quantifying the structure, dynamics, and energetics of the SARS-CoV-2 spike protein binding with nanobodies. In this review, a comprehensive analysis of the current structural, biophysical, and computational biology investigations of SARS-CoV-2 S proteins and their complexes with distinct classes of nanobodies targeting different binding sites is presented. The analysis of computational studies is supplemented by an in-depth examination of mutational scanning simulations and identification of binding energy hotspots for distinct nanobody classes. The review is focused on the analysis of mechanisms underlying synergistic binding of multivalent nanobodies that can be superior to single nanobodies and conventional nanobody cocktails in combating escape mutations by effectively leveraging binding avidity and allosteric cooperativity. We discuss how structural insights and protein engineering approaches together with computational biology tools can aid in the rational design of synergistic combinations that exhibit superior binding and neutralization characteristics owing to avidity-mediated mechanisms.

[end paper 9]

[begin paper 10]

PMCID = 9043892

Title = Passive Immunotherapy Against SARS-CoV-2: From Plasma-Based Therapy to Single Potent Antibodies in the Race to Stay Ahead of the Variants

The COVID-19 pandemic is now approaching 2 years old, with more than 440 million people infected and nearly six million dead worldwide, making it the most significant pandemic since the 1918 influenza pandemic. The severity and significance of SARS-CoV-2 was recognized immediately upon discovery, leading to innumerable companies and institutes designing and generating vaccines and therapeutic antibodies literally as soon as recombinant SARS-CoV-2 spike protein sequence was available. Within months of the pandemic start, several antibodies had been generated, tested, and moved into clinical trials, including Eli Lilly’s bamlanivimab and etesevimab, Regeneron’s mixture of imdevimab and casirivimab, Vir’s sotrovimab, Celltrion’s regdanvimab, and Lilly’s bebtelovimab. These antibodies all have now received at least Emergency Use Authorizations (EUAs) and some have received full approval in select countries. To date, more than three dozen antibodies or antibody combinations have been forwarded into clinical trials. These antibodies to SARS-CoV-2 all target the receptor-binding domain (RBD), with some blocking the ability of the RBD to bind human ACE2, while others bind core regions of the RBD to modulate spike stability or ability to fuse to host cell membranes. While these antibodies were being discovered and developed, new variants of SARS-CoV-2 have cropped up in real time, altering the antibody landscape on a moving basis. Over the past year, the search has widened to find antibodies capable of neutralizing the wide array of variants that have arisen, including Alpha, Beta, Gamma, Delta, and Omicron. The recent rise and dominance of the Omicron family of variants, including the rather disparate BA.1 and BA.2 variants, demonstrate the need to continue to find new approaches to neutralize the rapidly evolving SARS-CoV-2 virus. This review highlights both convalescent plasma- and polyclonal antibody-based approaches as well as the top approximately 50 antibodies to SARS-CoV-2, their epitopes, their ability to bind to SARS-CoV-2 variants, and how they are delivered. New approaches to antibody constructs, including single domain antibodies, bispecific antibodies, IgA- and IgM-based antibodies, and modified ACE2-Fc fusion proteins, are also described. Finally, antibodies being developed for palliative care of COVID-19 disease, including the ramifications of cytokine release syndrome (CRS) and acute respiratory distress syndrome (ARDS), are described.

The online version contains supplementary material available at 10.1007/s40259-022-00529-7.

[end paper 10]

[begin paper 11]

PMCID = 10715488

Title = Identification of nurse shark VNAR single-domain antibodies targeting the spike S2 subunit of SARS-CoV-2

SARS-CoV-2 is the etiological agent of the COVID-19 pandemic. Antibody-based therapeutics targeting the spike protein, specifically the S1 subunit or the receptor binding domain (RBD) of SARS-CoV-2, have gained attention due to their clinical efficacy in treating patients diagnosed with COVID-19. An alternative to conventional antibody therapeutics is the use of shark new antigen variable receptor domain (VNAR) antibodies. VNARs are small (<15 kDa) and can reach deep into the pockets or grooves of the target antigen. Here, we have isolated 53 VNARs that bind to the S2 subunit by phage panning from a naïve nurse shark VNAR phage display library constructed in our laboratory. Among those binders, S2A9 showed the best neutralization activity against the original pseudotyped SARS-CoV-2 virus. Several binders, including S2A9, showed cross-reactivity against S2 subunits from other β coronaviruses. Furthermore, S2A9 showed neutralization activity against all variants of concern (VOCs) from alpha to omicron (including BA1, BA2, BA4, and BA5) in both pseudovirus and live virus neutralization assays. Our findings suggest that S2A9 could be a promising lead molecule for the development of broadly neutralizing antibodies against SARS-CoV-2 and emerging variants. The nurse shark VNAR phage library offers a novel platform that can be used to rapidly isolate single-domain antibodies against emerging viral pathogens.

[end paper 11]

[begin paper 12]

PMCID = 8514607

Title = The role of single-domain antibodies (or nanobodies) in SARS-CoV-2 neutralization

The severe acute respiratory syndrome (SARS-CoV-2), a newly emerging of coronavirus, continues to infect humans in the absence of a viable treatment. Neutralizing antibodies that disrupt the interaction of RBD and ACE2 has been under the spotlight as a way of developing the COVID-19 treatment. Some animals, such as llamas, manufacture heavy-chain antibodies that have a single variable domain (VHH) instead of two variable domains (VH/VL) as opposed to typical antibodies. Nanobodies are antigen-specific, single-domain, changeable segments of camelid heavy chain-only antibodies that are recombinantly produced. These types of antibodies exhibit a wide range of strong physical and chemical properties, like high solubility, and stability. The VHH's high-affinity attachment to the receptor-binding domain (RBD) allowed the neutralization of SARS-CoV-2. To tackle COVID-19, some nanobodies are being developed against SARS-CoV-2, some of which have been recently included in clinical trials. Nanobody therapy may be useful in managing the COVID-19 pandemic as a potent and low-cost treatment. This paper describes the application of nanobodies as a new class of recombinant antibodies in COVID-19 treatment.

[end paper 12]

[begin paper 13]

PMCID = 10373554

Title = Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening

Members of the Sarbecovirus subgenus of Coronaviridae have twice caused deadly threats to humans. There is increasing concern about the rapid mutation of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), which has evolved into multiple generations of epidemic variants in 3 years. Broad neutralizing antibodies are of great importance for pandemic preparedness against SARS-CoV-2 variants and divergent zoonotic sarbecoviruses. Here, we analyzed the structural conservation of the receptor-binding domain (RBD) from representative sarbecoviruses and chose S2H97, a previously reported RBD antibody with ideal breadth and resistance to escape, as a template for computational design to enhance the neutralization activity and spectrum. A total of 35 designs were purified for evaluation. The neutralizing activity of a large proportion of these designs against multiple variants was increased from several to hundreds of times. Molecular dynamics simulation suggested that extra interface contacts and enhanced intermolecular interactions between the RBD and the designed antibodies are established. After light and heavy chain reconstitution, AI-1028, with five complementarity determining regions optimized, showed the best neutralizing activity across all tested sarbecoviruses, including SARS-CoV, multiple SARS-CoV-2 variants, and bat-derived viruses. AI-1028 recognized the same cryptic RBD epitope as the parental prototype antibody. In addition to computational design, chemically synthesized nanobody libraries are also a precious resource for rapid antibody development. By applying distinct RBDs as baits for reciprocal screening, we identified two novel nanobodies with broad activities. These findings provide potential pan-sarbecovirus neutralizing drugs and highlight new pathways to rapidly optimize therapeutic candidates when novel SARS-CoV-2 escape variants or new zoonotic coronaviruses emerge.

IMPORTANCE The subgenus Sarbecovirus includes human SARS-CoV, SARS-CoV-2, and hundreds of genetically related bat viruses. The continuous evolution of SARS-CoV-2 has led to the striking evasion of neutralizing antibody (NAb) drugs and convalescent plasma. Antibodies with broad activity across sarbecoviruses would be helpful to combat current SARS-CoV-2 mutations and longer term animal virus spillovers. The study of pan-sarbecovirus NAbs described here is significant for the following reasons. First, we established a structure-based computational pipeline to design and optimize NAbs to obtain more potent and broader neutralizing activity across multiple sarbecoviruses. Second, we screened and identified nanobodies from a highly diversified synthetic library with a broad neutralizing spectrum using an elaborate screening strategy. These methodologies provide guidance for the rapid development of antibody therapeutics against emerging pathogens with highly variable characteristics.

[end paper 13]

[begin paper 14]

PMCID = 9792944

Title = Broadly neutralizing and protective nanobodies against SARS-CoV-2 Omicron subvariants BA.1, BA.2, and BA.4/5 and diverse sarbecoviruses

As SARS-CoV-2 Omicron and other variants of concern (VOCs) continue spreading worldwide, development of antibodies and vaccines to confer broad and protective activity is a global priority. Here, we report on the identification of a special group of nanobodies from immunized alpaca with potency against diverse VOCs including Omicron subvariants BA.1, BA.2 and BA.4/5, SARS-CoV-1, and major sarbecoviruses. Crystal structure analysis of one representative nanobody, 3-2A2-4, discovers a highly conserved epitope located between the cryptic and the outer face of the receptor binding domain (RBD), distinctive from the receptor ACE2 binding site. Cryo-EM and biochemical evaluation reveal that 3-2A2-4 interferes structural alteration of RBD required for ACE2 binding. Passive delivery of 3-2A2-4 protects K18-hACE2 mice from infection of authentic SARS-CoV-2 Delta and Omicron. Identification of these unique nanobodies will inform the development of next generation antibody therapies and design of pan-sarbecovirus vaccines.

The authors identify nanobodies from immunized alpaca with broadly neutralizing activity against SARS-CoV-1, SARS-CoV-2 variants, and major sarbecoviruses. One representative nanobody binds to a highly conserved epitope on RBD and protects K18-hACE2 mice from Omicron and Delta infection.

[end paper 14]

[begin paper 15]

PMCID = 8260353

Title = Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants

Since the start of the COVID-19 pandemic, SARS-CoV-2 has caused millions of deaths worldwide. Although a number of vaccines have been deployed, the continual evolution of the receptor-binding domain (RBD) of the virus has challenged their efficacy. In particular, the emerging variants B.1.1.7, B.1.351 and P.1 (first detected in the UK, South Africa and Brazil, respectively) have compromised the efficacy of sera from patients who have recovered from COVID-19 and immunotherapies that have received emergency use authorization. One potential alternative to avert viral escape is the use of camelid VHHs (variable heavy chain domains of heavy chain antibody (also known as nanobodies)), which can recognize epitopes that are often inaccessible to conventional antibodies. Here, we isolate anti-RBD nanobodies from llamas and from mice that we engineered to produce VHHs cloned from alpacas, dromedaries and Bactrian camels. We identified two groups of highly neutralizing nanobodies. Group 1 circumvents antigenic drift by recognizing an RBD region that is highly conserved in coronaviruses but rarely targeted by human antibodies. Group 2 is almost exclusively focused to the RBD–ACE2 interface and does not neutralize SARS-CoV-2 variants that carry E484K or N501Y substitutions. However, nanobodies in group 2 retain full neutralization activity against these variants when expressed as homotrimers, and—to our knowledge—rival the most potent antibodies against SARS-CoV-2 that have been produced to date. These findings suggest that multivalent nanobodies overcome SARS-CoV-2 mutations through two separate mechanisms: enhanced avidity for the ACE2-binding domain and recognition of conserved epitopes that are largely inaccessible to human antibodies. Therefore, although new SARS-CoV-2 mutants will continue to emerge, nanobodies represent promising tools to prevent COVID-19 mortality when vaccines are compromised.

Multivalent nanobodies against SARS-CoV-2 from mice engineered to produce camelid nanobodies recognize conserved epitopes that are inaccessible to human antibodies and show promise as a strategy for dealing with viral escape mutations.

[end paper 15]

[begin paper 16]

PMCID = 9850341

Title = The Potential of Nanobodies for COVID-19 Diagnostics and Therapeutics

The infectious severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) is the causative agent for coronavirus disease 2019 (COVID-19). Globally, there have been millions of infections and fatalities. Unfortunately, the virus has been persistent and a contributing factor is the emergence of several variants. The urgency to combat COVID-19 led to the identification/development of various diagnosis (polymerase chain reaction and antigen tests) and treatment (repurposed drugs, convalescent plasma, antibodies and vaccines) options. These treatments may treat mild symptoms and decrease the risk of life-threatening disease. Although these options have been fairly beneficial, there are some challenges and limitations, such as cost of tests/drugs, specificity, large treatment dosages, intravenous administration, need for trained personal, lengthy production time, high manufacturing costs, and limited availability. Therefore, the development of more efficient COVID-19 diagnostic and therapeutic options are vital. Nanobodies (Nbs) are novel monomeric antigen-binding fragments derived from camelid antibodies. Advantages of Nbs include low immunogenicity, high specificity, stability and affinity. These characteristics allow for rapid Nb generation, inexpensive large-scale production, effective storage, and transportation, which is essential during pandemics. Additionally, the potential aerosolization and inhalation delivery of Nbs allows for targeted treatment delivery as well as patient self-administration. Therefore, Nbs are a viable option to target SARS-CoV-2 and overcome COVID-19. In this review we discuss (1) COVID-19; (2) SARS-CoV-2; (3) the present conventional COVID-19 diagnostics and therapeutics, including their challenges and limitations; (4) advantages of Nbs; and (5) the numerous Nbs generated against SARS-CoV-2 as well as their diagnostic and therapeutic potential.

[end paper 16]

[begin paper 17]

PMCID = 11148044

Title = Generation of nanobodies from transgenic ‘LamaMice’ lacking an endogenous immunoglobulin repertoire

Due to their exceptional solubility and stability, nanobodies have emerged as powerful building blocks for research tools and therapeutics. However, their generation in llamas is cumbersome and costly. Here, by inserting an engineered llama immunoglobulin heavy chain (IgH) locus into IgH-deficient mice, we generate a transgenic mouse line, which we refer to as ‘LamaMouse’. We demonstrate that LamaMice solely express llama IgH molecules without association to Igκ or λ light chains. Immunization of LamaMice with AAV8, the receptor-binding domain of the SARS-CoV-2 spike protein, IgE, IgG2c, and CLEC9A enabled us to readily select respective target-specific nanobodies using classical hybridoma and phage display technologies, single B cell screening, and direct cloning of the nanobody-repertoire into a mammalian expression vector. Our work shows that the LamaMouse represents a flexible and broadly applicable platform for a facilitated selection of target-specific nanobodies.

Nanobodies are normally made from immunized camelids, Ig transgenic mice or synthetic libraries. In this study, the authors introduce the llama Ig heavy chain locus into mice lacking this locus, thereby generating a line in which nanobodies can be made by direct immunization in the absence of an endogenous antibody repertoire.

[end paper 17]

[begin paper 18]

PMCID = 9365158

Title = A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike

Neutralizing antibodies targeting the SARS-CoV-2 spike protein have shown a great preventative/therapeutic potential. Here, we report a rapid and efficient strategy for the development and design of SARS-CoV-2 neutralizing humanized nanobody constructs with sub-nanomolar affinities and nanomolar potencies. CryoEM-based structural analysis of the nanobodies in complex with spike revealed two distinct binding modes. The most potent nanobody, RBD-1-2G(NCATS-BL8125), tolerates the N501Y RBD mutation and remains capable of neutralizing the B.1.1.7 (Alpha) variant. Molecular dynamics simulations provide a structural basis for understanding the neutralization process of nanobodies exclusively focused on the spike-ACE2 interface with and without the N501Y mutation on RBD. A primary human airway air-lung interface (ALI) ex vivo model showed that RBD-1-2G-Fc antibody treatment was effective at reducing viral burden following WA1 and B.1.1.7 SARS-CoV-2 infections. Therefore, this presented strategy will serve as a tool to mitigate the threat of emerging SARS-CoV-2 variants.

[end paper 18]

[begin paper 19]

PMCID = 9834170

Title = Two pan-SARS-CoV-2 nanobodies and their multivalent derivatives effectively prevent Omicron infections in mice

With the widespread vaccinations against coronavirus disease 2019 (COVID-19), we are witnessing gradually waning neutralizing antibodies and increasing cases of breakthrough infections, necessitating the development of drugs aside from vaccines, particularly ones that can be administered outside of hospitals. Here, we present two cross-reactive nanobodies (R14 and S43) and their multivalent derivatives, including decameric ones (fused to the immunoglobulin M [IgM] Fc) that maintain potent neutralizing activity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) after aerosolization and display not only pan-SARS-CoV-2 but also varied pan-sarbecovirus activities. Through respiratory administration to mice, monovalent and decameric R14 significantly reduce the lung viral RNAs at low dose and display potent pre- and post-exposure protection. Furthermore, structural studies reveal the neutralizing mechanisms of R14 and S43 and the multiple inhibition effects that the multivalent derivatives exert. Our work demonstrates promising convenient drug candidates via respiratory administration against SARS-CoV-2 infection, which can contribute to containing the COVID-19 pandemic.

Nanobodies R14 and S43 display not just pan-SARS-CoV-2 but varied pan-sarbecovirus activities

The multivalent nanobodies show enhanced neutralization, fitting the avidity effect

Intranasally administered MR14 potently prevents Omicron infections in mice

Structural studies reveal the neutralizing mechanisms of R14 and S43 as well as MR14

Liu et al. isolate two nanobodies (R14 and S43) and construct their multivalent derivatives with increased potencies and breadth against SARS-CoV-2 variants. Particularly, decameric R14, maintaining similar activities after aerosolization, efficiently prevents BA.2 infections in both pre- and post-exposure prophylactic settings in vivo and deserves further development.

[end paper 19]

[begin paper 20]

PMCID = 10968798

Title = Aptamers and Nanobodies as New Bioprobes for SARS-CoV-2 Diagnostic and Therapeutic System Applications

The global challenges posed by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic have underscored the critical importance of innovative and efficient control systems for addressing future pandemics. The most effective way to control the pandemic is to rapidly suppress the spread of the virus through early detection using a rapid, accurate, and easy-to-use diagnostic platform. In biosensors that use bioprobes, the binding affinity of molecular recognition elements (MREs) is the primary factor determining the dynamic range of the sensing platform. Furthermore, the sensitivity relies mainly on bioprobe quality with sufficient functionality. This comprehensive review investigates aptamers and nanobodies recently developed as advanced MREs for SARS-CoV-2 diagnostic and therapeutic applications. These bioprobes might be integrated into organic bioelectronic materials and devices, with promising enhanced sensitivity and specificity. This review offers valuable insights into advancing biosensing technologies for infectious disease diagnosis and treatment using aptamers and nanobodies as new bioprobes.

[end paper 20]

[begin paper 21]

PMCID = 8908544

Title = Structural basis of nanobodies neutralizing SARS-CoV-2 variants

Because of the evolutionary variants of SARS-CoV-2, development of broad-spectrum neutralizing antibodies resilient to virus escape is urgently needed. We identified a group of high-affinity nanobodies from camels immunized with receptor-binding domain (RBD) of SARS-CoV-2 spike protein and resolved the structures of two non-competing nanobodies (NB1A7 and NB1B11) in complex with RBD using X-ray crystallography. The structures show that NB1A7 targets the highly conserved cryptic epitope shared by SARS-CoV-2 variants and some other coronaviruses and blocks ACE2 receptor attachment of the spike protein, and NB1B11 epitope overlaps with the contacting surface of ACE2 and is different from the binding site of NB1A7. These two nanobodies were covalently linked into multivalent and bi-paratopic formats, which significantly improved the avidity and neutralization potency and may further inhibit viral escape. The results contribute to the structure-guided design of antibodies against future variants of SARS-CoV-2 virus to combat coronavirus epidemics and pandemics.

Shi et al. identify two neutralizing nanobodies against SARS-CoV-2. Crystal structures reveal how they prevent the virus and its variants from entering cells. The combination of two nanobodies significantly improves neutralization potency, which is a potential therapeutic candidate to reduce the probability of the emergence of escape variants of SARS-CoV-2.

[end paper 21]

[begin paper 22]

PMCID = 9226548

Title = Nucleocapsid Specific Diagnostics for the Detection of Divergent SARS-CoV-2 Variants

Since the start of the COVID-19 pandemic, multiple waves of SARS-CoV-2 variants have emerged. Of particular concern is the omicron variant, which harbors 28 mutations in the spike glycoprotein receptor binding and N-terminal domains relative to the ancestral strain. The high mutability of SARS-CoV-2 therefore poses significant hurdles for development of universal assays that rely on spike-specific immune detection. To address this, more conserved viral antigens need to be targeted. In this work, we comprehensively demonstrate the use of nucleocapsid (N)-specific detection across several assays using previously described nanobodies C2 and E2. We show that these nanobodies are highly sensitive and can detect divergent SARS-CoV-2 ancestral, delta and omicron variants across several assays. By comparison, spike-specific antibodies S309 and CR3022 only disparately detect SARS-CoV-2 variant targets. As such, we conclude that N-specific detection could provide a standardized universal target for detection of current and emerging SARS-CoV-2 variants of concern.

[end paper 22]

[begin paper 23]

PMCID = 9896449

Title = Shark nanobodies with potent SARS-CoV-2 neutralizing activity and broad sarbecovirus reactivity

Despite rapid and ongoing vaccine and therapeutic development, SARS-CoV-2 continues to evolve and evade, presenting a need for next-generation diverse therapeutic modalities. Here we show that nurse sharks immunized with SARS-CoV-2 recombinant receptor binding domain (RBD), RBD-ferritin (RFN), or spike protein ferritin nanoparticle (SpFN) immunogens elicit a set of new antigen receptor antibody (IgNAR) molecules that target two non-overlapping conserved epitopes on the spike RBD. Representative shark antibody variable NAR-Fc chimeras (ShAbs) targeting either of the two epitopes mediate cell-effector functions, with high affinity to all SARS-CoV-2 viral variants of concern, including the divergent Omicron strains. The ShAbs potently cross-neutralize SARS-CoV-2 WA-1, Alpha, Beta, Delta, Omicron BA.1 and BA.5, and SARS-CoV-1 pseudoviruses, and confer protection against SARS-CoV-2 challenge in the K18-hACE2 transgenic mouse model. Structural definition of the RBD-ShAb01-ShAb02 complex enabled design and production of multi-specific nanobodies with enhanced neutralization capacity, and picomolar affinity to divergent sarbecovirus clade 1a, 1b and 2 RBD molecules. These shark nanobodies represent potent immunotherapeutics both for current use, and future sarbecovirus pandemic preparation.

SARS-CoV-2 variants of concern continue to emerge, reducing vaccine efficacy and limiting therapeutic options. Here, Chen and colleagues describe the identification and design of shark nanobodies with pansarbecovirus activity.

[end paper 23]

[begin paper 24]

PMCID = 11252471

Title = Single-domain antibodies as therapeutics for solid tumor treatment

Single-domain antibodies (sdAbs), initially identified in camelids or sharks and commonly referred to as nanobodies or VNARs, have emerged as a promising alternative to conventional therapeutic antibodies. These sdAbs have many superior physicochemical and pharmacological properties, including small size, good solubility and thermostability, easier accessible epitopes, and strong tissue penetration. However, the inherent challenges associated with the animal origin of sdAbs limit their clinical use. In recent years, various innovative humanization technologies, including complementarity-determining region (CDR) grafting or complete engineering of fully human sdAbs, have been developed to mitigate potential immunogenicity issues and enhance their compatibility. This review provides a comprehensive exploration of sdAbs, emphasizing their distinctive features and the progress in humanization methodologies. In addition, we provide an overview of the recent progress in developing drugs and therapeutic strategies based on sdAbs and their potential in solid tumor treatment, such as sdAb–drug conjugates, multispecific sdAbs, sdAb-based delivery systems, and sdAb-based cell therapy.

This review provides a comprehensive exploration of sdAbs, emphasizing the progress in humanization methodologies and their distinctive applications in solid tumor treatment.

[end paper 24]

[begin paper 25]

PMCID = 9356539

Title = Rapid and Quantitative In-vitro Evaluation of SARS-CoV-2 Neutralizing Antibodies and Nanobodies

Neutralizing monoclonal antibodies and nanobodies have shown promising results as potential therapeutic agents for COVID-19. Identifying such antibodies and nanobodies requires evaluating the neutralization activity of a large number of lead molecules via biological assays, such as the virus neutralization test (VNT). These assays are typically time-consuming and demanding on lab facilities. Here, we present a rapid and quantitative assay that evaluates the neutralizing efficacy of an antibody or nanobody within 1.5 hours, does not require BSL-2 facilities, and consumes only 8 μL of low concentration (ng/mL) sample for each assay run. We tested the human angiotensin-converting enzyme 2 (ACE2) binding inhibition efficacy of seven antibodies and eight nanobodies and verified that the IC50 values of our assay are comparable with those from SARS-CoV-2 pseudovirus neutralization tests. We also found that our assay could evaluate the neutralizing efficacy against three widespread SARS-CoV-2 variants. We observed increased affinity of these variants for ACE2, including the Beta and Gamma variants. Finally, we demonstrated that our assay enables rapid identification of an immune-evasive mutation of the SARS-CoV-2 spike protein utilizing a set of nanobodies with known binding epitopes.

[end paper 25]

[begin paper 26]

PMCID = 10767048

Title = Cleavage-intermediate Lassa virus trimer elicits neutralizing responses, identifies neutralizing nanobodies, and reveals an apex-situated site-of-vulnerability

Lassa virus (LASV) infection is expanding outside its traditionally endemic areas in West Africa, posing a pandemic biothreat. LASV-neutralizing antibodies, moreover, have proven difficult to elicit. To gain insight into LASV neutralization, here we develop a prefusion-stabilized LASV glycoprotein trimer (GPC), pan it against phage libraries comprising single-domain antibodies (nanobodies) from shark and camel, and identify one, D5, which neutralizes LASV. Cryo-EM analyses reveal D5 to recognize a cleavage-dependent site-of-vulnerability at the trimer apex. The recognized site appears specific to GPC intermediates, with protomers lacking full cleavage between GP1 and GP2 subunits. Guinea pig immunizations with the prefusion-stabilized cleavage-intermediate LASV GPC, first as trimer and then as a nanoparticle, induce neutralizing responses, targeting multiple epitopes including that of D5; we identify a neutralizing antibody (GP23) from the immunized guinea pigs. Collectively, our findings define a prefusion-stabilized GPC trimer, reveal an apex-situated site-of-vulnerability, and demonstrate elicitation of LASV-neutralizing responses by a cleavage-intermediate LASV trimer.

Gorman et al. designed a Lassa virus prefusion-stabilized soluble glycoprotein complex trimer (GPC), with which they identified a Lassa virus-neutralizing nanobody that bound the GPC apex and elicited neutralizing antibody responses in guinea pigs.

[end paper 26]

[begin paper 27]

PMCID = 8982573

Title = Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance

The ongoing COVID‐19 pandemic represents an unprecedented global health crisis. Here, we report the identification of a synthetic nanobody (sybody) pair, Sb#15 and Sb#68, that can bind simultaneously to the SARS‐CoV‐2 spike RBD and efficiently neutralize pseudotyped and live viruses by interfering with ACE2 interaction. Cryo‐EM confirms that Sb#15 and Sb#68 engage two spatially discrete epitopes, influencing rational design of bispecific and tri‐bispecific fusion constructs that exhibit up to 100‐ and 1,000‐fold increase in neutralization potency, respectively. Cryo‐EM of the sybody‐spike complex additionally reveals a novel up‐out RBD conformation. While resistant viruses emerge rapidly in the presence of single binders, no escape variants are observed in the presence of the bispecific sybody. The multivalent bispecific constructs further increase the neutralization potency against globally circulating SARS‐CoV‐2 variants of concern. Our study illustrates the power of multivalency and biparatopic nanobody fusions for the potential development of therapeutic strategies that mitigate the emergence of new SARS‐CoV‐2 escape mutants.

Sybodies Sb#15 and Sb#68 inhibit SARS‐CoV‐2 infectivity by targeting non‐overlapping epitopes on the spike glycoprotein. Covalent sybody fusion and valency engineering enhances neutralization potency against variants and impedes emergence of escape mutants.  

[end paper 27]

[begin paper 28]

PMCID = 10588750

Title = A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern

The emergence of the coronavirus disease 2019 (COVID-19) pandemic and the new severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) variants of concern (VOCs) requires the continuous development of safe, effective, and affordable prevention and therapeutics. Nanobodies have demonstrated antiviral activity against a variety of viruses, providing a new candidate for the prevention and treatment of SARS-CoV-2 and its variants.

SARS-CoV-2 glycoprotein spike 1 subunit (S1) was selected as the target antigen for nanobody screening of a naïve phage display library. We obtained a nanobody, named Nb-H6, and then determined its affinity, inhibition, and stability by ELISA, Competitive ELISA, and Biolayer Interferometry (BLI). Infection assays of authentic and pseudotyped SARS-CoV-2 were performed to evaluate the neutralization of Nb-H6. The structure and mechanism of action were investigated by AlphaFold, docking, and residue mutation assays.

We isolated and characterized a nanobody, Nb-H6, which exhibits a broad affinity for S1 and the receptor binding domain (RBD) of SARS-CoV-2, or Alpha (B.1.1.7), Delta (B.1.617.2), Lambda (C.37), and Omicron (BA.2 and BA.5), and blocks receptor angiotensin-converting enzyme 2 (ACE2) binding. Moreover, Nb-H6 can retain its binding capability after pH or thermal treatment and effectively neutralize both pseudotyped and authentic SARS-CoV-2, as well as VOC Alpha (B.1.1.7), Delta (B.1.617.2), and Omicron (BA.2 and BA.5) pseudoviruses. We also confirmed that Nb-H6 binds two distinct amino acid residues of the RBD, preventing SARS-CoV-2 from interacting with the host receptor.

Our study highlights a novel nanobody, Nb-H6, that may be useful therapeutically in SARS-CoV-2 and VOC outbreaks and pandemics. These findings also provide a molecular foundation for further studies into how nanobodies neutralize SARS-CoV-2 and variants and imply potential therapeutic targets for the treatment of COVID-19.

[end paper 28]

[begin paper 29]

PMCID = 9482557

Title = Single domain antibodies derived from ancient animals as broadly neutralizing agents for SARS-CoV-2 and other coronaviruses

With severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) as an emergent human virus since December 2019, the world population is susceptible to coronavirus disease 2019 (COVID-19). SARS-CoV-2 has higher transmissibility than the previous coronaviruses, associated by the ribonucleic acid (RNA) virus nature with high mutation rate, caused SARS-CoV-2 variants to arise while circulating worldwide. Neutralizing antibodies are identified as immediate and direct-acting therapeutic against COVID-19. Single-domain antibodies (sdAbs), as small biomolecules with non-complex structure and intrinsic stability, can acquire antigen-binding capabilities comparable to conventional antibodies, which serve as an attractive neutralizing solution. SARS-CoV-2 spike protein attaches to human angiotensin-converting enzyme 2 (ACE2) receptor on lung epithelial cells to initiate viral infection, serves as potential therapeutic target. sdAbs have shown broad neutralization towards SARS-CoV-2 with various mutations, effectively stop and prevent infection while efficiently block mutational escape. In addition, sdAbs can be developed into multivalent antibodies or inhaled biotherapeutics against COVID-19.

.

[end paper 29]

[begin paper 30]

PMCID = 10986514

Title = Nanobody engineering for SARS-CoV-2 neutralization and detection

In response to the ongoing COVID-19 pandemic, the quest for coronavirus inhibitors has inspired research on a variety of small proteins beyond conventional antibodies, including robust single-domain antibody fragments, i.e., “nanobodies.” Here, we explore the potential of nanobody engineering in the development of antivirals and diagnostic tools. Through fusion of nanobody domains that target distinct binding sites, we engineered multimodular nanobody constructs that neutralize wild-type SARS-CoV-2 and the Alpha and Delta variants at high potency, with IC50 values as low as 50 pM. Despite simultaneous binding to distinct epitopes, Beta and Omicron variants were more resistant to neutralization by the multimodular nanobodies, which highlights the importance of accounting for antigenic drift in the design of biologics. To further explore the applications of nanobody engineering in outbreak management, we present an assay based on fusions of nanobodies with fragments of NanoLuc luciferase that can detect sub-nanomolar quantities of the SARS-CoV-2 spike protein in a single step. Our work showcases the potential of nanobody engineering to combat emerging infectious diseases.

Nanobodies, small protein binders derived from the camelid antibody, are highly potent inhibitors of respiratory viruses that offer several advantages over conventional antibodies as candidates for specific therapies, including high stability and low production costs. In this work, we leverage the unique properties of nanobodies and apply them as building blocks for new therapeutic and diagnostic tools. We report ultra-potent SARS-CoV-2 inhibition by engineered nanobodies comprising multiple modules in structure-guided combinations and develop nanobodies that carry signal molecules, allowing rapid detection of the SARS-CoV-2 spike protein. Our results highlight the potential of engineered nanobodies in the development of effective countermeasures, both therapeutic and diagnostic, to manage outbreaks of emerging viruses.

[end paper 30]

[begin paper 31]

PMCID = 9174178

Title = Superimmunity by pan-sarbecovirus nanobodies

Vaccine boosters and infection can facilitate the development of SARS-CoV-2 antibodies with improved potency and breadth. Here, we observe superimmunity in a camelid extensively immunized with the SARS-CoV-2 receptor-binding domain (RBD). We rapidly isolate a large repertoire of specific ultra-high-affinity nanobodies that bind strongly to all known sarbecovirus clades using integrative proteomics. These pan-sarbecovirus nanobodies (psNbs) are highly effective against SARS-CoV and SARS-CoV-2 variants, including Omicron, with the best median neutralization potency at single-digit nanograms per milliliter. A highly potent, inhalable, and bispecific psNb (PiN-31) is also developed. Structural determinations of 13 psNbs with the SARS-CoV-2 spike or RBD reveal five epitope classes, providing insights into the mechanisms and evolution of their broad activities. The highly evolved psNbs target small, flat, and flexible epitopes that contain over 75% of conserved RBD surface residues. Their potencies are strongly and negatively correlated with the distance of the epitopes from the receptor binding sites.

Xiang et al. characterize 100 diverse pan-sarbecovirus nanobodies (psNbs) that are highly effective against SARS-CoV and SARS-CoV-2 variants, including Omicron, with the best median neutralization potency at single-digit nanograms per milliliter. Structural determinations of 13 psNbs reveal five RBD epitope classes, providing insights into the mechanisms and evolution of their broad activities.

[end paper 31]

[begin paper 32]

PMCID = 10145278

Title = Identification of B-Cell Linear Epitopes in the Nucleocapsid (N) Protein B-Cell Linear Epitopes Conserved among the Main SARS-CoV-2 Variants

The Nucleocapsid (N) protein is highlighted as the main target for COVID-19 diagnosis by antigen detection due to its abundance in circulation early during infection. However, the effects of the described mutations in the N protein epitopes and the efficacy of antigen testing across SARS-CoV-2 variants remain controversial and poorly understood. Here, we used immunoinformatics to identify five epitopes in the SARS-CoV-2 N protein (N(34–48), N(89–104), N(185–197), N(277–287), and N(378–390)) and validate their reactivity against samples from COVID-19 convalescent patients. All identified epitopes are fully conserved in the main SARS-CoV-2 variants and highly conserved with SARS-CoV. Moreover, the epitopes N(185–197) and N(277–287) are highly conserved with MERS-CoV, while the epitopes N(34–48), N(89–104), N(277–287), and N(378–390) are lowly conserved with common cold coronaviruses (229E, NL63, OC43, HKU1). These data are in accordance with the observed conservation of amino acids recognized by the antibodies 7R98, 7N0R, and 7CR5, which are conserved in the SARS-CoV-2 variants, SARS-CoV and MERS-CoV but lowly conserved in common cold coronaviruses. Therefore, we support the antigen tests as a scalable solution for the population-level diagnosis of SARS-CoV-2, but we highlight the need to verify the cross-reactivity of these tests against the common cold coronaviruses.

[end paper 32]

[begin paper 33]

PMCID = 8043454

Title = Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2

Despite the great promise of vaccines, the COVID-19 pandemic is ongoing and future serious outbreaks are highly likely, so that multi-pronged containment strategies will be required for many years. Nanobodies are the smallest naturally occurring single domain antigen binding proteins identified to date, possessing numerous properties advantageous to their production and use. We present a large repertoire of high affinity nanobodies against SARS-CoV-2 Spike protein with excellent kinetic and viral neutralization properties, which can be strongly enhanced with oligomerization. This repertoire samples the epitope landscape of the Spike ectodomain inside and outside the receptor binding domain, recognizing a multitude of distinct epitopes and revealing multiple neutralization targets of pseudoviruses and authentic SARS-CoV-2, including in primary human airway epithelial cells. Combinatorial nanobody mixtures show highly synergistic activities, and are resistant to mutational escape and emerging viral variants of concern. These nanobodies establish an exceptional resource for superior COVID-19 prophylactics and therapeutics.

[end paper 33]

[begin paper 34]

PMCID = 10590698

Title = Nanobodies with cross-neutralizing activity provide prominent therapeutic efficacy in mild and severe COVID-19 rodent models

The weakened protective efficacy of COVID-19 vaccines and antibodies caused by SARS-CoV-2 variants presents a global health emergency, which underscores the urgent need for universal therapeutic antibody intervention for clinical patients. Here, we screened three alpacas-derived nanobodies (Nbs) with neutralizing activity from twenty RBD-specific Nbs. The three Nbs were fused with the Fc domain of human IgG, namely aVHH-11-Fc, aVHH-13-Fc and aVHH-14-Fc, which could specifically bind RBD protein and competitively inhibit the binding of ACE2 receptor to RBD. They effectively neutralized SARS-CoV-2 pseudoviruses D614G, Alpha, Beta, Gamma, Delta, and Omicron sub-lineages BA.1, BA.2, BA.4, and BA.5 and authentic SARS-CoV-2 prototype, Delta, and Omicron BA.1, BA.2 strains. In mice-adapted COVID-19 severe model, intranasal administration of aVHH-11-Fc, aVHH-13-Fc and aVHH-14-Fc effectively protected mice from lethal challenges and reduced viral loads in both the upper and lower respiratory tracts. In the COVID-19 mild model, aVHH-13-Fc, which represents the optimal neutralizing activity among the above three Nbs, effectively protected hamsters from the challenge of SARS-CoV-2 prototype, Delta, Omicron BA.1 and BA.2 by significantly reducing viral replication and pathological alterations in the lungs. In structural modeling of aVHH-13 and RBD, aVHH-13 binds to the receptor-binding motif region of RBD and interacts with some highly conserved epitopes. Taken together, our study illustrated that alpaca-derived Nbs offered a therapeutic countermeasure against SARS-CoV-2, including those Delta and Omicron variants which have evolved into global pandemic strains.

Alpaca-derived nanobodies showed a cross-neutralizing activity against SARS-CoV-2 variants.

aVHH-13-Fc provided complete protection in mild and severe COVID-19 rodent models.

The binding site between aVHH-13-Fc and the RBD variant of SARS-CoV-2 is partially located on the RBM.

[end paper 34]

[begin paper 35]

PMCID = 10057852

Title = Nanobodies: A Review of Generation, Diagnostics and Therapeutics

Nanobodies, also referred to as single domain-based VHHs, are antibody fragments derived from heavy-chain only IgG antibodies found in the Camelidae family. Due to their small size, simple structure, high antigen binding affinity, and remarkable stability in extreme conditions, nanobodies possess the potential to overcome several of the limitations of conventional monoclonal antibodies. For many years, nanobodies have been of great interest in a wide variety of research fields, particularly in the diagnosis and treatment of diseases. This culminated in the approval of the world’s first nanobody based drug (Caplacizumab) in 2018 with others following soon thereafter. This review will provide an overview, with examples, of (i) the structure and advantages of nanobodies compared to conventional monoclonal antibodies, (ii) methods used to generate and produce antigen-specific nanobodies, (iii) applications for diagnostics, and (iv) ongoing clinical trials for nanobody therapeutics as well as promising candidates for clinical development.

[end paper 35]

[begin paper 36]

PMCID = 11111979

Title = Next generation single-domain antibodies against respiratory zoonotic RNA viruses

The global impact of zoonotic viral outbreaks underscores the pressing need for innovative antiviral strategies, particularly against respiratory zoonotic RNA viruses. These viruses possess a high potential to trigger future epidemics and pandemics due to their high mutation rate, broad host range and efficient spread through airborne transmission. Recent pandemics caused by coronaviruses and influenza A viruses underscore the importance of developing targeted antiviral strategies. Single-domain antibodies (sdAbs), originating from camelids, also known as nanobodies or VHHs (Variable Heavy domain of Heavy chain antibodies), have emerged as promising tools to combat current and impending zoonotic viral threats. Their unique structure, coupled with attributes like robustness, compact size, and cost-effectiveness, positions them as strong alternatives to traditional monoclonal antibodies. This review describes the pivotal role of sdAbs in combating respiratory zoonotic viruses, with a primary focus on enhancing sdAb antiviral potency through optimization techniques and diverse administration strategies. We discuss both the promises and challenges within this dynamically growing field.

[end paper 36]

[begin paper 37]

PMCID = 7870875

Title = Potent neutralization of clinical isolates of SARS-CoV-2 D614 and G614 variants by a monomeric, sub-nanomolar affinity nanobody

Despite unprecedented global efforts to rapidly develop SARS-CoV-2 treatments, in order to reduce the burden placed on health systems, the situation remains critical. Effective diagnosis, treatment, and prophylactic measures are urgently required to meet global demand: recombinant antibodies fulfill these requirements and have marked clinical potential. Here, we describe the fast-tracked development of an alpaca Nanobody specific for the receptor-binding-domain (RBD) of the SARS-CoV-2 Spike protein with potential therapeutic applicability. We present a rapid method for nanobody isolation that includes an optimized immunization regimen coupled with VHH library E. coli surface display, which allows single-step selection of Nanobodies using a simple density gradient centrifugation of the bacterial library. The selected single and monomeric Nanobody, W25, binds to the SARS-CoV-2 S RBD with sub-nanomolar affinity and efficiently competes with ACE-2 receptor binding. Furthermore, W25 potently neutralizes SARS-CoV-2 wild type and the D614G variant with IC50 values in the nanomolar range, demonstrating its potential as antiviral agent.

[end paper 37]

[begin paper 38]

PMCID = 10858559

Title = Development and characterization of a novel nanobody with SRMV neutralizing activity

Peste des petits ruminants (PPR) is an acute, contact infectious disease caused by the small ruminant morbillivirus (SRMV), and its morbidity in goats and sheep can be up to 100% with significant mortality. Nanobody generated from camelid animals such as alpaca has attracted wide attention because of its unique advantages compared with conventional antibodies. The main objective of this study was to produce specific nanobodies against SRMV and identify its characteristics. To obtain the coding gene of SRMV-specific nanobodies, we first constructed an immune phage-displayed library from the VHH repertoire of alpaca that was immunized with SRMV-F and -H proteins. By using phage display technology, the target antigen-specific VHHs can be obtained after four consecutive rounds of biopanning. Results showed that the size of this VHH library was 2.26 × 1010 CFU/mL and the SRMV-F and -H specific phage particles were greatly enriched after four rounds of biopanning. The positive phage clones were selected and sequenced, and total of five independent different sequences of SRMV-specific nanobodies were identified. Subsequently, the DNA fragments of the five nanobodies were cloned into E. coli BL21(DE3), respectively, and three of them were successfully expressed and purified. Specificity and affinity towards inactivated SRMV of these purified nanobodies were then evaluated using the ELISA method. Results demonstrated that NbSRMV-1-1, NbSRMV-2-10, and NbSRMV-1-21 showed no cross-reactivity with other antigens, such as inactivated BTV, inactivated FMDV, His-tag labeled protein, and BSA. The ELISA titer of these three nanobodies against inactivated SRMV was up to 1:1000. However, only NbSRMV-1-21 displayed SRMV neutralizing activity at a maximum dilution of 1:4. The results indicate that the nanobodies against SRMV generated in this study could be useful in future applications. This study provided a novel antibody tool and laid a foundation for the treatment and detection of SRMV.

[end paper 38]

[begin paper 39]

PMCID = 8575140

Title = Camel nanobodies broadly neutralize SARS-CoV-2 variants

With the emergence of SARS-CoV-2 variants, there is urgent need to develop broadly neutralizing antibodies. Here, we isolate two VHH nanobodies (7A3 and 8A2) from dromedary camels by phage display, which have high affinity for the receptor-binding domain (RBD) and broad neutralization activities against SARS-CoV-2 and its emerging variants. Cryo-EM complex structures reveal that 8A2 binds the RBD in its up mode and 7A3 inhibits receptor binding by uniquely targeting a highly conserved and deeply buried site in the spike regardless of the RBD conformational state. 7A3 at a dose of ≥5 mg/kg efficiently protects K18-hACE2 transgenic mice from the lethal challenge of B.1.351 or B.1.617.2, suggesting that the nanobody has promising therapeutic potentials to curb the COVID-19 surge with emerging SARS-CoV-2 variants.

Dromedary camel (Camelus dromedarius) VHH phage libraries were built for isolation of the nanobodies that broadly neutralize SARS-CoV-2 variants.

[end paper 39]

[begin paper 40]

PMCID = 8448731

Title = A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies

Antibody engineering technologies face increasing demands for speed, reliability and scale. We develop CeVICA, a cell-free nanobody engineering platform that uses ribosome display for in vitro selection of nanobodies from a library of 1011 randomized sequences. We apply CeVICA to engineer nanobodies against the Receptor Binding Domain (RBD) of SARS-CoV-2 spike protein and identify >800 binder families using a computational pipeline based on CDR-directed clustering. Among 38 experimentally-tested families, 30 are true RBD binders and 11 inhibit SARS-CoV-2 pseudotyped virus infection. Affinity maturation and multivalency engineering increase nanobody binding affinity and yield a virus neutralizer with picomolar IC50. Furthermore, the capability of CeVICA for comprehensive binder prediction allows us to validate the fitness of our nanobody library. CeVICA offers an integrated solution for rapid generation of divergent synthetic nanobodies with tunable affinities in vitro and may serve as the basis for automated and highly parallel nanobody engineering.

Faster, higher throughput antibody engineering methods are needed. Here the authors present CeVICA, a cell-free nanobody engineering platform using ribosome display and computational clustering analysis for in vitro selection; they use this to develop nanobodies against the RBD of SARS-CoV-2 spike protein.

[end paper 40]

[begin paper 41]

PMCID = 9132424

Title = Deep mutational engineering of broadly-neutralizing nanobodies accommodating SARS-CoV-1 and 2 antigenic drift

Here, we report the molecular engineering of nanobodies that bind with picomolar affinity to both SARS-CoV-1 and SARS-CoV-2 receptor-binding domains (RBD) and are highly neutralizing. We applied deep mutational engineering to VHH72, a nanobody initially specific for SARS-CoV-1 RBD with little cross-reactivity to SARS-CoV-2 antigen. We first identified all the individual VHH substitutions that increase binding to SARS-CoV-2 RBD and then screened highly focused combinatorial libraries to isolate engineered nanobodies with improved properties. The corresponding VHH-Fc molecules show high affinities for SARS-CoV-2 antigens from various emerging variants and SARS-CoV-1, block the interaction between ACE2 and RBD, and neutralize the virus with high efficiency. Its rare specificity across sarbecovirus relies on its peculiar epitope outside the immunodominant regions. The engineered nanobodies share a common motif of three amino acids, which contribute to the broad specificity of recognition. Our results show that deep mutational engineering is a very powerful method, especially to rapidly adapt existing antibodies to new variants of pathogens.

[end paper 41]

[begin paper 42]

PMCID = 9393763

Title = Antibody-mediated immunity to SARS-CoV-2 spike

Despite effective spike-based vaccines and monoclonal antibodies, the SARS-CoV-2 pandemic continues more than two and a half years post-onset. Relentless investigation has outlined a causative dynamic between host-derived antibodies and reciprocal viral subversion. Integration of this paradigm into the architecture of next generation antiviral strategies, predicated on a foundational understanding of the virology and immunology of SARS-CoV-2, will be critical for success. This review aims to serve as a primer on the immunity endowed by antibodies targeting SARS-CoV-2 spike protein through a structural perspective. We begin by introducing the structure and function of spike, polyclonal immunity to SARS-CoV-2 spike, and the emergence of major SARS-CoV-2 variants that evade immunity. The remainder of the article comprises an in-depth dissection of all major epitopes on SARS-CoV-2 spike in molecular detail, with emphasis on the origins, neutralizing potency, mechanisms of action, cross-reactivity, and variant resistance of representative monoclonal antibodies to each epitope.

[end paper 42]

[begin paper 43]

PMCID = 8928829

Title = Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19

There remains an unmet need for globally deployable, low-cost therapeutics for the ongoing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic. Previously, we reported on the isolation and in vitro characterization of a potent single-domain nanobody, NIH-CoVnb-112, specific for the receptor-binding domain (RBD) of SARS-CoV-2. Here, we report on the molecular basis for the observed broad in vitro neutralization capability of NIH-CoVnb-112 against variant SARS-CoV-2 pseudoviruses. The structure of NIH-CoVnb-112 bound to SARS-CoV-2 RBD reveals a large contact surface area overlapping the angiotensin converting enzyme 2 (ACE2) binding site, which is largely unencumbered by the common RBD mutations. In an in vivo pilot study, we demonstrate effective reductions in weight loss, viral burden, and lung pathology in a Syrian hamster model of COVID-19 following nebulized delivery of NIH-CoVnb-112. These findings support the further development of NIH-CoVnb-112 as a potential adjunct preventative therapeutic for the treatment of SARS-CoV-2 infection.

Abbreviations: ACE2 – angiotensin converting enzyme 2BSA – buried surface areaCDR – complementary determining regionRBD – receptor binding domainRBM – receptor-binding motifSARS-CoV-2 - severe acute respiratory syndrome coronavirus 2

[end paper 43]

[begin paper 44]

PMCID = 8726723

Title = Rapid discovery of diverse neutralizing SARS-CoV-2 antibodies from large-scale synthetic phage libraries

Coronavirus disease 2019 (COVID-19) is an evolving global public health crisis in need of therapeutic options. Passive immunization of monoclonal antibodies (mAbs) represents a promising therapeutic strategy capable of conferring immediate protection from SARS-CoV-2 infection. Herein, we describe the discovery and characterization of neutralizing SARS-CoV-2 IgG and VHH antibodies from four large-scale phage libraries. Each library was constructed synthetically with shuffled complementarity-determining region loops from natural llama and human antibody repertoires. While most candidates targeted the receptor-binding domain of the S1 subunit of SARS-CoV-2 spike protein, we also identified a neutralizing IgG candidate that binds a unique epitope on the N-terminal domain. A select number of antibodies retained binding to SARS-CoV-2 variants Alpha, Beta, Gamma, Kappa and Delta. Overall, our data show that synthetic phage libraries can rapidly yield SARS-CoV-2 S1 antibodies with therapeutically desirable features, including high affinity, unique binding sites, and potent neutralizing activity in vitro, and a capacity to limit disease in vivo.

[end paper 44]

[begin paper 45]

PMCID = 9000494

Title = Structural Insights into the Design of Synthetic Nanobody Libraries

Single domain antibodies from camelids, or nanobodies, are a unique class of antibody fragments with several advantageous characteristics: small monomeric size, high stability and solubility and easy tailoring for multiple applications. Nanobodies are gaining increasing acceptance as diagnostic tools and promising therapeutic agents in cancer and other diseases. While most nanobodies are obtained from immunized animals of the camelid family, a few synthetic nanobody libraries constructed in recent years have shown the capability of generating high quality nanobodies in terms of affinity and stability. Since this synthetic approach has important advantages over the use of animals, the recent advances are indeed encouraging. Here we review over a dozen synthetic nanobody libraries reported so far and discuss the different approaches followed in their construction and validation, with an emphasis on framework and hypervariable loop design as critical issues defining their potential as high-class nanobody sources.

[end paper 45]

[begin paper 46]

PMCID = 9392344

Title = K205R specific nanobody-horseradish peroxidase fusions as reagents of competitive ELISA to detect African swine fever virus serum antibodies

African swine fever virus (ASFV) is a highly contagious hemorrhagic disease and often lethal, which has significant economic consequences for the swine industry. Due to lacking of commercial vaccine, the prevention and control of ASF largely depend on early large-scale detection and screening. So far, the commercial ELISA kits have a long operation time and are expensive, making it difficult to achieve large-scale clinical applications. Nanobodies are single-domain antibodies produced by camelid animals, and have unique advantages such as smaller molecular weight, easy genetic engineering modification and low-costing of mass production, thus exhibiting good application prospects.

The present study developed a new method for detection of ASFV specific antibodies using nanobody-horseradish peroxidase (Nb-HRP) fusion proteins as probe. By using camel immunization, phage library construction and phage display technology, five nanobodies against K205R protein were screened. Then, Nb-HRP fusion proteins were produced using genetic modification technology. Based on the Nb-HRP fusion protein as specific antibodies against K205R protein, a new type of cELISA was established to detect ASFV antibodies in pig serum. The cut-off value of the cELISA was 34.8%, and its sensitivity, specificity, and reproducibility were good. Furthermore, the developed cELISA exhibited 99.3% agreement rate with the commercial available ELISA kit (kappa value = 0.98).

The developed cELISA method has the advantages of simple operation, rapid and low-costing, and can be used for monitoring of ASFV infection in pigs, thus providing a new method for the prevention and control of ASF.

The online version contains supplementary material available at 10.1186/s12917-022-03423-0.

[end paper 46]

[begin paper 47]

PMCID = 11209230

Title = Comprehensive Overview of Broadly Neutralizing Antibodies against SARS-CoV-2 Variants

Currently, SARS-CoV-2 has evolved into various variants, including the numerous highly mutated Omicron sub-lineages, significantly increasing immune evasion ability. The development raises concerns about the possibly diminished effectiveness of available vaccines and antibody-based therapeutics. Here, we describe those representative categories of broadly neutralizing antibodies (bnAbs) that retain prominent effectiveness against emerging variants including Omicron sub-lineages. The molecular characteristics, epitope conservation, and resistance mechanisms of these antibodies are further detailed, aiming to offer suggestion or direction for the development of therapeutic antibodies, and facilitate the design of vaccines with broad-spectrum potential.

[end paper 47]

[begin paper 48]

PMCID = 9288967

Title = Accelerating PERx reaction enables covalent nanobodies for potent neutralization of SARS-CoV-2 and variants

The long-lasting COVID-19 pandemic and increasing SARS-CoV-2 variants demand effective drugs for prophylactics and treatment. Protein-based biologics offer high specificity, yet their noncovalent interactions often lead to drug dissociation and incomplete inhibition. Here, we have developed covalent nanobodies capable of binding with SARS-CoV-2 irreversibly via a proximity-enabled reactive therapeutic (PERx) mechanism. A latent bioreactive amino acid (FFY) was designed and genetically encoded into nanobodies to accelerate the PERx reaction rate. Compared with the noncovalent wild-type nanobody, the FFY-incorporated covalent nanobodies neutralized both wild-type SARS-CoV-2 and its Alpha, Delta, Epsilon, Lambda, and Omicron variants with drastically higher potency. This PERx-enabled covalent-nanobody strategy and the related insights into increased potency can be valuable to developing effective therapeutics for various viral infections.

The long-lasting COVID-19 pandemic and increasing SARS-CoV-2 variants demand effective therapeutic strategies. Herein, we have engineered unique unnatural amino acids into proteins and developed covalent nanobodies capable of irreversibly binding with the viral spike protein. These covalent nanobodies are able to neutralize both SARS-CoV-2 and its different variants with drastic higher potency than traditional nanobodies, affording a potential prophylactic and medication for COVID-19. In addition, our strategy could provide an original general route to developing effective therapeutics for various other viral infections, such as influenza, hepatitis, AIDS, and anthrax.

Traditional nanobodies interact with their targets reversibly, leading to dissociation and incomplete inhibition. Wang and co-workers have designed a latent bioreactive amino acid (FFY) and genetically encoded it into nanobodies. Upon nanobody binding to the viral spike protein, FFY covalently reacts with spike via a proximity-enabled reactive therapeutic (PERx) mechanism, enabling the nanobody to bind to SARS-CoV-2 irreversibly and effectively block viral infection. The FFY-incorporated covalent nanobodies neutralized both wild-type SARS-CoV-2 and its various variants with drastically higher potency than the noncovalent wild-type nanobodies.

[end paper 48]

[begin paper 49]

PMCID = 11515141

Title = Discovery of nanobodies: a comprehensive review of their applications and potential over the past five years

Nanobodies (Nbs) are antibody fragments derived from heavy-chain-only IgG antibodies found in the Camelidae family as well as cartilaginous fish. Their unique structural and functional properties, such as their small size, the ability to be engineered for high antigen-binding affinity, stability under extreme conditions, and ease of production, have made them promising tools for diagnostics and therapeutics. This potential was realized in 2018 with the approval of caplacizumab, the world’s first Nb-based drug. Currently, Nbs are being investigated in clinical trials for a broad range of treatments, including targeted therapies against PDL1 and Epidermal Growth Factor Receptor (EGFR), cardiovascular diseases, inflammatory conditions, and neurodegenerative disorders such as Alzheimer’s disease, Parkinson’s disease, and amyotrophic lateral sclerosis. They are also being studied for their potential for detecting and imaging autoimmune conditions and infectious diseases such as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). A variety of methods are now available to generate target-specific Nbs quickly and efficiently at low costs, increasing their accessibility. This article examines these diverse applications of Nbs and their promising roles. Only the most recent articles published in the last five years have been used to summarize the most advanced developments in the field.

[end paper 49]

[begin paper 50]

PMCID = 10060843

Title = Review of therapeutic mechanisms and applications based on SARS-CoV-2 neutralizing antibodies

COVID-19 pandemic is a global public health emergency. Despite extensive research, there are still few effective treatment options available today. Neutralizing-antibody-based treatments offer a broad range of applications, including the prevention and treatment of acute infectious diseases. Hundreds of SARS-CoV-2 neutralizing antibody studies are currently underway around the world, with some already in clinical applications. The development of SARS-CoV-2 neutralizing antibody opens up a new therapeutic option for COVID-19. We intend to review our current knowledge about antibodies targeting various regions (i.e., RBD regions, non-RBD regions, host cell targets, and cross-neutralizing antibodies), as well as the current scientific evidence for neutralizing-antibody-based treatments based on convalescent plasma therapy, intravenous immunoglobulin, monoclonal antibodies, and recombinant drugs. The functional evaluation of antibodies (i.e., in vitro or in vivo assays) is also discussed. Finally, some current issues in the field of neutralizing-antibody-based therapies are highlighted.

[end paper 50]

[begin paper 51]

PMCID = 11311977

Title = Enhanced Assessment of Cross-Reactive Antigenic Determinants within the Spike Protein

Despite successful vaccination efforts, the emergence of new SARS-CoV-2 variants poses ongoing challenges to control COVID-19. Understanding humoral responses regarding SARS-CoV-2 infections and their impact is crucial for developing future vaccines that are effective worldwide. Here, we identified 41 immunodominant linear B-cell epitopes in its spike glycoprotein with an SPOT synthesis peptide array probed with a pool of serum from hospitalized COVID-19 patients. The bioinformatics showed a restricted set of epitopes unique to SARS-CoV-2 compared to other coronavirus family members. Potential crosstalk was also detected with Dengue virus (DENV), which was confirmed by screening individuals infected with DENV before the COVID-19 pandemic in a commercial ELISA for anti-SARS-CoV-2 antibodies. A high-resolution evaluation of antibody reactivity against peptides representing epitopes in the spike protein identified ten sequences in the NTD, RBD, and S2 domains. Functionally, antibody-dependent enhancement (ADE) in SARS-CoV-2 infections of monocytes was observed in vitro with pre-pandemic Dengue-positive sera. A significant increase in viral load was measured compared to that of the controls, with no detectable neutralization or considerable cell death, suggesting its role in viral entry. Cross-reactivity against peptides from spike proteins was observed for the pre-pandemic sera. This study highlights the importance of identifying specific epitopes generated during the humoral response to a pathogenic infection to understand the potential interplay of previous and future infections on diseases and their impact on vaccinations and immunodiagnostics.

[end paper 51]

[begin paper 52]

PMCID = 11723632

Title = Discovery of Nanosota-EB1 and -EB2 as Novel Nanobody Inhibitors Against Ebola Virus Infection

The Ebola filovirus (EBOV) poses a serious threat to global health and national security. Nanobodies, a type of single-domain antibody, have demonstrated promising therapeutic potential. We identified two anti-EBOV nanobodies, Nanosota-EB1 and Nanosota-EB2, which specifically target the EBOV glycoprotein (GP). Cryo-EM and biochemical data revealed that Nanosota-EB1 binds to the glycan cap of GP1, preventing its protease cleavage, while Nanosota-EB2 binds to critical membrane-fusion elements in GP2, stabilizing it in the pre-fusion state. Nanosota-EB2 is a potent neutralizer of EBOV infection in vitro and offers excellent protection in a mouse model of EBOV challenge, while Nanosota-EB1 provides moderate neutralization and protection. Nanosota-EB1 and Nanosota-EB2 are the first nanobodies shown to inhibit authentic EBOV. Combined with our newly developed structure-guided in vitro evolution approach, they lay the foundation for nanobody-based therapies against EBOV and other viruses within the ebolavirus genus.

Ebola virus (EBOV) poses a serious global health threat due to its high fatality rate. Current antibody-based treatments face significant challenges, including reducing mortality, addressing viral persistence in some survivors, and ensuring affordability and accessibility. Furthermore, adapting existing human antibody therapies to combat EBOV-related viruses, such as Bundibugyo ebolavirus (BDBV) and Sudan ebolavirus (SUDV), remains difficult. Nanobodies, with their modular single-domain structure, present a promising solution to these challenges. In this study, we identified two anti-EBOV nanobodies, Nanosota-EB1 and Nanosota-EB2, which specifically target the EBOV GP protein. Cryo-EM and biochemical analyses revealed that Nanosota-EB1 binds to the glycan cap of GP1, preventing protease cleavage essential for viral infection. In contrast, Nanosota-EB2 binds to GP2, stabilizing it in the pre-fusion state and blocking its ability to mediate fusion between viral and host membranes. Nanosota-EB2 demonstrates strong neutralizing activity against EBOV infection in vitro and provides robust protection in a mouse model of EBOV challenge, while Nanosota-EB1 offers moderate neutralization and protection. As the first nanobodies shown to be effective against authentic EBOV, Nanosota-EB1 and Nanosota-EB2, together with our recently developed nanobody evolution strategy, establish a foundation for nanobody-based therapies targeting EBOV and related filoviruses.

[end paper 52]

[begin paper 53]

PMCID = 9529347

Title = Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2

The development of therapeutics to prevent or treat COVID-19 remains an area of intense focus. Protein biologics, including monoclonal antibodies and nanobodies that neutralize virus, have potential for the treatment of active disease. Here, we have used yeast display of a synthetic nanobody library to isolate nanobodies that bind the receptor-binding domain (RBD) of SARS-CoV-2 and neutralize the virus. We show that combining two clones with distinct binding epitopes within the RBD into a single protein construct to generate biparatopic reagents dramatically enhances their neutralizing capacity. Furthermore, the biparatopic nanobodies exhibit enhanced control over clinically relevant RBD variants that escaped recognition by the individual nanobodies. Structural analysis of biparatopic binding to spike (S) protein revealed a unique binding mode whereby the two nanobody paratopes bridge RBDs encoded by distinct S trimers. Accordingly, biparatopic nanobodies offer a way to rapidly generate powerful viral neutralizers with enhanced ability to control viral escape mutants.

Yeast display was used to generate SARS-CoV-2 RBD-specific nanobodies

Distinct nanobody clones were fused to create biparatopic reagents

Biparatopic nanobodies protect mice from SARS-CoV-2 infection

The biparatopics bridge distinct S proteins in a unique “twinned-S” conformation

Immunology; Microbiology; Structural biology.

[end paper 53]

[begin paper 54]

PMCID = 8458290

Title = A potent SARS-CoV-2 neutralising nanobody shows therapeutic efficacy in the Syrian golden hamster model of COVID-19

SARS-CoV-2 remains a global threat to human health particularly as escape mutants emerge. There is an unmet need for effective treatments against COVID-19 for which neutralizing single domain antibodies (nanobodies) have significant potential. Their small size and stability mean that nanobodies are compatible with respiratory administration. We report four nanobodies (C5, H3, C1, F2) engineered as homotrimers with pmolar affinity for the receptor binding domain (RBD) of the SARS-CoV-2 spike protein. Crystal structures show C5 and H3 overlap the ACE2 epitope, whilst C1 and F2 bind to a different epitope. Cryo Electron Microscopy shows C5 binding results in an all down arrangement of the Spike protein. C1, H3 and C5 all neutralize the Victoria strain, and the highly transmissible Alpha (B.1.1.7 first identified in Kent, UK) strain and C1 also neutralizes the Beta (B.1.35, first identified in South Africa). Administration of C5-trimer via the respiratory route showed potent therapeutic efficacy in the Syrian hamster model of COVID-19 and separately, effective prophylaxis. The molecule was similarly potent by intraperitoneal injection.

Neutralizing nanobodies (Nb) are of considerable interest as therapeutic agents for COVID-19 treatment. Here, the authors functionally and structurally characterize Nbs that bind with high affinity to the receptor binding domain of the SARS-CoV-2 spike protein and show that an engineered homotrimeric Nb prevents disease progression in a Syrian hamster model of COVID-19 when administered intranasally.

[end paper 54]

[begin paper 55]

PMCID = 8863326

Title = A potent alpaca-derived nanobody that neutralizes SARS-CoV-2 variants

The spike glycoprotein of SARS-CoV-2 engages with human ACE 2 to facilitate infection. Here, we describe an alpaca-derived heavy chain antibody fragment (VHH), saRBD-1, that disrupts this interaction by competitively binding to the spike protein receptor-binding domain. We further generated an engineered bivalent nanobody construct engineered by a flexible linker and a dimeric Fc conjugated nanobody construct. Both multivalent nanobodies blocked infection at picomolar concentrations and demonstrated no loss of potency against emerging variants of concern including Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), Epsilon (B.1.427/429), and Delta (B.1.617.2). saRBD-1 tolerates elevated temperature, freeze-drying, and nebulization, making it an excellent candidate for further development into a therapeutic approach for COVID-19.

SARS-CoV-2 variants effectively neutralized by saRBD-1 VHH with picomolar affinity

saRBD-1 neutralization increases when expressed as a bivalent or Fc construct

saRBD-1 binds SARS-CoV-2 RBD as a likely class 1 neutralizing antibody

saRBD-1 retains binding, neutralization after heat and nebulization treatments

Applied microbiology; Bioengineering; Nanotechnology

[end paper 55]

[begin paper 56]

PMCID = 10138814

Title = Structural Insights into the Binding of Red Fluorescent Protein mCherry-Specific Nanobodies

Red fluorescent proteins (RFPs) have broad applications in life science research, and the manipulation of RFPs using nanobodies can expand their potential uses. However, the structural information available for nanobodies that bind with RFPs is still insufficient. In this study, we cloned, expressed, purified, and crystallized complexes formed by mCherry with LaM1, LaM3, and LaM8. Then, we analyzed the biochemical properties of the complexes using mass spectrometry (MS), fluorescence-detected size exclusion chromatography (FSEC), isothermal titration calorimetry (ITC), and bio-layer interferometry (BLI) technology. We determined the crystal structure of mCherry-LaM1, mCherry-LaM3, and mCherry-LaM8, with resolutions of 2.05 Å, 3.29 Å, and 1.31 Å, respectively. In this study, we systematically compared various parameters of several LaM series nanobodies, including LaM1, LaM3, and LaM8, with previously reported data on LaM2, LaM4, and LaM6, specifically examining their structural information. After designing multivalent tandem LaM1-LaM8 and LaM8-LaM4 nanobodies based on structural information, we characterized their properties, revealing their higher affinity and specificity to mCherry. Our research provides novel structural insights that could aid in understanding nanobodies targeting a specific target protein. This could provide a starting point for developing enhanced mCherry manipulation tools.

[end paper 56]

[begin paper 57]

PMCID = 10054972

Title = Nanobodies against SARS-CoV-2 reduced virus load in the brain of challenged mice and neutralized Wuhan, Delta and Omicron Variants

In this work, we developed llama-derived nanobodies (Nbs) directed to the receptor binding domain (RBD) and other domains of the Spike (S) protein of SARS-CoV-2. Nanobodies were selected after the biopanning of two VHH-libraries, one of which was generated after the immunization of a llama (lama glama) with the bovine coronavirus (BCoV) Mebus, and another with the full-length pre-fused locked S protein (S-2P) and the RBD from the SARS-CoV-2 Wuhan strain (WT). Most of the neutralizing Nbs selected with either RBD or S-2P from SARS-CoV-2 were directed to RBD and were able to block S-2P/ACE2 interaction. Three Nbs recognized the N-terminal domain (NTD) of the S-2P protein as measured by competition with biliverdin, while some non-neutralizing Nbs recognize epitopes in the S2 domain. One Nb from the BCoV immune library was directed to RBD but was non-neutralizing. Intranasal administration of Nbs induced protection ranging from 40% to 80% against COVID-19 death in k18-hACE2 mice challenged with the WT strain. Interestingly, protection was not only associated with a significant reduction of virus replication in nasal turbinates and lungs, but also with a reduction of virus load in the brain. Employing pseudovirus neutralization assays, we were able to identify Nbs with neutralizing capacity against the Alpha, Beta, Delta and Omicron variants. Furthermore, cocktails of different Nbs performed better than individual Nbs to neutralize two Omicron variants (B.1.529 and BA.2). Altogether, the data suggest these Nbs can potentially be used as a cocktail for intranasal treatment to prevent or treat COVID-19 encephalitis, or modified for prophylactic administration to fight this disease.

[end paper 57]

[begin paper 58]

PMCID = 9512078

Title = Mucosal nanobody IgA as inhalable and affordable prophylactic and therapeutic treatment against SARS-CoV-2 and emerging variants

Anti-COVID antibody therapeutics have been developed but not widely used due to their high cost and escape of neutralization from the emerging variants. Here, we describe the development of VHH-IgA1.1, a nanobody IgA fusion molecule as an inhalable, affordable and less invasive prophylactic and therapeutic treatment against SARS-CoV-2 Omicron variants. VHH-IgA1.1 recognizes a conserved epitope of SARS-CoV-2 spike protein Receptor Binding Domain (RBD) and potently neutralizes major global SARS-CoV-2 variants of concern (VOC) including the Omicron variant and its sub lineages BA.1.1, BA.2 and BA.2.12.1. VHH-IgA1.1 is also much more potent against Omicron variants as compared to an IgG Fc fusion construct, demonstrating the importance of IgA mediated mucosal protection for Omicron infection. Intranasal administration of VHH-IgA1.1 prior to or after challenge conferred significant protection from severe respiratory disease in K18-ACE2 transgenic mice infected with SARS-CoV-2 VOC. More importantly, for cost-effective production, VHH-IgA1.1 produced in Pichia pastoris had comparable potency to mammalian produced antibodies. Our study demonstrates that intranasal administration of affordably produced VHH-IgA fusion protein provides effective mucosal immunity against infection of SARS-CoV-2 including emerging variants.

[end paper 58]

[begin paper 59]

PMCID = 8126837

Title = Nanobody cocktails potently neutralize SARS-CoV-2 D614G N501Y variant and protect mice

Neutralizing antibodies are important for immunity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and as therapeutics for the prevention and treatment of COVID-19. We identified high-affinity nanobodies against SARS-CoV-2 receptor-binding domain and found that nanobody cocktails consisting of two noncompeting nanobodies were able to block ACE2 engagement with RBD variants present in human populations and potently neutralize both wild-type SARS-CoV-2 and the N501Y D614G variant at low concentrations. Prophylactic administration of nanobody cocktails reduced viral loads in mice infected with the N501Y D614G SARS-CoV-2 virus, showing that nanobody cocktails are useful as prophylactic agents against SARS-CoV-2.

Neutralizing antibodies are important for immunity against SARS-CoV-2 and as therapeutics for the prevention and treatment of COVID-19. Here, we identified high-affinity nanobodies from alpacas immunized with coronavirus spike and receptor-binding domains (RBD) that disrupted RBD engagement with the human receptor angiotensin-converting enzyme 2 (ACE2) and potently neutralized SARS-CoV-2. Epitope mapping, X-ray crystallography, and cryo-electron microscopy revealed two distinct antigenic sites and showed two neutralizing nanobodies from different epitope classes bound simultaneously to the spike trimer. Nanobody-Fc fusions of the four most potent nanobodies blocked ACE2 engagement with RBD variants present in human populations and potently neutralized both wild-type SARS-CoV-2 and the N501Y D614G variant at concentrations as low as 0.1 nM. Prophylactic administration of either single nanobody-Fc or as mixtures reduced viral loads by up to 104-fold in mice infected with the N501Y D614G SARS-CoV-2 virus. These results suggest a role for nanobody-Fc fusions as prophylactic agents against SARS-CoV-2.

[end paper 59]

[begin paper 60]

PMCID = 10028217

Title = Efficient virus detection utilizing chitin-immobilized nanobodies synthesized in Ustilago maydis

The COVID-19 pandemic has greatly impacted the global economy and health care systems, illustrating the urgent need for timely and inexpensive responses to pandemic threats in the form of vaccines and antigen tests. Currently, antigen testing is mostly conducted by qualitative flow chromatography or via quantitative ELISA-type assays. The latter mostly utilize materials like protein-adhesive polymers and gold or latex particles. Here we present an alternative ELISA approach using inexpensive, biogenic materials and permitting quick detection based on components produced in the microbial model Ustilago maydis. In this fungus, heterologous proteins like biopharmaceuticals can be exported by fusion to unconventionally secreted chitinase Cts1. As a unique feature, the carrier chitinase binds to chitin allowing its additional use as a purification or immobilization tag. Recent work has demonstrated that nanobodies are suitable target proteins. These proteins represent a very versatile alternative antibody format and can quickly be adapted to detect novel antigens by camelidae immunization or synthetic libraries. In this study, we exemplarily produced different mono- and bivalent SARS-CoV-2 nanobodies directed against the spike protein receptor binding domain (RBD) as Cts1 fusions and screened their antigen binding affinity in vitro and in vivo. Functional nanobody-Cts1 fusions were immobilized on chitin forming an RBD tethering surface. This provides a solid base for future development of inexpensive antigen tests utilizing unconventionally secreted nanobodies as antigen trap and a matching ubiquitous and biogenic surface for immobilization.

[end paper 60]

[begin paper 61]

PMCID = 10878242

Title = A novel strategy for an anti-idiotype vaccine: nanobody mimicking neutralization epitope of porcine circovirus type 2

Vaccination is the most effective method to protect humans and animals from diseases. Anti-idiotype vaccines are safer due to their absence of pathogens. However, the commercial production of traditional anti-idiotype vaccines using monoclonal and polyclonal antibodies (mAb and pAb) is complex and has a high failure rate. The present study designed a novel, simple, low-cost strategy for developing anti-idiotype vaccines with nanobody technology. We used porcine circovirus type 2 (PCV2) as a viral model, which can result in serious economic loss in the pig industry. The neutralizing mAb-1E7 (Ab1) against PCV2 capsid protein (PCV2-Cap) was immunized in the camel. And 12 nanobodies against mAb-1E7 were screened. Among them, Nb61 (Ab2) targeted the idiotype epitope of mAb-1E7 and blocked mAb-1E7’s binding to PCV2-Cap. Additionally, a high-dose Nb61 vaccination can also protect mice and pigs from PCV2 infection. Epitope mapping showed that mAb-1E7 recognized the 75NINDFL80 of PCV2-Cap and 101NYNDFLG107 of Nb61. Subsequently, the mAb-3G4 (Ab3) against Nb61 was produced and can neutralize PCV2 infection in the PK-15 cells. Structure analysis showed that the amino acids of mAb-1E7 and mAb-3G4 respective binding to PCV2-Cap and Nb61 were also similar on the amino acids sequences and spatial conformation. Collectively, our study first provided a strategy for producing nanobody-based anti-idiotype vaccines and identified that anti-idiotype nanobodies could mimic the antigen on amino acids and structures. Importantly, as more and more neutralization mAbs against different pathogens are prepared, anti-idiotype nanobody vaccines can be easily produced against the disease with our strategy, especially for dangerous pathogens.

Anti-idiotype vaccines utilize idiotype-anti-idiotype network theory, eliminating the need for external antigens as vaccine candidates. Especially for dangerous pathogens, they were safer because they did not contact the live pathogenic microorganisms. However, developing anti-idiotype vaccines with traditional monoclonal and polyclonal antibodies is complex and has a high failure rate. We present a novel, universal, simple, low-cost strategy for producing anti-idiotype vaccines with nanobody technology. Using a neutralization antibody against PCV2-Cap, a nanobody (Ab2) was successfully produced and could mimic the neutralizing epitope of PCV2-Cap. The nanobody can induce protective immune responses against PCV2 infection in mice and pigs. It highlighted that the anti-idiotype vaccine using nanobody has a very good application in the future, especially for dangerous pathogens.

[end paper 61]

[begin paper 62]

PMCID = 10517151

Title = Camel nanobody-based B7-H3 CAR-T cells show high efficacy against large solid tumours

Rational design of chimeric antigen receptor T (CAR-T) cells based on the recognition of antigenic epitopes capable of evoking the most potent CAR activation is an important objective in optimizing immune therapy. In solid tumors, the B7-H3 transmembrane protein is an emerging target that harbours two distinct epitope motifs, IgC and IgV, in its ectodomain. Here, we generate dromedary camel nanobodies targeting B7-H3 and demonstrate that CAR-T cells, based on the nanobodies recognizing the IgC but not IgV domain, had potent antitumour activity against large tumors in female mice. These CAR-T cells are characterized by highly activated T cell signaling and significant tumor infiltration. Single-cell transcriptome RNA sequencing coupled with functional T-cell proteomics analysis uncovers the top-upregulated genes that might be critical for the persistence of polyfunctional CAR-T cells in mice. Our results highlight the importance of the specific target antigen epitope in governing optimal CAR-T activity and provide a nanobody-based B7-H3 CAR-T product for use in solid tumor therapy.

Chimeric antigen receptor T (CAR-T) cells represent an emerging form of immune therapy but success, especially in solid tumors, is limited by the scarcity of suitable target epitopes. Here authors show that a distinct epitope motif on the transmembrane protein B7-H3, recognized by a camel nanobody, triggers robust activation and anti-tumor function in cognate CAR-T cells.

[end paper 62]

[begin paper 63]

PMCID = 9656363

Title = Screening, Expression, and Identification of Nanobody against SARS-CoV-2 Spike Protein

Coronavirus disease 2019 (COVID-19) is caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), an infectious disease that has become a serious burden on global public health. This study screened and yielded specific nanobodies (Nbs) against SARS-CoV-2 spike protein receptor binding domain (RBD), following testing its basic characteristics. A nanobody phage library was established by immunizing a camel with RBD protein. After three rounds of panning, the positive colonies were screened by enzyme-linked immunosorbent assay (ELISA). By sequencing, four different sequences of nanobody gene fragments were selected. The four nanobody fusion proteins were expressed and purified, respectively. The specificity and affinity of the four nanobodies were identified by ELISA. Our results showed that an immune phage display library against SARS-CoV-2 has been successfully constructed with a library capacity of which was 4.7 × 108 CFU. The four purified nanobodies showed specific high-affinity binding SARS-CoV-2 S-RBD. Among these, the antigen binding affinity of Nb61 was more comparable to that of commercial rabbit anti-SARS-CoV-2 S-RBD antibodies. In sum, our study has obtained four nanobody strains against SARS-CoV-2 S-RBD with significant affinity and specificity, therefore laying an essential foundation for further research as well as the applications of diagnostic and therapeutic tools of SARS-CoV-2.

[end paper 63]

[begin paper 64]

PMCID = 11793827

Title = Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies

Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) continues to persist, demonstrating the risks posed by emerging infectious diseases to national security, public health, and the economy. Development of new vaccines and antibodies for emerging viral threats requires substantial resources and time, and traditional development platforms for vaccines and antibodies are often too slow to combat continuously evolving immunological escape variants, reducing their efficacy over time. Previously, we designed a next-generation synthetic humanized nanobody (Nb) phage display library and demonstrated that this library could be used to rapidly identify highly specific and potent neutralizing heavy chain-only antibodies (HCAbs) with prophylactic and therapeutic efficacy in vivo against the original SARS-CoV-2. In this study, we used a combination of high throughput screening and machine learning (ML) models to identify HCAbs with potent efficacy against SARS-CoV-2 viral variants of interest (VOIs) and concern (VOCs). To start, we screened our highly diverse Nb phage display library against several pre-Omicron VOI and VOC receptor binding domains (RBDs) to identify panels of cross-reactive HCAbs. Using HCAb affinity for SARS-CoV-2 VOI and VOCs (pre-Omicron variants) and model features from other published data, we were able to develop a ML model that successfully identified HCAbs with efficacy against Omicron variants, independent of our experimental biopanning workflow. This biopanning informed ML approach reduced the experimental screening burden by 78% to 90% for the Omicron BA.5 and Omicron BA.1 variants, respectively. The combined approach can be applied to other emerging viruses with pandemic potential to rapidly identify effective therapeutic antibodies against emerging variants.

We leveraged a robust experimental pipeline for heavy chain-only (HCAb) library screening to identify 59 potent HCAbs that can cross-neutralize different SARS-CoV-2 variants. Several of these HCAbs with efficacy against different variants also bind to different SARS-CoV-2 epitopes, suggesting they could be used in antibody cocktails or be engineered as bispecific antibodies that are cross-variant and resistant to viral escape. Using existing published data and data generated from our library of HCAbs against diverse pre-Omicron SARS-CoV-2 variants, we developed an ML model to rapidly predict which HCAbs would be effective against Omicron BA.1 and Omicron BA.5. Using this ML model three HCAbs with efficacy against Omicron BA.1 and BA.5 were identified without requiring additional biopanning. This integrated computational and experimental pipeline can be leveraged for future outbreaks to rapidly identify cross-variant countermeasures that are effective against potential emerging variants.

[end paper 64]

[begin paper 65]

PMCID = 9822811

Title = Biparatopic antibody BA7208/7125 effectively neutralizes SARS-CoV-2 variants including Omicron BA.1-BA.5

SARS-CoV-2 Omicron subvariants have demonstrated extensive evasion from monoclonal antibodies (mAbs) developed for clinical use, which raises an urgent need to develop new broad-spectrum mAbs. Here, we report the isolation and analysis of two anti-RBD neutralizing antibodies BA7208 and BA7125 from mice engineered to produce human antibodies. While BA7125 showed broadly neutralizing activity against all variants except the Omicron sublineages, BA7208 was potently neutralizing against all tested SARS-CoV-2 variants (including Omicron BA.1–BA.5) except Mu. By combining BA7208 and BA7125 through the knobs-into-holes technology, we generated a biparatopic antibody BA7208/7125 that was able to neutralize all tested circulating SARS-CoV-2 variants. Cryo-electron microscopy structure of these broad-spectrum antibodies in complex with trimeric Delta and Omicron spike indicated that the contact residues are highly conserved and had minimal interactions with mutational residues in RBD of current variants. In addition, we showed that administration of BA7208/7125 via the intraperitoneal, intranasal, or aerosol inhalation route showed potent therapeutic efficacy against Omicron BA.1 and BA.2 in hACE2-transgenic and wild-type mice and, separately, effective prophylaxis. BA7208/7125 thus has the potential to be an effective candidate as an intervention against COVID-19.

[end paper 65]

[begin paper 66]

PMCID = 7820838

Title = Perspective on therapeutic and diagnostic potential of camel nanobodies for coronavirus disease-19 (COVID-19)

In this paper, we focus on the camelid nanobodies as a revolutionary therapy that can guide efforts to discover new drugs for Coronavirus disease (COVID-19). The small size property makes nanobodies capable of penetrating efficiently into tissues and recognizing cryptic antigens. Strong antigen affinity and stability in the gastrointestinal tract allow them to be used via oral administration. In fact, the use of nanobodies as inhalant can be directly delivered to the target organ, conferring high pulmonary drug concentrations and low systemic drug concentrations and minimal systemic side effects. For that, nanobodies are referred as a class of next-generation antibodies. Nanobodies permit the construction of multivalent formats that may achieve ultra-high neutralization potency and then may prevent mutational escape and can neutralize a wide range of SARS-CoV-2 variants. Due to their distinctive characteristics, nanobodies can be of great use in the development of promising treatment or preventive strategies against SARS-CoV-2 infection. In this review, the state-of-the-art of camel nanobodies design strategies against the virus including SARS-CoV-2 are critically summarized. The application of general nanotechnology was also discussed to mitigate and control emerging SARS-CoV-2 infection.

[end paper 66]

[begin paper 67]

PMCID = 10275483

Title = Emerging Landscape of Nanobodies and Their Neutralizing Applications against SARS-CoV-2 Virus

The new severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) that causes the coronavirus disease 2019 (COVID-19) has significantly altered people’s way of life. Despite widespread knowledge of vaccination, mask use, and avoidance of close contact, COVID-19 is still spreading around the world. Numerous research teams are examining the SARS-CoV-2 infection process to discover strategies to identify, prevent, and treat COVID-19 to limit the spread of this chronic coronavirus illness and restore lives to normalcy. Nanobodies have advantages over polyclonal and monoclonal antibodies (Ab) and Ab fragments, including reduced size, high stability, simplicity in manufacture, compatibility with genetic engineering methods, and lack of solubility and aggregation issues. Recent studies have shown that nanobodies that target the SARS-CoV-2 receptor-binding domain and disrupt ACE2 interactions are helpful in the prevention and treatment of SARS-CoV-2-infected animal models, despite the lack of evidence in human patients. The creation and evaluation of nanobodies, as well as their diagnostic and therapeutic applications against COVID-19, are discussed in this paper.

[end paper 67]

[begin paper 68]

PMCID = 10757926

Title = Two birds with one stone: human SIRPα nanobodies for functional modulation and in vivo imaging of myeloid cells

Signal-regulatory protein α (SIRPα) expressed by myeloid cells is of particular interest for therapeutic strategies targeting the interaction between SIRPα and the “don’t eat me” ligand CD47 and as a marker to monitor macrophage infiltration into tumor lesions. To address both approaches, we developed a set of novel human SIRPα (hSIRPα)–specific nanobodies (Nbs). We identified high-affinity Nbs targeting the hSIRPα/hCD47 interface, thereby enhancing antibody-dependent cellular phagocytosis. For non-invasive in vivo imaging, we chose S36 Nb as a non-modulating binder. By quantitative positron emission tomography in novel hSIRPα/hCD47 knock-in mice, we demonstrated the applicability of 64Cu-hSIRPα-S36 Nb to visualize tumor infiltration of myeloid cells. We envision that the hSIRPα-Nbs presented in this study have potential as versatile theranostic probes, including novel myeloid-specific checkpoint inhibitors for combinatorial treatment approaches and for in vivo stratification and monitoring of individual responses during cancer immunotherapies.

[end paper 68]

[begin paper 69]

PMCID = 8748511

Title = A bispecific monomeric nanobody induces spike trimer dimers and neutralizes SARS-CoV-2 in vivo

Antibodies binding to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike have therapeutic promise, but emerging variants show the potential for virus escape. This emphasizes the need for therapeutic molecules with distinct and novel neutralization mechanisms. Here we describe the isolation of a nanobody that interacts simultaneously with two RBDs from different spike trimers of SARS-CoV-2, rapidly inducing the formation of spike trimer–dimers leading to the loss of their ability to attach to the host cell receptor, ACE2. We show that this nanobody potently neutralizes SARS-CoV-2, including the beta and delta variants, and cross-neutralizes SARS-CoV. Furthermore, we demonstrate the therapeutic potential of the nanobody against SARS-CoV-2 and the beta variant in a human ACE2 transgenic mouse model. This naturally elicited bispecific monomeric nanobody establishes an uncommon strategy for potent inactivation of viral antigens and represents a promising antiviral against emerging SARS-CoV-2 variants.

Here, the authors isolate and characterize a bispecific monomeric nanobody that induces dimerization of SARS-CoV-2 spike trimers, neutralizes variants of concerns as well as SARS-CoV, and inhibits SARS-CoV-2 infection in mice.

[end paper 69]

[begin paper 70]

PMCID = 11324105

Title = Structure defining of ultrapotent neutralizing nanobodies against MERS-CoV with novel epitopes on receptor binding domain

The Middle East Respiratory Syndrome Coronavirus (MERS-CoV) causes severe and fatal acute respiratory disease in humans. High fatality rates and continued infectiousness remain a pressing concern for global health preparedness. Antibodies targeted at the receptor-binding domain (RBD) are major countermeasures against human viral infection. Here, we report four potent nanobodies against MERS-CoV, which are isolated from alpaca, and especially the potency of Nb14 is highest in the pseudotyped virus assay. Structural studies show that Nb14 framework regions (FRs) are mainly involved in interactions targeting a novel epitope, which is entirely distinct from all previously reported antibodies, and disrupt the protein-carbohydrate interaction between residue W535 of RBD and hDPP4 N229-linked carbohydrate moiety (hDPP4-N229-glycan). Different from Nb14, Nb9 targets the cryptic face of RBD, which is distinctive from the hDPP4 binding site and the Nb14 epitope, and it induces the β5-β6 loop to inflect towards a shallow groove of the RBD and dampens the accommodation of a short helix of hDPP4. The particularly striking epitopes endow the two Nbs administrate synergistically in the pseudotyped MERS-CoV assays. These results not only character unprecedented epitopes for antibody recognition but also provide promising agents for prophylaxis and therapy of MERS-CoV infection.

MERS-CoV is one of the most prevalent zoonotic virus that has spread through 27 countries and infected more than 2,605 people since its first outbreak in Saudi Arabia in 2012. The high fatality rate and its persistent wide spread infectiousness in animal reservoirs have generated tremendous global public health concern. However, no licensed therapeutic agents or vaccines against MERS-CoV are currently available. To address the risk of MERS-CoV and its variants re-emergence, we developed four highly effective Nbs from alpaca and solved two crystal complexes of Nb9 and Nb14 with RBD. Remarkably, the neutralizing activity of Nb14 represents the highest of MERS-CoV antibodies reported to date. The novel epitopes of Nb14 and Nb9 locate outside the RBD and hDPP4 interface, and the unique different epitopes and mechanisms of Nb14 and Nb9 provide a better neutralizing for synergistically against MERS-CoV infection. Our findings provide insights into the cryptic epitopes on RBD and the development of antibodies against MERS-CoV infection.

[end paper 70]

[begin paper 71]

PMCID = 9990171

Title = Avidity engineering of human heavy-chain-only antibodies mitigates neutralization resistance of SARS-CoV-2 variants

Emerging SARS-CoV-2 variants have accrued mutations within the spike protein rendering most therapeutic monoclonal antibodies against COVID-19 ineffective. Hence there is an unmet need for broad-spectrum mAb treatments for COVID-19 that are more resistant to antigenically drifted SARS-CoV-2 variants. Here we describe the design of a biparatopic heavy-chain-only antibody consisting of six antigen binding sites recognizing two distinct epitopes in the spike protein NTD and RBD. The hexavalent antibody showed potent neutralizing activity against SARS-CoV-2 and variants of concern, including the Omicron sub-lineages BA.1, BA.2, BA.4 and BA.5, whereas the parental components had lost Omicron neutralization potency. We demonstrate that the tethered design mitigates the substantial decrease in spike trimer affinity seen for escape mutations for the hexamer components. The hexavalent antibody protected against SARS-CoV-2 infection in a hamster model. This work provides a framework for designing therapeutic antibodies to overcome antibody neutralization escape of emerging SARS-CoV-2 variants.

[end paper 71]

[begin paper 72]

PMCID = 10154539

Title = Fortuitous somatic mutations during antibody evolution endow broad neutralization against SARS-CoV-2 Omicron variants

Striking antibody evasion by emerging circulating severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants drives the identification of broadly neutralizing antibodies (bNAbs). However, how a bNAb acquires increased neutralization breadth during antibody evolution is still elusive. Here, we identify a clonally related antibody family from a convalescent individual. One of the members, XG005, exhibits potent and broad neutralizing activities against SARS-CoV-2 variants, while the other members show significant reductions in neutralization breadth and potency, especially against the Omicron sublineages. Structural analysis visualizing the XG005-Omicron spike binding interface reveals how crucial somatic mutations endow XG005 with greater neutralization potency and breadth. A single administration of XG005 with extended half-life, reduced antibody-dependent enhancement (ADE) effect, and increased antibody product quality exhibits a high therapeutic efficacy in BA.2- and BA.5-challenged mice. Our results provide a natural example to show the importance of somatic hypermutation during antibody evolution for SARS-CoV-2 neutralization breadth and potency.

Wu et al. identify that XG005, but not its family members, is a potent and broad SARS-CoV-2 neutralizer and exhibits a high therapeutic efficacy. Structural analysis identifies the key residues of XG005. This provides a natural example to show the importance of somatic hypermutation for SARS-CoV-2 neutralization breadth and potency.

[end paper 72]

[begin paper 73]

PMCID = 11277385

Title = Broad sarbecovirus neutralization by combined memory B cell antibodies to ancestral SARS-CoV-2

Antibodies play a pivotal role in protecting from SARS-CoV-2 infection, but their efficacy is challenged by the continuous emergence of viral variants. In this study, we describe two broadly neutralizing antibodies cloned from the memory B cells of a single convalescent individual after infection with ancestral SARS-CoV-2. Cv2.3194, a resilient class 1 anti-RBD antibody, remains active against Omicron sub-variants up to BA.2.86. Cv2.3132, a near pan-Sarbecovirus neutralizer, targets the heptad repeat 2 membrane proximal region. When combined, Cv2.3194 and Cv2.3132 form a complementary SARS-CoV-2 neutralizing antibody cocktail exhibiting a local dose-dependent synergy. Thus, remarkably robust neutralizing memory B cell antibodies elicited in response to ancestral SARS-CoV-2 infection can withstand viral evolution and immune escape. The cooperative effect of such antibody combination may confer a certain level of protection against the latest SARS-CoV-2 variants.

Cv2.3194 is a resilient class 1 RBD SARS-CoV-2 neutralizer active up to BA.2.86

Cv2.3132, a broadly sarbecovirus neutralizer, targets the HR2 membrane proximal region

Combined Cv2.3132 and Cv2.3194 cooperate to neutralize SARS-CoV-2 variants in vitro

Cv2.3132 and Cv2.3194 provide protection against SARS-CoV-2 infection in mouse models

Immunology; Virology

[end paper 73]

[begin paper 74]

PMCID = 8354634

Title = The development of Nanosota-1 as anti-SARS-CoV-2 nanobody drug candidates

Combating the COVID-19 pandemic requires potent and low-cost therapeutics. We identified a series of single-domain antibodies (i.e., nanobody), Nanosota-1, from a camelid nanobody phage display library. Structural data showed that Nanosota-1 bound to the oft-hidden receptor-binding domain (RBD) of SARS-CoV-2 spike protein, blocking viral receptor angiotensin-converting enzyme 2 (ACE2). The lead drug candidate possessing an Fc tag (Nanosota-1C-Fc) bound to SARS-CoV-2 RBD ~3000 times more tightly than ACE2 did and inhibited SARS-CoV-2 pseudovirus ~160 times more efficiently than ACE2 did. Administered at a single dose, Nanosota-1C-Fc demonstrated preventive and therapeutic efficacy against live SARS-CoV-2 infection in both hamster and mouse models. Unlike conventional antibodies, Nanosota-1C-Fc was produced at high yields in bacteria and had exceptional thermostability. Pharmacokinetic analysis of Nanosota-1C-Fc documented an excellent in vivo stability and a high tissue bioavailability. As effective and inexpensive drug candidates, Nanosota-1 may contribute to the battle against COVID-19.

[end paper 74]

[begin paper 75]

PMCID = 8967028

Title = Structure-based dual affinity optimization of a SARS-CoV-1/2 cross-reactive single-domain antibody

The SARS coronavirus 2 (SARS-CoV-2) spike (S) protein binding to the human ACE2 receptor is the molecular event that initiates viral entry into host cells and leads to infection and virus replication. There is a need for agents blocking viral entry into host cells that are cross-reactive with emerging virus variants. VHH-72 is an anti-SARS-CoV-1 single-domain antibody that also exhibits cross-specificity with SARS-CoV-2 but with decreased binding affinity. Here we applied a structure-based approach to affinity-mature VHH-72 for the SARS-CoV-2 spike protein while retaining the original affinity for SARS-CoV-1. This was achieved by employing the computational platform ADAPT in a constrained dual-affinity optimization mode as a means of broadening specificity. Select mutants designed by ADAPT were formatted as fusions with a human IgG1-Fc fragment. These mutants demonstrated improved binding to the SARS-CoV-2 spike protein due to decreased dissociation rates. Functional testing for virus neutralization revealed improvements relative to the parental VHH72-Fc up to 10-fold using a SARS-CoV-2 pseudotyped lentivirus and 20-fold against the SARS-CoV-2 authentic live virus (Wuhan variant). Binding and neutralization improvements were maintained for some other SARS-CoV-2 variants currently in circulation. These improved VHH-72 mutants are predicted to establish novel interactions with the S antigen. They will be useful, alone or as fusions with other functional modules, in the global quest for treatments of COVID-19 infections.

[end paper 75]

[begin paper 76]

PMCID = 11200683

Title = A VHH single-domain platform enabling discovery and development of monospecific antibodies and modular neutralizing bispecifics against SARS-CoV-2 variants

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) continues to evolve, escape coronavirus disease 2019 therapeutics and vaccines, and jeopardize public health. To combat SARS-CoV-2 antigenic escape, we developed a rapid, high-throughput pipeline to discover monospecific VHH antibodies and iteratively develop VHH-Fc-VHH bispecifics capable of neutralizing emerging SARS-CoV-2 variants. By panning VHH single-domain phage libraries against ancestral or beta spike proteins, we discovered high-affinity VHH antibodies with unique target epitopes. Combining two VHHs into a tetravalent bispecific construct conferred broad neutralization activity against multiple variants and was more resistant to antigenic escape than the monospecific antibody alone. Following the rise of the Omicron variant, a VHH in the original bispecific construct was replaced with another VHH discovered against the Omicron BA.1 receptor binding domain; the resulting bispecific exhibited neutralization against both BA.1 and BA.5 sublineage variants. A heavy chain-only tetravalent VHH-Fc-VHH bispecific platform derived from humanized synthetic libraries held a myriad of unique advantages: (i) synthetic preconstructed libraries minimized risk of liabilities and maximized discovery speed, (ii) VHH scaffolds allowed for a modular “plug-and-play” format that could be rapidly iterated upon as variants of concern arose, (iii) natural dimerization of single VHH-Fc-VHH polypeptides allowed for straightforward bispecific production and purification methods, and (iv) multivalent approaches enhanced avidity boosting effects and neutralization potency, and conferred more robust resistance to antigenic escape than monovalent approaches against specific variants. This iterative platform of rapid VHH discovery combined with modular bispecific design holds promise for long-term viral control efforts.

Statement of Significance: A modular antibody platform leveraging VHH heavy-chain constructs derived from humanized synthetic libraries to rapidly generate tetravalent VHH-Fc-VHH bispecifics was developed to combat viruses like SARS-CoV-2 that exhibit rapid antigenic drift. This platform was utilized to discover new antibodies and iteratively develop neutralizing bispecifics as new SARS-CoV-2 variants arose.

[end paper 76]

[begin paper 77]

PMCID = 10051120

Title = Biomimetic Nanotechnology for SARS-CoV-2 Treatment

More than 600 million people worldwide have been infected with severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), resulting in the pandemic of coronavirus disease 2019 (COVID-19). In particular, new waves of COVID-19 caused by emerging SARS-CoV-2 variants pose new health risks to the global population. Nanotechnology has developed excellent solutions to combat the virus pandemic, such as ACE2-based nanodecoys, nanobodies, nanovaccines, and drug nanocarriers. Lessons learned and strategies developed during this battle against SARS-CoV-2 variants may also serve as inspiration for developing nanotechnology-based strategies to combat other global infectious diseases and their variants in the future.

[end paper 77]

[begin paper 78]

PMCID = 8194341

Title = Prospects of Neutralizing Nanobodies Against SARS-CoV-2

Since December 2019, the SARS-CoV-2 has erupted on a large scale worldwide and spread rapidly. Passive immunization of antibody-related molecules provides opportunities for prevention and treatment of high-risk patients and children. Nanobodies (Nbs) have many strong physical and chemical properties. They can be atomized, administered by inhalation, and can be directly applied to the infected site, with fast onset, high local drug concentration/high bioavailability, and high patient compliance (no needles). It has very attractive potential in the treatment of respiratory viruses. Rapid and low-cost development of Nbs targeting SARS-CoV-2 can quickly be achieved. Nbs against SARS-CoV-2 mutant strains also can be utilized quickly to prevent the virus from escaping. It provides important technical supports for the treatment of the SARS-CoV-2 and has the potential to become an essential medicine in the toolbox against the SARS-CoV-2.

[end paper 78]

[begin paper 79]

PMCID = 9778468

Title = Plasmodium infection is associated with cross-reactive antibodies to carbohydrate epitopes on the SARS-CoV-2 Spike protein

Sero-surveillance can monitor and project disease burden and risk. However, SARS-CoV-2 antibody test results can produce false positive results, limiting their efficacy as a sero-surveillance tool. False positive SARS-CoV-2 antibody results are associated with malaria exposure, and understanding this association is essential to interpret sero-surveillance results from malaria-endemic countries. Here, pre-pandemic samples from eight malaria endemic and non-endemic countries and four continents were tested by ELISA to measure SARS-CoV-2 Spike S1 subunit reactivity. Individuals with acute malaria infection generated substantial SARS-CoV-2 reactivity. Cross-reactivity was not associated with reactivity to other human coronaviruses or other SARS-CoV-2 proteins, as measured by peptide and protein arrays. ELISAs with deglycosylated and desialated Spike S1 subunits revealed that cross-reactive antibodies target sialic acid on N-linked glycans of the Spike protein. The functional activity of cross-reactive antibodies measured by neutralization assays showed that cross-reactive antibodies did not neutralize SARS-CoV-2 in vitro. Since routine use of glycosylated or sialated assays could result in false positive SARS-CoV-2 antibody results in malaria endemic regions, which could overestimate exposure and population-level immunity, we explored methods to increase specificity by reducing cross-reactivity. Overestimating population-level exposure to SARS-CoV-2 could lead to underestimates of risk of continued COVID-19 transmission in sub-Saharan Africa.

[end paper 79]

[begin paper 80]

PMCID = 8677774

Title = Mechanisms of SARS-CoV-2 neutralization by shark variable new antigen receptors elucidated through X-ray crystallography

Single-domain Variable New Antigen Receptors (VNARs) from the immune system of sharks are the smallest naturally occurring binding domains found in nature. Possessing flexible paratopes that can recognize protein motifs inaccessible to classical antibodies, VNARs have yet to be exploited for the development of SARS-CoV-2 therapeutics. Here, we detail the identification of a series of VNARs from a VNAR phage display library screened against the SARS-CoV-2 receptor binding domain (RBD). The ability of the VNARs to neutralize pseudotype and authentic live SARS-CoV-2 virus rivalled or exceeded that of full-length immunoglobulins and other single-domain antibodies. Crystallographic analysis of two VNARs found that they recognized separate epitopes on the RBD and had distinctly different mechanisms of virus neutralization unique to VNARs. Structural and biochemical data suggest that VNARs would be effective therapeutic agents against emerging SARS-CoV-2 mutants, including the Delta variant, and coronaviruses across multiple phylogenetic lineages. This study highlights the utility of VNARs as effective therapeutics against coronaviruses and may serve as a critical milestone for nearing a paradigm shift of the greater biologic landscape.

Shark antibodies (Variable New Antigen Receptors, VNARs) are the smallest naturally occurring antibody fragments. Here, the authors screen a VNAR phage display library against the SARS-CoV2 receptor binding domain (RBD) and identify VNARs that neutralize the SARSCoV-2 virus and discuss their mechanisms of viral neutralization.

[end paper 80]

[begin paper 81]

PMCID = 10518403

Title = Unleashing the power of shark variable single domains (VNARs): broadly neutralizing tools for combating SARS-CoV-2

The pandemic caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) generated a joint global effort to develop vaccines and other treatments that could mitigate the negative effects and the rapid spread of the virus. Single-domain antibodies derived from various sources, including cartilaginous fish, camelids, and humans, have gained attention as promising therapeutic tools against coronavirus disease 2019. Shark-derived variable new antigen receptors (VNARs) have emerged as the smallest naturally occurring antigen-binding molecules. Here, we compile and review recent published studies on VNARs with the capacity to recognize and/or neutralize SARS-CoV-2. We found a close balance between the use of natural immune libraries and synthetic VNAR libraries for the screening against SARS-CoV-2, with phage display being the preferred display technology for the selection of VNARs against this virus. In addition, we discuss potential modifications and engineering strategies employed to improve the neutralization potential of VNARs, such as exploring fusion with the Fc domain of human Immunoglobulin G (IgG) to increase avidity and therapeutic potential. This research highlights the potential of VNARs as powerful molecular tools in the fight against infectious diseases.

[end paper 81]

[begin paper 82]

PMCID = 8537675

Title = Emerging SARS-CoV-2 Variants: A Review of Its Mutations, Its Implications and Vaccine Efficacy

The widespread increase in multiple severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) variants is causing a significant health concern in the United States and worldwide. These variants exhibit increased transmissibility, cause more severe disease, exhibit evasive immune properties, impair neutralization by antibodies from vaccinated individuals or convalescence sera, and reinfection. The Centers for Disease Control and Prevention (CDC) has classified SARS-CoV-2 variants into variants of interest, variants of concern, and variants of high consequence. Currently, four variants of concern (B.1.1.7, B.1.351, P.1, and B.1.617.2) and several variants of interests (B.1.526, B.1.525, and P.2) are characterized and are essential for close monitoring. In this review, we discuss the different SARS-CoV-2 variants, emphasizing variants of concern circulating the world and highlight the various mutations and how these mutations affect the characteristics of the virus. In addition, we discuss the most common vaccines and the various studies concerning the efficacy of these vaccines against different variants of concern.

[end paper 82]

[begin paper 83]

PMCID = 11099193

Title = Extracellular modulation of TREK-2 activity with nanobodies provides insight into the mechanisms of K2P channel regulation

Potassium channels of the Two-Pore Domain (K2P) subfamily, KCNK1-KCNK18, play crucial roles in controlling the electrical activity of many different cell types and represent attractive therapeutic targets. However, the identification of highly selective small molecule drugs against these channels has been challenging due to the high degree of structural and functional conservation that exists not only between K2P channels, but across the whole K+ channel superfamily. To address the issue of selectivity, here we generate camelid antibody fragments (nanobodies) against the TREK-2 (KCNK10) K2P K+ channel and identify selective binders including several that directly modulate channel activity. X-ray crystallography and CryoEM data of these nanobodies in complex with TREK-2 also reveal insights into their mechanisms of activation and inhibition via binding to the extracellular loops and Cap domain, as well as their suitability for immunodetection. These structures facilitate design of a biparatropic inhibitory nanobody with markedly improved sensitivity. Together, these results provide important insights into TREK channel gating and provide an alternative, more selective approach to modulation of K2P channel activity via their extracellular domains.

K2P channels are important regulators of cellular electrical activity. Here the authors show how nanobody fragments can be used to detect and modulate TREK2 K2P channel activity to provide insight into the mechanism of gating.

[end paper 83]

[begin paper 84]

PMCID = 10532743

Title = Shark IgNAR: The Next Broad Application Antibody in Clinical Diagnoses and Tumor Therapies?

Antibodies represent a relatively mature detection means and serve as therapeutic drug carriers in the clinical diagnosis and treatment of cancer—among which monoclonal antibodies (mAbs) currently occupy a dominant position. However, the emergence and development of small-molecule monodomain antibodies are inevitable due to the many limitations of mAbs, such as their large size, complex structure, and sensitivity to extreme temperature, and tumor microenvironments. Thus, since first discovered in Chondroid fish in 1995, IgNAR has become an alternative therapeutic strategy through which to replace monoclonal antibodies, thus entailing that this novel type of immunoglobulin has received wide attention with respect to clinical diagnoses and tumor therapies. The variable new antigen receptor (VNAR) of IgNAR provides an advantage for the development of new antitumor drugs due to its small size, high stability, high affinity, as well as other structural and functional characteristics. In that respect, a better understanding of the unique characteristics and therapeutic potential of IgNAR/VNAR in clinical and anti-tumor treatment is needed. This article reviews the advantages of its unique biochemical conditions and molecular structure for clinical diagnoses and novel anti-tumor drugs. At the same time, the main advantages of the existing conjugated drugs, which are based on single-domain antibodies, are introduced here, thereby providing new ideas and methods for the development of clinical diagnoses and anti-tumor therapies in the future.

[end paper 84]

[begin paper 85]

PMCID = 10386765

Title = Cross protection to SARS-CoV-2 variants in hamsters with naturally-acquired immunity

Since SARS-CoV-2 was first reported in late 2019, multiple variations of the original virus have emerged. Each variant harbors accumulations of mutations, particularly within the spike glycoprotein, that are associated with increased viral transmissibility and escape immunity. The different mutations in the spike protein of different variants shape the subsequent antibody and T cell responses, such that exposure to different spike proteins can result in reduced or enhanced responses to heterologous variants further down the line. Globally, people have been exposed and re-exposed to multiple variations of the Ancestral strain, including the five variants of concerns. Studies have shown that the protective immune response of an individual is influenced by which strain or combination of strains they are exposed to. The initial exposure to a specific strain may also shape their subsequent immune patterns and response to later infections with a heterologous virus. Most immunological observations were carried out early during the pandemic when the Ancestral strain was circulating. However, SARS-CoV-2 variants exhibit varying patterns of disease severity, waning immunity, immune evasion and sensitivity to therapeutics. Here we investigated the cross-protection in hamsters previously infected with a variant of concern (VOC) and subsequently re-infected with a heterologous variant. We also determined if cross-protection and immunity were dependent on the specific virus to which the hamster was first exposed. We further profiled the host cytokine response induced by each SARS-CoV-2 variants as well as subsequent to re-infection. A comparative analysis of the three VOCs revealed that Alpha variant was the most pathogenic VOC to emerge. We showed that naturally acquired immunity protected hamsters from subsequent re-infection with heterologous SARS-CoV-2 variant, regardless which variant the animal was first exposed to. Our study supports observations that heterologous infection of different SARS-CoV-2 variants do not exacerbate disease in subsequent re-infections. The continual emergence of new SARS-CoV-2 variants mandates a better understanding of cross-protection and immune imprinting in infected individuals. Such information is essential to guide vaccine strategy and public policy to emerging SARS-CoV-2 VOCs and future novel pandemic coronaviruses.

[end paper 85]

[begin paper 86]

PMCID = 9181070

Title = Robust anti-SARS-CoV2 single domain antibodies cross neutralize multiple viruses

We report robust SARS-CoV2 neutralizing sdAbs targeting the viral peptides encompassing the polybasic cleavage site (CSP) and in the receptor binding domain (RBD) of the spike (S) protein. Both the sdAbs inhibited infectivity of the CoV2 S protein expressing pseudoviruses (LV-CoV2S). Both anti-CSP and RBD intrabodies (IB) inhibited the output of LV(CoV2 S). Anti-CSP IB altered the proteolytic processing and targeted the viral S protein for degradation. Because of cross-reactive CSPs in the entry mediators, the anti-CSP sdAb neutralized in vitro and in vivo the infectivity of SARS-CoV2 unrelated viruses such as herpes simplex virus 1 (HSV1) and pestes des petits ruminants virus (PPRV). Conversely, anti-HSV1 and anti-PPRV sera neutralized LV(CoV2 S) owing to the presence of CSP reactive antibodies indicating that a prior infection with such pathogens could impact on the pattern of COVID-19.

Anti-CSP and anti-RBD sdAb efficiently neutralized LV (SARS-CoV2 S) pseudoviruses

Both intrabodies prevented the biogenesis of SARS-CoV2-S pseudovirions

Anti-CSP sdAb neutralized viruses such as HSV1 and PPRV

Anti-HSV1 and anti-PPRV sera neutralized LV(CoV2 S) pseudoviruses

Virology, Immunology, and Drugs,

[end paper 86]

[begin paper 87]

PMCID = 9201380

Title = Structural Characterization of a Neutralizing Nanobody With Broad Activity Against SARS-CoV-2 Variants

SARS-CoV-2 and its variants, such as the Omicron continue to threaten public health. The virus recognizes the host cell by attaching its Spike (S) receptor-binding domain (RBD) to the host receptor, ACE2. Therefore, RBD is a primary target for neutralizing antibodies and vaccines. Here, we report the isolation and biological and structural characterization of a single-chain antibody (nanobody) from RBD-immunized alpaca. The nanobody, named DL28, binds to RBD tightly with a KD of 1.56 nM and neutralizes the original SARS-CoV-2 strain with an IC50 of 0.41 μg mL−1. Neutralization assays with a panel of variants of concern (VOCs) reveal its wide-spectrum activity with IC50 values ranging from 0.35 to 1.66 μg mL−1 for the Alpha/Beta/Gamma/Delta and an IC50 of 0.66 μg mL−1 for the currently prevalent Omicron. Competition binding assays show that DL28 blocks ACE2-binding. However, structural characterizations and mutagenesis suggest that unlike most antibodies, the blockage by DL28 does not involve direct competition or steric hindrance. Rather, DL28 may use a “conformation competition” mechanism where it excludes ACE2 by keeping an RBD loop in a conformation incompatible with ACE2-binding.

[end paper 87]

[begin paper 88]

PMCID = 8837488

Title = Human inhalable antibody fragments neutralizing SARS-CoV-2 variants for COVID-19 therapy

As of December 2021, coronavirus disease 2019 (COVID-19), caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), remains a global emergency, and novel therapeutics are urgently needed. Here we describe human single-chain variable fragment (scFv) antibodies (76clAbs) that block an epitope of the SARS-CoV-2 spike protein essential for ACE2-mediated entry into cells. 76clAbs neutralize the Delta variant and other variants being monitored (VBMs) and inhibit spike-mediated pulmonary cell-cell fusion, a critical feature of COVID-19 pathology. In two independent animal models, intranasal administration counteracted the infection. Because of their high efficiency, remarkable stability, resilience to nebulization, and low cost of production, 76clAbs may become a relevant tool for rapid, self-administrable early intervention in SARS-CoV-2-infected subjects independently of their immune status.

De Santis and colleagues describe engineered human antibody fragments (scFvs), which are extremely effective at neutralizing SARS-CoV-2 variants. Because of their high stability and efficacy in preclinical models, intranasal or aerosol delivery of scFv antibodies represents a promising approach for halting SARS-CoV-2 infection at an early stage regardless of vaccination status.

[end paper 88]

[begin paper 89]

PMCID = 8526496

Title = Development of a highly specific and sensitive VHH-based sandwich immunoassay for the detection of the SARS-CoV-2 nucleoprotein

The current COVID-19 pandemic illustrates the importance of obtaining reliable methods for the rapid detection of SARS-CoV-2. A highly specific and sensitive diagnostic test able to differentiate the SARS-CoV-2 virus from common human coronaviruses is therefore needed. Coronavirus nucleoprotein (N) localizes to the cytoplasm and the nucleolus and is required for viral RNA synthesis. N is the most abundant coronavirus protein, so it is of utmost importance to develop specific antibodies for its detection. In this study, we developed a sandwich immunoassay to recognize the SARS-CoV-2 N protein. We immunized one alpaca with recombinant SARS-CoV-2 N and constructed a large single variable domain on heavy chain (VHH) antibody library. After phage display selection, seven VHHs recognizing the full N protein were identified by ELISA. These VHHs did not recognize the nucleoproteins of the four common human coronaviruses. Hydrogen Deuterium eXchange–Mass Spectrometry (HDX-MS) analysis also showed that these VHHs mainly targeted conformational epitopes in either the C-terminal or the N-terminal domains. All VHHs were able to recognize SARS-CoV-2 in infected cells or on infected hamster tissues. Moreover, the VHHs could detect the SARS variants B.1.17/alpha, B.1.351/beta, and P1/gamma. We propose that this sandwich immunoassay could be applied to specifically detect the SARS-CoV-2 N in human nasal swabs.

[end paper 89]

[begin paper 90]

PMCID = 10302031

Title = Status and Developing Strategies for Neutralizing Monoclonal Antibody Therapy in the Omicron Era of COVID-19

The monoclonal antibody (mAb)-based treatment is a highly valued therapy against COVID-19, especially for individuals who may not have strong immune responses to the vaccine. However, with the arrival of the Omicron variant and its evolving subvariants, along with the occurrence of remarkable resistance of these SARS-CoV-2 variants to the neutralizing antibodies, mAbs are facing tough challenges. Future strategies for developing mAbs with improved resistance to viral evasion will involve optimizing the targeting epitopes on SARS-CoV-2, enhancing the affinity and potency of mAbs, exploring the use of non-neutralizing antibodies that bind to conserved epitopes on the S protein, as well as optimizing immunization regimens. These approaches can improve the viability of mAb therapy in the fight against the evolving threat of the coronavirus.

[end paper 90]

[begin paper 91]

PMCID = 9774204

Title = Fortuitous Somatic Mutations during Antibody Evolution Endow Broad Neutralization against SARS-CoV-2 Omicron Variants

Striking antibody evasion by emerging circulating SARS-CoV-2 variants drives the identification of broadly neutralizing antibodies (bNAbs). However, how a bNAb acquires increased neutralization breadth during antibody evolution is still elusive. Here, we identified a clonally-related antibody family from a convalescent individual. One of the members, XG005, exhibited potent and broad neutralizing activities against SARS-CoV-2 variants, while the other members showed significant reductions in neutralization breadth and potency, especially against the Omicron sublineages. Structural analysis visualizing the XG005-Omicron spike binding interface revealed how crucial somatic mutations endowed XG005 with greater neutralization potency and breadth. A single administration of XG005 with extended half-life, reduced antibody-dependent enhancement (ADE) effect, and increased antibody product quality, exhibited a high therapeutic efficacy in BA.2- and BA.5-challenged mice. Our results provided a natural example to show the importance of somatic hypermutation during antibody evolution for SARS-CoV-2 neutralization breadth and potency.

[end paper 91]

[begin paper 92]

PMCID = 8564274

Title = A non-ACE2 competing human single-domain antibody confers broad neutralization against SARS-CoV-2 and circulating variants

The current COVID-19 pandemic has heavily burdened the global public health system and may keep simmering for years. The frequent emergence of immune escape variants have spurred the search for prophylactic vaccines and therapeutic antibodies that confer broad protection against SARS-CoV-2 variants. Here we show that the bivalency of an affinity maturated fully human single-domain antibody (n3113.1-Fc) exhibits exquisite neutralizing potency against SARS-CoV-2 pseudovirus, and confers effective prophylactic and therapeutic protection against authentic SARS-CoV-2 in the host cell receptor angiotensin-converting enzyme 2 (ACE2) humanized mice. The crystal structure of n3113 in complex with the receptor-binding domain (RBD) of SARS-CoV-2, combined with the cryo-EM structures of n3113 and spike ecto-domain, reveals that n3113 binds to the side surface of up-state RBD with no competition with ACE2. The binding of n3113 to this novel epitope stabilizes spike in up-state conformations but inhibits SARS-CoV-2 S mediated membrane fusion, expanding our recognition of neutralization by antibodies against SARS-CoV-2. Binding assay and pseudovirus neutralization assay show no evasion of recently prevalent SARS-CoV-2 lineages, including Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), and Delta (B.1.617.2) for n3113.1-Fc with Y58L mutation, demonstrating the potential of n3113.1-Fc (Y58L) as a promising candidate for clinical development to treat COVID-19.

[end paper 92]

[begin paper 93]

PMCID = 8700021

Title = Epitope profiling using computational structural modelling demonstrated on coronavirus-binding antibodies

Identifying the epitope of an antibody is a key step in understanding its function and its potential as a therapeutic. Sequence-based clonal clustering can identify antibodies with similar epitope complementarity, however, antibodies from markedly different lineages but with similar structures can engage the same epitope. We describe a novel computational method for epitope profiling based on structural modelling and clustering. Using the method, we demonstrate that sequence dissimilar but functionally similar antibodies can be found across the Coronavirus Antibody Database, with high accuracy (92% of antibodies in multiple-occupancy structural clusters bind to consistent domains). Our approach functionally links antibodies with distinct genetic lineages, species origins, and coronavirus specificities. This indicates greater convergence exists in the immune responses to coronaviruses than is suggested by sequence-based approaches. Our results show that applying structural analytics to large class-specific antibody databases will enable high confidence structure-function relationships to be drawn, yielding new opportunities to identify functional convergence hitherto missed by sequence-only analysis.

Antibodies are a key component of the immune system that combat pathogens by binding to a defined region of their molecular surface (known as an ‘epitope’). The ability to map which antibodies target the same epitopes is crucial when designing non-competing antibody therapeutics or predicting the influence of pathogen mutation on population immunity. While one can use laboratory experiments to deduce when pairs of antibodies engage the same epitope, such experiments are very expensive and time consuming if used to compare on the order of thousands of antibodies. In this work, we report a new computational algorithm (SPACE) that clusters antibodies that target the same epitope based on their predicted 3D structure, as binding site structure is a property often conserved between binders complementary to the same epitope. Unlike existing antibody epitope profiling tools which assume two antibodies must share a high sequence identity/similar genetic basis to engage the same region, our orthogonal method can detect broader patterns of convergent evolution across binders to different pathogen strains, and between antibodies with different genetic and even species origins.

[end paper 93]

[begin paper 94]

PMCID = 8899823

Title = A broadly cross-reactive antibody neutralizes and protects against sarbecovirus challenge in mice

Severe acute respiratory syndrome coronaviruses 1 (SARS-CoV) and 2 (SARS-CoV-2), including SARS-CoV-2 variants of concern, can cause deadly infections. The mortality associated with sarbecovirus infection underscores the importance of developing broadly effective countermeasures against them, which could be key in the prevention and mitigation of current and future zoonotic events. Here, we demonstrate the neutralization of SARS-CoV, bat coronaviruses WIV-1, RsSHC014, and SARS-CoV-2 variants D614G, B.1.1.7, B.1.351, P.1, B.1.429, B.1.526, B.1.617.1, and B.1.617.2 by a receptor-binding domain (RBD)-specific human antibody, DH1047. Prophylactic and therapeutic treatment with DH1047 was protective against SARS-CoV, WIV-1, RsSHC014, and SARS-CoV-2 B.1.351 infection in mice. Binding and structural analysis showed high affinity binding of DH1047 to an epitope that is highly conserved among sarbecoviruses. Thus, DH1047 is a broadly protective antibody that can prevent infection and mitigate outbreaks caused by SARS-related strains and SARS-CoV-2 variants. Our results also suggest that the conserved RBD epitope bound by DH1047 is a rational target for a universal sarbecovirus vaccine.

DH1047 broadly protects in vivo against pre-emergent bat SARS-related viruses, SARS-CoV, and SARS-CoV-2 variants.

[end paper 94]

[begin paper 95]

PMCID = 11100608

Title = Bispecific antibodies with broad neutralization potency against SARS-CoV-2 variants of concern

The ongoing emergence of SARS-CoV-2 variants of concern (VOCs) that reduce the effectiveness of antibody therapeutics necessitates development of next-generation antibody modalities that are resilient to viral evolution. Here, we characterized N-terminal domain (NTD) and receptor binding domain (RBD)-specific monoclonal antibodies previously isolated from COVID-19 convalescent donors for their activity against emergent SARS-CoV-2 VOCs. Among these, the NTD-specific antibody C1596 displayed the greatest breadth of binding to VOCs, with cryo-EM structural analysis revealing recognition of a distinct NTD epitope outside of the site i antigenic supersite. Given C1596’s favorable binding profile, we designed a series of bispecific antibodies (bsAbs) termed CoV2-biRNs, that featured both NTD and RBD specificities. Notably, two of the C1596-inclusive bsAbs, CoV2-biRN5 and CoV2-biRN7, retained potent in vitro neutralization activity against all Omicron variants tested, including XBB.1.5, EG.5.1, and BA.2.86, contrasting the diminished potency of parental antibodies delivered as monotherapies or as a cocktail. Furthermore, prophylactic delivery of CoV2-biRN5 significantly reduced the viral load within the lungs of K18-hACE2 mice following challenge with SARS-CoV-2 XBB.1.5. In conclusion, our NTD-RBD bsAbs offer promising potential for the design of resilient, next-generation antibody therapeutics against SARS-CoV-2 VOCs.

Bispecific antibodies with a highly cross-reactive NTD antibody demonstrate resilience to SARS-CoV-2 variants of concern.

[end paper 95]

[begin paper 96]

PMCID = 10170451

Title = Targetable elements in SARS-CoV-2 S2 subunit for the design of pan-coronavirus fusion inhibitors and vaccines

The ongoing global pandemic of coronavirus disease 2019 (COVID-19), caused by severe acute respiratory syndrome coronavirus 2 (SARS‐CoV‐2), has caused devastating impacts on the public health and the global economy. Rapid viral antigenic evolution has led to the continual generation of new variants. Of special note is the recently expanding Omicron subvariants that are capable of immune evasion from most of the existing neutralizing antibodies (nAbs). This has posed new challenges for the prevention and treatment of COVID-19. Therefore, exploring broad-spectrum antiviral agents to combat the emerging variants is imperative. In sharp contrast to the massive accumulation of mutations within the SARS-CoV-2 receptor-binding domain (RBD), the S2 fusion subunit has remained highly conserved among variants. Hence, S2-based therapeutics may provide effective cross-protection against new SARS-CoV-2 variants. Here, we summarize the most recently developed broad-spectrum fusion inhibitors (e.g., nAbs, peptides, proteins, and small-molecule compounds) and candidate vaccines targeting the conserved elements in SARS-CoV-2 S2 subunit. The main focus includes all the targetable S2 elements, namely, the fusion peptide, stem helix, and heptad repeats 1 and 2 (HR1-HR2) bundle. Moreover, we provide a detailed summary of the characteristics and action-mechanisms for each class of cross-reactive fusion inhibitors, which should guide and promote future design of S2-based inhibitors and vaccines against new coronaviruses.

[end paper 96]

[begin paper 97]

PMCID = 8890389

Title = Therapeutic antibodies for COVID-19: is a new age of IgM, IgA and bispecific antibodies coming?

Early humoral immune responses to severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) are dominated by IgM and IgA antibodies, which greatly contribute to virus neutralization at mucosal sites. Given the essential roles of IgM and IgA in the control and elimination of SARS-CoV-2 infection, the mucosal immunity could be exploited for therapeutic and prophylactic purposes. However, almost all neutralizing antibodies that are authorized for emergency use and under clinical development are IgG antibodies, and no vaccine has been developed to boost mucosal immunity for SARS-CoV-2 infection. In addition to IgM and IgA, bispecific antibodies (bsAbs) combine specificities of two antibodies in one molecule, representing an important alternative to monoclonal antibody cocktails. Here, we summarize the latest advances in studies on IgM, IgA and bsAbs against SARS-CoV-2. The current challenges and future directions in vaccine design and antibody-based therapeutics are also discussed.

[end paper 97]

[begin paper 98]

PMCID = 10562541

Title = A novel bispecific antibody dual-targeting approach for enhanced neutralization against fast-evolving SARS-CoV-2 variants

The emergence of new severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants has caused unprecedented health and socioeconomic crises, necessitating the immediate development of highly effective neutralizing antibodies. Despite recent advancements in anti-SARS-CoV-2 receptor-binding domain (RBD)-specific monoclonal antibodies (mAbs) derived from convalescent patient samples, their efficacy against emerging variants has been limited. In this study, we present a novel dual-targeting strategy using bispecific antibodies (bsAbs) that specifically recognize both the SARS-CoV-2 RBD and fusion peptide (FP), crucial domains for viral attachment to the host cell membrane and fusion in SARS-CoV-2 infection. 

Using phage display technology, we rapidly isolated FP-specific mAbs from an established human recombinant antibody library, identifying K107.1 with a nanomolar affinity for SARS-CoV-2 FP. Furthermore, we generated K203.A, a new bsAb built in immunoglobulin G4-(single-chain variable fragment)2 forms and demonstrating a high manufacturing yield and nanomolar affinity to both the RBD and FP, by fusing K102.1, our previously reported RBD-specific mAb, with K107.1. 

Our comprehensive in vitro functional analyses revealed that the K203.A bsAb significantly outperformed the parental RBD-specific mAb in terms of neutralization efficacy against SARS-CoV-2 variants. Furthermore, intravenous monotherapy with K203.A demonstrated potent in vivo neutralizing activity without significant in vivo toxicity in a mouse model infected with a SARS-CoV-2 variant. 

These findings present a novel bsAb dual-targeting strategy, directed at SARS-CoV-2 RBD and FP, as an effective approach for rapid development and management against continuously evolving SARS-CoV-2 variants.

[end paper 98]

[begin paper 99]

PMCID = 9035363

Title = LY-CoV1404 (bebtelovimab) potently neutralizes SARS-CoV-2 variants

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)-neutralizing monoclonal antibodies (mAbs) can reduce the risk of hospitalization from coronavirus disease 2019 (COVID-19) when administered early. However, SARS-CoV-2 variants of concern (VOCs) have negatively affected therapeutic use of some authorized mAbs. Using a high-throughput B cell screening pipeline, we isolated LY-CoV1404 (bebtelovimab), a highly potent SARS-CoV-2 spike glycoprotein receptor binding domain (RBD)-specific antibody. LY-CoV1404 potently neutralizes authentic SARS-CoV-2, B.1.1.7, B.1.351, and B.1.617.2. In pseudovirus neutralization studies, LY-CoV1404 potently neutralizes variants, including B.1.1.7, B.1.351, B.1.617.2, B.1.427/B.1.429, P.1, B.1.526, B.1.1.529, and the BA.2 subvariant. Structural analysis reveals that the contact residues of the LY-CoV1404 epitope are highly conserved, except for N439 and N501. The binding and neutralizing activity of LY-CoV1404 is unaffected by the most common mutations at these positions (N439K and N501Y). The broad and potent neutralization activity and the relatively conserved epitope suggest that LY-CoV1404 has the potential to be an effective therapeutic agent to treat all known variants.

Westendorf et al. show that LY-CoV1404 is a potent SARS-CoV-2-binding antibody that neutralizes all known variants of concern and that its epitope is rarely mutated.

[end paper 99]

[begin paper 100]

PMCID = 10847457

Title = Potent and broadly neutralizing antibodies against sarbecoviruses induced by sequential COVID-19 vaccination

The current SARS-CoV-2 variants strikingly evade all authorized monoclonal antibodies and threaten the efficacy of serum-neutralizing activity elicited by vaccination or prior infection, urging the need to develop antivirals against SARS-CoV-2 and related sarbecoviruses. Here, we identified both potent and broadly neutralizing antibodies from a five-dose vaccinated donor who exhibited cross-reactive serum-neutralizing activity against diverse coronaviruses. Through single B-cell sorting and sequencing followed by a tailor-made computational pipeline, we successfully selected 86 antibodies with potential cross-neutralizing ability from 684 antibody sequences. Among them, PW5-570 potently neutralized all SARS-CoV-2 variants that arose prior to Omicron BA.5, and the other three could broadly neutralize all current SARS-CoV-2 variants of concern, SARS-CoV and their related sarbecoviruses (Pangolin-GD, RaTG13, WIV-1, and SHC014). Cryo-EM analysis demonstrates that these antibodies have diverse neutralization mechanisms, such as disassembling spike trimers, or binding to RBM or SD1 to affect ACE2 binding. In addition, prophylactic administration of these antibodies significantly protects nasal turbinate and lung infections against BA.1, XBB.1, and SARS-CoV viral challenge in golden Syrian hamsters, respectively. Importantly, post-exposure treatment with PW5-5 and PW5-535 also markedly protects against XBB.1 challenge in these models. This study reveals the potential utility of computational process to assist screening cross-reactive antibodies, as well as the potency of vaccine-induced broadly neutralizing antibodies against current SARS-CoV-2 variants and related sarbecoviruses, offering promising avenues for the development of broad therapeutic antibody drugs.

[end paper 100]



