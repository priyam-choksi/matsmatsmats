## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "nanobodies" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on nanobodies to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on nanobodies using the query "nanobodies AND SARS-CoV-2 AND spike protein" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to nanobodies. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, I have selected the following papers that are directly relevant to the design and development of nanobodies targeting the SARS-CoV-2 spike protein. These papers provide insights into the structural, functional, and therapeutic aspects of nanobodies, which are crucial for your research project. Here is the Python dictionary mapping PMCID to title:

```python
{
    "8651292": "Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape",
    "11140877": "Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses",
    "7932109": "Structure-guided multivalent nanobodies block SARS-CoV-2 infection and suppress mutational escape",
    "11076045": "Nanobody repertoire generated against the spike protein of ancestral SARS-CoV-2 remains efficacious against the rapidly evolving virus",
    "8907017": "Broad neutralization of SARS-CoV-2 variants by an inhalable bispecific single-domain antibody",
    "9257560": "A panel of nanobodies recognizing conserved hidden clefts of all SARS-CoV-2 spike variants including Omicron",
    "8907979": "Single-Domain Antibodies Efficiently Neutralize SARS-CoV-2 Variants of Concern",
    "10301551": "Neutralizing and Enhancing Epitopes of the SARS-CoV-2 Receptor-Binding Domain (RBD) Identified by Nanobodies",
    "10986514": "Nanobody engineering for SARS-CoV-2 neutralization and detection",
    "9792944": "Broadly neutralizing and protective nanobodies against SARS-CoV-2 Omicron subvariants BA.1, BA.2, and BA.4/5 and diverse sarbecoviruses",
    "7755911": "High affinity nanobodies block SARS-CoV-2 spike receptor binding domain interaction with human angiotensin converting enzyme",
    "8420576": "Neutralization of SARS‐CoV‐2 by highly potent, hyperthermostable, and mutation‐tolerant nanobodies",
    "9170159": "Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants",
    "9869787": "A Multivalent and Thermostable Nanobody Neutralizing SARS-CoV-2 Omicron (B.1.1.529)",
    "10444775": "Discovery and multimerization of cross-reactive single-domain antibodies against SARS-like viruses to enhance potency and address emerging SARS-CoV-2 variants",
    "8956255": "Multivariate mining of an alpaca immune repertoire identifies potent cross-neutralizing SARS-CoV-2 nanobodies",
    "8126837": "Nanobody cocktails potently neutralize SARS-CoV-2 D614G N501Y variant and protect mice",
    "8043454": "Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2",
    "10268213": "Fully synthetic platform to rapidly generate tetravalent bispecific nanobody–based immunoglobulins",
    "11554226": "Super broad and protective nanobodies against Sarbecoviruses including SARS-CoV-1 and the divergent SARS-CoV-2 subvariant KP.3.1.1",
    "9365158": "A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike",
    "10892724": "SARS-CoV-2 Specific Nanobodies Neutralize Different Variants of Concern and Reduce Virus Load in the Brain of h-ACE2 Transgenic Mice",
    "8908544": "Structural basis of nanobodies neutralizing SARS-CoV-2 variants",
    "7870875": "Potent neutralization of clinical isolates of SARS-CoV-2 D614 and G614 variants by a monomeric, sub-nanomolar affinity nanobody",
    "8625641": "Structural Biology of Nanobodies against the Spike Protein of SARS-CoV-2",
    "8097376": "NeutrobodyPlex—monitoring SARS‐CoV‐2 neutralizing immune responses using nanobodies",
    "8260353": "Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants",
    "8223476": "Directed evolution of potent neutralizing nanobodies against SARS-CoV-2 using CDR-swapping mutagenesis",
    "8448731": "A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies",
    "10688893": "Development of a bispecific nanobody conjugate broadly neutralizes diverse SARS-CoV-2 variants and structural basis for its broad neutralization",
    "10865538": "Single-domain antibodies against SARS-CoV-2 RBD from a two-stage phage screening of universal and focused synthetic libraries",
    "10588750": "A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern",
    "8458290": "A potent SARS-CoV-2 neutralising nanobody shows therapeutic efficacy in the Syrian golden hamster model of COVID-19",
    "10369967": "Nanobody repertoire generated against the spike protein of ancestral SARS-CoV-2 remains efficacious against the rapidly evolving virus",
    "7844813": "Development of multivalent nanobodies blocking SARS-CoV-2 infection by targeting RBD of spike protein",
    "9884143": "Expanding and improving nanobody repertoires using a yeast display method: Targeting SARS-CoV-2",
    "8748511": "A bispecific monomeric nanobody induces spike trimer dimers and neutralizes SARS-CoV-2 in vivo",
    "11630572": "Discovery of Nanosota-9 as anti-Omicron nanobody therapeutic candidate",
    "8354634": "The development of Nanosota-1 as anti-SARS-CoV-2 nanobody drug candidates",
    "9850341": "The Potential of Nanobodies for COVID-19 Diagnostics and Therapeutics",
    "11140111": "Single domain antibodies from camelids in the treatment of microbial infections",
    "9461429": "Arsenal of nanobodies shows broad-spectrum neutralization against SARS-CoV-2 variants of concern in vitro and in vivo in hamster models",
    "11211474": "A broad neutralizing nanobody against SARS-CoV-2 engineered from an approved drug",
    "11460708": "Structure-guided in vitro evolution of nanobodies targeting new viral variants",
    "9230756": "Single-Domain Antibodies as Therapeutics for Respiratory RNA Virus Infections",
    "8982573": "Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance",
    "7744031": "Slaying SARS-CoV-2 One (Single-domain) Antibody at a Time",
    "7857409": "An ultrapotent synthetic nanobody neutralizes SARS-CoV-2 by stabilizing inactive Spike",
    "11148044": "Generation of nanobodies from transgenic ‘LamaMice’ lacking an endogenous immunoglobulin repertoire",
    "7199733": "Structural Basis for Potent Neutralization of Betacoronaviruses by Single-Domain Camelid Antibodies",
    "9219022": "Production and purification of shark and camel single domain antibodies from bacterial and mammalian cell expression systems",
    "9082315": "Nanobodies Protecting From Lethal SARS-CoV-2 Infection Target Receptor Binding Epitopes Preserved in Virus Variants Other Than Omicron",
    "8325376": "Nanobodies as powerful pulmonary targeted biotherapeutics against SARS-CoV-2, pharmaceutical point of view",
    "8726723": "Rapid discovery of diverse neutralizing SARS-CoV-2 antibodies from large-scale synthetic phage libraries",
    "7832525": "Nanobodies, the potent agents to detect and treat the Coronavirus infections: A systematic review",
    "8021111": "Dromedary camels as a natural source of neutralizing nanobodies against SARS-CoV-2",
    "9896449": "Shark nanobodies with potent SARS-CoV-2 neutralizing activity and broad sarbecovirus reactivity",
    "11762740": "Visual and High-Efficiency Secretion of SARS-CoV-2 Nanobodies with Escherichia coli",
    "10231691": "Plant-derived single domain COVID-19 antibodies",
    "9356539": "Rapid and Quantitative In-vitro Evaluation of SARS-CoV-2 Neutralizing Antibodies and Nanobodies",
    "10590698": "Nanobodies with cross-neutralizing activity provide prominent therapeutic efficacy in mild and severe COVID-19 rodent models",
    "9678391": "Nanobody derived using a peptide epitope from the spike protein receptor-binding motif inhibits entry of SARS-CoV-2 variants",
    "9479348": "A potent synthetic nanobody with broad-spectrum activity neutralizes SARS-CoV-2 virus and the Omicron variant BA.1 through a unique binding mode",
    "8811630": "Biparatopic nanobodies protect mice from lethal challenge with SARS‐CoV‐2 variants of concern",
    "9834170": "Two pan-SARS-CoV-2 nanobodies and their multivalent derivatives effectively prevent Omicron infections in mice",
    "8324831": "A synthetic nanobody targeting RBD protects hamsters from SARS-CoV-2 infection",
    "7857400": "Versatile and multivalent nanobodies efficiently neutralize SARS-CoV-2",
    "8967979": "Short-Term Instantaneous Prophylaxis and Efficient Treatment Against SARS-CoV-2 in hACE2 Mice Conferred by an Intranasal Nanobody (Nb22)",
    "9529347": "Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2",
    "7483421": "Humanized single domain antibodies neutralize SARS-CoV-2 by targeting the spike receptor binding domain",
    "9482557": "Single domain antibodies derived from ancient animals as broadly neutralizing agents for SARS-CoV-2 and other coronaviruses",
    "7556756": "Antibody-like proteins that capture and neutralize SARS-CoV-2",
    "11277559": "Single-Domain Antibodies—Novel Tools to Study and Treat Allergies",
    "8351461": "Structural Basis for SARS-CoV-2 Nucleocapsid Protein Recognition by Single-Domain Antibodies",
    "8344751": "Development of potent and effective synthetic SARS-CoV-2 neutralizing nanobodies",
    "9334538": "Hetero-bivalent nanobodies provide broad-spectrum protection against SARS-CoV-2 variants of concern including Omicron",
    "7642358": "Selection, biophysical and structural analysis of synthetic nanobodies that effectively neutralize SARS-CoV-2",
    "9512078": "Mucosal nanobody IgA as inhalable and affordable prophylactic and therapeutic treatment against SARS-CoV-2 and emerging variants",
    "9288967": "Accelerating PERx reaction enables covalent nanobodies for potent neutralization of SARS-CoV-2 and variants",
    "10167778": "Isolation and characterization of single domain antibodies from banded houndshark (Triakis scyllium) targeting SARS-CoV-2 spike RBD protein",
    "9966208": "Potent Therapeutic Strategies for COVID-19 with Single-Domain Antibody Immunoliposomes Neutralizing SARS-CoV-2 and Lip/cGAMP Enhancing Protective Immunity",
    "11640846": "Backstage Heroes—Yeast in COVID-19 Research",
    "9020654": "Isolation, characterization, and structure-based engineering of a neutralizing nanobody against SARS-CoV-2",
    "7473855": "An alpaca nanobody neutralizes SARS-CoV-2 by blocking receptor interaction",
    "8928829": "Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19"
}
```

These papers will provide a comprehensive understanding of the current landscape of nanobody research against SARS-CoV-2, focusing on their development, engineering, and therapeutic potential.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "nanobodies AND SARS-CoV-2 AND spike protein":

[begin paper 1]

PMCID = 8651292

Title = Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape

The emergence of SARS-CoV-2 variants threatens current vaccines and therapeutic antibodies and urgently demands powerful new therapeutics that can resist viral escape. We therefore generated a large nanobody repertoire to saturate the distinct and highly conserved available epitope space of SARS-CoV-2 spike, including the S1 receptor binding domain, N-terminal domain, and the S2 subunit, to identify new nanobody binding sites that may reflect novel mechanisms of viral neutralization. Structural mapping and functional assays show that indeed these highly stable monovalent nanobodies potently inhibit SARS-CoV-2 infection, display numerous neutralization mechanisms, are effective against emerging variants of concern, and are resistant to mutational escape. Rational combinations of these nanobodies that bind to distinct sites within and between spike subunits exhibit extraordinary synergy and suggest multiple tailored therapeutic and prophylactic strategies.

[end paper 1]

[begin paper 2]

PMCID = 11140877

Title = Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses

Nanobodies, single-domain antibodies derived from variable domain of camelid or shark heavy-chain antibodies, have unique properties with small size, strong binding affinity, easy construction in versatile formats, high neutralizing activity, protective efficacy, and manufactural capacity on a large-scale. Nanobodies have been arisen as an effective research tool for development of nanobiotechnologies with a variety of applications. Three highly pathogenic coronaviruses (CoVs), SARS-CoV-2, SARS-CoV, and MERS-CoV, have caused serious outbreaks or a global pandemic, and continue to post a threat to public health worldwide. The viral spike (S) protein and its cognate receptor-binding domain (RBD), which initiate viral entry and play a critical role in virus pathogenesis, are important therapeutic targets. This review describes pathogenic human CoVs, including viral structures and proteins, and S protein-mediated viral entry process. It also summarizes recent advances in development of nanobodies targeting these CoVs, focusing on those targeting the S protein and RBD. Finally, we discuss potential strategies to improve the efficacy of nanobodies against emerging SARS-CoV-2 variants and other CoVs with pandemic potential. It will provide important information for rational design and evaluation of therapeutic agents against emerging and reemerging pathogens.

[end paper 2]

[begin paper 3]

PMCID = 7932109

Title = Structure-guided multivalent nanobodies block SARS-CoV-2 infection and suppress mutational escape

Monoclonal antibodies are an important weapon in the battle against COVID-19. However, these large proteins are difficult to produce in the needed quantities and at low cost. Attention has turned to nanobodies, which are aptly named, single-domain antibodies that are easier to produce and have the potential to be administered by inhalation. Koenig et al. describe four nanobodies that bind to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein and prevent infection of cells (see the Perspective by Saelens and Schepens). Structures show that the nanobodies target two distinct epitopes on the SARS-CoV-2 spike protein. Multivalent nanobodies neutralize virus much more potently than single nanobodies, and multivalent nanobodies that bind two epitopes prevent the emergence of viral escape mutants.

Science, this issue p. eabe6230; see also p. 

SARS-CoV-2–neutralizing nanobodies were combined to design potent multivalent nanobodies.

The global scale and rapid spread of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pose unprecedented challenges to society, health care systems, and science. In addition to effective and safe vaccines, passive immunization by antibody-related molecules offers an opportunity to harness the vertebrate immune system to fight viral infections in high-risk patients. Variable domains of heavy-chain–only antibodies (VHHs), also known as nanobodies, are suitable lead molecules in such efforts, as they are small, extremely stable, easy to engineer, and economic to produce in simple expression systems.

We engineered improved multivalent nanobodies neutralizing SARS-CoV-2 on the basis of two principles: (i) detailed structural information of their epitopes and binding modes to the viral spike protein and (ii) mechanistic insights into viral fusion with cellular membranes catalyzed by the spike.

Nanobodies specific for the receptor binding domain (RBD) of SARS-CoV-2 spike were identified by phage display using nanobody libraries from an alpaca and a llama immunized with the RBD and inactivated virus. Four of the resulting nanobodies—VHHs E, U, V, and W—potently neutralize SARS-CoV-2 and SARS-CoV-2–pseudotyped vesicular stomatitis virus. X-ray crystallography revealed that the nanobodies bind to two distinct epitopes on the RBD, interfaces “E” and “UVW,” which can be synergistically targeted by combinations of nanobodies to inhibit infection. Cryo–electron microscopy (cryo-EM) of trimeric spike in complex with VHH E and VHH V revealed that VHH E stabilizes a conformation of the spike with all three RBDs in the “up” conformation (3-up), a state that is typically associated with activation by receptor binding. In line with this observation, we found that VHH E triggers the fusion activity of spike in the absence of the cognate receptor ACE2. VHH V, by contrast, stabilizes spike in a 2-up conformation and does not induce fusion. On the basis of the structural information, we designed bi- and trivalent nanobodies with improved neutralizing properties. VHH EEE most potently inhibited infection, did not activate fusion, and likely inactivated virions by outcompeting interaction of the virus with its receptor. Yet evolution experiments revealed emergence of escape mutants in the spike with single–amino acid changes that were completely insensitive to inhibition by VHH EEE. VHH VE also neutralized more efficiently than VHH E or VHH V alone; stabilized the 3-up conformation of spike, as determined by cryo-EM; and more strongly induced the spike fusogenic activity. We conclude that the premature activation of the fusion machinery on virions was an unexpected mechanism of neutralization, as enhanced neutralization could not be attributed simply to better blocking of virus-receptor interactions. Activation of spike in the absence of target membranes likely induces irreversible conformational changes to assume the energetically favorable postfusion conformation without catalyzing fusion per se. Simultaneous targeting of two independent epitopes by VHH VE largely prevented the emergence of resistant escape mutants in evolution experiments.

Our results demonstrate the strength of the modular combination of nanobodies for neutralization. Premature activation of spike by nanobodies reveals an unusual mode of neutralization and yields insights into the mechanism of fusion.

On virions, SARS-CoV-2 spike trimers are mostly in an inactive configuration with all RBDs in the down conformation (left). Binding of bivalent nanobody VE stabilizes the spike in an active conformation with all RBDs up (middle), triggering premature induction of the postfusion conformation, which irreversibly inactivates the spike protein (right).

The pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) continues to spread, with devastating consequences. For passive immunization efforts, nanobodies have size and cost advantages over conventional antibodies. In this study, we generated four neutralizing nanobodies that target the receptor binding domain of the SARS-CoV-2 spike protein. We used x-ray crystallography and cryo–electron microscopy to define two distinct binding epitopes. On the basis of these structures, we engineered multivalent nanobodies with more than 100 times the neutralizing activity of monovalent nanobodies. Biparatopic nanobody fusions suppressed the emergence of escape mutants. Several nanobody constructs neutralized through receptor binding competition, whereas other monovalent and biparatopic nanobodies triggered aberrant activation of the spike fusion machinery. These premature conformational changes in the spike protein forestalled productive fusion and rendered the virions noninfectious.

[end paper 3]

[begin paper 4]

PMCID = 7224157

Title = Identification of Human Single-Domain Antibodies against SARS-CoV-2

The worldwide spread of COVID-19 highlights the need for an efficient approach to rapidly develop therapeutics and prophylactics against SARS-CoV-2. The SARS-CoV-2 spike protein, containing the receptor-binding domain (RBD) and S1 subunit involved in receptor engagement, is a potential therapeutic target. We describe the development of a phage-displayed single-domain antibody library by grafting naive complementarity-determining regions (CDRs) into framework regions of a human germline immunoglobulin heavy chain variable region (IGHV) allele. Panning this library against SARS-CoV-2 RBD and S1 subunit identified fully human single-domain antibodies targeting five distinct epitopes on SARS-CoV-2 RBD with subnanomolar to low nanomolar affinities. Some of these antibodies neutralize SARS-CoV-2 by targeting a cryptic epitope located in the spike trimeric interface. Collectively, this work presents a versatile platform for rapid antibody isolation and identifies promising therapeutic anti-SARS-CoV-2 antibodies as well as the diverse immogneic profile of the spike protein.

A phage-displayed human single-domain antibody library is developed

Single-domain antibodies targeting five types of SARS-CoV-2 epitopes are identified

Some neutralizing antibodies target cryptic SARS-CoV-2 spike trimeric interface

Unique immunogenic profile of SARS-CoV-2 RBD is revealed

Wu et al. describe the development of a versatile platform for rapid isolation of fully human single-domain antibodies and apply this methodology to identify SARS-CoV-2-specific antibodies. These human single-domain antibodies target diverse epitopes within the SARS-CoV-2 spike protein receptor binding domain (RBD) and may yield potential therapeutic candidates for COVID-19.

[end paper 4]

[begin paper 5]

PMCID = 11076045

Title = Nanobody repertoire generated against the spike protein of ancestral SARS-CoV-2 remains efficacious against the rapidly evolving virus

To date, all major modes of monoclonal antibody therapy targeting SARS-CoV-2 have lost significant efficacy against the latest circulating variants. As SARS-CoV-2 omicron sublineages account for over 90% of COVID-19 infections, evasion of immune responses generated by vaccination or exposure to previous variants poses a significant challenge. A compelling new therapeutic strategy against SARS-CoV-2 is that of single-domain antibodies, termed nanobodies, which address certain limitations of monoclonal antibodies. Here, we demonstrate that our high-affinity nanobody repertoire, generated against wild-type SARS-CoV-2 spike protein (Mast et al., 2021), remains effective against variants of concern, including omicron BA.4/BA.5; a subset is predicted to counter resistance in emerging XBB and BQ.1.1 sublineages. Furthermore, we reveal the synergistic potential of nanobody cocktails in neutralizing emerging variants. Our study highlights the power of nanobody technology as a versatile therapeutic and diagnostic tool to combat rapidly evolving infectious diseases such as SARS-CoV-2.

[end paper 5]

[begin paper 6]

PMCID = 9797221

Title = Therapeutic applications of nanobodies against SARS-CoV-2 and other viral infections: Current update

In the last two years, the world encountered the SARS-CoV-2 virus, which is still dominating the population due to the absence of a viable treatment. To eradicate the global pandemic, scientists, doctors, and researchers took an exceptionally significant initiative towards the development of effective therapeutics to save many lifes. This review discusses about the single-domain antibodies (sdAbs), also called nanobodies, their structure, and their types against the infections of dreadful SARS-CoV-2 virus. A precise description highlights the nanobodies and their therapeutic application against the other selected viruses. It aims to focus on the extraordinary features of these antibodies compared to the conventional therapeutics like mAbs, convalescent plasma therapy, and vaccines. The stable structure of these nanobodies along with the suitable mechanism of action also confers greater resistance to the evolving variants with numerous mutations. The nanobodies developed against SARS-CoV-2 and its mutant variants have shown the greater neutralization potential than the primitive ones. Engineering of these specialized antibodies by modern biotechnological approaches will surely be more beneficial in treating this COVID-19 pandemic along with certain other viral infections.

[end paper 6]

[begin paper 7]

PMCID = 8907017

Title = Broad neutralization of SARS-CoV-2 variants by an inhalable bispecific single-domain antibody

The effectiveness of SARS-CoV-2 vaccines and therapeutic antibodies have been limited by the continuous emergence of viral variants and by the restricted diffusion of antibodies from circulation into the sites of respiratory virus infection. Here, we report the identification of two highly conserved regions on the Omicron variant receptor-binding domain recognized by broadly neutralizing antibodies. Furthermore, we generated a bispecific single-domain antibody that was able to simultaneously and synergistically bind these two regions on a single Omicron variant receptor-binding domain as revealed by cryo-EM structures. We demonstrated that this bispecific antibody can be effectively delivered to lung via inhalation administration and exhibits exquisite neutralization breadth and therapeutic efficacy in mouse models of SARS-CoV-2 infections. Importantly, this study also deciphered an uncommon and highly conserved cryptic epitope within the spike trimeric interface that may have implications for the design of broadly protective SARS-CoV-2 vaccines and therapeutics.

Therapeutic effects against SARS-CoV-2 infection are seen in mice after inhalation of an engineered, bispecific, single-domain antibody that simultaneously targets two points on the spike protein receptor-binding domain.

[end paper 7]

[begin paper 8]

PMCID = 8730734

Title = Outlook of therapeutic and diagnostic competency of nanobodies against SARS-CoV-2: A systematic review

The newly emerged coronavirus (SARS-CoV-2) continues to infect humans, and no completely efficient treatment has yet been found. Antibody therapy is one way to control infection caused by COVID-19, but the use of classical antibodies has many disadvantages. Heavy chain antibodies (HCAbs) are single-domain antibodies derived from the Camelidae family. The variable part of these antibodies (Nanobodies or VHH) has interesting properties such as small size, identify criptic epitopes, stability in harsh conditions, good tissue permeability and cost-effective production causing nanobodies have become a good candidate in the treatment and diagnosis of viral infections.

Totally 157 records (up to November 10, 2021), were recognized to be reviewed in this study. 62 studies were removed after first step screening due to their deviation from inclusion criteria. The remaining 95 studies were reviewed in details. After removing articles that were not in the study area, 45 remaining studies met the inclusion criteria and were qualified to be included in the systematic review.

In this systematic review, the application of nanobodies in the treatment and detection of COVID-19 infection was reviewed. The results of this study showed that extensive and sufficient studies have been performed in the field of production of nanobodies against SARS-CoV-2 virus and the obtained nanobodies have a great potential for use in patients infected with SARS-CoV-2 virus.

According to the obtained results, it was found that nanobodies can be used effectively in the treatment and diagnosis of SARS-CoV-2 virus.

[end paper 8]

[begin paper 9]

PMCID = 8951411

Title = Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering

Nanobodies provide important advantages over traditional antibodies, including their smaller size and robust biochemical properties such as high thermal stability, high solubility, and the ability to be bioengineered into novel multivalent, multi-specific, and high-affinity molecules, making them a class of emerging powerful therapies against SARS-CoV-2. Recent research efforts on the design, protein engineering, and structure-functional characterization of nanobodies and their binding with SARS-CoV-2 S proteins reflected a growing realization that nanobody combinations can exploit distinct binding epitopes and leverage the intrinsic plasticity of the conformational landscape for the SARS-CoV-2 S protein to produce efficient neutralizing and mutation resistant characteristics. Structural and computational studies have also been instrumental in quantifying the structure, dynamics, and energetics of the SARS-CoV-2 spike protein binding with nanobodies. In this review, a comprehensive analysis of the current structural, biophysical, and computational biology investigations of SARS-CoV-2 S proteins and their complexes with distinct classes of nanobodies targeting different binding sites is presented. The analysis of computational studies is supplemented by an in-depth examination of mutational scanning simulations and identification of binding energy hotspots for distinct nanobody classes. The review is focused on the analysis of mechanisms underlying synergistic binding of multivalent nanobodies that can be superior to single nanobodies and conventional nanobody cocktails in combating escape mutations by effectively leveraging binding avidity and allosteric cooperativity. We discuss how structural insights and protein engineering approaches together with computational biology tools can aid in the rational design of synergistic combinations that exhibit superior binding and neutralization characteristics owing to avidity-mediated mechanisms.

[end paper 9]

[begin paper 10]

PMCID = 8877688

Title = Allosteric Determinants of the SARS-CoV-2 Spike Protein Binding with Nanobodies: Examining Mechanisms of Mutational Escape and Sensitivity of the Omicron Variant

Structural and biochemical studies have recently revealed a range of rationally engineered nanobodies with efficient neutralizing capacity against the SARS-CoV-2 virus and resilience against mutational escape. In this study, we performed a comprehensive computational analysis of the SARS-CoV-2 spike trimer complexes with single nanobodies Nb6, VHH E, and complex with VHH E/VHH V nanobody combination. We combined coarse-grained and all-atom molecular simulations and collective dynamics analysis with binding free energy scanning, perturbation-response scanning, and network centrality analysis to examine mechanisms of nanobody-induced allosteric modulation and cooperativity in the SARS-CoV-2 spike trimer complexes with these nanobodies. By quantifying energetic and allosteric determinants of the SARS-CoV-2 spike protein binding with nanobodies, we also examined nanobody-induced modulation of escaping mutations and the effect of the Omicron variant on nanobody binding. The mutational scanning analysis supported the notion that E484A mutation can have a significant detrimental effect on nanobody binding and result in Omicron-induced escape from nanobody neutralization. Our findings showed that SARS-CoV-2 spike protein might exploit the plasticity of specific allosteric hotspots to generate escape mutants that alter response to binding without compromising activity. The network analysis supported these findings showing that VHH E/VHH V nanobody binding can induce long-range couplings between the cryptic binding epitope and ACE2-binding site through a broader ensemble of communication paths that is less dependent on specific mediating centers and therefore may be less sensitive to mutational perturbations of functional residues. The results suggest that binding affinity and long-range communications of the SARS-CoV-2 complexes with nanobodies can be determined by structurally stable regulatory centers and conformationally adaptable hotspots that are allosterically coupled and collectively control resilience to mutational escape.

[end paper 10]

[begin paper 11]

PMCID = 9043892

Title = Passive Immunotherapy Against SARS-CoV-2: From Plasma-Based Therapy to Single Potent Antibodies in the Race to Stay Ahead of the Variants

The COVID-19 pandemic is now approaching 2 years old, with more than 440 million people infected and nearly six million dead worldwide, making it the most significant pandemic since the 1918 influenza pandemic. The severity and significance of SARS-CoV-2 was recognized immediately upon discovery, leading to innumerable companies and institutes designing and generating vaccines and therapeutic antibodies literally as soon as recombinant SARS-CoV-2 spike protein sequence was available. Within months of the pandemic start, several antibodies had been generated, tested, and moved into clinical trials, including Eli Lilly’s bamlanivimab and etesevimab, Regeneron’s mixture of imdevimab and casirivimab, Vir’s sotrovimab, Celltrion’s regdanvimab, and Lilly’s bebtelovimab. These antibodies all have now received at least Emergency Use Authorizations (EUAs) and some have received full approval in select countries. To date, more than three dozen antibodies or antibody combinations have been forwarded into clinical trials. These antibodies to SARS-CoV-2 all target the receptor-binding domain (RBD), with some blocking the ability of the RBD to bind human ACE2, while others bind core regions of the RBD to modulate spike stability or ability to fuse to host cell membranes. While these antibodies were being discovered and developed, new variants of SARS-CoV-2 have cropped up in real time, altering the antibody landscape on a moving basis. Over the past year, the search has widened to find antibodies capable of neutralizing the wide array of variants that have arisen, including Alpha, Beta, Gamma, Delta, and Omicron. The recent rise and dominance of the Omicron family of variants, including the rather disparate BA.1 and BA.2 variants, demonstrate the need to continue to find new approaches to neutralize the rapidly evolving SARS-CoV-2 virus. This review highlights both convalescent plasma- and polyclonal antibody-based approaches as well as the top approximately 50 antibodies to SARS-CoV-2, their epitopes, their ability to bind to SARS-CoV-2 variants, and how they are delivered. New approaches to antibody constructs, including single domain antibodies, bispecific antibodies, IgA- and IgM-based antibodies, and modified ACE2-Fc fusion proteins, are also described. Finally, antibodies being developed for palliative care of COVID-19 disease, including the ramifications of cytokine release syndrome (CRS) and acute respiratory distress syndrome (ARDS), are described.

The online version contains supplementary material available at 10.1007/s40259-022-00529-7.

[end paper 11]

[begin paper 12]

PMCID = 11407660

Title = Dual-role epitope on SARS-CoV-2 spike enhances and neutralizes viral entry across different variants

Grasping the roles of epitopes in viral glycoproteins is essential for unraveling the structure and function of these proteins. Up to now, all identified epitopes have been found to either neutralize, have no effect on, or enhance viral entry into cells. Here, we used nanobodies (single-domain antibodies) as probes to investigate a unique epitope on the SARS-CoV-2 spike protein, located outside the protein’s receptor-binding domain. Nanobody binding to this epitope enhances the cell entry of prototypic SARS-CoV-2, while neutralizing the cell entry of SARS-CoV-2 Omicron variant. Moreover, nanobody binding to this epitope promotes both receptor binding activity and post-attachment activity of prototypic spike, explaining the enhanced viral entry. The opposite occurs with Omicron spike, explaining the neutralized viral entry. This study reveals a unique epitope that can both enhance and neutralize viral entry across distinct viral variants, suggesting that epitopes may vary their roles depending on the viral context. Consequently, antibody therapies should be assessed across different viral variants to confirm their efficacy and safety.

Antibodies bind to specific epitopes on viral surface glycoproteins to perform their function: inhibiting viral infection, having no effect, or, in rare cases, enhancing viral infection. In our research, we used nanobodies (small, single-domain antibodies from camelid animals) as probes and identified a unique epitope on the SARS-CoV-2 spike protein with opposing functions across different SARS-CoV-2 variants. It enhances viral infection in pre-Omicron variants but inhibits it in the Omicron variant. We further investigated the molecular mechanism underlying these opposing effects. Although this epitope is not directly involved in receptor binding, nanobody binding to it modulates receptor binding and post-receptor-binding activities. This study marks the first discovery of a dual-function epitope on the SARS-CoV-2 spike protein that both enhances and inhibits viral infection across different viral variants. This discovery challenges the traditional categorization of epitopes, underscores the complex evolution of the SARS-CoV-2 spike protein, and offers new insights into antiviral antibody therapies.

[end paper 12]

[begin paper 13]

PMCID = 11285730

Title = Structural and functional characterization of nanobodies that neutralize Omicron variants of SARS-CoV-2

The Omicron strains of SARS-CoV-2 pose a significant challenge to the development of effective antibody-based treatments as immune evasion has compromised most available immune therapeutics. Therefore, in the ‘arms race’ with the virus, there is a continuing need to identify new biologics for the prevention or treatment of SARS-CoV-2 infections. Here, we report the isolation of nanobodies that bind to the Omicron BA.1 spike protein by screening nanobody phage display libraries previously generated from llamas immunized with either the Wuhan or Beta spike proteins. The structure and binding properties of three of these nanobodies (A8, H6 and B5-5) have been characterized in detail providing insight into their binding epitopes on the Omicron spike protein. Trimeric versions of H6 and B5-5 neutralized the SARS-CoV-2 variant of concern BA.5 both in vitro and in the hamster model of COVID-19 following nasal administration. Thus, either alone or in combination could serve as starting points for the development of new anti-viral immunotherapeutics.

[end paper 13]

[begin paper 14]

PMCID = 8514607

Title = The role of single-domain antibodies (or nanobodies) in SARS-CoV-2 neutralization

The severe acute respiratory syndrome (SARS-CoV-2), a newly emerging of coronavirus, continues to infect humans in the absence of a viable treatment. Neutralizing antibodies that disrupt the interaction of RBD and ACE2 has been under the spotlight as a way of developing the COVID-19 treatment. Some animals, such as llamas, manufacture heavy-chain antibodies that have a single variable domain (VHH) instead of two variable domains (VH/VL) as opposed to typical antibodies. Nanobodies are antigen-specific, single-domain, changeable segments of camelid heavy chain-only antibodies that are recombinantly produced. These types of antibodies exhibit a wide range of strong physical and chemical properties, like high solubility, and stability. The VHH's high-affinity attachment to the receptor-binding domain (RBD) allowed the neutralization of SARS-CoV-2. To tackle COVID-19, some nanobodies are being developed against SARS-CoV-2, some of which have been recently included in clinical trials. Nanobody therapy may be useful in managing the COVID-19 pandemic as a potent and low-cost treatment. This paper describes the application of nanobodies as a new class of recombinant antibodies in COVID-19 treatment.

[end paper 14]

[begin paper 15]

PMCID = 9257560

Title = A panel of nanobodies recognizing conserved hidden clefts of all SARS-CoV-2 spike variants including Omicron

We are amid the historic coronavirus infectious disease 2019 (COVID-19) pandemic. Imbalances in the accessibility of vaccines, medicines, and diagnostics among countries, regions, and populations, and those in war crises, have been problematic. Nanobodies are small, stable, customizable, and inexpensive to produce. Herein, we present a panel of nanobodies that can detect the spike proteins of five SARS-CoV-2 variants of concern (VOCs) including Omicron. Here we show via ELISA, lateral flow, kinetic, flow cytometric, microscopy, and Western blotting assays that our nanobodies can quantify the spike variants. This panel of nanobodies broadly neutralizes viral infection caused by pseudotyped and authentic SARS-CoV-2 VOCs. Structural analyses show that the P86 clone targets epitopes that are conserved yet unclassified on the receptor-binding domain (RBD) and contacts the N-terminal domain (NTD). Human antibodies rarely access both regions; consequently, the clone buries hidden crevasses of SARS-CoV-2 spike proteins that go undetected by conventional antibodies.

A panel of nanobodies are presented that can detect the spike proteins of five SARS-CoV-2 variants of concern and structural analyses show that one clone targets conserved epitopes on the receptor-binding domain and contacts the N-terminal domain.

[end paper 15]

[begin paper 16]

PMCID = 8907979

Title = Single-Domain Antibodies Efficiently Neutralize SARS-CoV-2 Variants of Concern

Virus-neutralizing antibodies are one of the few treatment options for COVID-19. The evolution of SARS-CoV-2 virus has led to the emergence of virus variants with reduced sensitivity to some antibody-based therapies. The development of potent antibodies with a broad spectrum of neutralizing activity is urgently needed. Here we isolated a panel of single-domain antibodies that specifically bind to the receptor-binding domain of SARS-CoV-2 S glycoprotein. Three of the selected antibodies exhibiting most robust neutralization potency were used to generate dimeric molecules. We observed that these modifications resulted in up to a 200-fold increase in neutralizing activity. The most potent heterodimeric molecule efficiently neutralized each of SARS-CoV-2 variant of concern, including Alpha, Beta, Gamma, Delta and Omicron variants. This heterodimeric molecule could be a promising drug candidate for a treatment for COVID-19 caused by virus variants of concern.

[end paper 16]

[begin paper 17]

PMCID = 10301551

Title = Neutralizing and Enhancing Epitopes of the SARS-CoV-2 Receptor-Binding Domain (RBD) Identified by Nanobodies

Engineered nanobodies (VHs) to the SARS-CoV-2 receptor-binding domain (RBD) were generated using phage display technology. A recombinant Wuhan RBD served as bait in phage panning to fish out nanobody-displaying phages from a VH/VHH phage display library. Sixteen phage-infected E. coli clones produced nanobodies with 81.79–98.96% framework similarity to human antibodies; thus, they may be regarded as human nanobodies. Nanobodies of E. coli clones 114 and 278 neutralized SARS-CoV-2 infectivity in a dose-dependent manner; nanobodies of clones 103 and 105 enhanced the virus’s infectivity by increasing the cytopathic effect (CPE) in an infected Vero E6 monolayer. These four nanobodies also bound to recombinant Delta and Omicron RBDs and native SARS-CoV-2 spike proteins. The neutralizing VH114 epitope contains the previously reported VYAWN motif (Wuhan RBD residues 350–354). The linear epitope of neutralizing VH278 at Wuhan RBD 319RVQPTESIVRFPNITN334 is novel. In this study, for the first time, we report SARS-CoV-2 RBD-enhancing epitopes, i.e., a linear VH103 epitope at RBD residues 359NCVADVSVLYNSAPFFTFKCYG380, and the VH105 epitope, most likely conformational and formed by residues in three RBD regions that are spatially juxtaposed upon the protein folding. Data obtained in this way are useful for the rational design of subunit SARS-CoV-2 vaccines that should be devoid of enhancing epitopes. VH114 and VH278 should be tested further for clinical use against COVID-19.

[end paper 17]

[begin paper 18]

PMCID = 8768012

Title = Bovine-derived antibodies and camelid-derived nanobodies as biotherapeutic weapons against SARS-CoV-2 and its variants: A review article

The Coronavirus Disease 2019 (COVID-19) pandemic, caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has infected 305 million individuals worldwide and killed about 5.5 million people as of January 10, 2022. SARS-CoV-2 is the third major outbreak caused by a new coronavirus in the previous two decades, following SARS-CoV and MERS-CoV. Even though vaccination against SARS-CoV-2 is considered a critical strategy for preventing virus spread in the population and limiting COVID-19 clinical manifestations, new therapeutic drugs, and management strategies are urgently needed, particularly in light of the growing number of SARS-CoV-2 variants (such as Delta and Omicron variants). However, the use of conventional antibodies has faced many challenges, such as viral escape mutants, increased instability, weak binding, large sizes, the need for large amounts of plasma, and high-cost manufacturing. Furthermore, the emergence of new SARS-CoV-2 variants in the human population and recurrent coronavirus spillovers highlight the need for broadly neutralizing antibodies that are not affected by an antigenic drift that could limit future zoonotic infection. Bovine-derived antibodies and camelid-derived nanobodies are more potent and protective than conventional human antibodies, thanks to their inbuilt characteristics, and can be produced in large quantities. In addition, it was reported that these biotherapeutics are effective against a broad spectrum of epitopes, reducing the opportunity of viral pathogens to develop mutational escape. In this review, we focus on the potential benefits behind our rationale for using bovine-derived antibodies and camelid-derived nanobodies in countering SARS-CoV-2 and its emerging variants and mutants.

[end paper 18]

[begin paper 19]

PMCID = 10968798

Title = Aptamers and Nanobodies as New Bioprobes for SARS-CoV-2 Diagnostic and Therapeutic System Applications

The global challenges posed by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic have underscored the critical importance of innovative and efficient control systems for addressing future pandemics. The most effective way to control the pandemic is to rapidly suppress the spread of the virus through early detection using a rapid, accurate, and easy-to-use diagnostic platform. In biosensors that use bioprobes, the binding affinity of molecular recognition elements (MREs) is the primary factor determining the dynamic range of the sensing platform. Furthermore, the sensitivity relies mainly on bioprobe quality with sufficient functionality. This comprehensive review investigates aptamers and nanobodies recently developed as advanced MREs for SARS-CoV-2 diagnostic and therapeutic applications. These bioprobes might be integrated into organic bioelectronic materials and devices, with promising enhanced sensitivity and specificity. This review offers valuable insights into advancing biosensing technologies for infectious disease diagnosis and treatment using aptamers and nanobodies as new bioprobes.

[end paper 19]

[begin paper 20]

PMCID = 9801699

Title = Applications of nanobodies in the prevention, detection, and treatment of the evolving SARS-CoV-2

Global health and economy are deeply influenced by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and its newly emerging variants. Nanobodies with nanometer-scale size are promising for the detection and treatment of SARS-CoV-2 and its variants because they are superior to conventional antibodies in terms of cryptic epitope accessibility, tissue penetration, cost, formatting adaptability, and especially protein stability, which enables their aerosolized specific delivery to lung tissues. This review summarizes the progress in the prevention, detection, and treatment of SARS-CoV-2 using nanobodies, as well as strategies to combat the evolving SARS-CoV-2 variants. Generally, highly efficient generation of potent broad-spectrum nanobodies targeting conserved epitopes or further construction of multivalent formats targeting non-overlapping epitopes can promote neutralizing activity against SARS-CoV-2 variants and suppress immune escape.

[end paper 20]

[begin paper 21]

PMCID = 10715488

Title = Identification of nurse shark VNAR single-domain antibodies targeting the spike S2 subunit of SARS-CoV-2

SARS-CoV-2 is the etiological agent of the COVID-19 pandemic. Antibody-based therapeutics targeting the spike protein, specifically the S1 subunit or the receptor binding domain (RBD) of SARS-CoV-2, have gained attention due to their clinical efficacy in treating patients diagnosed with COVID-19. An alternative to conventional antibody therapeutics is the use of shark new antigen variable receptor domain (VNAR) antibodies. VNARs are small (<15 kDa) and can reach deep into the pockets or grooves of the target antigen. Here, we have isolated 53 VNARs that bind to the S2 subunit by phage panning from a naïve nurse shark VNAR phage display library constructed in our laboratory. Among those binders, S2A9 showed the best neutralization activity against the original pseudotyped SARS-CoV-2 virus. Several binders, including S2A9, showed cross-reactivity against S2 subunits from other β coronaviruses. Furthermore, S2A9 showed neutralization activity against all variants of concern (VOCs) from alpha to omicron (including BA1, BA2, BA4, and BA5) in both pseudovirus and live virus neutralization assays. Our findings suggest that S2A9 could be a promising lead molecule for the development of broadly neutralizing antibodies against SARS-CoV-2 and emerging variants. The nurse shark VNAR phage library offers a novel platform that can be used to rapidly isolate single-domain antibodies against emerging viral pathogens.

[end paper 21]

[begin paper 22]

PMCID = 11368078

Title = Screening and affinity optimization of single domain antibody targeting the SARS-CoV-2 nucleocapsid protein

The coronavirus disease 2019 (COVID-19) pandemic, which caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), lead to a crisis with devastating disasters to global public economy and health. Several studies suggest that the SARS-CoV-2 nucleocapsid protein (N protein) is one of uppermost structural constituents of SARS-CoV-2 and is relatively conserved which could become a specific diagnostic marker. In this study, eight single domain antibodies recognized the N protein specifically which were named pN01–pN08 were screened using human phage display library. According to multiple sequence alignment and molecular docking analyses, the interaction mechanism between antibody and N protein was predicted. ELISA results indicated pN01–pN08 with high affinity to protein N. To improve their efficacy, two fusion proteins were prepared and their affinity was tested. These finding showed that fusion proteins had higher affinity than single domain antibodies and will be used as diagnosis for the pandemic of SARS-CoV-2.

[end paper 22]

[begin paper 23]

PMCID = 10986514

Title = Nanobody engineering for SARS-CoV-2 neutralization and detection

In response to the ongoing COVID-19 pandemic, the quest for coronavirus inhibitors has inspired research on a variety of small proteins beyond conventional antibodies, including robust single-domain antibody fragments, i.e., “nanobodies.” Here, we explore the potential of nanobody engineering in the development of antivirals and diagnostic tools. Through fusion of nanobody domains that target distinct binding sites, we engineered multimodular nanobody constructs that neutralize wild-type SARS-CoV-2 and the Alpha and Delta variants at high potency, with IC50 values as low as 50 pM. Despite simultaneous binding to distinct epitopes, Beta and Omicron variants were more resistant to neutralization by the multimodular nanobodies, which highlights the importance of accounting for antigenic drift in the design of biologics. To further explore the applications of nanobody engineering in outbreak management, we present an assay based on fusions of nanobodies with fragments of NanoLuc luciferase that can detect sub-nanomolar quantities of the SARS-CoV-2 spike protein in a single step. Our work showcases the potential of nanobody engineering to combat emerging infectious diseases.

Nanobodies, small protein binders derived from the camelid antibody, are highly potent inhibitors of respiratory viruses that offer several advantages over conventional antibodies as candidates for specific therapies, including high stability and low production costs. In this work, we leverage the unique properties of nanobodies and apply them as building blocks for new therapeutic and diagnostic tools. We report ultra-potent SARS-CoV-2 inhibition by engineered nanobodies comprising multiple modules in structure-guided combinations and develop nanobodies that carry signal molecules, allowing rapid detection of the SARS-CoV-2 spike protein. Our results highlight the potential of engineered nanobodies in the development of effective countermeasures, both therapeutic and diagnostic, to manage outbreaks of emerging viruses.

[end paper 23]

[begin paper 24]

PMCID = 9792944

Title = Broadly neutralizing and protective nanobodies against SARS-CoV-2 Omicron subvariants BA.1, BA.2, and BA.4/5 and diverse sarbecoviruses

As SARS-CoV-2 Omicron and other variants of concern (VOCs) continue spreading worldwide, development of antibodies and vaccines to confer broad and protective activity is a global priority. Here, we report on the identification of a special group of nanobodies from immunized alpaca with potency against diverse VOCs including Omicron subvariants BA.1, BA.2 and BA.4/5, SARS-CoV-1, and major sarbecoviruses. Crystal structure analysis of one representative nanobody, 3-2A2-4, discovers a highly conserved epitope located between the cryptic and the outer face of the receptor binding domain (RBD), distinctive from the receptor ACE2 binding site. Cryo-EM and biochemical evaluation reveal that 3-2A2-4 interferes structural alteration of RBD required for ACE2 binding. Passive delivery of 3-2A2-4 protects K18-hACE2 mice from infection of authentic SARS-CoV-2 Delta and Omicron. Identification of these unique nanobodies will inform the development of next generation antibody therapies and design of pan-sarbecovirus vaccines.

The authors identify nanobodies from immunized alpaca with broadly neutralizing activity against SARS-CoV-1, SARS-CoV-2 variants, and major sarbecoviruses. One representative nanobody binds to a highly conserved epitope on RBD and protects K18-hACE2 mice from Omicron and Delta infection.

[end paper 24]

[begin paper 25]

PMCID = 7755911

Title = High affinity nanobodies block SARS-CoV-2 spike receptor binding domain interaction with human angiotensin converting enzyme

There are currently few approved effective treatments for SARS-CoV-2, the virus responsible for the COVID-19 pandemic. Nanobodies are 12–15 kDa single-domain antibody fragments that can be delivered by inhalation and are amenable to relatively inexpensive large scale production compared to other biologicals. We have isolated nanobodies that bind to the SARS-CoV-2 spike protein receptor binding domain and block spike protein interaction with the angiotensin converting enzyme 2 (ACE2) with 1–5 nM affinity. The lead nanobody candidate, NIH-CoVnb-112, blocks SARS-CoV-2 spike pseudotyped lentivirus infection of HEK293 cells expressing human ACE2 with an EC50 of 0.3 µg/mL. NIH-CoVnb-112 retains structural integrity and potency after nebulization. Furthermore, NIH-CoVnb-112 blocks interaction between ACE2 and several high affinity variant forms of the spike protein. These nanobodies and their derivatives have therapeutic, preventative, and diagnostic potential.

[end paper 25]

[begin paper 26]

PMCID = 9351521

Title = Correlation between the binding affinity and the conformational entropy of nanobody SARS-CoV-2 spike protein complexes

Understanding the structural principles that determine the binding affinity of nanobodies to the spike protein of severe acute respiratory syndrome coronavirus 2 has been difficult. We analyzed electron microscopy maps of nanobody-spike complexes and quantified the conformational entropy of binding. This informed the design of an engineered nanobody with improved binding to the spike protein. This result offers a guiding principle for the rational maturation of nanobodies directed against the spike. High-binding potency nanobodies have been shown to be effective in animal models; thus, this technology could have application in future pandemics.

Camelid single-domain antibodies, also known as nanobodies, can be readily isolated from naïve libraries for specific targets but often bind too weakly to their targets to be immediately useful. Laboratory-based genetic engineering methods to enhance their affinity, termed maturation, can deliver useful reagents for different areas of biology and potentially medicine. Using the receptor binding domain (RBD) of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein and a naïve library, we generated closely related nanobodies with micromolar to nanomolar binding affinities. By analyzing the structure–activity relationship using X-ray crystallography, cryoelectron microscopy, and biophysical methods, we observed that higher conformational entropy losses in the formation of the spike protein–nanobody complex are associated with tighter binding. To investigate this, we generated structural ensembles of the different complexes from electron microscopy maps and correlated the conformational fluctuations with binding affinity. This insight guided the engineering of a nanobody with improved affinity for the spike protein.

[end paper 26]

[begin paper 27]

PMCID = 8420576

Title = Neutralization of SARS‐CoV‐2 by highly potent, hyperthermostable, and mutation‐tolerant nanobodies

Monoclonal anti‐SARS‐CoV‐2 immunoglobulins represent a treatment option for COVID‐19. However, their production in mammalian cells is not scalable to meet the global demand. Single‐domain (VHH) antibodies (also called nanobodies) provide an alternative suitable for microbial production. Using alpaca immune libraries against the receptor‐binding domain (RBD) of the SARS‐CoV‐2 Spike protein, we isolated 45 infection‐blocking VHH antibodies. These include nanobodies that can withstand 95°C. The most effective VHH antibody neutralizes SARS‐CoV‐2 at 17–50 pM concentration (0.2–0.7 µg per liter), binds the open and closed states of the Spike, and shows a tight RBD interaction in the X‐ray and cryo‐EM structures. The best VHH trimers neutralize even at 40 ng per liter. We constructed nanobody tandems and identified nanobody monomers that tolerate the K417N/T, E484K, N501Y, and L452R immune‐escape mutations found in the Alpha, Beta, Gamma, Epsilon, Iota, and Delta/Kappa lineages. We also demonstrate neutralization of the Beta strain at low‐picomolar VHH concentrations. We further discovered VHH antibodies that enforce native folding of the RBD in the E. coli cytosol, where its folding normally fails. Such “fold‐promoting” nanobodies may allow for simplified production of vaccines and their adaptation to viral escape‐mutations.

Single‐domain camelid antibodies that neutralize a range of common and emerging immune‐escape mutant strains of SARS‐CoV‐2 may constitute an easily‐producible option for treatment of COVID‐19 patients.  

[end paper 27]

[begin paper 28]

PMCID = 9170159

Title = Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants

Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. In this report, we built dromedary camel (Camelus dromedarius) VHH phage libraries for the isolation of high-affinity nanobodies that broadly neutralize SARS-CoV-2 variants. Cryo-EM complex structures reveal that one dromedary camel VHH nanobody (8A2) binds the S1 subunit of the viral spike protein, and the other (7A3) targets a deeply buried region that uniquely extends to the S2 subunit beyond the S1 subunit. These nanobodies can protect mice from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting the therapeutic potential of these nanobodies against COVID-19. The dromedary camel VHH libraries could be helpful to isolate neutralizing nanobodies against future emerging viruses quickly.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike is a trimer of S1/S2 heterodimers with three receptor-binding domains (RBDs) at the S1 subunit for human angiotensin-converting enzyme 2 (hACE2). Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. To isolate high-affinity nanobodies, large libraries with great diversity are highly desirable. Dromedary camels (Camelus dromedarius) are natural reservoirs of coronaviruses like Middle East respiratory syndrome CoV (MERS-CoV) that are transmitted to humans. Here, we built large dromedary camel VHH phage libraries to isolate nanobodies that broadly neutralize SARS-CoV-2 variants. We isolated two VHH nanobodies, NCI-CoV-7A3 (7A3) and NCI-CoV-8A2 (8A2), which have a high affinity for the RBD via targeting nonoverlapping epitopes and show broad neutralization activity against SARS-CoV-2 and its emerging variants of concern. Cryoelectron microscopy (cryo-EM) complex structures revealed that 8A2 binds the RBD in its up mode with a long CDR3 loop directly involved in the ACE2 binding residues and that 7A3 targets a deeply buried region that uniquely extends from the S1 subunit to the apex of the S2 subunit regardless of the conformational state of the RBD. At a dose of ≥5 mg/kg, 7A3 efficiently protected transgenic mice expressing hACE2 from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting its therapeutic use against COVID-19 variants. The dromedary camel VHH phage libraries could be helpful as a unique platform ready for quickly isolating potent nanobodies against future emerging viruses.

[end paper 28]

[begin paper 29]

PMCID = 9869787

Title = A Multivalent and Thermostable Nanobody Neutralizing SARS-CoV-2 Omicron (B.1.1.529)

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) Omicron variants have risen to dominance, which contains far more mutations in the spike protein in comparison to previously reported variants, compromising the efficacy of most existing vaccines or therapeutic monoclonal antibodies. Nanobody screened from high-throughput naïve libraries is a potential candidate for developing preventive and therapeutic antibodies.

Four nanobodies specific to the SARS-CoV-2 wild-type receptor-binding domain (RBD) were screened from a naïve phage display library. Their affinity and neutralizing activity were evaluated by surface plasmon resonance assays, surrogate virus neutralization tests, and pseudovirus neutralization assays. Preliminary identification of the binding epitopes of nanobodies by peptide-based ELISA and competition assay. Then four multivalent nanobodies were engineered by attaching the monovalent nanobodies to an antibody-binding nanoplatform constructed based on the lumazine synthase protein cage nanoparticles isolated from the Aquifex aeolicus (AaLS). Finally, the differences in potency between the monovalent and multivalent nanobodies were compared using the same methods.

Three of the four specific nanobodies could maintain substantial inhibitory activity against the Omicron (B.1.1.529), of them, B-B2 had the best neutralizing activity against the Omicron (B.1.1.529) pseudovirus (IC50 = 1.658 μg/mL). The antiviral ability of multivalent nanobody LS-B-B2 was improved in the Omicron (B.1.1.529) pseudovirus assays (IC50 = 0.653 μg/mL). The results of peptide-based ELISA indicated that LS-B-B2 might react with the linear epitopes in the SARS-CoV-2 RBD conserved regions, which would clarify the mechanisms for the maintenance of potent neutralization of Omicron (B.1.1.529) preliminary.

Our study indicated that the AaLS could be used as an antibody-binding nanoplatform to present nanobodies on its surface and improve the potency of nanobodies. The multivalent nanobody LS-B-B2 may serve as a potential agent for the neutralization of SARS-CoV-2 variants.

[end paper 29]

[begin paper 30]

PMCID = 10444775

Title = Discovery and multimerization of cross-reactive single-domain antibodies against SARS-like viruses to enhance potency and address emerging SARS-CoV-2 variants

Coronaviruses have been the causative agent of three epidemics and pandemics in the past two decades, including the ongoing COVID-19 pandemic. A broadly-neutralizing coronavirus therapeutic is desirable not only to prevent and treat COVID-19, but also to provide protection for high-risk populations against future emergent coronaviruses. As all coronaviruses use spike proteins on the viral surface to enter the host cells, and these spike proteins share sequence and structural homology, we set out to discover cross-reactive biologic agents targeting the spike protein to block viral entry. Through llama immunization campaigns, we have identified single domain antibodies (VHHs) that are cross-reactive against multiple emergent coronaviruses (SARS-CoV, SARS-CoV-2, and MERS). Importantly, a number of these antibodies show sub-nanomolar potency towards all SARS-like viruses including emergent CoV-2 variants. We identified nine distinct epitopes on the spike protein targeted by these VHHs. Further, by engineering VHHs targeting distinct, conserved epitopes into multi-valent formats, we significantly enhanced their neutralization potencies compared to the corresponding VHH cocktails. We believe this approach is ideally suited to address both emerging SARS-CoV-2 variants during the current pandemic as well as potential future pandemics caused by SARS-like coronaviruses.

[end paper 30]

[begin paper 31]

PMCID = 8956255

Title = Multivariate mining of an alpaca immune repertoire identifies potent cross-neutralizing SARS-CoV-2 nanobodies

Conventional approaches to isolate and characterize nanobodies are laborious. We combine phage display, multivariate enrichment, next-generation sequencing, and a streamlined screening strategy to identify numerous anti–severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) nanobodies. We characterize their potency and specificity using neutralization assays and hydrogen/deuterium exchange mass spectrometry (HDX-MS). The most potent nanobodies bind to the receptor binding motif of the receptor binding domain (RBD), and we identify two exceptionally potent members of this category (with monomeric half-maximal inhibitory concentrations around 13 and 16 ng/ml). Other nanobodies bind to a more conserved epitope on the side of the RBD and are able to potently neutralize the SARS-CoV-2 founder virus (42 ng/ml), the Beta variant (B.1.351/501Y.V2) (35 ng/ml), and also cross-neutralize the more distantly related SARS-CoV-1 (0.46 μg/ml). The approach presented here is well suited for the screening of phage libraries to identify functional nanobodies for various biomedical and biochemical applications.

High-throughput mining of camelid immune repertoires identified broad and potent nanobodies from an immunized alpaca.

[end paper 31]

[begin paper 32]

PMCID = 8126837

Title = Nanobody cocktails potently neutralize SARS-CoV-2 D614G N501Y variant and protect mice

Neutralizing antibodies are important for immunity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and as therapeutics for the prevention and treatment of COVID-19. We identified high-affinity nanobodies against SARS-CoV-2 receptor-binding domain and found that nanobody cocktails consisting of two noncompeting nanobodies were able to block ACE2 engagement with RBD variants present in human populations and potently neutralize both wild-type SARS-CoV-2 and the N501Y D614G variant at low concentrations. Prophylactic administration of nanobody cocktails reduced viral loads in mice infected with the N501Y D614G SARS-CoV-2 virus, showing that nanobody cocktails are useful as prophylactic agents against SARS-CoV-2.

Neutralizing antibodies are important for immunity against SARS-CoV-2 and as therapeutics for the prevention and treatment of COVID-19. Here, we identified high-affinity nanobodies from alpacas immunized with coronavirus spike and receptor-binding domains (RBD) that disrupted RBD engagement with the human receptor angiotensin-converting enzyme 2 (ACE2) and potently neutralized SARS-CoV-2. Epitope mapping, X-ray crystallography, and cryo-electron microscopy revealed two distinct antigenic sites and showed two neutralizing nanobodies from different epitope classes bound simultaneously to the spike trimer. Nanobody-Fc fusions of the four most potent nanobodies blocked ACE2 engagement with RBD variants present in human populations and potently neutralized both wild-type SARS-CoV-2 and the N501Y D614G variant at concentrations as low as 0.1 nM. Prophylactic administration of either single nanobody-Fc or as mixtures reduced viral loads by up to 104-fold in mice infected with the N501Y D614G SARS-CoV-2 virus. These results suggest a role for nanobody-Fc fusions as prophylactic agents against SARS-CoV-2.

[end paper 32]

[begin paper 33]

PMCID = 8043454

Title = Nanobody Repertoires for Exposing Vulnerabilities of SARS-CoV-2

Despite the great promise of vaccines, the COVID-19 pandemic is ongoing and future serious outbreaks are highly likely, so that multi-pronged containment strategies will be required for many years. Nanobodies are the smallest naturally occurring single domain antigen binding proteins identified to date, possessing numerous properties advantageous to their production and use. We present a large repertoire of high affinity nanobodies against SARS-CoV-2 Spike protein with excellent kinetic and viral neutralization properties, which can be strongly enhanced with oligomerization. This repertoire samples the epitope landscape of the Spike ectodomain inside and outside the receptor binding domain, recognizing a multitude of distinct epitopes and revealing multiple neutralization targets of pseudoviruses and authentic SARS-CoV-2, including in primary human airway epithelial cells. Combinatorial nanobody mixtures show highly synergistic activities, and are resistant to mutational escape and emerging viral variants of concern. These nanobodies establish an exceptional resource for superior COVID-19 prophylactics and therapeutics.

[end paper 33]

[begin paper 34]

PMCID = 10268213

Title = Fully synthetic platform to rapidly generate tetravalent bispecific nanobody–based immunoglobulins

Nanobodies are a promising class of biologics that can be used to prevent or treat viral infections. Here, we describe the production and validation of a discovery library that produces single-domain nanobodies using an engineered human antibody variable gene segment. As a test case, anti-SARS-CoV-2 nanobodies were isolated from this library and pairs of complementary nanobodies were incorporated into an antibody-like molecule that targets the receptor-binding domain using a biparatopic mode of engagement. This modular bispecific format enabled the rapid testing of nanobody pairs, and we show that incorporating pairs of nanobodies with different specificities can have synergistic effects on neutralization breadth and potency.

Nanobodies bind a target antigen with a kinetic profile similar to a conventional antibody, but exist as a single heavy chain domain that can be readily multimerized to engage antigen via multiple interactions. Presently, most nanobodies are produced by immunizing camelids; however, platforms for animal-free production are growing in popularity. Here, we describe the development of a fully synthetic nanobody library based on an engineered human VH3-23 variable gene and a multispecific antibody-like format designed for biparatopic target engagement. To validate our library, we selected nanobodies against the SARS-CoV-2 receptor–binding domain and employed an on-yeast epitope binning strategy to rapidly map the specificities of the selected nanobodies. We then generated antibody-like molecules by replacing the VH and VL domains of a conventional antibody with two different nanobodies, designed as a molecular clamp to engage the receptor-binding domain biparatopically. The resulting bispecific tetra-nanobody immunoglobulins neutralized diverse SARS-CoV-2 variants with potencies similar to antibodies isolated from convalescent donors. Subsequent biochemical analyses confirmed the accuracy of the on-yeast epitope binning and structures of both individual nanobodies, and a tetra-nanobody immunoglobulin revealed that the intended mode of interaction had been achieved. This overall workflow is applicable to nearly any protein target and provides a blueprint for a modular workflow for the development of multispecific molecules.

[end paper 34]

[begin paper 35]

PMCID = 11554226

Title = Super broad and protective nanobodies against Sarbecoviruses including SARS-CoV-1 and the divergent SARS-CoV-2 subvariant KP.3.1.1

The ongoing evolution and immune escape of SARS-CoV-2, alongside the potential threat of SARS-CoV-1 and other sarbecoviruses, underscore the urgent need for effective strategies against their infection and transmission. This study highlights the discovery of nanobodies from immunized alpacas, which demonstrate exceptionally broad and potent neutralizing capabilities against the recently emerged and more divergent SARS-CoV-2 Omicron subvariants including JD.1.1, JN.1, KP.3, KP.3.1.1, as well as SARS-CoV-1 and coronaviruses from bats and pangolins utilizing receptor ACE2. Among these, Tnb04-1 emerges as the most broad and potent, binding to a conserved hydrophobic pocket in the spike’s receptor-binding domain, distinct from the ACE2 binding site. This interaction disrupts the formation of a proteinase K-resistant core, crucial for viral-cell fusion. Notably, intranasal administration of Tnb04-1 in Syrian hamsters effectively prevented respiratory infection and transmission of the authentic Omicron XBB.1.5 subvariant. Thus, Thb04-1 holds promise in combating respiratory acquisition and transmission of diverse sarbecoviruses.

As SARS-CoV-2 evolves and evades antibody immunity, new antibody intervention strategies are urgently needed to prevent viral acquisition and transmission. We identified a cluster of nanobodies from immunized alpacas with broad and potent neutralizing activity against all major SARS-CoV-2 variants, SARS-CoV-1, and ACE2-utilizing coronaviruses from bats and pangolins. Tnb04-1 was the most effective, binding to a conserved epitope in spike and providing strong protection against contact and respiratory infection of Omicron XBB.1.5 in Syrian hamsters. Tnb04-1 shows promise for next-generation antibody interventions against diverse sarbecoviruses.

[end paper 35]

[begin paper 36]

PMCID = 9365158

Title = A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike

Neutralizing antibodies targeting the SARS-CoV-2 spike protein have shown a great preventative/therapeutic potential. Here, we report a rapid and efficient strategy for the development and design of SARS-CoV-2 neutralizing humanized nanobody constructs with sub-nanomolar affinities and nanomolar potencies. CryoEM-based structural analysis of the nanobodies in complex with spike revealed two distinct binding modes. The most potent nanobody, RBD-1-2G(NCATS-BL8125), tolerates the N501Y RBD mutation and remains capable of neutralizing the B.1.1.7 (Alpha) variant. Molecular dynamics simulations provide a structural basis for understanding the neutralization process of nanobodies exclusively focused on the spike-ACE2 interface with and without the N501Y mutation on RBD. A primary human airway air-lung interface (ALI) ex vivo model showed that RBD-1-2G-Fc antibody treatment was effective at reducing viral burden following WA1 and B.1.1.7 SARS-CoV-2 infections. Therefore, this presented strategy will serve as a tool to mitigate the threat of emerging SARS-CoV-2 variants.

[end paper 36]

[begin paper 37]

PMCID = 10892724

Title = SARS-CoV-2 Specific Nanobodies Neutralize Different Variants of Concern and Reduce Virus Load in the Brain of h-ACE2 Transgenic Mice

Since the beginning of the COVID-19 pandemic, there has been a significant need to develop antivirals and vaccines to combat the disease. In this work, we developed llama-derived nanobodies (Nbs) directed against the receptor binding domain (RBD) and other domains of the Spike (S) protein of SARS-CoV-2. Most of the Nbs with neutralizing properties were directed to RBD and were able to block S-2P/ACE2 interaction. Three neutralizing Nbs recognized the N-terminal domain (NTD) of the S-2P protein. Intranasal administration of Nbs induced protection ranging from 40% to 80% after challenge with the WA1/2020 strain in k18-hACE2 transgenic mice. Interestingly, protection was associated with a significant reduction in virus replication in nasal turbinates and a reduction in virus load in the brain. Employing pseudovirus neutralization assays, we identified Nbs with neutralizing capacity against the Alpha, Beta, Delta, and Omicron variants, including a Nb capable of neutralizing all variants tested. Furthermore, cocktails of different Nbs performed better than individual Nbs at neutralizing two Omicron variants (B.1.529 and BA.2). Altogether, the data suggest the potential of SARS-CoV-2 specific Nbs for intranasal treatment of COVID-19 encephalitis.

[end paper 37]

[begin paper 38]

PMCID = 8908544

Title = Structural basis of nanobodies neutralizing SARS-CoV-2 variants

Because of the evolutionary variants of SARS-CoV-2, development of broad-spectrum neutralizing antibodies resilient to virus escape is urgently needed. We identified a group of high-affinity nanobodies from camels immunized with receptor-binding domain (RBD) of SARS-CoV-2 spike protein and resolved the structures of two non-competing nanobodies (NB1A7 and NB1B11) in complex with RBD using X-ray crystallography. The structures show that NB1A7 targets the highly conserved cryptic epitope shared by SARS-CoV-2 variants and some other coronaviruses and blocks ACE2 receptor attachment of the spike protein, and NB1B11 epitope overlaps with the contacting surface of ACE2 and is different from the binding site of NB1A7. These two nanobodies were covalently linked into multivalent and bi-paratopic formats, which significantly improved the avidity and neutralization potency and may further inhibit viral escape. The results contribute to the structure-guided design of antibodies against future variants of SARS-CoV-2 virus to combat coronavirus epidemics and pandemics.

Shi et al. identify two neutralizing nanobodies against SARS-CoV-2. Crystal structures reveal how they prevent the virus and its variants from entering cells. The combination of two nanobodies significantly improves neutralization potency, which is a potential therapeutic candidate to reduce the probability of the emergence of escape variants of SARS-CoV-2.

[end paper 38]

[begin paper 39]

PMCID = 7870875

Title = Potent neutralization of clinical isolates of SARS-CoV-2 D614 and G614 variants by a monomeric, sub-nanomolar affinity nanobody

Despite unprecedented global efforts to rapidly develop SARS-CoV-2 treatments, in order to reduce the burden placed on health systems, the situation remains critical. Effective diagnosis, treatment, and prophylactic measures are urgently required to meet global demand: recombinant antibodies fulfill these requirements and have marked clinical potential. Here, we describe the fast-tracked development of an alpaca Nanobody specific for the receptor-binding-domain (RBD) of the SARS-CoV-2 Spike protein with potential therapeutic applicability. We present a rapid method for nanobody isolation that includes an optimized immunization regimen coupled with VHH library E. coli surface display, which allows single-step selection of Nanobodies using a simple density gradient centrifugation of the bacterial library. The selected single and monomeric Nanobody, W25, binds to the SARS-CoV-2 S RBD with sub-nanomolar affinity and efficiently competes with ACE-2 receptor binding. Furthermore, W25 potently neutralizes SARS-CoV-2 wild type and the D614G variant with IC50 values in the nanomolar range, demonstrating its potential as antiviral agent.

[end paper 39]

[begin paper 40]

PMCID = 8625641

Title = Structural Biology of Nanobodies against the Spike Protein of SARS-CoV-2

Nanobodies are 130 amino acid single-domain antibodies (VHH) derived from the unique heavy-chain-only subclass of Camelid immunogloblins. Their small molecular size, facile expression, high affinity and stability have combined to make them unique targeting reagents with numerous applications in the biomedical sciences. The first nanobody agent has now entered the clinic as a treatment against a blood disorder. The spread of the SARS-CoV-2 virus has seen the global scientific endeavour work to accelerate the development of technologies to try to defeat a pandemic that has now killed over four million people. In a remarkably short period of time, multiple studies have reported nanobodies directed against the viral Spike protein. Several agents have been tested in culture and demonstrate potent neutralisation of the virus or pseudovirus. A few agents have completed animal trials with very encouraging results showing their potential for treating infection. Here, we discuss the structural features that guide the nanobody recognition of the receptor binding domain of the Spike protein of SARS-CoV-2.

[end paper 40]

[begin paper 41]

PMCID = 8097376

Title = NeutrobodyPlex—monitoring SARS‐CoV‐2 neutralizing immune responses using nanobodies

In light of the COVID‐19 pandemic, there is an ongoing need for diagnostic tools to monitor the immune status of large patient cohorts and the effectiveness of vaccination campaigns. Here, we present 11 unique nanobodies (Nbs) specific for the SARS‐CoV‐2 spike receptor‐binding domain (RBD), of which 8 Nbs potently inhibit the interaction of RBD with angiotensin‐converting enzyme 2 (ACE2) as the major viral docking site. Following detailed epitope mapping and structural analysis, we select two inhibitory Nbs, one of which binds an epitope inside and one of which binds an epitope outside the RBD:ACE2 interface. Based on these, we generate a biparatopic nanobody (bipNb) with viral neutralization efficacy in the picomolar range. Using bipNb as a surrogate, we establish a competitive multiplex binding assay (“NeutrobodyPlex”) for detailed analysis of the presence and performance of neutralizing RBD‐binding antibodies in serum of convalescent or vaccinated patients. We demonstrate that NeutrobodyPlex enables high‐throughput screening and detailed analysis of neutralizing immune responses in infected or vaccinated individuals, to monitor immune status or to guide vaccine design.

This study describes the generation of biparatopic nanobodies as surrogates for monitoring neutralizing immune responses in SARS‐CoV‐2 infected individuals. NeutrobodyPlex enables high‐throughput screening and detailed analyses of infected or vaccinated individuals.  

[end paper 41]

[begin paper 42]

PMCID = 8260353

Title = Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants

Since the start of the COVID-19 pandemic, SARS-CoV-2 has caused millions of deaths worldwide. Although a number of vaccines have been deployed, the continual evolution of the receptor-binding domain (RBD) of the virus has challenged their efficacy. In particular, the emerging variants B.1.1.7, B.1.351 and P.1 (first detected in the UK, South Africa and Brazil, respectively) have compromised the efficacy of sera from patients who have recovered from COVID-19 and immunotherapies that have received emergency use authorization. One potential alternative to avert viral escape is the use of camelid VHHs (variable heavy chain domains of heavy chain antibody (also known as nanobodies)), which can recognize epitopes that are often inaccessible to conventional antibodies. Here, we isolate anti-RBD nanobodies from llamas and from mice that we engineered to produce VHHs cloned from alpacas, dromedaries and Bactrian camels. We identified two groups of highly neutralizing nanobodies. Group 1 circumvents antigenic drift by recognizing an RBD region that is highly conserved in coronaviruses but rarely targeted by human antibodies. Group 2 is almost exclusively focused to the RBD–ACE2 interface and does not neutralize SARS-CoV-2 variants that carry E484K or N501Y substitutions. However, nanobodies in group 2 retain full neutralization activity against these variants when expressed as homotrimers, and—to our knowledge—rival the most potent antibodies against SARS-CoV-2 that have been produced to date. These findings suggest that multivalent nanobodies overcome SARS-CoV-2 mutations through two separate mechanisms: enhanced avidity for the ACE2-binding domain and recognition of conserved epitopes that are largely inaccessible to human antibodies. Therefore, although new SARS-CoV-2 mutants will continue to emerge, nanobodies represent promising tools to prevent COVID-19 mortality when vaccines are compromised.

Multivalent nanobodies against SARS-CoV-2 from mice engineered to produce camelid nanobodies recognize conserved epitopes that are inaccessible to human antibodies and show promise as a strategy for dealing with viral escape mutations.

[end paper 42]

[begin paper 43]

PMCID = 8223476

Title = Directed evolution of potent neutralizing nanobodies against SARS-CoV-2 using CDR-swapping mutagenesis

There is widespread interest in facile methods for generating potent neutralizing antibodies, nanobodies, and other affinity proteins against SARS-CoV-2 and related viruses to address current and future pandemics. While isolating antibodies from animals and humans are proven approaches, these methods are limited to the affinities, specificities, and functional activities of antibodies generated by the immune system. Here we report a surprisingly simple directed evolution method for generating nanobodies with high affinities and neutralization activities against SARS-CoV-2. We demonstrate that complementarity-determining region swapping between low-affinity lead nanobodies, which we discovered unintentionally but find is simple to implement systematically, results in matured nanobodies with unusually large increases in affinity. Importantly, the matured nanobodies potently neutralize both SARS-CoV-2 pseudovirus and live virus, and possess drug-like biophysical properties. We expect that our methods will improve in vitro nanobody discovery and accelerate the generation of potent neutralizing nanobodies against diverse coronaviruses.

Zupancic et al. report potent neutralizing nanobodies against SARS-CoV-2. They demonstrate an approach that involves swapping the complementarity-determining regions of low-affinity clones to generate matured nanobodies with large increases in affinity and neutralization activity.

[end paper 43]

[begin paper 44]

PMCID = 8448731

Title = A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies

Antibody engineering technologies face increasing demands for speed, reliability and scale. We develop CeVICA, a cell-free nanobody engineering platform that uses ribosome display for in vitro selection of nanobodies from a library of 1011 randomized sequences. We apply CeVICA to engineer nanobodies against the Receptor Binding Domain (RBD) of SARS-CoV-2 spike protein and identify >800 binder families using a computational pipeline based on CDR-directed clustering. Among 38 experimentally-tested families, 30 are true RBD binders and 11 inhibit SARS-CoV-2 pseudotyped virus infection. Affinity maturation and multivalency engineering increase nanobody binding affinity and yield a virus neutralizer with picomolar IC50. Furthermore, the capability of CeVICA for comprehensive binder prediction allows us to validate the fitness of our nanobody library. CeVICA offers an integrated solution for rapid generation of divergent synthetic nanobodies with tunable affinities in vitro and may serve as the basis for automated and highly parallel nanobody engineering.

Faster, higher throughput antibody engineering methods are needed. Here the authors present CeVICA, a cell-free nanobody engineering platform using ribosome display and computational clustering analysis for in vitro selection; they use this to develop nanobodies against the RBD of SARS-CoV-2 spike protein.

[end paper 44]

[begin paper 45]

PMCID = 10688893

Title = Development of a bispecific nanobody conjugate broadly neutralizes diverse SARS-CoV-2 variants and structural basis for its broad neutralization

The continuous emergence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants with increased transmissibility and profound immune-escape capacity makes it an urgent need to develop broad-spectrum therapeutics. Nanobodies have recently attracted extensive attentions due to their excellent biochemical and binding properties. Here, we report two high-affinity nanobodies (Nb-015 and Nb-021) that target non-overlapping epitopes in SARS-CoV-2 S-RBD. Both nanobodies could efficiently neutralize diverse viruses of SARS-CoV-2. The neutralizing mechanisms for the two nanobodies are further delineated by high-resolution nanobody/S-RBD complex structures. In addition, an Fc-based tetravalent nanobody format is constructed by combining Nb-015 and Nb-021. The resultant nanobody conjugate, designated as Nb-X2-Fc, exhibits significantly enhanced breadth and potency against all-tested SARS-CoV-2 variants, including Omicron sub-lineages. These data demonstrate that Nb-X2-Fc could serve as an effective drug candidate for the treatment of SARS-CoV-2 infection, deserving further in-vivo evaluations in the future.

The pandemic of SARS-CoV-2 has led to a severe public health crisis. In view of the wide clinical-applications of antibody drugs, it has always been a research focus to identify new antibodies and/or nanobodies with anti-viral neutralizing activity. In this study, we identified two neutralizing nanobodies (Nb-015 and Nb-021) that target non-overlapping epitopes in SARS-CoV-2 S-RBD with high affinity. Both nanobodies could broadly neutralize diverse SARS-CoV-2 variants. Furthermore, we revealed two distinct neutralization mechanisms of Nb-015 and Nb-021 by high-resolution nanobody/S-RBD complex structures. Eventually, we engineered a bispecific nanobody conjugate which exhibits excellent inhibitory activity against SARS-CoV-2, including the original strain and the VOI and VOC variants. We believe that the nanobody conjugate identified in this study could serve as an effective anti-viral drug reserve for potential treatment of COVID-19.

[end paper 45]

[begin paper 46]

PMCID = 10865538

Title = Single-domain antibodies against SARS-CoV-2 RBD from a two-stage phage screening of universal and focused synthetic libraries

Coronavirus disease 2019 (COVID-19) is an evolving global pandemic, and nanobodies, as well as other single-domain antibodies (sdAbs), have been recognized as a potential diagnostic and therapeutic tool for infectious diseases. High-throughput screening techniques such as phage display have been developed as an alternative to in vivo immunization for the discovery of antibody-like target-specific binders.

We designed and constructed a highly diverse synthetic phage library sdAb-U (single-domain Antibody - Universal library ) based on a human framework. The SARS-CoV-2 receptor-binding domain (RBD) was expressed and purified. The universal library sdAb-U was panned against the RBD protein target for two rounds, followed by monoclonal phage ELISA (enzyme-linked immunosorbent assay) to identify RBD-specific binders (the first stage). High-affinity binders were sequenced and the obtained CDR1 and CDR2 sequences were combined with fully randomized CDR3 to construct a targeted (focused) phage library sdAb-RBD, for subsequent second-stage phage panning (also two rounds) and screening. Then, sequences with high single-to-background ratios in phage ELISA were selected for expression. The binding affinities of sdAbs to RBD were measured by an ELISA-based method. In addition, we conducted competition ELISA (using ACE2 ectodomain S19-D615) and SARS-CoV-2 pseudovirus neutralization assays for the high-affinity RBD-binding sdAb39.

Significant enrichments were observed in both the first-stage (universal library) and the second-stage (focused library) phage panning. Five RBD-specific binders were identified in the first stage with high ELISA signal-to-background ratios. In the second stage, we observed a much higher possibility of finding RBD-specific clones in phage ELISA. Among 45 selected RBD-positive sequences, we found eight sdAbs can be well expressed, and five of them show high-affinity to RBD (EC50 < 100nM). We finally found that sdAb39 (EC50 ~ 4nM) can compete with ACE2 for binding to RBD.

Overall, this two-stage strategy of synthetic phage display libraries enables rapid selection of SARS-CoV-2 RBD sdAb with potential therapeutic activity, and this two-stage strategy can potentially be used for rapid discovery of sdAbs against other targets.

The online version contains supplementary material available at 10.1186/s12879-024-09022-8.

[end paper 46]

[begin paper 47]

PMCID = 10588750

Title = A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern

The emergence of the coronavirus disease 2019 (COVID-19) pandemic and the new severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) variants of concern (VOCs) requires the continuous development of safe, effective, and affordable prevention and therapeutics. Nanobodies have demonstrated antiviral activity against a variety of viruses, providing a new candidate for the prevention and treatment of SARS-CoV-2 and its variants.

SARS-CoV-2 glycoprotein spike 1 subunit (S1) was selected as the target antigen for nanobody screening of a naïve phage display library. We obtained a nanobody, named Nb-H6, and then determined its affinity, inhibition, and stability by ELISA, Competitive ELISA, and Biolayer Interferometry (BLI). Infection assays of authentic and pseudotyped SARS-CoV-2 were performed to evaluate the neutralization of Nb-H6. The structure and mechanism of action were investigated by AlphaFold, docking, and residue mutation assays.

We isolated and characterized a nanobody, Nb-H6, which exhibits a broad affinity for S1 and the receptor binding domain (RBD) of SARS-CoV-2, or Alpha (B.1.1.7), Delta (B.1.617.2), Lambda (C.37), and Omicron (BA.2 and BA.5), and blocks receptor angiotensin-converting enzyme 2 (ACE2) binding. Moreover, Nb-H6 can retain its binding capability after pH or thermal treatment and effectively neutralize both pseudotyped and authentic SARS-CoV-2, as well as VOC Alpha (B.1.1.7), Delta (B.1.617.2), and Omicron (BA.2 and BA.5) pseudoviruses. We also confirmed that Nb-H6 binds two distinct amino acid residues of the RBD, preventing SARS-CoV-2 from interacting with the host receptor.

Our study highlights a novel nanobody, Nb-H6, that may be useful therapeutically in SARS-CoV-2 and VOC outbreaks and pandemics. These findings also provide a molecular foundation for further studies into how nanobodies neutralize SARS-CoV-2 and variants and imply potential therapeutic targets for the treatment of COVID-19.

[end paper 47]

[begin paper 48]

PMCID = 8458290

Title = A potent SARS-CoV-2 neutralising nanobody shows therapeutic efficacy in the Syrian golden hamster model of COVID-19

SARS-CoV-2 remains a global threat to human health particularly as escape mutants emerge. There is an unmet need for effective treatments against COVID-19 for which neutralizing single domain antibodies (nanobodies) have significant potential. Their small size and stability mean that nanobodies are compatible with respiratory administration. We report four nanobodies (C5, H3, C1, F2) engineered as homotrimers with pmolar affinity for the receptor binding domain (RBD) of the SARS-CoV-2 spike protein. Crystal structures show C5 and H3 overlap the ACE2 epitope, whilst C1 and F2 bind to a different epitope. Cryo Electron Microscopy shows C5 binding results in an all down arrangement of the Spike protein. C1, H3 and C5 all neutralize the Victoria strain, and the highly transmissible Alpha (B.1.1.7 first identified in Kent, UK) strain and C1 also neutralizes the Beta (B.1.35, first identified in South Africa). Administration of C5-trimer via the respiratory route showed potent therapeutic efficacy in the Syrian hamster model of COVID-19 and separately, effective prophylaxis. The molecule was similarly potent by intraperitoneal injection.

Neutralizing nanobodies (Nb) are of considerable interest as therapeutic agents for COVID-19 treatment. Here, the authors functionally and structurally characterize Nbs that bind with high affinity to the receptor binding domain of the SARS-CoV-2 spike protein and show that an engineered homotrimeric Nb prevents disease progression in a Syrian hamster model of COVID-19 when administered intranasally.

[end paper 48]

[begin paper 49]

PMCID = 10369967

Title = Nanobody repertoire generated against the spike protein of ancestral SARS-CoV-2 remains efficacious against the rapidly evolving virus

To date, all major modes of monoclonal antibody therapy targeting SARS-CoV-2 have lost significant efficacy against the latest circulating variants. As SARS-CoV-2 omicron sublineages account for over 90% of COVID-19 infections, evasion of immune responses generated by vaccination or exposure to previous variants poses a significant challenge. A compelling new therapeutic strategy against SARS-CoV-2 is that of single domain antibodies, termed nanobodies, which address certain limitations of monoclonal antibodies. Here we demonstrate that our high-affinity nanobody repertoire, generated against wild-type SARS-CoV-2 spike protein, remains effective against variants of concern, including omicron BA.4/BA.5; a subset is predicted to counter resistance in emerging XBB and BQ.1.1 sublineages. Furthermore, we reveal the synergistic potential of nanobody cocktails in neutralizing emerging variants. Our study highlights the power of nanobody technology as a versatile therapeutic and diagnostic tool to combat rapidly evolving infectious diseases such as SARS-CoV-2.

[end paper 49]

[begin paper 50]

PMCID = 7844813

Title = Development of multivalent nanobodies blocking SARS-CoV-2 infection by targeting RBD of spike protein

The outbreak and pandemic of coronavirus SARS-CoV-2 caused significant threaten to global public health and economic consequences. It is extremely urgent that global people must take actions to develop safe and effective preventions and therapeutics. Nanobodies, which are derived from single‑chain camelid antibodies, had shown antiviral properties in various challenge viruses. In this study, multivalent nanobodies with high affinity blocking SARS-CoV-2 spike interaction with ACE2 protein were developed.

Totally, four specific nanobodies against spike protein and its RBD domain were screened from a naïve VHH library. Among them, Nb91-hFc and Nb3-hFc demonstrated antiviral activity by neutralizing spike pseudotyped viruses in vitro. Subsequently, multivalent nanobodies were constructed to improve the neutralizing capacity. As a result, heterodimer nanobody Nb91-Nb3-hFc exhibited the strongest RBD-binding affinity and neutralizing ability against SARS-CoV-2 pseudoviruses with an IC50 value at approximately 1.54 nM.

The present study indicated that naïve VHH library could be used as a potential resource for rapid acquisition and exploitation of antiviral nanobodies. Heterodimer nanobody Nb91-Nb3-hFc may serve as a potential therapeutic agent for the treatment of COVID-19.

[end paper 50]

[begin paper 51]

PMCID = 9884143

Title = Expanding and improving nanobody repertoires using a yeast display method: Targeting SARS-CoV-2

COVID-19, caused by the coronavirus SARS-CoV-2, represents a serious worldwide health issue, with continually emerging new variants challenging current therapeutics. One promising alternate therapeutic avenue is represented by nanobodies, small single-chain antibodies derived from camelids with numerous advantageous properties and the potential to neutralize the virus. For identification and characterization of a broad spectrum of anti–SARS-CoV-2 Spike nanobodies, we further optimized a yeast display method, leveraging a previously published mass spectrometry-based method, using B-cell complementary DNA from the same immunized animals as a source of VHH sequences. Yeast display captured many of the sequences identified by the previous approach, as well as many additional sequences that proved to encode a large new repertoire of nanobodies with high affinities and neutralization activities against different SARS-CoV-2 variants. We evaluated DNA shuffling applied to the three complementarity-determining regions of antiviral nanobodies. The results suggested a surprising degree of modularity to complementarity-determining region function. Importantly, the yeast display approach applied to nanobody libraries from immunized animals allows parallel interrogation of a vast number of nanobodies. For example, we employed a modified yeast display to carry out massively parallel epitope binning. The current yeast display approach proved comparable in efficiency and specificity to the mass spectrometry–based approach, while requiring none of the infrastructure and expertise required for that approach, making these highly complementary approaches that together appear to comprehensively explore the paratope space. The larger repertoires produced maximize the likelihood of discovering broadly specific reagents and those that powerfully synergize in mixtures.

[end paper 51]

[begin paper 52]

PMCID = 8748511

Title = A bispecific monomeric nanobody induces spike trimer dimers and neutralizes SARS-CoV-2 in vivo

Antibodies binding to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike have therapeutic promise, but emerging variants show the potential for virus escape. This emphasizes the need for therapeutic molecules with distinct and novel neutralization mechanisms. Here we describe the isolation of a nanobody that interacts simultaneously with two RBDs from different spike trimers of SARS-CoV-2, rapidly inducing the formation of spike trimer–dimers leading to the loss of their ability to attach to the host cell receptor, ACE2. We show that this nanobody potently neutralizes SARS-CoV-2, including the beta and delta variants, and cross-neutralizes SARS-CoV. Furthermore, we demonstrate the therapeutic potential of the nanobody against SARS-CoV-2 and the beta variant in a human ACE2 transgenic mouse model. This naturally elicited bispecific monomeric nanobody establishes an uncommon strategy for potent inactivation of viral antigens and represents a promising antiviral against emerging SARS-CoV-2 variants.

Here, the authors isolate and characterize a bispecific monomeric nanobody that induces dimerization of SARS-CoV-2 spike trimers, neutralizes variants of concerns as well as SARS-CoV, and inhibits SARS-CoV-2 infection in mice.

[end paper 52]

[begin paper 53]

PMCID = 11630572

Title = Discovery of Nanosota-9 as anti-Omicron nanobody therapeutic candidate

Omicron subvariants of SARS-CoV-2 continue to pose a significant global health threat. Nanobodies, single-domain antibodies derived from camelids, are promising therapeutic tools against pandemic viruses due to their favorable properties. In this study, we identified a novel nanobody, Nanosota-9, which demonstrates high potency against a wide range of Omicron subvariants both in vitro and in a mouse model. Cryo-EM data revealed that Nanosota-9 neutralizes Omicron through a unique mechanism: two Nanosota-9 molecules crosslink two receptor-binding domains (RBDs) of the trimeric Omicron spike protein, preventing the RBDs from binding to the ACE2 receptor. This mechanism explains its strong anti-Omicron potency. Additionally, the Nanosota-9 binding epitopes on the spike protein are relatively conserved among Omicron subvariants, contributing to its broad anti-Omicron spectrum. Combined with our recently developed structure-guided in vitro evolution approach for nanobodies, Nanosota-9 has the potential to serve as the foundation for a superior anti-Omicron therapeutic.

Omicron continues to evolve and spread among humans, making the development of effective and broad-spectrum treatments a top priority for global health. Nanobodies, which are single-domain antibodies derived from camelids, show great promise due to their small size and ability to target conserved, hidden regions on the virus. In this study, we identified a novel nanobody called Nanosota-9, which effectively inhibits most major Omicron subvariants. Our cryo-EM analysis revealed a unique structural mechanism by which Nanosota-9 neutralizes Omicron. Like many other anti-SARS-CoV-2 antibodies and nanobodies, Nanosota-9 binds to the receptor-binding domain (RBD) of Omicron spike proteins, blocking receptor binding. However, unlike others, two Nanosota-9 molecules crosslink two RBDs of the trimeric spike, enhancing its anti-Omicron potency. One of the two binding sites on the RBD that Nanosota-9 targets significantly overlaps with the receptor-binding site and is relatively conserved, while the other site is fully conserved, expanding its anti-Omicron spectrum. Conventional antibodies cannot access the same binding sites on the spike protein as Nanosota-9, highlighting its unique neutralization mechanism. Combined with our newly developed structure-guided in vitro evolution approach for nanobodies, Nanosota-9 has strong potential as a superior anti-Omicron therapeutic.

[end paper 53]

[begin paper 54]

PMCID = 8354634

Title = The development of Nanosota-1 as anti-SARS-CoV-2 nanobody drug candidates

Combating the COVID-19 pandemic requires potent and low-cost therapeutics. We identified a series of single-domain antibodies (i.e., nanobody), Nanosota-1, from a camelid nanobody phage display library. Structural data showed that Nanosota-1 bound to the oft-hidden receptor-binding domain (RBD) of SARS-CoV-2 spike protein, blocking viral receptor angiotensin-converting enzyme 2 (ACE2). The lead drug candidate possessing an Fc tag (Nanosota-1C-Fc) bound to SARS-CoV-2 RBD ~3000 times more tightly than ACE2 did and inhibited SARS-CoV-2 pseudovirus ~160 times more efficiently than ACE2 did. Administered at a single dose, Nanosota-1C-Fc demonstrated preventive and therapeutic efficacy against live SARS-CoV-2 infection in both hamster and mouse models. Unlike conventional antibodies, Nanosota-1C-Fc was produced at high yields in bacteria and had exceptional thermostability. Pharmacokinetic analysis of Nanosota-1C-Fc documented an excellent in vivo stability and a high tissue bioavailability. As effective and inexpensive drug candidates, Nanosota-1 may contribute to the battle against COVID-19.

[end paper 54]

[begin paper 55]

PMCID = 9850341

Title = The Potential of Nanobodies for COVID-19 Diagnostics and Therapeutics

The infectious severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) is the causative agent for coronavirus disease 2019 (COVID-19). Globally, there have been millions of infections and fatalities. Unfortunately, the virus has been persistent and a contributing factor is the emergence of several variants. The urgency to combat COVID-19 led to the identification/development of various diagnosis (polymerase chain reaction and antigen tests) and treatment (repurposed drugs, convalescent plasma, antibodies and vaccines) options. These treatments may treat mild symptoms and decrease the risk of life-threatening disease. Although these options have been fairly beneficial, there are some challenges and limitations, such as cost of tests/drugs, specificity, large treatment dosages, intravenous administration, need for trained personal, lengthy production time, high manufacturing costs, and limited availability. Therefore, the development of more efficient COVID-19 diagnostic and therapeutic options are vital. Nanobodies (Nbs) are novel monomeric antigen-binding fragments derived from camelid antibodies. Advantages of Nbs include low immunogenicity, high specificity, stability and affinity. These characteristics allow for rapid Nb generation, inexpensive large-scale production, effective storage, and transportation, which is essential during pandemics. Additionally, the potential aerosolization and inhalation delivery of Nbs allows for targeted treatment delivery as well as patient self-administration. Therefore, Nbs are a viable option to target SARS-CoV-2 and overcome COVID-19. In this review we discuss (1) COVID-19; (2) SARS-CoV-2; (3) the present conventional COVID-19 diagnostics and therapeutics, including their challenges and limitations; (4) advantages of Nbs; and (5) the numerous Nbs generated against SARS-CoV-2 as well as their diagnostic and therapeutic potential.

[end paper 55]

[begin paper 56]

PMCID = 11140111

Title = Single domain antibodies from camelids in the treatment of microbial infections

Infectious diseases continue to pose significant global health challenges. In addition to the enduring burdens of ailments like malaria and HIV, the emergence of nosocomial outbreaks driven by antibiotic-resistant pathogens underscores the ongoing threats. Furthermore, recent infectious disease crises, exemplified by the Ebola and SARS-CoV-2 outbreaks, have intensified the pursuit of more effective and efficient diagnostic and therapeutic solutions. Among the promising options, antibodies have garnered significant attention due to their favorable structural characteristics and versatile applications. Notably, nanobodies (Nbs), the smallest functional single-domain antibodies of heavy-chain only antibodies produced by camelids, exhibit remarkable capabilities in stable antigen binding. They offer unique advantages such as ease of expression and modification and enhanced stability, as well as improved hydrophilicity compared to conventional antibody fragments (antigen-binding fragments (Fab) or single-chain variable fragments (scFv)) that can aggregate due to their low solubility. Nanobodies directly target antigen epitopes or can be engineered into multivalent Nbs and Nb-fusion proteins, expanding their therapeutic potential. This review is dedicated to charting the progress in Nb research, particularly those derived from camelids, and highlighting their diverse applications in treating infectious diseases, spanning both human and animal contexts.

[end paper 56]

[begin paper 57]

PMCID = 9461429

Title = Arsenal of nanobodies shows broad-spectrum neutralization against SARS-CoV-2 variants of concern in vitro and in vivo in hamster models

Nanobodies offer several potential advantages over mAbs for the control of SARS-CoV-2. Their ability to access cryptic epitopes conserved across SARS-CoV-2 variants of concern (VoCs) and feasibility to engineer modular, multimeric designs, make these antibody fragments ideal candidates for developing broad-spectrum therapeutics against current and continually emerging SARS-CoV-2 VoCs. Here we describe a diverse collection of 37 anti-SARS-CoV-2 spike glycoprotein nanobodies extensively characterized as both monovalent and IgG Fc-fused bivalent modalities. The nanobodies were collectively shown to have high intrinsic affinity; high thermal, thermodynamic and aerosolization stability; broad subunit/domain specificity and cross-reactivity across existing VoCs; wide-ranging epitopic and mechanistic diversity and high and broad in vitro neutralization potencies. A select set of Fc-fused nanobodies showed high neutralization efficacies in hamster models of SARS-CoV-2 infection, reducing viral burden by up to six orders of magnitude to below detectable levels. In vivo protection was demonstrated with anti-RBD and previously unreported anti-NTD and anti-S2 nanobodies. This collection of nanobodies provides a potential therapeutic toolbox from which various cocktails or multi-paratopic formats could be built to combat multiple SARS-CoV-2 variants.

Isolation and extensive characterization of a collection of 37 anti-SARS-CoV-2 spike glycoprotein nanobodies show broad neutralization efficacies in vitro and in vivo in a hamster model of SARS-CoV-2 infection.

[end paper 57]

[begin paper 58]

PMCID = 11211474

Title = A broad neutralizing nanobody against SARS-CoV-2 engineered from an approved drug

SARS-CoV-2 infection is initiated by Spike glycoprotein binding to the human angiotensin-converting enzyme 2 (ACE2) receptor via its receptor binding domain. Blocking this interaction has been proven to be an effective approach to inhibit virus infection. Here we report the discovery of a neutralizing nanobody named VHH60, which was directly produced from an engineering nanobody library based on a commercialized nanobody within a very short period. VHH60 competes with human ACE2 to bind the receptor binding domain of the Spike protein at S351, S470-471and S493-494 as determined by structural analysis, with an affinity of 2.56 nM. It inhibits infections of both ancestral SARS-CoV-2 strain and pseudotyped viruses harboring SARS-CoV-2 wildtype, key mutations or variants at the nanomolar level. Furthermore, VHH60 suppressed SARS-CoV-2 infection and propagation 50-fold better and protected mice from death for twice as long as the control group after SARS-CoV-2 nasal infections in vivo. Therefore, VHH60 is not only a powerful nanobody with a promising profile for disease control but also provides evidence for a highly effective and rapid approach to generating therapeutic nanobodies.

[end paper 58]

[begin paper 59]

PMCID = 11460708

Title = Structure-guided in vitro evolution of nanobodies targeting new viral variants

A major challenge in antiviral antibody therapy is keeping up with the rapid evolution of viruses. Our research shows that nanobodies - single-domain antibodies derived from camelids - can be rapidly re-engineered to combat new viral strains through structure-guided in vitro evolution. Specifically, for viral mutations occurring at nanobody-binding sites, we introduce randomized amino acid sequences into nanobody residues near these mutations. We then select nanobody variants that effectively bind to the mutated viral target from a phage display library. As a proof of concept, we used this approach to adapt Nanosota-3, a nanobody originally identified to target the receptor-binding domain (RBD) of early Omicron subvariants, making it highly effective against recent Omicron subvariants. Remarkably, this adaptation process can be completed in less than two weeks, allowing drug development to keep pace with viral evolution and provide timely protection to humans.

The ongoing fight against viral diseases is made more difficult by the rapid evolution of viruses, which often outpaces current antiviral therapies. Our study addresses this critical issue by demonstrating the flexibility and rapid adaptability of antiviral nanobodies - single-domain antibodies derived from camelids. We have developed a structure-guided in vitro evolution strategy to quickly re-engineer nanobodies, enabling them to neutralize newly emerging viral variants. For viral mutations occurring at nanobody-binding sites, this approach introduces randomized amino acid sequences into nanobody residues near these mutations. Nanobody variants that effectively bind to the mutated viral target are then selected using a phage display library. Using this method, we successfully modified the nanobody Nanosota-3, originally an inhibitor of early Omicron subvariants, into a potent inhibitor of more recent Omicron subvariants. Remarkably, this adaptation process can be completed in under two weeks, significantly narrowing the gap between viral mutation rates and the development of effective therapeutic agents. Our findings have profound implications for antiviral drug development, particularly in light of ongoing global health challenges, providing a pathway to more timely and responsive treatments.

[end paper 59]

[begin paper 60]

PMCID = 9230756

Title = Single-Domain Antibodies as Therapeutics for Respiratory RNA Virus Infections

Over the years, infectious diseases with high morbidity and mortality disrupted human healthcare systems and devastated economies globally. Respiratory viruses, especially emerging or re-emerging RNA viruses, including influenza and human coronavirus, are the main pathogens of acute respiratory diseases that cause epidemics or even global pandemics. Importantly, due to the rapid mutation of viruses, there are few effective drugs and vaccines for the treatment and prevention of these RNA virus infections. Of note, a class of antibodies derived from camelid and shark, named nanobody or single-domain antibody (sdAb), was characterized by smaller size, lower production costs, more accessible binding epitopes, and inhalable properties, which have advantages in the treatment of respiratory diseases compared to conventional antibodies. Currently, a number of sdAbs have been developed against various respiratory RNA viruses and demonstrated potent therapeutic efficacy in mouse models. Here, we review the current status of the development of antiviral sdAb and discuss their potential as therapeutics for respiratory RNA viral diseases.

[end paper 60]

[begin paper 61]

PMCID = 8982573

Title = Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance

The ongoing COVID‐19 pandemic represents an unprecedented global health crisis. Here, we report the identification of a synthetic nanobody (sybody) pair, Sb#15 and Sb#68, that can bind simultaneously to the SARS‐CoV‐2 spike RBD and efficiently neutralize pseudotyped and live viruses by interfering with ACE2 interaction. Cryo‐EM confirms that Sb#15 and Sb#68 engage two spatially discrete epitopes, influencing rational design of bispecific and tri‐bispecific fusion constructs that exhibit up to 100‐ and 1,000‐fold increase in neutralization potency, respectively. Cryo‐EM of the sybody‐spike complex additionally reveals a novel up‐out RBD conformation. While resistant viruses emerge rapidly in the presence of single binders, no escape variants are observed in the presence of the bispecific sybody. The multivalent bispecific constructs further increase the neutralization potency against globally circulating SARS‐CoV‐2 variants of concern. Our study illustrates the power of multivalency and biparatopic nanobody fusions for the potential development of therapeutic strategies that mitigate the emergence of new SARS‐CoV‐2 escape mutants.

Sybodies Sb#15 and Sb#68 inhibit SARS‐CoV‐2 infectivity by targeting non‐overlapping epitopes on the spike glycoprotein. Covalent sybody fusion and valency engineering enhances neutralization potency against variants and impedes emergence of escape mutants.  

[end paper 61]

[begin paper 62]

PMCID = 7744031

Title = Slaying SARS-CoV-2 One (Single-domain) Antibody at a Time

Camelid-derived and synthetic single-domain antibodies (sdAbs) are emerging as potent weapons against the novel coronavirus, SARS-CoV-2. sdAbs are small, compact, thermostable immunoglobulin elements capable of binding targets with subnanomolar affinities. By leveraging the power of phage- and yeast surface-display technologies, rare sdAbs can be isolated from highly diverse and complex antibody libraries. Once in hand, sdAbs can be engineered to improve binding affinity, avidity, target specificities, and biodistribution. In this Opinion piece we highlight a series of sophisticated studies describing the identification of ultrapotent sdAbs directed against the receptor-binding domain (RBD) of the SARS-CoV-2 Spike protein. We discuss the possible applications of these antibodies in the global fight against COVID-19.

[end paper 62]

[begin paper 63]

PMCID = 7857409

Title = An ultrapotent synthetic nanobody neutralizes SARS-CoV-2 by stabilizing inactive Spike

Monoclonal antibodies that bind to the spike protein of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) show therapeutic promise but must be produced in mammalian cells and need to be delivered intravenously. By contrast, single-domain antibodies called nanobodies can be produced in bacteria or yeast, and their stability may enable aerosol delivery. Two papers now report nanobodies that bind tightly to spike and efficiently neutralize SARS-CoV-2 in cells. Schoof et al. screened a yeast surface display of synthetic nanobodies and Xiang et al. screened anti-spike nanobodies produced by a llama. Both groups identified highly potent nanobodies that lock the spike protein in an inactive conformation. Multivalent constructs of selected nanobodies achieved even more potent neutralization.

Science, this issue p. 1473, p. 

Potent neutralizers of SARS-CoV-2 are identified by screening either synthetic or llama-produced nanobodies.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) virus enters host cells via an interaction between its Spike protein and the host cell receptor angiotensin-converting enzyme 2 (ACE2). By screening a yeast surface-displayed library of synthetic nanobody sequences, we developed nanobodies that disrupt the interaction between Spike and ACE2. Cryo–electron microscopy (cryo-EM) revealed that one nanobody, Nb6, binds Spike in a fully inactive conformation with its receptor binding domains locked into their inaccessible down state, incapable of binding ACE2. Affinity maturation and structure-guided design of multivalency yielded a trivalent nanobody, mNb6-tri, with femtomolar affinity for Spike and picomolar neutralization of SARS-CoV-2 infection. mNb6-tri retains function after aerosolization, lyophilization, and heat treatment, which enables aerosol-mediated delivery of this potent neutralizer directly to the airway epithelia.

[end paper 63]

[begin paper 64]

PMCID = 11148044

Title = Generation of nanobodies from transgenic ‘LamaMice’ lacking an endogenous immunoglobulin repertoire

Due to their exceptional solubility and stability, nanobodies have emerged as powerful building blocks for research tools and therapeutics. However, their generation in llamas is cumbersome and costly. Here, by inserting an engineered llama immunoglobulin heavy chain (IgH) locus into IgH-deficient mice, we generate a transgenic mouse line, which we refer to as ‘LamaMouse’. We demonstrate that LamaMice solely express llama IgH molecules without association to Igκ or λ light chains. Immunization of LamaMice with AAV8, the receptor-binding domain of the SARS-CoV-2 spike protein, IgE, IgG2c, and CLEC9A enabled us to readily select respective target-specific nanobodies using classical hybridoma and phage display technologies, single B cell screening, and direct cloning of the nanobody-repertoire into a mammalian expression vector. Our work shows that the LamaMouse represents a flexible and broadly applicable platform for a facilitated selection of target-specific nanobodies.

Nanobodies are normally made from immunized camelids, Ig transgenic mice or synthetic libraries. In this study, the authors introduce the llama Ig heavy chain locus into mice lacking this locus, thereby generating a line in which nanobodies can be made by direct immunization in the absence of an endogenous antibody repertoire.

[end paper 64]

[begin paper 65]

PMCID = 7199733

Title = Structural Basis for Potent Neutralization of Betacoronaviruses by Single-Domain Camelid Antibodies

Coronaviruses make use of a large envelope protein called spike (S) to engage host cell receptors and catalyze membrane fusion. Because of the vital role that these S proteins play, they represent a vulnerable target for the development of therapeutics. Here, we describe the isolation of single-domain antibodies (VHHs) from a llama immunized with prefusion-stabilized coronavirus spikes. These VHHs neutralize MERS-CoV or SARS-CoV-1 S pseudotyped viruses, respectively. Crystal structures of these VHHs bound to their respective viral targets reveal two distinct epitopes, but both VHHs interfere with receptor binding. We also show cross-reactivity between the SARS-CoV-1 S-directed VHH and SARS-CoV-2 S and demonstrate that this cross-reactive VHH neutralizes SARS-CoV-2 S pseudotyped viruses as a bivalent human IgG Fc-fusion. These data provide a molecular basis for the neutralization of pathogenic betacoronaviruses by VHHs and suggest that these molecules may serve as useful therapeutics during coronavirus outbreaks.

VHHs isolated from a llama immunized with prefusion-stabilized coronavirus spikes

Structural characterization of VHHs reveals conserved mechanism of neutralization

SARS-CoV-1 S-directed VHH cross-reacts with SARS-CoV-2 S

Bivalent VHH neutralizes SARS-CoV-2 pseudoviruses

Using llamas immunized with prefusion-stabilized betacoronavirus spike proteins, Wrapp et al. identify neutralizing cross-reactive single-domain camelid antibodies, which may serve not only as useful reagents for researchers studying the viruses causing MERS, SARS, and COVID-19, but also potential therapeutic candidates. Crystal structures further reveal how these antibodies bind spike proteins to prevent virus entry into cells.

[end paper 65]

[begin paper 66]

PMCID = 9219022

Title = Production and purification of shark and camel single domain antibodies from bacterial and mammalian cell expression systems

Single domain antibodies, including the antigen binding variable domains of the shark immunoglobulin new antigen receptor and the camelid variable region of the heavy chain, are the smallest antigen recognition domains (~15 kDa) and have unique characteristics compared to conventional antibodies. They are capable of binding epitopes that are hard to access for classical antibodies and can also be used for therapeutics or diagnostics, or modular building blocks for multi-domain constructs, antibody-drug conjugates, immunotoxins, or chimeric antigen receptor therapy. This protocol contains detailed procedures for the purification and validation of two single domain antibodies (one shark and one camel), which bind to the S2 subunit of the SARS-CoV-2 spike protein, using both bacterial and mammalian cell expression systems. It provides a comprehensive reference for the production of single domain antibodies with high yield, good quality, and purity.

[end paper 66]

[begin paper 67]

PMCID = 9082315

Title = Nanobodies Protecting From Lethal SARS-CoV-2 Infection Target Receptor Binding Epitopes Preserved in Virus Variants Other Than Omicron

The emergence of SARS-CoV-2 variants that escape from immune neutralization are challenging vaccines and antibodies developed to stop the COVID-19 pandemic. Thus, it is important to establish therapeutics directed toward multiple or specific SARS-CoV-2 variants. The envelope spike (S) glycoprotein of SARS-CoV-2 is the key target of neutralizing antibodies (Abs). We selected a panel of nine nanobodies (Nbs) from dromedary camels immunized with the receptor-binding domain (RBD) of the S, and engineered Nb fusions as humanized heavy chain Abs (hcAbs). Nbs and derived hcAbs bound with subnanomolar or picomolar affinities to the S and its RBD, and S-binding cross-competition clustered them in two different groups. Most of the hcAbs hindered RBD binding to its human ACE2 (hACE2) receptor, blocked cell entry of viruses pseudotyped with the S protein and neutralized SARS-CoV-2 infection in cell cultures. Four potent neutralizing hcAbs prevented the progression to lethal SARS-CoV-2 infection in hACE2-transgenic mice, demonstrating their therapeutic potential. Cryo-electron microscopy identified Nb binding epitopes in and out the receptor binding motif (RBM), and showed different ways to prevent virus binding to its cell entry receptor. The Nb binding modes were consistent with its recognition of SARS-CoV-2 RBD variants; mono and bispecific hcAbs efficiently bound all variants of concern except omicron, which emphasized the immune escape capacity of this latest variant.

[end paper 67]

[begin paper 68]

PMCID = 8325376

Title = Nanobodies as powerful pulmonary targeted biotherapeutics against SARS-CoV-2, pharmaceutical point of view

Background

Since December 2019, the newly emerged SARS-CoV-2 virus continues to infect humans and many people died from severe Covid-19 during the last 2 years worldwide. Different approaches are being used for treatment of this infection and its consequences, but limited results have been achieved and new therapeutics are still needed. One of the most interesting biotherapeutics in this era are Nanobodies which have shown very promising results in recent researches.

Scope of review

Here, we have reviewed the potentials of Nanobodies in Covid-19 treatment. We have also discussed the properties of these biotherapeutics that make them very suitable for pulmonary drug delivery, which seems to be very important route of administration in this disease.

Major conclusion

Nanobodies with their special biological and biophysical characteristics and their resistance against harsh manufacturing condition, can be considered as promising, targeted biotherapeutics which can be administered by pulmonary delivery pharmaceutical systems against Covid-19.

General significance

Covid-19 has become a global problem during the last two years and with emerging mutant strains, prophylactic and therapeutic approaches are still highly needed. Nanobodies with their specific properties can be considered as valuable and promising candidates in Covid-19 therapy.

[end paper 68]

[begin paper 69]

PMCID = 8726723

Title = Rapid discovery of diverse neutralizing SARS-CoV-2 antibodies from large-scale synthetic phage libraries

Coronavirus disease 2019 (COVID-19) is an evolving global public health crisis in need of therapeutic options. Passive immunization of monoclonal antibodies (mAbs) represents a promising therapeutic strategy capable of conferring immediate protection from SARS-CoV-2 infection. Herein, we describe the discovery and characterization of neutralizing SARS-CoV-2 IgG and VHH antibodies from four large-scale phage libraries. Each library was constructed synthetically with shuffled complementarity-determining region loops from natural llama and human antibody repertoires. While most candidates targeted the receptor-binding domain of the S1 subunit of SARS-CoV-2 spike protein, we also identified a neutralizing IgG candidate that binds a unique epitope on the N-terminal domain. A select number of antibodies retained binding to SARS-CoV-2 variants Alpha, Beta, Gamma, Kappa and Delta. Overall, our data show that synthetic phage libraries can rapidly yield SARS-CoV-2 S1 antibodies with therapeutically desirable features, including high affinity, unique binding sites, and potent neutralizing activity in vitro, and a capacity to limit disease in vivo.

[end paper 69]

[begin paper 70]

PMCID = 7832525

Title = Nanobodies, the potent agents to detect and treat the Coronavirus infections: A systematic review

The newly emerged coronavirus (SARS-CoV-2) continues to infect humans, and no effective treatment has yet been found. Antibody therapy is one way to control infection caused by COVID-19. However, the use of classical antibodies raises complex issues. Heavy chain antibodies (HCAbs) are single-domain antibodies derived from the Camelidae family. The variable part of these antibodies (Nanobodies or VHH) has interesting properties such as small size, cost-effective production, and good tissue permeability, causing VHH to be regarded as an antiviral therapeutics. However, the small size of nanobodies may lead to low antigen binding affinity and rapid renal clearance. In this systematic review, the application of nanobodies in the treatment of COVID-19 infection and other similar infections (MERS and SARS) was reviewed.

The clinical use of classical antibodies raises complex issues.

Nanobodies as the single-chain antibodies are now widely used and tested in the treatment of many diseases.

Many nanobodies have been produced against various viruses.

The nanobodies can be considered as prominent agents to treat COVID-19 infectious disease.

[end paper 70]

[begin paper 71]

PMCID = 8021111

Title = Dromedary camels as a natural source of neutralizing nanobodies against SARS-CoV-2

The development of prophylactic and therapeutic agents for coronavirus disease 2019 (COVID-19) is a current global health priority. Here, we investigated the presence of cross-neutralizing antibodies against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) in dromedary camels that were Middle East respiratory syndrome coronavirus (MERS-CoV) seropositive but MERS-CoV free. The tested 229 dromedaries had anti–MERS-CoV camel antibodies with variable cross-reactivity patterns against SARS-CoV-2 proteins, including the S trimer and M, N, and E proteins. Using SARS-CoV-2 competitive immunofluorescence immunoassays and pseudovirus neutralization assays, we found medium-to-high titers of cross-neutralizing antibodies against SARS-CoV-2 in these animals. Through linear B cell epitope mapping using phage immunoprecipitation sequencing and a SARS-CoV-2 peptide/proteome microarray, we identified a large repertoire of Betacoronavirus cross-reactive antibody specificities in these dromedaries and demonstrated that the SARS-CoV-2–specific VHH antibody repertoire is qualitatively diverse. This analysis revealed not only several SARS-CoV-2 epitopes that are highly immunogenic in humans, including a neutralizing epitope, but also epitopes exclusively targeted by camel antibodies. The identified SARS-CoV-2 cross-neutralizing camel antibodies are not proposed as a potential treatment for COVID-19. Rather, their presence in nonimmunized camels supports the development of SARS-CoV-2 hyperimmune camels, which could be a prominent source of therapeutic agents for the prevention and treatment of COVID-19.

[end paper 71]

[begin paper 72]

PMCID = 9896449

Title = Shark nanobodies with potent SARS-CoV-2 neutralizing activity and broad sarbecovirus reactivity

Despite rapid and ongoing vaccine and therapeutic development, SARS-CoV-2 continues to evolve and evade, presenting a need for next-generation diverse therapeutic modalities. Here we show that nurse sharks immunized with SARS-CoV-2 recombinant receptor binding domain (RBD), RBD-ferritin (RFN), or spike protein ferritin nanoparticle (SpFN) immunogens elicit a set of new antigen receptor antibody (IgNAR) molecules that target two non-overlapping conserved epitopes on the spike RBD. Representative shark antibody variable NAR-Fc chimeras (ShAbs) targeting either of the two epitopes mediate cell-effector functions, with high affinity to all SARS-CoV-2 viral variants of concern, including the divergent Omicron strains. The ShAbs potently cross-neutralize SARS-CoV-2 WA-1, Alpha, Beta, Delta, Omicron BA.1 and BA.5, and SARS-CoV-1 pseudoviruses, and confer protection against SARS-CoV-2 challenge in the K18-hACE2 transgenic mouse model. Structural definition of the RBD-ShAb01-ShAb02 complex enabled design and production of multi-specific nanobodies with enhanced neutralization capacity, and picomolar affinity to divergent sarbecovirus clade 1a, 1b and 2 RBD molecules. These shark nanobodies represent potent immunotherapeutics both for current use, and future sarbecovirus pandemic preparation.

SARS-CoV-2 variants of concern continue to emerge, reducing vaccine efficacy and limiting therapeutic options. Here, Chen and colleagues describe the identification and design of shark nanobodies with pansarbecovirus activity.

[end paper 72]

[begin paper 73]

PMCID = 11762740

Title = Visual and High-Efficiency Secretion of SARS-CoV-2 Nanobodies with Escherichia coli

Nanobodies have gained attention as potential therapeutic and diagnostic agents for severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) due to their ability to bind and neutralize the virus. However, rapid, scalable, and robust production of nanobodies for SARS-CoV-2 remains a crucial challenge. In this study, we developed a visual and high-efficiency biomanufacturing method for nanobodies with Escherichia coli by fusing the super-folder green fluorescent protein (sfGFP) to the N-terminus or C-terminus of the nanobody. Several receptor-binding domain (RBD)-specific nanobodies of the SARS-CoV-2 spike protein (S) were secreted onto the surface of E. coli cells and even into the culture medium, including Fu2, ANTE, mNb6, MR3-MR3, and n3113.1. The nanobodies secreted by E. coli retained equal activity as prior research, regardless of whether sfGFP was removed. Since some of the nanobodies bound to different regions of the RBD, we combined two nanobodies to improve the affinity. Fu2-sfGFP-ANTE was constructed to be bispecific for the RBD, and the bispecific nanobody exhibited significantly higher affinity than Fu2 (35.0-fold), ANTE (7.3-fold), and the combination of the two nanobodies (3.3-fold). Notably, Fu2-sfGFP-ANTE can be normally secreted into the culture medium and outer membrane. The novel nanobody production system enhances the efficiency of nanobody expression and streamlines the downstream purification process, enabling large-scale, cost-effective nanobody production. In addition, E. coli cells secreting the nanobodies on their surface facilitates screening and characterization of antigen-binding clones.

[end paper 73]

[begin paper 74]

PMCID = 10231691

Title = Plant-derived single domain COVID-19 antibodies

Data show a decrease in the risk of hospitalization and death from COVID-19. To date, global vaccinations for SARS-CoV-2 protections are underway, but additional treatments are urgently needed to prevent and cure infection among naïve and even vaccinated people. Neutralizing monoclonal antibodies are very promising for prophylaxis and therapy of SARS-CoV-2 infections. However, traditional large-scale methods of producing such antibodies are slow, extremely expensive and possess a high risk of contamination with viruses, prions, oncogenic DNA and other pollutants. The present study is aimed at developing an approach of producing monoclonal antibodies (mAbs) against SARS-CoV-2 spike (S) protein in plant systems which offers unique advantages, such as the lack of human and animal pathogens or bacterial toxins, relatively low-cost manufacturing, and ease of production scale-up. We selected a single N-terminal domain functional camelid-derived heavy (H)-chain antibody fragments (VHH, AKA nanobodies) targeted to receptor binding domain of SARS-CoV-2 spike protein and developed methods of their rapid production using transgenic plants and plant cell suspensions. Isolated and purified plant-derived VHH antibodies were compared with mAbs produced in traditional mammalian and bacterial expression systems. It was found that plant generated VHH using the proposed methods of transformation and purification possess the ability to bind to SARS-CoV-2 spike protein comparable to that of monoclonal antibodies derived from bacterial and mammalian cell cultures. The results of the present studies confirm the visibility of producing monoclonal single-chain antibodies with a high ability to bind the targeted COVID-19 spike protein in plant systems within a relatively shorter time span and at a lower cost when compared with traditional methods. Moreover, similar plant biotechnology approaches can be used for producing monoclonal neutralizing antibodies against other types of viruses.

[end paper 74]

[begin paper 75]

PMCID = 9356539

Title = Rapid and Quantitative In-vitro Evaluation of SARS-CoV-2 Neutralizing Antibodies and Nanobodies

Neutralizing monoclonal antibodies and nanobodies have shown promising results as potential therapeutic agents for COVID-19. Identifying such antibodies and nanobodies requires evaluating the neutralization activity of a large number of lead molecules via biological assays, such as the virus neutralization test (VNT). These assays are typically time-consuming and demanding on lab facilities. Here, we present a rapid and quantitative assay that evaluates the neutralizing efficacy of an antibody or nanobody within 1.5 hours, does not require BSL-2 facilities, and consumes only 8 μL of low concentration (ng/mL) sample for each assay run. We tested the human angiotensin-converting enzyme 2 (ACE2) binding inhibition efficacy of seven antibodies and eight nanobodies and verified that the IC50 values of our assay are comparable with those from SARS-CoV-2 pseudovirus neutralization tests. We also found that our assay could evaluate the neutralizing efficacy against three widespread SARS-CoV-2 variants. We observed increased affinity of these variants for ACE2, including the Beta and Gamma variants. Finally, we demonstrated that our assay enables rapid identification of an immune-evasive mutation of the SARS-CoV-2 spike protein utilizing a set of nanobodies with known binding epitopes.

[end paper 75]

[begin paper 76]

PMCID = 10590698

Title = Nanobodies with cross-neutralizing activity provide prominent therapeutic efficacy in mild and severe COVID-19 rodent models

The weakened protective efficacy of COVID-19 vaccines and antibodies caused by SARS-CoV-2 variants presents a global health emergency, which underscores the urgent need for universal therapeutic antibody intervention for clinical patients. Here, we screened three alpacas-derived nanobodies (Nbs) with neutralizing activity from twenty RBD-specific Nbs. The three Nbs were fused with the Fc domain of human IgG, namely aVHH-11-Fc, aVHH-13-Fc and aVHH-14-Fc, which could specifically bind RBD protein and competitively inhibit the binding of ACE2 receptor to RBD. They effectively neutralized SARS-CoV-2 pseudoviruses D614G, Alpha, Beta, Gamma, Delta, and Omicron sub-lineages BA.1, BA.2, BA.4, and BA.5 and authentic SARS-CoV-2 prototype, Delta, and Omicron BA.1, BA.2 strains. In mice-adapted COVID-19 severe model, intranasal administration of aVHH-11-Fc, aVHH-13-Fc and aVHH-14-Fc effectively protected mice from lethal challenges and reduced viral loads in both the upper and lower respiratory tracts. In the COVID-19 mild model, aVHH-13-Fc, which represents the optimal neutralizing activity among the above three Nbs, effectively protected hamsters from the challenge of SARS-CoV-2 prototype, Delta, Omicron BA.1 and BA.2 by significantly reducing viral replication and pathological alterations in the lungs. In structural modeling of aVHH-13 and RBD, aVHH-13 binds to the receptor-binding motif region of RBD and interacts with some highly conserved epitopes. Taken together, our study illustrated that alpaca-derived Nbs offered a therapeutic countermeasure against SARS-CoV-2, including those Delta and Omicron variants which have evolved into global pandemic strains.

Alpaca-derived nanobodies showed a cross-neutralizing activity against SARS-CoV-2 variants.

aVHH-13-Fc provided complete protection in mild and severe COVID-19 rodent models.

The binding site between aVHH-13-Fc and the RBD variant of SARS-CoV-2 is partially located on the RBM.

[end paper 76]

[begin paper 77]

PMCID = 9678391

Title = Nanobody derived using a peptide epitope from the spike protein receptor-binding motif inhibits entry of SARS-CoV-2 variants

The emergence of new escape mutants of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has escalated its penetration among the human population and has reinstated its status as a global pandemic. Therefore, developing effective antiviral therapy against emerging SARS-CoV variants and other viruses in a short period becomes essential. Blocking SARS-CoV-2 entry into human host cells by disrupting the spike glycoprotein-angiotensin-converting enzyme 2 interaction has already been exploited for vaccine development and monoclonal antibody therapy. Unlike the previous reports, our study used a nine-amino acid peptide from the receptor-binding motif of the spike protein as an epitope. We report the identification of an efficacious nanobody N1.2 that blocks the entry of pseudovirus-containing SARS-CoV-2 spike as the surface glycoprotein. Moreover, using mCherry fluorescence-based reporter assay, we observe a more potent neutralizing effect against both the hCoV19 (Wuhan/WIV04/2019) and the Omicron (BA.1) pseudotyped spike virus with a bivalent version of the N1.2 nanobody. In summary, our study presents a rapid and efficient methodology to use peptide sequences from a protein–receptor interaction interface as epitopes for screening nanobodies against potential pathogenic targets. We propose that this approach can also be widely extended to target other viruses and pathogens in the future.

[end paper 77]

[begin paper 78]

PMCID = 9479348

Title = A potent synthetic nanobody with broad-spectrum activity neutralizes SARS-CoV-2 virus and the Omicron variant BA.1 through a unique binding mode

The major challenge to controlling the COVID pandemic is the rapid mutation rate of the SARS-CoV-2 virus, leading to the escape of the protection of vaccines and most of the neutralizing antibodies to date. Thus, it is essential to develop neutralizing antibodies with broad-spectrum activity targeting multiple SARS-CoV-2 variants. Here, we report a synthetic nanobody (named C5G2) obtained by phage display and subsequent antibody engineering. C5G2 has a single-digit nanomolar binding affinity to the RBD domain and inhibits its binding to ACE2 with an IC50 of 3.7 nM. Pseudovirus assays indicated that monovalent C5G2 could protect the cells from infection with SARS-CoV-2 wild-type virus and most of the viruses of concern, i.e., Alpha, Beta, Gamma and Omicron variants. Strikingly, C5G2 has the highest potency against Omicron BA.1 among all the variants, with an IC50 of 4.9 ng/mL. The cryo-EM structure of C5G2 in complex with the spike trimer showed that C5G2 binds to RBD mainly through its CDR3 at a conserved region that does not overlap with the ACE2 binding surface. Additionally, C5G2 binds simultaneously to the neighboring NTD domain of the spike trimer through the same CDR3 loop, which may further increase its potency against viral infection. Third, the steric hindrance caused by FR2 of C5G2 could inhibit the binding of ACE2 to RBD as well. Thus, this triple-function nanobody may serve as an effective drug for prophylaxis and therapy against Omicron as well as future variants.

The online version contains supplementary material available at 10.1186/s12951-022-01619-y.

[end paper 78]

[begin paper 79]

PMCID = 8811630

Title = Biparatopic nanobodies protect mice from lethal challenge with SARS‐CoV‐2 variants of concern

The ongoing COVID‐19 pandemic and the emergence of new SARS‐CoV‐2 variants of concern (VOCs) requires continued development of effective therapeutics. Recently, we identified high‐affinity neutralizing nanobodies (Nbs) specific for the receptor‐binding domain (RBD) of SARS‐CoV‐2. Taking advantage of detailed epitope mapping, we generate two biparatopic Nbs (bipNbs) targeting a conserved epitope outside and two different epitopes inside the RBD:ACE2 interface. Both bipNbs bind all currently circulating VOCs with high affinities and are capable to neutralize cellular infection with VOC B.1.351 (Beta) and B.1.617.2 (Delta) in vitro. To assess if the bipNbs NM1267 and NM1268 confer protection against SARS‐CoV‐2 infection in vivo, human ACE2 transgenic mice are treated intranasally before infection with a lethal dose of SARS‐CoV‐2 B.1, B.1.351 (Beta) or B.1.617.2 (Delta). Nb‐treated mice show significantly reduced disease progression and increased survival rates. Histopathological analyses further reveal a drastically reduced viral load and inflammatory response in lungs. These data suggest that both bipNbs are broadly active against a variety of emerging SARS‐CoV‐2 VOCs and represent easily applicable drug candidates.

This study reports on two highly efficient RBD‐specific biparatopic nanobodies that bind and neutralize SARS‐CoV‐2 and its currently circulating variants of concern in vitro and in vivo, underscoring the potential of nanobodies to prevent or treat SARS‐CoV‐2 infection. 

[end paper 79]

[begin paper 80]

PMCID = 9834170

Title = Two pan-SARS-CoV-2 nanobodies and their multivalent derivatives effectively prevent Omicron infections in mice

With the widespread vaccinations against coronavirus disease 2019 (COVID-19), we are witnessing gradually waning neutralizing antibodies and increasing cases of breakthrough infections, necessitating the development of drugs aside from vaccines, particularly ones that can be administered outside of hospitals. Here, we present two cross-reactive nanobodies (R14 and S43) and their multivalent derivatives, including decameric ones (fused to the immunoglobulin M [IgM] Fc) that maintain potent neutralizing activity against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) after aerosolization and display not only pan-SARS-CoV-2 but also varied pan-sarbecovirus activities. Through respiratory administration to mice, monovalent and decameric R14 significantly reduce the lung viral RNAs at low dose and display potent pre- and post-exposure protection. Furthermore, structural studies reveal the neutralizing mechanisms of R14 and S43 and the multiple inhibition effects that the multivalent derivatives exert. Our work demonstrates promising convenient drug candidates via respiratory administration against SARS-CoV-2 infection, which can contribute to containing the COVID-19 pandemic.

Nanobodies R14 and S43 display not just pan-SARS-CoV-2 but varied pan-sarbecovirus activities

The multivalent nanobodies show enhanced neutralization, fitting the avidity effect

Intranasally administered MR14 potently prevents Omicron infections in mice

Structural studies reveal the neutralizing mechanisms of R14 and S43 as well as MR14

Liu et al. isolate two nanobodies (R14 and S43) and construct their multivalent derivatives with increased potencies and breadth against SARS-CoV-2 variants. Particularly, decameric R14, maintaining similar activities after aerosolization, efficiently prevents BA.2 infections in both pre- and post-exposure prophylactic settings in vivo and deserves further development.

[end paper 80]

[begin paper 81]

PMCID = 8324831

Title = A synthetic nanobody targeting RBD protects hamsters from SARS-CoV-2 infection

SARS-CoV-2, the causative agent of COVID-19, features a receptor-binding domain (RBD) for binding to the host cell ACE2 protein. Neutralizing antibodies that block RBD-ACE2 interaction are candidates for the development of targeted therapeutics. Llama-derived single-domain antibodies (nanobodies, ~15 kDa) offer advantages in bioavailability, amenability, and production and storage owing to their small sizes and high stability. Here, we report the rapid selection of 99 synthetic nanobodies (sybodies) against RBD by in vitro selection using three libraries. The best sybody, MR3 binds to RBD with high affinity (KD = 1.0 nM) and displays high neutralization activity against SARS-CoV-2 pseudoviruses (IC50 = 0.42 μg mL−1). Structural, biochemical, and biological characterization suggests a common neutralizing mechanism, in which the RBD-ACE2 interaction is competitively inhibited by sybodies. Various forms of sybodies with improved potency have been generated by structure-based design, biparatopic construction, and divalent engineering. Two divalent forms of MR3 protect hamsters from clinical signs after live virus challenge and a single dose of the Fc-fusion construct of MR3 reduces viral RNA load by 6 Log10. Our results pave the way for the development of therapeutic nanobodies against COVID-19 and present a strategy for rapid development of targeted medical interventions during an outbreak.

Here, the authors report the engineering, structural and biological characterization of synthetic nanobodies (sybodies) that display potent therapeutic activity against SARS-CoV-2 infection in animal models via targeting the virus receptor-binding domain.

[end paper 81]

[begin paper 82]

PMCID = 7857400

Title = Versatile and multivalent nanobodies efficiently neutralize SARS-CoV-2

Monoclonal antibodies that bind to the spike protein of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) show therapeutic promise but must be produced in mammalian cells and need to be delivered intravenously. By contrast, single-domain antibodies called nanobodies can be produced in bacteria or yeast, and their stability may enable aerosol delivery. Two papers now report nanobodies that bind tightly to spike and efficiently neutralize SARS-CoV-2 in cells. Schoof et al. screened a yeast surface display of synthetic nanobodies and Xiang et al. screened anti-spike nanobodies produced by a llama. Both groups identified highly potent nanobodies that lock the spike protein in an inactive conformation. Multivalent constructs of selected nanobodies achieved even more potent neutralization.

Science, this issue p. , p. 1479

Potent neutralizers of SARS-CoV-2 are identified by screening either synthetic or llama-produced nanobodies.

Cost-effective, efficacious therapeutics are urgently needed to combat the COVID-19 pandemic. In this study, we used camelid immunization and proteomics to identify a large repertoire of highly potent neutralizing nanobodies (Nbs) to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein receptor binding domain (RBD). We discovered Nbs with picomolar to femtomolar affinities that inhibit viral infection at concentrations below the nanograms-per-milliliter level, and we determined a structure of one of the most potent Nbs in complex with the RBD. Structural proteomics and integrative modeling revealed multiple distinct and nonoverlapping epitopes and indicated an array of potential neutralization mechanisms. We bioengineered multivalent Nb constructs that achieved ultrahigh neutralization potency (half-maximal inhibitory concentration as low as 0.058 ng/ml) and may prevent mutational escape. These thermostable Nbs can be rapidly produced in bulk from microbes and resist lyophilization and aerosolization.

[end paper 82]

[begin paper 83]

PMCID = 8967979

Title = Short-Term Instantaneous Prophylaxis and Efficient Treatment Against SARS-CoV-2 in hACE2 Mice Conferred by an Intranasal Nanobody (Nb22)

Current COVID-19 vaccines need to take at least one month to complete inoculation and then become effective. Around 51% of the global population is still not fully vaccinated. Instantaneous protection is an unmet need among those who are not fully vaccinated. In addition, breakthrough infections caused by SARS-CoV-2 are widely reported. All these highlight the unmet needing for short-term instantaneous prophylaxis (STIP) in the communities where SARS-CoV-2 is circulating. Previously, we reported nanobodies isolated from an alpaca immunized with the spike protein, exhibiting ultrahigh potency against SARS-CoV-2 and its variants. Herein, we found that Nb22, among our previously reported nanobodies, exhibited ultrapotent neutralization against Delta variant with an IC50 value of 0.41 ng/ml (5.13 pM). Furthermore, the crystal structural analysis revealed that the binding of Nb22 to WH01 and Delta RBDs both effectively blocked the binding of RBD to hACE2. Additionally, intranasal Nb22 exhibited protection against SARS-CoV-2 Delta variant in the post-exposure prophylaxis (PEP) and pre-exposure prophylaxis (PrEP). Of note, intranasal Nb22 also demonstrated high efficacy against SARS-CoV-2 Delta variant in STIP for seven days administered by single dose and exhibited long-lasting retention in the respiratory system for at least one month administered by four doses, providing a strategy of instantaneous short-term prophylaxis against SARS-CoV-2. Thus, ultrahigh potency, long-lasting retention in the respiratory system and stability at room-temperature make the intranasal or inhaled Nb22 to be a potential therapeutic or STIP agent against SARS-CoV-2.

[end paper 83]

[begin paper 84]

PMCID = 9529347

Title = Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2

The development of therapeutics to prevent or treat COVID-19 remains an area of intense focus. Protein biologics, including monoclonal antibodies and nanobodies that neutralize virus, have potential for the treatment of active disease. Here, we have used yeast display of a synthetic nanobody library to isolate nanobodies that bind the receptor-binding domain (RBD) of SARS-CoV-2 and neutralize the virus. We show that combining two clones with distinct binding epitopes within the RBD into a single protein construct to generate biparatopic reagents dramatically enhances their neutralizing capacity. Furthermore, the biparatopic nanobodies exhibit enhanced control over clinically relevant RBD variants that escaped recognition by the individual nanobodies. Structural analysis of biparatopic binding to spike (S) protein revealed a unique binding mode whereby the two nanobody paratopes bridge RBDs encoded by distinct S trimers. Accordingly, biparatopic nanobodies offer a way to rapidly generate powerful viral neutralizers with enhanced ability to control viral escape mutants.

Yeast display was used to generate SARS-CoV-2 RBD-specific nanobodies

Distinct nanobody clones were fused to create biparatopic reagents

Biparatopic nanobodies protect mice from SARS-CoV-2 infection

The biparatopics bridge distinct S proteins in a unique “twinned-S” conformation

Immunology; Microbiology; Structural biology.

[end paper 84]

[begin paper 85]

PMCID = 7483421

Title = Humanized single domain antibodies neutralize SARS-CoV-2 by targeting the spike receptor binding domain

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spreads worldwide and leads to an unprecedented medical burden and lives lost. Neutralizing antibodies provide efficient blockade for viral infection and are a promising category of biological therapies. Here, using SARS-CoV-2 spike receptor-binding domain (RBD) as a bait, we generate a panel of humanized single domain antibodies (sdAbs) from a synthetic library. These sdAbs reveal binding kinetics with the equilibrium dissociation constant (KD) of 0.99–35.5 nM. The monomeric sdAbs show half maximal neutralization concentration (EC50) of 0.0009–0.07 µg/mL and 0.13–0.51 µg/mL against SARS-CoV-2 pseudotypes, and authentic SARS-CoV-2, respectively. Competitive ligand-binding experiments suggest that the sdAbs either completely block or significantly inhibit the association between SARS-CoV-2 RBD and viral entry receptor ACE2. Fusion of the human IgG1 Fc to sdAbs improve their neutralization activity by up to ten times. These results support neutralizing sdAbs as a potential alternative for antiviral therapies.

Here, using a humanized phage display library with recombinant SARS- CoV-2 receptor binding domain (RBD) proteins, the authors identify a number of single domain antibodies (sdAbs) that neutralize SARS-CoV-2 in vitro by inhibiting the interaction of the RDB with the host entry receptor ACE2.

[end paper 85]

[begin paper 86]

PMCID = 9482557

Title = Single domain antibodies derived from ancient animals as broadly neutralizing agents for SARS-CoV-2 and other coronaviruses

With severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) as an emergent human virus since December 2019, the world population is susceptible to coronavirus disease 2019 (COVID-19). SARS-CoV-2 has higher transmissibility than the previous coronaviruses, associated by the ribonucleic acid (RNA) virus nature with high mutation rate, caused SARS-CoV-2 variants to arise while circulating worldwide. Neutralizing antibodies are identified as immediate and direct-acting therapeutic against COVID-19. Single-domain antibodies (sdAbs), as small biomolecules with non-complex structure and intrinsic stability, can acquire antigen-binding capabilities comparable to conventional antibodies, which serve as an attractive neutralizing solution. SARS-CoV-2 spike protein attaches to human angiotensin-converting enzyme 2 (ACE2) receptor on lung epithelial cells to initiate viral infection, serves as potential therapeutic target. sdAbs have shown broad neutralization towards SARS-CoV-2 with various mutations, effectively stop and prevent infection while efficiently block mutational escape. In addition, sdAbs can be developed into multivalent antibodies or inhaled biotherapeutics against COVID-19.

.

[end paper 86]

[begin paper 87]

PMCID = 7556756

Title = Antibody-like proteins that capture and neutralize SARS-CoV-2

SARS-CoV-2–neutralizing antibody-like proteins were generated by a high-speed in vitro selection method.

To combat severe acute respiratory syndrome–related coronavirus 2 (SARS-CoV-2) and any unknown emerging pathogens in the future, the development of a rapid and effective method to generate high-affinity antibodies or antibody-like proteins is of critical importance. We here report high-speed in vitro selection of multiple high-affinity antibody-like proteins against various targets including the SARS-CoV-2 spike protein. The sequences of monobodies against the SARS-CoV-2 spike protein were successfully procured within only 4 days. Furthermore, the obtained monobody efficiently captured SARS-CoV-2 particles from the nasal swab samples of patients and exhibited a high neutralizing activity against SARS-CoV-2 infection (half-maximal inhibitory concentration, 0.5 nanomolar). High-speed in vitro selection of antibody-like proteins is a promising method for rapid development of a detection method for, and of a neutralizing protein against, a virus responsible for an ongoing, and possibly a future, pandemic.

[end paper 87]

[begin paper 88]

PMCID = 11277559

Title = Single-Domain Antibodies—Novel Tools to Study and Treat Allergies

IgE-mediated allergies represent a major health problem in the modern world. Apart from allergen-specific immunotherapy (AIT), the only disease-modifying treatment, researchers focus on biologics that target different key molecules such as allergens, IgE, or type 2 cytokines to ameliorate allergic symptoms. Single-domain antibodies, or nanobodies, are the newcomers in biotherapeutics, and their huge potential is being investigated in various research fields since their discovery 30 years ago. While they are dominantly applied for theranostics of cancer and treatment of infectious diseases, nanobodies have become increasingly substantial in allergology over the last decade. In this review, we discuss the prerequisites that we consider to be important for generating useful nanobody-based drug candidates for treating allergies. We further summarize the available research data on nanobodies used as allergen monitoring and detection probes and for therapeutic approaches. We reflect on the limitations that have to be addressed during the development process, such as in vivo half-life and immunogenicity. Finally, we speculate about novel application formats for allergy treatment that might be available in the future.

[end paper 88]

[begin paper 89]

PMCID = 8351461

Title = Structural Basis for SARS-CoV-2 Nucleocapsid Protein Recognition by Single-Domain Antibodies

The COVID-19 pandemic, caused by the coronavirus SARS-CoV-2, is the most severe public health event of the twenty-first century. While effective vaccines against SARS-CoV-2 have been developed, there remains an urgent need for diagnostics to quickly and accurately detect infections. Antigen tests, particularly those that detect the abundant SARS-CoV-2 Nucleocapsid protein, are a proven method for detecting active SARS-CoV-2 infections. Here we report high-resolution crystal structures of three llama-derived single-domain antibodies that bind the SARS-CoV-2 Nucleocapsid protein with high affinity. Each antibody recognizes a specific folded domain of the protein, with two antibodies recognizing the N-terminal RNA binding domain and one recognizing the C-terminal dimerization domain. The two antibodies that recognize the RNA binding domain affect both RNA binding affinity and RNA-mediated phase separation of the Nucleocapsid protein. All three antibodies recognize highly conserved surfaces on the Nucleocapsid protein, suggesting that they could be used to develop affordable diagnostic tests to detect all circulating SARS-CoV-2 variants.

[end paper 89]

[begin paper 90]

PMCID = 8344751

Title = Development of potent and effective synthetic SARS-CoV-2 neutralizing nanobodies

The respiratory virus responsible for coronavirus disease 2019 (COVID-19), severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has affected nearly every aspect of life worldwide, claiming the lives of over 3.9 million people globally, at the time of this publication. Neutralizing humanized nanobody (VHH)-based antibodies (VHH-huFc) represent a promising therapeutic intervention strategy to address the current SARS-CoV-2 pandemic and provide a powerful toolkit to address future virus outbreaks. Using a synthetic, high-diversity VHH bacteriophage library, several potent neutralizing VHH-huFc antibodies were identified and evaluated for their capacity to tightly bind to the SARS-CoV-2 receptor-binding domain, to prevent binding of SARS-CoV-2 spike (S) to the cellular receptor angiotensin-converting enzyme 2, and to neutralize viral infection. Preliminary preclinical evaluation of multiple VHH-huFc antibody candidates demonstrate that they are prophylactically and therapeutically effective in vivo against wildtype SARS-CoV-2. The identified and characterized VHH-huFc antibodies described herein represent viable candidates for further preclinical evaluation and another tool to add to our therapeutic arsenal to address the COVID-19 pandemic.

[end paper 90]

[begin paper 91]

PMCID = 9334538

Title = Hetero-bivalent nanobodies provide broad-spectrum protection against SARS-CoV-2 variants of concern including Omicron

SARS-CoV-2 variants with adaptive mutations have continued to emerge, causing fresh waves of infection even amongst vaccinated population. The development of broad-spectrum antivirals is thus urgently needed. We previously developed two hetero-bivalent nanobodies (Nbs), aRBD-2-5 and aRBD-2-7, with potent neutralization activity against the wild-type (WT) Wuhan isolated SARS-CoV-2, by fusing aRBD-2 with aRBD-5 and aRBD-7, respectively. Here, we resolved the crystal structures of these Nbs in complex with the receptor-binding domain (RBD) of the spike protein, and found that aRBD-2 contacts with highly-conserved RBD residues and retains binding to the RBD of the Alpha, Beta, Gamma, Delta, Delta plus, Kappa, Lambda, Omicron BA.1, and BA.2 variants. In contrast, aRBD-5 and aRBD-7 bind to less-conserved RBD epitopes non-overlapping with the epitope of aRBD-2, and do not show apparent binding to the RBD of some variants. However, when fused with aRBD-2, they effectively enhance the overall binding affinity. Consistently, aRBD-2-5-Fc and aRBD-2-7-Fc potently neutralized all of the tested authentic or pseudotyped viruses, including WT, Alpha, Beta, Gamma, Delta, and Omicron BA.1, BA.1.1 and BA.2. Furthermore, aRBD-2-5-Fc provided prophylactic protection against the WT and mouse-adapted SARS-CoV-2 in mice, and conferred protection against the Omicron BA.1 variant in hamsters prophylactically and therapeutically, indicating that aRBD-2-5-Fc could potentially benefit the prevention and treatment of COVID-19 caused by the emerging variants of concern. Our strategy provides new solutions in the development of broad-spectrum therapeutic antibodies for COVID-19.

[end paper 91]

[begin paper 92]

PMCID = 7642358

Title = Selection, biophysical and structural analysis of synthetic nanobodies that effectively neutralize SARS-CoV-2

The coronavirus SARS-CoV-2 is the cause of the ongoing COVID-19 pandemic. Therapeutic neutralizing antibodies constitute a key short-to-medium term approach to tackle COVID-19. However, traditional antibody production is hampered by long development times and costly production. Here, we report the rapid isolation and characterization of nanobodies from a synthetic library, known as sybodies (Sb), that target the receptor-binding domain (RBD) of the SARS-CoV-2 spike protein. Several binders with low nanomolar affinities and efficient neutralization activity were identified of which Sb23 displayed high affinity and neutralized pseudovirus with an IC50 of 0.6 µg/ml. A cryo-EM structure of the spike bound to Sb23 showed that Sb23 binds competitively in the ACE2 binding site. Furthermore, the cryo-EM reconstruction revealed an unusual conformation of the spike where two RBDs are in the ‘up’ ACE2-binding conformation. The combined approach represents an alternative, fast workflow to select binders with neutralizing activity against newly emerging viruses.

Here, the authors isolate several nanobodies from a synthetic library that bind the receptor-binding domain (RBD) of SARS-CoV-2 spike protein (S) and neutralize S pseudotyped viruses. Cryo-EM structure of Spike with one nanobody and further biophysical analysis shows competition with ACE2 binding.

[end paper 92]

[begin paper 93]

PMCID = 9512078

Title = Mucosal nanobody IgA as inhalable and affordable prophylactic and therapeutic treatment against SARS-CoV-2 and emerging variants

Anti-COVID antibody therapeutics have been developed but not widely used due to their high cost and escape of neutralization from the emerging variants. Here, we describe the development of VHH-IgA1.1, a nanobody IgA fusion molecule as an inhalable, affordable and less invasive prophylactic and therapeutic treatment against SARS-CoV-2 Omicron variants. VHH-IgA1.1 recognizes a conserved epitope of SARS-CoV-2 spike protein Receptor Binding Domain (RBD) and potently neutralizes major global SARS-CoV-2 variants of concern (VOC) including the Omicron variant and its sub lineages BA.1.1, BA.2 and BA.2.12.1. VHH-IgA1.1 is also much more potent against Omicron variants as compared to an IgG Fc fusion construct, demonstrating the importance of IgA mediated mucosal protection for Omicron infection. Intranasal administration of VHH-IgA1.1 prior to or after challenge conferred significant protection from severe respiratory disease in K18-ACE2 transgenic mice infected with SARS-CoV-2 VOC. More importantly, for cost-effective production, VHH-IgA1.1 produced in Pichia pastoris had comparable potency to mammalian produced antibodies. Our study demonstrates that intranasal administration of affordably produced VHH-IgA fusion protein provides effective mucosal immunity against infection of SARS-CoV-2 including emerging variants.

[end paper 93]

[begin paper 94]

PMCID = 9288967

Title = Accelerating PERx reaction enables covalent nanobodies for potent neutralization of SARS-CoV-2 and variants

The long-lasting COVID-19 pandemic and increasing SARS-CoV-2 variants demand effective drugs for prophylactics and treatment. Protein-based biologics offer high specificity, yet their noncovalent interactions often lead to drug dissociation and incomplete inhibition. Here, we have developed covalent nanobodies capable of binding with SARS-CoV-2 irreversibly via a proximity-enabled reactive therapeutic (PERx) mechanism. A latent bioreactive amino acid (FFY) was designed and genetically encoded into nanobodies to accelerate the PERx reaction rate. Compared with the noncovalent wild-type nanobody, the FFY-incorporated covalent nanobodies neutralized both wild-type SARS-CoV-2 and its Alpha, Delta, Epsilon, Lambda, and Omicron variants with drastically higher potency. This PERx-enabled covalent-nanobody strategy and the related insights into increased potency can be valuable to developing effective therapeutics for various viral infections.

The long-lasting COVID-19 pandemic and increasing SARS-CoV-2 variants demand effective therapeutic strategies. Herein, we have engineered unique unnatural amino acids into proteins and developed covalent nanobodies capable of irreversibly binding with the viral spike protein. These covalent nanobodies are able to neutralize both SARS-CoV-2 and its different variants with drastic higher potency than traditional nanobodies, affording a potential prophylactic and medication for COVID-19. In addition, our strategy could provide an original general route to developing effective therapeutics for various other viral infections, such as influenza, hepatitis, AIDS, and anthrax.

Traditional nanobodies interact with their targets reversibly, leading to dissociation and incomplete inhibition. Wang and co-workers have designed a latent bioreactive amino acid (FFY) and genetically encoded it into nanobodies. Upon nanobody binding to the viral spike protein, FFY covalently reacts with spike via a proximity-enabled reactive therapeutic (PERx) mechanism, enabling the nanobody to bind to SARS-CoV-2 irreversibly and effectively block viral infection. The FFY-incorporated covalent nanobodies neutralized both wild-type SARS-CoV-2 and its various variants with drastically higher potency than the noncovalent wild-type nanobodies.

[end paper 94]

[begin paper 95]

PMCID = 10167778

Title = Isolation and characterization of single domain antibodies from banded houndshark (Triakis scyllium) targeting SARS-CoV-2 spike RBD protein

The COVID-19 pandemic has significantly impacted human health for three years. To mitigate the spread of SARS-CoV-2, the development of neutralizing antibodies has been accelerated, including the exploration of alternative antibody formats such as single-domain antibodies. In this study, we identified variable new antigen receptors (VNARs) specific for the receptor binding domain (RBD) of SARS-CoV-2 by immunizing a banded houndshark (Triakis scyllium) with recombinant wild-type RBD. Notably, the CoV2NAR-1 clone showed high binding affinities in the nanomolar range to various RBDs and demonstrated neutralizing activity against SARS-CoV-2 pseudoviruses. These results highlight the potential of the banded houndshark as an animal model for the development of VNAR-based therapeutics or diagnostics against future pandemics.

[end paper 95]

[begin paper 96]

PMCID = 9966208

Title = Potent Therapeutic Strategies for COVID-19 with Single-Domain Antibody Immunoliposomes Neutralizing SARS-CoV-2 and Lip/cGAMP Enhancing Protective Immunity

The worldwide spread of COVID-19 continues to impact our lives and has led to unprecedented damage to global health and the economy. This highlights the need for an efficient approach to rapidly develop therapeutics and prophylactics against SARS-CoV-2. We modified a single-domain antibody, SARS-CoV-2 VHH, to the surface of the liposomes. These immunoliposomes demonstrated a good neutralizing ability, but could also carry therapeutic compounds. Furthermore, we used the 2019-nCoV RBD-SD1 protein as an antigen with Lip/cGAMP as the adjuvant to immunize mice. Lip/cGAMP enhanced the immunity well. It was demonstrated that the combination of RBD-SD1 and Lip/cGAMP was an effective preventive vaccine. This work presented potent therapeutic anti-SARS-CoV-2 drugs and an effective vaccine to prevent the spread of COVID-19.

[end paper 96]

[begin paper 97]

PMCID = 11640846

Title = Backstage Heroes—Yeast in COVID-19 Research

The extremely rapid development of understanding and technology that led to the containment of the COVID-19 pandemic resulted from collaborative efforts in the fields of Betacoronavirus pandemicum (SARS-CoV-2) biology, pharmacology, vaccinology, and medicine. Perhaps surprisingly, much of the research was conducted using simple and efficient yeast models. In this manuscript, we describe how yeast, eukaryotic microorganisms, have been used to research this global challenge, focusing on the therapeutic potential of the studies discussed herein. Thus, we outline the role of yeast in studying viral protein interactions with the host cell proteome, including the binding of the SARS-CoV-2 virus spike protein to the human ACE2 receptor and its modulation. The production and exploration of viral antigens in yeast systems, which led to the development of two approved COVID-19 vaccines, are also detailed. Moreover, yeast platforms facilitating the discovery and production of single-domain antibodies (nanobodies) against SARS-CoV-2 are described. Methods guiding modern and efficient drug discovery are explained at length. In particular, we focus on studies designed to search for inhibitors of the main protease (Mpro), a unique target for anti-coronaviral therapies. We highlight the adaptability of the techniques used, providing opportunities for rapid modification and implementation alongside the evolution of the SARS-CoV-2 virus. Approaches introduced in yeast systems that may have universal potential application in studies of emerging viral diseases are also described.

[end paper 97]

[begin paper 98]

PMCID = 9020654

Title = Isolation, characterization, and structure-based engineering of a neutralizing nanobody against SARS-CoV-2

SARS-CoV-2 engages with human cells through the binding of its Spike receptor-binding domain (S-RBD) to the receptor ACE2. Molecular blocking of this engagement represents a proven strategy to treat COVID-19. Here, we report a single-chain antibody (nanobody, DL4) isolated from immunized alpaca with picomolar affinity to RBD. DL4 neutralizes SARS-CoV-2 pseudoviruses with an IC50 of 0.101 μg mL−1 (6.2 nM). A crystal structure of the DL4-RBD complex at 1.75-Å resolution unveils the interaction detail and reveals a direct competition mechanism for DL4's ACE2-blocking and hence neutralizing activity. The structural information allows us to rationally design a mutant with higher potency. Our work adds diversity of neutralizing nanobodies against SARS-CoV-2 and should encourage protein engineering to improve antibody affinities in general.

[end paper 98]

[begin paper 99]

PMCID = 7473855

Title = An alpaca nanobody neutralizes SARS-CoV-2 by blocking receptor interaction

SARS-CoV-2 enters host cells through an interaction between the spike glycoprotein and the angiotensin converting enzyme 2 (ACE2) receptor. Directly preventing this interaction presents an attractive possibility for suppressing SARS-CoV-2 replication. Here, we report the isolation and characterization of an alpaca-derived single domain antibody fragment, Ty1, that specifically targets the receptor binding domain (RBD) of the SARS-CoV-2 spike, directly preventing ACE2 engagement. Ty1 binds the RBD with high affinity, occluding ACE2. A cryo-electron microscopy structure of the bound complex at 2.9 Å resolution reveals that Ty1 binds to an epitope on the RBD accessible in both the ‘up’ and ‘down’ conformations, sterically hindering RBD-ACE2 binding. While fusion to an Fc domain renders Ty1 extremely potent, Ty1 neutralizes SARS-CoV-2 spike pseudovirus as a 12.8 kDa nanobody, which can be expressed in high quantities in bacteria, presenting opportunities for manufacturing at scale. Ty1 is therefore an excellent candidate as an intervention against COVID-19.

Here, Hanke et al. immunize an alpaca with SARS-CoV-2 spike protein domains and identify a nanobody that binds the receptor binding domain of spike in both the up and down conformations and sterically hinders ACE2 engagement.

[end paper 99]

[begin paper 100]

PMCID = 8928829

Title = Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19

There remains an unmet need for globally deployable, low-cost therapeutics for the ongoing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic. Previously, we reported on the isolation and in vitro characterization of a potent single-domain nanobody, NIH-CoVnb-112, specific for the receptor-binding domain (RBD) of SARS-CoV-2. Here, we report on the molecular basis for the observed broad in vitro neutralization capability of NIH-CoVnb-112 against variant SARS-CoV-2 pseudoviruses. The structure of NIH-CoVnb-112 bound to SARS-CoV-2 RBD reveals a large contact surface area overlapping the angiotensin converting enzyme 2 (ACE2) binding site, which is largely unencumbered by the common RBD mutations. In an in vivo pilot study, we demonstrate effective reductions in weight loss, viral burden, and lung pathology in a Syrian hamster model of COVID-19 following nebulized delivery of NIH-CoVnb-112. These findings support the further development of NIH-CoVnb-112 as a potential adjunct preventative therapeutic for the treatment of SARS-CoV-2 infection.

Abbreviations: ACE2 – angiotensin converting enzyme 2BSA – buried surface areaCDR – complementary determining regionRBD – receptor binding domainRBM – receptor-binding motifSARS-CoV-2 - severe acute respiratory syndrome coronavirus 2

[end paper 100]



