## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "nanobodies" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on nanobodies to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on nanobodies using the query "nanobody affinity maturation AND SARS-CoV-2" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to nanobodies. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, I have identified the following papers as directly relevant to nanobodies and their application in designing SARS-CoV-2 nanobody binders. Here is the Python dictionary mapping PMCID to the title of the papers I wish to fine-tune myself on:

```python
{
    "8651292": "Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape",
    "8877688": "Allosteric Determinants of the SARS-CoV-2 Spike Protein Binding with Nanobodies: Examining Mechanisms of Mutational Escape and Sensitivity of the Omicron Variant",
    "8951411": "Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering",
    "8448731": "A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies",
    "8223476": "Directed evolution of potent neutralizing nanobodies against SARS-CoV-2 using CDR-swapping mutagenesis",
    "8928829": "Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19",
    "10588750": "A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern",
    "8354634": "The development of Nanosota-1 as anti-SARS-CoV-2 nanobody drug candidates",
    "9365158": "A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike",
    "9351521": "Correlation between the binding affinity and the conformational entropy of nanobody SARS-CoV-2 spike protein complexes",
    "11140877": "Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses",
    "9529347": "Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2",
    "9676808": "Nanobody-peptide-conjugate (NPC) for passive immunotherapy against SARS-CoV-2 variants of concern (VoC): a prospective pan-coronavirus therapeutics",
    "10268213": "Fully synthetic platform to rapidly generate tetravalent bispecific nanobody–based immunoglobulins",
    "9884143": "Expanding and improving nanobody repertoires using a yeast display method: Targeting SARS-CoV-2",
    "9479348": "A potent synthetic nanobody with broad-spectrum activity neutralizes SARS-CoV-2 virus and the Omicron variant BA.1 through a unique binding mode",
    "11407660": "Dual-role epitope on SARS-CoV-2 spike enhances and neutralizes viral entry across different variants",
    "8956255": "Multivariate mining of an alpaca immune repertoire identifies potent cross-neutralizing SARS-CoV-2 nanobodies",
    "9257560": "A panel of nanobodies recognizing conserved hidden clefts of all SARS-CoV-2 spike variants including Omicron",
    "8730734": "Outlook of therapeutic and diagnostic competency of nanobodies against SARS-CoV-2: A systematic review",
    "8597880": "Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological and pathological disease in a Syrian hamster model of COVID-19",
    "7685322": "The Development of a Novel Nanobody Therapeutic for SARS-CoV-2",
    "10966652": "Quantitative Assessment of Energetic Contributions of Residues in a SARS-CoV-2 Viral Enzyme/Nanobody Interface",
    "9801699": "Applications of nanobodies in the prevention, detection, and treatment of the evolving SARS-CoV-2",
    "8906176": "Monobodies with potent neutralizing activity against SARS-CoV-2 Delta and other variants of concern",
    "9170159": "Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants",
    "9524272": "Isolation of an escape-resistant SARS-CoV-2 neutralizing nanobody from a novel synthetic nanobody library",
    "8863326": "A potent alpaca-derived nanobody that neutralizes SARS-CoV-2 variants",
    "7473855": "An alpaca nanobody neutralizes SARS-CoV-2 by blocking receptor interaction",
    "8061467": "Integrative overview of antibodies against SARS-CoV-2 and their possible applications in COVID-19 prophylaxis and treatment",
    "8971115": "Circular RNA vaccines against SARS-CoV-2 and emerging variants",
    "11609455": "Nanobodies: From High-Throughput Identification to Therapeutic Development",
    "8463502": "Rapid generation of potent antibodies by autonomous hypermutation in yeast",
    "10275483": "Emerging Landscape of Nanobodies and Their Neutralizing Applications against SARS-CoV-2 Virus",
    "7959386": "A high-affinity RBD-targeting nanobody improves fusion partner’s potency against SARS-CoV-2",
    "7857409": "An ultrapotent synthetic nanobody neutralizes SARS-CoV-2 by stabilizing inactive Spike",
    "9132424": "Deep mutational engineering of broadly-neutralizing nanobodies accommodating SARS-CoV-1 and 2 antigenic drift",
    "9393763": "Antibody-mediated immunity to SARS-CoV-2 spike",
    "8982573": "Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance",
    "8646660": "Uncovering a conserved vulnerability site in SARS‐CoV‐2 by a human antibody",
    "11793827": "Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies",
    "8344751": "Development of potent and effective synthetic SARS-CoV-2 neutralizing nanobodies",
    "8771608": "Structure genomics of SARS-CoV-2 and its Omicron variant: drug design templates for COVID-19",
    "8248807": "SARS-CoV-2: Origin, Evolution, and Targeting Inhibition",
    "8724751": "Monoclonal antibodies for COVID-19 therapy and SARS-CoV-2 detection",
    "8260353": "Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants",
    "7833398": "Structural Analysis of Neutralizing Epitopes of the SARS-CoV-2 Spike to Guide Therapy and Vaccine Design Strategies",
    "8528929": "Discovery and characterization of high-affinity, potent SARS-CoV-2 neutralizing antibodies via single B cell screening",
    "11228396": "Structural basis of broad SARS-CoV-2 cross-neutralization by affinity-matured public antibodies",
    "9313406": "Plasmonic Approaches for the Detection of SARS-CoV-2 Viral Particles",
    "11640846": "Backstage Heroes—Yeast in COVID-19 Research",
    "8491763": "Mechanisms of SARS-CoV-2 entry into cells",
    "8447893": "Structural biology of SARS-CoV-2 and implications for therapeutic development",
    "7430568": "An ultra-potent synthetic nanobody neutralizes SARS-CoV-2 by locking Spike into an inactive conformation",
    "7556756": "Antibody-like proteins that capture and neutralize SARS-CoV-2",
    "9900178": "SARS-CoV-2 spike HexaPro formulated in aluminium hydroxide and administered in an accelerated vaccination schedule partially protects Syrian Hamsters against viral challenge despite low neutralizing antibody responses",
    "9896449": "Shark nanobodies with potent SARS-CoV-2 neutralizing activity and broad sarbecovirus reactivity",
    "11209230": "Comprehensive Overview of Broadly Neutralizing Antibodies against SARS-CoV-2 Variants",
    "10695562": "Lys417 acts as a molecular switch that regulates the conformation of SARS-CoV-2 spike protein",
    "8853550": "Structural and antigenic variations in the spike protein of emerging SARS-CoV-2 variants",
    "10051120": "Biomimetic Nanotechnology for SARS-CoV-2 Treatment",
    "8625641": "Structural Biology of Nanobodies against the Spike Protein of SARS-CoV-2",
    "8193598": "Molecular mechanism of interaction between SARS-CoV-2 and host cells and interventional therapy",
    "10702856": "In vitro generated antibodies guide thermostable ADDomer nanoparticle design for nasal vaccination and passive immunization against SARS-CoV-2",
    "11148769": "A structure-function analysis shows SARS-CoV-2 BA.2.86 balances antibody escape and ACE2 affinity",
    "9181103": "Newly Emerged Antiviral Strategies for SARS-CoV-2: From Deciphering Viral Protein Structural Function to the Development of Vaccines, Antibodies, and Small Molecules",
    "8368053": "Dynamic landscape mapping of humoral immunity to SARS-CoV-2 identifies non-structural protein antibodies associated with the survival of critical COVID-19 patients",
    "9850341": "The Potential of Nanobodies for COVID-19 Diagnostics and Therapeutics",
    "8782626": "Monoclonal antibodies targeting two immunodominant epitopes on the Spike protein neutralize emerging SARS-CoV-2 variants of concern",
    "8325376": "Nanobodies as powerful pulmonary targeted biotherapeutics against SARS-CoV-2, pharmaceutical point of view",
    "10148623": "Perturbation of the host cell Ca2+ homeostasis and ER-mitochondria contact sites by the SARS-CoV-2 structural proteins E and M",
    "11788755": "Nanobody engineering: computational modelling and design for biomedical and therapeutic applications",
    "9941073": "Emerging trends in point-of-care biosensing strategies for molecular architectures and antibodies of SARS-CoV-2",
    "11228696": "Current development of severe acute respiratory syndrome coronavirus 2 neutralizing antibodies (Review)",
    "8301790": "COVID-19: Unmasking Emerging SARS-CoV-2 Variants, Vaccines and Therapeutic Strategies",
    "11003867": "Antibody-independent protection against heterologous SARS-CoV-2 challenge conferred by prior infection or vaccination",
    "10891761": "Serial Llama Immunization with Various SARS-CoV-2 RBD Variants Induces Broad Spectrum Virus-Neutralizing Nanobodies",
    "8324831": "A synthetic nanobody targeting RBD protects hamsters from SARS-CoV-2 infection",
    "8564274": "A non-ACE2 competing human single-domain antibody confers broad neutralization against SARS-CoV-2 and circulating variants",
    "8006950": "Lessons learned 1 year after SARS-CoV-2 emergence leading to COVID-19 pandemic",
    "8476643": "Cross-neutralizing antibodies bind a SARS-CoV-2 cryptic site and resist circulating variants",
    "10315470": "SARS-CoV-2 and the host-immune response",
    "9407951": "An Antibody from Single Human VH-rearranging Mouse Neutralizes All SARS-CoV-2 Variants Through BA.5 by Inhibiting Membrane Fusion",
    "9355996": "Conformational flexibility in neutralization of SARS-CoV-2 by naturally elicited anti-SARS-CoV-2 antibodies",
    "8139655": "Potent Neutralization of SARS-CoV-2 by Hetero-Bivalent Alpaca Nanobodies Targeting the Spike Receptor-Binding Domain",
    "10373554": "Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening",
    "7822889": "A COVID-19 vaccine candidate using SpyCatcher multimerization of the SARS-CoV-2 spike protein receptor-binding domain induces potent neutralising antibody responses",
    "8201996": "Potent Molecular Feature-based Neutralizing Monoclonal Antibodies as Promising Therapeutics Against SARS-CoV-2 Infection",
    "9688389": "Angiotensin-Converting Enzyme 2-Based Biosensing Modalities and Devices for Coronavirus Detection",
    "9481167": "Biosynthetic proteins targeting the SARS-CoV-2 spike as anti-virals",
    "10060843": "Review of therapeutic mechanisms and applications based on SARS-CoV-2 neutralizing antibodies",
    "9169858": "An extended conformation of SARS-CoV-2 main protease reveals allosteric targets",
    "11100608": "Bispecific antibodies with broad neutralization potency against SARS-CoV-2 variants of concern",
    "8606259": "Emerging mutations in the SARS-CoV-2 variants and their role in antibody escape to small molecule-based therapeutic resistance",
    "8221914": "Ultrapotent miniproteins targeting the SARS-CoV-2 receptor-binding domain protect against infection and disease",
    "9491202": "Sensitivity to Vaccines, Therapeutic Antibodies, and Viral Entry Inhibitors and Advances To Counter the SARS-CoV-2 Omicron Variant",
    "9174178": "Superimmunity by pan-sarbecovirus nanobodies",
    "10578311": "Comparative Study of the Mutations Observed in the SARS-CoV-2 RBD Variants of Concern and Their Impact on the Interaction with the ACE2 Protein"
}
```

These papers focus on the development, engineering, and application of nanobodies against SARS-CoV-2, which are directly relevant to my research project.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "nanobody affinity maturation AND SARS-CoV-2":

[begin paper 1]

PMCID = 8651292

Title = Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape

The emergence of SARS-CoV-2 variants threatens current vaccines and therapeutic antibodies and urgently demands powerful new therapeutics that can resist viral escape. We therefore generated a large nanobody repertoire to saturate the distinct and highly conserved available epitope space of SARS-CoV-2 spike, including the S1 receptor binding domain, N-terminal domain, and the S2 subunit, to identify new nanobody binding sites that may reflect novel mechanisms of viral neutralization. Structural mapping and functional assays show that indeed these highly stable monovalent nanobodies potently inhibit SARS-CoV-2 infection, display numerous neutralization mechanisms, are effective against emerging variants of concern, and are resistant to mutational escape. Rational combinations of these nanobodies that bind to distinct sites within and between spike subunits exhibit extraordinary synergy and suggest multiple tailored therapeutic and prophylactic strategies.

[end paper 1]

[begin paper 2]

PMCID = 8877688

Title = Allosteric Determinants of the SARS-CoV-2 Spike Protein Binding with Nanobodies: Examining Mechanisms of Mutational Escape and Sensitivity of the Omicron Variant

Structural and biochemical studies have recently revealed a range of rationally engineered nanobodies with efficient neutralizing capacity against the SARS-CoV-2 virus and resilience against mutational escape. In this study, we performed a comprehensive computational analysis of the SARS-CoV-2 spike trimer complexes with single nanobodies Nb6, VHH E, and complex with VHH E/VHH V nanobody combination. We combined coarse-grained and all-atom molecular simulations and collective dynamics analysis with binding free energy scanning, perturbation-response scanning, and network centrality analysis to examine mechanisms of nanobody-induced allosteric modulation and cooperativity in the SARS-CoV-2 spike trimer complexes with these nanobodies. By quantifying energetic and allosteric determinants of the SARS-CoV-2 spike protein binding with nanobodies, we also examined nanobody-induced modulation of escaping mutations and the effect of the Omicron variant on nanobody binding. The mutational scanning analysis supported the notion that E484A mutation can have a significant detrimental effect on nanobody binding and result in Omicron-induced escape from nanobody neutralization. Our findings showed that SARS-CoV-2 spike protein might exploit the plasticity of specific allosteric hotspots to generate escape mutants that alter response to binding without compromising activity. The network analysis supported these findings showing that VHH E/VHH V nanobody binding can induce long-range couplings between the cryptic binding epitope and ACE2-binding site through a broader ensemble of communication paths that is less dependent on specific mediating centers and therefore may be less sensitive to mutational perturbations of functional residues. The results suggest that binding affinity and long-range communications of the SARS-CoV-2 complexes with nanobodies can be determined by structurally stable regulatory centers and conformationally adaptable hotspots that are allosterically coupled and collectively control resilience to mutational escape.

[end paper 2]

[begin paper 3]

PMCID = 8951411

Title = Structural and Computational Studies of the SARS-CoV-2 Spike Protein Binding Mechanisms with Nanobodies: From Structure and Dynamics to Avidity-Driven Nanobody Engineering

Nanobodies provide important advantages over traditional antibodies, including their smaller size and robust biochemical properties such as high thermal stability, high solubility, and the ability to be bioengineered into novel multivalent, multi-specific, and high-affinity molecules, making them a class of emerging powerful therapies against SARS-CoV-2. Recent research efforts on the design, protein engineering, and structure-functional characterization of nanobodies and their binding with SARS-CoV-2 S proteins reflected a growing realization that nanobody combinations can exploit distinct binding epitopes and leverage the intrinsic plasticity of the conformational landscape for the SARS-CoV-2 S protein to produce efficient neutralizing and mutation resistant characteristics. Structural and computational studies have also been instrumental in quantifying the structure, dynamics, and energetics of the SARS-CoV-2 spike protein binding with nanobodies. In this review, a comprehensive analysis of the current structural, biophysical, and computational biology investigations of SARS-CoV-2 S proteins and their complexes with distinct classes of nanobodies targeting different binding sites is presented. The analysis of computational studies is supplemented by an in-depth examination of mutational scanning simulations and identification of binding energy hotspots for distinct nanobody classes. The review is focused on the analysis of mechanisms underlying synergistic binding of multivalent nanobodies that can be superior to single nanobodies and conventional nanobody cocktails in combating escape mutations by effectively leveraging binding avidity and allosteric cooperativity. We discuss how structural insights and protein engineering approaches together with computational biology tools can aid in the rational design of synergistic combinations that exhibit superior binding and neutralization characteristics owing to avidity-mediated mechanisms.

[end paper 3]

[begin paper 4]

PMCID = 8448731

Title = A cell-free nanobody engineering platform rapidly generates SARS-CoV-2 neutralizing nanobodies

Antibody engineering technologies face increasing demands for speed, reliability and scale. We develop CeVICA, a cell-free nanobody engineering platform that uses ribosome display for in vitro selection of nanobodies from a library of 1011 randomized sequences. We apply CeVICA to engineer nanobodies against the Receptor Binding Domain (RBD) of SARS-CoV-2 spike protein and identify >800 binder families using a computational pipeline based on CDR-directed clustering. Among 38 experimentally-tested families, 30 are true RBD binders and 11 inhibit SARS-CoV-2 pseudotyped virus infection. Affinity maturation and multivalency engineering increase nanobody binding affinity and yield a virus neutralizer with picomolar IC50. Furthermore, the capability of CeVICA for comprehensive binder prediction allows us to validate the fitness of our nanobody library. CeVICA offers an integrated solution for rapid generation of divergent synthetic nanobodies with tunable affinities in vitro and may serve as the basis for automated and highly parallel nanobody engineering.

Faster, higher throughput antibody engineering methods are needed. Here the authors present CeVICA, a cell-free nanobody engineering platform using ribosome display and computational clustering analysis for in vitro selection; they use this to develop nanobodies against the RBD of SARS-CoV-2 spike protein.

[end paper 4]

[begin paper 5]

PMCID = 8223476

Title = Directed evolution of potent neutralizing nanobodies against SARS-CoV-2 using CDR-swapping mutagenesis

There is widespread interest in facile methods for generating potent neutralizing antibodies, nanobodies, and other affinity proteins against SARS-CoV-2 and related viruses to address current and future pandemics. While isolating antibodies from animals and humans are proven approaches, these methods are limited to the affinities, specificities, and functional activities of antibodies generated by the immune system. Here we report a surprisingly simple directed evolution method for generating nanobodies with high affinities and neutralization activities against SARS-CoV-2. We demonstrate that complementarity-determining region swapping between low-affinity lead nanobodies, which we discovered unintentionally but find is simple to implement systematically, results in matured nanobodies with unusually large increases in affinity. Importantly, the matured nanobodies potently neutralize both SARS-CoV-2 pseudovirus and live virus, and possess drug-like biophysical properties. We expect that our methods will improve in vitro nanobody discovery and accelerate the generation of potent neutralizing nanobodies against diverse coronaviruses.

Zupancic et al. report potent neutralizing nanobodies against SARS-CoV-2. They demonstrate an approach that involves swapping the complementarity-determining regions of low-affinity clones to generate matured nanobodies with large increases in affinity and neutralization activity.

[end paper 5]

[begin paper 6]

PMCID = 8928829

Title = Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological, and pathological disease in a Syrian hamster model of COVID-19

There remains an unmet need for globally deployable, low-cost therapeutics for the ongoing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic. Previously, we reported on the isolation and in vitro characterization of a potent single-domain nanobody, NIH-CoVnb-112, specific for the receptor-binding domain (RBD) of SARS-CoV-2. Here, we report on the molecular basis for the observed broad in vitro neutralization capability of NIH-CoVnb-112 against variant SARS-CoV-2 pseudoviruses. The structure of NIH-CoVnb-112 bound to SARS-CoV-2 RBD reveals a large contact surface area overlapping the angiotensin converting enzyme 2 (ACE2) binding site, which is largely unencumbered by the common RBD mutations. In an in vivo pilot study, we demonstrate effective reductions in weight loss, viral burden, and lung pathology in a Syrian hamster model of COVID-19 following nebulized delivery of NIH-CoVnb-112. These findings support the further development of NIH-CoVnb-112 as a potential adjunct preventative therapeutic for the treatment of SARS-CoV-2 infection.

Abbreviations: ACE2 – angiotensin converting enzyme 2BSA – buried surface areaCDR – complementary determining regionRBD – receptor binding domainRBM – receptor-binding motifSARS-CoV-2 - severe acute respiratory syndrome coronavirus 2

[end paper 6]

[begin paper 7]

PMCID = 10588750

Title = A Naïve Phage Display Library-Derived Nanobody Neutralizes SARS-CoV-2 and Three Variants of Concern

The emergence of the coronavirus disease 2019 (COVID-19) pandemic and the new severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) variants of concern (VOCs) requires the continuous development of safe, effective, and affordable prevention and therapeutics. Nanobodies have demonstrated antiviral activity against a variety of viruses, providing a new candidate for the prevention and treatment of SARS-CoV-2 and its variants.

SARS-CoV-2 glycoprotein spike 1 subunit (S1) was selected as the target antigen for nanobody screening of a naïve phage display library. We obtained a nanobody, named Nb-H6, and then determined its affinity, inhibition, and stability by ELISA, Competitive ELISA, and Biolayer Interferometry (BLI). Infection assays of authentic and pseudotyped SARS-CoV-2 were performed to evaluate the neutralization of Nb-H6. The structure and mechanism of action were investigated by AlphaFold, docking, and residue mutation assays.

We isolated and characterized a nanobody, Nb-H6, which exhibits a broad affinity for S1 and the receptor binding domain (RBD) of SARS-CoV-2, or Alpha (B.1.1.7), Delta (B.1.617.2), Lambda (C.37), and Omicron (BA.2 and BA.5), and blocks receptor angiotensin-converting enzyme 2 (ACE2) binding. Moreover, Nb-H6 can retain its binding capability after pH or thermal treatment and effectively neutralize both pseudotyped and authentic SARS-CoV-2, as well as VOC Alpha (B.1.1.7), Delta (B.1.617.2), and Omicron (BA.2 and BA.5) pseudoviruses. We also confirmed that Nb-H6 binds two distinct amino acid residues of the RBD, preventing SARS-CoV-2 from interacting with the host receptor.

Our study highlights a novel nanobody, Nb-H6, that may be useful therapeutically in SARS-CoV-2 and VOC outbreaks and pandemics. These findings also provide a molecular foundation for further studies into how nanobodies neutralize SARS-CoV-2 and variants and imply potential therapeutic targets for the treatment of COVID-19.

[end paper 7]

[begin paper 8]

PMCID = 8354634

Title = The development of Nanosota-1 as anti-SARS-CoV-2 nanobody drug candidates

Combating the COVID-19 pandemic requires potent and low-cost therapeutics. We identified a series of single-domain antibodies (i.e., nanobody), Nanosota-1, from a camelid nanobody phage display library. Structural data showed that Nanosota-1 bound to the oft-hidden receptor-binding domain (RBD) of SARS-CoV-2 spike protein, blocking viral receptor angiotensin-converting enzyme 2 (ACE2). The lead drug candidate possessing an Fc tag (Nanosota-1C-Fc) bound to SARS-CoV-2 RBD ~3000 times more tightly than ACE2 did and inhibited SARS-CoV-2 pseudovirus ~160 times more efficiently than ACE2 did. Administered at a single dose, Nanosota-1C-Fc demonstrated preventive and therapeutic efficacy against live SARS-CoV-2 infection in both hamster and mouse models. Unlike conventional antibodies, Nanosota-1C-Fc was produced at high yields in bacteria and had exceptional thermostability. Pharmacokinetic analysis of Nanosota-1C-Fc documented an excellent in vivo stability and a high tissue bioavailability. As effective and inexpensive drug candidates, Nanosota-1 may contribute to the battle against COVID-19.

[end paper 8]

[begin paper 9]

PMCID = 9365158

Title = A humanized nanobody phage display library yields potent binders of SARS CoV-2 spike

Neutralizing antibodies targeting the SARS-CoV-2 spike protein have shown a great preventative/therapeutic potential. Here, we report a rapid and efficient strategy for the development and design of SARS-CoV-2 neutralizing humanized nanobody constructs with sub-nanomolar affinities and nanomolar potencies. CryoEM-based structural analysis of the nanobodies in complex with spike revealed two distinct binding modes. The most potent nanobody, RBD-1-2G(NCATS-BL8125), tolerates the N501Y RBD mutation and remains capable of neutralizing the B.1.1.7 (Alpha) variant. Molecular dynamics simulations provide a structural basis for understanding the neutralization process of nanobodies exclusively focused on the spike-ACE2 interface with and without the N501Y mutation on RBD. A primary human airway air-lung interface (ALI) ex vivo model showed that RBD-1-2G-Fc antibody treatment was effective at reducing viral burden following WA1 and B.1.1.7 SARS-CoV-2 infections. Therefore, this presented strategy will serve as a tool to mitigate the threat of emerging SARS-CoV-2 variants.

[end paper 9]

[begin paper 10]

PMCID = 9351521

Title = Correlation between the binding affinity and the conformational entropy of nanobody SARS-CoV-2 spike protein complexes

Understanding the structural principles that determine the binding affinity of nanobodies to the spike protein of severe acute respiratory syndrome coronavirus 2 has been difficult. We analyzed electron microscopy maps of nanobody-spike complexes and quantified the conformational entropy of binding. This informed the design of an engineered nanobody with improved binding to the spike protein. This result offers a guiding principle for the rational maturation of nanobodies directed against the spike. High-binding potency nanobodies have been shown to be effective in animal models; thus, this technology could have application in future pandemics.

Camelid single-domain antibodies, also known as nanobodies, can be readily isolated from naïve libraries for specific targets but often bind too weakly to their targets to be immediately useful. Laboratory-based genetic engineering methods to enhance their affinity, termed maturation, can deliver useful reagents for different areas of biology and potentially medicine. Using the receptor binding domain (RBD) of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike protein and a naïve library, we generated closely related nanobodies with micromolar to nanomolar binding affinities. By analyzing the structure–activity relationship using X-ray crystallography, cryoelectron microscopy, and biophysical methods, we observed that higher conformational entropy losses in the formation of the spike protein–nanobody complex are associated with tighter binding. To investigate this, we generated structural ensembles of the different complexes from electron microscopy maps and correlated the conformational fluctuations with binding affinity. This insight guided the engineering of a nanobody with improved affinity for the spike protein.

[end paper 10]

[begin paper 11]

PMCID = 11140877

Title = Therapeutic nanobodies against SARS-CoV-2 and other pathogenic human coronaviruses

Nanobodies, single-domain antibodies derived from variable domain of camelid or shark heavy-chain antibodies, have unique properties with small size, strong binding affinity, easy construction in versatile formats, high neutralizing activity, protective efficacy, and manufactural capacity on a large-scale. Nanobodies have been arisen as an effective research tool for development of nanobiotechnologies with a variety of applications. Three highly pathogenic coronaviruses (CoVs), SARS-CoV-2, SARS-CoV, and MERS-CoV, have caused serious outbreaks or a global pandemic, and continue to post a threat to public health worldwide. The viral spike (S) protein and its cognate receptor-binding domain (RBD), which initiate viral entry and play a critical role in virus pathogenesis, are important therapeutic targets. This review describes pathogenic human CoVs, including viral structures and proteins, and S protein-mediated viral entry process. It also summarizes recent advances in development of nanobodies targeting these CoVs, focusing on those targeting the S protein and RBD. Finally, we discuss potential strategies to improve the efficacy of nanobodies against emerging SARS-CoV-2 variants and other CoVs with pandemic potential. It will provide important information for rational design and evaluation of therapeutic agents against emerging and reemerging pathogens.

[end paper 11]

[begin paper 12]

PMCID = 9529347

Title = Biparatopic nanobodies targeting the receptor binding domain efficiently neutralize SARS-CoV-2

The development of therapeutics to prevent or treat COVID-19 remains an area of intense focus. Protein biologics, including monoclonal antibodies and nanobodies that neutralize virus, have potential for the treatment of active disease. Here, we have used yeast display of a synthetic nanobody library to isolate nanobodies that bind the receptor-binding domain (RBD) of SARS-CoV-2 and neutralize the virus. We show that combining two clones with distinct binding epitopes within the RBD into a single protein construct to generate biparatopic reagents dramatically enhances their neutralizing capacity. Furthermore, the biparatopic nanobodies exhibit enhanced control over clinically relevant RBD variants that escaped recognition by the individual nanobodies. Structural analysis of biparatopic binding to spike (S) protein revealed a unique binding mode whereby the two nanobody paratopes bridge RBDs encoded by distinct S trimers. Accordingly, biparatopic nanobodies offer a way to rapidly generate powerful viral neutralizers with enhanced ability to control viral escape mutants.

Yeast display was used to generate SARS-CoV-2 RBD-specific nanobodies

Distinct nanobody clones were fused to create biparatopic reagents

Biparatopic nanobodies protect mice from SARS-CoV-2 infection

The biparatopics bridge distinct S proteins in a unique “twinned-S” conformation

Immunology; Microbiology; Structural biology.

[end paper 12]

[begin paper 13]

PMCID = 9676808

Title = Nanobody-peptide-conjugate (NPC) for passive immunotherapy against SARS-CoV-2 variants of concern (VoC): a prospective pan-coronavirus therapeutics

The COVID-19 crisis, incited by the zoonotic SARS-CoV-2 virus, has quickly escalated into a catastrophic public health issue and a grave threat to humankind owing to the advent of mutant viruses. Multiple pharmaceutical therapies or biologics envision stopping the virus from spreading further; however, WHO has voiced concerns about the variants of concern (VoCs) inability to respond. Nanobodies are a new class of antibody mimics with binding affinity and specificity similar to classical mAbs, as well as the privileges of a small molecular weight, ease of entry into solid tissues, and binding cryptic epitopes of the antigen. Herein, we investigated multiple putative anti-SARS-CoV-2 nanobodies targeting the Receptor binding domain of the WHO-listed SARS-CoV-2 variants of concern using a comprehensive computational immunoinformatics methodology. With affinity maturation via alanine scanning mutagenesis, we remodeled an ultrapotent nanobody with substantial breadth and potency, exhibiting pico-molar binding affinities against all the VoCs. An antiviral peptide with specificity for ACE-2 receptors was affixed to make it multispecific and discourage viral entry. Collectively, we constructed a broad-spectrum therapeutic biparatopic nanobody-peptide conjugate (NPC) extending coverage to SARS-CoV-2 VoCs RBDs. We PEGylated the biparatopic construct with 20kD maleimide-terminated PEG (MAL-(PEG)n-OMe) to improve its clinical efficacy limiting rapid renal clearance, and performed in silico cloning to facilitate future experimental studies. Our findings suggest that combining biparatopic nanobody conjugate with standard treatment may be a promising bivariate tool for combating viral entry during COVID-19 illness.

The online version contains supplementary material available at 10.1007/s11030-022-10570-x.

[end paper 13]

[begin paper 14]

PMCID = 10268213

Title = Fully synthetic platform to rapidly generate tetravalent bispecific nanobody–based immunoglobulins

Nanobodies are a promising class of biologics that can be used to prevent or treat viral infections. Here, we describe the production and validation of a discovery library that produces single-domain nanobodies using an engineered human antibody variable gene segment. As a test case, anti-SARS-CoV-2 nanobodies were isolated from this library and pairs of complementary nanobodies were incorporated into an antibody-like molecule that targets the receptor-binding domain using a biparatopic mode of engagement. This modular bispecific format enabled the rapid testing of nanobody pairs, and we show that incorporating pairs of nanobodies with different specificities can have synergistic effects on neutralization breadth and potency.

Nanobodies bind a target antigen with a kinetic profile similar to a conventional antibody, but exist as a single heavy chain domain that can be readily multimerized to engage antigen via multiple interactions. Presently, most nanobodies are produced by immunizing camelids; however, platforms for animal-free production are growing in popularity. Here, we describe the development of a fully synthetic nanobody library based on an engineered human VH3-23 variable gene and a multispecific antibody-like format designed for biparatopic target engagement. To validate our library, we selected nanobodies against the SARS-CoV-2 receptor–binding domain and employed an on-yeast epitope binning strategy to rapidly map the specificities of the selected nanobodies. We then generated antibody-like molecules by replacing the VH and VL domains of a conventional antibody with two different nanobodies, designed as a molecular clamp to engage the receptor-binding domain biparatopically. The resulting bispecific tetra-nanobody immunoglobulins neutralized diverse SARS-CoV-2 variants with potencies similar to antibodies isolated from convalescent donors. Subsequent biochemical analyses confirmed the accuracy of the on-yeast epitope binning and structures of both individual nanobodies, and a tetra-nanobody immunoglobulin revealed that the intended mode of interaction had been achieved. This overall workflow is applicable to nearly any protein target and provides a blueprint for a modular workflow for the development of multispecific molecules.

[end paper 14]

[begin paper 15]

PMCID = 9884143

Title = Expanding and improving nanobody repertoires using a yeast display method: Targeting SARS-CoV-2

COVID-19, caused by the coronavirus SARS-CoV-2, represents a serious worldwide health issue, with continually emerging new variants challenging current therapeutics. One promising alternate therapeutic avenue is represented by nanobodies, small single-chain antibodies derived from camelids with numerous advantageous properties and the potential to neutralize the virus. For identification and characterization of a broad spectrum of anti–SARS-CoV-2 Spike nanobodies, we further optimized a yeast display method, leveraging a previously published mass spectrometry-based method, using B-cell complementary DNA from the same immunized animals as a source of VHH sequences. Yeast display captured many of the sequences identified by the previous approach, as well as many additional sequences that proved to encode a large new repertoire of nanobodies with high affinities and neutralization activities against different SARS-CoV-2 variants. We evaluated DNA shuffling applied to the three complementarity-determining regions of antiviral nanobodies. The results suggested a surprising degree of modularity to complementarity-determining region function. Importantly, the yeast display approach applied to nanobody libraries from immunized animals allows parallel interrogation of a vast number of nanobodies. For example, we employed a modified yeast display to carry out massively parallel epitope binning. The current yeast display approach proved comparable in efficiency and specificity to the mass spectrometry–based approach, while requiring none of the infrastructure and expertise required for that approach, making these highly complementary approaches that together appear to comprehensively explore the paratope space. The larger repertoires produced maximize the likelihood of discovering broadly specific reagents and those that powerfully synergize in mixtures.

[end paper 15]

[begin paper 16]

PMCID = 9043892

Title = Passive Immunotherapy Against SARS-CoV-2: From Plasma-Based Therapy to Single Potent Antibodies in the Race to Stay Ahead of the Variants

The COVID-19 pandemic is now approaching 2 years old, with more than 440 million people infected and nearly six million dead worldwide, making it the most significant pandemic since the 1918 influenza pandemic. The severity and significance of SARS-CoV-2 was recognized immediately upon discovery, leading to innumerable companies and institutes designing and generating vaccines and therapeutic antibodies literally as soon as recombinant SARS-CoV-2 spike protein sequence was available. Within months of the pandemic start, several antibodies had been generated, tested, and moved into clinical trials, including Eli Lilly’s bamlanivimab and etesevimab, Regeneron’s mixture of imdevimab and casirivimab, Vir’s sotrovimab, Celltrion’s regdanvimab, and Lilly’s bebtelovimab. These antibodies all have now received at least Emergency Use Authorizations (EUAs) and some have received full approval in select countries. To date, more than three dozen antibodies or antibody combinations have been forwarded into clinical trials. These antibodies to SARS-CoV-2 all target the receptor-binding domain (RBD), with some blocking the ability of the RBD to bind human ACE2, while others bind core regions of the RBD to modulate spike stability or ability to fuse to host cell membranes. While these antibodies were being discovered and developed, new variants of SARS-CoV-2 have cropped up in real time, altering the antibody landscape on a moving basis. Over the past year, the search has widened to find antibodies capable of neutralizing the wide array of variants that have arisen, including Alpha, Beta, Gamma, Delta, and Omicron. The recent rise and dominance of the Omicron family of variants, including the rather disparate BA.1 and BA.2 variants, demonstrate the need to continue to find new approaches to neutralize the rapidly evolving SARS-CoV-2 virus. This review highlights both convalescent plasma- and polyclonal antibody-based approaches as well as the top approximately 50 antibodies to SARS-CoV-2, their epitopes, their ability to bind to SARS-CoV-2 variants, and how they are delivered. New approaches to antibody constructs, including single domain antibodies, bispecific antibodies, IgA- and IgM-based antibodies, and modified ACE2-Fc fusion proteins, are also described. Finally, antibodies being developed for palliative care of COVID-19 disease, including the ramifications of cytokine release syndrome (CRS) and acute respiratory distress syndrome (ARDS), are described.

The online version contains supplementary material available at 10.1007/s40259-022-00529-7.

[end paper 16]

[begin paper 17]

PMCID = 9479348

Title = A potent synthetic nanobody with broad-spectrum activity neutralizes SARS-CoV-2 virus and the Omicron variant BA.1 through a unique binding mode

The major challenge to controlling the COVID pandemic is the rapid mutation rate of the SARS-CoV-2 virus, leading to the escape of the protection of vaccines and most of the neutralizing antibodies to date. Thus, it is essential to develop neutralizing antibodies with broad-spectrum activity targeting multiple SARS-CoV-2 variants. Here, we report a synthetic nanobody (named C5G2) obtained by phage display and subsequent antibody engineering. C5G2 has a single-digit nanomolar binding affinity to the RBD domain and inhibits its binding to ACE2 with an IC50 of 3.7 nM. Pseudovirus assays indicated that monovalent C5G2 could protect the cells from infection with SARS-CoV-2 wild-type virus and most of the viruses of concern, i.e., Alpha, Beta, Gamma and Omicron variants. Strikingly, C5G2 has the highest potency against Omicron BA.1 among all the variants, with an IC50 of 4.9 ng/mL. The cryo-EM structure of C5G2 in complex with the spike trimer showed that C5G2 binds to RBD mainly through its CDR3 at a conserved region that does not overlap with the ACE2 binding surface. Additionally, C5G2 binds simultaneously to the neighboring NTD domain of the spike trimer through the same CDR3 loop, which may further increase its potency against viral infection. Third, the steric hindrance caused by FR2 of C5G2 could inhibit the binding of ACE2 to RBD as well. Thus, this triple-function nanobody may serve as an effective drug for prophylaxis and therapy against Omicron as well as future variants.

The online version contains supplementary material available at 10.1186/s12951-022-01619-y.

[end paper 17]

[begin paper 18]

PMCID = 11407660

Title = Dual-role epitope on SARS-CoV-2 spike enhances and neutralizes viral entry across different variants

Grasping the roles of epitopes in viral glycoproteins is essential for unraveling the structure and function of these proteins. Up to now, all identified epitopes have been found to either neutralize, have no effect on, or enhance viral entry into cells. Here, we used nanobodies (single-domain antibodies) as probes to investigate a unique epitope on the SARS-CoV-2 spike protein, located outside the protein’s receptor-binding domain. Nanobody binding to this epitope enhances the cell entry of prototypic SARS-CoV-2, while neutralizing the cell entry of SARS-CoV-2 Omicron variant. Moreover, nanobody binding to this epitope promotes both receptor binding activity and post-attachment activity of prototypic spike, explaining the enhanced viral entry. The opposite occurs with Omicron spike, explaining the neutralized viral entry. This study reveals a unique epitope that can both enhance and neutralize viral entry across distinct viral variants, suggesting that epitopes may vary their roles depending on the viral context. Consequently, antibody therapies should be assessed across different viral variants to confirm their efficacy and safety.

Antibodies bind to specific epitopes on viral surface glycoproteins to perform their function: inhibiting viral infection, having no effect, or, in rare cases, enhancing viral infection. In our research, we used nanobodies (small, single-domain antibodies from camelid animals) as probes and identified a unique epitope on the SARS-CoV-2 spike protein with opposing functions across different SARS-CoV-2 variants. It enhances viral infection in pre-Omicron variants but inhibits it in the Omicron variant. We further investigated the molecular mechanism underlying these opposing effects. Although this epitope is not directly involved in receptor binding, nanobody binding to it modulates receptor binding and post-receptor-binding activities. This study marks the first discovery of a dual-function epitope on the SARS-CoV-2 spike protein that both enhances and inhibits viral infection across different viral variants. This discovery challenges the traditional categorization of epitopes, underscores the complex evolution of the SARS-CoV-2 spike protein, and offers new insights into antiviral antibody therapies.

[end paper 18]

[begin paper 19]

PMCID = 8956255

Title = Multivariate mining of an alpaca immune repertoire identifies potent cross-neutralizing SARS-CoV-2 nanobodies

Conventional approaches to isolate and characterize nanobodies are laborious. We combine phage display, multivariate enrichment, next-generation sequencing, and a streamlined screening strategy to identify numerous anti–severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) nanobodies. We characterize their potency and specificity using neutralization assays and hydrogen/deuterium exchange mass spectrometry (HDX-MS). The most potent nanobodies bind to the receptor binding motif of the receptor binding domain (RBD), and we identify two exceptionally potent members of this category (with monomeric half-maximal inhibitory concentrations around 13 and 16 ng/ml). Other nanobodies bind to a more conserved epitope on the side of the RBD and are able to potently neutralize the SARS-CoV-2 founder virus (42 ng/ml), the Beta variant (B.1.351/501Y.V2) (35 ng/ml), and also cross-neutralize the more distantly related SARS-CoV-1 (0.46 μg/ml). The approach presented here is well suited for the screening of phage libraries to identify functional nanobodies for various biomedical and biochemical applications.

High-throughput mining of camelid immune repertoires identified broad and potent nanobodies from an immunized alpaca.

[end paper 19]

[begin paper 20]

PMCID = 9257560

Title = A panel of nanobodies recognizing conserved hidden clefts of all SARS-CoV-2 spike variants including Omicron

We are amid the historic coronavirus infectious disease 2019 (COVID-19) pandemic. Imbalances in the accessibility of vaccines, medicines, and diagnostics among countries, regions, and populations, and those in war crises, have been problematic. Nanobodies are small, stable, customizable, and inexpensive to produce. Herein, we present a panel of nanobodies that can detect the spike proteins of five SARS-CoV-2 variants of concern (VOCs) including Omicron. Here we show via ELISA, lateral flow, kinetic, flow cytometric, microscopy, and Western blotting assays that our nanobodies can quantify the spike variants. This panel of nanobodies broadly neutralizes viral infection caused by pseudotyped and authentic SARS-CoV-2 VOCs. Structural analyses show that the P86 clone targets epitopes that are conserved yet unclassified on the receptor-binding domain (RBD) and contacts the N-terminal domain (NTD). Human antibodies rarely access both regions; consequently, the clone buries hidden crevasses of SARS-CoV-2 spike proteins that go undetected by conventional antibodies.

A panel of nanobodies are presented that can detect the spike proteins of five SARS-CoV-2 variants of concern and structural analyses show that one clone targets conserved epitopes on the receptor-binding domain and contacts the N-terminal domain.

[end paper 20]

[begin paper 21]

PMCID = 8730734

Title = Outlook of therapeutic and diagnostic competency of nanobodies against SARS-CoV-2: A systematic review

The newly emerged coronavirus (SARS-CoV-2) continues to infect humans, and no completely efficient treatment has yet been found. Antibody therapy is one way to control infection caused by COVID-19, but the use of classical antibodies has many disadvantages. Heavy chain antibodies (HCAbs) are single-domain antibodies derived from the Camelidae family. The variable part of these antibodies (Nanobodies or VHH) has interesting properties such as small size, identify criptic epitopes, stability in harsh conditions, good tissue permeability and cost-effective production causing nanobodies have become a good candidate in the treatment and diagnosis of viral infections.

Totally 157 records (up to November 10, 2021), were recognized to be reviewed in this study. 62 studies were removed after first step screening due to their deviation from inclusion criteria. The remaining 95 studies were reviewed in details. After removing articles that were not in the study area, 45 remaining studies met the inclusion criteria and were qualified to be included in the systematic review.

In this systematic review, the application of nanobodies in the treatment and detection of COVID-19 infection was reviewed. The results of this study showed that extensive and sufficient studies have been performed in the field of production of nanobodies against SARS-CoV-2 virus and the obtained nanobodies have a great potential for use in patients infected with SARS-CoV-2 virus.

According to the obtained results, it was found that nanobodies can be used effectively in the treatment and diagnosis of SARS-CoV-2 virus.

[end paper 21]

[begin paper 22]

PMCID = 8597880

Title = Nebulized delivery of a broadly neutralizing SARS-CoV-2 RBD-specific nanobody prevents clinical, virological and pathological disease in a Syrian hamster model of COVID-19

There remains an unmet need for globally deployable, low-cost therapeutics for the ongoing severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic. Previously, we reported on the isolation and in vitro characterization of a potent single-domain nanobody, NIH-CoVnb-112, specific for the receptor binding domain (RBD) of SARS-CoV-2. Here, we report on the molecular basis for the observed broad in vitro neutralization capability of NIH-CoVnb-112 against variant SARS-CoV-2 pseudoviruses, including the currently dominant Delta variant. The structure of NIH-CoVnb-112 bound to SARS-CoV-2 RBD reveals a large contact surface area overlapping the angiotensin converting enzyme 2 (ACE2) binding site, which is largely unencumbered by the common RBD mutations. In an in vivo pilot study, we demonstrate effective reductions in weight loss, viral burden, and lung pathology in a Syrian hamster model of COVID-19 following nebulized delivery of NIH-CoVnb-112. These findings support the further development of NIH-CoVnb-112 as a potential adjunct preventative therapeutic for the treatment of SARS-CoV-2 infection.

[end paper 22]

[begin paper 23]

PMCID = 8106883

Title = SARS-CoV-2 exacerbates proinflammatory responses in myeloid cells through C-type lectin receptors and Tweety family member 2

Despite mounting evidence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) engagement with immune cells, most express little, if any, of the canonical receptor of SARS-CoV-2, angiotensin-converting enzyme 2 (ACE2). Here, using a myeloid cell receptor-focused ectopic expression screen, we identified several C-type lectins (DC-SIGN, L-SIGN, LSECtin, ASGR1, and CLEC10A) and Tweety family member 2 (TTYH2) as glycan-dependent binding partners of the SARS-CoV-2 spike. Except for TTYH2, these molecules primarily interacted with spike via regions outside of the receptor-binding domain. Single-cell RNA sequencing analysis of pulmonary cells from individuals with coronavirus disease 2019 (COVID-19) indicated predominant expression of these molecules on myeloid cells. Although these receptors do not support active replication of SARS-CoV-2, their engagement with the virus induced robust proinflammatory responses in myeloid cells that correlated with COVID-19 severity. We also generated a bispecific anti-spike nanobody that not only blocked ACE2-mediated infection but also the myeloid receptor-mediated proinflammatory responses. Our findings suggest that SARS-CoV-2-myeloid receptor interactions promote immune hyperactivation, which represents potential targets for COVID-19 therapy.

Most immune cells express little, if any, of the canonical SARS-CoV-2 receptor, ACE2. Lu et al. report that C-type lectins and TTYH2 act as SARS-CoV-2 myeloid cell-interacting partners that trigger immune hyperactivation but not infection. These findings raise the possibility that these virus-myeloid cell interactions are directly involved in COVID-19 immunopathogenesis and could be targeted for COVID-19 therapy.

[end paper 23]

[begin paper 24]

PMCID = 7685322

Title = The Development of a Novel Nanobody Therapeutic for SARS-CoV-2

Combating the COVID-19 pandemic requires potent and low-cost therapeutics. We identified a novel series of single-domain antibodies (i.e., nanobody), Nanosota-1, from a camelid nanobody phage display library. Structural data showed that Nanosota-1 bound to the oft-hidden receptor-binding domain (RBD) of SARS-CoV-2 spike protein, blocking out viral receptor ACE2. The lead drug possessing an Fc tag (Nanosota-1C-Fc) bound to SARS-CoV-2 RBD with a Kd of 15.7picomolar (~3000 times more tightly than ACE2 did) and inhibited SARS-CoV-2 infection with an ND50 of 0.16microgram/milliliter (~6000 times more potently than ACE2 did). Administered at a single dose, Nanosota-1C-Fc demonstrated preventive and therapeutic efficacy in hamsters subjected to SARS-CoV-2 infection. Unlike conventional antibody drugs, Nanosota-1C-Fc was produced at high yields in bacteria and had exceptional thermostability. Pharmacokinetic analysis of Nanosota-1C-Fc documented a greater than 10-day in vivo half-life efficacy and high tissue bioavailability. Nanosota-1C-Fc is a potentially effective and realistic solution to the COVID-19 pandemic.

[end paper 24]

[begin paper 25]

PMCID = 10966652

Title = Quantitative Assessment of Energetic Contributions of Residues in a SARS-CoV-2 Viral Enzyme/Nanobody Interface

The highly conserved protease enzyme from SARS-CoV-2 (MPro) is crucial for viral replication and is an attractive target for the design of novel inhibitory compounds. MPro is known to be conformationally flexible and has been stabilized in an extended conformation in a complex with a novel nanobody (NB2B4), which inhibits the dimerization of the enzyme via binding to an allosteric site. However, the energetic contributions of the nanobody residues stabilizing the MPro/nanobody interface remain unresolved. We probed these residues using all-atom MD simulations in combination with alchemical free energy calculations by studying the physical residue–residue interactions and discovered the role of hydrophobic and electrostatic interactions in stabilizing the complex. Specifically, we found via mutational analysis that three interfacial nanobody residues (Y59, R106, and L109) contributed significantly, two residues (L107 and P110) contributed moderately, and two residues (H112 and T113) contributed minimally to the overall binding affinity of the nanobody. We also discovered that the nanobody affinity could be enhanced via a charge-reversal mutation (D62R) that alters the local interfacial electrostatic environment of this residue in the complex. These findings are potentially useful in designing novel synthetic nanobodies as allosteric inhibitors of MPro.

[end paper 25]

[begin paper 26]

PMCID = 9801699

Title = Applications of nanobodies in the prevention, detection, and treatment of the evolving SARS-CoV-2

Global health and economy are deeply influenced by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and its newly emerging variants. Nanobodies with nanometer-scale size are promising for the detection and treatment of SARS-CoV-2 and its variants because they are superior to conventional antibodies in terms of cryptic epitope accessibility, tissue penetration, cost, formatting adaptability, and especially protein stability, which enables their aerosolized specific delivery to lung tissues. This review summarizes the progress in the prevention, detection, and treatment of SARS-CoV-2 using nanobodies, as well as strategies to combat the evolving SARS-CoV-2 variants. Generally, highly efficient generation of potent broad-spectrum nanobodies targeting conserved epitopes or further construction of multivalent formats targeting non-overlapping epitopes can promote neutralizing activity against SARS-CoV-2 variants and suppress immune escape.

[end paper 26]

[begin paper 27]

PMCID = 8906176

Title = Monobodies with potent neutralizing activity against SARS-CoV-2 Delta and other variants of concern

By extensive affinity maturation against the SARS-CoV-2 spike protein, we developed a monobody with potent neutralizing activity against SARS-CoV-2 B.1.1 and four other variants of concern.

Neutralizing antibodies against the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) are useful for patients’ treatment of the coronavirus disease 2019 (COVID-19). We report here affinity maturation of monobodies against the SARS-CoV-2 spike protein and their neutralizing activity against SARS-CoV-2 B.1.1 (Pango v.3.1.14) as well as four variants of concern. We selected matured monobodies from libraries with multi-site saturation mutagenesis on the recognition loops through in vitro selection. One clone, the C4-AM2 monobody, showed extremely high affinity (KD < 0.01 nM) against the receptor-binding domain of the SARS-CoV-2 B.1.1, even in monomer form. Furthermore, the C4-AM2 monobody efficiently neutralized the SARS-CoV-2 B.1.1 (IC50 = 46 pM, 0.62 ng/ml), and the Alpha (IC50 = 77 pM, 1.0 ng/ml), Beta (IC50 = 0.54 nM, 7.2 ng/ml), Gamma (IC50 = 0.55 nM, 7.4 ng/ml), and Delta (IC50 = 0.59 nM, 8.0 ng/ml) variants. The obtained monobodies would be useful as neutralizing proteins against current and potentially hazardous future SARS-CoV-2 variants.

[end paper 27]

[begin paper 28]

PMCID = 9170159

Title = Dromedary camel nanobodies broadly neutralize SARS-CoV-2 variants

Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. In this report, we built dromedary camel (Camelus dromedarius) VHH phage libraries for the isolation of high-affinity nanobodies that broadly neutralize SARS-CoV-2 variants. Cryo-EM complex structures reveal that one dromedary camel VHH nanobody (8A2) binds the S1 subunit of the viral spike protein, and the other (7A3) targets a deeply buried region that uniquely extends to the S2 subunit beyond the S1 subunit. These nanobodies can protect mice from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting the therapeutic potential of these nanobodies against COVID-19. The dromedary camel VHH libraries could be helpful to isolate neutralizing nanobodies against future emerging viruses quickly.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) spike is a trimer of S1/S2 heterodimers with three receptor-binding domains (RBDs) at the S1 subunit for human angiotensin-converting enzyme 2 (hACE2). Due to their small size, nanobodies can recognize protein cavities that are not accessible to conventional antibodies. To isolate high-affinity nanobodies, large libraries with great diversity are highly desirable. Dromedary camels (Camelus dromedarius) are natural reservoirs of coronaviruses like Middle East respiratory syndrome CoV (MERS-CoV) that are transmitted to humans. Here, we built large dromedary camel VHH phage libraries to isolate nanobodies that broadly neutralize SARS-CoV-2 variants. We isolated two VHH nanobodies, NCI-CoV-7A3 (7A3) and NCI-CoV-8A2 (8A2), which have a high affinity for the RBD via targeting nonoverlapping epitopes and show broad neutralization activity against SARS-CoV-2 and its emerging variants of concern. Cryoelectron microscopy (cryo-EM) complex structures revealed that 8A2 binds the RBD in its up mode with a long CDR3 loop directly involved in the ACE2 binding residues and that 7A3 targets a deeply buried region that uniquely extends from the S1 subunit to the apex of the S2 subunit regardless of the conformational state of the RBD. At a dose of ≥5 mg/kg, 7A3 efficiently protected transgenic mice expressing hACE2 from the lethal challenge of variants B.1.351 or B.1.617.2, suggesting its therapeutic use against COVID-19 variants. The dromedary camel VHH phage libraries could be helpful as a unique platform ready for quickly isolating potent nanobodies against future emerging viruses.

[end paper 28]

[begin paper 29]

PMCID = 9524272

Title = Isolation of an escape-resistant SARS-CoV-2 neutralizing nanobody from a novel synthetic nanobody library

The COVID−19 pandemic not only resulted in a global crisis, but also accelerated vaccine development and antibody discovery. Herein we report a synthetic humanized VHH library development pipeline for nanomolar-range affinity VHH binders to SARS-CoV-2 variants of concern (VoC) receptor binding domains (RBD) isolation. Trinucleotide-based randomization of CDRs by Kunkel mutagenesis with the subsequent rolling-cycle amplification resulted in more than 1011 diverse phage display library in a manageable for a single person number of electroporation reactions. We identified a number of nanomolar-range affinity VHH binders to SARS-CoV-2 variants of concern (VoC) receptor binding domains (RBD) by screening a novel synthetic humanized antibody library. In order to explore the most robust and fast method for affinity improvement, we performed affinity maturation by CDR1 and CDR2 shuffling and avidity engineering by multivalent trimeric VHH fusion protein construction. As a result, H7-Fc and G12x3-Fc binders were developed with the affinities in nM and pM range respectively. Importantly, these affinities are weakly influenced by most of SARS-CoV-2 VoC mutations and they retain moderate binding to BA.4\5. The plaque reduction neutralization test (PRNT) resulted in IC50 = 100 ng\ml and 9.6 ng\ml for H7-Fc and G12x3-Fc antibodies, respectively, for the emerging Omicron BA.1 variant. Therefore, these VHH could expand the present landscape of SARS-CoV-2 neutralization binders with the therapeutic potential for present and future SARS-CoV-2 variants.

[end paper 29]

[begin paper 30]

PMCID = 8863326

Title = A potent alpaca-derived nanobody that neutralizes SARS-CoV-2 variants

The spike glycoprotein of SARS-CoV-2 engages with human ACE 2 to facilitate infection. Here, we describe an alpaca-derived heavy chain antibody fragment (VHH), saRBD-1, that disrupts this interaction by competitively binding to the spike protein receptor-binding domain. We further generated an engineered bivalent nanobody construct engineered by a flexible linker and a dimeric Fc conjugated nanobody construct. Both multivalent nanobodies blocked infection at picomolar concentrations and demonstrated no loss of potency against emerging variants of concern including Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), Epsilon (B.1.427/429), and Delta (B.1.617.2). saRBD-1 tolerates elevated temperature, freeze-drying, and nebulization, making it an excellent candidate for further development into a therapeutic approach for COVID-19.

SARS-CoV-2 variants effectively neutralized by saRBD-1 VHH with picomolar affinity

saRBD-1 neutralization increases when expressed as a bivalent or Fc construct

saRBD-1 binds SARS-CoV-2 RBD as a likely class 1 neutralizing antibody

saRBD-1 retains binding, neutralization after heat and nebulization treatments

Applied microbiology; Bioengineering; Nanotechnology

[end paper 30]

[begin paper 31]

PMCID = 7473855

Title = An alpaca nanobody neutralizes SARS-CoV-2 by blocking receptor interaction

SARS-CoV-2 enters host cells through an interaction between the spike glycoprotein and the angiotensin converting enzyme 2 (ACE2) receptor. Directly preventing this interaction presents an attractive possibility for suppressing SARS-CoV-2 replication. Here, we report the isolation and characterization of an alpaca-derived single domain antibody fragment, Ty1, that specifically targets the receptor binding domain (RBD) of the SARS-CoV-2 spike, directly preventing ACE2 engagement. Ty1 binds the RBD with high affinity, occluding ACE2. A cryo-electron microscopy structure of the bound complex at 2.9 Å resolution reveals that Ty1 binds to an epitope on the RBD accessible in both the ‘up’ and ‘down’ conformations, sterically hindering RBD-ACE2 binding. While fusion to an Fc domain renders Ty1 extremely potent, Ty1 neutralizes SARS-CoV-2 spike pseudovirus as a 12.8 kDa nanobody, which can be expressed in high quantities in bacteria, presenting opportunities for manufacturing at scale. Ty1 is therefore an excellent candidate as an intervention against COVID-19.

Here, Hanke et al. immunize an alpaca with SARS-CoV-2 spike protein domains and identify a nanobody that binds the receptor binding domain of spike in both the up and down conformations and sterically hinders ACE2 engagement.

[end paper 31]

[begin paper 32]

PMCID = 8061467

Title = Integrative overview of antibodies against SARS-CoV-2 and their possible applications in COVID-19 prophylaxis and treatment

SARS-CoV-2 is a novel β-coronavirus that caused the COVID-19 pandemic disease, which spread rapidly, infecting more than 134 million people, and killing almost 2.9 million thus far. Based on the urgent need for therapeutic and prophylactic strategies, the identification and characterization of antibodies has been accelerated, since they have been fundamental in treating other viral diseases. Here, we summarized in an integrative manner the present understanding of the immune response and physiopathology caused by SARS-CoV-2, including the activation of the humoral immune response in SARS-CoV-2 infection and therefore, the synthesis of antibodies. Furthermore, we also discussed about the antibodies that can be generated in COVID-19 convalescent sera and their associated clinical studies, including a detailed characterization of a variety of human antibodies and identification of antibodies from other sources, which have powerful neutralizing capacities. Accordingly, the development of effective treatments to mitigate COVID-19 is expected. Finally, we reviewed the challenges faced in producing potential therapeutic antibodies and nanobodies by cell factories at an industrial level while ensuring their quality, efficacy, and safety.

The online version contains supplementary material available at 10.1186/s12934-021-01576-5.

[end paper 32]

[begin paper 33]

PMCID = 8971115

Title = Circular RNA vaccines against SARS-CoV-2 and emerging variants

As the emerging variants of SARS-CoV-2 continue to drive the worldwide pandemic, there is a constant demand for vaccines that offer more effective and broad-spectrum protection. Here, we report a circular RNA (circRNA) vaccine that elicited potent neutralizing antibodies and T cell responses by expressing the trimeric RBD of the spike protein, providing robust protection against SARS-CoV-2 in both mice and rhesus macaques. Notably, the circRNA vaccine enabled higher and more durable antigen production than the 1mΨ-modified mRNA vaccine and elicited a higher proportion of neutralizing antibodies and distinct Th1-skewed immune responses. Importantly, we found that the circRNARBD-Omicron vaccine induced effective neutralizing antibodies against the Omicron but not the Delta variant. In contrast, the circRNARBD-Delta vaccine protected against both Delta and Omicron or functioned as a booster after two doses of either native- or Delta-specific vaccination, making it a favorable choice against the current variants of concern (VOCs) of SARS-CoV-2.

A circular RNA (circRNA) vaccine that encodes the trimeric RBD antigens of SARS-CoV-2 spike provides protection and memory boosting against SARS-CoV-2 variants of concern, in mice and rhesus macaques.

[end paper 33]

[begin paper 34]

PMCID = 11609455

Title = Nanobodies: From High-Throughput Identification to Therapeutic Development

The camelid single-domain antibody fragment, commonly referred to as a nanobody, achieves the targeting power of conventional monoclonal antibodies (mAbs) at only a fraction of their size. Isolated from camelid species (including llamas, alpacas, and camels), their small size at ∼15 kDa, low structural complexity, and high stability compared with conventional antibodies have propelled nanobody technology into the limelight of biologic development. Nanobodies are proving themselves to be a potent complement to traditional mAb therapies, showing success in the treatment of, for example, autoimmune diseases and cancer, and more recently as therapeutic options to treat infectious diseases caused by rapidly evolving biological targets such as the SARS-CoV-2 virus. This review highlights the benefits of applying a proteomic approach to identify diverse nanobody sequences against a single antigen. This proteomic approach coupled with conventional yeast/phage display methods enables the production of highly diverse repertoires of nanobodies able to bind the vast epitope landscape of an antigen, with epitope sampling surpassing that of mAbs. Additionally, we aim to highlight recent findings illuminating the structural attributes of nanobodies that make them particularly amenable to comprehensive antigen sampling and to synergistic activity—underscoring the powerful advantage of acquiring a large, diverse nanobody repertoire against a single antigen. Lastly, we highlight the efforts being made in the clinical development of nanobodies, which have great potential as powerful diagnostic reagents and treatment options, especially when targeting infectious disease agents.

Successful methods used to produce high-affinity nanobodies, with a focus on proteomic approaches.

Nanobody features enabling comprehensive sampling of an antigen’s surface.

Advantages of a large diverse repertoire of nanobodies against an antigen.

Nanobodies as promising therapeutics targeting infectious diseases, e.g.SARS-CoV-2.

Key challenges when creating a nanobody-based therapeutic.

This review summarizes successful, widely used methods to produce high-affinity nanobodies against an antigen. We discuss the structural attributes of nanobodies that enable them to comprehensively sample their antigen surface and to engage in synergy pairs—underscoring the powerful advantages of acquiring a large, diverse nanobody repertoire against a single antigen. Lastly, we highlight the efforts being made in the development of nanobody therapeutics, which have great potential as diagnostic reagents and treatment options, especially when targeting infectious disease agents.

[end paper 34]

[begin paper 35]

PMCID = 8463502

Title = Rapid generation of potent antibodies by autonomous hypermutation in yeast

The predominant approach for antibody generation remains animal immunization, which can yield exceptionally selective and potent antibody clones owing to the powerful evolutionary process of somatic hypermutation. However, animal immunization is inherently slow, not always accessible, and poorly compatible with many antigens. Here, we describe Autonomous Hypermutation yEast surfAce Display (AHEAD), a synthetic recombinant antibody generation technology that imitates somatic hypermutation inside engineered yeast. By encoding antibody fragments on an error-prone orthogonal DNA replication system, surface-displayed antibody repertoires continuously mutate through simple cycles of yeast culturing and enrichment for antigen binding to produce high-affinity clones in as little as 2 weeks. We applied AHEAD to generate potent nanobodies against the SARS-CoV-2 S glycoprotein, a GPCR, and other targets, offering a template for streamlined antibody generation at large.

[end paper 35]

[begin paper 36]

PMCID = 10275483

Title = Emerging Landscape of Nanobodies and Their Neutralizing Applications against SARS-CoV-2 Virus

The new severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) that causes the coronavirus disease 2019 (COVID-19) has significantly altered people’s way of life. Despite widespread knowledge of vaccination, mask use, and avoidance of close contact, COVID-19 is still spreading around the world. Numerous research teams are examining the SARS-CoV-2 infection process to discover strategies to identify, prevent, and treat COVID-19 to limit the spread of this chronic coronavirus illness and restore lives to normalcy. Nanobodies have advantages over polyclonal and monoclonal antibodies (Ab) and Ab fragments, including reduced size, high stability, simplicity in manufacture, compatibility with genetic engineering methods, and lack of solubility and aggregation issues. Recent studies have shown that nanobodies that target the SARS-CoV-2 receptor-binding domain and disrupt ACE2 interactions are helpful in the prevention and treatment of SARS-CoV-2-infected animal models, despite the lack of evidence in human patients. The creation and evaluation of nanobodies, as well as their diagnostic and therapeutic applications against COVID-19, are discussed in this paper.

[end paper 36]

[begin paper 37]

PMCID = 7959386

Title = A high-affinity RBD-targeting nanobody improves fusion partner’s potency against SARS-CoV-2

A key step to the SARS-CoV-2 infection is the attachment of its Spike receptor-binding domain (S RBD) to the host receptor ACE2. Considerable research has been devoted to the development of neutralizing antibodies, including llama-derived single-chain nanobodies, to target the receptor-binding motif (RBM) and to block ACE2-RBD binding. Simple and effective strategies to increase potency are desirable for such studies when antibodies are only modestly effective. Here, we identify and characterize a high-affinity synthetic nanobody (sybody, SR31) as a fusion partner to improve the potency of RBM-antibodies. Crystallographic studies reveal that SR31 binds to RBD at a conserved and ‘greasy’ site distal to RBM. Although SR31 distorts RBD at the interface, it does not perturb the RBM conformation, hence displaying no neutralizing activities itself. However, fusing SR31 to two modestly neutralizing sybodies dramatically increases their affinity for RBD and neutralization activity against SARS-CoV-2 pseudovirus. Our work presents a tool protein and an efficient strategy to improve nanobody potency.

SARS-CoV-2 relies on the receptor-binding domain (RBD) of its envelope Spike protein to recognize and infect host cells. RBD is therefore an immunological hotspot to generate antibodies for therapeutic and detection purposes. Binding affinity is one of the key characteristics of such antibodies. Here, we report a single-chain antibody (nanobody, ~14 kDa) that binds RBD with nanomolar affinity. The nanobody, dubbed SR31, binds RBD at an epitope distal to the receptor-binding motif (RBM) which is the target of most neutralizing antibodies. SR31 can therefore bind RBD in addition to RBM-binders, and increase their affinity and potency by avidity effects when used as a fusion partner. Compared to other in vitro affinity maturation techniques such as library screening and structure-based design, the fusion strategy offers advantages in speed and simplicity. In addition, SR31, together with RBD-targeting nanobodies recognizing a wide spectrum of epitopes, provides a useful toolkit to probe epitopes of uncharacterized antibodies by competitive binding.

[end paper 37]

[begin paper 38]

PMCID = 7857409

Title = An ultrapotent synthetic nanobody neutralizes SARS-CoV-2 by stabilizing inactive Spike

Monoclonal antibodies that bind to the spike protein of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) show therapeutic promise but must be produced in mammalian cells and need to be delivered intravenously. By contrast, single-domain antibodies called nanobodies can be produced in bacteria or yeast, and their stability may enable aerosol delivery. Two papers now report nanobodies that bind tightly to spike and efficiently neutralize SARS-CoV-2 in cells. Schoof et al. screened a yeast surface display of synthetic nanobodies and Xiang et al. screened anti-spike nanobodies produced by a llama. Both groups identified highly potent nanobodies that lock the spike protein in an inactive conformation. Multivalent constructs of selected nanobodies achieved even more potent neutralization.

Science, this issue p. 1473, p. 

Potent neutralizers of SARS-CoV-2 are identified by screening either synthetic or llama-produced nanobodies.

The severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) virus enters host cells via an interaction between its Spike protein and the host cell receptor angiotensin-converting enzyme 2 (ACE2). By screening a yeast surface-displayed library of synthetic nanobody sequences, we developed nanobodies that disrupt the interaction between Spike and ACE2. Cryo–electron microscopy (cryo-EM) revealed that one nanobody, Nb6, binds Spike in a fully inactive conformation with its receptor binding domains locked into their inaccessible down state, incapable of binding ACE2. Affinity maturation and structure-guided design of multivalency yielded a trivalent nanobody, mNb6-tri, with femtomolar affinity for Spike and picomolar neutralization of SARS-CoV-2 infection. mNb6-tri retains function after aerosolization, lyophilization, and heat treatment, which enables aerosol-mediated delivery of this potent neutralizer directly to the airway epithelia.

[end paper 38]

[begin paper 39]

PMCID = 9132424

Title = Deep mutational engineering of broadly-neutralizing nanobodies accommodating SARS-CoV-1 and 2 antigenic drift

Here, we report the molecular engineering of nanobodies that bind with picomolar affinity to both SARS-CoV-1 and SARS-CoV-2 receptor-binding domains (RBD) and are highly neutralizing. We applied deep mutational engineering to VHH72, a nanobody initially specific for SARS-CoV-1 RBD with little cross-reactivity to SARS-CoV-2 antigen. We first identified all the individual VHH substitutions that increase binding to SARS-CoV-2 RBD and then screened highly focused combinatorial libraries to isolate engineered nanobodies with improved properties. The corresponding VHH-Fc molecules show high affinities for SARS-CoV-2 antigens from various emerging variants and SARS-CoV-1, block the interaction between ACE2 and RBD, and neutralize the virus with high efficiency. Its rare specificity across sarbecovirus relies on its peculiar epitope outside the immunodominant regions. The engineered nanobodies share a common motif of three amino acids, which contribute to the broad specificity of recognition. Our results show that deep mutational engineering is a very powerful method, especially to rapidly adapt existing antibodies to new variants of pathogens.

[end paper 39]

[begin paper 40]

PMCID = 9393763

Title = Antibody-mediated immunity to SARS-CoV-2 spike

Despite effective spike-based vaccines and monoclonal antibodies, the SARS-CoV-2 pandemic continues more than two and a half years post-onset. Relentless investigation has outlined a causative dynamic between host-derived antibodies and reciprocal viral subversion. Integration of this paradigm into the architecture of next generation antiviral strategies, predicated on a foundational understanding of the virology and immunology of SARS-CoV-2, will be critical for success. This review aims to serve as a primer on the immunity endowed by antibodies targeting SARS-CoV-2 spike protein through a structural perspective. We begin by introducing the structure and function of spike, polyclonal immunity to SARS-CoV-2 spike, and the emergence of major SARS-CoV-2 variants that evade immunity. The remainder of the article comprises an in-depth dissection of all major epitopes on SARS-CoV-2 spike in molecular detail, with emphasis on the origins, neutralizing potency, mechanisms of action, cross-reactivity, and variant resistance of representative monoclonal antibodies to each epitope.

[end paper 40]

[begin paper 41]

PMCID = 8982573

Title = Biparatopic sybodies neutralize SARS‐CoV‐2 variants of concern and mitigate drug resistance

The ongoing COVID‐19 pandemic represents an unprecedented global health crisis. Here, we report the identification of a synthetic nanobody (sybody) pair, Sb#15 and Sb#68, that can bind simultaneously to the SARS‐CoV‐2 spike RBD and efficiently neutralize pseudotyped and live viruses by interfering with ACE2 interaction. Cryo‐EM confirms that Sb#15 and Sb#68 engage two spatially discrete epitopes, influencing rational design of bispecific and tri‐bispecific fusion constructs that exhibit up to 100‐ and 1,000‐fold increase in neutralization potency, respectively. Cryo‐EM of the sybody‐spike complex additionally reveals a novel up‐out RBD conformation. While resistant viruses emerge rapidly in the presence of single binders, no escape variants are observed in the presence of the bispecific sybody. The multivalent bispecific constructs further increase the neutralization potency against globally circulating SARS‐CoV‐2 variants of concern. Our study illustrates the power of multivalency and biparatopic nanobody fusions for the potential development of therapeutic strategies that mitigate the emergence of new SARS‐CoV‐2 escape mutants.

Sybodies Sb#15 and Sb#68 inhibit SARS‐CoV‐2 infectivity by targeting non‐overlapping epitopes on the spike glycoprotein. Covalent sybody fusion and valency engineering enhances neutralization potency against variants and impedes emergence of escape mutants.  

[end paper 41]

[begin paper 42]

PMCID = 8646660

Title = Uncovering a conserved vulnerability site in SARS‐CoV‐2 by a human antibody

An essential step for SARS‐CoV‐2 infection is the attachment to the host cell receptor by its Spike receptor‐binding domain (RBD). Most of the existing RBD‐targeting neutralizing antibodies block the receptor‐binding motif (RBM), a mutable region with the potential to generate neutralization escape mutants. Here, we isolated and structurally characterized a non‐RBM‐targeting monoclonal antibody (FD20) from convalescent patients. FD20 engages the RBD at an epitope distal to the RBM with a K D of 5.6 nM, neutralizes SARS‐CoV‐2 including the current Variants of Concern such as B.1.1.7, B.1.351, P.1, and B.1.617.2 (Delta), displays modest cross‐reactivity against SARS‐CoV, and reduces viral replication in hamsters. The epitope coincides with a predicted “ideal” vulnerability site with high functional and structural constraints. Mutation of the residues of the conserved epitope variably affects FD20‐binding but confers little or no resistance to neutralization. Finally, in vitro mode‐of‐action characterization and negative‐stain electron microscopy suggest a neutralization mechanism by which FD20 destructs the Spike. Our results reveal a conserved vulnerability site in the SARS‐CoV‐2 Spike for the development of potential antiviral drugs.

A monoclonal antibody (FD20) from convalescent COVID‐19 patients has been isolated and structurally and biologically characterized. Various SARS‐CoV‐2 strains, including the Alpha, Beta, Gamma, and Delta variants, and naturally occurring epitope mutants, can be neutralized by FD20 with similar potency. 

[end paper 42]

[begin paper 43]

PMCID = 11793827

Title = Nanobody screening and machine learning guided identification of cross-variant anti-SARS-CoV-2 neutralizing heavy-chain only antibodies

Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) continues to persist, demonstrating the risks posed by emerging infectious diseases to national security, public health, and the economy. Development of new vaccines and antibodies for emerging viral threats requires substantial resources and time, and traditional development platforms for vaccines and antibodies are often too slow to combat continuously evolving immunological escape variants, reducing their efficacy over time. Previously, we designed a next-generation synthetic humanized nanobody (Nb) phage display library and demonstrated that this library could be used to rapidly identify highly specific and potent neutralizing heavy chain-only antibodies (HCAbs) with prophylactic and therapeutic efficacy in vivo against the original SARS-CoV-2. In this study, we used a combination of high throughput screening and machine learning (ML) models to identify HCAbs with potent efficacy against SARS-CoV-2 viral variants of interest (VOIs) and concern (VOCs). To start, we screened our highly diverse Nb phage display library against several pre-Omicron VOI and VOC receptor binding domains (RBDs) to identify panels of cross-reactive HCAbs. Using HCAb affinity for SARS-CoV-2 VOI and VOCs (pre-Omicron variants) and model features from other published data, we were able to develop a ML model that successfully identified HCAbs with efficacy against Omicron variants, independent of our experimental biopanning workflow. This biopanning informed ML approach reduced the experimental screening burden by 78% to 90% for the Omicron BA.5 and Omicron BA.1 variants, respectively. The combined approach can be applied to other emerging viruses with pandemic potential to rapidly identify effective therapeutic antibodies against emerging variants.

We leveraged a robust experimental pipeline for heavy chain-only (HCAb) library screening to identify 59 potent HCAbs that can cross-neutralize different SARS-CoV-2 variants. Several of these HCAbs with efficacy against different variants also bind to different SARS-CoV-2 epitopes, suggesting they could be used in antibody cocktails or be engineered as bispecific antibodies that are cross-variant and resistant to viral escape. Using existing published data and data generated from our library of HCAbs against diverse pre-Omicron SARS-CoV-2 variants, we developed an ML model to rapidly predict which HCAbs would be effective against Omicron BA.1 and Omicron BA.5. Using this ML model three HCAbs with efficacy against Omicron BA.1 and BA.5 were identified without requiring additional biopanning. This integrated computational and experimental pipeline can be leveraged for future outbreaks to rapidly identify cross-variant countermeasures that are effective against potential emerging variants.

[end paper 43]

[begin paper 44]

PMCID = 8344751

Title = Development of potent and effective synthetic SARS-CoV-2 neutralizing nanobodies

The respiratory virus responsible for coronavirus disease 2019 (COVID-19), severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has affected nearly every aspect of life worldwide, claiming the lives of over 3.9 million people globally, at the time of this publication. Neutralizing humanized nanobody (VHH)-based antibodies (VHH-huFc) represent a promising therapeutic intervention strategy to address the current SARS-CoV-2 pandemic and provide a powerful toolkit to address future virus outbreaks. Using a synthetic, high-diversity VHH bacteriophage library, several potent neutralizing VHH-huFc antibodies were identified and evaluated for their capacity to tightly bind to the SARS-CoV-2 receptor-binding domain, to prevent binding of SARS-CoV-2 spike (S) to the cellular receptor angiotensin-converting enzyme 2, and to neutralize viral infection. Preliminary preclinical evaluation of multiple VHH-huFc antibody candidates demonstrate that they are prophylactically and therapeutically effective in vivo against wildtype SARS-CoV-2. The identified and characterized VHH-huFc antibodies described herein represent viable candidates for further preclinical evaluation and another tool to add to our therapeutic arsenal to address the COVID-19 pandemic.

[end paper 44]

[begin paper 45]

PMCID = 8771608

Title = Structure genomics of SARS-CoV-2 and its Omicron variant: drug design templates for COVID-19

Coronavirus disease 2019 (COVID-19), caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), has brought an unprecedented public health crisis and persistently threatens to humanity. With tireless efforts from scientists around the world, understanding of the biology of coronavirus has been greatly enhanced over the past 2 years. Structural biology has demonstrated its powerful impact on uncovering structures and functions for the vast majority of SARS-CoV-2 proteins and guided the development of drugs and vaccines against COVID-19. In this review, we summarize current progress in the structural biology of SARS-CoV-2 and discuss important biological issues that remain to be addressed. We present the examples of structure-based design of Pfizer’s novel anti-SARS-CoV-2 drug PF-07321332 (Paxlovid), Merck’s nucleotide inhibitor molnupiravir (Lagevrio), and VV116, an oral drug candidate for COVID-19. These examples highlight the importance of structure in drug discovery to combat COVID-19. We also discussed the recent variants of Omicron and its implication in immunity escape from existing vaccines and antibody therapies.

[end paper 45]

[begin paper 46]

PMCID = 8248807

Title = SARS-CoV-2: Origin, Evolution, and Targeting Inhibition

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) caused an outbreak in Wuhan city, China and quickly spread worldwide. Currently, there are no specific drugs or antibodies that claim to cure severe acute respiratory diseases. For SARS-CoV-2, the spike (S) protein recognizes and binds to the angiotensin converting enzyme 2 (ACE2) receptor, allowing viral RNA to enter the host cell. The main protease (Mpro) is involved in the proteolytic process for mature non-structural proteins, and RNA-dependent RNA polymerase (RdRp) is responsible for the viral genome replication and transcription processes. Owing to the pivotal physiological roles in viral invasion and replication, S protein, Mpro, RdRp are regarded as the main therapeutic targets for coronavirus disease 2019 (COVID-19). In this review, we carried out an evolutionary analysis of SARS-CoV-2 in comparison with other mammal-infecting coronaviruses that have sprung up in the past few decades and described the pathogenic mechanism of SARS-CoV-2. We displayed the structural details of S protein, Mpro, and RdRp, as well as their complex structures with different chemical inhibitors or antibodies. Structural comparisons showed that some neutralizing antibodies and small molecule inhibitors could inhibit S protein, Mpro, or RdRp. Moreover, we analyzed the structural differences between SARS-CoV-2 ancestral S protein and D614G mutant, which led to a second wave of infection during the recent pandemic. In this context, we outline the methods that might potentially help cure COVID-19 and provide a summary of effective chemical molecules and neutralizing antibodies.

[end paper 46]

[begin paper 47]

PMCID = 8724751

Title = Monoclonal antibodies for COVID-19 therapy and SARS-CoV-2 detection

The coronavirus disease 2019 (COVID-19) pandemic is an exceptional public health crisis that demands the timely creation of new therapeutics and viral detection. Owing to their high specificity and reliability, monoclonal antibodies (mAbs) have emerged as powerful tools to treat and detect numerous diseases. Hence, many researchers have begun to urgently develop Ab-based kits for the detection of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and Ab drugs for use as COVID-19 therapeutic agents. The detailed structure of the SARS-CoV-2 spike protein is known, and since this protein is key for viral infection, its receptor-binding domain (RBD) has become a major target for therapeutic Ab development. Because SARS-CoV-2 is an RNA virus with a high mutation rate, especially under the selective pressure of aggressively deployed prophylactic vaccines and neutralizing Abs, the use of Ab cocktails is expected to be an important strategy for effective COVID-19 treatment. Moreover, SARS-CoV-2 infection may stimulate an overactive immune response, resulting in a cytokine storm that drives severe disease progression. Abs to combat cytokine storms have also been under intense development as treatments for COVID-19. In addition to their use as drugs, Abs are currently being utilized in SARS-CoV-2 detection tests, including antigen and immunoglobulin tests. Such Ab-based detection tests are crucial surveillance tools that can be used to prevent the spread of COVID-19. Herein, we highlight some key points regarding mAb-based detection tests and treatments for the COVID-19 pandemic.

[end paper 47]

[begin paper 48]

PMCID = 8260353

Title = Nanobodies from camelid mice and llamas neutralize SARS-CoV-2 variants

Since the start of the COVID-19 pandemic, SARS-CoV-2 has caused millions of deaths worldwide. Although a number of vaccines have been deployed, the continual evolution of the receptor-binding domain (RBD) of the virus has challenged their efficacy. In particular, the emerging variants B.1.1.7, B.1.351 and P.1 (first detected in the UK, South Africa and Brazil, respectively) have compromised the efficacy of sera from patients who have recovered from COVID-19 and immunotherapies that have received emergency use authorization. One potential alternative to avert viral escape is the use of camelid VHHs (variable heavy chain domains of heavy chain antibody (also known as nanobodies)), which can recognize epitopes that are often inaccessible to conventional antibodies. Here, we isolate anti-RBD nanobodies from llamas and from mice that we engineered to produce VHHs cloned from alpacas, dromedaries and Bactrian camels. We identified two groups of highly neutralizing nanobodies. Group 1 circumvents antigenic drift by recognizing an RBD region that is highly conserved in coronaviruses but rarely targeted by human antibodies. Group 2 is almost exclusively focused to the RBD–ACE2 interface and does not neutralize SARS-CoV-2 variants that carry E484K or N501Y substitutions. However, nanobodies in group 2 retain full neutralization activity against these variants when expressed as homotrimers, and—to our knowledge—rival the most potent antibodies against SARS-CoV-2 that have been produced to date. These findings suggest that multivalent nanobodies overcome SARS-CoV-2 mutations through two separate mechanisms: enhanced avidity for the ACE2-binding domain and recognition of conserved epitopes that are largely inaccessible to human antibodies. Therefore, although new SARS-CoV-2 mutants will continue to emerge, nanobodies represent promising tools to prevent COVID-19 mortality when vaccines are compromised.

Multivalent nanobodies against SARS-CoV-2 from mice engineered to produce camelid nanobodies recognize conserved epitopes that are inaccessible to human antibodies and show promise as a strategy for dealing with viral escape mutations.

[end paper 48]

[begin paper 49]

PMCID = 7833398

Title = Structural Analysis of Neutralizing Epitopes of the SARS-CoV-2 Spike to Guide Therapy and Vaccine Design Strategies

Coronavirus research has gained tremendous attention because of the COVID-19 pandemic, caused by the novel severe acute respiratory syndrome coronavirus (nCoV or SARS-CoV-2). In this review, we highlight recent studies that provide atomic-resolution structural details important for the development of monoclonal antibodies (mAbs) that can be used therapeutically and prophylactically and for vaccines against SARS-CoV-2. Structural studies with SARS-CoV-2 neutralizing mAbs have revealed a diverse set of binding modes on the spike’s receptor-binding domain and N-terminal domain and highlight alternative targets on the spike. We consider this structural work together with mAb effects in vivo to suggest correlations between structure and clinical applications. We also place mAbs against severe acute respiratory syndrome (SARS) and Middle East respiratory syndrome (MERS) coronaviruses in the context of the SARS-CoV-2 spike to suggest features that may be desirable to design mAbs or vaccines capable of conferring broad protection.

[end paper 49]

[begin paper 50]

PMCID = 8528929

Title = Discovery and characterization of high-affinity, potent SARS-CoV-2 neutralizing antibodies via single B cell screening

Monoclonal antibodies that target SARS-CoV-2 with high affinity are valuable for a wide range of biomedical applications involving novel coronavirus disease (COVID-19) diagnosis, treatment, and prophylactic intervention. Strategies for the rapid and reliable isolation of these antibodies, especially potent neutralizing antibodies, are critical toward improved COVID-19 response and informed future response to emergent infectious diseases. In this study, single B cell screening was used to interrogate antibody repertoires of immunized mice and isolate antigen-specific IgG1+ memory B cells. Using these methods, high-affinity, potent neutralizing antibodies were identified that target the receptor-binding domain of SARS-CoV-2. Further engineering of the identified molecules to increase valency resulted in enhanced neutralizing activity. Mechanistic investigation revealed that these antibodies compete with ACE2 for binding to the receptor-binding domain of SARS-CoV-2. These antibodies may warrant further development for urgent COVID-19 applications. Overall, these results highlight the potential of single B cell screening for the rapid and reliable identification of high-affinity, potent neutralizing antibodies for infectious disease applications.

[end paper 50]

[begin paper 51]

PMCID = 11228396

Title = Structural basis of broad SARS-CoV-2 cross-neutralization by affinity-matured public antibodies

Descendants of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) Omicron variant now account for almost all SARS-CoV-2 infections. The Omicron variant and its sublineages have spike glycoproteins that are highly diverged from the pandemic founder and first-generation vaccine strain, resulting in significant evasion from monoclonal antibody therapeutics and vaccines. Understanding how commonly elicited antibodies can broaden to cross-neutralize escape variants is crucial. We isolate IGHV3-53, using “public” monoclonal antibodies (mAbs) from an individual 7 months post infection with the ancestral virus and identify antibodies that exhibit potent and broad cross-neutralization, extending to the BA.1, BA.2, and BA.4/BA.5 sublineages of Omicron. Deep mutational scanning reveals these mAbs’ high resistance to viral escape. Structural analysis via cryoelectron microscopy of a representative broadly neutralizing antibody, CAB-A17, in complex with the Omicron BA.1 spike highlights the structural underpinnings of this broad neutralization. By reintroducing somatic hypermutations into a germline-reverted CAB-A17, we delineate the role of affinity maturation in the development of cross-neutralization by a public class of antibodies.

Some antibodies resulting from founder strain infection show a remarkable breadth

Deep mutational scanning shows high resistance to viral escape

Antibody affinity maturation was identified as key to developing breadth

Cryo-EM reveals the basis for breadth developed through affinity maturation

Sheward et al. find representative antibodies, raised following infection with the SARS-CoV-2 founder strain, that cross-neutralize Omicron variants. Combining deep mutational scanning, electron cryomicroscopy, and neutralization assays, they identify the structural basis for this cross-neutralization and reveal the role of somatic hypermutation in the broadening of the antibody response.

[end paper 51]

[begin paper 52]

PMCID = 9313406

Title = Plasmonic Approaches for the Detection of SARS-CoV-2 Viral Particles

The ongoing highly contagious Coronavirus disease 2019 (COVID-19) pandemic, caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), underlines the fundamental position of diagnostic testing in outbreak control by allowing a distinction of the infected from the non-infected people. Diagnosis of COVID-19 remains largely based on reverse transcription PCR (RT-PCR), identifying the genetic material of the virus. Molecular testing approaches have been largely proposed in addition to infectivity testing of patients via sensing the presence of viral particles of SARS-CoV-2 specific structural proteins, such as the spike glycoproteins (S1, S2) and the nucleocapsid (N) protein. While the S1 protein remains the main target for neutralizing antibody treatment upon infection and the focus of vaccine and therapeutic design, it has also become a major target for the development of point-of care testing (POCT) devices. This review will focus on the possibility of surface plasmon resonance (SPR)-based sensing platforms to convert the receptor-binding event of SARS-CoV-2 viral particles into measurable signals. The state-of-the-art SPR-based SARS-CoV-2 sensing devices will be provided, and highlights about the applicability of plasmonic sensors as POCT for virus particle as well as viral protein sensing will be discussed.

[end paper 52]

[begin paper 53]

PMCID = 11640846

Title = Backstage Heroes—Yeast in COVID-19 Research

The extremely rapid development of understanding and technology that led to the containment of the COVID-19 pandemic resulted from collaborative efforts in the fields of Betacoronavirus pandemicum (SARS-CoV-2) biology, pharmacology, vaccinology, and medicine. Perhaps surprisingly, much of the research was conducted using simple and efficient yeast models. In this manuscript, we describe how yeast, eukaryotic microorganisms, have been used to research this global challenge, focusing on the therapeutic potential of the studies discussed herein. Thus, we outline the role of yeast in studying viral protein interactions with the host cell proteome, including the binding of the SARS-CoV-2 virus spike protein to the human ACE2 receptor and its modulation. The production and exploration of viral antigens in yeast systems, which led to the development of two approved COVID-19 vaccines, are also detailed. Moreover, yeast platforms facilitating the discovery and production of single-domain antibodies (nanobodies) against SARS-CoV-2 are described. Methods guiding modern and efficient drug discovery are explained at length. In particular, we focus on studies designed to search for inhibitors of the main protease (Mpro), a unique target for anti-coronaviral therapies. We highlight the adaptability of the techniques used, providing opportunities for rapid modification and implementation alongside the evolution of the SARS-CoV-2 virus. Approaches introduced in yeast systems that may have universal potential application in studies of emerging viral diseases are also described.

[end paper 53]

[begin paper 54]

PMCID = 8491763

Title = Mechanisms of SARS-CoV-2 entry into cells

The unprecedented public health and economic impact of the COVID-19 pandemic caused by infection with severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has been met with an equally unprecedented scientific response. Much of this response has focused, appropriately, on the mechanisms of SARS-CoV-2 entry into host cells, and in particular the binding of the spike (S) protein to its receptor, angiotensin-converting enzyme 2 (ACE2), and subsequent membrane fusion. This Review provides the structural and cellular foundations for understanding the multistep SARS-CoV-2 entry process, including S protein synthesis, S protein structure, conformational transitions necessary for association of the S protein with ACE2, engagement of the receptor-binding domain of the S protein with ACE2, proteolytic activation of the S protein, endocytosis and membrane fusion. We define the roles of furin-like proteases, transmembrane protease, serine 2 (TMPRSS2) and cathepsin L in these processes, and delineate the features of ACE2 orthologues in reservoir animal species and S protein adaptations that facilitate efficient human transmission. We also examine the utility of vaccines, antibodies and other potential therapeutics targeting SARS-CoV-2 entry mechanisms. Finally, we present key outstanding questions associated with this critical process.

Entry of SARS-CoV-2 into host cells is mediated by the interaction between the viral spike protein and its receptor angiotensin-converting enzyme 2, followed by virus–cell membrane fusion. Worldwide research efforts have provided a detailed understanding of this process at the structural and cellular levels, enabling successful vaccine development for a rapid response to the COVID-19 pandemic.

[end paper 54]

[begin paper 55]

PMCID = 8447893

Title = Structural biology of SARS-CoV-2 and implications for therapeutic development

The COVID-19 pandemic, caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), is an unprecedented global health crisis. However, therapeutic options for treatment are still very limited. The development of drugs that target vital proteins in the viral life cycle is a feasible approach for treating COVID-19. Belonging to the subfamily Orthocoronavirinae with the largest RNA genome, SARS-CoV-2 encodes a total of 29 proteins. These non-structural, structural and accessory proteins participate in entry into host cells, genome replication and transcription, and viral assembly and release. SARS-CoV-2 proteins can individually perform essential physiological roles, be components of the viral replication machinery or interact with numerous host cellular factors. In this Review, we delineate the structural features of SARS-CoV-2 from the whole viral particle to the individual viral proteins and discuss their functions as well as their potential as targets for therapeutic interventions.

Elucidating the structure and function of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) proteins is vital for understanding the molecular mechanisms of viral replication and COVID-19 pathogenesis, and could lead to the development of novel therapeutics. In this Review, Yang and Rao delineate the structural features of SARS-CoV-2 from the whole viral particle to the individual viral proteins and discuss their functions as well as their potential as targets for therapeutic interventions.

[end paper 55]

[begin paper 56]

PMCID = 7430568

Title = An ultra-potent synthetic nanobody neutralizes SARS-CoV-2 by locking Spike into an inactive conformation

Without an effective prophylactic solution, infections from SARS-CoV-2 continue to rise worldwide with devastating health and economic costs. SARS-CoV-2 gains entry into host cells via an interaction between its Spike protein and the host cell receptor angiotensin converting enzyme 2 (ACE2). Disruption of this interaction confers potent neutralization of viral entry, providing an avenue for vaccine design and for therapeutic antibodies. Here, we develop single-domain antibodies (nanobodies) that potently disrupt the interaction between the SARS-CoV-2 Spike and ACE2. By screening a yeast surface-displayed library of synthetic nanobody sequences, we identified a panel of nanobodies that bind to multiple epitopes on Spike and block ACE2 interaction via two distinct mechanisms. Cryogenic electron microscopy (cryo-EM) revealed that one exceptionally stable nanobody, Nb6, binds Spike in a fully inactive conformation with its receptor binding domains (RBDs) locked into their inaccessible down-state, incapable of binding ACE2. Affinity maturation and structure-guided design of multivalency yielded a trivalent nanobody, mNb6-tri, with femtomolar affinity for SARS-CoV-2 Spike and picomolar neutralization of SARS-CoV-2 infection. mNb6-tri retains stability and function after aerosolization, lyophilization, and heat treatment. These properties may enable aerosol-mediated delivery of this potent neutralizer directly to the airway epithelia, promising to yield a widely deployable, patient-friendly prophylactic and/or early infection therapeutic agent to stem the worst pandemic in a century.

[end paper 56]

[begin paper 57]

PMCID = 7556756

Title = Antibody-like proteins that capture and neutralize SARS-CoV-2

SARS-CoV-2–neutralizing antibody-like proteins were generated by a high-speed in vitro selection method.

To combat severe acute respiratory syndrome–related coronavirus 2 (SARS-CoV-2) and any unknown emerging pathogens in the future, the development of a rapid and effective method to generate high-affinity antibodies or antibody-like proteins is of critical importance. We here report high-speed in vitro selection of multiple high-affinity antibody-like proteins against various targets including the SARS-CoV-2 spike protein. The sequences of monobodies against the SARS-CoV-2 spike protein were successfully procured within only 4 days. Furthermore, the obtained monobody efficiently captured SARS-CoV-2 particles from the nasal swab samples of patients and exhibited a high neutralizing activity against SARS-CoV-2 infection (half-maximal inhibitory concentration, 0.5 nanomolar). High-speed in vitro selection of antibody-like proteins is a promising method for rapid development of a detection method for, and of a neutralizing protein against, a virus responsible for an ongoing, and possibly a future, pandemic.

[end paper 57]

[begin paper 58]

PMCID = 9900178

Title = SARS-CoV-2 spike HexaPro formulated in aluminium hydroxide and administered in an accelerated vaccination schedule partially protects Syrian Hamsters against viral challenge despite low neutralizing antibody responses

SARS-CoV-2 continues to pose a threat to human health as new variants emerge and thus a diverse vaccine pipeline is needed. We evaluated SARS-CoV-2 HexaPro spike protein formulated in Alhydrogel® (aluminium oxyhydroxide) in Syrian hamsters, using an accelerated two dose regimen (given 10 days apart) and a standard regimen (two doses given 21 days apart). Both regimens elicited spike- and RBD-specific IgG antibody responses of similar magnitude, but in vitro virus neutralization was low or undetectable. Despite this, the accelerated two dose regimen offered reduction in viral load and protected against lung pathology upon challenge with homologous SARS-CoV-2 virus (Wuhan-Hu-1). This highlights that vaccine-induced protection against SARS-CoV-2 disease can be obtained despite low neutralizing antibody levels and suggests that accelerated vaccine schedules may be used to confer rapid protection against SARS-CoV-2 disease.

[end paper 58]

[begin paper 59]

PMCID = 9896449

Title = Shark nanobodies with potent SARS-CoV-2 neutralizing activity and broad sarbecovirus reactivity

Despite rapid and ongoing vaccine and therapeutic development, SARS-CoV-2 continues to evolve and evade, presenting a need for next-generation diverse therapeutic modalities. Here we show that nurse sharks immunized with SARS-CoV-2 recombinant receptor binding domain (RBD), RBD-ferritin (RFN), or spike protein ferritin nanoparticle (SpFN) immunogens elicit a set of new antigen receptor antibody (IgNAR) molecules that target two non-overlapping conserved epitopes on the spike RBD. Representative shark antibody variable NAR-Fc chimeras (ShAbs) targeting either of the two epitopes mediate cell-effector functions, with high affinity to all SARS-CoV-2 viral variants of concern, including the divergent Omicron strains. The ShAbs potently cross-neutralize SARS-CoV-2 WA-1, Alpha, Beta, Delta, Omicron BA.1 and BA.5, and SARS-CoV-1 pseudoviruses, and confer protection against SARS-CoV-2 challenge in the K18-hACE2 transgenic mouse model. Structural definition of the RBD-ShAb01-ShAb02 complex enabled design and production of multi-specific nanobodies with enhanced neutralization capacity, and picomolar affinity to divergent sarbecovirus clade 1a, 1b and 2 RBD molecules. These shark nanobodies represent potent immunotherapeutics both for current use, and future sarbecovirus pandemic preparation.

SARS-CoV-2 variants of concern continue to emerge, reducing vaccine efficacy and limiting therapeutic options. Here, Chen and colleagues describe the identification and design of shark nanobodies with pansarbecovirus activity.

[end paper 59]

[begin paper 60]

PMCID = 11209230

Title = Comprehensive Overview of Broadly Neutralizing Antibodies against SARS-CoV-2 Variants

Currently, SARS-CoV-2 has evolved into various variants, including the numerous highly mutated Omicron sub-lineages, significantly increasing immune evasion ability. The development raises concerns about the possibly diminished effectiveness of available vaccines and antibody-based therapeutics. Here, we describe those representative categories of broadly neutralizing antibodies (bnAbs) that retain prominent effectiveness against emerging variants including Omicron sub-lineages. The molecular characteristics, epitope conservation, and resistance mechanisms of these antibodies are further detailed, aiming to offer suggestion or direction for the development of therapeutic antibodies, and facilitate the design of vaccines with broad-spectrum potential.

[end paper 60]

[begin paper 61]

PMCID = 10695562

Title = Lys417 acts as a molecular switch that regulates the conformation of SARS-CoV-2 spike protein

SARS-CoV-2 spike protein plays a key role in mediating viral entry and inducing host immune responses. It can adopt either an open or closed conformation based on the position of its receptor-binding domain (RBD). It is yet unclear what causes these conformational changes or how they influence the spike’s functions. Here, we show that Lys417 in the RBD plays dual roles in the spike’s structure: it stabilizes the closed conformation of the trimeric spike by mediating inter-spike–subunit interactions; it also directly interacts with ACE2 receptor. Hence, a K417V mutation has opposing effects on the spike’s function: it opens up the spike for better ACE2 binding while weakening the RBD’s direct binding to ACE2. The net outcomes of this mutation are to allow the spike to bind ACE2 with higher probability and mediate viral entry more efficiently, but become more exposed to neutralizing antibodies. Given that residue 417 has been a viral mutational hotspot, SARS-CoV-2 may have been evolving to strike a balance between infection potency and immune evasion, contributing to its pandemic spread.

[end paper 61]

[begin paper 62]

PMCID = 8853550

Title = Structural and antigenic variations in the spike protein of emerging SARS-CoV-2 variants

The Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) virus is continuously evolving, and this poses a major threat to antibody therapies and currently authorized Coronavirus Disease 2019 (COVID-19) vaccines. It is therefore of utmost importance to investigate and predict the putative mutations on the spike protein that confer immune evasion. Antibodies are key components of the human immune system’s response to SARS-CoV-2, and the spike protein is a prime target of neutralizing antibodies (nAbs) as it plays critical roles in host cell recognition, fusion, and virus entry. The potency of therapeutic antibodies and vaccines partly depends on how readily the virus can escape neutralization. Recent structural and functional studies have mapped the epitope landscape of nAbs on the spike protein, which illustrates the footprints of several nAbs and the site of escape mutations. In this review, we discuss (1) the emerging SARS-CoV-2 variants; (2) the structural basis for antibody-mediated neutralization of SARS-CoV-2 and nAb classification; and (3) identification of the RBD escape mutations for several antibodies that resist antibody binding and neutralization. These escape maps are a valuable tool to predict SARS-CoV-2 fitness, and in conjunction with the structures of the spike-nAb complex, they can be utilized to facilitate the rational design of escape-resistant antibody therapeutics and vaccines.

[end paper 62]

[begin paper 63]

PMCID = 10051120

Title = Biomimetic Nanotechnology for SARS-CoV-2 Treatment

More than 600 million people worldwide have been infected with severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), resulting in the pandemic of coronavirus disease 2019 (COVID-19). In particular, new waves of COVID-19 caused by emerging SARS-CoV-2 variants pose new health risks to the global population. Nanotechnology has developed excellent solutions to combat the virus pandemic, such as ACE2-based nanodecoys, nanobodies, nanovaccines, and drug nanocarriers. Lessons learned and strategies developed during this battle against SARS-CoV-2 variants may also serve as inspiration for developing nanotechnology-based strategies to combat other global infectious diseases and their variants in the future.

[end paper 63]

[begin paper 64]

PMCID = 8625641

Title = Structural Biology of Nanobodies against the Spike Protein of SARS-CoV-2

Nanobodies are 130 amino acid single-domain antibodies (VHH) derived from the unique heavy-chain-only subclass of Camelid immunogloblins. Their small molecular size, facile expression, high affinity and stability have combined to make them unique targeting reagents with numerous applications in the biomedical sciences. The first nanobody agent has now entered the clinic as a treatment against a blood disorder. The spread of the SARS-CoV-2 virus has seen the global scientific endeavour work to accelerate the development of technologies to try to defeat a pandemic that has now killed over four million people. In a remarkably short period of time, multiple studies have reported nanobodies directed against the viral Spike protein. Several agents have been tested in culture and demonstrate potent neutralisation of the virus or pseudovirus. A few agents have completed animal trials with very encouraging results showing their potential for treating infection. Here, we discuss the structural features that guide the nanobody recognition of the receptor binding domain of the Spike protein of SARS-CoV-2.

[end paper 64]

[begin paper 65]

PMCID = 8193598

Title = Molecular mechanism of interaction between SARS-CoV-2 and host cells and interventional therapy

The pandemic of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection has resulted in an unprecedented setback for global economy and health. SARS-CoV-2 has an exceptionally high level of transmissibility and extremely broad tissue tropism. However, the underlying molecular mechanism responsible for sustaining this degree of virulence remains largely unexplored. In this article, we review the current knowledge and crucial information about how SARS-CoV-2 attaches on the surface of host cells through a variety of receptors, such as ACE2, neuropilin-1, AXL, and antibody–FcγR complexes. We further explain how its spike (S) protein undergoes conformational transition from prefusion to postfusion with the help of proteases like furin, TMPRSS2, and cathepsins. We then review the ongoing experimental studies and clinical trials of antibodies, peptides, or small-molecule compounds with anti-SARS-CoV-2 activity, and discuss how these antiviral therapies targeting host–pathogen interaction could potentially suppress viral attachment, reduce the exposure of fusion peptide to curtail membrane fusion and block the formation of six-helix bundle (6-HB) fusion core. Finally, the specter of rapidly emerging SARS-CoV-2 variants deserves a serious review of broad-spectrum drugs or vaccines for long-term prevention and control of COVID-19 in the future.

[end paper 65]

[begin paper 66]

PMCID = 10702856

Title =  In vitro generated antibodies guide thermostable ADDomer nanoparticle design for nasal vaccination and passive immunization against SARS-CoV-2

Due to COVID-19, pandemic preparedness emerges as a key imperative, necessitating new approaches to accelerate development of reagents against infectious pathogens.

Here, we developed an integrated approach combining synthetic, computational and structural methods with in vitro antibody selection and in vivo immunization to design, produce and validate nature-inspired nanoparticle-based reagents against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).

Our approach resulted in two innovations: (i) a thermostable nasal vaccine called ADDoCoV, displaying multiple copies of a SARS-CoV-2 receptor binding motif derived epitope and (ii) a multivalent nanoparticle superbinder, called Gigabody, against SARS-CoV-2 including immune-evasive variants of concern (VOCs). In vitro generated neutralizing nanobodies and electron cryo-microscopy established authenticity and accessibility of epitopes displayed by ADDoCoV. Gigabody comprising multimerized nanobodies prevented SARS-CoV-2 virion attachment with picomolar EC50. Vaccinating mice resulted in antibodies cross-reacting with VOCs including Delta and Omicron.

Our study elucidates Adenovirus-derived dodecamer (ADDomer)-based nanoparticles for use in active and passive immunization and provides a blueprint for crafting reagents to combat respiratory viral infections.

Statement of Significance: COVID-19 reminds us of the critical importance of pandemic preparedness, requiring sophisticated approaches to facilitate and accelerate reagent development against emerging threats. We described an integrated pipeline approach to rapidly craft in parallel ADDomer-based nanoparticles for vaccination and for passive immunization, applicable to many infectious diseases.

[end paper 66]

[begin paper 67]

PMCID = 11148769

Title = A structure-function analysis shows SARS-CoV-2 BA.2.86 balances antibody escape and ACE2 affinity

BA.2.86, a recently described sublineage of SARS-CoV-2 Omicron, contains many mutations in the spike gene. It appears to have originated from BA.2 and is distinct from the XBB variants responsible for many infections in 2023. The global spread and plethora of mutations in BA.2.86 has caused concern that it may possess greater immune-evasive potential, leading to a new wave of infection. Here, we examine the ability of BA.2.86 to evade the antibody response to infection using a panel of vaccinated or naturally infected sera and find that it shows marginally less immune evasion than XBB.1.5. We locate BA.2.86 in the antigenic landscape of recent variants and look at its ability to escape panels of potent monoclonal antibodies generated against contemporary SARS-CoV-2 infections. We demonstrate, and provide a structural explanation for, increased affinity of BA.2.86 to ACE2, which may increase transmissibility.

BA.2.86, highly mutated from BA.2, is mapped in the landscape of recent variants

It is marginally less evasive than XBB.1.5 but escapes a panel of BA.2 antibodies

It has increased affinity for ACE2, possibly aiding transmission

Its RBD is primed for further escape at residues 455 and/or 456, as seen in JN.1

BA.2.86 spike has evolved extensively from the early Omicron variant BA.2, allowing it to escape the vast majority of anti-BA.2 monoclonal antibodies. While it is not quite as evasive as some other recent variants such as XBB.1.5, it has high affinity for the virus receptor ACE2 and is primed for further antigenic escape by mutation at residues 455 and/or 456 of the spike.

[end paper 67]

[begin paper 68]

PMCID = 9181103

Title = Newly Emerged Antiviral Strategies for SARS-CoV-2: From Deciphering Viral Protein Structural Function to the Development of Vaccines, Antibodies, and Small Molecules

Coronavirus disease 2019 (COVID-19) caused by the infection of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has become the most severe health crisis, causing extraordinary economic disruption worldwide. SARS-CoV-2 is a single-stranded RNA-enveloped virus. The process of viral replication and particle packaging is finished in host cells. Viral proteins, including both structural and nonstructural proteins, play important roles in the viral life cycle, which also provides the targets of treatment. Therefore, a better understanding of the structural function of virus proteins is crucial to speed up the development of vaccines and therapeutic strategies. Currently, the structure and function of proteins encoded by the SARS-CoV-2 genome are reviewed by several studies. However, most of them are based on the analysis of SARS-CoV-1 particles, lacking a systematic review update for SARS-CoV-2. Here, we specifically focus on the structure and function of proteins encoded by SARS-CoV-2. Viral proteins that contribute to COVID-19 infection and disease pathogenesis are reviewed according to the most recent research findings. The structure-function correlation of viral proteins provides a fundamental rationale for vaccine development and targeted therapy. Then, current antiviral vaccines are updated, such as inactive viral vaccines and protein-based vaccines and DNA, mRNA, and circular RNA vaccines. A summary of other therapeutic options is also reviewed, including monoclonal antibodies such as a cross-neutralizer antibody, a constructed cobinding antibody, a dual functional monoclonal antibody, an antibody cocktail, and an engineered bispecific antibody, as well as peptide-based inhibitors, chemical compounds, and clustered regularly interspaced short palindromic repeats (CRISPR) exploration. Overall, viral proteins and their functions provide the basis for targeted therapy and vaccine development.

[end paper 68]

[begin paper 69]

PMCID = 8368053

Title = Dynamic landscape mapping of humoral immunity to SARS-CoV-2 identifies non-structural protein antibodies associated with the survival of critical COVID-19 patients

A comprehensive analysis of the humoral immune response to the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is essential in understanding COVID-19 pathogenesis and developing antibody-based diagnostics and therapy. In this work, we performed a longitudinal analysis of antibody responses to SARS-CoV-2 proteins in 104 serum samples from 49 critical COVID-19 patients using a peptide-based SARS-CoV-2 proteome microarray. Our data show that the binding epitopes of IgM and IgG antibodies differ across SARS-CoV-2 proteins and even within the same protein. Moreover, most IgM and IgG epitopes are located within nonstructural proteins (nsps), which are critical in inactivating the host’s innate immune response and enabling SARS-CoV-2 replication, transcription, and polyprotein processing. IgM antibodies are associated with a good prognosis and target nsp3 and nsp5 proteases, whereas IgG antibodies are associated with high mortality and target structural proteins (Nucleocapsid, Spike, ORF3a). The epitopes targeted by antibodies in patients with a high mortality rate were further validated using an independent serum cohort (n = 56) and using global correlation mapping analysis with the clinical variables that are associated with COVID-19 severity. Our data provide fundamental insight into humoral immunity during SARS-CoV-2 infection. SARS-CoV-2 immunogenic epitopes identified in this work could also help direct antibody-based COVID-19 treatment and triage patients.

[end paper 69]

[begin paper 70]

PMCID = 9850341

Title = The Potential of Nanobodies for COVID-19 Diagnostics and Therapeutics

The infectious severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) is the causative agent for coronavirus disease 2019 (COVID-19). Globally, there have been millions of infections and fatalities. Unfortunately, the virus has been persistent and a contributing factor is the emergence of several variants. The urgency to combat COVID-19 led to the identification/development of various diagnosis (polymerase chain reaction and antigen tests) and treatment (repurposed drugs, convalescent plasma, antibodies and vaccines) options. These treatments may treat mild symptoms and decrease the risk of life-threatening disease. Although these options have been fairly beneficial, there are some challenges and limitations, such as cost of tests/drugs, specificity, large treatment dosages, intravenous administration, need for trained personal, lengthy production time, high manufacturing costs, and limited availability. Therefore, the development of more efficient COVID-19 diagnostic and therapeutic options are vital. Nanobodies (Nbs) are novel monomeric antigen-binding fragments derived from camelid antibodies. Advantages of Nbs include low immunogenicity, high specificity, stability and affinity. These characteristics allow for rapid Nb generation, inexpensive large-scale production, effective storage, and transportation, which is essential during pandemics. Additionally, the potential aerosolization and inhalation delivery of Nbs allows for targeted treatment delivery as well as patient self-administration. Therefore, Nbs are a viable option to target SARS-CoV-2 and overcome COVID-19. In this review we discuss (1) COVID-19; (2) SARS-CoV-2; (3) the present conventional COVID-19 diagnostics and therapeutics, including their challenges and limitations; (4) advantages of Nbs; and (5) the numerous Nbs generated against SARS-CoV-2 as well as their diagnostic and therapeutic potential.

[end paper 70]

[begin paper 71]

PMCID = 8782626

Title = Monoclonal antibodies targeting two immunodominant epitopes on the Spike protein neutralize emerging SARS-CoV-2 variants of concern

The emergence of new SARS-CoV-2 variants of concern B.1.1.7 (Alpha), B.1.351 (Beta), P.1 (Gamma) and B.1.617.2 (Delta) that harbor mutations in the viral S protein raised concern about activity of current vaccines and therapeutic antibodies. Independent studies have shown that mutant variants are partially or completely resistant against some of the therapeutic antibodies authorized for emergency use.

We employed hybridoma technology, ELISA-based and cell-based S-ACE2 interaction assays combined with authentic virus neutralization assays to develop second-generation antibodies, which were specifically selected for their ability to neutralize the new variants of SARS-CoV-2.

AX290 and AX677, two monoclonal antibodies with non-overlapping epitopes, exhibit subnanomolar or nanomolar affinities to the receptor binding domain of the viral Spike protein carrying amino acid substitutions N501Y, N439K, E484K, K417N, and a combination N501Y/E484K/K417N found in the circulating virus variants. The antibodies showed excellent neutralization of an authentic SARS-CoV-2 virus representing strains circulating in Europe in spring 2020 and also the variants of concern B.1.1.7 (Alpha), B.1.351 (Beta) and B.1.617.2 (Delta). In addition, AX677 is able to bind Omicron Spike protein just like the wild type Spike. The combination of the two antibodies prevented the appearance of escape mutations of the authentic SARS-CoV-2 virus. Prophylactic administration of AX290 and AX677, either individually or in combination, effectively reduced viral burden and inflammation in the lungs, and prevented disease in a mouse model of SARS-CoV-2 infection.

The virus-neutralizing properties were fully reproduced in chimeric mouse-human versions of the antibodies, which may represent a promising tool for COVID-19 therapy.

The study was funded by AXON Neuroscience SE and AXON COVIDAX a.s.

[end paper 71]

[begin paper 72]

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

[end paper 72]

[begin paper 73]

PMCID = 10148623

Title = Perturbation of the host cell Ca2+ homeostasis and ER-mitochondria contact sites by the SARS-CoV-2 structural proteins E and M

Coronavirus disease (COVID-19) is a contagious respiratory disease caused by the SARS-CoV-2 virus. The clinical phenotypes are variable, ranging from spontaneous recovery to serious illness and death. On March 2020, a global COVID-19 pandemic was declared by the World Health Organization (WHO). As of February 2023, almost 670 million cases and 6,8 million deaths have been confirmed worldwide. Coronaviruses, including SARS-CoV-2, contain a single-stranded RNA genome enclosed in a viral capsid consisting of four structural proteins: the nucleocapsid (N) protein, in the ribonucleoprotein core, the spike (S) protein, the envelope (E) protein, and the membrane (M) protein, embedded in the surface envelope. In particular, the E protein is a poorly characterized viroporin with high identity amongst all the β-coronaviruses (SARS-CoV-2, SARS-CoV, MERS-CoV, HCoV-OC43) and a low mutation rate. Here, we focused our attention on the study of SARS-CoV-2 E and M proteins, and we found a general perturbation of the host cell calcium (Ca2+) homeostasis and a selective rearrangement of the interorganelle contact sites. In vitro and in vivo biochemical analyses revealed that the binding of specific nanobodies to soluble regions of SARS-CoV-2 E protein reversed the observed phenotypes, suggesting that the E protein might be an important therapeutic candidate not only for vaccine development, but also for the clinical management of COVID designing drug regimens that, so far, are very limited.

[end paper 73]

[begin paper 74]

PMCID = 11788755

Title = Nanobody engineering: computational modelling and design for biomedical and therapeutic applications

Nanobodies, the smallest functional antibody fragment derived from camelid heavy‐chain‐only antibodies, have emerged as powerful tools for diverse biomedical applications. In this comprehensive review, we discuss the structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We explore their unique antigen‐binding domains, highlighting the critical role of complementarity‐determining regions in target recognition and specificity. This review further underscores the advantages of nanobodies over conventional antibodies from a biosynthesis perspective, including their small size, stability, and solubility, which make them ideal candidates for economical antigen capture in diagnostics, therapeutics, and biosensing. We discuss the recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. Finally, we examine a direct example of how computational design strategies were implemented for improving a nanobody‐based immunosensor, known as a Quenchbody. Through combining experimental findings and computational insights, this review elucidates the transformative impact of nanobodies in biotechnology and biomedical research, offering a roadmap for future advancements and applications in healthcare and diagnostics.

This review focuses on structural characteristics, functional properties, and computational approaches driving the design and optimisation of synthetic nanobodies. We discuss recent advancements in computational methods for nanobody modelling, epitope prediction, and affinity maturation, shedding light on their intricate antigen‐binding mechanisms and conformational dynamics. It underscores their advantages over conventional antibodies from a biosynthesis perspective in diagnostics, therapeutics, and biosensing. 

[end paper 74]

[begin paper 75]

PMCID = 9941073

Title = Emerging trends in point-of-care biosensing strategies for molecular architectures and antibodies of SARS-CoV-2

COVID-19, a highly contagious viral infection caused by the occurrence of severe acute respiratory syndrome coronavirus (SARS-CoV-2), has turned out to be a viral pandemic then ravaged many countries worldwide. In the recent years, point-of-care (POC) biosensors combined with state-of-the-art bioreceptors, and transducing systems enabled the development of novel diagnostic tools for rapid and reliable detection of biomarkers associated with SARS-CoV-2. The present review thoroughly summarises and discusses various biosensing strategies developed for probing SARS-CoV-2 molecular architectures (viral genome, S Protein, M protein, E protein, N protein and non-structural proteins) and antibodies as a potential diagnostic tool for COVID-19. This review discusses the various structural components of SARS-CoV-2, their binding regions and the bioreceptors used for recognizing the structural components. The various types of clinical specimens investigated for rapid and POC detection of SARS-CoV-2 is also highlighted. The importance of nanotechnology and artificial intelligence (AI) approaches in improving the biosensor performance for real-time and reagent-free monitoring the biomarkers of SARS-CoV-2 is also summarized. This review also encompasses existing practical challenges and prospects for developing new POC biosensors for clinical monitoring of COVID-19.

[end paper 75]

[begin paper 76]

PMCID = 11228696

Title = Current development of severe acute respiratory syndrome coronavirus 2 neutralizing antibodies (Review)

The coronavirus disease 2019 pandemic due to severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) seriously affected global public health security. Studies on vaccines, neutralizing antibodies (NAbs) and small molecule antiviral drugs are currently ongoing. In particular, NAbs have emerged as promising therapeutic agents due to their well-defined mechanism, high specificity, superior safety profile, ease of large-scale production and simultaneous application for both prevention and treatment of viral infection. Numerous NAb therapeutics have entered the clinical research stages, demonstrating promising therapeutic and preventive effects. These agents have been used for outbreak prevention and control under urgent authorization processes. The present review summarizes the molecular targets of SARS-CoV-2-associated NAbs and screening and identification techniques for NAb development. Moreover, the current shortcomings and challenges that persist with the use of NAbs are discussed. The aim of the present review is to offer a reference for the development of NAbs for any future emergent infectious diseases, including SARS-CoV-2.

[end paper 76]

[begin paper 77]

PMCID = 8301790

Title = COVID-19: Unmasking Emerging SARS-CoV-2 Variants, Vaccines and Therapeutic Strategies

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is the etiological agent of the coronavirus disease 2019 (COVID-19) pandemic, which has been a topic of major concern for global human health. The challenge to restrain the COVID-19 pandemic is further compounded by the emergence of several SARS-CoV-2 variants viz. B.1.1.7 (Alpha), B.1.351 (Beta), P1 (Gamma) and B.1.617.2 (Delta), which show increased transmissibility and resistance towards vaccines and therapies. Importantly, there is convincing evidence of increased susceptibility to SARS-CoV-2 infection among individuals with dysregulated immune response and comorbidities. Herein, we provide a comprehensive perspective regarding vulnerability of SARS-CoV-2 infection in patients with underlying medical comorbidities. We discuss ongoing vaccine (mRNA, protein-based, viral vector-based, etc.) and therapeutic (monoclonal antibodies, small molecules, plasma therapy, etc.) modalities designed to curb the COVID-19 pandemic. We also discuss in detail, the challenges posed by different SARS-CoV-2 variants of concern (VOC) identified across the globe and their effects on therapeutic and prophylactic interventions.

[end paper 77]

[begin paper 78]

PMCID = 11003867

Title = Antibody-independent protection against heterologous SARS-CoV-2 challenge conferred by prior infection or vaccination

Vaccines have reduced severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) morbidity and mortality, yet emerging variants challenge their effectiveness. The prevailing approach to updating vaccines targets the antibody response, operating under the presumption that it is the primary defense mechanism following vaccination or infection. This perspective, however, can overlook the role of T cells, particularly when antibody levels are low or absent. Here we show, through studies in mouse models lacking antibodies but maintaining functional B cells and lymphoid organs, that immunity conferred by prior infection or mRNA vaccination can protect against SARS-CoV-2 challenge independently of antibodies. Our findings, using three distinct models inclusive of a novel human/mouse ACE2 hybrid, highlight that CD8+ T cells are essential for combating severe infections, whereas CD4+ T cells contribute to managing milder cases, with interferon-γ having an important function in this antibody-independent defense. These findings highlight the importance of T cell responses in vaccine development, urging a broader perspective on protective immunity beyond just antibodies.

Here the authors use three different mouse models to show that prior infection or mRNA vaccination can protect against severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) independently of antibodies, highlighting the importance of T cell-derived interferon-γ (IFN-γ) in host defense and the need to consider this measure of protection in vaccination.

[end paper 78]

[begin paper 79]

PMCID = 10891761

Title = Serial Llama Immunization with Various SARS-CoV-2 RBD Variants Induces Broad Spectrum Virus-Neutralizing Nanobodies

The emergence of SARS-CoV-2 mutant variants has posed a significant challenge to both the prevention and treatment of COVID-19 with anti-coronaviral neutralizing antibodies. The latest viral variants demonstrate pronounced resistance to the vast majority of human monoclonal antibodies raised against the ancestral Wuhan variant. Less is known about the susceptibility of the evolved virus to camelid nanobodies developed at the start of the pandemic. In this study, we compared nanobody repertoires raised in the same llama after immunization with Wuhan’s RBD variant and after subsequent serial immunization with a variety of RBD variants, including that of SARS-CoV-1. We show that initial immunization induced highly potent nanobodies, which efficiently protected Syrian hamsters from infection with the ancestral Wuhan virus. These nanobodies, however, mostly lacked the activity against SARS-CoV-2 omicron-pseudotyped viruses. In contrast, serial immunization with different RBD variants resulted in the generation of nanobodies demonstrating a higher degree of somatic mutagenesis and a broad range of neutralization. Four nanobodies recognizing distinct epitopes were shown to potently neutralize a spectrum of omicron variants, including those of the XBB sublineage. Our data show that nanobodies broadly neutralizing SARS-CoV-2 variants may be readily induced by a serial variant RBD immunization.

[end paper 79]

[begin paper 80]

PMCID = 8324831

Title = A synthetic nanobody targeting RBD protects hamsters from SARS-CoV-2 infection

SARS-CoV-2, the causative agent of COVID-19, features a receptor-binding domain (RBD) for binding to the host cell ACE2 protein. Neutralizing antibodies that block RBD-ACE2 interaction are candidates for the development of targeted therapeutics. Llama-derived single-domain antibodies (nanobodies, ~15 kDa) offer advantages in bioavailability, amenability, and production and storage owing to their small sizes and high stability. Here, we report the rapid selection of 99 synthetic nanobodies (sybodies) against RBD by in vitro selection using three libraries. The best sybody, MR3 binds to RBD with high affinity (KD = 1.0 nM) and displays high neutralization activity against SARS-CoV-2 pseudoviruses (IC50 = 0.42 μg mL−1). Structural, biochemical, and biological characterization suggests a common neutralizing mechanism, in which the RBD-ACE2 interaction is competitively inhibited by sybodies. Various forms of sybodies with improved potency have been generated by structure-based design, biparatopic construction, and divalent engineering. Two divalent forms of MR3 protect hamsters from clinical signs after live virus challenge and a single dose of the Fc-fusion construct of MR3 reduces viral RNA load by 6 Log10. Our results pave the way for the development of therapeutic nanobodies against COVID-19 and present a strategy for rapid development of targeted medical interventions during an outbreak.

Here, the authors report the engineering, structural and biological characterization of synthetic nanobodies (sybodies) that display potent therapeutic activity against SARS-CoV-2 infection in animal models via targeting the virus receptor-binding domain.

[end paper 80]

[begin paper 81]

PMCID = 8564274

Title = A non-ACE2 competing human single-domain antibody confers broad neutralization against SARS-CoV-2 and circulating variants

The current COVID-19 pandemic has heavily burdened the global public health system and may keep simmering for years. The frequent emergence of immune escape variants have spurred the search for prophylactic vaccines and therapeutic antibodies that confer broad protection against SARS-CoV-2 variants. Here we show that the bivalency of an affinity maturated fully human single-domain antibody (n3113.1-Fc) exhibits exquisite neutralizing potency against SARS-CoV-2 pseudovirus, and confers effective prophylactic and therapeutic protection against authentic SARS-CoV-2 in the host cell receptor angiotensin-converting enzyme 2 (ACE2) humanized mice. The crystal structure of n3113 in complex with the receptor-binding domain (RBD) of SARS-CoV-2, combined with the cryo-EM structures of n3113 and spike ecto-domain, reveals that n3113 binds to the side surface of up-state RBD with no competition with ACE2. The binding of n3113 to this novel epitope stabilizes spike in up-state conformations but inhibits SARS-CoV-2 S mediated membrane fusion, expanding our recognition of neutralization by antibodies against SARS-CoV-2. Binding assay and pseudovirus neutralization assay show no evasion of recently prevalent SARS-CoV-2 lineages, including Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), and Delta (B.1.617.2) for n3113.1-Fc with Y58L mutation, demonstrating the potential of n3113.1-Fc (Y58L) as a promising candidate for clinical development to treat COVID-19.

[end paper 81]

[begin paper 82]

PMCID = 8006950

Title = Lessons learned 1 year after SARS-CoV-2 emergence leading to COVID-19 pandemic

Without modern medical management and vaccines, the severity of the Coronavirus Disease 2019 (COVID-19) pandemic caused by severe acute respiratory syndrome (SARS) coronavirus 2 (SARS-CoV-2) might approach the magnitude of 1894-plague (12 million deaths) and 1918-A(H1N1) influenza (50 million deaths) pandemics. The COVID-19 pandemic was heralded by the 2003 SARS epidemic which led to the discovery of human and civet SARS-CoV-1, bat SARS-related-CoVs, Middle East respiratory syndrome (MERS)-related bat CoV HKU4 and HKU5, and other novel animal coronaviruses. The suspected animal-to-human jumping of 4 betacoronaviruses including the human coronaviruses OC43(1890), SARS-CoV-1(2003), MERS-CoV(2012), and SARS-CoV-2(2019) indicates their significant pandemic potential. The presence of a large reservoir of coronaviruses in bats and other wild mammals, culture of mixing and selling them in urban markets with suboptimal hygiene, habit of eating exotic mammals in highly populated areas, and the rapid and frequent air travels from these areas are perfect ingredients for brewing rapidly exploding epidemics. The possibility of emergence of a hypothetical SARS-CoV-3 or other novel viruses from animals or laboratories, and therefore needs for global preparedness should not be ignored. We reviewed representative publications on the epidemiology, virology, clinical manifestations, pathology, laboratory diagnostics, treatment, vaccination, and infection control of COVID-19 as of 20 January 2021, which is 1 year after person-to-person transmission of SARS-CoV-2 was announced. The difficulties of mass testing, labour-intensive contact tracing, importance of compliance to universal masking, low efficacy of antiviral treatment for severe disease, possibilities of vaccine or antiviral-resistant virus variants and SARS-CoV-2 becoming another common cold coronavirus are discussed.

[end paper 82]

[begin paper 83]

PMCID = 8476643

Title = Cross-neutralizing antibodies bind a SARS-CoV-2 cryptic site and resist circulating variants

The emergence of numerous variants of SARS-CoV-2, the causative agent of COVID-19, has presented new challenges to the global efforts to control the COVID-19 pandemic. Here, we obtain two cross-neutralizing antibodies (7D6 and 6D6) that target Sarbecoviruses’ receptor-binding domain (RBD) with sub-picomolar affinities and potently neutralize authentic SARS-CoV-2. Crystal structures show that both antibodies bind a cryptic site different from that recognized by existing antibodies and highly conserved across Sarbecovirus isolates. Binding of these two antibodies to the RBD clashes with the adjacent N-terminal domain and disrupts the viral spike. Both antibodies confer good resistance to mutations in the currently circulating SARS-CoV-2 variants. Thus, our results have direct relevance to public health as options for passive antibody therapeutics and even active prophylactics. They can also inform the design of pan-sarbecovirus vaccines.

Antibodies (Abs) targeting highly conserved epitopes are important tools against emerging virus variants. Here, the authors characterize Abs that recognize a cryptic epitope in the receptor-binding domain of SARS-CoV-2 spike that is well conserved and show that these Abs can neutralize several variants of concerns.

[end paper 83]

[begin paper 84]

PMCID = 10315470

Title = SARS-CoV-2 and the host-immune response

The SARS-CoV-2 pandemic and the COVID-19 disease have affected everyone globally, leading to one of recorded history’s most significant research surges. As our knowledge evolves, our approaches to the virus and treatments must also evolve. The evaluation of future research approaches to SARS-CoV-2 will necessitate reviewing the host immune response and viral antagonism of that response. This review provides an overview of the current knowledge on SARS-CoV-2 by summarizing the virus and human response. The focuses are on the viral genome, replication cycle, host immune activation, response, signaling, and antagonism. To effectively fight the pandemic, efforts must focus on the current state of research to help develop treatments and prepare for future outbreaks.

[end paper 84]

[begin paper 85]

PMCID = 9407951

Title = An Antibody from Single Human VH-rearranging Mouse Neutralizes All SARS-CoV-2 Variants Through BA.5 by Inhibiting Membrane Fusion

SARS-CoV-2 Omicron sub-variants have generated a world-wide health crisis due to resistance to most approved SARS-CoV-2 neutralizing antibodies and evasion of vaccination-induced antibodies. To manage Omicron sub-variants and prepare for potential new variants, additional means of isolating broad and potent humanized SARS-CoV-2-neutralizing antibodies are desirable. Here, we describe a mouse model in which the primary B cell receptor (BCR) repertoire is generated solely through V(D)J recombination of a human VH1-2 heavy chain (HC) and, substantially, a human Vκ1-33 light chain (LC). Thus, primary humanized BCR repertoire diversity in these mice derives from immensely diverse HC and LC antigen-contact complementarity-region-3 (CDR3) sequences generated by non-templated junctional modifications during V(D)J recombination. Immunizing the human VH1-2/Vκ1-33-rearranging mouse model with SARS-CoV-2 (Wuhan-Hu-1) spike protein immunogens elicited several VH1-2/Vκ1-33-based neutralizing antibodies that bound RBD in a different mode from each other and from those of many prior human patient-derived VH1-2-based neutralizing antibodies. Of these, SP1-77 potently and broadly neutralized all SARS-CoV-2 variants through BA.5. Cryo-EM studies revealed that SP1-77 bound RBD away from the receptor-binding-motif via a CDR3-dominated recognition mode. Lattice-light-sheet-microscopy-based studies showed that SP1-77 did not block ACE2-mediated viral attachment or endocytosis, but rather blocked viral-host membrane fusion. The broad and potent SP1-77 neutralization activity and non-traditonal mechanism of action suggest this antibody might have therapeutic potential. Likewise, the SP1-77 binding epitope may further inform on vacccine strategies. Finally, the general class of humanized mouse models we have described may contribute to identifying therapeutic antibodies against future SARS-CoV-2 variants and other pathogens.

A humanized antibody from a recently-developed mouse model potently neutralizes SARS-CoV-2 variants by inhibiting membrane fusion.

[end paper 85]

[begin paper 86]

PMCID = 9355996

Title = Conformational flexibility in neutralization of SARS-CoV-2 by naturally elicited anti-SARS-CoV-2 antibodies

As new variants of SARS-CoV-2 continue to emerge, it is important to assess the cross-neutralizing capabilities of antibodies naturally elicited during wild type SARS-CoV-2 infection. In the present study, we evaluate the activity of nine anti-SARS-CoV-2 monoclonal antibodies (mAbs), previously isolated from convalescent donors infected with the Wuhan-Hu-1 strain, against the SARS-CoV-2 variants of concern (VOC) Alpha, Beta, Gamma, Delta and Omicron. By testing an array of mutated spike receptor binding domain (RBD) proteins, cell-expressed spike proteins from VOCs, and neutralization of SARS-CoV-2 VOCs as pseudoviruses, or as the authentic viruses in culture, we show that mAbs directed against the ACE2 binding site (ACE2bs) are more sensitive to viral evolution compared to anti-RBD non-ACE2bs mAbs, two of which retain their potency against all VOCs tested. At the second part of our study, we reveal the neutralization mechanisms at high molecular resolution of two anti-SARS-CoV-2 neutralizing mAbs by structural characterization. We solve the structures of the Delta-neutralizing ACE2bs mAb TAU-2303 with the SARS-CoV-2 spike trimer and RBD at 4.5 Å and 2.42 Å resolutions, respectively, revealing a similar mode of binding to that between the RBD and ACE2. Furthermore, we provide five additional structures (at resolutions of 4.7 Å, 7.3 Å, 6.4 Å, 3.3 Å, and 6.1 Å) of a second antibody, TAU-2212, complexed with the SARS-CoV-2 spike trimer. TAU-2212 binds an exclusively quaternary epitope, and exhibits a unique, flexible mode of neutralization that involves transitioning between five different conformations, with both arms of the antibody recruited for cross linking intra- and inter-spike RBD subunits. Our study provides additional mechanistic understanding about how antibodies neutralize SARS-CoV-2 and its emerging variants and provides insights on the likelihood of reinfections.

The neutralization of SARS-CoV-2 and variants of concern by nine monoclonal antibodies (mAb) isolated from convalescent donors infected with the Wuhan-Hu-1 strain alongside structural characterization of two of the mAbs in complex with the RBD and spike are presented.

[end paper 86]

[begin paper 87]

PMCID = 8139655

Title = Potent Neutralization of SARS-CoV-2 by Hetero-Bivalent Alpaca Nanobodies Targeting the Spike Receptor-Binding Domain

To date, SARS-CoV-2 has caused tremendous loss of human life and economic output worldwide. Although a few COVID-19 vaccines have been approved in several countries, the development of effective therapeutics, including SARS-CoV-2 targeting antibodies, remains critical.

Cellular entry of SARS-CoV-2 requires the binding between the receptor-binding domain (RBD) of the viral Spike protein and the cellular angiotensin-converting enzyme 2 (ACE2). As such, RBD has become the major target for vaccine development, whereas RBD-specific antibodies are pursued as therapeutics. Here, we report the development and characterization of SARS-CoV-2 RBD-specific VHH/nanobody (Nb) from immunized alpacas. Seven RBD-specific Nbs with high stability were identified using phage display. They bind to SARS-CoV-2 RBD with affinity KD ranging from 2.6 to 113 nM, and six of them can block RBD-ACE2 interaction. The fusion of the Nbs with IgG1 Fc resulted in homodimers with greatly improved RBD-binding affinities (KD ranging from 72.7 pM to 4.5 nM) and nanomolar RBD-ACE2 blocking abilities. Furthermore, the fusion of two Nbs with nonoverlapping epitopes resulted in hetero-bivalent Nbs, namely, aRBD-2-5 and aRBD-2-7, with significantly higher RBD binding affinities (KD of 59.2 pM and 0.25 nM) and greatly enhanced SARS-CoV-2 neutralizing potency. The 50% neutralization doses of aRBD-2-5 and aRBD-2-7 were 1.22 ng/ml (∼0.043 nM) and 3.18 ng/ml (∼0.111 nM), respectively. These high-affinity SARS-CoV-2 blocking Nbs could be further developed into therapeutics, as well as diagnostic reagents for COVID-19.

IMPORTANCE To date, SARS-CoV-2 has caused tremendous loss of human life and economic output worldwide. Although a few COVID-19 vaccines have been approved in several countries, the development of effective therapeutics, including SARS-CoV-2 targeting antibodies, remains critical. Due to their small size (13 to 15 kDa), high solubility, and stability, Nbs are particularly well suited for pulmonary delivery and more amenable to engineer into multivalent formats than the conventional antibody. Here, we report a series of new anti-SARS-CoV-2 Nbs isolated from immunized alpaca and two engineered hetero-bivalent Nbs. These potent neutralizing Nbs showed promise as potential therapeutics against COVID-19.

[end paper 87]

[begin paper 88]

PMCID = 10373554

Title = Broad Sarbecovirus Neutralizing Antibodies Obtained by Computational Design and Synthetic Library Screening

Members of the Sarbecovirus subgenus of Coronaviridae have twice caused deadly threats to humans. There is increasing concern about the rapid mutation of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), which has evolved into multiple generations of epidemic variants in 3 years. Broad neutralizing antibodies are of great importance for pandemic preparedness against SARS-CoV-2 variants and divergent zoonotic sarbecoviruses. Here, we analyzed the structural conservation of the receptor-binding domain (RBD) from representative sarbecoviruses and chose S2H97, a previously reported RBD antibody with ideal breadth and resistance to escape, as a template for computational design to enhance the neutralization activity and spectrum. A total of 35 designs were purified for evaluation. The neutralizing activity of a large proportion of these designs against multiple variants was increased from several to hundreds of times. Molecular dynamics simulation suggested that extra interface contacts and enhanced intermolecular interactions between the RBD and the designed antibodies are established. After light and heavy chain reconstitution, AI-1028, with five complementarity determining regions optimized, showed the best neutralizing activity across all tested sarbecoviruses, including SARS-CoV, multiple SARS-CoV-2 variants, and bat-derived viruses. AI-1028 recognized the same cryptic RBD epitope as the parental prototype antibody. In addition to computational design, chemically synthesized nanobody libraries are also a precious resource for rapid antibody development. By applying distinct RBDs as baits for reciprocal screening, we identified two novel nanobodies with broad activities. These findings provide potential pan-sarbecovirus neutralizing drugs and highlight new pathways to rapidly optimize therapeutic candidates when novel SARS-CoV-2 escape variants or new zoonotic coronaviruses emerge.

IMPORTANCE The subgenus Sarbecovirus includes human SARS-CoV, SARS-CoV-2, and hundreds of genetically related bat viruses. The continuous evolution of SARS-CoV-2 has led to the striking evasion of neutralizing antibody (NAb) drugs and convalescent plasma. Antibodies with broad activity across sarbecoviruses would be helpful to combat current SARS-CoV-2 mutations and longer term animal virus spillovers. The study of pan-sarbecovirus NAbs described here is significant for the following reasons. First, we established a structure-based computational pipeline to design and optimize NAbs to obtain more potent and broader neutralizing activity across multiple sarbecoviruses. Second, we screened and identified nanobodies from a highly diversified synthetic library with a broad neutralizing spectrum using an elaborate screening strategy. These methodologies provide guidance for the rapid development of antibody therapeutics against emerging pathogens with highly variable characteristics.

[end paper 88]

[begin paper 89]

PMCID = 7822889

Title = A COVID-19 vaccine candidate using SpyCatcher multimerization of the SARS-CoV-2 spike protein receptor-binding domain induces potent neutralising antibody responses

There is need for effective and affordable vaccines against SARS-CoV-2 to tackle the ongoing pandemic. In this study, we describe a protein nanoparticle vaccine against SARS-CoV-2. The vaccine is based on the display of coronavirus spike glycoprotein receptor-binding domain (RBD) on a synthetic virus-like particle (VLP) platform, SpyCatcher003-mi3, using SpyTag/SpyCatcher technology. Low doses of RBD-SpyVLP in a prime-boost regimen induce a strong neutralising antibody response in mice and pigs that is superior to convalescent human sera. We evaluate antibody quality using ACE2 blocking and neutralisation of cell infection by pseudovirus or wild-type SARS-CoV-2. Using competition assays with a monoclonal antibody panel, we show that RBD-SpyVLP induces a polyclonal antibody response that recognises key epitopes on the RBD, reducing the likelihood of selecting neutralisation-escape mutants. Moreover, RBD-SpyVLP is thermostable and can be lyophilised without losing immunogenicity, to facilitate global distribution and reduce cold-chain dependence. The data suggests that RBD-SpyVLP provides strong potential to address clinical and logistic challenges of the COVID-19 pandemic.

Vaccines for SARS-COV-2 are needed in the ongoing pandemic. Here the authors characterize a vaccine candidate that presents the receptor-binding domain (RBD) of SARS-CoV-2 spike protein on a synthetic VLP platform using SpyTag/SpyCatcher technology and show immunogenicity of a prime-boost regimen in mice and pigs.

[end paper 89]

[begin paper 90]

PMCID = 8201996

Title = Potent Molecular Feature-based Neutralizing Monoclonal Antibodies as Promising Therapeutics Against SARS-CoV-2 Infection

The 2019–2020 winter was marked by the emergence of a new coronavirus (SARS-CoV-2) related disease (COVID-19), which started in Wuhan, China. Its high human-to-human transmission ability led to a worldwide spread within few weeks and has caused substantial human loss. Mechanical antiviral control approach, drug repositioning, and use of COVID-19 convalescent plasmas (CPs) were the first line strategies utilized to mitigate the viral spread, yet insufficient. The urgent need to contain this deadly pandemic has led searchers and pharmaceutical companies to develop vaccines. However, not all vaccines manufactured are safe. Besides, an alternative and effective treatment option for such an infectious disease would include pure anti-viral neutralizing monoclonal antibodies (NmAbs), which can block the virus at specific molecular targets from entering cells by inhibiting virus-cell structural complex formation, with more safety and efficiency than the CP. Indeed, there is a lot of molecular evidence about the protector effect and the use of molecular feature-based NmAbs as promising therapeutics to contain COVID-19. Thus, from the scientific publication database screening, we here retrieved antibody-related papers and summarized the repertory of characterized NmAbs against SARS-CoV-2, their molecular neutralization mechanisms, and their immunotherapeutic pros and cons. About 500 anti-SARS-CoV-2 NmAbs, characterized through competitive binding assays and neutralization efficacy, were reported at the writing time (January 2021). All NmAbs bind respectively to SARS-CoV-2 S and exhibit high molecular neutralizing effects against wild-type and/or pseudotyped virus. Overall, we defined six NmAb groups blocking SARS-CoV-2 through different molecular neutralization mechanisms, from which five potential neutralization sites on SARS-CoV-2 S protein are described. Therefore, more efforts are needed to develop NmAbs-based cocktails to mitigate COVID-19.

[end paper 90]

[begin paper 91]

PMCID = 9688389

Title = Angiotensin-Converting Enzyme 2-Based Biosensing Modalities and Devices for Coronavirus Detection

Rapid and cost-effective diagnostic tests for severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) are a critical and valuable weapon for the coronavirus disease 2019 (COVID-19) pandemic response. SARS-CoV-2 invasion is primarily mediated by human angiotensin-converting enzyme 2 (hACE2). Recent developments in ACE2-based SARS-CoV-2 detection modalities accentuate the potential of this natural host-virus interaction for developing point-of-care (POC) COVID-19 diagnostic systems. Although research on harnessing ACE2 for SARS-CoV-2 detection is in its infancy, some interesting biosensing devices have been developed, showing the commercial viability of this intriguing new approach. The exquisite performance of the reported ACE2-based COVID-19 biosensors provides opportunities for researchers to develop rapid detection tools suitable for virus detection at points of entry, workplaces, or congregate scenarios in order to effectively implement pandemic control and management plans. However, to be considered as an emerging approach, the rationale for ACE2-based biosensing needs to be critically and comprehensively surveyed and discussed. Herein, we review the recent status of ACE2-based detection methods, the signal transduction principles in ACE2 biosensors and the development trend in the future. We discuss the challenges to development of ACE2-biosensors and delineate prospects for their use, along with recommended solutions and suggestions.

[end paper 91]

[begin paper 92]

PMCID = 9481167

Title = Biosynthetic proteins targeting the SARS-CoV-2 spike as anti-virals

The binding of the SARS-CoV-2 spike to angiotensin-converting enzyme 2 (ACE2) promotes virus entry into the cell. Targeting this interaction represents a promising strategy to generate antivirals. By screening a phage-display library of biosynthetic protein sequences build on a rigid alpha-helicoidal HEAT-like scaffold (named αReps), we selected candidates recognizing the spike receptor binding domain (RBD). Two of them (F9 and C2) bind the RBD with affinities in the nM range, displaying neutralisation activity in vitro and recognizing distinct sites, F9 overlapping the ACE2 binding motif. The F9-C2 fusion protein and a trivalent αRep form (C2-foldon) display 0.1 nM affinities and EC50 of 8–18 nM for neutralization of SARS-CoV-2. In hamsters, F9-C2 instillation in the nasal cavity before or during infections effectively reduced the replication of a SARS-CoV-2 strain harbouring the D614G mutation in the nasal epithelium. Furthermore, F9-C2 and/or C2-foldon effectively neutralized SARS-CoV-2 variants (including delta and omicron variants) with EC50 values ranging from 13 to 32 nM. With their high stability and their high potency against SARS-CoV-2 variants, αReps provide a promising tool for SARS-CoV-2 therapeutics to target the nasal cavity and mitigate virus dissemination in the proximal environment.

The entry of SARS-CoV-2 in permissive cells is mediated by the binding of its spike to angiotensin-converting enzyme 2 (ACE2) on the cell surface. To select ligands able to block this interaction, we screened a library of phages encoding biosynthetic proteins (named αReps) for binding to its receptor binding domain (RBD). Two of them were able to bind the RBD with high affinity and block efficiently the virus entry in cultured cells. Assembled αReps through covalent or non-covalent linkages blocked virus entry at lower concentration than their precursors (with around 20-fold activity increase for a trimeric αRep). These αReps derivates neutralize efficiently SARS-CoV-2 β, γ, δ and Omicron virus variants. Instillation of an αRep dimer in the nasal cavity effectively reduced virus replication in the hamster model of SARS-CoV-2 and pathogenicity.

[end paper 92]

[begin paper 93]

PMCID = 10060843

Title = Review of therapeutic mechanisms and applications based on SARS-CoV-2 neutralizing antibodies

COVID-19 pandemic is a global public health emergency. Despite extensive research, there are still few effective treatment options available today. Neutralizing-antibody-based treatments offer a broad range of applications, including the prevention and treatment of acute infectious diseases. Hundreds of SARS-CoV-2 neutralizing antibody studies are currently underway around the world, with some already in clinical applications. The development of SARS-CoV-2 neutralizing antibody opens up a new therapeutic option for COVID-19. We intend to review our current knowledge about antibodies targeting various regions (i.e., RBD regions, non-RBD regions, host cell targets, and cross-neutralizing antibodies), as well as the current scientific evidence for neutralizing-antibody-based treatments based on convalescent plasma therapy, intravenous immunoglobulin, monoclonal antibodies, and recombinant drugs. The functional evaluation of antibodies (i.e., in vitro or in vivo assays) is also discussed. Finally, some current issues in the field of neutralizing-antibody-based therapies are highlighted.

[end paper 93]

[begin paper 94]

PMCID = 9169858

Title = An extended conformation of SARS-CoV-2 main protease reveals allosteric targets

The coronavirus main protease (Mpro) is required for viral replication. Here, we obtained the extended conformation of the native monomer of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) Mpro by trapping it with nanobodies and found that the catalytic domain and the helix domain dissociate, revealing allosteric targets. Another monomeric state is termed compact conformation and is similar to one protomer of the dimeric form. We designed a Nanoluc Binary Techonology (NanoBiT)-based high-throughput allosteric inhibitor assay based on structural conformational change. Our results provide insight into the maturation, dimerization, and catalysis of the coronavirus Mpro and pave a way to develop an anticoronaviral drug through targeting the maturation process to inhibit the autocleavage of Mpro.

The coronavirus main protease (Mpro) is required for viral replication and has enzymatical activity as a homodimer. Thus, targeting its dimerization is an effective strategy for developing allosteric inhibitors to suppress mutation escape. In this study, we obtained the extended conformation of the native monomer of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) Mpro by trapping it with nanobodies, and found that the catalytic domain and the helix domain dissociate, revealing allosteric targets. We also found another state, a compact conformation, similar to the dimeric form. Our data support that the Mpro may be in equilibrium among the monomeric extended conformation as the precursor of all other states, the compact conformation as the intermediate state, and the dimeric conformation as the active state. We designed an innovative Nanoluc Binary Technology (NanoBiT)-based high-throughput allosteric inhibitor assay based on the rearranged conformation. In addition, we identified a set of allosteric inhibitory nanobodies against Mpro, one of which is also a competitive inhibitor of Mpro. Our results provide insight into the maturation of the coronavirus Mpro and a way to develop anticoronaviral drugs through targeting the folding process to inhibit the autocleavage of the main protease.

[end paper 94]

[begin paper 95]

PMCID = 11100608

Title = Bispecific antibodies with broad neutralization potency against SARS-CoV-2 variants of concern

The ongoing emergence of SARS-CoV-2 variants of concern (VOCs) that reduce the effectiveness of antibody therapeutics necessitates development of next-generation antibody modalities that are resilient to viral evolution. Here, we characterized N-terminal domain (NTD) and receptor binding domain (RBD)-specific monoclonal antibodies previously isolated from COVID-19 convalescent donors for their activity against emergent SARS-CoV-2 VOCs. Among these, the NTD-specific antibody C1596 displayed the greatest breadth of binding to VOCs, with cryo-EM structural analysis revealing recognition of a distinct NTD epitope outside of the site i antigenic supersite. Given C1596’s favorable binding profile, we designed a series of bispecific antibodies (bsAbs) termed CoV2-biRNs, that featured both NTD and RBD specificities. Notably, two of the C1596-inclusive bsAbs, CoV2-biRN5 and CoV2-biRN7, retained potent in vitro neutralization activity against all Omicron variants tested, including XBB.1.5, EG.5.1, and BA.2.86, contrasting the diminished potency of parental antibodies delivered as monotherapies or as a cocktail. Furthermore, prophylactic delivery of CoV2-biRN5 significantly reduced the viral load within the lungs of K18-hACE2 mice following challenge with SARS-CoV-2 XBB.1.5. In conclusion, our NTD-RBD bsAbs offer promising potential for the design of resilient, next-generation antibody therapeutics against SARS-CoV-2 VOCs.

Bispecific antibodies with a highly cross-reactive NTD antibody demonstrate resilience to SARS-CoV-2 variants of concern.

[end paper 95]

[begin paper 96]

PMCID = 8606259

Title = Emerging mutations in the SARS-CoV-2 variants and their role in antibody escape to small molecule-based therapeutic resistance

Several clinical trials started during the COVID-19 pandemic to discover effective therapeutics led to identify a few candidates from the major clinical trials. However, in the past several months, quite a few SARS-CoV-2 variants have emerged with significant mutations. Major mutations in the S-glycoprotein and other parts of the genome have led to the antibody's escape to small molecule-based therapeutic resistance. The mutations in S-glycoprotein trigger the antibody escape/resistance, and mutations in RdRp might cause remdesivir resistance. The article illustrates emerging mutations that have resulted in antibody escape to therapeutics resistance. In this direction, the article illustrates presently developed neutralizing antibodies (with their preclinical, clinical stages) and antibody escapes and associated mutations. Finally, owing to the RdRp mutations, the antiviral small molecules resistance is illustrated.

[end paper 96]

[begin paper 97]

PMCID = 8221914

Title = Ultrapotent miniproteins targeting the SARS-CoV-2 receptor-binding domain protect against infection and disease

Despite the introduction of public health measures and spike protein-based vaccines to mitigate the COVID-19 pandemic, SARS-CoV-2 infections and deaths continue to have a global impact. Previously, we used a structural design approach to develop picomolar range miniproteins targeting the SARS-CoV-2 spike receptor-binding domain. Here, we investigated the capacity of modified versions of one lead miniprotein, LCB1, to protect against SARS-CoV-2-mediated lung disease in mice. Systemic administration of LCB1-Fc reduced viral burden, diminished immune cell infiltration and inflammation, and completely prevented lung disease and pathology. A single intranasal dose of LCB1v1.3 reduced SARS-CoV-2 infection in the lung when given as many as 5 days before or 2 days after virus inoculation. Importantly, LCB1v1.3 protected in vivo against a historical strain (WA1/2020), an emerging B.1.1.7 strain, and a strain encoding key E484K and N501Y spike protein substitutions. These data support development of LCB1v1.3 for prevention or treatment of SARS-CoV-2 infection.

Case et al. show that de novo-designed SARS-CoV-2 RBD-specific miniproteins administered as prophylaxis or therapy protect mice against SARS-CoV-2-mediated disease, inflammation, and pathology. Injected or intranasal administration of miniproteins showed efficacy when given any time over an 8-day treatment window.

[end paper 97]

[begin paper 98]

PMCID = 9491202

Title = Sensitivity to Vaccines, Therapeutic Antibodies, and Viral Entry Inhibitors and Advances To Counter the SARS-CoV-2 Omicron Variant

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) keeps evolving and mutating into newer variants over time, which gain higher transmissibility, disease severity, and spread in communities at a faster rate, resulting in multiple waves of surge in Coronavirus Disease 2019 (COVID-19) cases. A highly mutated and transmissible SARS-CoV-2 Omicron variant has recently emerged, driving the extremely high peak of infections in almost all continents at an unprecedented speed and scale. The Omicron variant evades the protection rendered by vaccine-induced antibodies and natural infection, as well as overpowers the antibody-based immunotherapies, raising the concerns of current effectiveness of available vaccines and monoclonal antibody-based therapies. This review outlines the most recent advancements in studying the virology and biology of the Omicron variant, highlighting its increased resistance to current antibody-based therapeutics and its immune escape against vaccines. However, the Omicron variant is highly sensitive to viral fusion inhibitors targeting the HR1 motif in the spike protein, enzyme inhibitors, involving the endosomal fusion pathway, and ACE2-based entry inhibitors. Omicron variant-associated infectivity and entry mechanisms of Omicron variant are essentially distinct from previous characterized variants. Innate sensing and immune evasion of SARS-CoV-2 and T cell immunity to the virus provide new perspectives of vaccine and drug development. These findings are important for understanding SARS-CoV-2 viral biology and advances in developing vaccines, antibody-based therapies, and more effective strategies to mitigate the transmission of the Omicron variant or the next SARS-CoV-2 variant of concern.

[end paper 98]

[begin paper 99]

PMCID = 9174178

Title = Superimmunity by pan-sarbecovirus nanobodies

Vaccine boosters and infection can facilitate the development of SARS-CoV-2 antibodies with improved potency and breadth. Here, we observe superimmunity in a camelid extensively immunized with the SARS-CoV-2 receptor-binding domain (RBD). We rapidly isolate a large repertoire of specific ultra-high-affinity nanobodies that bind strongly to all known sarbecovirus clades using integrative proteomics. These pan-sarbecovirus nanobodies (psNbs) are highly effective against SARS-CoV and SARS-CoV-2 variants, including Omicron, with the best median neutralization potency at single-digit nanograms per milliliter. A highly potent, inhalable, and bispecific psNb (PiN-31) is also developed. Structural determinations of 13 psNbs with the SARS-CoV-2 spike or RBD reveal five epitope classes, providing insights into the mechanisms and evolution of their broad activities. The highly evolved psNbs target small, flat, and flexible epitopes that contain over 75% of conserved RBD surface residues. Their potencies are strongly and negatively correlated with the distance of the epitopes from the receptor binding sites.

Xiang et al. characterize 100 diverse pan-sarbecovirus nanobodies (psNbs) that are highly effective against SARS-CoV and SARS-CoV-2 variants, including Omicron, with the best median neutralization potency at single-digit nanograms per milliliter. Structural determinations of 13 psNbs reveal five RBD epitope classes, providing insights into the mechanisms and evolution of their broad activities.

[end paper 99]

[begin paper 100]

PMCID = 10578311

Title = Comparative Study of the Mutations Observed in the SARS-CoV-2 RBD Variants of Concern and Their Impact on the Interaction with the ACE2 Protein

SARS-CoV-2 strains have made an appearance across the globe, causing over 757 million cases and over 6.85 million deaths at the time of writing. The emergence of these variants shows the amplitude of genetic variation to which the wild-type strains have been subjected. The rise of the different SARS-CoV-2 variants resulting from such genetic modification has significantly affected COVD-19’s major impact on proliferation, virulence, and clinics. With the emergence of the variants of concern, the spike protein has been identified as a possible therapeutic target due to its critical role in binding to human cells and pathogenesis. These mutations could be linked to functional heterogeneity and use a different infection strategy. For example, the Omicron variant’s multiple mutations should be carefully examined, as they represent one of the most widely spread strains and hint to us that there may be more genetic changes in the virus. As a result, we applied a common protocol where we reconstructed SARS-CoV-2 variants of concern and performed molecular dynamics simulations to study the stability of the ACE2–RBD complex in each variant. We also carried out free energy calculations to compare the binding and biophysical properties of the different SARS-CoV-2 variants when they interact with ACE2. Therefore, we were able to obtain consistent results and uncover new crucial residues that were essential for preserving a balance between maintaining a high affinity for ACE2 and the capacity to evade RBD-targeted antibodies. Our detailed structural analysis showed that SARS-CoV-2 variants of concern show a higher affinity for ACE2 compared to the Wuhan strain. Additionally, residues K417N and E484K/A might play a crucial role in antibody evasion, whereas Q498R and N501Y are specifically mutated to strengthen RBD affinity to ACE2 and, thereby, increase the viral effect of the COVID-19 virus.

[end paper 100]



