## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "SARS-CoV-2 spike protein" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on SARS-CoV-2 spike protein to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on SARS-CoV-2 spike protein using the query "SARS-CoV-2 spike protein variants and mutations" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to SARS-CoV-2 spike protein. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, here are the papers that are directly relevant to the SARS-CoV-2 spike protein and would be beneficial for fine-tuning on the topic of designing nanobody binders for SARS-CoV-2:

```python
{
    "9043892": "Passive Immunotherapy Against SARS-CoV-2: From Plasma-Based Therapy to Single Potent Antibodies in the Race to Stay Ahead of the Variants",
    "8920968": "The spike glycoprotein of SARS-CoV-2: A review of how mutations of spike glycoproteins have driven the emergence of variants with high transmissibility and immune escape",
    "8776496": "Mutations in SARS-CoV-2 variants of concern link to increased spike cleavage and virus transmission",
    "9555207": "The Spike-Stabilizing D614G Mutation Interacts with S1/S2 Cleavage Site Mutations To Promote the Infectious Potential of SARS-CoV-2 Variants",
    "8651292": "Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape",
    "11666439": "Unraveling the impact of SARS-CoV-2 mutations on immunity: insights from innate immune recognition to antibody and T cell responses",
    "8651051": "Bispecific antibodies targeting distinct regions of the spike protein potently neutralize SARS-CoV-2 variants of concern",
    "11609190": "Research progress of spike protein mutation of SARS-CoV-2 mutant strain and antibody development",
    "11668528": "Accurate predictions of SARS-CoV-2 infectivity from comprehensive analysis",
    "10238279": "Changes within the P681 residue of spike dictate cell fusion and syncytia formation of Delta and Omicron variants of SARS-CoV-2 with no effects on neutralization or infectivity",
    "10265947": "Identification of broad, potent antibodies to functionally constrained regions of SARS-CoV-2 spike following a breakthrough infection",
    "7891930": "The Spike D614G mutation increases SARS-CoV-2 infection of multiple human cell types"
}
```

These papers focus on the spike protein's role in immune evasion, transmission, and the development of therapeutic antibodies, which are crucial for designing effective nanobody binders.

## User

Tool Output:

Here are the top 100 articles on PubMed Central for the query "SARS-CoV-2 spike protein variants and mutations":

[begin paper 1]

PMCID = 9043892

Title = Passive Immunotherapy Against SARS-CoV-2: From Plasma-Based Therapy to Single Potent Antibodies in the Race to Stay Ahead of the Variants

The COVID-19 pandemic is now approaching 2 years old, with more than 440 million people infected and nearly six million dead worldwide, making it the most significant pandemic since the 1918 influenza pandemic. The severity and significance of SARS-CoV-2 was recognized immediately upon discovery, leading to innumerable companies and institutes designing and generating vaccines and therapeutic antibodies literally as soon as recombinant SARS-CoV-2 spike protein sequence was available. Within months of the pandemic start, several antibodies had been generated, tested, and moved into clinical trials, including Eli Lilly’s bamlanivimab and etesevimab, Regeneron’s mixture of imdevimab and casirivimab, Vir’s sotrovimab, Celltrion’s regdanvimab, and Lilly’s bebtelovimab. These antibodies all have now received at least Emergency Use Authorizations (EUAs) and some have received full approval in select countries. To date, more than three dozen antibodies or antibody combinations have been forwarded into clinical trials. These antibodies to SARS-CoV-2 all target the receptor-binding domain (RBD), with some blocking the ability of the RBD to bind human ACE2, while others bind core regions of the RBD to modulate spike stability or ability to fuse to host cell membranes. While these antibodies were being discovered and developed, new variants of SARS-CoV-2 have cropped up in real time, altering the antibody landscape on a moving basis. Over the past year, the search has widened to find antibodies capable of neutralizing the wide array of variants that have arisen, including Alpha, Beta, Gamma, Delta, and Omicron. The recent rise and dominance of the Omicron family of variants, including the rather disparate BA.1 and BA.2 variants, demonstrate the need to continue to find new approaches to neutralize the rapidly evolving SARS-CoV-2 virus. This review highlights both convalescent plasma- and polyclonal antibody-based approaches as well as the top approximately 50 antibodies to SARS-CoV-2, their epitopes, their ability to bind to SARS-CoV-2 variants, and how they are delivered. New approaches to antibody constructs, including single domain antibodies, bispecific antibodies, IgA- and IgM-based antibodies, and modified ACE2-Fc fusion proteins, are also described. Finally, antibodies being developed for palliative care of COVID-19 disease, including the ramifications of cytokine release syndrome (CRS) and acute respiratory distress syndrome (ARDS), are described.

The online version contains supplementary material available at 10.1007/s40259-022-00529-7.

[end paper 1]

[begin paper 2]

PMCID = 10222255

Title = Quantitative Mutation Analysis of Genes and Proteins of Major SARS-CoV-2 Variants of Concern and Interest

Of various SARS-CoV-2 variants, some have drawn special concern or interest because of their heightened disease threat. The mutability of individual SARS-CoV-2 genes/proteins presumably varies. The present study quantified gene/protein mutations in 13 major SARS-CoV-2 variants of concern/interest, and analyzed viral protein antigenicity using bioinformatics. The results from 187 carefully perused genome clones showed significantly higher mean percent mutations in the spike, ORF8, nucleocapsid, and NSP6 than in other viral proteins. The ORF8 and spike proteins also tolerated higher maximal percent mutations. The omicron variant presented more percent mutations in the NSP6 and structural proteins, whereas the delta featured more in the ORF7a. Omicron subvariant BA.2 exhibited more mutations in ORF6, and omicron BA.4 had more in NSP1, ORF6, and ORF7b, relative to omicron BA.1. Delta subvariants AY.4 and AY.5 bore more mutations in ORF7b and ORF8 than delta B.1.617.2. Predicted antigen ratios of SARS-CoV-2 proteins significantly vary (range: 38–88%). To overcome SARS-CoV-2 immune evasion, the relatively conserved, potentially immunogenic NSP4, NSP13, NSP14, membrane, and ORF3a viral proteins may serve as more suitable targets for molecular vaccines or therapeutics than the mutation-prone NSP6, spike, ORF8, or nucleocapsid protein. Further investigation into distinct mutations of the variants/subvariants may help understand SARS-CoV-2 pathogenesis.

[end paper 2]

[begin paper 3]

PMCID = 8920968

Title = The spike glycoprotein of SARS-CoV-2: A review of how mutations of spike glycoproteins have driven the emergence of variants with high transmissibility and immune escape

Late in 2019, SARS-CoV-2 (severe acute respiratory syndrome coronavirus-2) emerged, causing an unknown type of pneumonia today called coronaviruses disease 2019 (COVID-19). COVID-19 is still an ongoing global outbreak that has claimed and threatened many lives worldwide. Along with the fastest vaccine developed in history to fight SARS-CoV-2 came a critical problem, SARS-CoV-2. These new variants are a result of the accumulation of mutations in the sequence and structure of spike (S) glycoprotein, which is by far the most critical protein for SARS-CoV-2 to recognize cells and escape the immune system, in addition to playing a role in SARS-CoV-2 infection, pathogenicity, transmission, and evolution. In this review, we discuss mutation of S protein and how these mutations have led to new variants that are usually more transmissible and can thus mitigate the immunity produced by vaccination. Here, analysis of S protein sequences and structures from variants point out the mutations among them, how they emerge, and the behavior of S protein from each variant. This review brings details in an understandable way about how the variants of SARS-CoV-2 are a result of mutations in S protein, making them more transmissible and even more aggressive than their relatives.

[end paper 3]

[begin paper 4]

PMCID = 8776496

Title = Mutations in SARS-CoV-2 variants of concern link to increased spike cleavage and virus transmission

SARS-CoV-2 lineages have diverged into highly prevalent variants termed “variants of concern” (VOCs). Here, we characterized emerging SARS-CoV-2 spike polymorphisms in vitro and in vivo to understand their impact on transmissibility and virus pathogenicity and fitness. We demonstrate that the substitution S:655Y, represented in the gamma and omicron VOCs, enhances viral replication and spike protein cleavage. The S:655Y substitution was transmitted more efficiently than its ancestor S:655H in the hamster infection model and was able to outcompete S:655H in the hamster model and in a human primary airway system. Finally, we analyzed a set of emerging SARS-CoV-2 variants to investigate how different sets of mutations may impact spike processing. All VOCs tested exhibited increased spike cleavage and fusogenic capacity. Taken together, our study demonstrates that the spike mutations present in VOCs that become epidemiologically prevalent in humans are linked to an increase in spike processing and virus transmission.

Escalera et al. show that spike mutation H655Y, which is present in SARS-CoV-2 variants gamma and omicron, enhances spike protein cleavage, cell-cell fusion, and transmission in the hamster model. Additionally, SARS-CoV-2 variants of concern are shown to have independently acquired mutations associated with a gain in spike cleavage and syncytia formation.

[end paper 4]

[begin paper 5]

PMCID = 9555207

Title = The Spike-Stabilizing D614G Mutation Interacts with S1/S2 Cleavage Site Mutations To Promote the Infectious Potential of SARS-CoV-2 Variants

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) remained genetically stable during the first 3 months of the pandemic, before acquiring a D614G spike mutation that rapidly spread worldwide and then generating successive waves of viral variants with increasingly high transmissibility. We set out to evaluate possible epistatic interactions between the early-occurring D614G mutation and the more recently emerged cleavage site mutations present in spike of the Alpha, Delta, and Omicron variants of concern. The P681H/R mutations at the S1/S2 cleavage site increased spike processing and fusogenicity but limited its incorporation into pseudoviruses. In addition, the higher cleavage rate led to higher shedding of the spike S1 subunit, resulting in a lower infectivity of the P681H/R-carrying pseudoviruses compared to those expressing the Wuhan wild-type spike. The D614G mutation increased spike expression at the cell surface and limited S1 shedding from pseudovirions. As a consequence, the D614G mutation preferentially increased the infectivity of P681H/R-carrying pseudoviruses. This enhancement was more marked in cells where the endosomal route predominated, suggesting that more stable spikes could better withstand the endosomal environment. Taken together, these findings suggest that the D614G mutation stabilized S1/S2 association and enabled the selection of mutations that increased S1/S2 cleavage, leading to the emergence of SARS-CoV-2 variants expressing highly fusogenic spikes.

IMPORTANCE The first SARS-CoV-2 variant that spread worldwide in early 2020 carried a D614G mutation in the viral spike, making this protein more stable in its cleaved form at the surface of virions. The Alpha and Delta variants, which spread in late 2020 and early 2021, respectively, proved increasingly transmissible and pathogenic compared to the original strain. Interestingly, Alpha and Delta both carried the mutations P681H/R in a cleavage site that made the spike more cleaved and more efficient at mediating viral fusion. We show here that variants with increased spike cleavage due to P681H/R were even more dependent on the stabilizing effect of the D614G mutation, which limited the shedding of cleaved S1 subunits from viral particles. These findings suggest that the worldwide spread of the D614G mutation was a prerequisite for the emergence of more pathogenic SARS-CoV-2 variants with highly fusogenic spikes.

[end paper 5]

[begin paper 6]

PMCID = 10986611

Title = Vero cell-adapted SARS-CoV-2 strain shows increased viral growth through furin-mediated efficient spike cleavage

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) utilizes several host proteases to cleave the spike (S) protein to enter host cells. SARS-CoV-2 S protein is cleaved into S1 and S2 subunits by furin, which is closely involved in the pathogenicity of SARS-CoV-2. However, the effects of the modulated protease cleavage activity due to S protein mutations on viral replication and pathogenesis remain unclear. Herein, we serially passaged two SARS-CoV-2 strains in Vero cells and characterized the cell-adapted SARS-CoV-2 strains in vitro and in vivo. The adapted strains showed high viral growth, effective S1/S2 cleavage of the S protein, and low pathogenicity compared with the wild-type strain. Furthermore, the viral growth and S1/S2 cleavage were enhanced by the combination of the Δ68–76 and H655Y mutations using recombinant SARS-CoV-2 strains generated by the circular polymerase extension reaction. The recombinant SARS-CoV-2 strain, which contained the mutation of the adapted strain, showed increased susceptibility to the furin inhibitor, suggesting that the adapted SARS-CoV-2 strain utilized furin more effectively than the wild-type strain. Pathogenicity was attenuated by infection with effectively cleaved recombinant SARS-CoV-2 strains, suggesting that the excessive cleavage of the S proteins decreases virulence. Finally, the high-growth-adapted SARS-CoV-2 strain could be used as the seed for a low-cost inactivated vaccine; immunization with this vaccine can effectively protect the host from SARS-CoV-2 variants. Our findings provide novel insights into the growth and pathogenicity of SARS-CoV-2 in the evolution of cell-cell transmission.

The efficacy of the S protein cleavage generally differs among the SARS-CoV-2 variants, resulting in distinct viral characteristics. The relationship between a mutation and the entry of SARS-CoV-2 into host cells remains unclear. In this study, we analyzed the sequence of high-growth Vero cell-adapted SARS-CoV-2 and factors determining the enhancement of the growth of the adapted virus and confirmed the characteristics of the adapted strain by analyzing the recombinant SARS-CoV-2 strain. We successfully identified mutations Δ68-76 and H655Y, which enhance viral growth and the S protein cleavage by furin. Using recombinant viruses enabled us to conduct a virus challenge experiment in vivo. The pathogenicity of SARS-CoV-2 introduced with the mutations Δ68-76, H655Y, P812L, and Q853L was attenuated in hamsters, indicating the possibility of the attenuation of excessive cleaved SARS-CoV-2. These findings provide novel insights into the infectivity and pathogenesis of SARS-CoV-2 strains, thereby significantly contributing to the field of virology.

[end paper 6]

[begin paper 7]

PMCID = 8732965

Title = Modeling SARS-CoV-2 spike/ACE2 protein–protein interactions for predicting the binding affinity of new spike variants for ACE2, and novel ACE2 structurally related human protein targets, for COVID-19 handling in the 3PM context

The rapid spread of new SARS-CoV-2 variants has highlighted the crucial role played in the infection by mutations occurring at the SARS-CoV-2 spike receptor binding domain (RBD) in the interactions with the human ACE2 receptor. In this context, it urgently needs to develop new rapid tools for quickly predicting the affinity of ACE2 for the SARS-CoV-2 spike RBD protein variants to be used with the ongoing SARS-CoV-2 genomic sequencing activities in the clinics, aiming to gain clues about the transmissibility and virulence of new variants, to prevent new outbreaks and to quickly estimate the severity of the disease in the context of the 3PM.

In our study, we used a computational pipeline for calculating the interaction energies at the SARS-CoV-2 spike RBD/ACE2 protein–protein interface for a selected group of characterized infectious variants of concern/interest (VoC/VoI). By using our pipeline, we built 3D comparative models of the SARS-CoV-2 spike RBD/ACE2 protein complexes for the VoC B.1.1.7-United Kingdom (carrying the mutations of concern/interest N501Y, S494P, E484K at the RBD), P.1-Japan/Brazil (RBD mutations: K417T, E484K, N501Y), B.1.351-South Africa (RBD mutations: K417N, E484K, N501Y), B.1.427/B.1.429-California (RBD mutations: L452R), the B.1.141 (RBD mutations: N439K), and the recent B.1.617.1-India (RBD mutations: L452R; E484Q) and the B.1.620 (RBD mutations: S477N; E484K). Then, we used the obtained 3D comparative models of the SARS-CoV-2 spike RBD/ACE2 protein complexes for predicting the interaction energies at the protein–protein interface.

Along SARS-CoV-2 mutation database screening and mutation localization analysis, it was ascertained that the most dangerous mutations at VoC/VoI spike proteins are located mainly at three regions of the SARS-CoV-2 spike “boat-shaped” receptor binding motif, on the RBD domain. Notably, the P.1 Japan/Brazil variant present three mutations, K417T, E484K, N501Y, located along the entire receptor binding motif, which apparently determines the highest interaction energy at the SARS-CoV-2 spike RBD/ACE2 protein–protein interface, among those calculated. Conversely, it was also observed that the replacement of a single acidic/hydrophilic residue with a basic residue (E484K or N439K) at the “stern” or “bow” regions, of the boat-shaped receptor binding motif on the RBD, appears to determine an interaction energy with ACE2 receptor higher than that observed with single mutations occurring at the “hull” region or with other multiple mutants. In addition, our pipeline allowed searching for ACE2 structurally related proteins, i.e., THOP1 and NLN, which deserve to be investigated for their possible involvement in interactions with the SARS-CoV-2 spike protein, in those tissues showing a low expression of ACE2, or as a novel receptor for future spike variants. A freely available web-tool for the in silico calculation of the interaction energy at the SARS-CoV-2 spike RBD/ACE2 protein–protein interface, starting from the sequences of the investigated spike and/or ACE2 variants, was made available for the scientific community at: https://www.mitoairm.it/covid19affinities.

In the context of the PPPM/3PM, the employment of the described pipeline through the provided webservice, together with the ongoing SARS-CoV-2 genomic sequencing, would help to predict the transmissibility of new variants sequenced from future patients, depending on SARS-CoV-2 genomic sequencing activities and on the specific amino acid replacement and/or on its location on the SARS-CoV-2 spike RBD, to put in play all the possible counteractions for preventing the most deleterious scenarios of new outbreaks, taking into consideration that a greater transmissibility has not to be necessarily related to a more severe manifestation of the disease.

The online version contains supplementary material available at 10.1007/s13167-021-00267-w.

[end paper 7]

[begin paper 8]

PMCID = 10535197

Title = The Effect of Select SARS-CoV-2 N-Linked Glycan and Variant of Concern Spike Protein Mutations on C-Type Lectin-Receptor-Mediated Infection

The SARS-CoV-2 virion has shown remarkable resilience, capable of mutating to escape immune detection and re-establishing infectious capabilities despite new vaccine rollouts. Therefore, there is a critical need to identify relatively immutable epitopes on the SARS-CoV-2 virion that are resistant to future mutations the virus may accumulate. While hACE2 has been identified as the receptor that mediates SARS-CoV-2 susceptibility, it is only modestly expressed in lung tissue. C-type lectin receptors like DC-SIGN can act as attachment sites to enhance SARS-CoV-2 infection of cells with moderate or low hACE2 expression. We developed an easy-to-implement assay system that allows for the testing of SARS-CoV-2 trans-infection. Using our assay, we assessed how SARS-CoV-2 Spike S1-domain glycans and spike proteins from different strains affected the ability of pseudotyped lentivirions to undergo DC-SIGN-mediated trans-infection. Through our experiments with seven glycan point mutants, two glycan cluster mutants and four strains of SARS-CoV-2 spike, we found that glycans N17 and N122 appear to have significant roles in maintaining COVID-19′s infectious capabilities. We further found that the virus cannot retain infectivity upon the loss of multiple glycosylation sites, and that Omicron BA.2 pseudovirions may have an increased ability to bind to other non-lectin receptor proteins on the surface of cells. Taken together, our work opens the door to the development of new therapeutics that can target overlooked epitopes of the SARS-CoV-2 virion to prevent C-type lectin-receptor-mediated trans-infection in lung tissue.

[end paper 8]

[begin paper 9]

PMCID = 11722104

Title = Ongoing evolution of SARS-CoV-2 drives escape from mRNA vaccine-induced humoral immunity

With the onset of the COVID-19 pandemic 4 years ago, viral sequencing continues to document numerous individual mutations in the viral spike protein across many variants. To determine the ability of vaccine-mediated humoral immunity to combat continued SARS-CoV-2 evolution, we construct a comprehensive panel of pseudoviruses harboring each individual mutation spanning 4 years of the pandemic to understand the fitness cost and resistance benefits of each. These efforts identify numerous mutations that escape from vaccine-induced humoral immunity. Across 50 variants and 131 mutants we construct, we observe progressive loss of neutralization across variants, irrespective of vaccine doses, as well as increasing infectivity and ACE2 binding. Importantly, the recent XBB.1.5 booster significantly increases titers against most variants but not JN.1, KP.2, or KP.3. These findings demonstrate that variants continue to evade updated mRNA vaccines, highlighting the need for different approaches to control SARS-CoV-2 transmission.

Primary vaccination reveals regions of vulnerability within the mutational landscape

mRNA booster vaccination significantly enhances neutralization of SARS-CoV-2 mutants

Individual omicron mutants reveal gaps in mRNA-boosted humoral immunity

Variant-specific vaccines markedly improve neutralization breadth

Roederer et al. examine how the SARS-CoV-2 virus evolves over time and how well updated vaccines protect against variants. Variants of the virus are becoming harder for vaccine-mediated humoral immunity to neutralize, even with updated boosters, suggesting alternate vaccination methods may be necessary to combat SARS-CoV-2 evolution.

[end paper 9]

[begin paper 10]

PMCID = 11192331

Title = Mapping the Evolutionary Space of SARS-CoV-2 Variants to Anticipate Emergence of Subvariants Resistant to COVID-19 Therapeutics

New sublineages of SARS-CoV-2 variants-of-concern (VOCs) continuously emerge with mutations in the spike glycoprotein. In most cases, the sublineage-defining mutations vary between the VOCs. It is unclear whether these differences reflect lineage-specific likelihoods for mutations at each spike position or the stochastic nature of their appearance. Here we show that SARS-CoV-2 lineages have distinct evolutionary spaces (a probabilistic definition of the sequence states that can be occupied by expanding virus subpopulations). This space can be accurately inferred from the patterns of amino acid variability at the whole-protein level. Robust networks of co-variable sites identify the highest-likelihood mutations in new VOC sublineages and predict remarkably well the emergence of subvariants with resistance mutations to COVID-19 therapeutics. Our studies reveal the contribution of low frequency variant patterns at heterologous sites across the protein to accurate prediction of the changes at each position of interest.

We describe a systematic approach to identify clues that can predict the mutational profiles of SARS-CoV-2 subvariants at early stages after emergence of their parental lineages. We found that the likelihood for mutations at each position of spike is lineage-specific, and is estimated well by the mutational patterns at all positions of the protein. As examples, we apply the model to forecast emergence of resistance to the antibody therapeutics Bebtelovimab and Cilgavimab, which appeared in sublineages of VOCs BA.5 and BA.4, respectively. Emergence of these subvariants required changes in treatment protocols of COVID-19 patients. In both cases, the escape mutations were predicted well by the mutational profiles at all positions of spike at early time points after emergence of the parental VOCs. Accurate estimations of the imminent changes in SARS-CoV-2 lineages can contribute to design and selection of therapeutics that maintain their efficacy against future forms of this virus.

[end paper 10]

[begin paper 11]

PMCID = 9511875

Title = Non-uniform aspects of the SARS-CoV-2 intraspecies evolution reopen question of its origin

Several hypotheses have been presented on the origin of severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) from its identification as the agent causing the current coronavirus disease 19 (COVID-19) pandemic. So far, no solid evidence has been found to support any hypothesis on the origin of this virus, and the issue continue to resurface over and over again. Here we have unfolded a pattern of distribution of several mutations in the SARS-CoV-2 proteins in 24 geo-locations across different continents. The results showed an evenly uneven distribution of the unique protein variants, distinct mutations, unique frequency of common conserved residues, and mutational residues across these 24 geo-locations. Furthermore, ample mutations were identified in the evolutionarily conserved invariant regions in the SARS-CoV-2 proteins across almost all geo-locations studied. This pattern of mutations potentially breaches the law of evolutionary conserved functional units of the beta-coronavirus genus. These mutations may lead to several novel SARS-CoV-2 variants with a high degree of transmissibility and virulence. A thorough investigation on the origin and characteristics of SARS-CoV-2 needs to be conducted in the interest of science and for the preparation of meeting the challenges of potential future pandemics.

[end paper 11]

[begin paper 12]

PMCID = 8651292

Title = Highly synergistic combinations of nanobodies that target SARS-CoV-2 and are resistant to escape

The emergence of SARS-CoV-2 variants threatens current vaccines and therapeutic antibodies and urgently demands powerful new therapeutics that can resist viral escape. We therefore generated a large nanobody repertoire to saturate the distinct and highly conserved available epitope space of SARS-CoV-2 spike, including the S1 receptor binding domain, N-terminal domain, and the S2 subunit, to identify new nanobody binding sites that may reflect novel mechanisms of viral neutralization. Structural mapping and functional assays show that indeed these highly stable monovalent nanobodies potently inhibit SARS-CoV-2 infection, display numerous neutralization mechanisms, are effective against emerging variants of concern, and are resistant to mutational escape. Rational combinations of these nanobodies that bind to distinct sites within and between spike subunits exhibit extraordinary synergy and suggest multiple tailored therapeutic and prophylactic strategies.

[end paper 12]

[begin paper 13]

PMCID = 8447121

Title = The biological and clinical significance of emerging SARS-CoV-2 variants

The past several months have witnessed the emergence of SARS-CoV-2 variants with novel spike protein mutations that are influencing the epidemiological and clinical aspects of the COVID-19 pandemic. These variants can increase rates of virus transmission and/or increase the risk of reinfection and reduce the protection afforded by neutralizing monoclonal antibodies and vaccination. These variants can therefore enable SARS-CoV-2 to continue its spread in the face of rising population immunity while maintaining or increasing its replication fitness. The identification of four rapidly expanding virus lineages since December 2020, designated variants of concern, has ushered in a new stage of the pandemic. The four variants of concern, the Alpha variant (originally identified in the UK), the Beta variant (originally identified in South Africa), the Gamma variant (originally identified in Brazil) and the Delta variant (originally identified in India), share several mutations with one another as well as with an increasing number of other recently identified SARS-CoV-2 variants. Collectively, these SARS-CoV-2 variants complicate the COVID-19 research agenda and necessitate additional avenues of laboratory, epidemiological and clinical research.

In this Review, the authors describe our latest understanding of the emergence and properties of SARS-CoV-2 genetic variants, particularly those designated as WHO (World Health Organization) ‘variants of concern’. They focus on the consequences of these variants for antibody-mediated virus neutralization, with important implications for reinfection risk and for vaccine effectiveness.

The past several months have witnessed the emergence of four SARS-CoV-2 variants of concern (Alpha, Beta, Gamma and Delta) associated with increased transmissibility, increased risk of reinfection and/or reduced vaccine efficacy.

Many additional SARS-CoV-2 variants sharing mutations and biological features with these variants are also increasingly being identified.

The increasing number of SARS-CoV-2 variants share a repertoire of mutations that is enabling the virus to spread despite rising population immunity while maintaining or increasing its replication fitness.

Whereas most emerging mutations reduce the protective effects of neutralizing antibodies generated by infection and vaccination, several recently identified mutations appear to antagonize the innate immune response to initial infection.

The emergence of SARS-CoV-2 variants requires an expanded research agenda to improve our understanding of emerging SARS-CoV-2 mutations and the correlates of protective immunity against variants with these mutations.

[end paper 13]

[begin paper 14]

PMCID = 11666439

Title = Unraveling the impact of SARS-CoV-2 mutations on immunity: insights from innate immune recognition to antibody and T cell responses

Throughout the COVID-19 pandemic, the emergence of new viral variants has challenged public health efforts, often evading antibody responses generated by infections and vaccinations. This immune escape has led to waves of breakthrough infections, raising questions about the efficacy and durability of immune protection. Here we focus on the impact of SARS-CoV-2 Delta and Omicron spike mutations on ACE-2 receptor binding, protein stability, and immune response evasion. Delta and Omicron variants had 3–5 times higher binding affinities to ACE-2 than the ancestral strain (KDwt = 23.4 nM, KDDelta = 8.08 nM, KDBA.1 = 4.77 nM, KDBA.2 = 4.47 nM). The pattern recognition molecule mannose-binding lectin (MBL) has been shown to recognize the spike protein. Here we found that MBL binding remained largely unchanged across the variants, even after introducing mutations at single glycan sites. Although MBL binding decreased post-vaccination, it increased by 2.6-fold upon IgG depletion, suggesting a compensatory or redundant role in immune recognition. Notably, we identified two glycan sites (N717 and N801) as potentially essential for the structural integrity of the spike protein. We also evaluated the antibody and T cell responses. Neutralization by serum immunoglobulins was predominantly mediated by IgG rather than IgA and was markedly impaired against the Delta (5.8-fold decrease) and Omicron variants BA.1 (17.4-fold) and BA.2 (14.2-fold). T cell responses, initially conserved, waned rapidly within 3 months post-Omicron infection. Our data suggests that immune imprinting may have hindered antibody and T cell responses toward the variants. Overall, despite decreased antibody neutralization, MBL recognition and T cell responses were generally unaffected by the variants. These findings extend our understanding of the complex interplay between viral adaptation and immune response, underscoring the importance of considering MBL interactions, immune imprinting, and viral evolution dynamics in developing new vaccine and treatment strategies.

[end paper 14]

[begin paper 15]

PMCID = 10114651

Title = Quantitative profiling of N-glycosylation of SARS-CoV-2 spike protein variants

With the global spread of the corona virus disease-2019 pandemic, new spike variants of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) continuously emerge due to increased possibility of virus adaptive amino acid mutations. However, the N-glycosylation profiles of different spike variants are yet to be explored extensively, although the spike protein is heavily glycosylated and surface glycans are well-established to play key roles in viral infection and immune response. Here, we investigated quantitatively the N-glycosylation profiles of seven major emerging spike variants including Original, Alpha (B.1.1.7), Beta (B.1.351), Gamma (P.1), Kappa (B.1.671.1), Delta (B.1.671.2), and Omicron (B.1.1.529). The aim was to understand the changing pattern of N-glycan profiles in SARS-CoV-2 evolution in addition to the widely studied amino acid mutations. Different spike variants exhibit substantial variations in the relative abundance of different glycan peaks and subclasses, although no specific glycan species are exclusively present in or absent from any specific variant. Cluster analysis shows that the N-glycosylation profiles may hold the potential for SARS-CoV-2 spike variants classification. Alpha and Beta variants exhibit the closest similarity to the Original, and the Delta variant displays substantial similarity to Gamma and Kappa variants, while the Omicron variant is significantly different from its counterparts. We demonstrated that there is a quantifiable difference in N-glycosylation profiles among different spike variants. The current study and observations herein provide a valuable framework for quantitative N-glycosylation profiling of new emerging viral variants and give us a more comprehensive picture of COVID-19 evolution.

[end paper 15]

[begin paper 16]

PMCID = 11254757

Title = Spike deep mutational scanning helps predict success of SARS-CoV-2 clades

SARS-CoV-2 variants acquire mutations in the spike protein that promote immune evasion and affect other properties that contribute to viral fitness, such as ACE2 receptor binding and cell entry. Knowledge of how mutations affect these spike phenotypes can provide insight into the current and potential future evolution of the virus. Here we use pseudovirus deep mutational scanning to measure how more than 9,000 mutations across the full XBB.1.5 and BA.2 spikes affect ACE2 binding, cell entry or escape from human sera. We find that mutations outside the receptor-binding domain (RBD) have meaningfully affected ACE2 binding during SARS-CoV-2 evolution. We also measure how mutations to the XBB.1.5 spike affect neutralization by serum from individuals who recently had SARS-CoV-2 infections. The strongest serum escape mutations are in the RBD at sites 357, 420, 440, 456 and 473; however, the antigenic effects of these mutations vary across individuals. We also identify strong escape mutations outside the RBD; however, many of them decrease ACE2 binding, suggesting they act by modulating RBD conformation. Notably, the growth rates of human SARS-CoV-2 clades can be explained in substantial part by the measured effects of mutations on spike phenotypes, suggesting our data could enable better prediction of viral evolution.

Pseudovirus-based SARS-CoV-2 spike deep mutational scanning is used to measure how mutations across the spikes affect ACE2 binding, cell entry or escape from human sera, producing data that could enable better prediction of viral evolution.

[end paper 16]

[begin paper 17]

PMCID = 9922669

Title = A pseudovirus system enables deep mutational scanning of the full SARS-CoV-2 spike

A major challenge in understanding SARS-CoV-2 evolution is interpreting the antigenic and functional effects of emerging mutations in the viral spike protein. Here, we describe a deep mutational scanning platform based on non-replicative pseudotyped lentiviruses that directly quantifies how large numbers of spike mutations impact antibody neutralization and pseudovirus infection. We apply this platform to produce libraries of the Omicron BA.1 and Delta spikes. These libraries each contain ∼7,000 distinct amino acid mutations in the context of up to ∼135,000 unique mutation combinations. We use these libraries to map escape mutations from neutralizing antibodies targeting the receptor-binding domain, N-terminal domain, and S2 subunit of spike. Overall, this work establishes a high-throughput and safe approach to measure how ∼105 combinations of mutations affect antibody neutralization and spike-mediated infection. Notably, the platform described here can be extended to the entry proteins of many other viruses.

Method for genotype-phenotype-linked lentivirus pseudotyping

Make full SARS-CoV-2 spike deep mutational scanning libraries

Use libraries to map antibody-escape mutations across spike

Measure effects of mutations in spike on virus cell entry

A high-throughput deep mutational scanning platform that uses non-replicative pseudotyped lentiviruses is able to directly quantify how SARS-CoV-2 spike mutations affect antibody neutralization and spike-mediated infection.

[end paper 17]

[begin paper 18]

PMCID = 10785701

Title = Differences in syncytia formation by SARS-CoV-2 variants modify host chromatin accessibility and cellular senescence via TP53

Coronavirus disease 2019 (COVID-19) remains a significant public health threat due to the ability of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants to evade the immune system and cause breakthrough infections. Although pathogenic coronaviruses such as SARS-CoV-2 and Middle East respiratory syndrome (MERS)-CoV lead to severe respiratory infections, how these viruses affect the chromatin proteomic composition upon infection remains largely uncharacterized. Here, we use our recently developed integrative DNA And Protein Tagging methodology to identify changes in host chromatin accessibility states and chromatin proteomic composition upon infection with pathogenic coronaviruses. SARS-CoV-2 infection induces TP53 stabilization on chromatin, which contributes to its host cytopathic effect. We mapped this TP53 stabilization to the SARS-CoV-2 spike and its propensity to form syncytia, a consequence of cell-cell fusion. Differences in SARS-CoV-2 spike variant-induced syncytia formation modify chromatin accessibility, cellular senescence, and inflammatory cytokine release via TP53. Our findings suggest that differences in syncytia formation alter senescence-associated inflammation, which varies among SARS-CoV-2 variants.

Lee et al. find that spike-mediated cell-cell fusion triggers TP53 stabilization, chromatin accessibility changes, and activation of senescence. The extent of these effects differs based on the fusogenicity of various SARS-CoV-2 variant spike sequences.

[end paper 18]

[begin paper 19]

PMCID = 8853550

Title = Structural and antigenic variations in the spike protein of emerging SARS-CoV-2 variants

The Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) virus is continuously evolving, and this poses a major threat to antibody therapies and currently authorized Coronavirus Disease 2019 (COVID-19) vaccines. It is therefore of utmost importance to investigate and predict the putative mutations on the spike protein that confer immune evasion. Antibodies are key components of the human immune system’s response to SARS-CoV-2, and the spike protein is a prime target of neutralizing antibodies (nAbs) as it plays critical roles in host cell recognition, fusion, and virus entry. The potency of therapeutic antibodies and vaccines partly depends on how readily the virus can escape neutralization. Recent structural and functional studies have mapped the epitope landscape of nAbs on the spike protein, which illustrates the footprints of several nAbs and the site of escape mutations. In this review, we discuss (1) the emerging SARS-CoV-2 variants; (2) the structural basis for antibody-mediated neutralization of SARS-CoV-2 and nAb classification; and (3) identification of the RBD escape mutations for several antibodies that resist antibody binding and neutralization. These escape maps are a valuable tool to predict SARS-CoV-2 fitness, and in conjunction with the structures of the spike-nAb complex, they can be utilized to facilitate the rational design of escape-resistant antibody therapeutics and vaccines.

[end paper 19]

[begin paper 20]

PMCID = 11486371

Title = A genome-wide arrayed CRISPR screen identifies PLSCR1 as an intrinsic barrier to SARS-CoV-2 entry that recent virus variants have evolved to resist

Interferons (IFNs) play a crucial role in the regulation and evolution of host–virus interactions. Here, we conducted a genome-wide arrayed CRISPR knockout screen in the presence and absence of IFN to identify human genes that influence Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) infection. We then performed an integrated analysis of genes interacting with SARS-CoV-2, drawing from a selection of 67 large-scale studies, including our own. We identified 28 genes of high relevance in both human genetic studies of Coronavirus Disease 2019 (COVID-19) patients and functional genetic screens in cell culture, with many related to the IFN pathway. Among these was the IFN-stimulated gene PLSCR1. PLSCR1 did not require IFN induction to restrict SARS-CoV-2 and did not contribute to IFN signaling. Instead, PLSCR1 specifically restricted spike-mediated SARS-CoV-2 entry. The PLSCR1-mediated restriction was alleviated by TMPRSS2 overexpression, suggesting that PLSCR1 primarily restricts the endocytic entry route. In addition, recent SARS-CoV-2 variants have adapted to circumvent the PLSCR1 barrier via currently undetermined mechanisms. Finally, we investigate the functional effects of PLSCR1 variants present in humans and discuss an association between PLSCR1 and severe COVID-19 reported recently.

Interferons (IFNs) have a role in the regulation of virus-host interactions. This study uses genome-wide CRISPR knockout screen data to identify 28 genes that impact SARS-CoV-2 infection, including PLSCR1, which restricts spike-mediated SARS-CoV-2 entry independently of its IFN-related role.

[end paper 20]

[begin paper 21]

PMCID = 11099032

Title = Sequential glycosylations at the multibasic cleavage site of SARS-CoV-2 spike protein regulate viral activity

The multibasic furin cleavage site at the S1/S2 boundary of the spike protein is a hallmark of SARS-CoV-2 and plays a crucial role in viral infection. However, the mechanism underlying furin activation and its regulation remain poorly understood. Here, we show that GalNAc-T3 and T7 jointly initiate clustered O-glycosylations in the furin cleavage site of the SARS-CoV-2 spike protein, which inhibit furin processing, suppress the incorporation of the spike protein into virus-like-particles and affect viral infection. Mechanistic analysis reveals that the assembly of the spike protein into virus-like particles relies on interactions between the furin-cleaved spike protein and the membrane protein of SARS-CoV-2, suggesting a possible mechanism for furin activation. Interestingly, mutations in the spike protein of the alpha and delta variants of the virus confer resistance against glycosylation by GalNAc-T3 and T7. In the omicron variant, additional mutations reverse this resistance, making the spike protein susceptible to glycosylation in vitro and sensitive to GalNAc-T3 and T7 expression in human lung cells. Our findings highlight the role of glycosylation as a defense mechanism employed by host cells against SARS-CoV-2 and shed light on the evolutionary interplay between the host and the virus.

Here, the authors show that GalNAc-T3 and T7 regulate furin cleavage of the SARS-CoV-2 spike protein via O-glycosylation. This influences viral assembly and infection, highlighting glycosylation as a host defense mechanism.

[end paper 21]

[begin paper 22]

PMCID = 11727448

Title = Structural Immunology of SARS‐CoV‐2

The SARS‐CoV‐2 spike (S) protein has undergone significant evolution, enhancing both receptor binding and immune evasion. In this review, we summarize ongoing efforts to develop antibodies targeting various epitopes of the S protein, focusing on their neutralization potency, breadth, and escape mechanisms. Antibodies targeting the receptor‐binding site (RBS) typically exhibit high neutralizing potency but are frequently evaded by mutations in SARS‐CoV‐2 variants. In contrast, antibodies targeting conserved regions, such as the S2 stem helix and fusion peptide, exhibit broader reactivity but generally lower neutralization potency. However, several broadly neutralizing antibodies have demonstrated exceptional efficacy against emerging variants, including the latest omicron subvariants, underscoring the potential of targeting vulnerable sites such as RBS‐A and RBS‐D/CR3022. We also highlight public classes of antibodies targeting different sites on the S protein. The vulnerable sites targeted by public antibodies present opportunities for germline‐targeting vaccine strategies. Overall, developing escape‐resistant, potent antibodies and broadly effective vaccines remains crucial for combating future variants. This review emphasizes the importance of identifying key epitopes and utilizing antibody affinity maturation to inform future therapeutic and vaccine design.

[end paper 22]

[begin paper 23]

PMCID = 9240077

Title = Molecular characteristics, immune evasion, and impact of SARS-CoV-2 variants

The persistent COVID-19 pandemic since 2020 has brought an enormous public health burden to the global society and is accompanied by various evolution of the virus genome. The consistently emerging SARS-CoV-2 variants harboring critical mutations impact the molecular characteristics of viral proteins and display heterogeneous behaviors in immune evasion, transmissibility, and the clinical manifestation during infection, which differ each strain and endow them with distinguished features during populational spread. Several SARS-CoV-2 variants, identified as Variants of Concern (VOC) by the World Health Organization, challenged global efforts on COVID-19 control due to the rapid worldwide spread and enhanced immune evasion from current antibodies and vaccines. Moreover, the recent Omicron variant even exacerbated the global anxiety in the continuous pandemic. Its significant evasion from current medical treatment and disease control even highlights the necessity of combinatory investigation of the mutational pattern and influence of the mutations on viral dynamics against populational immunity, which would greatly facilitate drug and vaccine development and benefit the global public health policymaking. Hence in this review, we summarized the molecular characteristics, immune evasion, and impacts of the SARS-CoV-2 variants and focused on the parallel comparison of different variants in mutational profile, transmissibility and tropism alteration, treatment effectiveness, and clinical manifestations, in order to provide a comprehensive landscape for SARS-CoV-2 variant research.

[end paper 23]

[begin paper 24]

PMCID = 8863680

Title = A Detailed Overview of Immune Escape, Antibody Escape, Partial Vaccine Escape of SARS-CoV-2 and Their Emerging Variants With Escape Mutations

The infective SARS-CoV-2 is more prone to immune escape. Presently, the significant variants of SARS-CoV-2 are emerging in due course of time with substantial mutations, having the immune escape property. Simultaneously, the vaccination drive against this virus is in progress worldwide. However, vaccine evasion has been noted by some of the newly emerging variants. Our review provides an overview of the emerging variants’ immune escape and vaccine escape ability. We have illustrated a broad view related to viral evolution, variants, and immune escape ability. Subsequently, different immune escape approaches of SARS-CoV-2 have been discussed. Different innate immune escape strategies adopted by the SARS-CoV-2 has been discussed like, IFN-I production dysregulation, cytokines related immune escape, immune escape associated with dendritic cell function and macrophages, natural killer cells and neutrophils related immune escape, PRRs associated immune evasion, and NLRP3 inflammasome associated immune evasion. Simultaneously we have discussed the significant mutations related to emerging variants and immune escape, such as mutations in the RBD region (N439K, L452R, E484K, N501Y, K444R) and other parts (D614G, P681R) of the S-glycoprotein. Mutations in other locations such as NSP1, NSP3, NSP6, ORF3, and ORF8 have also been discussed. Finally, we have illustrated the emerging variants’ partial vaccine (BioNTech/Pfizer mRNA/Oxford-AstraZeneca/BBIBP-CorV/ZF2001/Moderna mRNA/Johnson & Johnson vaccine) escape ability. This review will help gain in-depth knowledge related to immune escape, antibody escape, and partial vaccine escape ability of the virus and assist in controlling the current pandemic and prepare for the next.

[end paper 24]

[begin paper 25]

PMCID = 10865805

Title = Surface-modified measles vaccines encoding oligomeric, prefusion-stabilized SARS-CoV-2 spike glycoproteins boost neutralizing antibody responses to Omicron and historical variants, independent of measles seropositivity

Serum titers of SARS-CoV-2-neutralizing antibodies (nAbs) correlate well with protection from symptomatic COVID-19 but decay rapidly in the months following vaccination or infection. In contrast, measles-protective nAb titers are lifelong after measles vaccination, possibly due to persistence of the live-attenuated virus in lymphoid tissues. We, therefore, sought to generate a live recombinant measles vaccine capable of driving high SARS-CoV-2 nAb responses. Since previous clinical testing of a live measles vaccine encoding a SARS-CoV-2 spike glycoprotein resulted in suboptimal anti-spike antibody titers, our new vectors were designed to encode prefusion-stabilized SARS-CoV-2 spike glycoproteins, trimerized via an inserted peptide domain, and displayed on a dodecahedral miniferritin scaffold. Additionally, to circumvent the blunting of vaccine efficacy by preformed anti-measles antibodies, we extensively modified the measles surface glycoproteins. Comprehensive in vivo mouse testing demonstrated the potent induction of high titer nAbs in measles-immune mice and confirmed the significant contributions to overall potency afforded by prefusion stabilization, trimerization, and miniferritin display of the SARS-CoV-2 spike glycoprotein. In animals primed and boosted with a measles virus (MeV) vaccine encoding the ancestral SARS-CoV-2 spike, high-titer nAb responses against ancestral virus strains were only weakly cross-reactive with the Omicron variant. However, in primed animals that were boosted with a MeV vaccine encoding the Omicron BA.1 spike, antibody titers to both ancestral and Omicron strains were robustly elevated, and the passive transfer of serum from these animals protected K18-ACE2 mice from infection and morbidity after exposure to BA.1 and WA1/2020 strains. Our results demonstrate that by engineering the antigen, we can develop potent measles-based vaccine candidates against SARS-CoV-2.

Although the live-attenuated measles virus (MeV) is one of the safest and most efficacious human vaccines, a measles-vectored COVID-19 vaccine candidate expressing the SARS-CoV-2 spike failed to elicit neutralizing antibody (nAb) responses in a phase-1 clinical trial, especially in measles-immune individuals. Here, we constructed a comprehensive panel of MeV-based COVID-19 vaccine candidates using a MeV with extensive modifications on the envelope glycoproteins (MeV-MR). We show that artificial trimerization of the spike is critical for the induction of nAbs and that their magnitude can be significantly augmented when the spike protein is synchronously fused to a dodecahedral scaffold. Furthermore, preexisting measles immunity did not abolish heterologous immunity elicited by our vector. Our results highlight the importance of antigen optimization in the development of spike-based COVID-19 vaccines and therapies.

[end paper 25]

[begin paper 26]

PMCID = 8341613

Title = Signatures in SARS-CoV-2 spike protein conferring escape to neutralizing antibodies

Understanding SARS-CoV-2 evolution and host immunity is critical to control COVID-19 pandemics. At the core is an arms-race between SARS-CoV-2 antibody and angiotensin-converting enzyme 2 (ACE2) recognition, a function of the viral protein spike. Mutations in spike impacting antibody and/or ACE2 binding are appearing worldwide, imposing the need to monitor SARS-CoV2 evolution and dynamics in the population. Determining signatures in SARS-CoV-2 that render the virus resistant to neutralizing antibodies is critical. We engineered 25 spike-pseudotyped lentiviruses containing individual and combined mutations in the spike protein, including all defining mutations in the variants of concern, to identify the effect of single and synergic amino acid substitutions in promoting immune escape. We confirmed that E484K evades antibody neutralization elicited by infection or vaccination, a capacity augmented when complemented by K417N and N501Y mutations. In silico analysis provided an explanation for E484K immune evasion. E484 frequently engages in interactions with antibodies but not with ACE2. Importantly, we identified a novel amino acid of concern, S494, which shares a similar pattern. Using the already circulating mutation S494P, we found that it reduces antibody neutralization of convalescent and post-immunization sera, particularly when combined with E484K and with mutations able to increase binding to ACE2, such as N501Y. Our analysis of synergic mutations provides a signature for hotspots for immune evasion and for targets of therapies, vaccines and diagnostics.

For a SARS-CoV-2 virion to enter a cell, the spike protein displayed at its surface must be recognized by the host ACE2 receptor. Serum neutralizing antibodies, shown to develop upon natural SARS-CoV-2 infection or vaccination, bind spike protein preventing the recognition by ACE2 and, consequently, infection. However, SARS-CoV-2 virus is constantly evolving, and can acquire mutations in spike that render this protein resistant to neutralizing antibodies and make vaccines ineffective. In this paper, we tested how single and a combination of mutations naturally occurring in spike, including in variants of concern, would synergize to affect antibody neutralizing capacity. We then integrated these findings with in silico analyses of amino acids binding to ACE2 and antibodies, and distributed them in a grid as amino acids important for binding to ACE2 or antibodies, or both. We found that changes in amino such as E484 and S494, which frequently interact with antibodies but not with ACE2, promptly evolve immune escape mutants, elicited by infection or vaccination, if the mutation severely alters the binding specificity of the antibody. Our work also suggests that the combination of these mutations with others promoting ACE2 binding, such as N501Y, increases their ability to escape neutralizing-antibody responses.

[end paper 26]

[begin paper 27]

PMCID = 11603013

Title = A substitution at the cytoplasmic tail of the spike protein enhances SARS-CoV-2 infectivity and immunogenicity

Global dissemination of SARS-CoV-2 Omicron sublineages has provided a sufficient opportunity for natural selection, thus enabling beneficial mutations to emerge. Characterisation of these mutations uncovers the underlying machinery responsible for the fast transmission of Omicron variants and guides vaccine development for combating the COVID-19 pandemic.

Through systematic bioinformatics analysis of 496,606 sequences of Omicron variants, we obtained 40 amino acid substitutions that occurred with high frequency in the S protein. Utilising pseudoviruses and a trans-complementation system of SARS-CoV-2, we identified the effect of high-frequency mutations on viral infectivity and elucidated the molecular mechanisms. Finally, we evaluated the impact of a key emerging mutation on the immune protection induced by the SARS-CoV-2 VLP mRNA vaccine in a murine model.

We identified a proline-to-leucine substitution at the 1263rd residue of the Spike protein, and upon investigating the relative frequencies across multiple Omicron sublineages, we found a trend of increasing frequency for P1263L. The substitution significantly enhances the capacity for S-mediated viral entry and improves the immunogenicity of a virus-like particle mRNA vaccine. Mechanistic studies showed that this mutation is located in the FERM binding motif of the cytoplasmic tail and impairs the interaction between the S protein and the Ezrin/Radixin/Moesin proteins. Additionally, this mutation facilitates the incorporation of S proteins into SARS-CoV-2 virions.

This study offers mechanistic insight into the constantly increasing transmissibility of SARS-CoV-2 Omicron variants and provides a meaningful optimisation strategy for vaccine development against SARS-CoV-2.

This study was supported by grants from the  (, ,  and ), the  (, , , , , , , ,  and ),  (, ), the  (),  Project for Prevention and Research on Vector-borne Diseases (), Yunnan Provincial Science and Technology Project at  (). The New Cornerstone Science Foundation through the New Cornerstone Investigator Program, and the Xplorer Prize from Tencent Foundation.

[end paper 27]

[begin paper 28]

PMCID = 9393763

Title = Antibody-mediated immunity to SARS-CoV-2 spike

Despite effective spike-based vaccines and monoclonal antibodies, the SARS-CoV-2 pandemic continues more than two and a half years post-onset. Relentless investigation has outlined a causative dynamic between host-derived antibodies and reciprocal viral subversion. Integration of this paradigm into the architecture of next generation antiviral strategies, predicated on a foundational understanding of the virology and immunology of SARS-CoV-2, will be critical for success. This review aims to serve as a primer on the immunity endowed by antibodies targeting SARS-CoV-2 spike protein through a structural perspective. We begin by introducing the structure and function of spike, polyclonal immunity to SARS-CoV-2 spike, and the emergence of major SARS-CoV-2 variants that evade immunity. The remainder of the article comprises an in-depth dissection of all major epitopes on SARS-CoV-2 spike in molecular detail, with emphasis on the origins, neutralizing potency, mechanisms of action, cross-reactivity, and variant resistance of representative monoclonal antibodies to each epitope.

[end paper 28]

[begin paper 29]

PMCID = 9655607

Title = In Silico Approach for the Evaluation of the Potential Antiviral Activity of Extra Virgin Olive Oil (EVOO) Bioactive Constituents Oleuropein and Oleocanthal on Spike Therapeutic Drug Target of SARS-CoV-2

Since there is an urgent need for novel treatments to combat the current coronavirus disease 2019 (COVID-19) pandemic, in silico molecular docking studies were implemented as an attempt to explore the ability of selected bioactive constituents of extra virgin olive oil (EVOO) to act as potent SARS-CoV-2 (severe acute respiratory syndrome coronavirus 2) antiviral compounds, aiming to explore their ability to interact with SARS-CoV-2 Spike key therapeutic target protein. Our results suggest that EVOO constituents display substantial capacity for binding and interfering with Spike (S) protein, both wild-type and mutant, via the receptor-binding domain (RBD) of Spike, or other binding targets such as angiotensin-converting enzyme 2 (ACE2) or the RBD-ACE2 protein complex, inhibiting the interaction of the virus with host cells. This in silico study provides useful insights for the understanding of the mechanism of action of the studied compounds at a molecular level. From the present study, it could be suggested that the studied active phytochemicals could potentially inhibit the Spike protein, contributing thus to the understanding of the role that they can play in future drug designing and the development of anti-COVID-19 therapeutics.

[end paper 29]

[begin paper 30]

PMCID = 8167834

Title = SARS-CoV-2 variants, spike mutations and immune escape

Although most mutations in the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) genome are expected to be either deleterious and swiftly purged or relatively neutral, a small proportion will affect functional properties and may alter infectivity, disease severity or interactions with host immunity. The emergence of SARS-CoV-2 in late 2019 was followed by a period of relative evolutionary stasis lasting about 11 months. Since late 2020, however, SARS-CoV-2 evolution has been characterized by the emergence of sets of mutations, in the context of ‘variants of concern’, that impact virus characteristics, including transmissibility and antigenicity, probably in response to the changing immune profile of the human population. There is emerging evidence of reduced neutralization of some SARS-CoV-2 variants by postvaccination serum; however, a greater understanding of correlates of protection is required to evaluate how this may impact vaccine effectiveness. Nonetheless, manufacturers are preparing platforms for a possible update of vaccine sequences, and it is crucial that surveillance of genetic and antigenic changes in the global virus population is done alongside experiments to elucidate the phenotypic impacts of mutations. In this Review, we summarize the literature on mutations of the SARS-CoV-2 spike protein, the primary antigen, focusing on their impacts on antigenicity and contextualizing them in the protein structure, and discuss them in the context of observed mutation frequencies in global sequence datasets.

The evolution of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has been characterized by the emergence of mutations and so-called variants of concern that impact virus characteristics, including transmissibility and antigenicity. In this Review, members of the COVID-19 Genomics UK (COG-UK) Consortium and colleagues summarize mutations of the SARS-CoV-2 spike protein, focusing on their impacts on antigenicity and contextualizing them in the protein structure, and discuss them in the context of observed mutation frequencies in global sequence datasets.

[end paper 30]

[begin paper 31]

PMCID = 9350683

Title = Antibody escape and cryptic cross-domain stabilization in the SARS-CoV-2 Omicron spike protein

The worldwide spread of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has led to the repeated emergence of variants of concern. For the Omicron variant, sub-lineages BA.1 and BA.2, respectively, contain 33 and 29 nonsynonymous and indel spike protein mutations. These amino acid substitutions and indels are implicated in increased transmissibility and enhanced immune evasion. By reverting individual spike mutations of BA.1 or BA.2, we characterize the molecular effects of the Omicron spike mutations on expression, ACE2 receptor affinity, and neutralizing antibody recognition. We identified key mutations enabling escape from neutralizing antibodies at a variety of epitopes. Stabilizing mutations in the N-terminal and S2 domains of the spike protein can compensate for destabilizing mutations in the receptor binding domain, enabling the record number of mutations in Omicron. Our results provide a comprehensive account of the mutational effects in the Omicron spike protein and illustrate previously uncharacterized mechanisms of host evasion.

The Omicron BA.1 and BA.2 variants have unprecedented numbers of nonsynonymous and indel spike protein mutations. Javanmardi et al. report the antigenicity, expression, and hACE2 affinity changes due to these mutations, in different protein contexts. This study reveals cryptic cross-domain interactions that enhance antibody escape and stabilize the spike protein.

[end paper 31]

[begin paper 32]

PMCID = 9554601

Title = Potent monoclonal antibodies neutralize Omicron sublineages and other SARS-CoV-2 variants

The emergence and global spread of the SARS-CoV-2 Omicron variants, which carry an unprecedented number of mutations, raise serious concerns due to the reduced efficacy of current vaccines and resistance to therapeutic antibodies. Here, we report the generation and characterization of two potent human monoclonal antibodies, NA8 and NE12, against the receptor-binding domain of the SARS-CoV-2 spike protein. NA8 interacts with a highly conserved region and has a breadth of neutralization with picomolar potency against the Beta variant and the Omicron BA.1 and BA.2 sublineages and nanomolar potency against BA.2.12.1 and BA.4. Combination of NA8 and NE12 retains potent neutralizing activity against the major SARS-CoV-2 variants of concern. Cryo-EM analysis provides the structural basis for the broad and complementary neutralizing activity of these two antibodies. We confirm the in vivo protective and therapeutic efficacies of NA8 and NE12 in the hamster model. These results show that broad and potent human antibodies can overcome the continuous immune escape of evolving SARS-CoV-2 variants.

The emergence of new SARS-CoV-2 variants that escape neutralization can jeopardize the efficacy of vaccines and therapeutic antibodies. Chen et al. report the isolation of potent human antibodies that neutralize emerging variants of concern, including various Omicron sublineages. These antibodies show prophylactic and therapeutic efficacy in the hamster model.

[end paper 32]

[begin paper 33]

PMCID = 10972307

Title = Genomic Surveillance and Mutation Analysis of SARS-CoV-2 Variants among Patients in Saudi Arabia

The genome of severe acute respiratory coronavirus-2 (SARS-CoV-2), the virus responsible for coronavirus disease 2019 (COVID-19), has undergone a rapid evolution, resulting in the emergence of multiple SARS-CoV-2 variants with amino acid changes. This study aimed to sequence the whole genome of SARS-CoV-2 and detect the variants present in specimens from Saudi Arabia. Furthermore, we sought to analyze and characterize the amino acid changes in the various proteins of the identified SARS-CoV-2 variants. A total of 1161 samples from patients diagnosed with COVID-19 in Saudi Arabia, between 1 April 2021 and 31 July 2023, were analyzed. Whole genome sequencing was employed for variant identification and mutation analysis. The statistical analysis was performed using the Statistical Analytical Software SAS, version 9.4, and GraphPad, version 9.0. This study identified twenty-three variants and subvariants of SARS-CoV-2 within the population, with the Omicron BA.1 (21K) variant (37.0%) and the Delta (21J) variant (12%) being the most frequently detected. Notably, the Omicron subvariants exhibited a higher mean mutation rate. Amino acid mutations were observed in twelve proteins. Among these, the spike (S), ORF1a, nucleocapsid (N), and ORF1b proteins showed a higher frequency of amino acid mutations compared to other the viral proteins. The S protein exhibited the highest incidence of amino acid mutations (47.6%). Conversely, the ORF3a, ORF8, ORF7a, ORF6, and ORF7b proteins appeared more conserved, demonstrating the lowest percentage and frequency of amino acid mutations. The investigation of structural protein regions revealed the N-terminal S1 subunit of the S protein to frequently harbor mutations, while the N-terminal domain of the envelope (E) protein displayed the lowest mutation frequency. This study provides insights into the variants and genetic diversity of SARS-CoV-2, underscoring the need for further research to comprehend its genome evolution and the occurrence of mutations. These findings are pertinent to the development of testing approaches, therapeutics, and vaccine strategies.

[end paper 33]

[begin paper 34]

PMCID = 8604694

Title = Mutations of SARS-CoV-2 spike protein: Implications on immune evasion and vaccine-induced immunity

Responsible for more than 4.9 million deaths so far, COVID-19, caused by SARS-CoV-2, is instigating devastating effects on the global health care system whose impacts could be longer for the years to come. Acquiring a comprehensive knowledge of host-virus interaction is critical for designing effective vaccines and/or drugs. Understanding the evolution of the virus and the impact of genetic variability on host immune evasion and vaccine efficacy is helpful to design novel strategies to minimize the effects of the emerging variants of concern (VOC). Most vaccines under development and/or in current use target the spike protein owning to its unique function of host receptor binding, relatively conserved nature, potent immunogenicity in inducing neutralizing antibodies, and being a good target of T cell responses. However, emerging SARS-CoV-2 strains are exhibiting variability on the spike protein which could affect the efficacy of vaccines and antibody-based therapies in addition to enhancing viral immune evasion mechanisms. Currently, the degree to which mutations on the spike protein affect immunity and vaccination, and the ability of the current vaccines to confer protection against the emerging variants attracts much attention. This review discusses the implications of SARS-CoV-2 spike protein mutations on immune evasion and vaccine-induced immunity and forward directions which could contribute to future studies focusing on designing effective vaccines and/or immunotherapies to consider viral evolution. Combining vaccines derived from different regions of the spike protein that boost both the humoral and cellular wings of adaptive immunity could be the best options to cope with the emerging VOC.

[end paper 34]

[begin paper 35]

PMCID = 11324185

Title = The S2 subunit of spike encodes diverse targets for functional antibody responses to SARS-CoV-2

The SARS-CoV-2 virus responsible for the COVID-19 global pandemic has exhibited a striking capacity for viral evolution that drives continued evasion from vaccine and infection-induced immune responses. Mutations in the receptor binding domain of the S1 subunit of the spike glycoprotein have led to considerable escape from antibody responses, reducing the efficacy of vaccines and monoclonal antibody (mAb) therapies. Therefore, there is a need to interrogate more constrained regions of spike, such as the S2 subdomain. Here, we present a collection of S2 mAbs from two SARS-CoV-2 convalescent individuals that target multiple regions in S2, including regions outside of those commonly reported. One of the S2 mAbs, C20.119, which bound to a highly conserved epitope in the fusion peptide, was able to broadly neutralize across SARS-CoV-2 variants, SARS-CoV-1, and closely related zoonotic sarbecoviruses. The majority of the mAbs were non-neutralizing; however, many of them could mediate antibody-dependent cellular cytotoxicity (ADCC) at levels similar to the S1-targeting mAb S309 that was previously authorized for treatment of SARS-CoV-2 infections. Several of the mAbs with ADCC function also bound to spike trimers from other human coronaviruses (HCoVs), such as MERS-CoV and HCoV-HKU1. Our findings suggest S2 mAbs can target diverse epitopes in S2, including functional mAbs with HCoV and sarbecovirus breadth that likely target functionally constrained regions of spike. These mAbs could be developed for potential future pandemics, while also providing insight into ideal epitopes for eliciting a broad HCoV response.

The early successes of vaccines and antibody therapies against SARS-CoV-2, the virus responsible for the COVID-19 global pandemic, leveraged the considerable antibody response to the viral entry protein, spike, after vaccination or infection. These initial interventions were highly effective at protecting from infection and reducing severe disease or death. However, the rapid emergence of new SARS-CoV-2 variants has severely compromised the utility of COVID-19 vaccines and antibody-based treatments. SARS-CoV-2 shows no sign of abating, with the continued rise of new variants with distinct alterations most significantly in regions of the spike protein that elicit most of the anti-viral functional antibody response, which in turn facilitates viral escape from the immune response. These findings suggest a critical need to identify vaccine approaches and therapies that provide the broadest possible antibody responses, focused on regions of spike critical for SARS-CoV-2 infection and, therefore, do not undergo changes that could lead to immune evasion. Our study describes a panel of functional antibodies, from individuals after SARS-CoV-2 infection, that recognize regions of spike that appear conserved across SARS-CoV-2 variants and other closely related viruses, that could guide more effective vaccine design in the face of continued viral evolution.

[end paper 35]

[begin paper 36]

PMCID = 8651051

Title = Bispecific antibodies targeting distinct regions of the spike protein potently neutralize SARS-CoV-2 variants of concern

Bispecific antibodies targeting multiple regions of the SARS-CoV-2 spike protein comparably neutralize variants of concern and wild-type virus.

Monoclonal antibodies (mAbs) have become an essential component of coronavirus disease 2019 (COVID-19) treatment. Most mAbs are administered as a cocktail of multiple antibodies that target different regions of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) virion. Here, Cho et al. further improved upon this strategy by identifying mAbs that, when combined as bispecific antibodies, neutralized SARS-CoV-2 better than cocktails of the parental mAbs. These bispecific antibodies bound to distinct regions of the SARS-CoV-2 spike protein, neutralized variants of concern, including the Delta variant, and conferred protection when administered to hamsters before SARS-CoV-2 infection. Together, these findings suggest that bispecific antibodies merit further consideration as variant-resistant SARS-CoV-2 therapeutics.

The emergence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants of concern threatens the efficacy of existing vaccines and therapeutic antibodies and underscores the need for additional antibody-based tools that potently neutralize variants by targeting multiple sites of the spike protein. We isolated 216 monoclonal antibodies targeting SARS-CoV-2 from plasmablasts and memory B cells collected from patients with coronavirus disease 2019. The three most potent antibodies targeted distinct regions of the receptor binding domain (RBD), and all three neutralized the SARS-CoV-2 Alpha and Beta variants. The crystal structure of the most potent antibody, CV503, revealed that it binds to the ridge region of SARS-CoV-2 RBD, competes with the angiotensin-converting enzyme 2 receptor, and has limited contact with key variant residues K417, E484, and N501. We designed bispecific antibodies by combining nonoverlapping specificities and identified five bispecific antibodies that inhibit SARS-CoV-2 infection at concentrations of less than 1 ng/ml. Through a distinct mode of action, three bispecific antibodies cross-linked adjacent spike proteins using dual N-terminal domain–RBD specificities. One bispecific antibody was greater than 100-fold more potent than a cocktail of its parent monoclonals in vitro and prevented clinical disease in a hamster model at a dose of 2.5 mg/kg. Two bispecific antibodies in our panel comparably neutralized the Alpha, Beta, Gamma, and Delta variants and wild-type virus. Furthermore, a bispecific antibody that neutralized the Beta variant protected hamsters against SARS-CoV-2 expressing the E484K mutation. Thus, bispecific antibodies represent a promising next-generation countermeasure against SARS-CoV-2 variants of concern.

[end paper 36]

[begin paper 37]

PMCID = 8994061

Title = Spike protein of SARS-CoV-2 variants: a brief review and practical implications

The scientific community has been alarmed by the possible immunological evasion, higher infectivity, and severity of disease caused by the newest variants of SARS-CoV-2. The spike protein has an important role in the cellular invasion of viruses and is the target of several vaccines and therapeutic resources, such as monoclonal antibodies. In addition, some of the most relevant mutations in the different variants are on the spike (S) protein gene sequence that leads to structural alterations in the predicted protein, thus causing concern about the protection mediated by vaccines against these new strains. The present review highlights the most recent knowledge about COVID-19 and vaccines, emphasizing the different spike protein structures of SARS-CoV-2 and updating the reader about the emerging viral variants and their classifications, the more common viral mutations described and their distribution in Brazil. It also compiles a table with the most recent knowledge about all of the Omicron spike mutations.

[end paper 37]

[begin paper 38]

PMCID = 8836237

Title = Conformational Flexibility and Local Frustration in the Functional States of the SARS-CoV-2 Spike B.1.1.7 and B.1.351 Variants: Mutation-Induced Allosteric Modulation Mechanism of Functional Dynamics and Protein Stability

Structural and functional studies of the SARS-CoV-2 spike proteins have recently determined distinct functional states of the B.1.1.7 and B.1.351 spike variants, providing a molecular framework for understanding the mechanisms that link the effect of mutations with the enhanced virus infectivity and transmissibility. A detailed dynamic and energetic analysis of these variants was undertaken in the present work to quantify the effects of different mutations on functional conformational changes and stability of the SARS-CoV-2 spike protein. We employed the efficient and accurate coarse-grained (CG) simulations of multiple functional states of the D614G mutant, B.1.1.7 and B.1.351 spike variants to characterize conformational dynamics of the SARS-CoV-2 spike proteins and identify dynamic signatures of the functional regions that regulate transitions between the closed and open forms. By combining molecular simulations with full atomistic reconstruction of the trajectories and the ensemble-based mutational frustration analysis, we characterized how the intrinsic flexibility of specific spike regions can control functional conformational changes required for binding with the host-cell receptor. Using the residue-based mutational scanning of protein stability, we determined protein stability hotspots and identified potential energetic drivers favoring the receptor-accessible open spike states for the B.1.1.7 and B.1.351 spike variants. The results suggested that modulation of the energetic frustration at the inter-protomer interfaces can serve as a mechanism for allosteric couplings between mutational sites and the inter-protomer hinges of functional motions. The proposed mechanism of mutation-induced energetic frustration may result in greater adaptability and the emergence of multiple conformational states in the open form. This study suggested that SARS-CoV-2 B.1.1.7 and B.1.351 variants may leverage the intrinsic plasticity of functional regions in the spike protein for mutation-induced modulation of protein dynamics and allosteric regulation to control binding with the host cell receptor.

[end paper 38]

[begin paper 39]

PMCID = 11609190

Title = Research progress of spike protein mutation of SARS-CoV-2 mutant strain and antibody development

The coronavirus disease 2019 (COVID-19) is a respiratory disease with a very high infectious rate caused by the Severe Acute Respiratory Syndrome Coronavirus-2(SARS-CoV-2). Because SARS-CoV-2 is easy to mutate, the continuous emergence of SARS-CoV-2 variant strains not only enhances the infectivity of the SARS-CoV-2 but also brings great obstacles to the treatment of COVID-19. Neutralizing antibodies have achieved good results in the clinical application of the novel coronavirus pneumonia, which can be used for pre-infection protection and treatment of novel coronavirus patients. This review makes a detailed introduction to the mutation characteristics of SARS-CoV-2, focusing on the molecular mechanism of mutation affecting the infectivity of SARS-CoV-2, and the impact of mutation on monoclonal antibody therapy, providing scientific reference for the prevention of SARS-CoV-2 variant strains and the research and development of antibody drugs.

[end paper 39]

[begin paper 40]

PMCID = 10154619

Title = Evaluating the effect of SARS-CoV-2 spike mutations with a linear doubly robust learner

Driven by various mutations on the viral Spike protein, diverse variants of SARS-CoV-2 have emerged and prevailed repeatedly, significantly prolonging the pandemic. This phenomenon necessitates the identification of key Spike mutations for fitness enhancement. To address the need, this manuscript formulates a well-defined framework of causal inference methods for evaluating and identifying key Spike mutations to the viral fitness of SARS-CoV-2. In the context of large-scale genomes of SARS-CoV-2, it estimates the statistical contribution of mutations to viral fitness across lineages and therefore identifies important mutations. Further, identified key mutations are validated by computational methods to possess functional effects, including Spike stability, receptor-binding affinity, and potential for immune escape. Based on the effect score of each mutation, individual key fitness-enhancing mutations such as D614G and T478K are identified and studied. From individual mutations to protein domains, this paper recognizes key protein regions on the Spike protein, including the receptor-binding domain and the N-terminal domain. This research even makes further efforts to investigate viral fitness via mutational effect scores, allowing us to compute the fitness score of different SARS-CoV-2 strains and predict their transmission capacity based solely on their viral sequence. This prediction of viral fitness has been validated using BA.2.12.1, which is not used for regression training but well fits the prediction. To the best of our knowledge, this is the first research to apply causal inference models to mutational analysis on large-scale genomes of SARS-CoV-2. Our findings produce innovative and systematic insights into SARS-CoV-2 and promotes functional studies of its key mutations, serving as reliable guidance about mutations of interest.

[end paper 40]

[begin paper 41]

PMCID = 9030214

Title = Functional Analysis of Spike from SARS-CoV-2 Variants Reveals the Role of Distinct Mutations in Neutralization Potential and Viral Infectivity

Enhanced viral transmission and escape from vaccine–elicited neutralizing antibodies drive worldwide spread of SARS-CoV-2 variants and promote disease progression. However, the impact of specific spike mutations that are carried by different viral variants on viral infectivity and neutralization sensitivity has not been completely defined. Here, we use pseudoviruses to assess the contribution of spike mutations within the Receptor Binding Domain (RBD) and the Furin Cleavage Site (FCS), and appear in circulating viral variants, on viral infectivity and neutralization potential against sera that was drawn from fully vaccinated individuals. Our functional analysis demonstrates that single, P681H, P681R or A701V–FCS mutations do not play a role in viral infectivity and neutralization potential. However, when in conjunction with the RBD–N501Y mutation, viral infectivity is enhanced. Similarly, combining the E484K–RBD mutation to the spike that carries FCS mutations reduces neutralization sensitivity with no effects on viral infectivity. Employing a similar approach onto the spike from Delta or Lota SARS-CoV-2 variants further reveals that specific RBD mutations affect neutralization sensitivity or viral infectivity differently. Our results validate the efficacy of the Pfizer third dose vaccine against Delta and Lota SARS-CoV-2 variants, and outline the significance of distinct RBD mutations in promoting viral infectivity and neutralization sensitivity to post–vaccination sera.

[end paper 41]

[begin paper 42]

PMCID = 11668528

Title = Accurate predictions of SARS-CoV-2 infectivity from comprehensive analysis

An unprecedented amount of SARS-CoV-2 data has been accumulated compared with previous infectious diseases, enabling insights into its evolutionary process and more thorough analyses. This study investigates SARS-CoV-2 features as it evolved to evaluate its infectivity. We examined viral sequences and identified the polarity of amino acids in the receptor binding motif (RBM) region. We detected an increased frequency of amino acid substitutions to lysine (K) and arginine (R) in variants of concern (VOCs). As the virus evolved to Omicron, commonly occurring mutations became fixed components of the new viral sequence. Furthermore, at specific positions of VOCs, only one type of amino acid substitution and a notable absence of mutations at D467 were detected. We found that the binding affinity of SARS-CoV-2 lineages to the ACE2 receptor was impacted by amino acid substitutions. Based on our discoveries, we developed APESS, an evaluation model evaluating infectivity from biochemical and mutational properties. In silico evaluation using real-world sequences and in vitro viral entry assays validated the accuracy of APESS and our discoveries. Using Machine Learning, we predicted mutations that had the potential to become more prominent. We created AIVE, a web-based system, accessible at https://ai-ve.org to provide infectivity measurements of mutations entered by users. Ultimately, we established a clear link between specific viral properties and increased infectivity, enhancing our understanding of SARS-CoV-2 and enabling more accurate predictions of the virus.

[end paper 42]

[begin paper 43]

PMCID = 8941872

Title = Omicron: What Makes the Latest SARS-CoV-2 Variant of Concern So Concerning?

Emerging strains of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of the coronavirus disease 2019 (COVID-19) pandemic, that show increased transmission fitness and/or immune evasion are classified as “variants of concern” (VOCs). Recently, a SARS-CoV-2 variant first identified in November 2021 in South Africa has been recognized as a fifth VOC, termed “Omicron.” What makes this VOC so alarming is the high number of changes, especially in the viral Spike protein, and accumulating evidence for increased transmission efficiency and escape from neutralizing antibodies. In an amazingly short time, the Omicron VOC has outcompeted the previously dominating Delta VOC. However, it seems that the Omicron VOC is overall less pathogenic than other SARS-CoV-2 VOCs. Here, we provide an overview of the mutations in the Omicron genome and the resulting changes in viral proteins compared to other SARS-CoV-2 strains and discuss their potential functional consequences.

[end paper 43]

[begin paper 44]

PMCID = 10206564

Title = SARS-CoV-2 omicron variants harbor spike protein mutations responsible for their attenuated fusogenic phenotype

Since the emergence of the Omicron variants at the end of 2021, they quickly became the dominant variants globally. The Omicron variants may be more easily transmitted compared to the earlier Wuhan and the other variants. In this study, we aimed to elucidate mechanisms of the altered infectivity associated with the Omicron variants. We systemically evaluated mutations located in the S2 sequence of spike and identified mutations that are responsible for altered viral fusion. We demonstrated that mutations near the S1/S2 cleavage site decrease S1/S2 cleavage, resulting in reduced fusogenicity. Mutations in the HR1 and other S2 sequences also affect cell-cell fusion. Based on nuclear magnetic resonance (NMR) studies and in silico modeling, these mutations affect fusogenicity possibly at multiple steps of the viral fusion. Our findings reveal that the Omicron variants have accumulated mutations that contribute to reduced syncytial formation and hence an attenuated pathogenicity.

A combination of in vitro, in silico, and biochemical approaches demonstrate that mutations in the S2 domain of the SARS-CoV-2 spike protein are responsible for the attenuated viral fusion phenotype of the Omicron variant BA.1.

[end paper 44]

[begin paper 45]

PMCID = 9359488

Title = Host genetic diversity and genetic variations of SARS-CoV-2 in COVID-19 pathogenesis and the effectiveness of vaccination

The severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2), responsible for the outbreak of coronavirus disease 2019 (COVID-19), has shown a vast range of clinical manifestations from asymptomatic to life-threatening symptoms. To figure out the cause of this heterogeneity, studies demonstrated the trace of genetic diversities whether in the hosts or the virus itself. With this regard, this review provides a comprehensive overview of how host genetic such as those related to the entry of the virus, the immune-related genes, gender-related genes, disease-related genes, and also host epigenetic could influence the severity of COVID-19. Besides, the mutations in the genome of SARS-CoV-2 __leading to emerging of new variants__ per se affect the affinity of the virus to the host cells and enhance the immune escape capacity. The current review discusses these variants and also the latest data about vaccination effectiveness facing the most important variants.

[end paper 45]

[begin paper 46]

PMCID = 10409611

Title = The effects of amino acid substitution of spike protein and genomic recombination on the evolution of SARS-CoV-2

Over three years’ pandemic of 2019 novel coronavirus disease (COVID-19), multiple variants and novel subvariants have emerged successively, outcompeted earlier variants and become predominant. The sequential emergence of variants reflects the evolutionary process of mutation-selection-adaption of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Amino acid substitution/insertion/deletion in the spike protein causes altered viral antigenicity, transmissibility, and pathogenicity of SARS-CoV-2. Early in the pandemic, D614G mutation conferred virus with advantages over previous variants and increased transmissibility, and it also laid a conservative background for subsequent substantial mutations. The role of genomic recombination in the evolution of SARS-CoV-2 raised increasing concern with the occurrence of novel recombinants such as Deltacron, XBB.1.5, XBB.1.9.1, and XBB.1.16 in the late phase of pandemic. Co-circulation of different variants and co-infection in immunocompromised patients accelerate the emergence of recombinants. Surveillance for SARS-CoV-2 genomic variations, particularly spike protein mutation and recombination, is essential to identify ongoing changes in the viral genome and antigenic epitopes and thus leads to the development of new vaccine strategies and interventions.

[end paper 46]

[begin paper 47]

PMCID = 11562025

Title = Neutralisation resistance of SARS-CoV-2 spike-variants is primarily mediated by synergistic receptor binding domain substitutions

The evolution of SARS-CoV-2 has led to the emergence of numerous variants of concern (VOCs), marked by changes in the viral spike glycoprotein, the primary target for neutralising antibody (nAb) responses. Emerging VOCs, particularly omicron sub-lineages, show resistance to nAbs induced by prior infection or vaccination. The precise spike protein changes contributing to this resistance remain unclear in infectious cell culture systems. In the present study, a large panel of infectious SARS-CoV-2 mutant viruses, each with spike protein changes found in VOCs, including omicron JN.1 and its derivatives KP.2 and KP.3, was generated using a reverse genetic system. The susceptibility of these viruses to antibody neutralisation was measured using plasma from convalescent and vaccinated individuals. Synergistic roles of combined substitutions in the spike receptor binding domain (RBD) were observed in neutralisation resistance. However, recombinant viruses with the entire spike protein from a specific VOC showed enhanced resistance, indicating that changes outside the RBD are also significant. In silico analyses of spike antibody epitopes suggested that changes in neutralisation could be due to altered antibody binding affinities. Assessing ACE2 usage for entry through anti-ACE2 antibody blocking and ACE2 siRNA revealed that omicron BA.2.86 and JN.1 mutant viruses were less dependent on ACE2 for entry. However, surface plasmon resonance analysis showed increased affinity for ACE2 for both BA.2.86 and JN.1 compared to the ancestral spike. This detailed analysis of specific changes in the SARS-CoV-2 spike enhances understanding of coronavirus evolution, particularly regarding neutralising antibody evasion and ACE2 entry receptor dependence.

[end paper 47]

[begin paper 48]

PMCID = 9801140

Title = Intrinsic D614G and P681R/H mutations in SARS-CoV-2 VoCs Alpha, Delta, Omicron and viruses with D614G plus key signature mutations in spike protein alters fusogenicity and infectivity

The SARS-CoV-2 virus has been rapidly evolving over the time and the genetic variation has led to the generation of Variants of Concerns (VoC), which have shown increased fitness. These VoC viruses contain the key mutations in the spike protein which have allowed better survival and evasion of host defense mechanisms. The D614G mutation in the spike domain is found in the majority of VoC; additionally, the P681R/H mutation at the S1/S2 furin cleavage site junction is also found to be highly conserved in major VoCs; Alpha, Delta, Omicron, and its’ current variants. The impact of these genetic alterations of the SARS-CoV-2 VoCs on the host cell entry, transmissibility, and infectivity has not been clearly identified. In our study, Delta and D614G + P681R synthetic double mutant pseudoviruses showed a significant increase in the cell entry, cell-to-cell fusion and infectivity. In contrast, the Omicron and P681H synthetic single mutant pseudoviruses showed TMPRSS2 independent cell entry, less fusion and infectivity as compared to Delta and D614G + P681R double mutants. Addition of exogenous trypsin further enhanced fusion in Delta viruses as compared to Omicron. Furthermore, Delta viruses showed susceptibility to both E64d and Camostat mesylate inhibitors suggesting, that the Delta virus could exploit both endosomal and TMPRSS2 dependent entry pathways as compared to the Omicron virus. Taken together, these results indicate that the D614G and P681R/H mutations in the spike protein are pivotal which might be favoring the VoC replication in different host compartments, and thus allowing a balance of mutation vs selection for better long-term adaptation.

The online version contains supplementary material available at 10.1007/s00430-022-00760-7.

[end paper 48]

[begin paper 49]

PMCID = 10683510

Title = Structural understanding of SARS-CoV-2 virus entry to host cells

Coronavirus disease 2019 (COVID-19), caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), is a major global health concern associated with millions of fatalities worldwide. Mutant variants of the virus have further exacerbated COVID-19 mortality and infection rates, emphasizing the urgent need for effective preventive strategies. Understanding the viral infection mechanism is crucial for developing therapeutics and vaccines. The entry of SARS-CoV-2 into host cells is a key step in the infection pathway and has been targeted for drug development. Despite numerous reviews of COVID-19 and the virus, there is a lack of comprehensive reviews focusing on the structural aspects of viral entry. In this review, we analyze structural changes in Spike proteins during the entry process, dividing the entry process into prebinding, receptor binding, proteolytic cleavage, and membrane fusion steps. By understanding the atomic-scale details of viral entry, we can better target the entry step for intervention strategies. We also examine the impacts of mutations in Spike proteins, including the Omicron variant, on viral entry. Structural information provides insights into the effects of mutations and can guide the development of therapeutics and vaccines. Finally, we discuss available structure-based approaches for the development of therapeutics and vaccines. Overall, this review provides a detailed analysis of the structural aspects of SARS-CoV-2 viral entry, highlighting its significance in the development of therapeutics and vaccines against COVID-19. Therefore, our review emphasizes the importance of structural information in combating SARS-CoV-2 infection.

[end paper 49]

[begin paper 50]

PMCID = 10197516

Title = Evolving spike-protein N-glycosylation in SARS-CoV-2 variants

Since >3 years, SARS-CoV-2 has plunged humans into a colossal pandemic. Henceforth, multiple waves of infection have swept through the human population, led by variants that were able to partially evade acquired immunity. The co-evolution of SARS-CoV-2 variants with human immunity provides an excellent opportunity to study the interaction between viral pathogens and their human hosts. The heavily N-glycosylated spike-protein of SARS-CoV-2 plays a pivotal role in initiating infection and is the target for host immune-response, both of which are impacted by host-installed N-glycans. Using highly-sensitive DeGlyPHER approach, we compared the N-glycan landscape on spikes of the SARS-CoV-2 Wuhan-Hu-1 strain to seven WHO-defined variants of concern/interest, using recombinantly expressed, soluble spike-protein trimers, sharing same stabilizing-mutations. We found that N-glycan processing is conserved at most sites. However, in multiple variants, processing of N-glycans from high mannose- to complex-type is reduced at sites N165, N343 and N616, implicated in spike-protein function.

[end paper 50]

[begin paper 51]

PMCID = 9281186

Title = Continent-wide evolutionary trends of emerging SARS-CoV-2 variants: dynamic profiles from Alpha to Omicron

The ongoing SARS-CoV-2 evolution process has generated several variants due to its continuous mutations, making pandemics more critical. The present study illustrates SARS-CoV-2 evolution and its emerging mutations in five directions. First, the significant mutations in the genome and S-glycoprotein were analyzed in different variants. Three linear models were developed with the regression line to depict the mutational load for S-glycoprotein, total genome excluding S-glycoprotein, and whole genome. Second, the continent-wide evolution of SARS-CoV-2 and its variants with their clades and divergence were evaluated. It showed the region-wise evolution of the SARS-CoV-2 variants and their clustering event. The major clades for each variant were identified. One example is clade 21K, a major clade of the Omicron variant. Third, lineage dynamics and comparison between SARS-CoV-2 lineages across different countries are also illustrated, demonstrating dominant variants in various countries over time. Fourth, gene-wise mutation patterns and genetic variability of SARS-CoV-2 variants across various countries are illustrated. High mutation patterns were found in the ORF10, ORF6, S, and low mutation pattern E genes. Finally, emerging AA point mutations (T478K, L452R, N501Y, S477N, E484A, Q498R, and Y505H), their frequencies, and country-wise occurrence were identified, and the highest event of two mutations (T478K and L452R) was observed.

[end paper 51]

[begin paper 52]

PMCID = 9749455

Title = The P681H Mutation in the Spike Glycoprotein of the Alpha Variant of SARS-CoV-2 Escapes IFITM Restriction and Is Necessary for Type I Interferon Resistance

The appearance of new dominant variants of concern (VOC) of severe acute respiratory syndrome coronavirus type 2 (SARS-CoV-2) threatens the global response to the coronavirus disease 2019 (COVID-19) pandemic. Of these, the alpha variant (also known as B.1.1.7), which appeared initially in the United Kingdom, became the dominant variant in much of Europe and North America in the first half of 2021. The spike (S) glycoprotein of alpha acquired seven mutations and two deletions compared to the ancestral virus, including the P681H mutation adjacent to the polybasic cleavage site, which has been suggested to enhance S cleavage. Here, we show that the alpha spike protein confers a level of resistance to beta interferon (IFN-β) in human lung epithelial cells. This correlates with resistance to an entry restriction mediated by interferon-induced transmembrane protein 2 (IFITM2) and a pronounced infection enhancement by IFITM3. Furthermore, the P681H mutation is essential for resistance to IFN-β and context-dependent resistance to IFITMs in the alpha S. P681H reduces dependence on endosomal cathepsins, consistent with enhanced cell surface entry. However, reversion of H681 does not reduce cleaved spike incorporation into particles, indicating that it exerts its effect on entry and IFN-β downstream of furin cleavage. Overall, we suggest that, in addition to adaptive immune escape, mutations associated with VOC may well also confer a replication and/or transmission advantage through adaptation to resist innate immune mechanisms.

IMPORTANCE Accumulating evidence suggests that variants of concern (VOC) of SARS-CoV-2 evolve to evade the human immune response, with much interest focused on mutations in the spike protein that escape from antibodies. However, resistance to the innate immune response is essential for efficient viral replication and transmission. Here, we show that the alpha (B.1.1.7) VOC of SARS-CoV-2 is substantially more resistant to type I interferons than the parental Wuhan-like virus. This correlates with resistance to the antiviral protein IFITM2 and enhancement by its paralogue IFITM3. The key determinant of this is a proline-to-histidine change at position 681 in S adjacent to the furin cleavage site, which in the context of the alpha spike modulates cell entry pathways of SARS-CoV-2. Reversion of the mutation is sufficient to restore interferon and IFITM2 sensitivity, highlighting the dynamic nature of the SARS CoV-2 as it adapts to both innate and adaptive immunity in the humans.

[end paper 52]

[begin paper 53]

PMCID = 10963270

Title = High-throughput screening of genetic and cellular drivers of syncytium formation induced by the spike protein of SARS-CoV-2

Mapping mutations and discovering cellular determinants that cause the spike protein of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) to induce infected cells to form syncytia would facilitate the development of strategies for blocking the formation of such cell–cell fusion. Here we describe high-throughput screening methods based on droplet microfluidics and the size-exclusion selection of syncytia, coupled with large-scale mutagenesis and genome-wide knockout screening via clustered regularly interspaced short palindromic repeats (CRISPR), for the large-scale identification of determinants of cell–cell fusion. We used the methods to perform deep mutational scans in spike-presenting cells to pinpoint mutable syncytium-enhancing substitutions in two regions of the spike protein (the fusion peptide proximal region and the furin-cleavage site). We also used a genome-wide CRISPR screen in cells expressing the receptor angiotensin-converting enzyme 2 to identify inhibitors of clathrin-mediated endocytosis that impede syncytium formation, which we validated in hamsters infected with SARS-CoV-2. Finding genetic and cellular determinants of the formation of syncytia may reveal insights into the physiological and pathological consequences of cell–cell fusion.

High-throughput methods based on droplet microfluidics and size-exclusion selection allow for the identification of genetic and cellular drivers of syncytium formation induced by the spike protein of SARS-CoV-2 and of inhibitors of such cell–cell fusion.

[end paper 53]

[begin paper 54]

PMCID = 10278895

Title = Altered hACE2 binding affinity and S1/S2 cleavage efficiency of SARS-CoV-2 spike protein mutants affect viral cell entry

SARS-CoV-2 variants are constantly emerging, hampering public health measures in controlling the number of infections. While it is well established that mutations in spike proteins observed for the different variants directly affect virus entry into host cells, there remains a need for further expansion of systematic and multifaceted comparisons. Here, we comprehensively studied the effect of spike protein mutations on spike expression and proteolytic activation, binding affinity, viral entry efficiency and host cell tropism of eight variants of concern (VOC) and variants of interest (VOI). We found that both the full-length spike and its receptor-binding domain (RBD) of Omicron bind to hACE2 with an affinity similar to that of the wild-type. In addition, Alpha, Beta, Delta and Lambda pseudoviruses gained significantly enhanced cell entry ability compared to the wild-type, while the Omicron pseudoviruses showed a slightly increased cell entry, suggesting the vastly increased rate of transmission observed for Omicron variant is not associated with its affinity to hACE2. We also found that the spikes of Omicron and Mu showed lower S1/S2 cleavage efficiency and inefficiently utilized TMPRSS2 to enter host cells than others, suggesting that they prefer the endocytosis pathway to enter host cells. Furthermore, all variants' pseudoviruses we tested gained the ability to enter the animal ACE2-expressing cells. Especially the infection potential of rats and mice showed significantly increased, strongly suggesting that rodents possibly become a reservoir for viral evolution. The insights gained from this study provide valuable guidance for a targeted approach to epidemic control, and contribute to a better understanding of SARS-CoV-2 evolution.

All tested SARS-CoV-2 variants, except Omicron, showed increased binding affinities of their full-length S/RBD to hACE2.

Omicron pseudoviruses exhibited slightly increased cell entry ability.

S-Omicron and S-Mu showed lower S1/S2 cleavage efficiency and corresponding pseudoviruses inefficiently utilized TMPRSS2 to enter host cells compared with others.

All tested variants' pseudoviruses gain the ability to use animal ACE2 for host cell entry.

[end paper 54]

[begin paper 55]

PMCID = 10655031

Title = Analysis of spike protein variants evolved in a novel in vivo long-term replication model for SARS-CoV-2

The spectrum of SARS-CoV-2 mutations have increased over time, resulting in the emergence of several variants of concern. Persistent infection is assumed to be involved in the evolution of the variants. Calu-3 human lung cancer cells persistently grow without apoptosis and release low virus titers after infection.

We established a novel in vivo long-term replication model using xenografts of Calu-3 human lung cancer cells in immunodeficient mice. Virus replication in the tumor was monitored for 30 days and occurrence of mutations in the viral genome was determined by whole-genome deep sequencing. Viral isolates with mutations were selected after plaque forming assays and their properties were determined in cells and in K18-hACE2 mice.

After infection with parental SARS-CoV-2, viruses were found in the tumor tissues for up to 30 days and acquired various mutations, predominantly in the spike (S) protein, some of which increased while others fluctuated for 30 days. Three viral isolates with different combination of mutations produced higher virus titers than the parental virus in Calu-3 cells without cytopathic effects. In K18-hACE2 mice, the variants were less lethal than the parental virus. Infection with each variant induced production of cross-reactive antibodies to the receptor binding domain of parental SARS-CoV-2 S protein and provided protective immunity against subsequent challenge with parental virus.

These results suggest that most of the SARS-CoV-2 variants acquired mutations promoting host adaptation in the Calu-3 xenograft mice. This model can be used in the future to further study SARS-CoV-2 variants upon long-term replication in vivo.

[end paper 55]

[begin paper 56]

PMCID = 10954308

Title = Murine alveolar macrophages rapidly accumulate intranasally administered SARS-CoV-2 Spike protein leading to neutrophil recruitment and damage

The trimeric SARS-CoV-2 Spike protein mediates viral attachment facilitating cell entry. Most COVID-19 vaccines direct mammalian cells to express the Spike protein or deliver it directly via inoculation to engender a protective immune response. The trafficking and cellular tropism of the Spike protein in vivo and its impact on immune cells remains incompletely elucidated. In this study, we inoculated mice intranasally, intravenously, and subcutaneously with fluorescently labeled recombinant SARS-CoV-2 Spike protein. Using flow cytometry and imaging techniques, we analyzed its localization, immune cell tropism, and acute functional impact. Intranasal administration led to rapid lung alveolar macrophage uptake, pulmonary vascular leakage, and neutrophil recruitment and damage. When injected near the inguinal lymph node medullary, but not subcapsular macrophages, captured the protein, while scrotal injection recruited and fragmented neutrophils. Widespread endothelial and liver Kupffer cell uptake followed intravenous administration. Human peripheral blood cells B cells, neutrophils, monocytes, and myeloid dendritic cells all efficiently bound Spike protein. Exposure to the Spike protein enhanced neutrophil NETosis and augmented human macrophage TNF-α (tumor necrosis factor-α) and IL-6 production. Human and murine immune cells employed C-type lectin receptors and Siglecs to help capture the Spike protein. This study highlights the potential toxicity of the SARS-CoV-2 Spike protein for mammalian cells and illustrates the central role for alveolar macrophage in pathogenic protein uptake.

[end paper 56]

[begin paper 57]

PMCID = 10081308

Title = S:D614G and S:H655Y are gateway mutations that act epistatically to promote SARS-CoV-2 variant fitness

SARS-CoV-2 variants bearing complex combinations of mutations that confer increased transmissibility, COVID-19 severity, and immune escape, were first detected after S:D614G had gone to fixation, and likely originated during persistent infection of immunocompromised hosts. To test the hypothesis that S:D614G facilitated emergence of such variants, S:D614G was reverted to the ancestral sequence in the context of sequential Spike sequences from an immunocompromised individual, and within each of the major SARS-CoV-2 variants of concern. In all cases, infectivity of the S:D614G revertants was severely compromised. The infectivity of atypical SARS-CoV-2 lineages that propagated in the absence of S:D614G was found to be dependent upon either S:Q613H or S:H655Y. Notably, Gamma and Omicron variants possess both S:D614G and S:H655Y, each of which contributed to infectivity of these variants. Among sarbecoviruses, S:Q613H, S:D614G, and S:H655Y are only detected in SARS-CoV-2, which is also distinguished by a polybasic S1/S2 cleavage site. Genetic and biochemical experiments here showed that S:Q613H, S:D614G, and S:H655Y each stabilize Spike on virions, and that they are dispensable in the absence of S1/S2 cleavage, consistent with selection of these mutations by the S1/S2 cleavage site. CryoEM revealed that either S:D614G or S:H655Y shift the Spike receptor binding domain (RBD) towards the open conformation required for ACE2-binding and therefore on pathway for infection. Consistent with this, an smFRET reporter for RBD conformation showed that both S:D614G and S:H655Y spontaneously adopt the conformation that ACE2 induces in the parental Spike. Data from these orthogonal experiments demonstrate that S:D614G and S:H655Y are convergent adaptations to the polybasic S1/S2 cleavage site which stabilize S1 on the virion in the open RBD conformation and act epistatically to promote the fitness of variants bearing complex combinations of clinically significant mutations.

[end paper 57]

[begin paper 58]

PMCID = 9710838

Title = SARS-CoV-2 variant Alpha has a spike-dependent replication advantage over the ancestral B.1 strain in human cells with low ACE2 expression

Epidemiological data demonstrate that Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) variants of concern (VOCs) Alpha and Delta are more transmissible, infectious, and pathogenic than previous variants. Phenotypic properties of VOC remain understudied. Here, we provide an extensive functional study of VOC Alpha replication and cell entry phenotypes assisted by reverse genetics, mutational mapping of spike in lentiviral pseudotypes, viral and cellular gene expression studies, and infectivity stability assays in an enhanced range of cell and epithelial culture models. In almost all models, VOC Alpha spread less or equally efficiently as ancestral (B.1) SARS-CoV-2. B.1. and VOC Alpha shared similar susceptibility to serum neutralization. Despite increased relative abundance of specific sgRNAs in the context of VOC Alpha infection, immune gene expression in infected cells did not differ between VOC Alpha and B.1. However, inferior spreading and entry efficiencies of VOC Alpha corresponded to lower abundance of proteolytically cleaved spike products presumably linked to the T716I mutation. In addition, we identified a bronchial cell line, NCI-H1299, which supported 24-fold increased growth of VOC Alpha and is to our knowledge the only cell line to recapitulate the fitness advantage of VOC Alpha compared to B.1. Interestingly, also VOC Delta showed a strong (595-fold) fitness advantage over B.1 in these cells. Comparative analysis of chimeric viruses expressing VOC Alpha spike in the backbone of B.1, and vice versa, showed that the specific replication phenotype of VOC Alpha in NCI-H1299 cells is largely determined by its spike protein. Despite undetectable ACE2 protein expression in NCI-H1299 cells, CRISPR/Cas9 knock-out and antibody-mediated blocking experiments revealed that multicycle spread of B.1 and VOC Alpha required ACE2 expression. Interestingly, entry of VOC Alpha, as opposed to B.1 virions, was largely unaffected by treatment with exogenous trypsin or saliva prior to infection, suggesting enhanced resistance of VOC Alpha spike to premature proteolytic cleavage in the extracellular environment of the human respiratory tract. This property may result in delayed degradation of VOC Alpha particle infectivity in conditions typical of mucosal fluids of the upper respiratory tract that may be recapitulated in NCI-H1299 cells closer than in highly ACE2-expressing cell lines and models. Our study highlights the importance of cell model evaluation and comparison for in-depth characterization of virus variant-specific phenotypes and uncovers a fine-tuned interrelationship between VOC Alpha- and host cell-specific determinants that may underlie the increased and prolonged virus shedding detected in patients infected with VOC Alpha.

Epidemiological data show that SARS-CoV-2 variants of concern Alpha and Delta are more transmissible, infectious and pathogenic than previous variants, but their phenotypic properties remain understudied. A functional study of the Alpha variant identifies a single bronchial cell line that reflects its enhanced in vivo transmissibility.

[end paper 58]

[begin paper 59]

PMCID = 8789181

Title = SARS-CoV-2 variants with reduced infectivity and varied sensitivity to the BNT162b2 vaccine are developed during the course of infection

In-depth analysis of SARS-CoV-2 quasispecies is pivotal for a thorough understating of its evolution during infection. The recent deployment of COVID-19 vaccines, which elicit protective anti-spike neutralizing antibodies, has stressed the importance of uncovering and characterizing SARS-CoV-2 variants with mutated spike proteins. Sequencing databases have allowed to follow the spread of SARS-CoV-2 variants that are circulating in the human population, and several experimental platforms were developed to study these variants. However, less is known about the SARS-CoV-2 variants that are developed in the respiratory system of the infected individual. To gain further insight on SARS-CoV-2 mutagenesis during natural infection, we preformed single-genome sequencing of SARS-CoV-2 isolated from nose-throat swabs of infected individuals. Interestingly, intra-host SARS-CoV-2 variants with mutated S genes or N genes were detected in all individuals who were analyzed. These intra-host variants were present in low frequencies in the swab samples and were rarely documented in current sequencing databases. Further examination of representative spike variants identified by our analysis showed that these variants have impaired infectivity capacity and that the mutated variants showed varied sensitivity to neutralization by convalescent plasma and to plasma from vaccinated individuals. Notably, analysis of the plasma neutralization activity against these variants showed that the L1197I mutation at the S2 subunit of the spike can affect the plasma neutralization activity. Together, these results suggest that SARS-CoV-2 intra-host variants should be further analyzed for a more thorough characterization of potential circulating variants.

The global employment of mRNA-based COVID-19 vaccines, together with their high protection, has enabled to control the pandemic in places with high vaccination rate. However, the recent increase in the frequency of SARS-CoV-2 variants with reduced sensitivity to antibody neutralization has raised concerns about the ability of COVID-19 vaccines to eradicate SARS-CoV-2. Despite in-depth characterization of such variants, little is known about SARS-CoV-2 evolution within the host. Here, we preformed in-depth analysis of swab samples isolated from infected individuals and demonstrated that intra-host variants with reduced infectivity are developed during the course of the infection. Furthermore, these variants have varied sensitivity to the BNT162b2 vaccine and to convalescent plasma. In addition, we identified a point mutation at the S2 subunit of the spike protein, which impairs antibody neutralization activity. Our findings shed light on SARS-CoV-2 adaptation to the human host and highlight the importance of mapping the landscape of SARS-CoV-2 intra-host variability for better evaluating variants of concern.

[end paper 59]

[begin paper 60]

PMCID = 8782626

Title = Monoclonal antibodies targeting two immunodominant epitopes on the Spike protein neutralize emerging SARS-CoV-2 variants of concern

The emergence of new SARS-CoV-2 variants of concern B.1.1.7 (Alpha), B.1.351 (Beta), P.1 (Gamma) and B.1.617.2 (Delta) that harbor mutations in the viral S protein raised concern about activity of current vaccines and therapeutic antibodies. Independent studies have shown that mutant variants are partially or completely resistant against some of the therapeutic antibodies authorized for emergency use.

We employed hybridoma technology, ELISA-based and cell-based S-ACE2 interaction assays combined with authentic virus neutralization assays to develop second-generation antibodies, which were specifically selected for their ability to neutralize the new variants of SARS-CoV-2.

AX290 and AX677, two monoclonal antibodies with non-overlapping epitopes, exhibit subnanomolar or nanomolar affinities to the receptor binding domain of the viral Spike protein carrying amino acid substitutions N501Y, N439K, E484K, K417N, and a combination N501Y/E484K/K417N found in the circulating virus variants. The antibodies showed excellent neutralization of an authentic SARS-CoV-2 virus representing strains circulating in Europe in spring 2020 and also the variants of concern B.1.1.7 (Alpha), B.1.351 (Beta) and B.1.617.2 (Delta). In addition, AX677 is able to bind Omicron Spike protein just like the wild type Spike. The combination of the two antibodies prevented the appearance of escape mutations of the authentic SARS-CoV-2 virus. Prophylactic administration of AX290 and AX677, either individually or in combination, effectively reduced viral burden and inflammation in the lungs, and prevented disease in a mouse model of SARS-CoV-2 infection.

The virus-neutralizing properties were fully reproduced in chimeric mouse-human versions of the antibodies, which may represent a promising tool for COVID-19 therapy.

The study was funded by AXON Neuroscience SE and AXON COVIDAX a.s.

[end paper 60]

[begin paper 61]

PMCID = 10334861

Title = Interspecies transmission of SARS CoV-2 with special emphasis on viral mutations and ACE-2 receptor homology roles

COVID-19 outbreak was first reported in 2019, Wuhan, China. The spillover of the disease caused by Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2), to a wide range of pet, zoo, wild, and farm animals has emphasized potential zoonotic and reverse zoonotic viral transmission. Furthermore, it has evoked inquiries about susceptibility of different animal species to SARS-CoV-2 infection and role of these animals as viral reservoirs. Therefore, studying susceptible and non-susceptible hosts for SARS-CoV-2 infection could give a better understanding for the virus and will help in preventing further outbreaks. Here, we review structural aspects of SARS-CoV-2 spike protein, the effect of the different mutations observed in the spike protein, and the impact of ACE2 receptor variations in different animal hosts on inter-species transmission. Moreover, the SARS-CoV-2 spillover chain was reviewed. Combination of SARS-CoV-2 high mutation rate and homology of cellular ACE2 receptors enable the virus to transcend species barriers and facilitate its transmission between humans and animals.

[end paper 61]

[begin paper 62]

PMCID = 10142771

Title = Understanding Mutations in Human SARS-CoV-2 Spike Glycoprotein: A Systematic Review & Meta-Analysis

Genetic variant(s) of concern (VoC) of SARS-CoV-2 have been emerging worldwide due to mutations in the gene encoding spike glycoprotein. We performed comprehensive analyses of spike protein mutations in the significant variant clade of SARS-CoV-2, using the data available on the Nextstrain server. We selected various mutations, namely, A222V, N439K, N501Y, L452R, Y453F, E484K, K417N, T478K, L981F, L212I, N856K, T547K, G496S, and Y369C for this study. These mutations were chosen based on their global entropic score, emergence, spread, transmission, and their location in the spike receptor binding domain (RBD). The relative abundance of these mutations was mapped with global mutation D614G as a reference. Our analyses suggest the rapid emergence of newer global mutations alongside D614G, as reported during the recent waves of COVID-19 in various parts of the world. These mutations could be instrumentally imperative for the transmission, infectivity, virulence, and host immune system’s evasion of SARS-CoV-2. The probable impact of these mutations on vaccine effectiveness, antigenic diversity, antibody interactions, protein stability, RBD flexibility, and accessibility to human cell receptor ACE2 was studied in silico. Overall, the present study can help researchers to design the next generation of vaccines and biotherapeutics to combat COVID-19 infection.

[end paper 62]

[begin paper 63]

PMCID = 8877688

Title = Allosteric Determinants of the SARS-CoV-2 Spike Protein Binding with Nanobodies: Examining Mechanisms of Mutational Escape and Sensitivity of the Omicron Variant

Structural and biochemical studies have recently revealed a range of rationally engineered nanobodies with efficient neutralizing capacity against the SARS-CoV-2 virus and resilience against mutational escape. In this study, we performed a comprehensive computational analysis of the SARS-CoV-2 spike trimer complexes with single nanobodies Nb6, VHH E, and complex with VHH E/VHH V nanobody combination. We combined coarse-grained and all-atom molecular simulations and collective dynamics analysis with binding free energy scanning, perturbation-response scanning, and network centrality analysis to examine mechanisms of nanobody-induced allosteric modulation and cooperativity in the SARS-CoV-2 spike trimer complexes with these nanobodies. By quantifying energetic and allosteric determinants of the SARS-CoV-2 spike protein binding with nanobodies, we also examined nanobody-induced modulation of escaping mutations and the effect of the Omicron variant on nanobody binding. The mutational scanning analysis supported the notion that E484A mutation can have a significant detrimental effect on nanobody binding and result in Omicron-induced escape from nanobody neutralization. Our findings showed that SARS-CoV-2 spike protein might exploit the plasticity of specific allosteric hotspots to generate escape mutants that alter response to binding without compromising activity. The network analysis supported these findings showing that VHH E/VHH V nanobody binding can induce long-range couplings between the cryptic binding epitope and ACE2-binding site through a broader ensemble of communication paths that is less dependent on specific mediating centers and therefore may be less sensitive to mutational perturbations of functional residues. The results suggest that binding affinity and long-range communications of the SARS-CoV-2 complexes with nanobodies can be determined by structurally stable regulatory centers and conformationally adaptable hotspots that are allosterically coupled and collectively control resilience to mutational escape.

[end paper 63]

[begin paper 64]

PMCID = 9159519

Title = Methodology-Centered Review of Molecular Modeling, Simulation, and Prediction of SARS-CoV-2

Despite tremendous efforts in the past two years, our understanding of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), virus–host interactions, immune response, virulence, transmission, and evolution is still very limited. This limitation calls for further in-depth investigation. Computational studies have become an indispensable component in combating coronavirus disease 2019 (COVID-19) due to their low cost, their efficiency, and the fact that they are free from safety and ethical constraints. Additionally, the mechanism that governs the global evolution and transmission of SARS-CoV-2 cannot be revealed from individual experiments and was discovered by integrating genotyping of massive viral sequences, biophysical modeling of protein–protein interactions, deep mutational data, deep learning, and advanced mathematics. There exists a tsunami of literature on the molecular modeling, simulations, and predictions of SARS-CoV-2 and related developments of drugs, vaccines, antibodies, and diagnostics. To provide readers with a quick update about this literature, we present a comprehensive and systematic methodology-centered review. Aspects such as molecular biophysics, bioinformatics, cheminformatics, machine learning, and mathematics are discussed. This review will be beneficial to researchers who are looking for ways to contribute to SARS-CoV-2 studies and those who are interested in the status of the field.

[end paper 64]

[begin paper 65]

PMCID = 8989121

Title = Impairment of SARS-CoV-2 spike glycoprotein maturation and fusion activity by nitazoxanide: an effect independent of spike variants emergence

SARS-CoV-2, the causative agent of COVID-19, has caused an unprecedented global health crisis. The SARS-CoV-2 spike, a surface-anchored trimeric class-I fusion glycoprotein essential for viral entry, represents a key target for developing vaccines and therapeutics capable of blocking virus invasion. The emergence of SARS-CoV-2 spike variants that facilitate virus spread and may affect vaccine efficacy highlights the need to identify novel antiviral strategies for COVID-19 therapy. Here, we demonstrate that nitazoxanide, an antiprotozoal agent with recognized broad-spectrum antiviral activity, interferes with SARS-CoV-2 spike maturation, hampering its terminal glycosylation at an endoglycosidase H-sensitive stage. Engineering multiple SARS-CoV-2 variant-pseudoviruses and utilizing quantitative cell–cell fusion assays, we show that nitazoxanide-induced spike modifications hinder progeny virion infectivity as well as spike-driven pulmonary cell–cell fusion, a critical feature of COVID-19 pathology. Nitazoxanide, being equally effective against the ancestral SARS-CoV-2 Wuhan-spike and different emerging variants, including the Delta variant of concern, may represent a useful tool in the fight against COVID-19 infections. 

The online version contains supplementary material available at 10.1007/s00018-022-04246-w.

[end paper 65]

[begin paper 66]

PMCID = 8306177

Title = Structural Evaluation of the Spike Glycoprotein Variants on SARS-CoV-2 Transmission and Immune Evasion

The emergence of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) presents significant social, economic and political challenges worldwide. SARS-CoV-2 has caused over 3.5 million deaths since late 2019. Mutations in the spike (S) glycoprotein are of particular concern because it harbours the domain which recognises the angiotensin-converting enzyme 2 (ACE2) receptor and is the target for neutralising antibodies. Mutations in the S protein may induce alterations in the surface spike structures, changing the conformational B-cell epitopes and leading to a potential reduction in vaccine efficacy. Here, we summarise how the more important variants of SARS-CoV-2, which include cluster 5, lineages B.1.1.7 (Alpha variant), B.1.351 (Beta), P.1 (B.1.1.28/Gamma), B.1.427/B.1.429 (Epsilon), B.1.526 (Iota) and B.1.617.2 (Delta) confer mutations in their respective spike proteins which enhance viral fitness by improving binding affinity to the ACE2 receptor and lead to an increase in infectivity and transmission. We further discuss how these spike protein mutations provide resistance against immune responses, either acquired naturally or induced by vaccination. This information will be valuable in guiding the development of vaccines and other therapeutics for protection against the ongoing coronavirus disease 2019 (COVID-19) pandemic.

[end paper 66]

[begin paper 67]

PMCID = 11112362

Title = A systematic mutation analysis of 13 major SARS-CoV-2 variants

Systematic analyses of thirteen major SARS-COV-2 variants.

Omicron variant exhibited the most abundant and complex mutation sites.

Omicron variant showed higher indices of hydrophobicity and flexibility than others.

Immune escape sites 339/373/477/496 are first reported to be specific to Omicron variant.

SARS-CoV-2 evolves constantly with various novel mutations. Due to their enhanced infectivity, transmissibility and immune evasion, a comprehensive understanding of the association between these mutations and the respective functional changes is crucial. However, previous mutation studies of major SARS-CoV-2 variants remain limited. Here, we performed systematic analyses of full-length amino acids mutation, phylogenetic features, protein physicochemical properties, molecular dynamics and immune escape as well as pseudotype virus infection assays among thirteen major SARS-CoV-2 variants. We found that Omicron exhibited the most abundant and complex mutation sites, higher indices of hydrophobicity and flexibility than other variants. The results of molecular dynamics simulation suggest that Omicron has the highest number of hydrogen bonds and strongest binding free energy between the S protein and ACE2 receptor. Furthermore, we revealed 10 immune escape sites in 13 major variants, some of them were reported previously, but four of which (i.e. 339/373/477/496) are first reported to be specific to Omicron, whereas 462 is specific to Epslion. The infectivity of these variants was confirmed by the pseudotype virus infection assays. Our findings may help us understand the functional consequences of the mutations within various variants and the underlying mechanisms of the immune escapes conferred by the S proteins.

[end paper 67]

[begin paper 68]

PMCID = 11143066

Title = SARS-CoV-2 journey: from alpha variant to omicron and its sub-variants

The COVID-19 pandemic has affected hundreds of millions of individuals and caused more than six million deaths. The prolonged pandemic duration and the continual inter-individual transmissibility have contributed to the emergence of a wide variety of SARS-CoV-2 variants. Genomic surveillance and phylogenetic studies have shown that substantial mutations in crucial supersites of spike glycoprotein modulate the binding affinity of the evolved SARS-COV-2 lineages to ACE2 receptors and modify the binding of spike protein with neutralizing antibodies. The immunological spike mutations have been associated with differential transmissibility, infectivity, and therapeutic efficacy of the vaccines and the immunological therapies among the new variants. This review highlights the diverse genetic mutations assimilated in various SARS-CoV-2 variants. The implications of the acquired mutations related to viral transmission, infectivity, and COVID-19 severity are discussed. This review also addresses the effectiveness of human neutralizing antibodies induced by SARS-CoV-2 infection or immunization and the therapeutic antibodies against the ascended variants.

[end paper 68]

[begin paper 69]

PMCID = 9206116

Title = Potent human broadly SARS-CoV-2–neutralizing IgA and IgG antibodies effective against Omicron BA.1 and BA.2

This study identifies broadly SARS-CoV-2–neutralizing IgA and IgG antibodies from Wuhan COVID-19 convalescents active against the variants of concern Alpha, Beta, Gamma, Delta, and Omicron BA.1 and BA.2.

Memory B-cell and antibody responses to the SARS-CoV-2 spike protein contribute to long-term immune protection against severe COVID-19, which can also be prevented by antibody-based interventions. Here, wide SARS-CoV-2 immunoprofiling in Wuhan COVID-19 convalescents combining serological, cellular, and monoclonal antibody explorations revealed humoral immunity coordination. Detailed characterization of a hundred SARS-CoV-2 spike memory B-cell monoclonal antibodies uncovered diversity in their repertoire and antiviral functions. The latter were influenced by the targeted spike region with strong Fc-dependent effectors to the S2 subunit and potent neutralizers to the receptor-binding domain. Amongst those, Cv2.1169 and Cv2.3194 antibodies cross-neutralized SARS-CoV-2 variants of concern, including Omicron BA.1 and BA.2. Cv2.1169, isolated from a mucosa-derived IgA memory B cell demonstrated potency boost as IgA dimers and therapeutic efficacy as IgG antibodies in animal models. Structural data provided mechanistic clues to Cv2.1169 potency and breadth. Thus, potent broadly neutralizing IgA antibodies elicited in mucosal tissues can stem SARS-CoV-2 infection, and Cv2.1169 and Cv2.3194 are prime candidates for COVID-19 prevention and treatment.

[end paper 69]

[begin paper 70]

PMCID = 9941545

Title = Variant-specific deleterious mutations in the SARS-CoV-2 genome reveal immune responses and potentials for prophylactic vaccine development

 Introduction: Coronavirus disease 2019 (COVID-19), caused by SARS-CoV-2, has had a disastrous effect worldwide during the previous three years due to widespread infections with SARS-CoV-2 and its emerging variations. More than 674 million confirmed cases and over 6.7 million deaths have been attributed to successive waves of SARS-CoV-2 infections as of 29th January 2023. Similar to other RNA viruses, SARS-CoV-2 is more susceptible to genetic evolution and spontaneous mutations over time, resulting in the continual emergence of variants with distinct characteristics. Spontaneous mutations of SARS-CoV-2 variants increase its transmissibility, virulence, and disease severity and diminish the efficacy of therapeutics and vaccines, resulting in vaccine-breakthrough infections and re-infection, leading to high mortality and morbidity rates.

 Materials and methods: In this study, we evaluated 10,531 whole genome sequences of all reported variants globally through a computational approach to assess the spread and emergence of the mutations in the SARS-CoV-2 genome. The available data sources of NextCladeCLI 2.3.0 (https://clades.nextstrain.org/) and NextStrain (https://nextstrain.org/) were searched for tracking SARS-CoV-2 mutations, analysed using the PROVEAN, Polyphen-2, and Predict SNP mutational analysis tools and validated by Machine Learning models.

 Result: Compared to the Wuhan-Hu-1 reference strain NC 045512.2, genome-wide annotations showed 16,954 mutations in the SARS-CoV-2 genome. We determined that the Omicron variant had 6,307 mutations (retrieved sequence:1947), including 67.8% unique mutations, more than any other variant evaluated in this study. The spike protein of the Omicron variant harboured 876 mutations, including 443 deleterious mutations. Among these deleterious mutations, 187 were common and 256 were unique non-synonymous mutations. In contrast, after analysing 1,884 sequences of the Delta variant, we discovered 4,468 mutations, of which 66% were unique, and not previously reported in other variants. Mutations affecting spike proteins are mostly found in RBD regions for Omicron, whereas most of the Delta variant mutations drawn to focus on amino acid regions ranging from 911 to 924 in the context of epitope prediction (B cell & T cell) and mutational stability impact analysis protruding that Omicron is more transmissible.

 Discussion: The pathogenesis of the Omicron variant could be prevented if the deleterious and persistent unique immunosuppressive mutations can be targeted for vaccination or small-molecule inhibitor designing. Thus, our findings will help researchers monitor and track the continuously evolving nature of SARS-CoV-2 strains, the associated genetic variants, and their implications for developing effective control and prophylaxis strategies.

[end paper 70]

[begin paper 71]

PMCID = 11131022

Title = Mutations in the SARS-CoV-2 spike receptor binding domain and their delicate balance between ACE2 affinity and antibody evasion

Intensive selection pressure constrains the evolutionary trajectory of SARS-CoV-2 genomes and results in various novel variants with distinct mutation profiles. Point mutations, particularly those within the receptor binding domain (RBD) of SARS-CoV-2 spike (S) protein, lead to the functional alteration in both receptor engagement and monoclonal antibody (mAb) recognition. Here, we review the data of the RBD point mutations possessed by major SARS-CoV-2 variants and discuss their individual effects on ACE2 affinity and immune evasion. Many single amino acid substitutions within RBD epitopes crucial for the antibody evasion capacity may conversely weaken ACE2 binding affinity. However, this weakened effect could be largely compensated by specific epistatic mutations, such as N501Y, thus maintaining the overall ACE2 affinity for the spike protein of all major variants. The predominant direction of SARS-CoV-2 evolution lies neither in promoting ACE2 affinity nor evading mAb neutralization but in maintaining a delicate balance between these two dimensions. Together, this review interprets how RBD mutations efficiently resist antibody neutralization and meanwhile how the affinity between ACE2 and spike protein is maintained, emphasizing the significance of comprehensive assessment of spike mutations.

[end paper 71]

[begin paper 72]

PMCID = 11507407

Title = Cross-Species Susceptibility of Emerging Variants of SARS-CoV-2 Spike

Background: The continuous evolution of SARS-CoV-2 and the emergence of novel variants with numerous mutations have heightened concerns surrounding the possibility of cross-species transmission and the establishment of natural animal reservoirs for the virus, but the host range of emerging SARS-CoV-2 variants has not been fully explored yet. Methods: We employed an in vitro model comprising VSV∆G* pseudotyped viruses bearing SARS-CoV-2 spike proteins to explore the plausible host range of SARS-CoV-2 emerging variants. Results: The overall host tropism of emerging SARS-CoV-2 variants are consistent with that of the SARS-CoV-2 wuhan-hu-1 strain with minor difference. Pseudotyped viruses bearing spike protein from RaTG13 and RmYN02 can enter cell cultures from a broad range of mammalian species, revealing that mink and hamsters may act as potential intermediate hosts. We further investigated 95 potential site-specific mutations in the SARS-CoV-2 spike protein that could impact viral infectivity across different species. The results showed that 13 of these mutations notably increased the transduction rates by more than two-fold when compared to the wild-type spike protein. Further examination of these 13 mutations within cell cultures from 31 different species revealed heightened sensitivity in cells derived from palm civets, minks, and Chinese horseshoe bats to the VSV∆G*-SARS2-S mutants. Specific mutations, such as L24F, R158G, and L212I, were seen to significantly enhance the capacity for SARS-CoV-2 of cross-species transmission. Conclusions: This study offers critical insights for the ongoing surveillance and monitoring efforts of SARS-CoV-2 evolution, emphasizing the need for the vigilant monitoring of specific mutations in both human and animal populations.

[end paper 72]

[begin paper 73]

PMCID = 11542851

Title = Delineating the functional activity of antibodies with cross-reactivity to SARS-CoV-2, SARS-CoV-1 and related sarbecoviruses

The recurring spillover of pathogenic coronaviruses and demonstrated capacity of sarbecoviruses, such SARS-CoV-2, to rapidly evolve in humans underscores the need to better understand immune responses to this virus family. For this purpose, we characterized the functional breadth and potency of antibodies targeting the receptor binding domain (RBD) of the spike glycoprotein that exhibited cross-reactivity against SARS-CoV-2 variants, SARS-CoV-1 and sarbecoviruses from diverse clades and animal origins with spillover potential. One neutralizing antibody, C68.61, showed remarkable neutralization breadth against both SARS-CoV-2 variants and viruses from different sarbecovirus clades. C68.61, which targets a conserved RBD class 5 epitope, did not select for escape variants of SARS-CoV-2 or SARS-CoV-1 in culture nor have predicted escape variants among circulating SARS-CoV-2 strains, suggesting this epitope is functionally constrained. We identified 11 additional SARS-CoV-2/SARS-CoV-1 cross-reactive antibodies that target the more sequence conserved class 4 and class 5 epitopes within RBD that show activity against a subset of diverse sarbecoviruses with one antibody binding every single sarbecovirus RBD tested. A subset of these antibodies exhibited Fc-mediated effector functions as potent as antibodies that impact infection outcome in animal models. Thus, our study identified antibodies targeting conserved regions across SARS-CoV-2 variants and sarbecoviruses that may serve as therapeutics for pandemic preparedness as well as blueprints for the design of immunogens capable of eliciting cross-neutralizing responses.

There is a large collection of sarbecoviruses related to SARS-CoV-2 circulating in animal reservoirs with the potential to spillover into humans. Neutralizing antibodies have the potential to protect against infection, although viral escape is common. In this study, we isolated several monoclonal antibodies that show broad activity against different sarbecoviruses. The antibodies target epitopes in the core of the receptor binding domain that are highly conserved in sequence across sarbecoviruses and emerging SARS-CoV-2 variants. One antibody showed remarkable breadth against both SARS-CoV-1 variants as well as diverse sarbecoviruses. The results of deep mutational scanning suggest that mutations at these predicted sites of escape may functionally constrain viral fitness. Our functional profiling of cross-reactive antibodies highlights vulnerable sites of sarbecoviruses, with some antibodies poised as broadly neutralizing candidates for therapeutic use against future sarbecovirus emergence.

[end paper 73]

[begin paper 74]

PMCID = 8724751

Title = Monoclonal antibodies for COVID-19 therapy and SARS-CoV-2 detection

The coronavirus disease 2019 (COVID-19) pandemic is an exceptional public health crisis that demands the timely creation of new therapeutics and viral detection. Owing to their high specificity and reliability, monoclonal antibodies (mAbs) have emerged as powerful tools to treat and detect numerous diseases. Hence, many researchers have begun to urgently develop Ab-based kits for the detection of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and Ab drugs for use as COVID-19 therapeutic agents. The detailed structure of the SARS-CoV-2 spike protein is known, and since this protein is key for viral infection, its receptor-binding domain (RBD) has become a major target for therapeutic Ab development. Because SARS-CoV-2 is an RNA virus with a high mutation rate, especially under the selective pressure of aggressively deployed prophylactic vaccines and neutralizing Abs, the use of Ab cocktails is expected to be an important strategy for effective COVID-19 treatment. Moreover, SARS-CoV-2 infection may stimulate an overactive immune response, resulting in a cytokine storm that drives severe disease progression. Abs to combat cytokine storms have also been under intense development as treatments for COVID-19. In addition to their use as drugs, Abs are currently being utilized in SARS-CoV-2 detection tests, including antigen and immunoglobulin tests. Such Ab-based detection tests are crucial surveillance tools that can be used to prevent the spread of COVID-19. Herein, we highlight some key points regarding mAb-based detection tests and treatments for the COVID-19 pandemic.

[end paper 74]

[begin paper 75]

PMCID = 10588992

Title = Antibodies targeting a quaternary site on SARS-CoV-2 spike glycoprotein prevent viral receptor engagement by conformational locking

SARS-CoV-2 continues to evolve, with many variants evading clinically authorized antibodies. To isolate monoclonal antibodies (mAbs) with broadly neutralizing capacities against the virus, we screened serum samples from convalescing COVID-19 patients. We isolated two mAbs, 12-16 and 12-19, which neutralized all SARS-CoV-2 variants tested, including the XBB subvariants, and prevented infection in hamsters challenged with Omicron BA.1 intranasally. Structurally, both antibodies targeted a conserved quaternary epitope located at the interface between the N-terminal domain and subdomain 1, uncovering a site of vulnerability on SARS-CoV-2 spike. These antibodies prevented viral receptor engagement by locking the receptor-binding domain (RBD) of spike in the down conformation, revealing a mechanism of virus neutralization for non-RBD antibodies. Deep mutational scanning showed that SARS-CoV-2 could mutate to escape 12-19, but such mutations are rarely found in circulating viruses. Antibodies 12-16 and 12-19 hold promise as prophylactic agents for immunocompromised persons who do not respond robustly to COVID-19 vaccines.

Isolated bnAbs 12-16 and 12-19 from a SARS-CoV-2 recovered/vaccinated individual

These mAbs target a conserved quaternary epitope at the interface between NTD-SD1

The mAbs neutralize all current SARS-CoV-2 VOCs by locking RBD in down conformation

12-19 escape mutations are rarely found in circulating SARS-CoV-2 viruses

Current variants of SARS-CoV-2 can evade clinically authorized antibodies. Liu et al. demonstrate that two monoclonal antibodies isolated from convalescing COVID-19 patients neutralize all current SARS-CoV-2 variants of concern via interaction with a mechanism that locks the RBD in the down conformation. Mutations in the epitope targeted by these mAbs are rarely found in circulating SARS-CoV-2 viruses, suggesting clinical applicability.

[end paper 75]

[begin paper 76]

PMCID = 8743546

Title = Phylogenomics and population genomics of SARS-CoV-2 in Mexico during the pre-vaccination stage reveals variants of interest B.1.1.28.4 and B.1.1.222 or B.1.1.519 and the nucleocapsid mutation S194L associated with symptoms

Understanding the evolution of the SARS-CoV-2 virus in various regions of the world during the Covid-19 pandemic is essential to help mitigate the effects of this devastating disease. We describe the phylogenomic and population genetic patterns of the virus in Mexico during the pre-vaccination stage, including asymptomatic carriers. A real-time quantitative PCR screening and phylogenomic reconstructions directed at sequence/structure analysis of the spike glycoprotein revealed mutation of concern E484K in genomes from central Mexico, in addition to the nationwide prevalence of the imported variant 20C/S:452R (B.1.427/9). Overall, the detected variants in Mexico show spike protein mutations in the N-terminal domain (i.e. R190M), in the receptor-binding motif (i.e. T478K, E484K), within the S1–S2 subdomains (i.e. P681R/H, T732A), and at the basis of the protein, V1176F, raising concerns about the lack of phenotypic and clinical data available for the variants of interest we postulate: 20B/478K.V1 (B.1.1.222 or B.1.1.519) and 20B/P.4 (B.1.1.28.4). Moreover, the population patterns of single nucleotide variants from symptomatic and asymptomatic carriers obtained with a self-sampling scheme confirmed the presence of several fixed variants, and differences in allelic frequencies among localities. We identified the mutation N:S194L of the nucleocapsid protein associated with symptomatic patients. Phylogenetically, this mutation is frequent in Mexican sub-clades. Our results highlight the dual and complementary role of spike and nucleocapsid proteins in adaptive evolution of SARS-CoV-2 to their hosts and provide a baseline for specific follow-up of mutations of concern during the vaccination stage.

[end paper 76]

[begin paper 77]

PMCID = 9241865

Title = Effects of Spike Mutations in SARS-CoV-2 Variants of Concern on Human or Animal ACE2-Mediated Virus Entry and Neutralization

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) is a zoonotic agent capable of infecting humans and a wide range of animal species. Over the duration of the pandemic, mutations in the SARS-CoV-2 spike (S) protein have arisen, culminating in the spread of several variants of concern (VOCs) with various degrees of altered virulence, transmissibility, and neutralizing antibody escape. In this study, we used pseudoviruses that express specific SARS-CoV-2 S protein substitutions and cell lines that express angiotensin-converting enzyme 2 (ACE2) from nine different animal species to gain insights into the effects of VOC mutations on viral entry and antibody neutralization capability. All animal ACE2 receptors tested, except mink, support viral cell entry for pseudoviruses expressing the ancestral prototype S at levels comparable to human ACE2. Most single S substitutions did not significantly change virus entry, although 614G and 484K resulted in a decreased efficiency. Conversely, combinatorial VOC substitutions in the S protein were associated with increased entry of pseudoviruses. Neutralizing titers in sera from various animal species were significantly reduced against pseudoviruses expressing the S proteins of Beta, Delta, or Omicron VOCs compared to the parental S protein. Especially, substitutions in the S protein of the Omicron variant significantly reduced the neutralizing titers of the sera. This study reveals important insights into the host range of SARS-CoV-2 and the effect of recently emergent S protein substitutions on viral entry, virus replication, and antibody-mediated viral neutralization.

IMPORTANCE The ongoing coronavirus disease 2019 (COVID-19) pandemic, caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), continues to have devastating impacts on global health and socioeconomics. The recent emergence of SARS-CoV-2 variants of concern, which contain mutations that can affect the virulence, transmission, and effectiveness of licensed vaccines and therapeutic antibodies, are currently becoming the common strains circulating in humans worldwide. In addition, SARS-CoV-2 has been shown to infect a wide variety of animal species, which could result in additional mutations of the SARS-CoV-2 virus. In this study, we investigate the effect of mutations present in SARS-CoV-2 variants of concern and determine the effects of these mutations on cell entry, virulence, and antibody neutralization activity in humans and a variety of animals that might be susceptible to SARS-CoV-2 infection. This information is essential to understand the effects of important SARS-CoV-2 mutations and to inform public policy to create better strategies to control the COVID-19 pandemic.

[end paper 77]

[begin paper 78]

PMCID = 9847462

Title = SARS-CoV-2 variant biology: immune escape, transmission and fitness

In late 2020, after circulating for almost a year in the human population, severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) exhibited a major step change in its adaptation to humans. These highly mutated forms of SARS-CoV-2 had enhanced rates of transmission relative to previous variants and were termed ‘variants of concern’ (VOCs). Designated Alpha, Beta, Gamma, Delta and Omicron, the VOCs emerged independently from one another, and in turn each rapidly became dominant, regionally or globally, outcompeting previous variants. The success of each VOC relative to the previously dominant variant was enabled by altered intrinsic functional properties of the virus and, to various degrees, changes to virus antigenicity conferring the ability to evade a primed immune response. The increased virus fitness associated with VOCs is the result of a complex interplay of virus biology in the context of changing human immunity due to both vaccination and prior infection. In this Review, we summarize the literature on the relative transmissibility and antigenicity of SARS-CoV-2 variants, the role of mutations at the furin spike cleavage site and of non-spike proteins, the potential importance of recombination to virus success, and SARS-CoV-2 evolution in the context of T cells, innate immunity and population immunity. SARS-CoV-2 shows a complicated relationship among virus antigenicity, transmission and virulence, which has unpredictable implications for the future trajectory and disease burden of COVID-19.

In this Review, the authors summarize the mutations harboured by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants of concern. They describe the impact of mutations on virus infectivity and transmissibility, and discuss SARS-CoV-2 evolution in the context of T cells, innate immunity and population immunity.

[end paper 78]

[begin paper 79]

PMCID = 8118711

Title = Structural Modeling of the SARS-CoV-2 Spike/Human ACE2 Complex Interface can Identify High-Affinity Variants Associated with Increased Transmissibility

The COVID-19 pandemic has triggered concerns about the emergence of more infectious and pathogenic viral strains. As a public health measure, efficient screening methods are needed to determine the functional effects of new sequence variants. Here we show that structural modeling of SARS-CoV-2 Spike protein binding to the human ACE2 receptor, the first step in host-cell entry, predicts many novel variant combinations with enhanced binding affinities. By focusing on natural variants at the Spike-hACE2 interface and assessing over 700 mutant complexes, our analysis reveals that high-affinity Spike mutations (including N440K, S443A, G476S, E484R, G502P) tend to cluster near known human ACE2 recognition sites (K31 and K353). These Spike regions are structurally flexible, allowing certain mutations to optimize interface interaction energies. Although most human ACE2 variants tend to weaken binding affinity, they can interact with Spike mutations to generate high-affinity double mutant complexes, suggesting variation in individual susceptibility to infection. Applying structural analysis to highly transmissible variants, we find that circulating point mutations S477N, E484K and N501Y form high-affinity complexes (~40% more than wild-type). By combining predicted affinities and available antibody escape data, we show that fast-spreading viral variants exploit combinatorial mutations possessing both enhanced affinity and antibody resistance, including S477N/E484K, E484K/N501Y and K417T/E484K/N501Y. Thus, three-dimensional modeling of the Spike/hACE2 complex predicts changes in structure and binding affinity that correlate with transmissibility and therefore can help inform future intervention strategies.

[end paper 79]

[begin paper 80]

PMCID = 10578311

Title = Comparative Study of the Mutations Observed in the SARS-CoV-2 RBD Variants of Concern and Their Impact on the Interaction with the ACE2 Protein

SARS-CoV-2 strains have made an appearance across the globe, causing over 757 million cases and over 6.85 million deaths at the time of writing. The emergence of these variants shows the amplitude of genetic variation to which the wild-type strains have been subjected. The rise of the different SARS-CoV-2 variants resulting from such genetic modification has significantly affected COVD-19’s major impact on proliferation, virulence, and clinics. With the emergence of the variants of concern, the spike protein has been identified as a possible therapeutic target due to its critical role in binding to human cells and pathogenesis. These mutations could be linked to functional heterogeneity and use a different infection strategy. For example, the Omicron variant’s multiple mutations should be carefully examined, as they represent one of the most widely spread strains and hint to us that there may be more genetic changes in the virus. As a result, we applied a common protocol where we reconstructed SARS-CoV-2 variants of concern and performed molecular dynamics simulations to study the stability of the ACE2–RBD complex in each variant. We also carried out free energy calculations to compare the binding and biophysical properties of the different SARS-CoV-2 variants when they interact with ACE2. Therefore, we were able to obtain consistent results and uncover new crucial residues that were essential for preserving a balance between maintaining a high affinity for ACE2 and the capacity to evade RBD-targeted antibodies. Our detailed structural analysis showed that SARS-CoV-2 variants of concern show a higher affinity for ACE2 compared to the Wuhan strain. Additionally, residues K417N and E484K/A might play a crucial role in antibody evasion, whereas Q498R and N501Y are specifically mutated to strengthen RBD affinity to ACE2 and, thereby, increase the viral effect of the COVID-19 virus.

[end paper 80]

[begin paper 81]

PMCID = 9029021

Title = Severe Acute Respiratory Syndrome Coronavirus 2 Variants of Concern: A Perspective for Emerging More Transmissible and Vaccine-Resistant Strains

Novel severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants of concern (VOC) are constantly threatening global public health. With no end date, the pandemic persists with the emergence of novel variants that threaten the effectiveness of diagnostic tests and vaccines. Mutations in the Spike surface protein of the virus are regularly observed in the new variants, potentializing the emergence of novel viruses with different tropism from the current ones, which may change the severity and symptoms of the disease. Growing evidence has shown that mutations are being selected in favor of variants that are more capable of evading the action of neutralizing antibodies. In this context, the most important factor guiding the evolution of SARS-CoV-2 is its interaction with the host’s immune system. Thus, as current vaccines cannot block the transmission of the virus, measures complementary to vaccination, such as the use of masks, hand hygiene, and keeping environments ventilated remain essential to delay the emergence of new variants. Importantly, in addition to the involvement of the immune system in the evolution of the virus, we highlight several chemical parameters that influence the molecular interactions between viruses and host cells during invasion and are also critical tools making novel variants more transmissible. In this review, we dissect the impacts of the Spike mutations on biological parameters such as (1) the increase in Spike binding affinity to hACE2; (2) bound time for the receptor to be cleaved by the proteases; (3) how mutations associate with the increase in RBD up-conformation state in the Spike ectodomain; (4) expansion of uncleaved Spike protein in the virion particles; (5) increment in Spike concentration per virion particles; and (6) evasion of the immune system. These factors play key roles in the fast spreading of SARS-CoV-2 variants of concern, including the Omicron.

[end paper 81]

[begin paper 82]

PMCID = 7843029

Title = Circulating SARS-CoV-2 spike N439K variants maintain fitness while evading antibody-mediated immunity

SARS-CoV-2 can mutate and evade immunity, with consequences for efficacy of emerging vaccines and antibody therapeutics. Here, we demonstrate that the immunodominant SARS-CoV-2 spike (S) receptor binding motif (RBM) is a highly variable region of S and provide epidemiological, clinical, and molecular characterization of a prevalent, sentinel RBM mutation, N439K. We demonstrate N439K S protein has enhanced binding affinity to the hACE2 receptor, and N439K viruses have similar in vitro replication fitness and cause infections with similar clinical outcomes as compared to wild type. We show the N439K mutation confers resistance against several neutralizing monoclonal antibodies, including one authorized for emergency use by the US Food and Drug Administration (FDA), and reduces the activity of some polyclonal sera from persons recovered from infection. Immune evasion mutations that maintain virulence and fitness such as N439K can emerge within SARS-CoV-2 S, highlighting the need for ongoing molecular surveillance to guide development and usage of vaccines and therapeutics.

The receptor-binding motif (RBM) is a highly variable region of SARS-CoV-2 spike

RBM mutation N439K has emerged independently in multiple lineages

N439K increases spike affinity for hACE2; viral fitness and disease are unchanged

N439K confers resistance to several mAbs and escapes some polyclonal responses

Epidemiological, clinical, molecular, and structural characterization of the N439K mutation in the SARS-CoV-2 spike receptor binding motif demonstrates that it results in similar viral fitness compared to wild-type while conferring resistance against some neutralizing monoclonal antibodies and reducing the activity of some polyclonal antibody responses.

[end paper 82]

[begin paper 83]

PMCID = 11076047

Title = Some mechanistic underpinnings of molecular adaptations of SARS-COV-2 spike protein by integrating candidate adaptive polymorphisms with protein dynamics

We integrate evolutionary predictions based on the neutral theory of molecular evolution with protein dynamics to generate mechanistic insight into the molecular adaptations of the SARS-COV-2 spike (S) protein. With this approach, we first identified candidate adaptive polymorphisms (CAPs) of the SARS-CoV-2 S protein and assessed the impact of these CAPs through dynamics analysis. Not only have we found that CAPs frequently overlap with well-known functional sites, but also, using several different dynamics-based metrics, we reveal the critical allosteric interplay between SARS-CoV-2 CAPs and the S protein binding sites with the human ACE2 (hACE2) protein. CAPs interact far differently with the hACE2 binding site residues in the open conformation of the S protein compared to the closed form. In particular, the CAP sites control the dynamics of binding residues in the open state, suggesting an allosteric control of hACE2 binding. We also explored the characteristic mutations of different SARS-CoV-2 strains to find dynamic hallmarks and potential effects of future mutations. Our analyses reveal that Delta strain-specific variants have non-additive (i.e., epistatic) interactions with CAP sites, whereas the less pathogenic Omicron strains have mostly additive mutations. Finally, our dynamics-based analysis suggests that the novel mutations observed in the Omicron strain epistatically interact with the CAP sites to help escape antibody binding.

[end paper 83]

[begin paper 84]

PMCID = 10324094

Title = Evolution of Sequence and Structure of SARS-CoV-2 Spike Protein: A Dynamic Perspective

Novel coronavirus (SARS-CoV-2) enters its host cell through a surface spike protein. The viral spike protein has undergone several modifications/mutations at the genomic level, through which it modulated its structure–function and passed through several variants of concern. Recent advances in high-resolution structure determination and multiscale imaging techniques, cost-effective next-generation sequencing, and development of new computational methods (including information theory, statistical methods, machine learning, and many other artificial intelligence-based techniques) have hugely contributed to the characterization of sequence, structure, function of spike proteins, and its different variants to understand viral pathogenesis, evolutions, and transmission. Laying on the foundation of the sequence–structure–function paradigm, this review summarizes not only the important findings on structure/function but also the structural dynamics of different spike components, highlighting the effects of mutations on them. As dynamic fluctuations of three-dimensional spike structure often provide important clues for functional modulation, quantifying time-dependent fluctuations of mutational events over spike structure and its genetic/amino acidic sequence helps identify alarming functional transitions having implications for enhanced fusogenicity and pathogenicity of the virus. Although these dynamic events are more difficult to capture than quantifying a static, average property, this review encompasses those challenging aspects of characterizing the evolutionary dynamics of spike sequence and structure and their implications for functions.

[end paper 84]

[begin paper 85]

PMCID = 9257560

Title = A panel of nanobodies recognizing conserved hidden clefts of all SARS-CoV-2 spike variants including Omicron

We are amid the historic coronavirus infectious disease 2019 (COVID-19) pandemic. Imbalances in the accessibility of vaccines, medicines, and diagnostics among countries, regions, and populations, and those in war crises, have been problematic. Nanobodies are small, stable, customizable, and inexpensive to produce. Herein, we present a panel of nanobodies that can detect the spike proteins of five SARS-CoV-2 variants of concern (VOCs) including Omicron. Here we show via ELISA, lateral flow, kinetic, flow cytometric, microscopy, and Western blotting assays that our nanobodies can quantify the spike variants. This panel of nanobodies broadly neutralizes viral infection caused by pseudotyped and authentic SARS-CoV-2 VOCs. Structural analyses show that the P86 clone targets epitopes that are conserved yet unclassified on the receptor-binding domain (RBD) and contacts the N-terminal domain (NTD). Human antibodies rarely access both regions; consequently, the clone buries hidden crevasses of SARS-CoV-2 spike proteins that go undetected by conventional antibodies.

A panel of nanobodies are presented that can detect the spike proteins of five SARS-CoV-2 variants of concern and structural analyses show that one clone targets conserved epitopes on the receptor-binding domain and contacts the N-terminal domain.

[end paper 85]

[begin paper 86]

PMCID = 9937729

Title = Integrative network pharmacology and in silico analyses identify the anti-omicron SARS-CoV-2 potential of eugenol

Eugenol as a natural product is the source of isoniazid, and purified eugenol is extensively used in the cosmetics industry and the productive processes of edible spices. Accumulating evidence suggested that eugenol exerted potent anti-microorganism and anti-inflammation effects. Application of eugenol effectively reduced the risk of atherosclerosis, arterial embolism, and Type 2 diabetes. A previous study confirmed that treatment with eugenol attenuated lung inflammation and improved heart functions in SARS-CoV-2 spike S1-intoxicated mice. In addition to the study, based on a series of public datasets, computational analyses were conducted to characterize the acting targets of eugenol and the functional roles of these targets in COVID-19. The binding capacities of eugenol to conservative sites of SARS-CoV-2 like RNA-dependent RNA polymerase (RdRp) and mutable site as spike (S) protein, were calculated by using molecular docking following the molecular dynamics simulation with RMSD, RMSF, and MM-GBSA methods. The results of network pharmacology indicated that six targets, including PLAT, HMOX1, NUP88, CTSL, ITGB1 andTMPRSS2 were eugenol-SARS-CoV-2 interacting proteins. The omics results of in-silico study further implicated that eugenol increased the expression of SCARB1, HMOX1 and GDF15, especially HMOX1, which were confirmed the potential interacting targets between eugenol and SARS-CoV-2 antigens. Enrichment analyses indicated that eugenol exerted extensive biological effects such as regulating immune infiltration of macrophage, lipid localization, monooxyenase activity, iron ion binding and PPAR signaling. The results of the integrated analysis of eugenol targets and immunotranscription profile of COVID-19 cases shows that eugenol also plays an important role in strengthen of immunologic functions and regulating cytokine signaling. As a complement to the integrated analysis, the results of molecular docking indicated the potential binding interactions between eugenol and four proteins relating to cytokine production/release and the function of T type lymphocytes, including human TLR-4, TCR, NF-κB, JNK and AP-1. Furthermore, results of molecular docking and molecular dynamics (100ns) simulations implicated that stimulated modification of eugenol to the SARS-CoV-2 Omicron Spike-ACE2 complex, especially for human ACE2, and the molecular interaction of eugenol to SARS-CoV-2 RdRp, were no less favorable than two positive controls, molnupiravir and nilotinib. Dynamics (200ns) simulations indicated that the binding capacities and stabilities of eugenol to finger subdomain of RdRp is no less than molnupiravir. However, the simulated binding capacity of eugenol to SARS-CoV-2 wild type RBD and Omicron mutant RBD were less than nilotinib. Eugenol was predicted to have more favor LD50 value and lower cytotoxicity than two positive controls, and eugenol can pass through the blood-brain barrier (BBB). In a brief, eugenol is helpful for attenuating systemic inflammation induced by SARS-CoV-2 infection, due to the direct interaction of eugenol to SARS-CoV-2 proteins and extensive bio-manipulation of pro-inflammatory factors. This study carefully suggests eugenol is a candidate compound of developing drugs and supplement agents against SARS-CoV-2 and its Omicron variants.

[end paper 86]

[begin paper 87]

PMCID = 8611377

Title = Effect of natural mutations of SARS-CoV-2 on spike structure, conformation, and antigenicity

As battles to contain the COVID-19 pandemic continue, attention is focused on emerging variants of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) virus that have been deemed variants of concern because they are resistant to antibodies elicited by infection or vaccination or they increase transmissibility or disease severity. Three papers used functional and structural studies to explore how mutations in the viral spike protein affect its ability to infect host cells and to evade host immunity. Gobeil et al. looked at a variant spike protein involved in transmission between minks and humans, as well as the B1.1.7 (alpha), B.1.351 (beta), and P1 (gamma) spike variants; Cai et al. focused on the alpha and beta variants; and McCallum et al. discuss the properties of the spike protein from the B1.1.427/B.1.429 (epsilon) variant. Together, these papers show a balance among mutations that enhance stability, those that increase binding to the human receptor ACE2, and those that confer resistance to neutralizing antibodies. —VV

How SARS-CoV-2 variants gain enhanced infectivity and evade host immune responses.

Variants of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) have been circulating worldwide since the beginning of the pandemic. Some are termed Variants of Concern (VOC) because they show evidence for increased transmissibility, higher disease severity, resistance to neutralizing antibodies elicited by current vaccines or from previous infection, reduced efficacy of treatments, or failure of diagnostic detection methods. VOCs accumulate mutations in the spike (S) glycoprotein. Some VOCs that arose independently in different geographical locations show identical changes, implying convergent evolution and selective advantages of the acquired variations. A set of three amino acid substitutions in the receptor-binding domain (RBD)—Lys417 → Asn (K417N), Glu484 → Lys (E484K), and Asn501 → Tyr (N501Y)—occurred in the B.1.1.28 and B.1.351 lineages that originated in Brazil and South Africa, respectively. The P.1 lineage that branched off B.1.1.28 harbored a Lys417 → Thr (K417T) substitution while retaining the E484K and N501Y changes. The E484K substitution has attracted attention as a result of its location within the epitope of many potent neutralizing antibodies. The N501Y substitution also occurred in the B.1.1.7 variant that originated in the UK and was implicated in increased receptor binding and higher transmissibility of the variant. The B.1.1.7 variant, in turn, shares the His69/Val70 spike deletion mutation with spike from a variant that was implicated in transmission between humans and minks (ΔFVI).

Global sequencing initiatives and in vitro neutralization and antibody binding assays have rapidly provided critical and timely information on the VOCs. Here, by combining cryo–electron microscopy (cryo-EM) structural determination with binding assays and computational analyses on the variant spikes, we sought to visualize the impact of the amino acid substitutions on spike conformation to understand how these changes affect their biological function.

We measured angiotensin-converting enzyme 2 (ACE2) receptor and antibody binding for 19 SARS-CoV-2 S ectodomain constructs harboring amino acid changes found in circulating variants. These included a variant involved in interspecies SARS-CoV-2 transmission between humans and minks, as well as several VOCs including the B.1.1.7, B.1.1.28/P.1, and B.1.351 variants. Consistent with published neutralization data, B.1.1.7 showed decreased binding to N-terminal domain (NTD)–directed antibodies, whereas P.1 and B.1.351 showed reduced binding to both NTD- and RBD-directed antibodies. All variants showed increased binding to ACE2, which was mediated by higher propensity for RBD-up states, and affinity-enhancing mutations in the RBD. We observed spike instability in the mink-associated variant, highlighted by the presence of a population in the cryo-EM dataset with missing density for the S1 subunit of one protomer. Modulation of contacts between the SD1 and HR1 regions led to increased RBD-up states of the B.1.1.7 spike, with the protein stability maintained by a balance of stabilizing and destabilizing mutations. A local destabilizing effect of the RBD E484K mutation was implicated in resistance of the B.1.1.28/P.1 and B.1.351 variants to some potent RBD-directed neutralizing antibodies.

Our study revealed details of how amino acid substitutions affect spike conformation in circulating SARS-CoV-2 VOCs. We define communication networks that modulate spike allostery and show that the S protein uses different mechanisms to converge upon similar solutions for altering the RBD up/down positioning.

Naturally occurring amino acid variations are represented by colored spheres. Spike mutations from a mink-associated (ΔFV) (top left), B.1.1.7 (top right), B.1.351 (bottom right), and a spike with three RBD mutations (bottom left) are shown. Relative proportions of the RBD down and up populations are indicated for each. The three amino acid substitutions in the RBD—K417N/T, E484K, and N501Y—were found in the B.1.1.28 variant and are shared with the P.1 and B.1.351 lineages.

Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants with multiple spike mutations enable increased transmission and antibody resistance. We combined cryo–electron microscopy (cryo-EM), binding, and computational analyses to study variant spikes, including one that was involved in transmission between minks and humans, and others that originated and spread in human populations. All variants showed increased angiotensin-converting enzyme 2 (ACE2) receptor binding and increased propensity for receptor binding domain (RBD)–up states. While adaptation to mink resulted in spike destabilization, the B.1.1.7 (UK) spike balanced stabilizing and destabilizing mutations. A local destabilizing effect of the RBD E484K mutation was implicated in resistance of the B.1.1.28/P.1 (Brazil) and B.1.351 (South Africa) variants to neutralizing antibodies. Our studies revealed allosteric effects of mutations and mechanistic differences that drive either interspecies transmission or escape from antibody neutralization.

[end paper 87]

[begin paper 88]

PMCID = 9430374

Title = Spike Protein Cleavage-Activation in the Context of the SARS-CoV-2 P681R Mutation: an Analysis from Its First Appearance in Lineage A.23.1 Identified in Uganda

Based on its predicted ability to affect transmissibility and pathogenesis, surveillance studies have highlighted the role of a specific mutation (P681R) in the S1/S2 furin cleavage site of the SARS-CoV-2 spike protein. Here we analyzed A.23.1, first identified in Uganda, as a P681R-containing virus several months prior to the emergence of B.1.617.2 (Delta variant). We performed assays using peptides mimicking the S1/S2 from A.23.1 and B.1.617 and observed significantly increased cleavability with furin compared to both an original B lineage (Wuhan-Hu1) and B.1.1.7 (Alpha variant). We also performed cell–cell fusion and functional infectivity assays using pseudotyped particles and observed an increase in activity for A.23.1 compared to an original B lineage spike. However, these changes in activity were not reproduced in the B lineage spike bearing only the P681R substitution. Our findings suggest that while A.23.1 has increased furin-mediated cleavage linked to the P681R substitution, this substitution needs to occur on the background of other spike protein changes to enable its functional consequences.

IMPORTANCE During the course of the SARS-CoV-2 pandemic, viral variants have emerged that often contain notable mutations in the spike gene. Mutations that encode changes in the spike S1/S2 (furin) activation site have been considered especially impactful. The S1/S2 change from proline to arginine at position 681 (P681R) first emerged in the A.23.1 variant in Uganda, and subsequently occurred in the more widely transmitted Delta variant. We show that the A.23.1 spike is more readily activated by the host cell protease furin, but that this is not reproduced in an original SARS-CoV-2 spike containing the P681R mutation. Changes to the S1/S2 (furin) activation site play a role in SARS-CoV-2 infection and spread, but successful viruses combine these mutations with other less well identified changes, occurring as part of natural selection.

[end paper 88]

[begin paper 89]

PMCID = 8433594

Title = Contribution of single mutations to selected SARS-CoV-2 emerging variants spike antigenicity

Towards the end of 2020, multiple variants of concern (VOCs) and variants of interest (VOIs) have arisen from the original SARS-CoV-2 Wuhan-Hu-1 strain. Mutations in the Spike protein are highly scrutinized for their impact on transmissibility, pathogenesis and vaccine efficacy. Here, we contribute to the growing body of literature on emerging variants by evaluating the impact of single mutations on the overall antigenicity of selected variants and their binding to the ACE2 receptor. We observe a differential contribution of single mutants to the global variants phenotype related to ACE2 interaction and antigenicity. Using biolayer interferometry, we observe that enhanced ACE2 interaction is mostly modulated by a decrease in off-rate. Finally, we made the interesting observation that the Spikes from tested emerging variants bind better to ACE2 at 37°C compared to the D614G variant. Whether improved ACE2 binding at higher temperature facilitates emerging variants transmission remain to be demonstrated.

[end paper 89]

[begin paper 90]

PMCID = 9161165

Title = The ORF8 Protein of SARS-CoV-2 Modulates the Spike Protein and Its Implications in Viral Transmission

COVID-19 is currently global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Accompanying the rapid spread of the error-prone RNA-based genome, several dominant SARS-CoV-2 variants have been genetically identified. The mutations in the spike protein, which are essential for receptor binding and fusion, have been intensively investigated for their contributions to viral transmission. Nevertheless, the importance of other viral proteins and their mutations in SARS-CoV-2 lifecycle and transmission remains fairly understood. Here, we report the strong potency of an accessory protein ORF8 in modulating the level and processing of the spike protein. The expression of ORF8 protein does not affect propagation but expression of spike protein, which may lead to pseudovirions with less spike protein on the surface, therefore less infection potential. At the protein level, ORF8 expression led to downregulation and insufficient S1/S2 cleavage of the spike protein in a dose-dependent manner. ORF8 exhibits a strong interaction with the spike protein mainly at S1 domains and mediates its degradation through multiple pathways. The dominant clinical isolated ORF8 variants with the reduced protein stability exhibited the increased capacity of viral transmission without compromising their inhibitory effects on HLA-A2. Although the increase in spike protein level and Spike pseudovirus production observed by using highly transmissible clinical spike variants, there was no significant compromise in ORF8-mediated downregulation. Because ORF8 is important for immune surveillance and might be required for viral fitness in vivo, the alteration of the spike protein might be an optional strategy used by SARS-CoV-2 to promote viral transmission by escaping the inhibitory effects of ORF8. Therefore, our report emphasized the importance of ORF8 in SARS-CoV-2 spike protein production, maturation, and possible evolution.

[end paper 90]

[begin paper 91]

PMCID = 7676316

Title = Complete Mapping of Mutations to the SARS-CoV-2 Spike Receptor-Binding Domain that Escape Antibody Recognition

Antibodies targeting the SARS-CoV-2 spike receptor-binding domain (RBD) are being developed as therapeutics and are a major contributor to neutralizing antibody responses elicited by infection. Here, we describe a deep mutational scanning method to map how all amino-acid mutations in the RBD affect antibody binding and apply this method to 10 human monoclonal antibodies. The escape mutations cluster on several surfaces of the RBD that broadly correspond to structurally defined antibody epitopes. However, even antibodies targeting the same surface often have distinct escape mutations. The complete escape maps predict which mutations are selected during viral growth in the presence of single antibodies. They further enable the design of escape-resistant antibody cocktails—including cocktails of antibodies that compete for binding to the same RBD surface but have different escape mutations. Therefore, complete escape-mutation maps enable rational design of antibody therapeutics and assessment of the antigenic consequences of viral evolution.

Develop system to map all SARS-CoV-2 RBD mutations that escape antibody binding

Escape maps predict which mutations emerge when virus grown in presence of antibody

Escape maps inform surveillance for possible antigenic evolution

Greaney et al. develop a method to map all mutations to the SARS-CoV-2 RBD that escape antibody binding and apply this method to 10 antibodies. The resulting escape maps predict which mutations arise when virus is grown in the presence of antibody and can inform the design of antibody therapeutics.

[end paper 91]

[begin paper 92]

PMCID = 10232722

Title = How helpful were molecular dynamics simulations in shaping our understanding of SARS-CoV-2 spike protein dynamics?

The SARS-CoV-2 spike protein (S) represents an important viral component that is required for successful viral infection in humans owing to its essential role in recognition of and entry to host cells. The spike is also an appealing target for drug designers who develop vaccines and antivirals. This article is important as it summarizes how molecular simulations successfully shaped our understanding of spike conformational behavior and its role in viral infection. MD simulations found that the higher affinity of SARS-CoV-2-S to ACE2 is linked to its unique residues that add extra electrostatic and van der Waal interactions in comparison to the SARS-CoV S. This illustrates the spread potential of the pandemic SARS-CoV-2 relative to the epidemic SARS-CoV. Different mutations at the S-ACE2 interface, which is believed to increase the transmission of the new variants, affected the behavior and binding interactions in different simulations. The contributions of glycans to the opening of S were revealed via simulations. The immune evasion of S was linked to the spatial distribution of glycans. This help the virus to escape the immune system recognition. This article is important as it summarizes how molecular simulations successfully shaped our understanding of spike conformational behavior and its role in viral infection. This will pave the way to us preparing for the next pandemic as the computational tools are tailored to help fight new challenges.

[end paper 92]

[begin paper 93]

PMCID = 7904240

Title = A Biochemical Perspective of the Nonstructural Proteins (NSPs) and the Spike Protein of SARS CoV-2

The global pandemic that shut down the world in 2020 was caused by the virus, SARS CoV-2. The chemistry of the various nonstructural proteins (NSP3, NSP5, NSP12, NSP13, NSP14, NSP15, NSP16) of SARS CoV-2 is discussed. Secondly, a recent major focus of this pandemic is the variant strains of SARS CoV-2 that are increasingly occurring and more transmissible. One strain, called “D614G”, possesses a glycine (G) instead of an aspartate (D) at position 614 of the spike protein. Additionally, other emerging strains called “501Y.V1” and “501Y.V2” have several differences in the receptor binding domain of the spike protein (N501Y) as well as other locations. These structural changes may enhance the interaction between the spike protein and the ACE2 receptor of the host, increasing infectivity. The global pandemic caused by SARS CoV-2 is a rapidly evolving situation, emphasizing the importance of continuing the efforts to interrogate and understand this virus.

The online version contains supplementary material available at 10.1007/s10930-021-09967-8.

[end paper 93]

[begin paper 94]

PMCID = 8015430

Title = Neutralization potency of monoclonal antibodies recognizing dominant and subdominant epitopes on SARS-CoV-2 Spike is impacted by the B.1.1.7 variant

Interaction of the SARS-CoV-2 Spike receptor binding domain (RBD) with the receptor ACE2 on host cells is essential for viral entry. RBD is the dominant target for neutralizing antibodies, and several neutralizing epitopes on RBD have been molecularly characterized. Analysis of circulating SARS-CoV-2 variants has revealed mutations arising in the RBD, N-terminal domain (NTD) and S2 subunits of Spike. To understand how these mutations affect Spike antigenicity, we isolated and characterized >100 monoclonal antibodies targeting epitopes on RBD, NTD, and S2 from SARS-CoV-2-infected individuals. Approximately 45% showed neutralizing activity, of which ∼20% were NTD specific. NTD-specific antibodies formed two distinct groups: the first was highly potent against infectious virus, whereas the second was less potent and displayed glycan-dependant neutralization activity. Mutations present in B.1.1.7 Spike frequently conferred neutralization resistance to NTD-specific antibodies. This work demonstrates that neutralizing antibodies targeting subdominant epitopes should be considered when investigating antigenic drift in emerging variants.

Potent nAbs were isolated from an asymptomatic donor with low plasma neutralization

RBD-specific nAbs target epitopes overlapping with known RBD antibody classes

NTD mutations in B.1.1.7 Spike confer neutralization resistance to NTD-specific nAbs

Most RBD-specific nAbs retain potent neutralization of the B.1.1.7 variant

The impact of mutations arising in SARS-CoV-2 Spike on antigenicity is still not known. Graham et al. isolate potent neutralizing monoclonal antibodies from individuals experiencing a range of COVID-19 disease severity that target RBD, NTD, and non-S1 epitopes. The B.1.1.7 variant of concern was most resistant to NTD-specific nAbs whereas RBD-specific nAbs retained potent neutralization.

[end paper 94]

[begin paper 95]

PMCID = 8252702

Title = SARS-CoV-2 Mutations and their Viral Variants

Mutations in the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) occur spontaneously during replication. Thousands of mutations have accumulated and continue to since the emergence of the virus. As novel mutations continue appearing at the scene, naturally, new variants are increasingly observed.

Since the first occurrence of the SARS-CoV-2 infection, a wide variety of drug compounds affecting the binding sites of the virus have begun to be studied. As the drug and vaccine trials are continuing, it is of utmost importance to take into consideration the SARS-CoV-2 mutations and their respective frequencies since these data could lead the way to multi-drug combinations. The lack of effective therapeutic and preventive strategies against human coronaviruses (hCoVs) necessitates research that is of interest to the clinical applications.

The reason why the mutations in glycoprotein S lead to vaccine escape is related to the location of the mutation and the affinity of the protein. At the same time, it can be said that variations should occur in areas such as the receptor-binding domain (RBD), and vaccines and antiviral drugs should be formulated by targeting more than one viral protein.

In this review, a literature survey in the scope of the increasing SARS-CoV-2 mutations and the viral variations is conducted. In the light of current knowledge, the various disguises of the mutant SARS-CoV-2 forms and their apparent differences from the original strain are examined as they could possibly aid in finding the most appropriate therapeutic approaches.

[end paper 95]

[begin paper 96]

PMCID = 9958328

Title = Global landscape of SARS-CoV-2 mutations and conserved regions

At the end of December 2019, a novel strain of Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2) disease (COVID-19) has been identified in Wuhan, a central city in China, and then spread to every corner of the globe. As of October 8, 2022, the total number of COVID-19 cases had reached over 621 million worldwide, with more than 6.56 million confirmed deaths. Since SARS-CoV-2 genome sequences change due to mutation and recombination, it is pivotal to surveil emerging variants and monitor changes for improving pandemic management.

10,287,271 SARS-CoV-2 genome sequence samples were downloaded in FASTA format from the GISAID databases from February 24, 2020, to April 2022. Python programming language (version 3.8.0) software was utilized to process FASTA files to identify variants and sequence conservation. The NCBI RefSeq SARS-CoV-2 genome (accession no. NC_045512.2) was considered as the reference sequence.

Six mutations had more than 50% frequency in global SARS-CoV-2. These mutations include the P323L (99.3%) in NSP12, D614G (97.6) in S, the T492I (70.4) in NSP4, R203M (62.8%) in N, T60A (61.4%) in Orf9b, and P1228L (50.0%) in NSP3. In the SARS-CoV-2 genome, no mutation was observed in more than 90% of nsp11, nsp7, nsp10, nsp9, nsp8, and nsp16 regions. On the other hand, N, nsp3, S, nsp4, nsp12, and M had the maximum rate of mutations. In the S protein, the highest mutation frequency was observed in aa 508–635(0.77%) and aa 381–508 (0.43%). The highest frequency of mutation was observed in aa 66–88 (2.19%), aa 7–14, and aa 164–246 (2.92%) in M, E, and N proteins, respectively.

Therefore, monitoring SARS-CoV-2 proteomic changes and detecting hot spots mutations and conserved regions could be applied to improve the SARS‐CoV‐2 diagnostic efficiency and design safe and effective vaccines against emerging variants.

The online version contains supplementary material available at 10.1186/s12967-023-03996-w.

[end paper 96]

[begin paper 97]

PMCID = 8597642

Title = SARS-CoV-2 Variants and Their Relevant Mutational Profiles: Update Summer 2021

Since the beginning of the coronavirus disease 2019 (COVID-19) pandemic caused by it, severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) has been undergoing a genetic diversification leading to the emergence of new variants. Nevertheless, a clear definition of the genetic signatures underlying the circulating variants is still missing. Here, we provide a comprehensive insight into mutational profiles characterizing each SARS-CoV-2 variant, focusing on spike mutations known to modulate viral infectivity and/or antigenicity. We focused on variants and on specific relevant mutations reported by GISAID, Nextstrain, Outbreak.info, Pango, and Stanford database websites that were associated with any clinical/diagnostic impact, according to published manuscripts. Furthermore, 1,223,338 full-length high-quality SARS-CoV-2 genome sequences were retrieved from GISAID and used to accurately define the specific mutational patterns in each variant. Finally, mutations were mapped on the three-dimensional structure of the SARS-CoV-2 spike protein to assess their localization in the different spike domains. Overall, this review sheds light and assists in defining the genetic signatures characterizing the currently circulating variants and their clinical relevance.

IMPORTANCE Since the emergence of SARS-CoV-2, several recurrent mutations, particularly in the spike protein, arose during human-to-human transmission or spillover events between humans and animals, generating distinct worrisome variants of concern (VOCs) or of interest (VOIs), designated as such due to their clinical and diagnostic impacts. Characterizing these variants and their related mutations is important in tracking SAR-CoV-2 evolution and understanding the efficacy of vaccines and therapeutics based on monoclonal antibodies, convalescent-phase sera, and direct antivirals. Our study provides a comprehensive survey of the mutational profiles characterizing the important SARS-CoV-2 variants, focusing on spike mutations and highlighting other protein mutations.

[end paper 97]

[begin paper 98]

PMCID = 10238279

Title = Changes within the P681 residue of spike dictate cell fusion and syncytia formation of Delta and Omicron variants of SARS-CoV-2 with no effects on neutralization or infectivity

The rapid spread and dominance of the Omicron SARS-CoV-2 lineages have posed severe health challenges worldwide. While extensive research on the role of the Receptor Binding Domain (RBD) in promoting viral infectivity and vaccine sensitivity has been well documented, the functional significance of the 681PRRAR/SV687 polybasic motif of the viral spike is less clear. In this work, we monitored the infectivity levels and neutralization potential of the wild-type human coronavirus 2019 (hCoV-19), Delta, and Omicron SARS-CoV-2 pseudoviruses against sera samples drawn four months post administration of a third dose of the BNT162b2 mRNA vaccine. Our findings show that in comparison to hCoV-19 and Delta SARS-CoV-2, Omicron lineages BA.1 and BA.2 exhibit enhanced infectivity and a sharp decline in their sensitivity to vaccine-induced neutralizing antibodies. Interestingly, P681 mutations within the viral spike do not play a role in the neutralization potential or infectivity of SARS Cov-2 pseudoviruses carrying mutations in this position. The P681 residue however, dictates the ability of the spike protein to promote fusion and syncytia formation between infected cells. While spike from hCoV-19 (P681) and Omicron (H681) promote only modest cell fusion and formation of syncytia between cells that express the spike-protein, Delta spike (R681) displays enhanced fusogenic activity and promotes syncytia formation. Additional analysis shows that a single P681R mutation within the hCoV-19 spike, or H681R within the Omicron spike, restores fusion potential to similar levels observed for the Delta R681 spike. Conversely, R681P point mutation within the spike of Delta pseudovirus abolishes efficient fusion and syncytia formation. Our investigation also demonstrates that spike proteins from hCoV-19 and Delta SARS-CoV-2 are efficiently incorporated into viral particles relative to the spike of Omicron lineages. We conclude that the third dose of the Pfizer-BNT162b2 provides appreciable protection against the newly emerged Omicron sub-lineages. However, the neutralization sensitivity of these new variants is diminished relative to that of the hCoV-19 or Delta SARS-CoV-2. We further show that the P681 residue within spike dictates cell fusion and syncytia formation with no effects on the infectivity of the specific viral variant and on its sensitivity to vaccine-mediated neutralization.

[end paper 98]

[begin paper 99]

PMCID = 10265947

Title = Identification of broad, potent antibodies to functionally constrained regions of SARS-CoV-2 spike following a breakthrough infection

SARS-CoV-2 infections can result in diverse clinical outcomes, including severe disease. Monoclonal antibodies (mAbs) have been used therapeutically to treat infection, but the emergence of variants has compromised their efficacy. Thus, identifying mAbs that are more durable in the face of SARS-CoV-2 evolution is a pressing need. Here, we describe four mAbs isolated from a Delta-breakthrough infection that can potently neutralize diverse variants, including multiple Omicron variants. In addition, one mAb shows broader activity against coronaviruses. The breadth of these mAbs is due to their focus on highly conserved regions of the viral protein antigen, including regions that are required for the virus to enter the cell. These properties make them promising candidates for therapeutic use.

The antiviral benefit of antibodies can be compromised by viral escape especially for rapidly evolving viruses. Therefore, durable, effective antibodies must be both broad and potent to counter newly emerging, diverse strains. Discovery of such antibodies is critically important for SARS-CoV-2 as the global emergence of new variants of concern (VOC) has compromised the efficacy of therapeutic antibodies and vaccines. We describe a collection of broad and potent neutralizing monoclonal antibodies (mAbs) isolated from an individual who experienced a breakthrough infection with the Delta VOC. Four mAbs potently neutralize the Wuhan-Hu-1 vaccine strain, the Delta VOC, and also retain potency against the Omicron VOCs through BA.4/BA.5 in both pseudovirus-based and authentic virus assays. Three mAbs also retain potency to recently circulating VOCs XBB.1.5 and BQ.1.1 and one also potently neutralizes SARS-CoV-1. The potency of these mAbs was greater against Omicron VOCs than all but one of the mAbs that had been approved for therapeutic applications. The mAbs target distinct epitopes on the spike glycoprotein, three in the receptor-binding domain (RBD) and one in an invariant region downstream of the RBD in subdomain 1 (SD1). The escape pathways we defined at single amino acid resolution with deep mutational scanning show they target conserved, functionally constrained regions of the glycoprotein, suggesting escape could incur a fitness cost. Overall, these mAbs are unique in their breadth across VOCs, their epitope specificity, and include a highly potent mAb targeting a rare epitope outside of the RBD in SD1.

[end paper 99]

[begin paper 100]

PMCID = 7891930

Title = The Spike D614G mutation increases SARS-CoV-2 infection of multiple human cell types

A novel variant of the SARS-CoV-2 virus carrying a point mutation in the Spike protein (D614G) has recently emerged and rapidly surpassed others in prevalence. This mutation is in linkage disequilibrium with an ORF1b protein variant (P314L), making it difficult to discern the functional significance of the Spike D614G mutation from population genetics alone. Here, we perform site-directed mutagenesis on wild-type human-codon-optimized Spike to introduce the D614G variant. Using multiple human cell lines, including human lung epithelial cells, we found that the lentiviral particles pseudotyped with Spike D614G are more effective at transducing cells than ones pseudotyped with wild-type Spike. The increased transduction with Spike D614G ranged from 1.3- to 2.4-fold in Caco-2 and Calu-3 cells expressing endogenous ACE2 and from 1.5- to 7.7-fold in A549ACE2 and Huh7.5ACE2 overexpressing ACE2. Furthermore, trans-complementation of SARS-CoV-2 virus with Spike D614G showed an increased infectivity in human cells. Although there is minimal difference in ACE2 receptor binding between the D614 and G614 Spike variants, the G614 variant is more resistant to proteolytic cleavage, suggesting a possible mechanism for the increased transduction.

[end paper 100]



