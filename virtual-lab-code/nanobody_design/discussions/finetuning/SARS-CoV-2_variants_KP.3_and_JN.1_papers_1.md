## User

This is the beginning of an individual meeting with Immunologist to discuss your research project.

Here is the agenda for the meeting:

You are working on a research project to use machine learning to develop antibodies or nanobodies for the newest variant of the SARS-CoV-2 spike protein that also, ideally, have activity against other circulating minor variants and past variants. Your team previous decided to modify existing nanobodies to improve their binding to the newest variant of the SARS-CoV-2 spike protein. You are responsible for understanding the topic "SARS-CoV-2 variants KP.3 and JN.1" in the context of designing nanobody binders for SARS-CoV-2. You need to fine-tune yourself on the relevant literature on SARS-CoV-2 variants KP.3 and JN.1 to improve your ability to design SARS-CoV-2 nanobody binders. Please use PubMed Central and search for relevant papers on SARS-CoV-2 variants KP.3 and JN.1 using the query "SARS-CoV-2 variant KP.3 spike protein structure" and request 100 articles with abstracts only. Read all of the abstracts and based on each abstract individually, decide whether you want to fine-tune yourself on the full text of that paper. Include as many papers as possible, but only include papers that are directly relevant to SARS-CoV-2 variants KP.3 and JN.1. Please provide the PMCIDs and titles of all the papers that you wish to fine-tune yourself on as a Python dictionary mapping PMCID as a double-quoted string to title as a double-quoted string.

Here are the agenda questions that must be answered:

1. What are the PMCIDs and titles of the papers you wish to fine-tune yourself on (as a Python dictionary mapping PMCID as a double-quoted string to title as double-quoted string)?

Immunologist, please provide your response to the agenda.

## Immunologist

Based on the abstracts provided, here are the PMCIDs and titles of the papers that are directly relevant to SARS-CoV-2 variants KP.3 and JN.1, which I would like to fine-tune myself on:

```python
{
    "11661143": "Structural and Functional Insights into the Evolution of SARS-CoV-2 KP.3.1.1 Spike Protein",
    "11437503": "AlphaFold2 Modeling and Molecular Dynamics Simulations of the Conformational Ensembles for the SARS-CoV-2 Spike Omicron JN.1, KP.2 and KP.3 Variants: Mutational Profiling of Binding Energetics Reveals Epistatic Drivers of the ACE2 Affinity and Escape Hotspots of Antibody Resistance",
    "11554226": "Super broad and protective nanobodies against Sarbecoviruses including SARS-CoV-1 and the divergent SARS-CoV-2 subvariant KP.3.1.1",
    "11562025": "Neutralisation resistance of SARS-CoV-2 spike-variants is primarily mediated by synergistic receptor binding domain substitutions",
    "11722104": "Ongoing evolution of SARS-CoV-2 drives escape from mRNA vaccine-induced humoral immunity",
    "11601353": "Immune Evasion, Cell-Cell Fusion, and Spike Stability of the SARS-CoV-2 XEC Variant: Role of Glycosylation Mutations at the N-terminal Domain",
    "11754117": "Evolving antibody response to SARS-CoV-2 antigenic shift from XBB to JN.1",
    "11461813": "Lineage-specific pathogenicity, immune evasion, and virological features of SARS-CoV-2 BA.2.86/JN.1 and EG.5.1/HK.3",
    "11407393": "Recurrent SARS-CoV-2 spike mutations confer growth advantages to select JN.1 sublineages",
    "11406931": "Ipsilateral or contralateral boosting of mice with mRNA vaccines confers equivalent immunity and protection against a SARS-CoV-2 Omicron strain"
}
```

These papers provide insights into the structural, functional, and immunological aspects of the KP.3 and JN.1 variants, which are crucial for designing effective nanobody binders.

## User

Tool Output:

Here are the top 54 articles on PubMed Central for the query "SARS-CoV-2 variant KP.3 spike protein structure":

[begin paper 1]

PMCID = 11661143

Title = Structural and Functional Insights into the Evolution of SARS-CoV-2 KP.3.1.1 Spike Protein

The JN.1-sublineage KP.3.1.1 recently emerged as the globally prevalent SARS-CoV-2 variant, demonstrating increased infectivity and antibody escape. We investigated how mutations and a deletion in the KP.3.1.1 spike protein (S) affect ACE2 binding and antibody escape. Mass spectrometry revealed a new glycan site at residue N30 and altered glycoforms at neighboring N61. Cryo-EM structures showed that the N30 glycan and rearrangement of adjacent residues did not significantly change the overall spike structure, up-down ratio of the receptor-binding domains (RBDs), or ACE2 binding. Furthermore, a KP.3.1.1 S structure with hACE2 further confirmed an epistatic effect between F456L and Q493E on ACE2 binding. Our analysis shows SARS-CoV-2 variants that emerged after late 2023 are now incorporating reversions to residues found in other sarbecoviruses, including the N30 glycan, Q493E, and others. Overall, these results inform on the structural and functional consequences of the KP.3.1.1 mutations, the current SARS-CoV-2 evolutionary trajectory, and immune evasion.

[end paper 1]

[begin paper 2]

PMCID = 11437503

Title = AlphaFold2 Modeling and Molecular Dynamics Simulations of the Conformational Ensembles for the SARS-CoV-2 Spike Omicron JN.1, KP.2 and KP.3 Variants: Mutational Profiling of Binding Energetics Reveals Epistatic Drivers of the ACE2 Affinity and Escape Hotspots of Antibody Resistance

The most recent wave of SARS-CoV-2 Omicron variants descending from BA.2 and BA.2.86 exhibited improved viral growth and fitness due to convergent evolution of functional hotspots. These hotspots operate in tandem to optimize both receptor binding for effective infection and immune evasion efficiency, thereby maintaining overall viral fitness. The lack of molecular details on structure, dynamics and binding energetics of the latest FLiRT and FLuQE variants with the ACE2 receptor and antibodies provides a considerable challenge that is explored in this study. We combined AlphaFold2-based atomistic predictions of structures and conformational ensembles of the SARS-CoV-2 spike complexes with the host receptor ACE2 for the most dominant Omicron variants JN.1, KP.1, KP.2 and KP.3 to examine the mechanisms underlying the role of convergent evolution hotspots in balancing ACE2 binding and antibody evasion. Using the ensemble-based mutational scanning of the spike protein residues and computations of binding affinities, we identified binding energy hotspots and characterized the molecular basis underlying epistatic couplings between convergent mutational hotspots. The results suggested the existence of epistatic interactions between convergent mutational sites at L455, F456, Q493 positions that protect and restore ACE2-binding affinity while conferring beneficial immune escape. To examine immune escape mechanisms, we performed structure-based mutational profiling of the spike protein binding with several classes of antibodies that displayed impaired neutralization against BA.2.86, JN.1, KP.2 and KP.3. The results confirmed the experimental data that JN.1, KP.2 and KP.3 harboring the L455S and F456L mutations can significantly impair the neutralizing activity of class 1 monoclonal antibodies, while the epistatic effects mediated by F456L can facilitate the subsequent convergence of Q493E changes to rescue ACE2 binding. Structural and energetic analysis provided a rationale to the experimental results showing that BD55-5840 and BD55-5514 antibodies that bind to different binding epitopes can retain neutralizing efficacy against all examined variants BA.2.86, JN.1, KP.2 and KP.3. The results support the notion that evolution of Omicron variants may favor emergence of lineages with beneficial combinations of mutations involving mediators of epistatic couplings that control balance of high ACE2 affinity and immune evasion.

[end paper 2]

[begin paper 3]

PMCID = 11554226

Title = Super broad and protective nanobodies against Sarbecoviruses including SARS-CoV-1 and the divergent SARS-CoV-2 subvariant KP.3.1.1

The ongoing evolution and immune escape of SARS-CoV-2, alongside the potential threat of SARS-CoV-1 and other sarbecoviruses, underscore the urgent need for effective strategies against their infection and transmission. This study highlights the discovery of nanobodies from immunized alpacas, which demonstrate exceptionally broad and potent neutralizing capabilities against the recently emerged and more divergent SARS-CoV-2 Omicron subvariants including JD.1.1, JN.1, KP.3, KP.3.1.1, as well as SARS-CoV-1 and coronaviruses from bats and pangolins utilizing receptor ACE2. Among these, Tnb04-1 emerges as the most broad and potent, binding to a conserved hydrophobic pocket in the spike’s receptor-binding domain, distinct from the ACE2 binding site. This interaction disrupts the formation of a proteinase K-resistant core, crucial for viral-cell fusion. Notably, intranasal administration of Tnb04-1 in Syrian hamsters effectively prevented respiratory infection and transmission of the authentic Omicron XBB.1.5 subvariant. Thus, Thb04-1 holds promise in combating respiratory acquisition and transmission of diverse sarbecoviruses.

As SARS-CoV-2 evolves and evades antibody immunity, new antibody intervention strategies are urgently needed to prevent viral acquisition and transmission. We identified a cluster of nanobodies from immunized alpacas with broad and potent neutralizing activity against all major SARS-CoV-2 variants, SARS-CoV-1, and ACE2-utilizing coronaviruses from bats and pangolins. Tnb04-1 was the most effective, binding to a conserved epitope in spike and providing strong protection against contact and respiratory infection of Omicron XBB.1.5 in Syrian hamsters. Tnb04-1 shows promise for next-generation antibody interventions against diverse sarbecoviruses.

[end paper 3]

[begin paper 4]

PMCID = 11562025

Title = Neutralisation resistance of SARS-CoV-2 spike-variants is primarily mediated by synergistic receptor binding domain substitutions

The evolution of SARS-CoV-2 has led to the emergence of numerous variants of concern (VOCs), marked by changes in the viral spike glycoprotein, the primary target for neutralising antibody (nAb) responses. Emerging VOCs, particularly omicron sub-lineages, show resistance to nAbs induced by prior infection or vaccination. The precise spike protein changes contributing to this resistance remain unclear in infectious cell culture systems. In the present study, a large panel of infectious SARS-CoV-2 mutant viruses, each with spike protein changes found in VOCs, including omicron JN.1 and its derivatives KP.2 and KP.3, was generated using a reverse genetic system. The susceptibility of these viruses to antibody neutralisation was measured using plasma from convalescent and vaccinated individuals. Synergistic roles of combined substitutions in the spike receptor binding domain (RBD) were observed in neutralisation resistance. However, recombinant viruses with the entire spike protein from a specific VOC showed enhanced resistance, indicating that changes outside the RBD are also significant. In silico analyses of spike antibody epitopes suggested that changes in neutralisation could be due to altered antibody binding affinities. Assessing ACE2 usage for entry through anti-ACE2 antibody blocking and ACE2 siRNA revealed that omicron BA.2.86 and JN.1 mutant viruses were less dependent on ACE2 for entry. However, surface plasmon resonance analysis showed increased affinity for ACE2 for both BA.2.86 and JN.1 compared to the ancestral spike. This detailed analysis of specific changes in the SARS-CoV-2 spike enhances understanding of coronavirus evolution, particularly regarding neutralising antibody evasion and ACE2 entry receptor dependence.

[end paper 4]

[begin paper 5]

PMCID = 11722104

Title = Ongoing evolution of SARS-CoV-2 drives escape from mRNA vaccine-induced humoral immunity

With the onset of the COVID-19 pandemic 4 years ago, viral sequencing continues to document numerous individual mutations in the viral spike protein across many variants. To determine the ability of vaccine-mediated humoral immunity to combat continued SARS-CoV-2 evolution, we construct a comprehensive panel of pseudoviruses harboring each individual mutation spanning 4 years of the pandemic to understand the fitness cost and resistance benefits of each. These efforts identify numerous mutations that escape from vaccine-induced humoral immunity. Across 50 variants and 131 mutants we construct, we observe progressive loss of neutralization across variants, irrespective of vaccine doses, as well as increasing infectivity and ACE2 binding. Importantly, the recent XBB.1.5 booster significantly increases titers against most variants but not JN.1, KP.2, or KP.3. These findings demonstrate that variants continue to evade updated mRNA vaccines, highlighting the need for different approaches to control SARS-CoV-2 transmission.

Primary vaccination reveals regions of vulnerability within the mutational landscape

mRNA booster vaccination significantly enhances neutralization of SARS-CoV-2 mutants

Individual omicron mutants reveal gaps in mRNA-boosted humoral immunity

Variant-specific vaccines markedly improve neutralization breadth

Roederer et al. examine how the SARS-CoV-2 virus evolves over time and how well updated vaccines protect against variants. Variants of the virus are becoming harder for vaccine-mediated humoral immunity to neutralize, even with updated boosters, suggesting alternate vaccination methods may be necessary to combat SARS-CoV-2 evolution.

[end paper 5]

[begin paper 6]

PMCID = 11601353

Title = Immune Evasion, Cell-Cell Fusion, and Spike Stability of the SARS-CoV-2 XEC Variant: Role of Glycosylation Mutations at the N-terminal Domain

SARS-CoV-2 continues to evolve, producing new variants that drive global COVID-19 surges. XEC, a recombinant of KS.1.1 and KP.3.3, contains T22N and F59S mutations in the spike protein’s N-terminal domain (NTD). The T22N mutation, similar to the DelS31 mutation in KP.3.1.1, introduces a potential N-linked glycosylation site in XEC. In this study, we examined the neutralizing antibody (nAb) response and mutation effects in sera from bivalent-vaccinated healthcare workers, BA.2.86/JN.1 wave-infected patients, and XBB.1.5 monovalent-vaccinated hamsters, assessing responses to XEC alongside D614G, JN.1, KP.3, and KP.3.1.1. XEC demonstrated significantly reduced neutralization titers across all cohorts, largely due to the F59S mutation. Notably, removal of glycosylation sites in XEC and KP.3.1.1 substantially restored nAb titers. Antigenic cartography analysis revealed XEC to be more antigenically distinct from its common ancestral BA.2.86/JN.1 compared to KP.3.1.1, with the F59S mutation as a determining factor. Similar to KP.3.1.1, XEC showed reduced cell-cell fusion relative to its parental KP.3, a change attributed to the T22N glycosylation. We also observed reduced S1 shedding for XEC and KP.3.1.1, which was reversed by ablation of T22N and DelS31 glycosylation mutations, respectively. Molecular modeling suggests that T22N and F59S mutations of XEC alters hydrophobic interactions with adjacent spike protein residues, impacting both conformational stability and neutralization. Overall, our findings underscore the pivotal role of NTD mutations in shaping SARS-CoV-2 spike biology and immune escape mechanisms.

[end paper 6]

[begin paper 7]

PMCID = 11727448

Title = Structural Immunology of SARS‐CoV‐2

The SARS‐CoV‐2 spike (S) protein has undergone significant evolution, enhancing both receptor binding and immune evasion. In this review, we summarize ongoing efforts to develop antibodies targeting various epitopes of the S protein, focusing on their neutralization potency, breadth, and escape mechanisms. Antibodies targeting the receptor‐binding site (RBS) typically exhibit high neutralizing potency but are frequently evaded by mutations in SARS‐CoV‐2 variants. In contrast, antibodies targeting conserved regions, such as the S2 stem helix and fusion peptide, exhibit broader reactivity but generally lower neutralization potency. However, several broadly neutralizing antibodies have demonstrated exceptional efficacy against emerging variants, including the latest omicron subvariants, underscoring the potential of targeting vulnerable sites such as RBS‐A and RBS‐D/CR3022. We also highlight public classes of antibodies targeting different sites on the S protein. The vulnerable sites targeted by public antibodies present opportunities for germline‐targeting vaccine strategies. Overall, developing escape‐resistant, potent antibodies and broadly effective vaccines remains crucial for combating future variants. This review emphasizes the importance of identifying key epitopes and utilizing antibody affinity maturation to inform future therapeutic and vaccine design.

[end paper 7]

[begin paper 8]

PMCID = 11680242

Title = Characterization and Fluctuations of an Ivermectin Binding Site at the Lipid Raft Interface of the N-Terminal Domain (NTD) of the Spike Protein of SARS-CoV-2 Variants

Most studies on the docking of ivermectin on the spike protein of SARS-CoV-2 concern the receptor binding domain (RBD) and, more precisely, the RBD interface recognized by the ACE2 receptor. The N-terminal domain (NTD), which controls the initial attachment of the virus to lipid raft gangliosides, has not received the attention it deserves. In this study, we combined molecular modeling and physicochemical approaches to analyze the mode of interaction of ivermectin with the interface of the NTD-facing lipid rafts of the host cell membrane. We characterize a binding area that presents point mutations and deletions in successive SARS-CoV-2 variants from the initial strain to omicron KP.3 circulating in many countries in 2024. We show that ivermectin has exceptional flexibility, allowing the drug to bind to the spike protein of all variants tested. The energy of interaction is specific to each variant, allowing a classification according to their affinity for ivermectin in the following ascending order: Omicron KP.3 < Delta < Omicron BA.5 < Alpha < Wuhan (B.1) < Omicron BA.1. The binding site of ivermectin is subject to important variations of the NTD, including the Y144 deletion. It overlaps with the ganglioside binding domain of the NTD, as demonstrated by docking and physicochemical studies. These results suggest a new mechanism of antiviral action for ivermectin based on competitive inhibition for initial virus attachment to lipid rafts. The current KP.3 variant is still recognized by ivermectin, although with an affinity slightly lower than the Wuhan strain.

[end paper 8]

[begin paper 9]

PMCID = 11609190

Title = Research progress of spike protein mutation of SARS-CoV-2 mutant strain and antibody development

The coronavirus disease 2019 (COVID-19) is a respiratory disease with a very high infectious rate caused by the Severe Acute Respiratory Syndrome Coronavirus-2(SARS-CoV-2). Because SARS-CoV-2 is easy to mutate, the continuous emergence of SARS-CoV-2 variant strains not only enhances the infectivity of the SARS-CoV-2 but also brings great obstacles to the treatment of COVID-19. Neutralizing antibodies have achieved good results in the clinical application of the novel coronavirus pneumonia, which can be used for pre-infection protection and treatment of novel coronavirus patients. This review makes a detailed introduction to the mutation characteristics of SARS-CoV-2, focusing on the molecular mechanism of mutation affecting the infectivity of SARS-CoV-2, and the impact of mutation on monoclonal antibody therapy, providing scientific reference for the prevention of SARS-CoV-2 variant strains and the research and development of antibody drugs.

[end paper 9]

[begin paper 10]

PMCID = 11414647

Title = Deep mutational scanning of SARS-CoV-2 Omicron BA.2.86 and epistatic emergence of the KP.3 variant

Deep mutational scanning experiments aid in the surveillance and forecasting of viral evolution by providing prospective measurements of mutational effects on viral traits, but epistatic shifts in the impacts of mutations can hinder viral forecasting when measurements were made in outdated strain backgrounds. Here, we report measurements of the impact of all single amino acid mutations on ACE2-binding affinity and protein folding and expression in the SARS-CoV-2 Omicron BA.2.86 spike receptor-binding domain. As with other SARS-CoV-2 variants, we find a plastic and evolvable basis for receptor binding, with many mutations at the ACE2 interface maintaining or even improving ACE2-binding affinity. Despite its large genetic divergence, mutational effects in BA.2.86 have not diverged greatly from those measured in its Omicron BA.2 ancestor. However, we do identify strong positive epistasis among subsequent mutations that have accrued in BA.2.86 descendants. Specifically, the Q493E mutation that decreased ACE2-binding affinity in all previous SARS-CoV-2 backgrounds is reversed in sign to enhance human ACE2-binding affinity when coupled with L455S and F456L in the currently emerging KP.3 variant. Our results point to a modest degree of epistatic drift in mutational effects during recent SARS-CoV-2 evolution but highlight how these small epistatic shifts can have important consequences for the emergence of new SARS-CoV-2 variants.

[end paper 10]

[begin paper 11]

PMCID = 11754117

Title = Evolving antibody response to SARS-CoV-2 antigenic shift from XBB to JN.1

The continuous evolution of SARS-CoV-2, particularly the emergence of the BA.2.86/JN.1 lineage replacing XBB, necessitates re-evaluation of vaccine compositions. Here, we provide a comprehensive analysis of the humoral immune response to XBB and JN.1 human exposure. We demonstrate the antigenic distinctiveness of XBB and JN.1 lineages in SARS-CoV-2-naive individuals and show that infection with JN.1 elicits superior plasma neutralization against its subvariants. We highlight the strong immune evasion and receptor-binding capability of KP.3, supporting its foreseeable prevalence. Extensive analysis of the B cell receptor repertoire, in which we isolate approximately 2,000 receptor-binding-domain-specific antibodies, with targeting epitopes characterized by deep mutational scanning, underscores the superiority of JN.1-elicited memory B cells. Class 1 IGHV3-53/3-66-derived neutralizing antibodies (NAbs) are important contributors to the wild-type reactivity of NAbs against JN.1. However, KP.2 and KP.3 evade a substantial subset of these antibodies, even those induced by JN.1, supporting a need for booster updates. JN.1-induced Omicron-specific antibodies also demonstrate high potency across Omicron. Escape hotspots for these NAbs have already been mutated, resulting in a higher immune barrier to escape and indicating probable recovery of escaped NAbs. In addition, the prevalence of IGHV3-53/3-66-derived antibodies and their ability to compete with all Omicron-specific NAbs suggests that they have an inhibitory effect on the activation of Omicron-specific naive B cells, potentially explaining the heavy immune imprinting in mRNA-vaccinated individuals. These findings delineate the evolving antibody response to the antigenic shift of Omicron from XBB to JN.1 and highlight the importance of developing the JN.1 lineage, especially KP.2- and KP.3-based vaccine boosters.

A deep mutational scanning approach identifies mutations that allow subvariants of the prevalent JN.1 SARS-CoV-2 strain to evade Omicron-induced neutralizing antibodies, indicating a need for careful monitoring of this lineage in vaccine development.

[end paper 11]

[begin paper 12]

PMCID = 11291116

Title = Deep mutational scanning of SARS-CoV-2 Omicron BA.2.86 and epistatic emergence of the KP.3 variant

Deep mutational scanning experiments aid in the surveillance and forecasting of viral evolution by providing prospective measurements of mutational effects on viral traits, but epistatic shifts in the impacts of mutations can hinder viral forecasting when measurements were made in outdated strain backgrounds. Here, we report measurements of the impact of all single amino acid mutations on ACE2-binding affinity and protein folding and expression in the SARS-CoV-2 Omicron BA.2.86 spike receptor-binding domain (RBD). As with other SARS-CoV-2 variants, we find a plastic and evolvable basis for receptor binding, with many mutations at the ACE2 interface maintaining or even improving ACE2-binding affinity. Despite its large genetic divergence, mutational effects in BA.2.86 have not diverged greatly from those measured in its Omicron BA.2 ancestor. However, we do identify strong positive epistasis among subsequent mutations that have accrued in BA.2.86 descendants. Specifically, the Q493E mutation that decreased ACE2-binding affinity in all previous SARS-CoV-2 backgrounds is reversed in sign to enhance human ACE2-binding affinity when coupled with L455S and F456L in the currently emerging KP.3 variant. Our results point to a modest degree of epistatic drift in mutational effects during recent SARS-CoV-2 evolution but highlight how these small epistatic shifts can have important consequences for the emergence of new SARS-CoV-2 variants.

[end paper 12]

[begin paper 13]

PMCID = 11630572

Title = Discovery of Nanosota-9 as anti-Omicron nanobody therapeutic candidate

Omicron subvariants of SARS-CoV-2 continue to pose a significant global health threat. Nanobodies, single-domain antibodies derived from camelids, are promising therapeutic tools against pandemic viruses due to their favorable properties. In this study, we identified a novel nanobody, Nanosota-9, which demonstrates high potency against a wide range of Omicron subvariants both in vitro and in a mouse model. Cryo-EM data revealed that Nanosota-9 neutralizes Omicron through a unique mechanism: two Nanosota-9 molecules crosslink two receptor-binding domains (RBDs) of the trimeric Omicron spike protein, preventing the RBDs from binding to the ACE2 receptor. This mechanism explains its strong anti-Omicron potency. Additionally, the Nanosota-9 binding epitopes on the spike protein are relatively conserved among Omicron subvariants, contributing to its broad anti-Omicron spectrum. Combined with our recently developed structure-guided in vitro evolution approach for nanobodies, Nanosota-9 has the potential to serve as the foundation for a superior anti-Omicron therapeutic.

Omicron continues to evolve and spread among humans, making the development of effective and broad-spectrum treatments a top priority for global health. Nanobodies, which are single-domain antibodies derived from camelids, show great promise due to their small size and ability to target conserved, hidden regions on the virus. In this study, we identified a novel nanobody called Nanosota-9, which effectively inhibits most major Omicron subvariants. Our cryo-EM analysis revealed a unique structural mechanism by which Nanosota-9 neutralizes Omicron. Like many other anti-SARS-CoV-2 antibodies and nanobodies, Nanosota-9 binds to the receptor-binding domain (RBD) of Omicron spike proteins, blocking receptor binding. However, unlike others, two Nanosota-9 molecules crosslink two RBDs of the trimeric spike, enhancing its anti-Omicron potency. One of the two binding sites on the RBD that Nanosota-9 targets significantly overlaps with the receptor-binding site and is relatively conserved, while the other site is fully conserved, expanding its anti-Omicron spectrum. Conventional antibodies cannot access the same binding sites on the spike protein as Nanosota-9, highlighting its unique neutralization mechanism. Combined with our newly developed structure-guided in vitro evolution approach for nanobodies, Nanosota-9 has strong potential as a superior anti-Omicron therapeutic.

[end paper 13]

[begin paper 14]

PMCID = 11406931

Title = Ipsilateral or contralateral boosting of mice with mRNA vaccines confers equivalent immunity and protection against a SARS-CoV-2 Omicron strain

Boosting with mRNA vaccines encoding variant-matched spike proteins has been implemented to mitigate their reduced efficacy against emerging SARS-CoV-2 variants. Nonetheless, in humans, it remains unclear whether boosting in the ipsilateral or contralateral arm with respect to the priming doses impacts immunity and protection. Here, we boosted K18-hACE2 mice with either monovalent mRNA-1273 (Wuhan-1 spike) or bivalent mRNA-1273.214 (Wuhan-1 + BA.1 spike) vaccine in the ipsilateral or contralateral leg after a two-dose priming series with mRNA-1273. Boosting in the ipsilateral or contralateral leg elicited equivalent levels of serum IgG and neutralizing antibody responses against Wuhan-1 and BA.1. While contralateral boosting with mRNA vaccines resulted in the expansion of spike-specific B and T cells beyond the ipsilateral draining lymph node (DLN) to the contralateral DLN, administration of a third mRNA vaccine dose at either site resulted in similar levels of antigen-specific germinal center B cells, plasmablasts/plasma cells, T follicular helper cells, and CD8+ T cells in the DLNs and the spleen. Furthermore, ipsilateral and contralateral boosting with mRNA-1273 or mRNA-1273.214 vaccines conferred similar homologous or heterologous immune protection against SARS-CoV-2 BA.1 virus challenge with equivalent reductions in viral RNA and infectious virus in the nasal turbinates and lungs. Collectively, our data show limited differences in B and T cell immune responses after ipsilateral and contralateral site boosting by mRNA vaccines that do not substantively impact protection against an Omicron strain.

Sequential boosting with mRNA vaccines has been an effective strategy to overcome waning immunity and neutralization escape by emerging SARS-CoV-2 variants. However, it remains unclear how the site of boosting relative to the primary vaccination series shapes optimal immune responses or breadth of protection against variants. In K18-hACE2 transgenic mice, we observed that intramuscular boosting with historical monovalent or variant-matched bivalent vaccines in the ipsilateral or contralateral limb elicited comparable levels of serum spike-specific antibody and antigen-specific B and T cell responses. Moreover, boosting on either side conferred equivalent protection against a SARS-CoV-2 Omicron challenge strain. Our data in mice suggest that the site of intramuscular boosting with an mRNA vaccine does not substantially impact immunity or protection against SARS-CoV-2 infection.

[end paper 14]

[begin paper 15]

PMCID = 11312261

Title = Comprehensive Review of COVID-19: Epidemiology, Pathogenesis, Advancement in Diagnostic and Detection Techniques, and Post-Pandemic Treatment Strategies

At present, COVID-19 remains a public health concern due to the ongoing evolution of SARS-CoV-2 and its prevalence in particular countries. This paper provides an updated overview of the epidemiology and pathogenesis of COVID-19, with a focus on the emergence of SARS-CoV-2 variants and the phenomenon known as ‘long COVID’. Meanwhile, diagnostic and detection advances will be mentioned. Though many inventions have been made to combat the COVID-19 pandemic, some outstanding ones include multiplex RT-PCR, which can be used for accurate diagnosis of SARS-CoV-2 infection. ELISA-based antigen tests also appear to be potential diagnostic tools to be available in the future. This paper also discusses current treatments, vaccination strategies, as well as emerging cell-based therapies for SARS-CoV-2 infection. The ongoing evolution of SARS-CoV-2 underscores the necessity for us to continuously update scientific understanding and treatments for it.

[end paper 15]

[begin paper 16]

PMCID = 11358953

Title = Subsequent Waves of Convergent Evolution in SARS-CoV-2 Genes and Proteins

Beginning in 2022, following widespread infection and vaccination among the global population, the SARS-CoV-2 virus mainly evolved to evade immunity derived from vaccines and past infections. This review covers the convergent evolution of structural, nonstructural, and accessory proteins in SARS-CoV-2, with a specific look at common mutations found in long-lasting infections that hint at the virus potentially reverting to an enteric sarbecovirus type.

[end paper 16]

[begin paper 17]

PMCID = 11593367

Title = SARS-CoV-2 Infection and Alpha-Synucleinopathies: Potential Links and Underlying Mechanisms

Alpha-synuclein (α-syn) is a 140-amino-acid, intrinsically disordered, soluble protein that is abundantly present in the brain. It plays a crucial role in maintaining cellular structures and organelle functions, particularly in supporting synaptic plasticity and regulating neurotransmitter turnover. However, for reasons not yet fully understood, α-syn can lose its physiological role and begin to aggregate. This altered α-syn disrupts dopaminergic transmission and causes both presynaptic and postsynaptic dysfunction, ultimately leading to cell death. A group of neurodegenerative diseases known as α-synucleinopathies is characterized by the intracellular accumulation of α-syn deposits in specific neuronal and glial cells within certain brain regions. In addition to Parkinson’s disease (PD), these conditions include dementia with Lewy bodies (DLBs), multiple system atrophy (MSA), pure autonomic failure (PAF), and REM sleep behavior disorder (RBD). Given that these disorders are associated with α-syn-related neuroinflammation—and considering that SARS-CoV-2 infection has been shown to affect the nervous system, with COVID-19 patients experiencing neurological symptoms—it has been proposed that COVID-19 may contribute to neurodegeneration in PD and other α-synucleinopathies by promoting α-syn misfolding and aggregation. In this review, we focus on whether SARS-CoV-2 could act as an environmental trigger that facilitates the onset or progression of α-synucleinopathies. Specifically, we present new evidence on the potential role of SARS-CoV-2 in modulating α-syn function and discuss the causal relationship between SARS-CoV-2 infection and the development of parkinsonism-like symptoms.

[end paper 17]

[begin paper 18]

PMCID = 11461813

Title = Lineage-specific pathogenicity, immune evasion, and virological features of SARS-CoV-2 BA.2.86/JN.1 and EG.5.1/HK.3

SARS-CoV-2 JN.1 with an additional L455S mutation on spike when compared with its parental variant BA.2.86 has outcompeted all earlier variants to become the dominant circulating variant. Recent studies investigated the immune resistance of SARS-CoV-2 JN.1 but additional factors are speculated to contribute to its global dominance, which remain elusive until today. Here, we find that SARS-CoV-2 JN.1 has a higher infectivity than BA.2.86 in differentiated primary human nasal epithelial cells (hNECs). Mechanistically, we demonstrate that the gained infectivity of SARS-CoV-2 JN.1 over BA.2.86 associates with increased entry efficiency conferred by L455S and better spike cleavage in hNECs. Structurally, S455 altered the mode of binding of JN.1 spike protein to ACE2 when compared to BA.2.86 spike at ACE2H34, and modified the internal structure of JN.1 spike protein by increasing the number of hydrogen bonds with neighboring residues. These findings indicate that a single mutation (L455S) enhances virus entry in hNECs and increases immune evasiveness, which contribute to the robust transmissibility of SARS-CoV-2 JN.1. We further evaluate the in vitro and in vivo virological characteristics between SARS-CoV-2 BA.2.86/JN.1 and EG.5.1/HK.3, and identify key lineage-specific features of the two Omicron sublineages that contribute to our understanding on Omicron antigenicity, transmissibility, and pathogenicity.

Here, the authors show that a single mutation in JN.1 spike, L455S, confers better spike cleavage and enhances virus infectivity in differentiated primary human nasal epithelial cells while increasing immune evasiveness, contributing to the efficient spread of JN.1.

[end paper 18]

[begin paper 19]

PMCID = 11476394

Title = In Silico Design of miniACE2 Decoys with In Vitro Enhanced Neutralization Activity against SARS-CoV-2, Encompassing Omicron Subvariants

The COVID-19 pandemic has overwhelmed healthcare systems and triggered global economic downturns. While vaccines have reduced the lethality rate of SARS-CoV-2 to 0.9% as of October 2024, the continuous evolution of variants remains a significant public health challenge. Next-generation medical therapies offer hope in addressing this threat, especially for immunocompromised individuals who experience prolonged infections and severe illnesses, contributing to viral evolution. These cases increase the risk of new variants emerging. This study explores miniACE2 decoys as a novel strategy to counteract SARS-CoV-2 variants. Using in silico design and molecular dynamics, blocking proteins (BPs) were developed with stronger binding affinity for the receptor-binding domain of multiple variants than naturally soluble human ACE2. The BPs were expressed in E. coli and tested in vitro, showing promising neutralizing effects. Notably, miniACE2 BP9 exhibited an average IC50 of 4.9 µg/mL across several variants, including the Wuhan strain, Mu, Omicron BA.1, and BA.2 This low IC50 demonstrates the potent neutralizing ability of BP9, indicating its efficacy at low concentrations.Based on these findings, BP9 has emerged as a promising therapeutic candidate for combating SARS-CoV-2 and its evolving variants, thereby positioning it as a potential emergency biopharmaceutical.

[end paper 19]

[begin paper 20]

PMCID = 11769326

Title = SARS-CoV-2 Evolution: Implications for Diagnosis, Treatment, Vaccine Effectiveness and Development

The COVID-19 pandemic, driven by the rapid evolution of the SARS-CoV-2 virus, presents ongoing challenges to global public health. SARS-CoV-2 is characterized by rapidly evolving mutations, especially in (but not limited to) the spike protein, complicating predictions about its evolutionary trajectory. These mutations have significantly affected transmissibility, immune evasion, and vaccine efficacy, leading to multiple pandemic waves with over half a billion cases and seven million deaths globally. Despite several strategies, from rapid vaccine development and administration to the design and availability of antivirals, including monoclonal antibodies, already having been employed, the persistent circulation of the virus and the emergence of new variants continue to result in high case numbers and fatalities. In the past four years, immense research efforts have contributed much to our understanding of the viral pathogenesis mechanism, the COVID-19 syndrome, and the host–microbe interactions, leading to the development of effective vaccines, diagnostic tools, and treatments. The focus of this review is to provide a comprehensive analysis of the functional impact of mutations on diagnosis, treatments, and vaccine effectiveness. We further discuss vaccine safety in pregnancy and the implications of hybrid immunity on long-term protection against infection, as well as the latest developments on a pan-coronavirus vaccine and nasal formulations, emphasizing the need for continued surveillance, research, and adaptive public health strategies in response to the ongoing SARS-CoV-2 evolution race.

[end paper 20]

[begin paper 21]

PMCID = 11481514

Title = Molecular and structural insights into SARS-CoV-2 evolution: from BA.2 to XBB subvariants

Due to the incessant emergence of various SARS-CoV-2 variants with enhanced fitness in the human population, controlling the COVID-19 pandemic has been challenging. Understanding how the virus enhances its fitness during a pandemic could offer valuable insights for more effective control of viral epidemics. In this manuscript, we review the evolution of SARS-CoV-2 from early 2022 to the end of 2023—from Omicron BA.2 to XBB descendants. Focusing on viral evolution during this period, we provide concrete examples that SARS-CoV-2 has increased its fitness by enhancing several functions of the spike (S) protein, including its binding affinity to the ACE2 receptor and its ability to evade humoral immunity. Furthermore, we explore how specific mutations modify these functions of the S protein through structural alterations. This review provides evolutionary, molecular, and structural insights into how SARS-CoV-2 has increased its fitness and repeatedly caused epidemic surges during the pandemic.

[end paper 21]

[begin paper 22]

PMCID = 11776705

Title = SARS-CoV-2 CoCoPUTs: analyzing GISAID and NCBI data to obtain codon statistics, mutations, and free energy over a multiyear period

A consistent area of interest since the beginning of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) pandemic has been the sequence composition of the virus and how it has changed over time. Many resources have been developed for the storage and analysis of SARS-CoV-2 data, such as GISAID (Global Initiative on Sharing All Influenza Data), NCBI, Nextstrain, and outbreak.info. However, relatively little has been done to compile codon usage data, codon-level mutation data, and secondary structure data into a single database. Here, we assemble the aforementioned data and many additional virus attributes in a new database entitled SARS-CoV-2 CoCoPUTs. We begin with an overview of the composition and overlap between two of the largest sources of SARS-CoV-2 sequence data: GISAID and NCBI Virus (GenBank). We then evaluate different types of sequence curation strategies to reduce the dataset of millions of sequences to only one sequence per Pango lineage variant. We then performed specific analyses on the coding sequences (CDSs), including calculating codon usage, codon pair usage, dinucleotides, junction dinucleotides, mutations, GC content, effective number of codons (ENCs), and effective number of codon pairs (ENCPs). We have also performed whole-genome secondary RNA structure prediction calculations for each variant, using the LinearPartition software and modified selective 2ʹ-hydroxyl acylation analyzed by primer extension (SHAPE) data that are available online. Finally, we compiled all the data into our resource, SARS-CoV-2 CoCoPUTs, and paired many of the resulting statistics with variant proportion data over time in order to derive trends in viral evolution. Although the overall codon usage of SARS-CoV-2 did not change drastically, in line with the previous literature on this subject, we did observe that while overall GC% content decreased, GC% of the third position in the codon was more positive relative to overall GC% content between February 2021 and July 2023. Over the same interval, we noted that both synonymous and nonsynonymous mutations increased in number, with nonsynonymous mutations outpacing synonymous mutations at a rate of 3:1. We noted that the predicted whole-genome secondary structures nearly all contained the previously described virus-activated inhibitor of translation (VAIT) stem loops, validating for the first time their existence in a whole-genome secondary structure prediction for many SARS-CoV-2 variants (as opposed to previous local secondary structure predictions). We also separately produced a synonymous mutation-deprived set of SARS-CoV-2 variant sequences and repeated the secondary structure calculations on this set. This revealed an interesting trend of reduced ensemble free energy compared to the unaltered variant structures, indicating that synonymous mutations play a role in increasing the free energy of viral RNA molecules. These data both validate previous studies describing increases in viral free energy in human viruses over time and indicate a possible role for synonymous mutations in viral biology.

[end paper 22]

[begin paper 23]

PMCID = 11688500

Title = Evolution of SARS-CoV-2 spike trimers towards optimized heparan sulfate cross-linking and inter-chain mobility

The heparan sulfate (HS)-rich extracellular matrix (ECM) serves as an initial interaction site for the homotrimeric spike (S) protein of SARS-CoV-2 to facilitate subsequent docking to angiotensin-converting enzyme 2 (ACE2) receptors and cellular infection. More recent variants, notably Omicron, have evolved by swapping several amino acids to positively charged residues to enhance the interaction of the S-protein trimer with the negatively charged HS. However, these enhanced interactions may reduce Omicron’s ability to move through the HS-rich ECM to effectively find ACE2 receptors and infect cells, raising the question of how to mechanistically explain HS-associated viral movement. In this work, we show that Omicron S proteins have evolved to balance HS interaction stability and dynamics, resulting in enhanced mobility on an HS-functionalized artificial matrix. This property is achieved by the ability of Omicron S-proteins to cross-link at least two HS chains, allowing direct S-protein switching between chains as a prerequisite for cell surface mobility. Optimized HS interactions can be targeted pharmaceutically, as an HS mimetic significantly suppressed surface binding and cellular infection specifically of the Omicron variant. These findings suggest a robust way to interfere with SARS-CoV-2 Omicron infection and potentially future variants.

The online version contains supplementary material available at 10.1038/s41598-024-84276-5.

[end paper 23]

[begin paper 24]

PMCID = 11374805

Title = Structural basis for the evolution and antibody evasion of SARS-CoV-2 BA.2.86 and JN.1 subvariants

The Omicron subvariants of SARS-CoV-2, especially for BA.2.86 and JN.1, have rapidly spread across multiple countries, posing a significant threat in the ongoing COVID-19 pandemic. Distinguished by 34 additional mutations on the Spike (S) protein compared to its BA.2 predecessor, the implications of BA.2.86 and its evolved descendant, JN.1 with additional L455S mutation in receptor-binding domains (RBDs), are of paramount concern. In this work, we systematically examine the neutralization susceptibilities of SARS-CoV-2 Omicron subvariants and reveal the enhanced antibody evasion of BA.2.86 and JN.1. We also determine the cryo-EM structures of the trimeric S proteins from BA.2.86 and JN.1 in complex with the host receptor ACE2, respectively. The mutations within the RBDs of BA.2.86 and JN.1 induce a remodeling of the interaction network between the RBD and ACE2. The L455S mutation of JN.1 further induces a notable shift of the RBD–ACE2 interface, suggesting the notably reduced binding affinity of JN.1 than BA.2.86. An analysis of the broadly neutralizing antibodies possessing core neutralizing epitopes reveals the antibody evasion mechanism underlying the evolution of Omicron BA.2.86 subvariant. In general, we construct a landscape of evolution in virus-receptor of the circulating Omicron subvariants.

Here, the authors evaluate the neutralization susceptibilities of BA.2.86 and JN.1 and determine the structural basis for binding to the ACE2 receptor and monoclonal neutralizing antibodies, revealing their evolution and antibody evasion mechanism.

[end paper 24]

[begin paper 25]

PMCID = 11520098

Title = Omicron-specific ultra-potent SARS-CoV-2 neutralizing antibodies targeting the N1/N2 loop of Spike N-terminal domain

A multitude of functional mutations continue to emerge on the N-terminal domain (NTD) of the spike protein in SARS-CoV-2 Omicron subvariants. Understanding the immunogenicity of Omicron NTD and the properties of antibodies elicited by it is crucial for comprehending the impact of NTD mutations on viral fitness and guiding vaccine design. In this study, we find that most of NTD-targeting antibodies isolated from individuals with BA.5/BF.7 breakthrough infection (BTI) are ancestral (wild-type or WT)-reactive and non-neutralizing. Surprisingly, we identified five ultra-potent neutralizing antibodies (NAbs) that can only bind to Omicron but not WT NTD. Structural analysis revealed that they bind to a unique epitope on the N1/N2 loop of NTD and interact with the receptor-binding domain (RBD) via the light chain. These Omicron-specific NAbs achieve neutralization through ACE2 competition and blockage of ACE2-mediated S1 shedding. However, BA.2.86 and BA.2.87.1, which carry insertions or deletions on the N1/N2 loop, can evade these antibodies. Together, we provided a detailed map of the NTD-targeting antibody repertoire in the post-Omicron era, demonstrating their vulnerability to NTD mutations enabled by its evolutionary flexibility, despite their potent neutralization. These results revealed the function of the indels in the NTD of BA.2.86/JN.1 sublineage in evading neutralizing antibodies and highlighted the importance of considering the immunogenicity of NTD in vaccine design.

[end paper 25]

[begin paper 26]

PMCID = 11661108

Title = Defining a highly conserved B cell epitope in the receptor binding motif of SARS-CoV-2 spike glycoprotein

SARS-CoV-2 mRNA vaccines induce robust and persistent germinal centre (GC) B cell responses in humans. It remains unclear how the continuous evolution of the virus impacts the breadth of the induced GC B cell response. Using ultrasound-guided fine needle aspiration, we examined draining lymph nodes of nine healthy adults following bivalent booster immunization. We show that 77.8% of the B cell clones in the GC expressed as representative monoclonal antibodies recognized the spike protein, with a third (37.8%) of these targeting the receptor binding domain (RBD). Strikingly, only one RBD-targeting mAb, mAb-52, neutralized all tested SARS-CoV-2 strains, including the recent KP.2 variant. mAb-52 utilizes the IGHV3–66 public clonotype, protects hamsters challenged against the EG.5.1 variant and targets the class I/II RBD epitope, closely mimicking the binding footprint of ACE2. Finally, we show that the remarkable breadth of mAb-52 is due to the somatic hypermutations accumulated within vaccine-induced GC reaction.

Booster SARS-CoV-2 mRNA vaccine recruits and broadens GC B cell responses targeting a highly conserved site on receptor binding domain of spike glycoprotein.

[end paper 26]

[begin paper 27]

PMCID = 11272943

Title = Universal subunit vaccine protects against multiple SARS-CoV-2 variants and SARS-CoV

Although Omicron RBD of SARS-CoV-2 accumulates many mutations, the backbone region (truncated RBD) of spike protein is highly conserved. Here, we designed several subunit vaccines by keeping the conserved spike backbone region of SARS-CoV-2 Omicron BA.1 subvariant (S-6P-no-RBD), or inserting the RBD of Delta variant (S-6P-Delta-RBD), Omicron (BA.5) variant (S-6P-BA5-RBD), or ancestral SARS-CoV-2 (S-6P-WT-RBD) to the above backbone construct, and evaluated their ability to induce immune responses and cross-protective efficacy against various SARS-CoV-2 variants and SARS-CoV. Among the four subunit vaccines, S-6P-Delta-RBD protein elicited broad and potent neutralizing antibodies against all SARS-CoV-2 variants tested, including Alpha, Beta, Gamma, and Delta variants, the BA.1, BA.2, BA.2.75, BA.4.6, and BA.5 Omicron subvariants, and the ancestral strain of SARS-CoV-2. This vaccine prevented infection and replication of SARS-CoV-2 Omicron, and completely protected immunized mice against lethal challenge with the SARS-CoV-2 Delta variant and SARS-CoV. Sera from S-6P-Delta-RBD-immunized mice protected naive mice against challenge with the Delta variant, with significantly reduced viral titers and without pathological effects. Protection correlated positively with the serum neutralizing antibody titer. Overall, the designed vaccine has potential for development as a universal COVID-19 vaccine and/or a pan-sarbecovirus subunit vaccine that will prevent current and future outbreaks caused by SARS-CoV-2 variants and SARS-related CoVs.

[end paper 27]

[begin paper 28]

PMCID = 11407393

Title = Recurrent SARS-CoV-2 spike mutations confer growth advantages to select JN.1 sublineages

The recently dominant SARS-CoV-2 Omicron JN.1 has evolved into multiple sublineages, with recurrent spike mutations R346T, F456L, and T572I, some of which exhibit growth advantages, such as KP.2 and KP.3. We investigated these mutations in JN.1, examining their individual and combined effects on immune evasion, ACE2 receptor affinity, and in vitro infectivity. F456L increased resistance to neutralization by human sera, including those after JN.1 breakthrough infections, and by RBD class-1 monoclonal antibodies, significantly altering JN.1 antigenicity. R346T enhanced ACE2-binding affinity and modestly boosted the infectivity of JN.1 pseudovirus, without a discernible effect on serum neutralization, while T572I slightly bolstered evasion of SD1-directed mAbs against JN.1's ancestor, BA.2, possibly by altering SD1 conformation. Importantly, expanding sublineages such as KP.2 containing R346T, F456L, and V1104L, showed similar neutralization resistance as JN.1 with R346T and F456L, suggesting V1104L does not appreciably affect antibody evasion. Furthermore, the hallmark mutation Q493E in KP.3 significantly reduced ACE2-binding affinity and viral infectivity, without noticeably impacting serum neutralization. Our findings illustrate how certain JN.1 mutations confer growth advantages in the population and could inform the design of the next COVID-19 vaccine booster.

[end paper 28]

[begin paper 29]

PMCID = 11724309

Title = High resistance barrier and prophylactic protection in preclinical models of SARS-CoV-2 with two siRNA combination

RNA interference is a natural antiviral mechanism that could be harnessed to combat SARS-CoV-2 infection by targeting and destroying the viral RNA. We identified potent lipophilic small interfering RNA (siRNA) conjugates targeting highly conserved regions of SARS-CoV-2 outside of the spike-encoding region capable of achieving ≥3-log viral reduction. Serial passaging studies demonstrated that a two-siRNA combination prevented development of resistance compared to a single siRNA approach. Viral resistance to single siRNA treatment occurred due to emergence of point mutations at critical positions required for siRNA-mediated target binding and cleavage, which led to a loss of siRNA efficacy. With a two-siRNA combination, emergence of mutations within the siRNA binding site was abolished. When delivered intranasally, two-siRNA combination protected Syrian hamsters from weight loss and lung pathology by viral infection upon prophylactic administration but not following onset of infection. Together, the data support potential utility of RNAi as a prophylactic approach with high resistance barrier to counteract SARS-CoV-2 emergent variants and complement vaccination. Most importantly, given that the siRNAs can be rapidly developed from a new pathogen sequence, this strategy has implications as a new type of preventive medicine that may protect against future coronavirus pandemics.

[end paper 29]

[begin paper 30]

PMCID = 11728909

Title = Ancestral SARS-CoV-2 immune imprinting persists on RBD but not NTD after sequential Omicron infections

Whether Omicron exposures could overcome ancestral SARS-CoV-2 immune imprinting remains controversial. Here we analyzed B cell responses evoked by sequential Omicron infections in vaccinated and unvaccinated individuals. Plasma neutralizing antibody titers against ancestral SARS-CoV-2 and variants indicate that immune imprinting is not consistently induced by inactivated or recombinant protein vaccines. However, once robustly induced, immune imprinting is not countered by successive Omicron challenges. We compared binding specificities, neutralizing capacities, developing origins and targeting epitopes of monoclonal antibodies from those individuals. Although receptor-binding domain (RBD) and N-terminal domain (NTD) of spike are both primary targets for neutralizing antibodies, immune imprinting only shapes antibody responses to RBD by impeding the production of Omicron-specific neutralizing antibodies while facilitating the development of broadly neutralizing antibodies. We propose that immune imprinting can be either neglected by NTD-based vaccines to induce variant-specific antibodies or leveraged by RBD-containing vaccines to induce broadly neutralizing antibodies.

Immune imprinting is not consistently induced by vaccination

Repeated Omicron infections could not counter immune imprinting

Immune imprinting shapes antibody responses to RBD but not NTD

Immune imprinting leads to broadly neutralizing antibodies

Health sciences; Medicine; Medical specialty; Immunology; Natural sciences; Biological sciences; Immunology; Immune response; Microbiology; Virology

[end paper 30]

[begin paper 31]

PMCID = 11275766

Title = Dissecting human monoclonal antibody responses from mRNA- and protein-based XBB.1.5 COVID-19 monovalent vaccines

The emergence of highly contagious and immune-evasive severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) variants has required reformulation of coronavirus disease 2019 (COVID-19) vaccines to target those new variants specifically. While previous infections and booster vaccinations can enhance variant neutralization, it is unclear whether the monovalent version, administered using either mRNA or protein-based vaccine platforms, can elicit de novo B-cell responses specific for Omicron XBB.1.5 variants. Here, we dissected the genetic antibody repertoire of 603 individual plasmablasts derived from five individuals who received a monovalent XBB.1.5 vaccination either with mRNA (Moderna or Pfizer/BioNtech) or adjuvanted protein (Novavax). From these sequences, we expressed 100 human monoclonal antibodies and determined binding, affinity and protective potential against several SARS-CoV-2 variants, including JN.1. We then select two vaccine-induced XBB.1.5 mAbs, M2 and M39. M2 mAb was a de novo, antibody, i.e., specific for XBB.1.5 but not ancestral SARS-CoV-2. M39 bound and neutralized both XBB.1.5 and JN.1 strains. Our high-resolution cryo-electron microscopy (EM) structures of M2 and M39 in complex with the XBB.1.5 spike glycoprotein defined the epitopes engaged and revealed the molecular determinants for the mAbs’ specificity. These data show, at the molecular level, that monovalent, variant-specific vaccines can elicit functional antibodies, and shed light on potential functional and genetic differences of mAbs induced by vaccinations with different vaccine platforms.\

[end paper 31]

[begin paper 32]

PMCID = 11601794

Title = Rapid response to fast viral evolution using AlphaFold 3-assisted topological deep learning

The fast evolution of SARS-CoV-2 and other infectious viruses poses a grand challenge to the rapid response in terms of viral tracking, diagnostics, and design and manufacture of monoclonal antibodies (mAbs) and vaccines, which are both time-consuming and costly. This underscores the need for efficient computational approaches. Recent advancements, like topological deep learning (TDL), have introduced powerful tools for forecasting emerging dominant variants, yet they require deep mutational scanning (DMS) of viral surface proteins and associated three-dimensional (3D) protein-protein interaction (PPI) complex structures. We propose an AlphaFold 3 (AF3)-assisted multi-task topological Laplacian (MT-TopLap) strategy to address this need. MT-TopLap combines deep learning with topological data analysis (TDA) models, such as persistent Laplacians (PL) to extract detailed topological and geometric characteristics of PPIs, thereby enhancing the prediction of DMS and binding free energy (BFE) changes upon virus mutations. Validation with four experimental DMS datasets of SARS-CoV-2 spike receptor-binding domain (RBD) and the human angiotensin-converting enzyme-2 (ACE2) complexes indicates that our AF3 assisted MT-TopLap strategy maintains robust performance, with only an average 1.1% decrease in Pearson correlation coefficients (PCC) and an average 9.3% increase in root mean square errors (RMSE), compared with the use of experimental structures. Additionally, AF3-assisted MT-TopLap achieved a PCC of 0.81 when tested with a SARS-CoV-2 HK.3 variant DMS dataset, confirming its capability to accurately predict BFE changes and adapt to new experimental data, thereby showcasing its potential for rapid and effective response to fast viral evolution.

[end paper 32]

[begin paper 33]

PMCID = 11763035

Title = Pseudotyped Viruses: A Useful Platform for Pre-Clinical Studies Conducted in a BSL-2 Laboratory Setting

The study of pathogenic viruses has always posed significant biosafety challenges. In particular, the study of highly pathogenic viruses requires methods with low biological risk but relatively high sensitivity and convenience in detection. In recent years, pseudoviruses, which consist of a backbone of one virus and envelope proteins of another virus, have become one of the most widely used tools for exploring the mechanisms of viruses binding to cells, membrane fusion and viral entry, as well as for screening the libraries of antiviral substances, evaluating the potential of neutralizing monoclonal antibodies, developing neutralization tests, and therapeutic platforms. During the outbreak of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), pseudotyped virus-based assays played a pivotal role in advancing our understanding of virus–cell interactions and the role of its proteins in disease pathogenesis. Such tools facilitated the search for potential therapeutic agents and accelerated epidemiological studies on post-infection and post-vaccination humoral immunity. This review focuses on the use of pseudoviruses as a model for large-scale applications to study enveloped viruses.

[end paper 33]

[begin paper 34]

PMCID = 11663983

Title = A predictive language model for SARS-CoV-2 evolution

Modeling and predicting mutations are critical for COVID-19 and similar pandemic preparedness. However, existing predictive models have yet to integrate the regularity and randomness of viral mutations with minimal data requirements. Here, we develop a non-demanding language model utilizing both regularity and randomness to predict candidate SARS-CoV-2 variants and mutations that might prevail. We constructed the “grammatical frameworks” of the available S1 sequences for dimension reduction and semantic representation to grasp the model’s latent regularity. The mutational profile, defined as the frequency of mutations, was introduced into the model to incorporate randomness. With this model, we successfully identified and validated several variants with significantly enhanced viral infectivity and immune evasion by wet-lab experiments. By inputting the sequence data from three different time points, we detected circulating strains or vital mutations for XBB.1.16, EG.5, JN.1, and BA.2.86 strains before their emergence. In addition, our results also predicted the previously unknown variants that may cause future epidemics. With both the data validation and experiment evidence, our study represents a fast-responding, concise, and promising language model, potentially generalizable to other viral pathogens, to forecast viral evolution and detect crucial hot mutation spots, thus warning the emerging variants that might raise public health concern.

[end paper 34]

[begin paper 35]

PMCID = 11680146

Title = mRNA Vaccines Against COVID-19 as Trailblazers for Other Human Infectious Diseases

mRNA vaccines represent a milestone in the history of vaccinology, because they are safe, very effective, quick and cost-effective to produce, easy to adapt should the antigen vary, and able to induce humoral and cellular immunity. Methods: To date, only two COVID-19 mRNA and one RSV vaccines have been approved. However, several mRNA vaccines are currently under development for the prevention of human viral (influenza, human immunodeficiency virus [HIV], Epstein–Barr virus, cytomegalovirus, Zika, respiratory syncytial virus, metapneumovirus/parainfluenza 3, Chikungunya, Nipah, rabies, varicella zoster virus, and herpes simplex virus 1 and 2), bacterial (tuberculosis), and parasitic (malaria) diseases. Results: RNA viruses, such as severe acute respiratory syndrome coronavirus (SARS-CoV)-2, HIV, and influenza, are characterized by high variability, thus creating the need to rapidly adapt the vaccines to the circulating viral strain, a task that mRNA vaccines can easily accomplish; however, the speed of variability may be higher than the time needed for a vaccine to be adapted. mRNA vaccines, using lipid nanoparticles as the delivery system, may act as adjuvants, thus powerfully stimulating innate as well as adaptive immunity, both humoral, which is rapidly waning, and cell-mediated, which is highly persistent. Safety profiles were satisfactory, considering that only a slight increase in prognostically favorable anaphylactic reactions in young females and myopericarditis in young males has been observed. Conclusions: The COVID-19 pandemic determined a shift in the use of RNA: after having been used in medicine as micro-RNAs and tumor vaccines, the new era of anti-infectious mRNA vaccines has begun, which is currently in great development, to either improve already available, but unsatisfactory, vaccines or develop protective vaccines against infectious agents for which no preventative tools have been realized yet.

[end paper 35]

[begin paper 36]

PMCID = 11545109

Title = Advancing CRISPR-Based Solutions for COVID-19 Diagnosis and Therapeutics

Since the onset of the COVID-19 pandemic, a variety of diagnostic approaches, including RT-qPCR, RAPID, and LFA, have been adopted, with RT-qPCR emerging as the gold standard. However, a significant challenge in COVID-19 diagnostics is the wide range of symptoms presented by patients, necessitating early and accurate diagnosis for effective management. Although RT-qPCR is a precise molecular technique, it is not immune to false-negative results. In contrast, CRISPR-based detection methods for SARS-CoV-2 offer several advantages: they are cost-effective, time-efficient, highly sensitive, and specific, and they do not require sophisticated instruments. These methods also show promise for scalability, enabling diagnostic tests. CRISPR technology can be customized to target any genomic region of interest, making it a versatile tool with applications beyond diagnostics, including therapeutic development. The CRISPR/Cas systems provide precise gene targeting with immense potential for creating next-generation diagnostics and therapeutics. One of the key advantages of CRISPR/Cas-based therapeutics is the ability to perform multiplexing, where different sgRNAs or crRNAs can target multiple sites within the same gene, reducing the likelihood of viral escape mutants. Among the various CRISPR systems, CRISPR/Cas13 and CARVER (Cas13-assisted restriction of viral expression and readout) are particularly promising. These systems can target a broad range of single-stranded RNA viruses, making them suitable for the diagnosis and treatment of various viral diseases, including SARS-CoV-2. However, the efficacy and safety of CRISPR-based therapeutics must be thoroughly evaluated in pre-clinical and clinical settings. While CRISPR biotechnologies have not yet been fully harnessed to control the current COVID-19 pandemic, there is an optimism that the limitations of the CRISPR/Cas system can be overcome soon. This review discusses how CRISPR-based strategies can revolutionize disease diagnosis and therapeutic development, better preparing us for future viral threats.

[end paper 36]

[begin paper 37]

PMCID = 11768752

Title = Reflections on COVID-19: A Literature Review of SARS-CoV-2 Testing

Although the Coronavirus disease 2019 (COVID-19) pandemic has ended, there are still many important lessons we can learn, as the pandemic profoundly affected every area of laboratory practice. During the pandemic, extensive changes to laboratory staffing had to be implemented, as many healthcare institutions required regular screening of all healthcare staff. Several studies examined the effectiveness of different screening regimens and concluded that repeated testing, even with lower sensitivity tests, could rival the performance of gold-standard RT-PCR testing in the detection of new cases. Many assay evaluations were performed both in the earlier and later periods of the pandemic. They included both nucleocapsid/spike antibodies and automated antigen assays. Early in the pandemic, it was generally agreed that the initial nucleocapsid antibody assays had poor sensitivity when used before 14 days of disease onset, with total or IgG antibodies being preferred over the use of IgM. Spike antibody assays gradually replaced nucleocapsid antibody assays, as most people were vaccinated. Spike antibodies tracked the rise in antibodies after vaccination with mRNA vaccines and became invaluable in the assessment of vaccine response. Studies demonstrated robust antibody secretion with each vaccine dose and could last for several months post-vaccination. When antigen testing was introduced, they became effective tools to identify affected patients when used serially or in an orthogonal fashion with RT-PCR testing. Despite the numerous findings during the pandemic period, research in COVID-19 has slowed. To this day it is difficult to identify a true neutralizing antibody test for the virus. An appropriate antibody level that would confer protective immunity against the plethora of new variants remains elusive. We hope that a summary of events during the pandemic could provide important insights to consider in planning for the next viral pandemic.

[end paper 37]

[begin paper 38]

PMCID = 11707167

Title = Inferring effects of mutations on SARS-CoV-2 transmission from genomic surveillance data

New and more transmissible variants of SARS-CoV-2 have arisen multiple times over the course of the pandemic. Rapidly identifying mutations that affect transmission could improve our understanding of viral biology and highlight new variants that warrant further study. Here we develop a generic, analytical epidemiological model to infer the transmission effects of mutations from genomic surveillance data. Applying our model to SARS-CoV-2 data across many regions, we find multiple mutations that substantially affect the transmission rate, both within and outside the Spike protein. The mutations that we infer to have the largest effects on transmission are strongly supported by experimental evidence from prior studies. Importantly, our model detects lineages with increased transmission even at low frequencies. As an example, we infer significant transmission advantages for the Alpha, Delta, and Omicron variants shortly after their appearances in regional data, when they comprised only around 1-2% of sample sequences. Our model thus facilitates the rapid identification of variants and mutations that affect transmission from genomic surveillance data.

Identifying mutations in pathogen genomes that confer transmission advantages could improve understanding of viral biology and flag variants for further study. Here, the authors develop a model for identifying mutations that affect transmission rates, validate it in simulations, then apply it to SARS-CoV-2 genomic surveillance data.

[end paper 38]

[begin paper 39]

PMCID = 11686202

Title = Trivalent recombinant protein vaccine induces cross-neutralization against XBB lineage and JN.1 subvariants: preclinical and phase 1 clinical trials

The immune escape capacities of XBB variants necessitate the authorization of vaccines with these antigens. In this study, we produce three recombinant trimeric proteins from the RBD sequences of Delta, BA.5, and XBB.1.5, formulating a trivalent vaccine (Tri-Vac) with an MF59-like adjuvant at a 1:1:4 ratio. Tri-Vac demonstrates immunogenicity in female NIH mice, inducing cross-neutralization against various SARS-CoV-2 variants, including pre-Omicron and Omicron BA.2.75, BA.5, and XBB lineages. It elicits measurable antigen-specific T cell responses, germinal center B cell responses, and T follicular helper responses, effectively protecting against live Omicron XBB.1.16 challenges. Protective immunity is maintained long-term, with sustained neutralizing antibodies and T cell responses, as well as memory B cells and long-lived plasma cells observed by day 210 post-immunization. Tri-Vac also serves as a candidate booster for enhancing immunity after three doses of inactivated virus or mRNA vaccines. A phase 1 investigator-initiated trial was initiated to assess safety and immunogenicity in humans, focusing on the primary endpoint of adverse reactions within 7 days and key secondary endpoints including the geometric mean titers (GMTs) of serum neutralizing antibodies within 30 days and 6 months post-vaccination, as well as adverse events within 30 days and serious adverse events within 6 months post-vaccination. Preliminary data indicate Tri-Vac has good safety and immunogenicity, improving neutralization against multiple variants, including JN.1, in previously vaccinated individuals, highlighting its clinical potential for protecting against SARS-CoV-2 variants. The registration number of this clinical trial is ChiCTR2200067245.

SARS-CoV-2 evolves into new subtypes necessitating further vaccine development. Here the authors generate a trivalent vaccine with Receptor Binding Domains from Delta, BA.5 and XBB.1.5 variants and characterize immunogenicity and protection in mice and safety in a phase I vaccine trial in humans.

[end paper 39]

[begin paper 40]

PMCID = 11599021

Title = First-in-Human Phase I Trial to Assess the Safety and Immunogenicity of an Orf Virus-Based COVID-19 Vaccine Booster

The emergence of SARS-CoV-2 has necessitated the development of versatile vaccines capable of addressing evolving variants. Prime-2-CoV_Beta, a novel Orf virus-based COVID-19 vaccine, was developed to express the SARS-CoV-2 spike and nucleocapsid antigens. This first-in-human, phase I, dose-finding clinical trial was conducted to assess the safety, reactogenicity, and immunogenicity of Prime-2-CoV_Beta as a booster in healthy adults. From June 2022 to June 2023, 60 participants in Germany received varying doses of Prime-2-CoV_Beta. The study demonstrated a favorable safety profile, with no serious adverse events (AEs) reported. All AEs were mild (107) or moderate (10), with the most common symptoms being pain at the injection site, fatigue, and headache. Immunogenicity assessments revealed robust vaccine-induced antigen-specific immune responses. High doses notably elicited significant increases in antibodies against the spike and nucleocapsid proteins as well as neutralizing antibodies against SARS-CoV-2 and its variants. Additionally, the vaccine did not induce ORFV-neutralizing antibodies, indicating the potential for repeated administration. In conclusion, Prime-2-CoV_Beta was safe, well tolerated, and immunogenic, demonstrating potential as a broadly protective vaccine against SARS-CoV-2 and its variants. These promising results support further evaluation of higher doses and additional studies to confirm efficacy and long-term protection. This trial was registered at ClinicalTrials, NCT05389319.

[end paper 40]

[begin paper 41]

PMCID = 11317227

Title = Predictive Assessment of the Antiviral Properties of Imperata cylindrica against SARS-CoV-2

The omicron variant and its sublineages are highly contagious, and they still constitute a global source of concern despite vaccinations. Hospitalizations and mortality rates resulting from infections by these variants of concern are still common. The existing therapeutic alternatives have presented various setbacks such as low potency, poor pharmacokinetic profiles, and drug resistance. The need for alternative therapeutic options cannot be overemphasized. Plants and their phytochemicals present interesting characteristics that make them suitable candidates for the development of antiviral therapeutic agents. This study aimed to investigate the antiviral potential of Imperata cylindrica (I. cylindrica). Specifically, the objective of this study was to identify I. cylindrica phytochemicals that display inhibitory effects against SARS-CoV-2 main protease (Mpro), a highly conserved protein among coronaviruses. Molecular docking and in silico pharmacokinetic assays were used to assess 72 phytocompounds that are found in I. cylindrica as ligands and Mpro (6LU7) as the target. Only eight phytochemicals (bifendate, cylindrene, tabanone, siderin, 5-hydroxy-2-[2-(2-hydroxyphenyl)ethyl]-4H-1-benzopyran-4-one, maritimin, 5-methoxyflavone, and flavone) displayed high binding affinities with Mpro with docking scores ranging from −5.6 kcal/mol to −9.1 kcal/mol. The in silico pharmacokinetic and toxicological assays revealed that tabanone was the best and safest phytochemical for the development of an inhibitory agent against coronavirus main protease. Thus, the study served as a baseline for further in vitro and in vivo assessment of this phytochemical against Mpro of SARS-CoV-2 variants of concern to validate these in silico findings.

[end paper 41]

[begin paper 42]

PMCID = 9254363

Title = Andrographolide induces anti-SARS-CoV-2 response through host-directed mechanism: an in silico study

Aim: Considering the present alarming situation of COVID-19 pandemic, we concentrated on evaluating the efficacy of a novel natural antiviral drug-candidate andrographolide against SARS-CoV-2 through an in silico model of study. Materials & methods: Interaction of andrographolide against the major host molecules that are responsible for SARS-CoV-2 pathogenesis were determined using bio-computational tools, in other words, molecular docking, molecular dynamics simulation and pharmacodynamics–pharmacokinetics analysis. Result: Computational findings represent that andrographolide efficiently interacts with the major human–host-associated putative drug-targets of viral-entry points like furin (-10.54 kcal/mol), TMPRSS-2 (-9.50 kcal/mol), ACE2 (-8.99 kcal/mol) and Cathepsin L (-8.98 kcal/mol). Moreover, it also blocks the inflammatory regulators including TLR4-MD2 and IL-6, which promote virus-induced inflammation leading to cytokine storm in the host body. Conclusion: This work elucidates that, the candidature of andrographolide can be utilized as a potent natural agent for the therapeutic intervention of SARS-CoV-2 through host-directed treatment.

[end paper 42]

[begin paper 43]

PMCID = 11598945

Title = Factors Predicting COVID-19 Vaccine Effectiveness and Longevity of Humoral Immune Responses

The COVID-19 pandemic, caused by SARS-CoV-2, prompted global efforts to develop vaccines to control the disease. Various vaccines, including mRNA (BNT162b2, mRNA-1273), adenoviral vector (ChAdOx1, Ad26.COV2.S), and inactivated virus platforms (BBIBP-CorV, CoronaVac), elicit high-titer, protective antibodies against the virus, but long-term antibody durability and effectiveness vary. The objective of this study is to elucidate the factors that influence vaccine effectiveness (VE) and the longevity of humoral immune responses to COVID-19 vaccines through a review of the relevant literature, including clinical and real-world studies. Here, we discuss the humoral immune response to different COVID-19 vaccines and identify factors influencing VE and antibody longevity. Despite initial robust immune responses, vaccine-induced immunity wanes over time, particularly with the emergence of variants, such as Delta and Omicron, that exhibit immune escape mechanisms. Additionally, the durability of the humoral immune responses elicited by different vaccine platforms, along with the identification of essential determinants of long-term protection—like pre-existing immunity, booster doses, hybrid immunity, and demographic factors—are critical for protecting against severe COVID-19. Booster vaccinations substantially restore neutralizing antibody levels, especially against immune-evasive variants, while individuals with hybrid immunity have a more durable and potent immune response. Importantly, comorbidities such as diabetes, cardiovascular disease, chronic kidney disease, and cancer significantly reduce the magnitude and longevity of vaccine-induced protection. Immunocompromised individuals, particularly those undergoing chemotherapy and those with hematologic malignancies, have diminished humoral responses and benefit disproportionately from booster vaccinations. Age and sex also influence immune responses, with older adults experiencing accelerated antibody decline and females generally exhibiting stronger humoral responses compared to males. Understanding the variables affecting immune protection is crucial to improving vaccine strategies and predicting VE and protection against COVID-19.

[end paper 43]

[begin paper 44]

PMCID = 11681626

Title = Identification of virus-rich intermediate cells as crucial players in SARS-CoV-2 infection and differentiation dynamics of human airway epithelium

Understanding the early interactions between severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) and human airway epithelial cells is essential for unraveling viral replication and spread mechanisms. In this study, we investigated the early dynamics of airway epithelial cells during SARS-CoV-2 infection using well-differentiated human nasal and tracheal epithelial cell cultures by incorporating three publicly available single-cell RNA sequencing datasets. We identified a previously uncharacterized cell population, termed virus-rich intermediate (VRI) cells, representing an intermediate differentiation stage between basal and ciliated cells. These VRI cells exhibited high viral loads at all infection time points, strong interferon and inflammatory responses, increased mRNA expression of microvilli-related genes (PAK1, PAK4, VIL1), and suppression of apoptosis markers (BAX, CASP3) alongside increased anti-apoptotic gene expression (BCL2). Cell-cell interaction analysis revealed that VRI cells send signals to basal cells via receptor-ligand pathways such as EPHA and VEGF, likely promoting basal cell differentiation and proliferation through MAPK signaling. These findings suggest that SARS-CoV-2 utilizes VRI cells as a primary site for replication and spread, leveraging these cells’ unique differentiation state to evade host cell death and facilitate viral propagation. This study provides insights into the early cellular responses to SARS-CoV-2 infection and highlights potential therapeutic targets to limit viral spread.

[end paper 44]

[begin paper 45]

PMCID = 11433956

Title = Analysis of Cultured Gut Microbiota Using MALDI-TOF MS in COVID-19 Patients from Serbia during the Predominance of the SARS-CoV-2 Omicron Variant

The currently dominant SARS-CoV-2 omicron variant, while causing mild respiratory symptoms, exhibits high transmissibility, drug resistance, and immune evasion. We investigated whether the presence of the SARS-CoV-2 affected the dynamics of fecal microbial composition isolated in culture in moderate COVID-19 patients. Blood, stool, and medical records were collected from 50 patients with confirmed SARS-CoV-2 infection. Two samples were taken per patient, at disease onset (within 5 days) and after symptom resolution (30–35 days). The part of the gut microbiota identifiable using MALDI-TOF MS was analyzed, and inflammatory cytokines and blood markers were measured in serum. The analysis identified 566 isolates at the species level, including 83 bacterial and 9 fungal species. Our findings indicate a change in the gut microbiota composition isolated in culture during the initial phase of infection, characterized by the proliferation of opportunistic bacteria such as Enterococcus spp. and Citrobacter spp., at the expense of beneficial commensal bacteria from the genus Bacillus and Lactobacillus. Additionally, the enrichment of fungal pathogens in fecal samples collected 30 days after the cessation of disease symptoms might suggest a prolonged disruption of the gut microbiota even after the resolution of COVID-19 symptoms. This study contributes to a growing body of evidence on the systemic effects of SARS-CoV-2 and highlights the importance of considering gastrointestinal involvement in the management and treatment of COVID-19.

[end paper 45]

[begin paper 46]

PMCID = 11695372

Title = Development of a two-component recombinant vaccine for COVID-19

Though COVID-19 as a public health emergency of international concern (PHEIC) was declared to be ended by the WHO, it continues to pose a significant threat to human society. Vaccination remains one of the most effective methods for preventing COVID-19. While most of the antigenic regions are found in the receptor binding domain (RBD), the N-terminal domain (NTD) of the S protein is another crucial region for inducing neutralizing antibodies (nAbs) against COVID-19.

In the two-dose immunization experiment, female BALB/c mice were intramuscularly immunized with different ratios of RBD-Fc and NTD-Fc proteins, with a total protein dose of 8 μg per mouse. Mice were immunized on day 0 and boosted on day 7. In the sequential immunization experiment, groups of female BALB/c mice were immunized with two doses of the inactivated SARS-CoV-2 vaccine (prototype strain) on day 0 and 7. On day 28, mice were boosted with RBD-Fc, NTD-Fc, RBD-Fc/NTD-Fc (9:1), RBD-Fc/NTD-Fc (3:1), inactivated SARS-CoV-2 vaccine (protoype strain), inactivated SARS-CoV-2 vaccine (omicron strain), individually. The IgG antibodies were detected using ELISA, while the neutralizing antibodies were measured through a microneutralization assay utilizing both the prototype and omicron strains. The ELISPOT assays were performed to measure the secretion of IL-4 and IFN-γ, and the concentrations of secreted IL-2 and IL-10 in the supernatants were measured by ELISA.

We have first developed a two-component recombinant vaccine for COVID-19 based on RBD-Fc and NTD-Fc proteins, with an optimal RBD-Fc/NTD-Fc ratio of 3:1. This novel two-component vaccine demonstrated the ability to induce durable and potent IgG antibodies, as well as the neutralizing antibodies in both the two-dose homologous and sequential vaccinations. Heterologous booster with this two-component vaccine could induce higher neutralizing antibody titers than the homologous group. Additionally, the vaccine elicited relatively balanced Th1- and Th2-cell immune responses.

This novel two-component recombinant vaccine exhibits high immunogenicity and offers a potential booster strategy for COVID-19 vaccine development.

[end paper 46]

[begin paper 47]

PMCID = 11722225

Title = Cross-reactive sarbecovirus antibodies induced by mosaic RBD-nanoparticles

Therapeutic monoclonal antibodies (mAbs) against SARS-CoV-2 become obsolete as spike substitutions reduce antibody binding. To induce antibodies against conserved receptor-binding domain (RBD) regions for protection against SARS-CoV-2 variants of concern and zoonotic sarbecoviruses, we developed mosaic-8b RBD-nanoparticles presenting eight sarbecovirus RBDs arranged randomly on a 60-mer nanoparticle. Mosaic-8b immunizations protected animals from challenges from viruses whose RBDs were matched or mismatched to those on nanoparticles. Here, we describe neutralizing mAbs from mosaic-8b–immunized rabbits, some on par with Pemgarda (the only currently FDA-approved therapeutic mAb). Deep mutational scanning, in vitro selection of spike resistance mutations, and cryo-EM structures of spike-antibody complexes demonstrated targeting of conserved epitopes. Rabbit mAbs included critical D-gene segment features in common with human anti-RBD mAbs, despite rabbit genomes lacking an equivalent human D-gene segment. Thus, mosaic RBD-nanoparticle immunization coupled with multiplexed screening represent an efficient way to generate and select therapeutic pan-sarbecovirus and pan-SARS-2 variant mAbs.

[end paper 47]

[begin paper 48]

PMCID = 11680907

Title = Daphnetin may protect from SARS-CoV-2 infection by reducing ACE2

To combat the SARS-CoV-2 pandemic, innovative prevention strategies are needed, including reducing ACE2 expression on respiratory cells. This study screened approved drugs in China for their ability to downregulate ACE2. Daphnetin (DAP) was found to significantly reduce ACE2 mRNA and protein levels in PC9 cells. DAP exerts its inhibitory effects on ACE2 expression by targeting HIF-1α and JAK2, thereby impeding the transcription of the ACE2 gene. The SARS-CoV-2 pseudovirus infection assay confirmed that DAP-treated PC9 cells exhibited decreased susceptibility to viral infection. At therapeutic doses, DAP effectively lowers ACE2 expression in the respiratory systems of mice and humans. This suggests that DAP, already approved for other conditions, could be a new preventive measure against SARS-CoV-2, offering a cost-effective and accessible way to reduce SARS-CoV-2 spread.

The online version contains supplementary material available at 10.1038/s41598-024-79734-z.

[end paper 48]

[begin paper 49]

PMCID = 11463117

Title = A prediction of mutations in infectious viruses using artificial intelligence

Many subtypes of SARS-CoV-2 have emerged since its early stages, with mutations showing regional and racial differences. These mutations significantly affected the infectivity and severity of the virus. This study aimed to predict the mutations that occur during the evolution of SARS-CoV-2 and identify the key characteristics for making these predictions. We collected and organized data on the lineage, date, clade, and mutations of SARS-CoV-2 from publicly available databases and processed them to predict the mutations. In addition, we utilized various artificial intelligence models to predict newly emerging mutations and created various training sets based on clade information. Using only mutation information resulted in low performance of the learning models, whereas incorporating clade differentiation resulted in high performance in machine learning models, including XGBoost (accuracy: 0.999). However, mutations fixed in the receptor-binding motif (RBM) region of Omicron resulted in decreased predictive performance. Using these models, we predicted potential mutation positions for 24C, following the recently emerged 24A and 24B clades. We identified a mutation at position Q493 in the RBM region. Our study developed effective artificial intelligence models and characteristics for predicting new mutations in continuously evolving infectious viruses.

The online version contains supplementary material available at 10.1186/s44342-024-00019-y.

[end paper 49]

[begin paper 50]

PMCID = 11579292

Title = Preclinical characterization of the Omicron XBB.1.5-adapted BNT162b2 COVID-19 vaccine

As SARS-CoV-2 evolves, increasing in potential for greater transmissibility and immune escape, updated vaccines are needed to boost adaptive immunity to protect against COVID-19 caused by circulating strains. Here, we report features of the monovalent Omicron XBB.1.5-adapted BNT162b2 vaccine, which contains XBB.1.5-specific sequence changes, relative to the original BNT162b2 backbone, in the encoded prefusion-stabilized SARS-CoV-2 spike protein (S(P2)). Biophysical characterization of Omicron XBB.1.5 S(P2) demonstrated that it maintains a prefusion conformation and adopts a flexible, predominantly open, state, with high affinity for the human ACE-2 receptor. When administered as a 4th dose in BNT162b2-experienced mice, the monovalent Omicron XBB.1.5 vaccine elicited substantially higher serum neutralizing titers against pseudotyped viruses of Omicron XBB.1.5, XBB.1.16, XBB.1.16.1, XBB.2.3, EG.5.1 and HV.1 sublineages and phylogenetically distant BA.2.86 lineage than the bivalent Wild Type + Omicron BA.4/5 vaccine. Similar trends were observed against Omicron XBB sublineage pseudoviruses when the vaccine was administered as a 2-dose series in naive mice. Strong S-specific Th1 CD4+ and IFNγ+ CD8+ T cell responses were also observed. These findings, together with real world performance of the XBB.1.5-adapted vaccine, suggest that preclinical data for the monovalent Omicron XBB.1.5-adapted BNT162b2 was predictive of protective immunity against dominant SARS-CoV-2 strains.

[end paper 50]

[begin paper 51]

PMCID = 11577066

Title = Intranasal delivery of a subunit protein vaccine provides protective immunity against JN.1 and XBB-lineage variants

The mucosal immune response plays a crucial role in the prevention of respiratory viruses. Given the risk of recurrent SARS-CoV-2 infections in the population, the rapid development of next-generation intranasal COVID-19 vaccines with high safety and efficacy is paramount. In the current study, we developed a protein-based intranasal vaccine comprising the XBB.1.5 receptor binding domain (RBD)-derived trimeric recombinant protein (RBDXBB.1.5-HR) and an MF59-like oil-in-water adjuvant. Intranasal administration of RBDXBB.1.5-HR vaccine elicited robust and sustained humoral immune responses in mice and rats, resulting in high levels of neutralizing antibodies against XBB-lineage subvariants, with protection lasting for at least six months. The intranasal RBDXBB.1.5-HR vaccine generated potent mucosal immune responses, characterized by the inductions of tissue-resident T (TRM) cells, local cellular immunity, germinal center, and memory B cell responses in the respiratory tract. The combination of intramuscular and intranasal delivery of the RBDXBB.1.5-HR vaccine demonstrated exceptional systemic and mucosal protective immunity. Furthermore, intranasal delivery of RBDXBB.1.5-HR vaccine as a heterologous booster shot showed more effective boosting effects after mRNA administration compared to homologous vaccination, as evidenced by the induction of superior systemic and extra mucosal immune response. Importantly, the intranasal RBDXBB.1.5-HR vaccine conferred efficient protection against the challenge with authentic EG.5.1 viruses in vivo. These findings identify the intranasal RBDXBB.1.5-HR vaccine as a potential mucosal vaccine candidate for the prevention of SARS-CoV-2 infection.

[end paper 51]

[begin paper 52]

PMCID = 8264290

Title = A Computational Approach Identified Andrographolide as a Potential Drug for Suppressing COVID-19-Induced Cytokine Storm

The newly identified betacoronavirus SARS-CoV-2 is the causative pathogen of the coronavirus disease of 2019 (COVID-19) that killed more than 3.5 million people till now. The cytokine storm induced in severe COVID-19 patients causes hyper-inflammation, is the primary reason for respiratory and multi-organ failure and fatality. This work uses a rational computational strategy to identify the existing drug molecules to target host pathways to reduce the cytokine storm.

We used a “host response signature network” consist of 36 genes induced by SARS-CoV-2 infection and associated with cytokine storm. In order to attenuate the cytokine storm, potential drug molecules were searched against “host response signature network”. Our study identified that drug molecule andrographolide, naturally present in a medicinal plant Andrographis paniculata, has the potential to bind with crucial proteins to block the TNF-induced NFkB1 signaling pathway responsible for cytokine storm in COVID-19 patients. The molecular docking method showed the binding of andrographolide with TNF and covalent binding with NFkB1 proteins of the TNF signaling pathway.

We used a rational computational approach to repurpose existing drugs targeting host immunomodulating pathways. Our study suggests that andrographolide could bind with TNF and NFkB1 proteins, block TNF-induced cytokine storm in COVID-19 patients, and warrant further experimental validation.

[end paper 52]

[begin paper 53]

PMCID = 11729908

Title = mRNA vaccines with RBD mutations have broad-spectrum activity against SARS-CoV-2 variants in mice

The emergence of SARS-CoV-2 variants with defined mutations that enhance pathogenicity or facilitate immune evasion has resulted in a continual decline in the protective efficacy of existing vaccines. Therefore, there is a pressing need for a vaccine capable of combating future variants. In this study, we designed new mRNA vaccines, BSCoV05 and BSCoV06, and generated point mutations in the receptor-binding domain (RBD) of the original Wuhan strain to increase their broad-spectrum antiviral activity. Additionally, we used the BA.1 RBD as a control. Both vaccines elicited a robust immune response in BALB/c and K18-hACE2 mice, generating high levels of specific binding antibodies against the BA.2 RBD. Moreover, all three vaccines induced neutralizing antibodies against the prototype viral strain and relevant variants, including the Alpha and Beta strains and the Omicron variants BA.1, BA.2, BA.5, XBB.1.5, XBB.1.16, EG.5.1, and EG.5.1.1, with BSCoV06 demonstrating broader neutralizing antibody activity. Both BSCoV05 and BSCoV06 also elicited a cellular immune response. After the challenge, both BSCoV05 and BSCOV06 provided protection against the EG.5.1 strain in both mouse strains. Therefore, these two vaccines merit further evaluation in nonhuman primates, and this vaccine design strategy should be explored for its potential application in combating future SARS-CoV-2 variants, offering valuable insights into broad-spectrum vaccine development.

[end paper 53]

[begin paper 54]

PMCID = 11375728

Title = Repurposed Drugs during the Outbreak of Pandemic COVID-19: A Mini-Review on Their Molecular Structures and Hit-and-Trial Results

One of the most significant threats to global public health in the 21st century is the novel coronavirus disease (COVID-19) caused by SARS-CoV-2. It rapidly turned into a global pandemic after it was identified in late 2019, and the World Health Organization announced the end of the pandemic on May 5, 2023. Current strategies for managing this disease include vaccination and repurposing antimalarial and antibiotic medications to alleviate symptoms like fever and throat pain, which are associated with acute respiratory distress syndrome (ARDS). Antiviral drugs such as chloroquine, hydroxychloroquine, azithromycin, remdesivir, and favipiravir have been repurposed for the treatment of COVID-19. They were previously recommended for treating SARS-CoV and MERS-CoV. However, the inefficacy and adverse side effects of these repurposed drugs led to a decrease in their widespread use in treating COVID-19 patients. The lack of approved drugs for combating this coronavirus and its unpredictable variants remains a significant challenge.

[end paper 54]



