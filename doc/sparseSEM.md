---
title: "Elastic Net Enabled Sparse-Aware Maximum Likelihood for Structural Equation Models in Inferring Gene Regulatory Networks"
author: 
  - Anhui Huang
date: "`r format(Sys.time(), '%B %d, %Y')`"
output:
  pdf_document:
    fig_caption: yes
    toc: yes
    toc_depth: 3
vignette: >
  %\VignetteIndexEntry{Elastic Net Enabled Sparse-Aware Maximum Likelihood for Structural Equation Models in Inferring Gene Regulatory Networks}
  %\VignetteEngine{knitr::rmarkdown}
  %\usepackage[utf8]{inputenc}    
---


<a id="top"></a>


<a id="summary"></a>

## Summary
<br>
<br>
Understanding the multiple levels of gene regulations is the key for the prediction of complex cellular behavior.
Integrating genetic perturbations with gene expression data has been proven to be more accurate in learning of causal 
regulatory relationships among genes comparing to treating each gene expression level as an individual quantitative trait. 
The previously designed sparse-aware maximum likelihood method for structural equation models (SEM-SML) has been shown
to be able to integrate such information to infer gene regulatory networks (GRN) systematically and offer significant 
better performance than state-of-the-art algorithms.  We extended the SEM-SML to incorporate adaptive elastic net (EN) penalty for the likelihood function of the SEMs, 
and implemented the SEM-EN software in efficient C/C++ with parallel computational capability by Message Passing Interface (MPI). 
The parallel design is capable of scaling up the network structure inference in a computer cluster, and enables SEM-EN to infer 
a network structure with thousands of nodes. Simulation studies demonstrated that SEM-EN was capable of inferring a large 
network within affordable computational time while achieved more accurate power of detection than SEM-SML. The software was 
further applied to infer the GRN in budding yeast systematically, in which two set of experimental perturbations with 
co-regulated gene set information were available on the AMN1 and LEU2 genes. 
The SEM-EN identified GRN had two clusters with hubs and members in line with the experimental perturbations, 
corroborating the strength of SEM-EN. While the parallel version of the SEM-EN software for computer cluster is 
implemented with command line interface, the SEM-EN method is also implemented in C/C++ with a user-friendly R interface for personal computers. 
An R software package sparseSEM with both SEM-SML and SEM-EN features is available on the Comprehensive R Archive Network (CRAN),
and the command-line software is freely available upon request.

### Key Words:
\begin{itemize}
\item{SEM: structural equation models}
\item{EN: elastic net}
\item{GRN: gen regulatory network}
\item{sparseML: sparse-aware maximum likelihood}

\end{itemize}

[Back to Top](#top)

<a id="intro"></a>

## Introduction 
<br>  
<br>

Understanding biological network at system-level is crucial to gaining insights into gene functions and cellular dynamics. To elucidate the complexity of gene regulatory networks (GRNs)  and uncover the mechanism of gene regulations that lead to complex biologically diversified phenotypes, a large number of studies have been conducted at genetic, transcriptomic, proteomic and metabolomic level [1].  Experimental approaches in deducing physical interactions of individual genes are time consuming and labor intensive, whereas computational methods that exploit genome-wide expression data and genetic perturbations from high-throughput technologies are efficient and cost-effective [2]. 
<br>  
<br>

A number of computational methods have been developed to leverage different intermediate phenotypes, such as transcript, protein or metabolite level, to understand cell regulation process comprehensively. For example, a co-expression or relevance network infers network structures through measuring the similarity level in gene expression [3, 4], a Bayesian network evaluates the dependence structure among genes [5], and a Gaussian graphic model approach evaluates the presence of an edge if the pair of genes are conditionally dependent given expression levels of all other genes [6, 7]. Another approach employs regularized linear regression models to find the co-occurrence among genes to infer the gene network [8-11]. Recently, a powerful structural equation models (SEMs) for the GRN modeling was developed [12], which systematically integrated both genetic perturbation and gene expression data, and inferred the GRN through a sparse-aware maximum likelihood (SML) method. The SEM-SML applied an adaptive l1-norm regularization term on the likelihood function, which was then optimized via an efficient block coordinate ascent algorithm. Simulations of the SML algorithm demonstrated that it accurately inferred regulatory relations among genes and offered significant better performance than state-of-the-art algorithms [12].
<br>  
<br>

The SEM-SML algorithm was motivated by the fact that the gene networks are sparse [13-15]. While this is the first study that infers sparse SEM systematically, other penalty functions, especially the penalty function of the elastic net (EN) [16, 17] may improve the inference accuracy for GRNs.  This is based on the following observations. Although the Lasso-based methods achieve good performance in the inference of GRNs and are ranked top on the list of a number of methods for GRN inference [18], they tend to miss interactions in feed-forward loops, fan-in motifs and fan-out motifs. This is likely due to the fact that Lasso typically chooses only one variable among several highly correlated variables. On the other hand, it has been known through experimentation that a gene regulator in GRN can typically shape the expression profile of a set of genes, meaning that the expression of the set of co-regulated genes can be highly correlated [1, 2]. For example, in gene expression microarray analysis, researchers aim at finding a group of up and down-regulated gene expression patterns under different experimental treatments, and discover novel and unexpected functional relationships among genes [19]. Such observations make the elastic net the right fit since it retains correlated variables while still yielding a sparse model [16]. 
<br>  
<br>

In this paper, we developed an SEM-based method for the inference of GRNs that maximizes the l1/l2-regularized likelihood function similar to the one used in the adaptive EN [16, 17]. The SEM with adaptive elastic net penalty algorithm (SEM-EN) was maximized through a parallelized efficient block coordinate ascent algorithm, which inferred the network structure on each node in parallel. Considering that the MATLAB implementation of the SEM-SML algorithm in [12] was time consuming and not applicable to large network with thousands of nodes, here we further developed a software tool in C/C++ with message passing interface (MPI) to accelerate the computation through parallelization. Thanks to the elastic net penalty, the SEM-EN algorithm encourages a grouping effect that not only predicts causal regulatory genes, but also elucidates the complexity of cell regulation at system level. Computer simulation demonstrated that SEM-EN outperformed SEM-SML with higher power of detection (PD) and similar false discovery rate (FDR). The SEM-EN algorithm was further applied to infer the GRN of a previously described budding yeast (Saccharomyces cerevisiae) dataset.


[Back to Top](#top)


<a id="methods"></a>

## Methods 

### Sparse SEM model for gene regulatory networks

Consider the expression levels of $N_{g}$ genes from $N$ individuals measured in microarray or RNAseq experiments. Following the design of [12], the gene regulatory network is postulated to obey the SEM:

$$
\tag{1}
\mathbf{y}_{i}=\mathbf{B} \mathbf{y}_{i}+\mathbf{F} \mathbf{x}_{i}+\boldsymbol{\mu}+\varepsilon_{i}, \quad i=1, \cdots, N
$$

where $\mathbf{y}_{i}:=\left[y_{i 1}, \cdots, y_{i N_{g}}\right]^{T}$ is the expression levels of $N_{g}$ genes from $N$ individuals, and $\mathbf{x}_{i}:=\left[x_{i 1}, \cdots, x_{i N_{q}}\right]^{T}$ denotes the genotype of $N_{q} \geqslant N_{g}$ eQTLs of individual $i, i=1, \ldots, N$. In this paper, we focus on the genetic variations observed at expression quantitative trait loci (eQTLs), and the developed methods can be applied to other genetic variations such as single nucleotide polymorphisms (SNPs), copy number variations (CNVs) and gene knockdown by RNA interference (RNAi) or controlled gene overexpression. B is an $N_{g} \times N_{g}$ matrix contains unknown parameters defining the network structure, and is assumed to be sparse; $\mathbf{F}$ is an $N_{g} \times N_{q}$ matrix captures the effect of $N_{q}$ eQTLs for $N_{g}$ genes; $\boldsymbol{\mu}$ is an $N_{g} \times 1$ vector accounts for possible model bias; and $\varepsilon_{i}$ is an $N_{g} \times 1$ vector captures the residual error. Typically, $\varepsilon_{i}$ is modeled as a zero-mean Gaussian vector with covariance $\sigma^{2}$, where I denotes the $N_{g} \times N_{g}$ identity matrix.

We assume that locations of $N_{q}$ eQTLs have been determined using existing eQTL mapping methods, thus $\mathbf{F}$ has $N_{q}$ entries with known locations but unknown effect size, and $N_{g} N_{q}-N_{q}$ of zero entries. Note that there are two structural properties in the GRN. First, as noted in [12], GRN and other general biochemical networks are sparse, meaning that only a relative smaller number of genes can be regulators of a given gene, thus matrix $\mathbf{B}$ is sparse. Second, a regulator typically can shape the expression profile of a set of genes, meaning that the expression of the set of co-regulated genes can be highly correlated [1, 2]. Based on the SEM-SML algorithm designed in [12], we developed a network inference algorithm that exploits the aforementioned structural properties.

### Structural equation models with adaptive elastic net penalty (SEM-EN)

Let us define $\mathbf{Y}=\left[\mathbf{y}_{1}, \cdots, \mathbf{y}_{N}\right], \mathbf{X}=\left[\mathbf{x}_{1}, \cdots, \mathbf{x}_{N}\right]$ and $\mathbf{E}=\left[\varepsilon_{1}, \cdots, \varepsilon_{N}\right]$, then we can write the SEM in (1) as $\mathbf{Y}=\mathbf{B Y}+\mathbf{F X}+\boldsymbol{\mu} \mathbf{1}^{T}+\mathbf{E}$. The SEM-EN algorithm applied the $I_{1} / I_{2}$-norm penalty to the log-likelihood function of SEM in eq (2) of [12]. Let $\tilde{\boldsymbol{y}}_{i}=\boldsymbol{y}_{i}-\overline{\boldsymbol{y}}_{i}$ and $\tilde{\mathbf{x}}_{i}=\mathbf{x}_{i}-\overline{\mathbf{x}}, i=1, \cdots, N$, where $\overline{\boldsymbol{y}}=\sum_{i=1}^{N} \boldsymbol{y}_{i} / N$ and $\overline{\mathbf{x}}=\sum_{i=1}^{N} \mathbf{x}_{i} / N$, and collect them into matrices $\tilde{\mathbf{Y}}=\left[\tilde{\mathbf{y}}_{1}, \cdots, \tilde{\mathbf{y}}_{N}\right], \tilde{\mathbf{X}}=\left[\tilde{\mathbf{x}}_{1}, \cdots, \tilde{\mathbf{x}}_{N}\right]$. Then SEM-EN infers the model parameters through $I_{1} / I_{2}$ penalized $\mathrm{ML}$ estimation:

$$
\tag{2}
(\hat{\mathbf{B}}, \hat{\mathbf{F}})_{E N}=\arg \max _{\mathbf{B}, \mathbf{F}} N \sigma^{2} \log |\operatorname{det}(\mathbf{I}-\mathbf{B})|-\frac{1}{2}\|\tilde{\mathbf{Y}}-\mathbf{B} \tilde{\mathbf{Y}}-\mathbf{F} \tilde{\mathbf{X}}\|_{F}^{2}-\lambda \alpha\|\mathbf{B}\|_{1, W}-\frac{1}{2}(1-\alpha) \lambda\|\mathbf{B}\|_{2}
$$$$
\text { subject to } B_{i i}=0, \forall i=1, \cdots, N_{g}, F_{j k}=0, \forall(j, k) \in S_{q}
$$

where $\|\mathbf{B}\|_{1, W}:=\sum_{i=1}^{N g} \sum_{j=1}^{N g} w_{i j}\left|B_{i j}\right|$ with $B_{i j}$ denotes the $(i, j)$ th entry of $\mathbf{B},\|\cdot\|_{F}$ denotes the Frobenius norm, and $S_{q}$ denotes the set of row and column indices of the entries of $\mathbf{F}$ know to be zero. Parameters $\lambda>0$ and $\alpha \in[0,1]$ are the penalty parameters following the design of the elastic net for linear regression [16]. Weights $w_{i j}$ in the $I_{1}$ penalty term are introduced in line with the adaptive the elastic net (EN) [17] and are selected as the estimated coefficients of the ridge regression same as the one described in [12]. Then the SML algorithm is modified to obtain the estimated parameters. Specifically, with the $I_{1} / I_{2}$ penalty $-\lambda \alpha w_{i j}\left|B_{i j}\right|-(1-\alpha) \lambda B_{i j}^{2} / 2$ applied to eq(10) in [12], we have the following objective function:

$$
\tag{3}
g_{i j}\left(B_{i j}\right):=N \hat{\sigma}^{2} \log \left|\alpha_{0}-c_{i j} B_{i j}\right|+\alpha_{1} B_{i j}-\left[(1-\alpha) \lambda+\frac{1}{2} \alpha_{2}\right] B_{i j}^{2}-\lambda w_{i j}\left|B_{i j}\right|
$$

where $\hat{\sigma}^{2}$ is the variance estimate, $c_{i j}$ denotes the $(i, j)$ th co-factor of matrix $(\mathbf{I}-\hat{\mathbf{B}})$ with $\hat{\mathbf{B}}$ being the estimate of matrix $\mathbf{B}, \alpha_{0}:=\operatorname{det}(\mathbf{I}-\hat{\mathbf{B}})+c_{i j} \hat{B}_{i j}$ with $\hat{B}_{i j}$ being the estimate of $B_{i j}$, $\alpha_{1}:=\left[\left(\mathbf{I}-\hat{\mathbf{B}}+\mathbf{e}_{i} \mathbf{e}_{j}^{T} \hat{B}_{i j}\right) \tilde{\mathbf{Y}} \tilde{\mathbf{Y}}^{T}-\hat{\mathbf{F}}^{n e w} \tilde{\mathbf{X}} \tilde{\mathbf{Y}}^{T}\right]_{i j}$ with $\mathbf{e}_{i}$ and $\mathbf{e}_{j}$ being the ith and $j$ th canonical vectors in $\mathfrak{R}^{N g}, \hat{\mathbf{F}}^{\text {new }}$ being the estimate of $\mathbf{F}$, and $\alpha_{2}:=\left\|\tilde{\mathbf{Y}}^{T} \mathbf{e}_{j}\right\|_{2}^{2}$. Let us define $\tilde{\alpha}_{2}=2(1-\alpha) \lambda+\alpha_{2}$, then the solution to the objective function in (3) is identical to that of [12] with $\alpha_{2}$ replaced by $\tilde{\alpha}_{2}$, and eqs(1216) in [12] are applied to inter the network parameters. Furthermore, let $\tilde{Q}_{i j}(\lambda)$ denote the derivative of the differential part of $(2)$ :

$$
\tag{4}
\tilde{Q}_{i j}(\lambda)=\frac{N \sigma^{2} c_{i j}(\lambda)}{\operatorname{det}(\mathbf{I}-\hat{\mathbf{B}}(\lambda))}+\left[\tilde{\mathbf{Y}} \tilde{\mathbf{Y}}^{T}-\hat{\mathbf{B}}(\lambda) \tilde{\mathbf{Y}} \tilde{\mathbf{Y}}^{T}-\hat{\mathbf{F}}(\lambda) \tilde{\mathbf{X}} \tilde{\mathbf{Y}}^{T}\right]_{i j}-(1-\alpha) \lambda \hat{\mathbf{B}}(\lambda)
$$

where $\hat{\mathbf{B}}(\lambda)$ and $\hat{\mathbf{F}}(\lambda)$ denote the optimal estimate of (2) for a given $\lambda$ with fixed $\alpha$, and $\sigma^{2}$ can be estimated as $\hat{\sigma}^{2}=\frac{1}{N N_{g}}\|\tilde{\mathbf{Y}}-\hat{\mathbf{B}}(\lambda) \tilde{\mathbf{Y}}-\hat{\mathbf{F}}(\lambda) \tilde{\mathbf{X}}\|_{F}^{2}$. Then the strong rule [39] for SEM-EN is available as following. Let $\lambda_{\max }$ denote the smallest $\lambda$ that yields $\hat{B}_{i j}=0, \forall i, j$, and $\lambda_{\max }>\lambda_{1}>\cdots>\lambda_{\min }$ is a decreasing set of values, the following discarding rule can be applied to find solution of (2):

$$
\tag{5}
\begin{aligned}
& \left|\tilde{Q}_{i j}\left(\lambda_{\max }\right)\right|<w_{i j}\left(2 \lambda_{l}-\lambda_{\max }\right) \Rightarrow \hat{B}_{i j}\left(\lambda_{l}\right)=0 \\
& \left|\tilde{Q}_{i j}\left(\lambda_{l-1}\right)\right|<w_{i j}\left(2 \lambda_{l}-\lambda_{l-1}\right) \quad \Rightarrow \quad \hat{B}_{i j}\left(\lambda_{l}\right)=0
\end{aligned}
$$

where $\lambda_{1}<\lambda_{\max }$ is a value in the path of $\lambda$. Note that for any $\lambda>\lambda_{\max }, \mathbf{B}=0$, and $\hat{\mathbf{F}}(\lambda)$ is fixed, thus $\tilde{Q}_{i j}(\lambda)$ is also fixed. Therefore, $\lambda_{\max }$ can be obtained from the following equation:

$$
\tag{6}
\lambda_{\max }=\max _{i, j=1, \cdots, N g}\left|\frac{Q_{i j}\left(\lambda_{\max }\right)}{w_{i j}}\right|
$$



### Software implementation

The block coordinate ascent algorithm in Algorithm 1 of [12] is updated with equations (3) and (4) along with the discarding rules in equations (5) and (6), and is parallelized to reduce execution time.

Specifically, a master computer node is designated to compute and check to convergence criterion, which is determined as $\operatorname{err}=\left\|\hat{\mathbf{B}}-\hat{\mathbf{B}}^{\text {new }}\right\|_{F}^{2} /\|\hat{\mathbf{B}}\|_{F}^{2}+\left\|\hat{\mathbf{F}}-\hat{\mathbf{F}}^{n e w}\right\|_{F}^{2} /\|\hat{\mathbf{F}}\|_{F}^{2}$ being smaller than a prespecified small value. And several slavery nodes will be assigned to compute the solution for each row of $\hat{\mathbf{B}}$. The parallelized computation is achieved in a high performance computing (HPC) clusters, and the software is implemented in $\mathrm{C} / \mathrm{C}++$ utilizing open MPI. On the other hand, when the scale and degree of a network to be inferred is small and computation is less demanding, we also provide the serial version of the $C / C++$ program with a user friendly $\mathrm{R}$ interface. To achieve fast computation, BLAS and LAPACK [37] were utilized in implementation of both software packages.








[Back to Top](#top)


<a id="results"></a>

## Results
<br>
<br>
### Scalability of the parallel network structure inference

To achieve feasible computation of inferring a GRN with hundreds or thousands of nodes, the parallel block coordinate ascent algorithm was implemented with a master-slavery paradigm. While one master node as designated for the computation of program initialization, decomposing the problem into small tasks, assign tasks for multiple slave nodes, gathering the results for determining convergence and generate the final result [20], number of slavery nodes can be supplied by users upon available system resources. To gain insights into the scaling property of the parallel computation, we also implemented the SEM-SML algorithms [12] with $\mathrm{C} / \mathrm{C}++$ and paralleled the block coordinate ascent algorithm with Open MPI. In fact, the SEM-SML algorithm is a special case of SEM-EN with shrinkage parameter $\alpha=1$ (see Methods section for details). The scaling property of the parallel computation was shown in Figure 1, where the performance was obtained from inference of a sparse DAG having $N=$ 300 samples, $N_{g}=300$ genes, $N_{e}=3$ edges, and $\sigma^{2}=0.05$. The computational time was the mean of 5 replicates. From Figure 1, a strong scaling pattern [21] was observed for both SEM-SML and SEM-EN.


![Scaling pattern for SEM-SML and SEM-EN](Fig1_nCPU.png){width=24%, height=35%}


### Simulation study

To evaluate the performance of SEM-EN, we compared PD and FDR with that of SEM-SML. If $\hat{B}_{i j} \neq 0$, then we consider there is an edge from gene $j$ to gene $i$. the PD and FDR of the SEM-SML and SEM-EN for different sample sizes are depicted in Figure 2 and Figure 3 corresponding to two directed acyclic graphs (DAGs) with number of genes $N_{g}=300$, expected number of edges per node $N_{e}=3$ and residual variance $\sigma^{2}=0.01$ and $\sigma^{2}=0.05$, respectively. The result of Figure 2 and 3 represents mean PD and FDR for 100 replicates for each for the 10 different sample sizes. It is observed that SEM-EN achieves higher PD and similar FDR comparing with that of SEM-SML for both DAGs despite of different sample sizes. Moreover, it can be seen that the performance gain of SEM-EN is more significant for the DAG with larger noise level (Figure 3). Take $N=500$ for example, the PD/FDR of SEM-EN for $\sigma^{2}=0.01$ are $0.9537 / 0.0433$, comparing to $0.9005 / 0.0375$ of SEM-SML. For $\sigma^{2}=0.05$ with the same sample size, the numbers are $0.9008 / 0.1300$, and $0.7663 / 0.1182$ for the two methods, respectively. 

![Performance of SEM-SML and SEM-EN for DAG simulation $\sigma^{2}=0.01$ ](Fig2_sigma01.png){width=20%, height=60%}


![Performance of SEM-SML and SEM-EN for DAG simulation $\sigma^{2}=0.05$ ](Fig3_sigma05.png){width=20%, height=60%}


![PD and FDR for SEM-SML and SEM-EN in analyzing the two simulated DAGs ](table1_data.png){width=30%, height=30%}


### Inference of the yeast GRN

The gene expression profile of 3,380 ORFs and genetic marker of 1,162 cis-eQTLs were used for network inference by SEM-EN. Two parameters controlling degree of sparseness in the GRN need to be learnt from data. Cross validation (CV) identified the optimal shrinkage parameters as $(\alpha, \lambda)=(0.45,0.0063)$ (see Methods section for the definitions of shrinkage parameters) for the SEM-EN method. With the pair of parameters, SEM-EN inferred a sparse GRN with 159 open reading frames (ORFs) involving 267 edges. The network was visualized via Cytoscape [22] shown in Figure 4 and the positional information of the ORFs involved in the GRN was depicted in Figure 5 by Circos software [23].

It has been known that in GRN, sub-network are typically associated with particular biological functions [24]. In our study, five major clusters of the GRN was identified via Cytoscape [22] shown in different colors in Figure 4, and gene ontology (GO) term enrichment analysis was performed for each clusters by Gorilla [25] (Table 1). Specific to molecular functions, Cluster 1 (lime color in Figure 4) includes 28 ORFs and is enriched with aldehyde dehydrogenase (NAD) activity and transferase activity that transferring aldehyde or ketonic groups. Cluster 2 (teal color) includes 10 ORFs and is enriched with asparaginase activity and iron ion transmembrane transporter activity. Cluster 3 (slate blue color) contains 30 ORFs and is enriched with carbamoyl-phosphate synthase activity, oxidoreductase activity, NAD binding, ad dicarboxylic acid transmembrane transporter activity. Cluster 4 (olive color) contains 13 ORFs and is enriched with mating pheromone activity, DNA binding and bending, and RNA polymerase II transcription. Cluster 5 (red color) contains 16 ORFs and is enriched with glucosidase activity. The corresponding significant terms were shown in Figure 4.

![Sparse budding yeast GRN inferred by SEM-EN](Fig4_cisTransNetworkYeast_clusters_manualAdjusted_GOterm.jpg){width=45%, height=99%}


![Enriched GO terms of the five clusters in yeast GRN ](table2_data.png){width=30%, height=60%}

Within a network cluster, nodes that are most relevant for the corresponding cluster function often have higher degree, meaning that there are more edges connected to them than other nodes [26]. Encoding the leucine biosynthetic enzyme, LEU2 is deleted in the RM parents in the segregants [27], and has been previously predicted as a causal regulator for the expression of a subset of genes by several independent studies [27-29]. In our GO term enrichment analysis shown in Table 1 , leucine biosynthetic process is significant for Cluster 3 $\left(p\right.$-value $\left.=4.77 \times 10^{-6}\right)$, and LEU2 is identified as regional hub in for the cluster with 20 genes directly connected to it (Figure 4). The parent lab strain BY4716 has AMN1 gene deleted. $A M N 1$ gene encodes a protein required for daughter cell separation, multiple mitotic checkpoints, and chromosome stability [27]. It has been verified in previous study that $A M N 1$ gene is responsible for the regulation of a set of genes (SCW11, DSE1, DSE2, DSE3, DSE4, ISR1, PRY3, EGT2, SUN4, BUD9) [27]. Among them, SCW11, DSE1, DSE2, DSE3, ISR1, EGT2, SUN4, and BUD9 are in Cluster 5 in Figure 4. These results from previous biological experiments and independent research studies corroborate the strength of the SEM-EN algorithm.


![Budding yeast gene interaction pattern with link edges identified by SEM-EN](Fig5_yeast_grn_genome_biology.png){width=30%, height=90%}


![eQTL hotspots identified in the original publication of the budding yeast dataset and predicted using SEM-EN](table3_data.png){width=30%, height=40%}


While the genetic engineering of $A M N 1$ and LEU2 genes serves as a direct positive control for the GRN inference in budding yeast $[27,29,30]$, the network structure inferred by SEM-EN shed light into the gene regulatory relationships. Previous study on single cis-eQTL mapping revealed 13 eQTL hot spots (Table 2), where eQTLs have pleiotropic effects on a number of expression traits [27]. The positions of eQTL hot spots can be visualized as the red ticks on the yeast chromosomes in Figure 5, and the regulator genes in the GRN inferred from SEM-EN are listed in Table 2. While the original study analyzed all 6,126 yeast ORFs and identified 8 regulators for the 13 eQTL hot spots, our study considered 3,380 ORFs that passed data quality control (see the Methods section for details) and identified regulators for 9 of the hot spots. Regulators such as $A M N 1, L E U 2$, and MATALPHA1 are consistent with previous study (Table 2). Particularly, SEM-EN identified regulators that are missed in previous study including RIB5 for hot spot 3 , TSL1 for hot spot 10 and SUN4 for hotspot 11. Among them, RIB5 is a gene encodes the riboflavin synthase that catalyzes the last step of the riboflavin biosynthesis pathway and involves in amino acid biosynthesis. RIB5 interacts with ARG1 in the SEM-EN identified GRN. Association of RIB5 and $A R G 1$ has been characterized in previous study [31], and both are key proteins for cellular growth. TSL1 interacts with CTT1, both of which have been identified to be associated with cellular growth under stress [32]. SUN4 is a gene involved in cell wall separation [33], and interacts with DSE2 and SCW11 in the SEM-EN inferred GRN (Figure 4). DSE2 is a daughter cell-specific secreted protein that plays a key role in daughter cell separation. During the separation process, DSE2 degrades cell wall from the daughter side and causes daughter cell to separate from the mother cell [34]. SCW11 is a cell wall protein that plays a key role in conjugation during mating [35], and its functional association with DSE2 have been experimentally characterized [36]. Given the genetic perturbation background in daughter cell separation and amino acid biosynthesis, previous molecular experiment results corroborate the predictive power of the inferred GRN and the strength of the SEM-EN method.




[Back to Top](#top)



<a id="discuss"></a>

## Discussion
<br>
<br>

Understanding the multiple levels of gene regulations is the key for the prediction of complex cellular behavior. Integrating genetic perturbations with gene expression data has been proven to be more accurate in learning causal regulatory relation among genes comparing to treating each gene expression level as an individual quantitative trait [12]. The SEM-SML has been shown to be able to integrate such information to infer GRN systematically and offer significant better performance than state-of-the-art algorithms. We extended the SEM-SML to incorporate adaptive elastic net penalty [16] for the likelihood function of the SEMs, and implemented the SEM-EN software in efficient $\mathrm{C} / \mathrm{C}++$ with parallel computational capability by MPI. Simulation studies demonstrated that SEM-EN is capable of inferring a large network within affordable computational time while achieving high power of detection than SEM-SML. The software is further applied to systematically infer the GRN in budding yeast.

The work in this paper improved the SEM-SML algorithm [12] from two directions. First, while previous work was implemented in MATLAB, the SEM-SML algorithm and the new SEM-EN algorithm were implemented in $\mathrm{C} / \mathrm{C}++$ with the fast basic linear algebra subprograms (BLAS) and linear algebra package (LAPACK) [37] in this paper. Simulation demonstrated that computational time for SEM-SML in $\mathrm{C} / \mathrm{C}++$ was reduced by more than 10 times compared to the one implemented in MATLAB using a single CPU node. To achieve computational feasibility for inferring large network with thousands of nodes, the block coordinate ascent technique in Algorithm 1 of [12] is parallelized with Open MPI [38]. A strong scaling pattern for the parallel implementation was observed and depicted in Figure 1, and the new software is capable of inferring a network structure more than 100 times faster. For example, while the MATLAB implementation takes several days to infer a network structure with 300 of nodes, and it is not realistic for it to infer a network structure with thousands of node, in the yeast GRN analysis, we demonstrated that SEM-EN is able to analyze a network with more than 3,000 nodes.


Second, with the adaptive elastic net $[16,17]$ in place of $l_{l}$-norm penalty for the SEM likelihood function, we showed that SEM-EN achieves higher PD while controlling a similar FDR to that of SEM-SML (Figures 2, 3). Superior performance of SEM-EN over SEM-SML is expected due to two reasons. First, it is known that $l_{1}$-norm penalty in regularized linear regression typically keeps only one out a group of correlated effects [16]. However, in cellular metabolisms, a gene regulator typically can shape the expression profile of a set of genes, resulting in highly correlated gene expression patterns $[1,2]$. Mathematically, it can be shown from SEM that gene expression levels have covariance $\operatorname{cov}(\mathbf{Y})=\left[(\mathbf{I}-\mathbf{B})^{-1}\right]^{T} \sigma^{2} \mathbf{I}(\mathbf{I}-\mathbf{B})^{-1}$, which is not diagonal. Therefore, the performance gain of SEM-EN over SEM-SML is expected given strong correlation among gene expression levels. Second, SEM-EN improves inference accuracy by taking the grouping effect into account, since it has been known that elastic net penalty enjoys a strong grouping effect than the $l_{l}$-norm penalty in linear regression. In fact, SEM-SML becomes a special case of SEM-EN with parameter $\alpha=1$.

In inferring the GRN of budding yeast, SEM-EN integrates genetic perturbations with genome-wide expression data. On the one hand, for the set of regulation that have been verified in previous studies such as the set of genes regulated by $A M N 1$ and $L E U 2$, the GRN obtained by SEM-EN is in line with previous findings, which corroborates the strength of the SEM-EN method. On the other hand, with the grouping effect encouraged by SEM-EN, we were also able to identify other ORFs interacting with known regulators. For example, seven ORFs in Custer 5 was not reported in the list of [27] that linked to AMN1. Among them, $R M A 1$ interacts with $D S E 2$ and $S C W 11$, both are directly linked to $A M N 1 ; A G P 2$ directly interacts with DSE 1 and has a distance of 2 to $A M N 1$, and TOF1 and HRR25 both interacts with DSE2. The genes in the cluster were known to specifically expressed in daughter cells during budding [36], and it may be worthy of experimental investigation to further study their roles in gene regulation affecting daughter-cell separation after budding.




[Back to Top](#top)






<a id="references"></a>

## References
<br>
<br>

1. Civelek M, Lusis AJ: Systems genetics approaches to understand complex traits. Nat Rev Genet 2014, 15:34-48.

2. Nuzhdin SV, Friesen ML, McIntyre LM: Genotype-phenotype mapping in a post-GWAS world. Trends in Genetics 2012, $28: 421-426$

3. Butte AJ, Tamayo $\mathrm{P}$, Slonim $\mathrm{D}$, Golub TR, Kohane IS: Discovering functional relationships between RNA expression and chemotherapeutic susceptibility using relevance networks. Proc Natl Acad Sci USA 2000, 97:12182-12186

4. Basso K, Margolin AA, Stolovitzky G, Klein U, Dalla-Favera R, Califano A: Reverse engineering of regulatory networks in human B cells. Nat Genet 2005, 37:382-390.

5. Friedman N, Linial M, Nachman I, Pe'er D: Using Bayesian networks to analyze expression data. $J$ Comput Biol 2000, 7:601-620

6. Dobra $\mathrm{A}$, Hans $\mathrm{C}$, Jones $\mathrm{B}$, Nevins $\mathrm{JR}$, Yao $\mathrm{G}$, West M: Sparse graphical models for exploring gene expression data. $J$ Multivar Anal 2004, 90:196-212

7. Schafer J, Strimmer K: An empirical Bayes approach to inferring large-scale gene association networks. Bioinformatics $2005,21: 754-764$

8. Bonneau R, Reiss DJ, Shannon P, Facciotti M, Hood L, Baliga NS, Thorsson V: The Inferelator: an algorithm for learning parsimonious regulatory networks from systems-biology data sets de novo. Genome Biol 2006, 7:R36.

9. di Bernardo D, Thompson MJ, Gardner TS, Chobot SE, Eastwood EL, Wojtovich AP, Elliott SJ, Schaus SE, Collins JJ Chemogenomic profiling on a genome-wide scale using reverse-engineered gene networks. Nat Biotechnol 2005, 23:377-383

10. Gardner TS, Di Bernardo D, Lorenz D, Collins JJ: Inferring genetic networks and identifying compound mode of action via expression profiling. Science 2003, 301:102-105

11. Schafer J, Strimmer K: A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics. Stat Appl Genet Mol Biol $2005, \mathbf{4}: 32$.

12. Cai X, Bazerque JA, Giannakis GB: Inference of gene regulatory networks with sparse structural equation models exploiting genetic perturbations. PLoS Comput Biol 2013, 9:e1003068.

13. Jeong H, Mason SP, Barabasi A-L, Oltvai ZN: Lethality and centrality in protein networks. Nature 2001, 411:41-42. 

14. Tegner J, Yeung MS, Hasty J, Collins JJ: Reverse engineering gene networks: integrating genetic perturbations with dynamical modeling. Proc Natl Acad Sci USA 2003, 100:5944-5949.

15. Thieffry D, Huerta AM, Perez-Rueda E, Collado-Vides J: From specific gene regulation to genomic networks: a global analysis of transcriptional regulation in Escherichia coli. Bioessays 1998, 20:433-440.

16. Zou H, Hastie T: Regularization and variable selection via the elastic net. J Roy Stat Soc B Met 2005, 67:301-320.

17. Zou H, Zhang HH: On the adaptive elastic-net with a diverging number of parameters. Ann Stat 2009 , 37:1733.

18. Marbach D, Costello JC, Kuffner R, Vega NM, Prill RJ, Camacho DM, Allison KR, Kellis M, Collins JJ, Stolovitzky G: Wisdom of crowds for robust gene network inference. Nat Meth 2012, 9:796-804

19. Slonim DK, Yanai I: Getting started in gene expression microarray analysis. PLoS Comput Biol 2009, 5:e1000543

20. Quinn MJ: Parallel Programming. TMH CSE; 2003.

21. Akl SG: Parallel computation: models and methods. Prentice-Hall, Inc.; 1997.

22. Shannon P, Markiel A, Ozier O, Baliga NS, Wang JT, Ramage D, Amin N, Schwikowski B, Ideker T: Cytoscape: A software environment for integrated models of biomolecular interaction networks. Genome Res 2003, 13:2498-2504.

23. Krzywinski M, Schein J, Birol I, Connors J, Gascoyne R, Horsman D, Jones SJ, Marra MA: Circos: an information aesthetic for comparative genomics. Genome Res $2009,19: 1639-1645$.

24. Langfelder P, Mischel PS, Horvath S: Whenis hub gene selection better than standard meta-analysis? PloS one 2013, $8: \mathrm{e} 61505$

25. Eden E, Navon R, Steinfeld I, Lipson D, Yakhini Z: GOrilla: a tool for discovery and visualization of enriched GO terms in ranked gene lists. BMC Bioinformatics $2009, \mathbf{1 0}: 48$

26. Almaas E: Biological impacts and context of network theory. Journal of Experimental Biology 2007, 210:1548-1558.

27. Yvert G, Brem RB, Whittle J, Akey JM, Foss E, Smith EN, Mackelprang R, Kruglyak L: Trans-acting regulatory variation in Saccharomyces cerevisiae and the role of transcription factors. Nat Genet 2003, 35:57-64

28. Zhu J, Sova P, Xu Q, Dombek KM, Xu EY, Vu H, Tu Z, Brem RB, Bumgarner RE, Schadt EE: Stitching together multiple data dimensions reveals interacting metabolomic and transcriptomic networks that modulate cell regulation. $P L o S$ Biol 2012,10:e1001301.

29. Zhu J, Zhang B, Smith EN, Drees B, Brem RB, Kruglyak L, Bumgarner RE, Schadt EE: Integrating large-scale functional genomic data to dissect the complexity of yeast regulatory networks. Nat Genet 2008, 40:854-861.

30. Brem RB, Yvert $\mathrm{G}$, Clinton R, Kruglyak L: Genetic dissection of transcriptional regulation in budding yeast. Science 2002 296:752-755

31. Creighton C, Hanash S: Mining gene expression databases for association rules. Bioinformatics 2003, 19:79-86.

32. Meadows R: Yeast survive by hedging their bets. PLoS Biol 2012, 10:e1001327.

33. Mouassite M, Camougrand N, Schwob E, Demaison G, Laclau M, Guerin M: The $S \boldsymbol{U}$, family: yeast $S \boldsymbol{N}$, $4 / S C W 3$ is involved in cell septation. Yeast $2000,16: 905-919$

34. Klis FM, Mol P, Hellingwerf K, Brul S: Dynamics of cell wall structure in Saccharomyces cerevisiae. FEMS microbiology reviews $2002, \mathbf{2 6 : 2 3 9 - 2 5 6}$

35. Zeitlinger J, Simon I, Harbison CT, Hannett NM, Volkert TL, Fink GR, Young RA: Program-specific distribution of a transcription factor dependent on partner transcription factor and MAPK signaling. Cell 2003, 113:395-404.

36. Colman-Lerner A, Chin TE, Brent R: Yeast $\boldsymbol{C b k} \boldsymbol{1}$ and $\boldsymbol{M o b} 2$ activate daughter-specific genetic programs to induce asymmetric cell fates. Cell $2001,107: 739-750$

37. Anderson E: LAPACK Users' Guide. Society for Industrial and Applied Mathematics; 1999.

38. Gabriel E, Fagg G, Bosilca G, Angskun T, Dongarra J, Squyres J, Sahay V, Kambadur P, Barrett B, Lumsdaine A, et al: Open MPI: goals, concept, and design of a next generation MPI implementation. In Recent Advances in Parallel Virtual Machine and Message Passing Interface. Volume 3241. Edited by Kranzlmuller D, Kacsuk P, Dongarra J: Springer Berlin Heidelberg; 2004: 97-104: Lecture Notes in Computer Science].

39. Tibshirani R, Bien J, Friedman J, Hastie T, Simon N, Taylor J, Tibshirani RJ: Strong rules for discarding predictors in lasso-type problems. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 2012, 74:245-266.

40. Brem RB, Kruglyak L: The landscape of genetic complexity across 5,700 gene expression traits in yeast. Proceedings of the National Academy of Sciences of the United States of America 2005, 102:1572-1577

41. Kellis M, Patterson N, Endrizzi M, Birren B, Lander ES: Sequencing and comparison of yeast species to identify genes and regulatory elements. Nature $2003, \mathbf{4 2 3}: 241-254$




[Back to Top](#top)


