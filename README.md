# ptalign

A Python tool for **pseudotime alignment** between reference and query single-cell datasets. 

ptalign places query cells at their best-matched position in the reference pseudotime, providing a framework for:
- comparative study of pseudotime expression dynamics across numerous query datasets
- transfer of contextual knowledge from the reference pseudotime trajectory

This process is enabled through a reference gene set capturing pseudotime dynamics, which is used to derive correlation-based pseudotime-similarity profiles that input to a pseudotime-predictive MLP to derive aligned pseudotimes.

![ptalign_scheme](https://github.com/user-attachments/assets/a09b2acf-c30f-4d30-9e22-fc106b8ae882)

A DTW-based permutation framework compares pseudotime expression dynamics between the reference and aligned query pseudotimes against those derived from expression-matched permuted gene sets. This way, aligned pseudotimes which do not better explain the underlzing reference expression dynamics than random can be excluded.

## Installation

Requirements:
`pandas, numpy, scipy, scikit-learn, matplotlib`

to install in a fresh conda environment, do:
```
conda create -n ptalign
conda activate ptalign
conda install numpy pandas scipy scikit-learn matplotlib
```

then navigate to where you want to install ptalign, and do:
```
git clone https://github.com/leoforster/ptalign.git
pip3 install -e ptalign/
```

this will install ptalign as a module accessible for python scripts or notebooks. 
ptalign does not currently have a CLI implemented.

Tested on Ubuntu 18.04.

## Minimal example of a ptalign run

ptalign ships with a toy dataset for demonstration purposes, found in `ptalign/toy_dataset`.
The following steps describe running ptalign on this dataset within a jupyter notebook or similar environment:

First check that ptalign is installed and that the module can be imported:
```
import ptalign
```

Also we will need pandas to manage the single-cell counts tables:
```
import pandas as pd
```

Then we load the supplied toy datasets, including a SVZ reference dataset and pseudotime and a xenografted GBM query dataset:
```
svz_ptime = pd.read_csv('toy_dataset/GSE115600_SVZ_10X_rep1.nocycle_pseudotime.csv.gz', index_col=0)['lineage_ptime']
svz_counts = pd.read_csv('toy_dataset/GSE115600_SVZ_10X_rep1.nocycle_counts.csv.gz', index_col=0)
gbm_counts = pd.read_csv('toy_dataset/GSM7707453_T6_xenograft_SS3_reporter_cells.counts.csv.gz', index_col=0)
```

We transpose the counts tables to have genes in rows:
```
svz_counts = svz_counts.T
gbm_counts = gbm_counts.T
```
Finally we need a pseudotime-predictive gene set, here the SVZ-QAD gene set supplied with the toy data:
```
qadgenes = pd.read_csv('toy_dataset/QAD_242_genes_geneset.csv', index_col=0)['gene'].values
```

Then pseudotime alignment is a matter of calling `ptalign.pseudotime_alignment` with those four data:
```
alignres = ptalign.pseudotime_alignment(refpt=svz_ptime, 
                                        raw_refcounts=svz_counts, 
                                        raw_tumcounts=gbm_counts, 
                                        geneset=qadgenes
                                       )
```
all other arguments to `ptalign.pseudotime_alignment` are optional. You can check the inline docstrings or do `ptalign.pseudotime_alignment?` in a jupyter notebook for full parameter details.

Finally, we can extract the aligned pseudotime from the resulting dictionary:
```
cellpt = alignres['cellpt'] # if permutations=0
cellpt = alignres[0]['cellpt'] # if permutations>0
```

Some example of possible downstream analysis using ptalign pseudotimes are demonstrated in [our paper](https://XXX).

