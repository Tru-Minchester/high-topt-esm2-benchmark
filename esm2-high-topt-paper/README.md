# Frozen ESM-2 embeddings improve high-Topt enzyme classification under strict low-homology benchmarking and external validation

This repository contains code and processed benchmark files for a study on high-Topt enzyme classification using frozen ESM-2 embeddings under strict low-homology benchmarking and external validation.

## Contents

- `01_parse_brenda.py` to extract temperature annotations from a local BRENDA JSON release
- `02_fetch_uniprot_sequences.py` to retrieve protein sequences from UniProt
- `10_collapse_exact_sequence_labels.py` to collapse exact duplicate sequences
- `11_build_strict_cdhit40_fasta.py` to prepare strict CD-HIT input
- `13_filter_mixed_label_clusters.py` to remove mixed-label clusters
- `14_extract_aac_dpc_from_strict_nomixed.py` to extract AAC/DPC features
- `15_extract_esm2_from_strict_nomixed.py` to extract frozen ESM-2 embeddings
- `16_evaluate_feature_sets_strict_nomixed.py` for nested cross-validation evaluation
- `17_multi_threshold_sensitivity.py` for 50/60/70 °C threshold analysis
- `18_prepare_external_seq2topt.py`, `19_recover_external_novel40.py`, `20_extract_external_esm2.py`, and `21_final_model_external_test.py` for external validation

These are the main scripts used in the manuscript workflow; additional intermediate scripts and helper outputs are included in the repository.

## Data

Raw source data were derived from third-party resources, including a local BRENDA JSON release for temperature annotations, UniProt accession-based sequence retrieval, and a Seq2Topt-derived external test set. Because some source records originate from third-party databases, this repository provides processed benchmark files and analysis-ready outputs where redistribution is permitted.

## Pretrained model

Protein language model used in this study: `facebook/esm2_t30_150M_UR50D`

## Reproducibility

The scripts are organized in approximate execution order from `01_` to `21_`.

## How to reproduce

A typical workflow is:

1. Parse BRENDA temperature annotations
2. Retrieve UniProt sequences
3. Build sequence FASTA files and run CD-HIT / CD-HIT-2D
4. Recover the strict no-mixed internal benchmark
5. Extract AAC, DPC, and frozen ESM-2 features
6. Run nested cross-validation and threshold sensitivity analysis
7. Prepare the external dataset and run final external validation

## Contact

Corresponding author: 1791820896@qq.com