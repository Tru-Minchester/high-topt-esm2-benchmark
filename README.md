# Frozen ESM-2 embeddings improve high-Topt enzyme classification

This repository contains code and processed benchmark files for a study on high-Topt enzyme classification under strict low-homology benchmarking and external validation.

## Repository structure

Most analysis scripts are currently stored in the `esm2-high-topt-paper/` folder.

## Main scripts

- `01_parse_brenda.py` for extracting temperature annotations from BRENDA JSON
- `02_fetch_uniprot_sequences.py` for retrieving protein sequences from UniProt
- `10_collapse_exact_sequence_labels.py` for collapsing exact duplicate sequences
- `11_build_strict_cdhit40_fasta.py` for preparing strict CD-HIT input
- `13_filter_mixed_label_clusters.py` for removing mixed-label clusters
- `14_extract_aac_dpc_from_strict_nomixed.py` for AAC/DPC extraction
- `15_extract_esm2_from_strict_nomixed.py` for frozen ESM-2 embedding extraction
- `16_evaluate_feature_sets_strict_nomixed.py` for nested cross-validation evaluation
- `17_multi_threshold_sensitivity.py` for threshold sensitivity analysis
- `18_prepare_external_seq2topt.py`, `19_recover_external_novel40.py`, `20_extract_external_esm2.py`, and `21_final_model_external_test.py` for external validation

## Model

Protein language model used: `facebook/esm2_t30_150M_UR50D`

## Data

Raw source data were derived from third-party resources including BRENDA, UniProt, and a Seq2Topt-derived external set. Processed benchmark files and analysis-ready outputs are provided where redistribution is permitted.

## Contact

Corresponding author: [1791820896@qq.com]
