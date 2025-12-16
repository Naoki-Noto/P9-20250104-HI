# P9-20250104-HI
# Table of Contents
code_result/

- Additional_baybe: Testing the search efficiency of another bayesian optimization algorithm.

- Additional_edbo_RF: Bayesian optimization with a random forest-based surrogate model.

- Additional_publicdata: Optimization for identifying the optimal OPSs in publicly available dataset.

- OPSsearch_OCH2F: TL-based optimization for identifying the optimal OPSs in PR13.

- OPSsearch_OoS: TL-based and RF-based optimizations for identifying OPS61, an effective photosensitizer not included in the source domain, in PR2.

- OPSsearch_P: TL-based optimization for identifying the optimal OPSs in PR14.

- OPSsearch_Si: TL-based optimization for identifying the optimal OPS in PR15.

- OPSsearch_edbo: Bayesian optimization (surrogate model: GPR) for identifying the optimal OPSs in PR13–PR15.

- OPSsearch_edbo_sc: Bayesian optimization using the standardized descriptor set.

- OPSsearch_others: TL-based optimization for identifying the optimal OPS in PR1–PR12.

- OPSsearch_random: Random search in PR1–PR15.

- OPSsearch_woTL: RF-based and LR-based optimizations without TL for identifying the optimal OPSs in PR1–PR15.

- Regression: Construction of RF-based and TrAB-based regression models in PR13–PR15.

- analysis: Code for analysing the results.

- demo: Demonstration of RF-based and TL-based optimizations (target: PR14).

- results/results_OoS/results_others/results_reg: Storage of ML results.

==========================================================================
  
data/

- data_60: Datasets involving 60 OPSs used for TL-based optimization.

- data_61: Datasets involving 61 OPSs used for extrapolative screening.

- data_BO: Datasets involving 60 OPSs used for BO.

- data_BO_sc: Datasets involving 60 OPSs used for BO, where descriptors are standardized.

- data_MF: Datasets involving 60 OPSs used for TL-based optimization, where Morgan fingerprint-based descriptors are used instead of DFT-based descriptors.

- data_woTL: Datasets involving 60 OPSs used for non-TL-based optimization.

==========================================================================

environment: Environments for performing each code are stored in this directory./

For performing BO (edbo.yml): Python (3.7.12) was used as a language, and used packages were edbo (0.1.0) and pandas (1.3.5).

For using general ML investigations (OPSsearch.yml): Python (3.9.20) was used as a language, and used packages were adapt (0.4.4), matplotlib (3.9.2), numpy (1.26.4), pandas (2.2.3), rdkit (2024.9.5), scikit-learn (1.5.2), scipy (1.12.0), seaborn (0.13.2), and xgboost (2.1.4).
