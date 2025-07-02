# P9-20240104-HI
# Table of Contents
code_result/
- Database_properties_ReL: Code and results for comparing chemical spaces and molecular weight distributions.

- Make_database_adapt1: Code and results for making databases.

  • HGB: Code for SHAP-based analysis based on HGB models.

  • data: Code for generating pre-training labels.

- MolGeneration_ReL: Code for reinforcement learning-based molecular generators.

- Pubchem_ReL: Code and results for checking whether molecules are registered in PubChem.

- pkl_files_Deep2: Code for generating pkl files.

- pkl_files_Deep2: Code for generating pkl files.

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
