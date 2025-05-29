# MoletoSolv
Machine Learning with Physically Inspired Descriptors to Predict Solvation Free Energies of Neutral and Ionic Solutes in Aqueous and Non-aqueous Solvents

## 📘 Project Overview

This project uses machine learning models to predict the **solvation free energies** of **neutral molecules, anions, and cations** in both **aqueous and non-aqueous solvents**.
Each model is constructed based on a series of *physically inspired molecular and solvent descriptors*, covering various factors such as molecular structure, charge distribution, solvent polarity, hydrogen-bond donating/accepting ability, aromaticity, and electronegative halogenic contributions.
Three separate models have been developed for different types of solutes:
- A model for predicting the solvation free energies of **neutral**;
- A model for predicting the solvation free energies of **anions**;
- A model for predicting the solvation free energies of **cations**.

This study extends our previously published model for predicting hydration free energies of neutral solutes (*J. Phys. Chem. Lett. 2023, 14, 1877–1884*), significantly enhancing its applicability to **ionic systems** and **non-aqueous environments**.

## 📂 Data Files

The `/data/` folder contains the following files:

- `prediction_results.csv`: Output file containing predicted solvation free energies for solutes in various solvents. Each row corresponds to a solute–solvent pair.
- `solutes_xyz.zip`: A ZIP archive containing `.xyz` files of solute molecules. All geometries were computed at the **B3LYP/def2-TZVPD** level of theory.
- `solvent_parameters.csv`: Physically inspired descriptors for each solvent.

These files are required for training, prediction, and validation tasks.
