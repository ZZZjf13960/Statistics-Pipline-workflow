# Statistical Analysis Pipeline

This repository contains a standardized, bilingual (Python & MATLAB) statistical analysis pipeline.
The core principle is **"Diagnosis First, Inference Second"**.

## Project Structure

```
stats_pipeline/
├── data/               # Data files (csv, mat)
├── python/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── eda.py          # Module 1: EDA & Diagnosis
│   │   ├── method_selector.py # Module 2: Method Selection Logic (Future)
│   │   └── inference.py    # Module 3: Statistical Inference (Future)
│   ├── tests/
│   │   └── test_eda_run.py # Tests for Module 1
├── matlab/
│   ├── src/
│   │   ├── EDA_Diagnosis.m # Module 1: EDA & Diagnosis (Class)
│   │   ├── MethodSelector.m # Module 2 (Future)
│   │   └── Inference.m     # Module 3 (Future)
```
