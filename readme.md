# Quadratic Weighted Kappa is Not Enough for Evaluating Automated Essay Scoring Models

**Accepted at LREC 2026**

## Overview

This repository accompanies the paper *"Quadratic Weighted Kappa is Not Enough for Evaluating Automated Essay Scoring Models"*. It provides comprehensive code, data, and experimental infrastructure for evaluating metrics beyond the standard Quadratic Weighted Kappa (QWK) for Automated Essay Scoring (AES).


## Quick Start

```bash
# Clone the repository
git clone https://github.com/salbatarni/LREC2026_QWK_for_AES.git
cd LREC2026_QWK_for_AES

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib irrCAC krippendorff

# Run research questions
python rq1.py          # Research Question 1
python rq2_1.py        # Research Question 2.1
python rq2_2.py        # Research Question 2.2
python rq3.py          # Research Question 3
```

### Requirements

- Python 3.7+
- NumPy
- Pandas
- SciPy
- scikit-learn
- matplotlib
- irrCAC (for Gwet's AC2)
- krippendorff (for Krippendorff's alpha)


## Project Structure

```
├── data/                          # Datasets
├── results_pickles/               # Experiment results
├── generate_systems.py            # Synthetic system generation
├── metrics.py                     # Evaluation metrics implementations
├── rq1.py                         # Research Question 1
├── rq2_1.py                       # Research Question 2.1
├── rq2_2.py                       # Research Question 2.2
├── rq3.py                         # Research Question 3
└── readme.md                     
```

## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@inproceedings{qwk_insufficient_2026,
  title={Quadratic Weighted Kappa is Not Enough for Evaluating Automated Essay Scoring Models},
  author={Salam Albatarni and Tamer Elsayed},
  booktitle={Proceedings of the 2026 Language Resources and Evaluation Conference (LREC)},
  year={2026}
}
```