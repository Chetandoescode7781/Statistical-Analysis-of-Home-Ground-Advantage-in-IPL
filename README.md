# Statistical Analysis of Home-Ground Advantage in the IPL

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![Numpy](https://img.shields.io/badge/Library-Numpy-013243.svg)](https://numpy.org/)

## 📌 Project Overview
This repository contains the statistical modeling and empirical analysis of home-ground advantage across all 19 seasons of the Indian Premier League (2008–2026). Our investigation evaluates whether playing on home grounds yields a statistically significant increase in win probability, higher scoring efficiency, and non-uniform team performance across iconic cricket venues.

---

## 👥 Team Information & Meta-Information
* **Course:** CS215: Data Analysis & Interpretation
* **Group Number:** Group 46
* **Mentor Name:** Yugesh
* **Tracking Document:** [Google Sheets - CS215 Team 46 Tracking Document](https://docs.google.com)

### Team Members & Division of Responsibilities

| Member | Project Role | Workflow & Core Responsibilities | Status |
| :--- | :--- | :--- | :--- |
| **Chetan** | Data Collection & Cleaning Lead | Standardize 19-season team/venue data, build automated Kaggle API ingestion pipeline, compute home vs. away sample metrics. | Completed |
| **Vinay** | Probability Modeling & Win Rates | Estimate home/away win probabilities, fit Bernoulli models for match outcomes, conduct binomial distribution comparisons across venues. | In Progress |
| **Ankit** | Batting & Bowling Metrics Specialist | Extract over-by-over run rates, bowling economy, and powerplay/death-over splits across home and away grounds using `pandas`. | In Progress |
| **Satyam** | Inference, Hypothesis Testing & BI | Execute two-sample t-tests/chi-square tests, construct 95% confidence intervals, synthesize results into an interactive Power BI dashboard. | In Progress |

---

## 🎯 Problem Statement & Sub-Hypotheses

While our primary inquiry evaluates whether playing on home grounds yields a statistically significant win probability increase, we break down our study into three core sub-hypotheses:

1. **$H_1$ (Overall Win Rates):** Home teams exhibit a win probability $P(\text{Win} \mid \text{Home}) > 0.50$ across all 19 seasons using a one-sided Z-test for proportions.
2. **$H_2$ (Scoring & Pitch Advantage):** Mean first-innings scores on home grounds are significantly higher than overall venue averages due to familiar pitch conditions ($p < 0.05$).
3. **$H_3$ (Franchise Variance):** Home advantage is non-uniform across teams—certain teams (e.g., CSK at Chepauk) exhibit a disproportionately higher home win rate compared to others.

---

## 📊 Dataset Overview

We have integrated two comprehensive datasets spanning all 19 seasons of the IPL (2008–2026):

* **260,000+** Ball-by-ball records
* **1,100+** Total IPL matches
* **19** Seasons covered (2008–2026)
* **35+** Unique cricket venues

| Dataset Identifier & Source | Rows | Cols | Key Attributes | Access / Permissions |
| :--- | :--- | :--- | :--- | :--- |
| **Ritesh Ojha IPL Dataset** (GitHub) | ~1,095 | 20 | `match_id`, `season`, `venue`, `team1`, `team2`, `toss_winner`, `winner`, `win_by_runs` | Public Open License |
| **Kaggle IPL Ball-by-Ball (2008–2026)** | ~260,000 | 22 | `match_id`, `inning`, `over`, `ball`, `batter`, `bowler`, `runs_off_bat`, `extras`, `venue` | Public Domain / Kaggle API |

---

## 🚀 Environment Setup & Execution Guide

Follow these steps to clone the repository, activate your Python virtual environment, and run the analysis code files and Jupyter Notebooks.

### 1. Prerequisites & Repository Setup
Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Chetandoescode7781/Statistical-Analysis-of-Home-Ground-Advantage-in-IPL.git
cd Statistical-Analysis-of-Home-Ground-Advantage-in-IPL
```

### 2. Dataset Placement
Ensure that the raw delivery dataset (`ipl_ball_by_ball_2008_2026.csv`) is placed in the expected relative data path. Place it either in the root directory or inside a parent `../ipl_data/` directory depending on your notebook configuration:
```text
├── ipl_data/
│   └── ipl_ball_by_ball_2008_2026.csv
├── data_of_csk.ipynb
├── data_of_kkr.ipynb
├── data_of_mi.ipynb
├── data_of_rcb.py
└── README.md
```

### 3. Virtual Environment (`.venv`) Setup & Activation

#### Creating a Virtual Environment (if not present):
```bash
python -m venv .venv
```

#### Activating `.venv`:
* **Windows (Command Prompt / PowerShell):**
  ```cmd
  .venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies
Install the required packages (`pandas`, `numpy`) within your active virtual environment:
```bash
pip install pandas numpy
```

### 5. Running the Analysis Notebooks & Scripts

#### Running Jupyter Notebooks:
1. Open your code editor (e.g., VS Code or Jupyter Lab).
2. Set the kernel selection to the Python environment inside `.venv`.
3. Open and run team-specific data extraction notebooks:
   * `data_of_csk.ipynb`
   * `data_of_kkr.ipynb`
   * `data_of_mi.ipynb`

#### Running Python Scripts:
Run the standalone analysis scripts directly through your terminal:
```bash
python data_of_rcb.py
```  
 
