# Research Initiative: LLM-Based Financial Constraints

**Status:** Ongoing Research / Experimental
**Author:** H. Naderi

## Objective
This project aims to construct a novel, more accurate measure of corporate financial constraints. Building on the latest academic research, it moves beyond traditional accounting-based indices to leverage the rich, unstructured information in corporate financial filings. 

The goal is to create a more dynamic and nuanced signal to improve systematic investment strategies by better identifying firms whose investment and financing decisions are materially affected by financing frictions.

## The Academic Consensus
A multi-decade academic debate has shown the limitations of traditional, accounting-based constraint indices (e.g., Kaplan-Zingales, Whited-Wu). Seminal work by *Farre-Mensa & Ljungqvist (2016)* demonstrated their failure to consistently predict actual firm behavior. 

A new consensus has emerged, pioneered by researchers like *Hoberg & Maksimovic (2015)* and *Bodnaruk, Loughran & McDonald (2015)*, suggesting that the most informative and robust signals are found in the text of 10-K filings rather than purely in the balance sheet.

## Methodology: From Keywords to Context
While early textual methods relied on keyword counts (bag-of-words) and static dictionaries, this research applies modern Natural Language Processing (NLP) and Large Language Models (LLMs) to capture semantic meaning and context. 

The methodology follows a hybrid, two-stage process inspired by recent literature (e.g., *Lin & Weagley, 2023*):

### Phase 1: Zero-Shot "Ground Truth" via LLMs
The first step is to create a high-fidelity measure of constraints directly from text.
* **Input:** Management Discussion & Analysis (MD&A) sections of 10-K filings.
* **Engine:** A Zero-Shot Classification LLM.
* **Technique:** By engineering precise prompts (e.g., *"Does this text discuss covenant violation risks?"* or *"Is the firm expressing difficulty in raising equity?"*), the model performs targeted thematic analysis. This captures the contextual richness of managerial disclosures that keyword counting misses.

### Phase 2: Broad Scalability via XGBoost
Direct textual analysis is computationally expensive and limited by the availability of machine-readable filings for older history.
* **Training:** We use the high-quality LLM scores from Phase 1 as "Training Labels."
* **Model:** A non-linear gradient boosting model (XGBoost) is trained to map standard accounting variables to these text-based labels.
* **Output:** The model learns the complex, non-linear relationships between financial data and the "true" constraint status. This results in a quantitative **"FC_Score"** that can be calculated for the entire universe of stocks over deep history.

## Investment Applications
This superior measure of financial constraints is designed for direct integration into the systematic investment process defined in the root of this repository.

### 1. Enhancing Factor Premia (Value & Growth)
The Value premium is likely conditional on a firm's ability to act on its apparent cheapness (limits to arbitrage). By using the `FC_Score`, a portfolio manager can:
* Isolate high Book-to-Market firms that are **financially unconstrained**, creating a sharper Value signal.
* Screen out high-growth firms that lack the capital to fund their own expansion (the "Growth Trap").

### 2. High-Frequency Signal Generation
Corporate filings and news releases are more timely than quarterly financial statements. The `FC_Score` can serve as a higher-frequency indicator of a company's changing financial health, providing an informational edge before accounting ratios update.

### 3. Granular Risk Management
By disaggregating the signal into **Debt Constraint** and **Equity Constraint** sub-scores, the model provides a nuanced view of firm-level risk. This differentiates between firms facing immediate covenant risks versus those facing informational asymmetry challenges in equity markets.