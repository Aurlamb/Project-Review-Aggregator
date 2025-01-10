# Project Review Aggregator: iWanIt 1.0

**Author:** Aurèle Lambert  
**Date:** January 2025  
**Purpose:** This repository is part of the IronHack AI Bootcamp and contains the implementation of the **iWanIt 1.0** project. The project aims to analyze product reviews and generate child-friendly sales pitches to help kids persuade parents to make purchases.

---

## Overview

iWanIt 1.0 is a prototype **mobile ad generator** designed to:
1. Analyze parent-written product reviews to identify the most persuasive arguments.
2. Organize and cluster products into meaningful categories.
3. Generate kid-friendly sales pitches that encourage parents to make purchases.

The system includes three main modules:
- **JoyScanner:** Sentiment analysis of reviews.
- **ToyFinder:** Product clustering based on categories.
- **BeggerMaker:** Generation of persuasive sales pitches.

---

## Features

### 1. JoyScanner
- **Purpose:** Analyzes the sentiment of product reviews to identify the most positive comments.
- **Implementation:** 
  - Used VADER for rule-based sentiment analysis (88.96% accuracy).
  - Used PySentimiento for transformer-based sentiment analysis (90.45% accuracy).
  - Calculates sentiment scores using both titles and reviews.

### 2. ToyFinder
- **Purpose:** Clusters products into categories for better targeting.
- **Implementation:** 
  - Preprocessed data (lemmatization, TF-IDF).
  - Experimented with 4 clustering methods; final model uses brand, name, and categories.
  - Created six clusters, e.g., "Kindles & e-readers," "Kids and toys."

### 3. BeggerMaker
- **Purpose:** Generates sales pitches tailored for children to persuade parents.
- **Implementation:**
  - Uses a fine-tuned **Llama 3.2** model with prompt engineering.
  - Prioritizes products based on ratings, reviews, and sentiment weights.
  - Generates child-like speech with elements of begging and urgency.

---

## Dataset

- **Source:** Kaggle Amazon reviews dataset.
- **Preprocessing Steps:**
  1. Merged 3 CSV files containing 74k reviews.
  2. Removed duplicates and irrelevant data, resulting in 48k reviews.
  3. Added a sentiment score column and performed clustering on product names.
  4. Generated a final catalog with 81 unique products.

---

## Architecture

The project integrates sentiment analysis, clustering, and natural language generation. The system builds on existing tools and models, enhancing them for the project's goals.

---

## Demos

### Demo 1: Kindle
- **Child pitch:**  
  "Mom, I really want this Kindle. It’s perfect for reading because it’s small, the screen is bright, and it’s easy to use. It will last forever because the battery lasts 3 days. My friend Emily has one, and my grandma says it’s like reading a paper book. Can we please get one?"  
  [Watch the demo](https://docs.google.com/file/d/1Tctlbxfg4g66dvstB-cnN7ZaLsQPAvr4/preview)

---

### Demo 2: Paintball Gun
- **Child pitch:**  
  "Mom, can I please get this paintball gun? It's so realistic, it feels like a real gun, and it's perfect for my school project. My teacher said we can use it for a project, and I want to get it so I can be the best. Plus, I saw my friend Timmy's dad has one, and he said it's really good."  
  [Watch the demo](https://docs.google.com/file/d/1Th5X__8KmhMq2_PXstA_DJkVBjqt0A9_/preview)

---

### Demo 3: Amazon Tablet
- **Child pitch:**  
  "Mom, can we get one of those tablets? People say it’s easy to use. My friend Emma, she can learn new things and play games! Plus, it’s really lightweight and durable."  
  [Watch the demo](https://docs.google.com/file/d/1TapSENX2lL90TFXG5SXRJw6IL714sMKl/preview)

---

## Challenges

- Balancing simplicity with accuracy in sentiment analysis.
- Efficient clustering for product categorization.
- Generating child-like language with persuasive elements.

---

## Requirements

- Python 3.9+
- Libraries:
  - `pandas`, `numpy`, `sklearn`, `VADER`, `PySentimiento`
- Pre-trained models:
  - Llama 3.2
  - Talking_Face_Avatar

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Aurlamb/Project-Review-Aggregator.git
   cd Project-Review-Aggregator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main pipeline:
   ```bash
   python main.py
   ```
4. View generated outputs in the `/Demos` directory.

---

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
