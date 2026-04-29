#  Decision Tree Algorithms — Implementation & Analysis

This project implements and analyzes **7 different decision tree–based machine learning algorithms** on a real-world dataset. It includes **model training, evaluation, and visual representation of trees**, along with a structured report for academic submission.

---

##  Overview

The objective of this project is to:

* Implement multiple decision tree algorithms
* Compare their performance on the same dataset
* Visualize tree structures
* Understand differences between single trees and ensemble methods

---

##  Algorithms Implemented

The following 7 models are implemented:

1. **ID3 (Custom Implementation)**

   * Uses entropy and information gain
   * Implemented manually

2. **CART (Classification and Regression Tree)**

   * Uses Gini Index

3. **Decision Tree (Entropy-based)**

   * Uses entropy (sklearn implementation)

4. **Random Forest**

   * Ensemble of decision trees using bagging

5. **Extra Trees (Extremely Randomized Trees)**

   * Uses random splits for better variance reduction

6. **Gradient Boosting Trees**

   * Sequential boosting technique

7. **AdaBoost (Adaptive Boosting)**

   * Focuses on misclassified samples

---

##  Dataset

* **Breast Cancer Dataset** (from sklearn)
* Binary classification problem
* 30 numerical features

---

##  Project Structure

```bash

├── decision_tree.ipynb     # Main notebook (from Colab)
└── README.md
```

---

##  How to Run

###  Option 1: Run in Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Run the first cell:

```bash
!pip install scikit-learn matplotlib graphviz
!apt-get install graphviz -y
```

3. Click **Runtime → Run All**

---

###  Option 2: Run Locally (VS Code / Terminal)

#### 1. Create virtual environment

```bash
python -m venv venv
```

#### 2. Activate environment

```bash
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

#### 3. Install dependencies

```bash
pip install scikit-learn matplotlib graphviz
```

#### 4. Run notebook

```bash
jupyter notebook
```

---

##  Tree Visualization

###  Single Trees

* CART (Gini)
* Entropy Tree

###  Ensemble Models

Since ensemble models consist of multiple trees, **only one representative tree is visualized**:

* Random Forest → `estimators_[0]`
* Extra Trees → `estimators_[0]`
* Gradient Boosting → `estimators_[0,0]`
* AdaBoost → `estimators_[0]`

###  ID3

* Displayed as a **text-based tree structure**

---

##  Results

| Model             | Accuracy |
| ----------------- | -------- |
| CART              | XX                 |
| Entropy Tree      | XX       |
| Random Forest     | XX       |
| Extra Trees       | XX       |
| Gradient Boosting | XX       |
| AdaBoost          | XX       |

> Replace `XX` with actual values from your notebook output.

---

##  Exporting Tree Images

Use this code inside your notebook:

```python
from sklearn.tree import export_graphviz
import graphviz

def save_tree(model, filename):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(filename, format="png")
```

---


##  Reproducibility

* Random seed used: `random_state = 42`
* Dataset is built-in (no external dependency)
* Compatible with Python 3.9+

---



##  License

This project is created for academic purposes.

---




