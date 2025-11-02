# Class Imbalance Analysis Report
**Team Member:** Johann | **Date:** October 2025 | **Status Update**

---

## Initial Situation: Class Imbalance Problem

The dataset exhibits significant class imbalance across 27 product type categories, with representation ranging from 0.9% (class 1180) to 12% (class 2583).

**Key Statistics - Before:**
- Most represented class: **2583** (12.0%)
- Least represented class: **1180** (0.9%)
- Critical imbalance ratio: **~13:1**
- Classes with <2% representation: **8 classes**

---
## Approach & Team Discussion

Our team explored class imbalance mitigation strategies, with ongoing discussion about implementation timing:

**Two Perspectives:**
- **Perspective 1:** Build full preprocessing pipeline with all balancing code included
- **Perspective 1:** Focus on minimal viable code, add complexity as needed ongoing

**Our Journey:**
1. Analyzed distribution patterns across all 27 categories
2. Evaluated multiple rebalancing techniques (oversampling, undersampling, SMOTE, hybrid)
3. Created comparison framework to assess impact with Linear Regression model

**Goal:** Maintain modular class-based structure with reusable functions, allowing flexible implementation without overengineering at this stage

---
## Preview of Class Balance Improvements Comparison:

================================================================================
SUMMARY: ALL PIPELINES
================================================================================

       Method  Accuracy  Precision   Recall  F1-Score  Training Size
     Baseline  0.779852   0.786024 0.743645  0.759897          67932
Undersampling  0.727096   0.699587 0.729557  0.706652          16497
 Oversampling  0.779616   0.754219 0.775004  0.761341         220509
        SMOTE  0.779969   0.751687 0.774126  0.759958         220509
Class Weights  0.772551   0.747266 0.770216  0.754058          67932
  Balanced RF  0.746055   0.726727 0.746635  0.726641          67932

**Key Findings:**
- Baseline performance is similiar to other methods
- Oversampling shows slight better performance (F1)
- Note: Results are based on raw dataset with Linear Regression (not cleaned)

---

## Implementation: Representative Code Structure

```python
class ClassImbalanceHandler:
    """Modular class for handling imbalanced datasets"""
    
    def __init__(self, method='oversampling'):
        self.method = method
    
    def process(self, df):
        """Apply selected balancing strategy"""
        if self.method == 'oversampling':
            return self._random_oversample(df)
        elif self.method == 'undersampling':
            return self._random_undersample(df)
        elif self.method == 'smote':
            return self._apply_smote(df)
```

**architecture decision:** Class-based structure provides flexibility for future enhancement while keeping current implementation lean and maintainable.

---
## Next Steps & Open Questions

**Status:** Early exploration phase

**Remaining Work:**
- finalize text preprocessing (HTML cleaning, translation via LLM/API)
- finalize special character handling strategy
- analyze impact of missing descriptions (35%) on model performance
- thorough check of rebalancing effects on classification metrics