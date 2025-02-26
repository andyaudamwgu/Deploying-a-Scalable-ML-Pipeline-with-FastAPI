# Model Card: Income Prediction Model

## Model Details
- **Developers**: Mienau, created as part of a scalable ML pipeline project.
- **Date**: February 25, 2025.
- **Version**: 1.0.
- **Type**: Random Forest Classifier.
- **Framework**: Scikit-learn.
- **License**: Open-source (assumed; specify if different).

## Intended Use
- **Purpose**: Predict whether an individual’s income exceeds $50K/year based on census data.
- **Users**: Data scientists, students, or developers exploring ML pipelines.
- **Use Case**: Educational analysis or prototype for income prediction tools; not for production without further validation.

## Training Data
- **Dataset**: UCI Adult Income dataset (`census.csv`), containing 32,561 instances.
- **Features**: 14 features (6 numeric, 8 categorical):
  - Numeric: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
- **Label**: `salary` (`<=50K` or `>50K`).
- **Preprocessing**: Numeric features unchanged, categorical features one-hot encoded via `OneHotEncoder`, label binarized (0 for `<=50K`, 1 for `>50K`) via `LabelBinarizer`.

## Evaluation Data
- **Split**: 20% test set (6,513 instances), split randomly with `random_state=42`.
- **Distribution**: Approximately 76% `<=50K`, 24% `>50K` (based on full dataset: 24,720 `<=50K`, 7,841 `>50K`).

## Metrics
- **Overall Performance**:
  - **Precision**: 0.7419 - 74.19% of predicted `>50K` instances are correct.
  - **Recall**: 0.6384 - 63.84% of actual `>50K` instances are identified.
  - **F1 Score**: 0.6863 - Balances precision and recall, accounting for class imbalance.
- **Slice Performance**: Detailed metrics for each categorical feature value are in `slice_output.txt`. Examples:
  - `education: Bachelors` (Count: 1,053): Precision 0.7523, Recall 0.7289, F1 0.7404.
  - `workclass: Private` (Count: 4,578): Precision 0.7376, Recall 0.6404, F1 0.6856.
  - `native-country: United-States` (Count: 5,870): Precision 0.7392, Recall 0.6321, F1 0.6814.
  - Small slices (e.g., `native-country: Yugoslavia`, Count: 2) often show F1 1.0000 due to low sample size.

## Quantitative Analysis
- The model excels with larger, well-represented groups (e.g., `Masters`, F1: 0.8409) but struggles with small or underrepresented slices (e.g., `7th-8th`, F1: 0.0000). Precision typically exceeds recall, indicating conservative `>50K` predictions.

## Ethical Considerations
- **Bias**: The dataset reflects 1994 census biases (e.g., race, sex imbalances), potentially skewing predictions.
- **Fairness**: Lower recall in slices like `Never-married` (0.4272) vs. `Married-civ-spouse` (0.6900) suggests disparity; further analysis needed.
- **Usage**: Not recommended for real-world decisions (e.g., hiring) without bias mitigation.

## Caveats and Recommendations
- **Limitations**: Uses a basic RandomForest without tuning; numeric features aren’t scaled (noted in `process_data`).
- **Improvements**: Scale continuous features, tune hyperparameters, or explore ensemble methods to boost F1, especially for minority slices.