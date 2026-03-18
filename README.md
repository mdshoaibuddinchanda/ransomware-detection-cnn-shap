# Detection of Ransomware Attacks Using Processor and Disk Usage Data

Final project repository for ransomware attack detection using processor and disk I/O behavior.

This project now includes:

- A complete Tkinter desktop app for model training/testing
- A notebook with paper-focused enhancements (XAI, robustness, CV, realtime simulation)
- Centralized output generation into `figures/`
- Combined PDF reporting (`all_outputs_report.pdf`)

## Project Status

| Area | Status | Notes |
|---|---|---|
| Core GUI pipeline | Completed | Dataset upload, preprocessing, model runs, comparison, prediction |
| ML/DL model training + evaluation | Completed | SVM, KNN, DT, RF, XGBoost, DNN, LSTM, CNN2D |
| Output artifact management | Completed | PNG/CSV/PDF outputs saved to `figures/` |
| Notebook research enhancements | Completed | SHAP, adversarial robustness, 10-fold stratified CV, realtime simulation |
| Batch automation (`run.bat`) | Completed | Runs `Main.py` then executes notebook |

## Implemented Features Checklist

- [x] GUI-based end-to-end workflow
- [x] Preprocessing and train/test split
- [x] Multiple model benchmarking (ML + DL)
- [x] Confusion matrices and metric reporting
- [x] Test-data prediction using CNN2D model
- [x] Model comparison chart and metrics table export
- [x] SHAP explainability outputs
- [x] Adversarial noise robustness evaluation
- [x] 10-fold stratified cross-validation outputs
- [x] Real-time stream simulation outputs
- [x] Centralized `figures/` output directory
- [x] Unified PDF report generation

## Repository Structure

| Path | Description |
|---|---|
| `Main.py` | Tkinter application with training, evaluation, predictions, and PDF generation button |
| `Ransomware_Paper_Enhancements.ipynb` | Notebook for paper-strength analyses and one-click export |
| `Requirements.txt` | Python package requirements (legacy stack) |
| `run.bat` | Windows automation script: GUI run + notebook execution |
| `Dataset/hpc_io_data.csv` | Main dataset for training/evaluation |
| `Dataset/testData.csv` | External test data used by GUI prediction step |
| `model/*.hdf5` | Saved model weights |
| `model/*.pckl` | Saved training histories |
| `figures/` | Generated figures/tables/PDF artifacts |

## Models Included

| Category | Models |
|---|---|
| Machine Learning | SVM, KNN, Decision Tree, Random Forest, XGBoost |
| Deep Learning | DNN, LSTM, CNN2D |

## Environment Setup

Recommended: Python 3.7 in conda (dependencies are legacy and pinned).

```bash
conda create -n ransomware-detect python=3.7 -y
conda activate ransomware-detect
uv pip install -r Requirements.txt
```

Fallback (if `uv` is not available):

```bash
conda create -n ransomware-detect python=3.7 -y
conda activate ransomware-detect
pip install -r Requirements.txt
```

## Run Options

### Option A: Full Automated Run (Recommended)

```bat
run.bat
```

What `run.bat` does:

1. Ensures `figures/` exists
2. Runs `Main.py`
3. Executes `Ransomware_Paper_Enhancements.ipynb` in-place via `nbconvert`
4. Attempts notebook package install if `nbconvert` execution fails

### Option B: GUI Only

```bash
python Main.py
```

### Option C: Notebook Only

```bash
python -m nbconvert --to notebook --execute Ransomware_Paper_Enhancements.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

## GUI Workflow (Button Order)

1. `Upload Attack Database`
2. `Preprocess & Split Dataset`
3. Run models (any/all):
   - `Run SVM Algorithm`
   - `Run KNN Algorithm`
   - `Run Decision Tree`
   - `Run Random Forest`
   - `Run XGBoost Algorithm`
   - `Run DNN Algorithm`
   - `Run LSTM Algorithm`
   - `Run CNN2D Algorithm`
4. `Comparison Graph`
5. `Predict Attack from Test Data`
6. `Generate PDF Report`

Prediction behavior:

- The app asks you to select the test CSV at runtime (not hardcoded).
- If the selected test CSV contains a `label` column, the app also computes validation metrics and a validation confusion matrix.

## Notebook Enhancements (Implemented)

| Enhancement | Output |
|---|---|
| SHAP Explainability | `shap_summary_plot.png`, `shap_feature_importance.png`, `shap_feature_importance_table.csv` |
| Adversarial Robustness | `adversarial_robustness.png`, `adversarial_robustness_table.csv` |
| 10-Fold Stratified CV | `kfold_cv_accuracy.png`, `kfold_cv_scores.csv`, `kfold_cv_summary.csv` |
| Real-Time Simulation | `realtime_simulation.png`, `realtime_summary.csv` |
| Model Comparison | `model_comparison.png`, `model_comparison_table.csv` |
| Combined Report | `all_outputs_report.pdf` |

Note: Cross-dataset validation is optional in the notebook and is skipped by default unless an explicit external dataset path is provided.

## Current Generated Artifacts

The `figures/` directory currently includes:

| File Type | Files |
|---|---|
| PNG | `dataset_class_distribution.png`, `svm_confusion_matrix.png`, `knn_confusion_matrix.png`, `decision_tree_confusion_matrix.png`, `random_forest_confusion_matrix.png`, `xgboost_confusion_matrix.png`, `dnn_confusion_matrix.png`, `lstm_confusion_matrix.png`, `extension_cnn2d_confusion_matrix.png`, `all_algorithms_performance_graph.png`, `test_prediction_distribution.png`, `baseline_confusion_matrix.png`, `shap_summary_plot.png`, `shap_feature_importance.png`, `adversarial_robustness.png`, `kfold_cv_accuracy.png`, `realtime_simulation.png`, `model_comparison.png`, `test_predictions_validation_confusion_matrix.png` |
| CSV | `algorithm_metrics_long.csv`, `test_predictions.csv`, `test_predictions_summary.csv`, `test_predictions_validation_metrics.csv`, `shap_feature_importance_table.csv`, `adversarial_robustness_table.csv`, `kfold_cv_scores.csv`, `kfold_cv_summary.csv`, `realtime_summary.csv`, `model_comparison_table.csv` |
| PDF | `all_outputs_report.pdf` |

## Key Outcome Snapshot

| Metric | Value |
|---|---|
| 10-Fold Stratified CV Accuracy | `99.08% ± 0.39%` |

## Troubleshooting

| Issue | Suggested Fix |
|---|---|
| Dependency install fails | Use Python 3.7 conda environment and reinstall requirements |
| `xgboost` install issues on Windows | Install Visual C++ build tools or use compatible wheel/environment |
| Notebook execution fails from batch | Run `python -m pip install --upgrade nbconvert nbclient ipykernel` |
| GUI not opening | Verify Tkinter is included in Python installation |
| Plot style error (`seaborn-v0_8-whitegrid`) | Update seaborn/matplotlib or use the built-in style fallback in current code |

## Notes

- Existing `model/*.hdf5` files are loaded if present, reducing retraining time.
- Delete model weight files to force fresh training.
- This project uses older TensorFlow/Keras versions by design (legacy compatibility).

## License

This repository includes a license file. See `LICENSE` for details.
