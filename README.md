# explaining_session_based_RSs
This repository contains the source code and the [appendix](https://github.com/simonebbruun/explaining_session_based_RSs/blob/main/ECIR2025_Explaining_SBRSs_appendix.pdf) for Feature Attribution Explanations of Session-based Recommendations.

## Requirements

- Python
- NumPy
- Pandas
- TensorFlow
- Scikit-learn
- Pickle
- Matplotlib
- Shap
- Lime
- Math
- Iterrtools
- Random
- Scipy


## Usage

1. Preprocess the datasets using   
   1_data_preprocessing_30music.py  
   1_data_preprocessing_diginetica.py  
2. Train the recommendation models using   
   2_train_model_gru4rec.py  
   2_train_model_sasrec.py  
3. Evaluate the effectiveness of the recommendation models using   
   3_evaluate_model.py  
4. Generate baseline explanations using   
   4_generate_explanations_deep_shap.py  
   4_generate_explanations_ig.py  
   4_generate_explanations_lime.py  
5. Analyze the effect of sequential dependencies and repeated interactions using   
   5_preliminary_analysis.py  
6. Generate session-based occlusion explanations using   
   6_generate_explanations_sb_occlusion.py  
7. Evaluate the explanations using   
   7_evaluate_explanations.py  
