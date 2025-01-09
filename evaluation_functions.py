import numpy as np
from scipy import stats
import random


def pred_diff(feature_attribution, test_data, n_pads, model, pad_value=0, class_index=0):
    preds = model.predict(test_data)[np.arange(len(test_data)), class_index]

    feature_attribution_1 = []
    pred_diffs = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i]
        test_data_masked = []
        attribution_instance_list = []
        for k in attribution_instance.keys():
            test_instance_masked = np.copy(test_data[i])
            test_instance_masked[(n_pads[i]+k),...] = pad_value
            test_data_masked.append(test_instance_masked)
            attribution_instance_list.append(attribution_instance[k])
        feature_attribution_1.append(attribution_instance_list)
        test_data_masked = np.stack(test_data_masked)
        if type(class_index) == int:
            diff = preds[i]-model.predict(test_data_masked)[np.arange(len(test_data_masked)), class_index]
        else:
            diff = preds[i]-model.predict(test_data_masked)[np.arange(len(test_data_masked)), class_index[i]]
        pred_diffs.append(diff)
    
    return feature_attribution_1, pred_diffs


def rank_correlation(feature_attribution, pred_diffs):      
    rank_correlation = []
    for i in range(len(feature_attribution)):
        rank_correlation.append(stats.kendalltau(np.abs(feature_attribution[i]), np.abs(pred_diffs[i]))[0])
    
    return rank_correlation


def attribute_correlation(feature_attribution, pred_diffs):      
    attribute_correlation = []
    for i in range(len(feature_attribution)):
        if len(feature_attribution[i]) < 2:
            attribute_correlation.append(np.nan)
        else:
            attribute_correlation.append(stats.pearsonr(feature_attribution[i], pred_diffs[i])[0])
    
    return attribute_correlation


def data_perturbated(test_data, items, n_pads, model):
    random.seed(42)
    preds = np.argmax(model.predict(test_data), axis=1)
    
    test_data_1 = []
    diffs_1 = []
    top_idx_1 = []
    for i in range(len(test_data)):
        n_items = test_data.shape[1]-n_pads[i]
        p=1/n_items
        test_data_perturbated = []
        diffs = []
        top_idx = []
        while len(test_data_perturbated)<50:
            test_instance_perturbated = np.copy(test_data[i])
            for j in range(n_items):
                r = random.random()
                items_to_select = items[items!=test_instance_perturbated[(n_pads[i]+j)]]
                if r<p:
                    test_instance_perturbated[(n_pads[i]+j),...] = random.choice(items_to_select)
            if (test_instance_perturbated!=test_data[i]).any():
                pred_perturbated = np.argmax(model.predict(np.expand_dims(test_instance_perturbated, axis=0)), axis=1)
                if pred_perturbated==preds[i]:
                    test_data_perturbated.append(test_instance_perturbated)
                    diff = np.where(test_instance_perturbated!=test_data[i],1,0)
                    diffs.append(diff[n_pads[i]:])
                    top_idx.append(preds[i])
        test_data_perturbated = np.stack(test_data_perturbated)
        diffs = np.stack(diffs, axis=-1).T
        test_data_1.append(test_data_perturbated)
        diffs_1.append(diffs)
        top_idx_1.append(top_idx)
    
    return test_data_1, diffs_1, top_idx_1


def relative_input_stability(explanations, perturbation_explanations, perturbation_diffs, n_pads):
    ris = []
    for i in range(100):
        explanation_instance = explanations[i]
        perturbation_explanation_instance = perturbation_explanations[i]
        perturbation_diffs_instance = perturbation_diffs[i]
        fraction = []
        for j in range(50):
            numerator = np.linalg.norm((explanation_instance[n_pads[i]:]-perturbation_explanation_instance[j,n_pads[i]:])/explanation_instance[n_pads[i]:])
            denominator = np.linalg.norm(perturbation_diffs_instance[j])
            fraction.append(numerator/denominator)
        ris.append(np.max(fraction))
    return ris