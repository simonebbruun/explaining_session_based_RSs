import numpy as np
import random


def repeated_split(test_data, n_pads, pad_value=0):
    actions_grp = []
    for i in range(len(test_data)):
        test_instance = test_data[i]
        test_instance = test_instance[n_pads[i]:,]
        if len(test_instance.shape) == 1:
            unknown_item = test_instance
        else:
            unknown_item = np.sum(test_instance, axis=1)
        test_instance = test_instance[unknown_item!=pad_value]
        actions_grp.append(np.unique(test_instance, axis=0, return_inverse=True)[1])
    
    actions_count = []
    for i in range(len(actions_grp)):
        repeated_instance = actions_grp[i]
        count_instance = np.bincount(repeated_instance)
        count_instance = count_instance[count_instance>1]
        actions_count.append(count_instance)
    
    actions_repeated = []
    for i in range(len(actions_count)):
        actions_repeated.append((actions_count[i]>1).any())
    
    return actions_repeated



# Sum two repeated.
def sum_repeated(feature_attribution, test_data, n_pads, pad_value=0, seed=42):
    np.random.seed(seed)
    actions_select = []
    for i in range(len(test_data)):
        test_instance = test_data[i,]
        test_instance = test_instance[n_pads[i]:,]
        unique, count = np.unique(test_instance, axis=0, return_counts=True)
        if len(test_instance.shape) == 1:
            unknown_item = test_instance
        else:
            unknown_item = np.sum(test_instance, axis=1)
        unique_1 = unique[(count>1) & (unknown_item!=pad_value)]
        unique_1 = unique_1[np.random.choice(unique_1.shape[0], 1),]
        actions_select.append(unique_1)
    
    feature_attribution_sum = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i]
        test_instance = test_data[i]
        sum_indices = tuple(np.random.choice(np.where((test_instance==actions_select[i]).all(axis=1))[0],size=2,replace=False)-n_pads[i])
        indices = [k for k in attribution_instance.keys() if k not in sum_indices]
        indices.append(sum_indices)
        attribution_instance_1 = {}
        for index in indices:
            attribution_instance_1[index] = sum(attribution_instance[(k,)] for k in index)
        feature_attribution_sum.append(attribution_instance_1)
    
    return feature_attribution_sum


# Sum two randoms.
def sum_random(feature_attribution, seed=42):       
    np.random.seed(seed)
    feature_attribution_sum = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i]
        sum_indices = tuple(np.random.choice(np.arange(len(attribution_instance)),size=2,replace=False))
        indices = [k for k in attribution_instance.keys() if k not in sum_indices]
        indices.append(sum_indices)
        attribution_instance_1 = {}
        for index in indices:
            attribution_instance_1[index] = sum(attribution_instance[(k,)] for k in index)
        feature_attribution_sum.append(attribution_instance_1)
    
    return feature_attribution_sum


def sum_repeated_random(feature_attribution, test_data, n_pads, pad_value=0, seed=42):
    actions_select = []
    for i in range(len(test_data)):
        unique, count = np.unique(test_data[i,], axis=0, return_counts=True)
        if len(test_data[i,].shape) == 1:
            unknown_item = unique
        else:
            unknown_item = np.sum(unique, axis=1)
        unique_1 = unique[(count>1) & (unknown_item!=pad_value)]
        unique_1 = unique_1[np.random.choice(unique_1.shape[0], 1),]
        actions_select.append(unique_1)
    
    np.random.seed(seed)
    feature_attribution_sum = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i]
        test_instance = test_data[i]
        index_1 = np.random.choice(np.where((test_instance==actions_select[i]).all(axis=1))[0],size=1,replace=False)-n_pads[i]
        index_2 = np.random.choice(np.delete(np.arange(len(attribution_instance)),index_1),size=1,replace=False)
        sum_indices = (index_1[0],index_2[0])
        indices = [k for k in attribution_instance.keys() if k not in sum_indices]
        indices.append(sum_indices)
        attribution_instance_1 = {}
        for index in indices:
            attribution_instance_1[index] = sum(attribution_instance[(k,)] for k in index)
        feature_attribution_sum.append(attribution_instance_1)
    
    return feature_attribution_sum


def sequential_dependencies(test_data, n_pads, model, class_index=0):
    preds = model.predict(test_data)[np.arange(len(test_data)), class_index]

    sequentials_max = []
    sequentials_min = []
    for i in range(len(test_data)):
        instance = test_data[i:(i+1)]
        
          
        n_feats = instance.shape[1]-n_pads[i]
        pair_indices = []
        for j in range(n_feats):
            for k in range(j + 1, n_feats):
                pair_indices.append((j, k))
    
        instances_flipped = []
        for p in pair_indices:
            instance_flipped = np.copy(instance)
            instance_flipped[:,n_pads[i]+np.array(p),] = instance_flipped[:,n_pads[i]+np.flip(np.array(p)),]
            instances_flipped.append(instance_flipped)
        instances_flipped = np.vstack(instances_flipped)
        
        preds_flipped = model.predict(instances_flipped)[np.arange(len(instances_flipped)), class_index[i]]
        diffs = np.abs(preds[i]-preds_flipped)
        
        sequential = {}
        for index, index_tuple in enumerate(pair_indices):
            sequential[index_tuple] = diffs[index]
        sequential = sorted(sequential.items(), key=lambda kv: -kv[1])
        
        sequential_max = sequential[0][0]
        sequential_min = sequential[-1][0]
        
        sequentials_max.append(sequential_max)
        sequentials_min.append(sequential_min)
    
    return sequentials_max, sequentials_min
    

def swap_actions(feature_attribution, test_data, n_pads, actions):
    feature_attribution_swapped = []
    test_data_swapped = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i].copy()
        test_instance = np.copy(test_data[i])
        action = actions[i]
        indices = np.array(action)
        attribution_instance[(indices[0],)], attribution_instance[(indices[1],)] = attribution_instance[(indices[1],)], attribution_instance[(indices[0],)]
        test_instance[(n_pads[i]+indices)] = test_instance[np.flip(n_pads[i]+indices)]
        feature_attribution_swapped.append(attribution_instance)
        test_data_swapped.append(test_instance)
    feature_attribution_swapped = feature_attribution_swapped
    test_data_swapped = np.stack(test_data_swapped)
    
    return feature_attribution_swapped, test_data_swapped


def select_repeated(feature_attribution, test_data, n_pads, pad_value=0):
    actions_select = []
    for i in range(len(test_data)):
        test_instance = test_data[i,]
        test_instance = test_instance[n_pads[i]:,]
        unique, count = np.unique(test_instance, axis=0, return_counts=True)
        if len(test_instance.shape) == 1:
            unknown_item = unique
        else:
            unknown_item = np.sum(unique, axis=1)
        unique_1 = unique[(count>1) & (unknown_item!=pad_value)]
        actions_select.append(unique_1)
    
        
    feature_attribution_1 = []
    n_repeated = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i]
        test_instance = test_data[i]
        action_instance = actions_select[i]
        n_pad = n_pads[i]
        indices = []
        for j in range(len(action_instance)):
            if len(test_instance.shape) == 1:
                indices.append(np.where((test_instance==action_instance[j]))[0]-n_pad)
            else:
                indices.append(np.where((test_instance==action_instance[j]).all(axis=1))[0]-n_pad)
        indices = np.hstack(indices)
        attribution_instance = {(k,): attribution_instance[(k,)] for k in indices}
        feature_attribution_1.append(attribution_instance)
        n_repeated.append(len(attribution_instance))
    
    return feature_attribution_1, n_repeated


def select_random(feature_attribution, test_data, n_repeated, seed=42):
    random.seed(seed)
    feature_attribution_1 = []
    for i in range(len(feature_attribution)):
        attribution_instance = feature_attribution[i]
        attribution_instance = dict(random.sample(attribution_instance.items(), n_repeated[i]))
        feature_attribution_1.append(attribution_instance)
    
    return feature_attribution_1

