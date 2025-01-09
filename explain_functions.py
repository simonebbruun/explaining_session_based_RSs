import numpy as np


def merge_overlapping_sets(lsts, output_ints=False):

    def locatebin(bins, n):
        while bins[n] != n:
            n = bins[n]
        return n

    data = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        data.append(set(lst))

    bins = list(range(len(data)))
    nums = dict()

    sets = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        sets.append(set(lst))

    for r, row in enumerate(data):
        for num in row:
            if num not in nums:
                nums[num] = r
                continue
            else:
                dest = locatebin(bins, nums[num])
                if dest == r:
                    continue

                if dest > r:
                    dest, r = r, dest

                data[dest].update(data[r])
                data[r] = None
                bins[r] = dest
                r = dest

    output = []
    for s in data:
        if s:
            if output_ints and len(s) == 1:
                output.append(next(iter(s)))
            else:
                output.append(tuple(sorted(s)))

    return output


def sequential_dependencies(test_data, model, class_index, n_pads, threshold, pad_value=0):
    sequentials = []
    for i in range(len(test_data)):
        instance = test_data[i:(i+1)]
        
          
        n_feats = instance.shape[1]-n_pads[i]
        pair_indices = []
        for j in range(n_feats):
            for k in range(j + 1, n_feats):
                pair_indices.append((j+n_pads[i], k+n_pads[i]))
        
        instances_baseline = []
        instances_flipped = []
        for p in pair_indices:
            instance_baseline = np.zeros(instance.shape)
            instance_baseline[:] = pad_value
            instance_baseline[:,(np.array(p)),] = instance[:,(np.array(p)),]
            instances_baseline.append(instance_baseline)
            instance_flipped = np.copy(instance_baseline)
            instance_flipped[:,np.array(p),] = instance_flipped[:,np.flip(np.array(p)),]
            instances_flipped.append(instance_flipped)
        instances_baseline = np.vstack(instances_baseline)
        instances_flipped = np.vstack(instances_flipped)
        
        preds_instances = model.predict(instances_baseline)[np.arange(len(instances_baseline)), class_index[i]]
        preds_flipped = model.predict(instances_flipped)[np.arange(len(instances_flipped)), class_index[i]]
        diffs = np.abs(preds_instances-preds_flipped)
        
        sequential = {}
        for index, index_tuple in enumerate(pair_indices):
            sequential[index_tuple] = diffs[index]
        sequential = {k: v for k, v in sequential.items() if v > threshold}
        
        sequential = merge_overlapping_sets(set(sequential.keys()))
        
        sequentials.append(sequential)
    
    return sequentials


def repeated_items(test_data, n_pads, sequential_indices):
    repeated = []
    for i in range(len(test_data)):
        records_array = np.copy(test_data[i][n_pads[i]:])
        vals, inverse, count = np.unique(records_array, return_inverse=True, return_counts=True, axis=0)
        
        idx_vals_repeated = np.where(count > 1)[0]
        # vals_repeated = vals[idx_vals_repeated]
        
        if len(sequential_indices[i])>0:
            inverse[sequential_indices[i]-n_pads[i]] = -1
        
        rows, cols = np.where((inverse == idx_vals_repeated[:, np.newaxis]))
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.array(np.split(cols, inverse_rows[1:]))+n_pads[i]
        res = [tuple(r) for r in res if len(r)>0]
        
        repeated.append(res)
    
    return repeated