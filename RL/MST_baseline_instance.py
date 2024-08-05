import torch
import networkx as nx

def mst_baseline(batch):
    """
    Compute the MST baseline for the given data.
    """
    
    mst_baseline_value = batch.mst_value
    # print(f'batch: {batch}')
    # for data in batch:
    #     print(f'data.mst_value: {data.mst_value}')
    #     mst_baseline_value.append(mst_val.mst_value)
    #     print(f'mst_baseline_value: {mst_baseline_value}')
    return mst_baseline_value