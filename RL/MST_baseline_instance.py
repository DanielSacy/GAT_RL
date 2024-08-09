import torch
import networkx as nx

def mst_baseline(batch):
    """
    Compute the MST baseline for the given data.
    """
    
    mst_baseline_value = batch.mst_value
    print(f'mst_baseline_value: {mst_baseline_value}\n')

    return mst_baseline_value.detach()