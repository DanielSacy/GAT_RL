import torch
import networkx as nx

def mst_baseline(batch):
    """
    Compute the MST baseline for the given data.
    """
    
    mst_baseline_value = batch.mst_value

    return mst_baseline_value.detach()