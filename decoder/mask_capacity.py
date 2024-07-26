import torch

def update_state(demand,dynamic_capacity,selected,c=20, depot_visits=0):#dynamic_capacity(num,1)

    depot = selected.squeeze(-1).eq(0)#Is there a group to access the depot
    current_demand = torch.gather(demand,1,selected)
    dynamic_capacity = dynamic_capacity-current_demand
    if depot.any():
        dynamic_capacity[depot.nonzero().squeeze()] = c
        depot_visits += 1

    return dynamic_capacity.detach(), depot_visits #(bach_size,1)

def update_mask(demand,capacity,selected,mask,i):
    go_depot = selected.squeeze(-1).eq(0)

    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    if (~go_depot).any():
        mask1[(~go_depot).nonzero(),0] = 0

    # if i+1>demand.size(1):
    if (mask1[:, 1:].sum(1) < (demand.size(1) - 1)).any():

        is_done = (mask1[:, 1:].sum(1) >= (demand.size(1) - 1)).float()
        combined = is_done.gt(0)
        mask1[combined.nonzero(), 0] = 0
        '''for i in range(demand.size(0)):
            if not mask1[i,1:].eq(0).any():
                mask1[i,0] = 0'''
    a = demand>capacity
    mask = a + mask1

    return mask.detach(),mask1.detach()