import torch

def update_mask(demand,capacity,selected,mask,i):
    go_depot = selected.squeeze(-1).eq(0)
    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    if (~go_depot).any():
        mask1[(~go_depot).nonzero(),0] = 0

    # This condition checks if there are any nodes the vehicle cannot serve due to capacity constraints
    # cannot_serve_any = (demand * (1 - mask1)).max(dim=1)[0] > capacity.squeeze(-1)
    
    # if cannot_serve_any.any():
    #     mask1[(~go_depot).nonzero(),0] = 0
    #     mask1[torch.logical_not(go_depot), 0] = 0

    if i+1>demand.size(1):
    # if (mask1[:, 1:].sum(1) < (demand.size(1) - 1)).any():
        is_done = (mask1[:, 1:].sum(1) >= (demand.size(1) - 1)).float()
        combined = is_done.gt(0)
        mask1[combined.nonzero(), 0] = 0
                
    # Update the mask for capacity constraints
    a = demand > capacity
    mask = a + mask1
    
    return mask.detach(),mask1.detach()

def update_state(demand,dynamic_capacity,selected,c=20):#, depot_visits=0)

    depot = selected.squeeze(-1).eq(0)#Is there a group to access the depot
    current_demand = torch.gather(demand,1,selected)
    dynamic_capacity = dynamic_capacity-current_demand
    if depot.any():
        dynamic_capacity[depot.nonzero().squeeze()] = c

    return dynamic_capacity.detach()#, depot_visits #(bach_size,1)
