import torch

def update_mask(demand,capacity,selected,mask,i):
    go_depot = selected.squeeze(-1).eq(0)
    print(f'selected: {selected}\n')
    print(f'go_depot: {go_depot}\n')
    

    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)
    print(f'mask1: {mask1}\n')
    

    if (~go_depot).any():
        mask1[(~go_depot).nonzero(),0] = 0
    print(f'mask1-2: {mask1}\n')

    # if i+1>demand.size(1):
    if (mask1[:, 1:].sum(1) < (demand.size(1) - 1)).any():
        is_done = (mask1[:, 1:].sum(1) >= (demand.size(1) - 1)).float()
        print(f'is_done: {is_done}\n')
        combined = is_done.gt(0)
        print(f'combined: {combined}\n')
        mask1[combined.nonzero(), 0] = 0
        
        '''for i in range(demand.size(0)):
            if not mask1[i,1:].eq(0).any():
                mask1[i,0] = 0'''
                
    print(f'demand value: {demand}\n')
    print(f'capacity value: {capacity}\n')
    a = demand>capacity
    print(f'a: {a}\n')
    mask = a + mask1
    print(f'mask final: {mask}\n')

    return mask.detach(),mask1.detach()

def update_state(demand,dynamic_capacity,selected,c=20):#, depot_visits=0)

    depot = selected.squeeze(-1).eq(0)#Is there a group to access the depot
    print(f'depot: {depot}\n')
    current_demand = torch.gather(demand,1,selected)
    print(f'current_demand: {current_demand}\n')
    dynamic_capacity = dynamic_capacity-current_demand
    print(f'dynamic_capacity: {dynamic_capacity}\n')
    if depot.any():
        dynamic_capacity[depot.nonzero().squeeze()] = c
        # depot_visits += 1

    return dynamic_capacity.detach()#, depot_visits #(bach_size,1)
