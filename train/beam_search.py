import torch

def beam_search(model, data, beam_width, n_steps):
    batch_size = data.num_graphs
    beam_candidates = [model(data, n_steps, greedy=False) for _ in range(beam_width)]
    
    log_probs = []
    actions = []
    for i in range(beam_width):
        actions.append(beam_candidates[i][0])
        log_probs.append(beam_candidates[i][1])
        
    print(f'Batch size: {batch_size}')
    # print(f'Actions shape: {[action.shape for action in actions]}')
    # print(f'Log probs shape: {[log_prob.shape for log_prob in log_probs]}')

    best_actions = []
    best_log_probs = []

    for batch_idx in range(batch_size):
        batch_actions = [actions[i][batch_idx] for i in range(beam_width)]
        batch_log_probs = [log_probs[i][batch_idx] for i in range(beam_width)]

        total_log_probs = [log_prob.sum().item() for log_prob in batch_log_probs]

        best_idx = total_log_probs.index(max(total_log_probs))
        best_actions.append(batch_actions[best_idx])
        best_log_probs.append(batch_log_probs[best_idx])

    # Convert lists back to tensors
    best_actions = torch.stack(best_actions)
    best_log_probs = torch.stack(best_log_probs)

    return best_actions, best_log_probs
