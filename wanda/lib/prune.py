import time 
import heapq 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from .ablate import AblateGPT 

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
# from transformers.utils import (
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_flash_attn_2_available,
#     is_flash_attn_greater_or_equal_2_10,
#     logging,
#     replace_return_docstrings,
# )
from transformers.models.llama.configuration_llama import LlamaConfig


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



def compute_attention_output_with_pruned_weights(
    inps,  # Input tensor (seq_len, hidden_dim)
    attention_mask,  # Attention mask (batch_size, 1, seq_len, seq_len)
    position_ids,  # Position IDs (batch_size, seq_len)
    pruned_weights,  # Dictionary of pruned weight matrices
    num_attention_heads=32,  # Number of attention heads
    hidden_dim=4096  # Hidden dimension size
):
    q_proj = pruned_weights['self_attn.q_proj']
    k_proj = pruned_weights['self_attn.k_proj']
    v_proj = pruned_weights['self_attn.v_proj']
    o_proj = pruned_weights['self_attn.o_proj']
    batch_size, seq_len, _ = inps.shape
    head_dim = hidden_dim // num_attention_heads
    
    output = torch.zeros_like(inps, device=inps.device)
    
    batch_size = 1
    i = 0
    # Step 1: Compute Q, K, V for slice `inps[i]`
    Q = torch.matmul(inps[i], q_proj.T)  # Shape: [seq_len, hidden_dim]
    K = torch.matmul(inps[i], k_proj.T)  # Shape: [seq_len, hidden_dim]
    V = torch.matmul(inps[i], v_proj.T)  # Shape: [seq_len, hidden_dim]

    # Step 2: Reshape Q, K, V for multi-head attention
    Q = Q.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)  # Shape: [num_heads, seq_len, head_dim]
    K = K.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)  # Shape: [num_heads, seq_len, head_dim]
    V = V.view(seq_len, num_attention_heads, head_dim).transpose(0, 1)  # Shape: [num_heads, seq_len, head_dim]

    # Step 3: Compute attention scores (scaled dot-product attention)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / head_dim**0.5  # Shape: [num_heads, seq_len, seq_len]
    
    # Apply attention mask
    attn_scores = attn_scores + attention_mask[0]  # Broadcasting the mask to [num_heads, seq_len, seq_len]

    # Step 4: Compute attention probabilities
    attn_probs = F.softmax(attn_scores, dim=-1)  # Shape: [num_heads, seq_len, seq_len]

    # Step 5: Compute attention output
    context = torch.matmul(attn_probs, V)  # Shape: [num_heads, seq_len, head_dim]

    # Step 6: Concatenate the outputs from all heads
    context = context.transpose(0, 1).contiguous().view(seq_len, hidden_dim)  # Shape: [seq_len, hidden_dim]

    # Step 7: Apply the output projection
    output = torch.matmul(context, o_proj.T)  # Shape: [seq_len, hidden_dim]
    pruned_weights.clear()

    return output
    
def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
            
def construct_permutation_matrix(index, device):

    num_cols = index.size(0)
    P_hard = torch.zeros((num_cols, num_cols), device=device)
    P_hard[torch.arange(num_cols), index] = 1.0
    return P_hard

def gumbel_sinkhorn(log_alpha, n_iters, tau, noise_factor=1.0, epsilon=1e-6, debug=True):
    """
    Applies the Gumbel-Sinkhorn algorithm to generate a soft permutation matrix.

    Args:
        log_alpha (torch.Tensor): Logits matrix of shape [n, n].
        n_iters (int): Number of Sinkhorn iterations.
        tau (float): Temperature parameter.
        noise_factor (float): Scaling factor for Gumbel noise.
        epsilon (float): Small constant for numerical stability.
        debug (bool): If True, prints debug information.
    
    Returns:
        torch.Tensor: Soft permutation matrix of shape [n, n].
    """
    
    
    # Ensure computations are in float32 for numerical stability
    log_alpha = log_alpha.float()
    
    # Step 1: Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_alpha) + epsilon) + epsilon)
    if debug:
        #print("Gumbel Noise Sampled:")
        #print_tensor_stats(gumbel_noise, "gumbel_noise")
        check_tensor(gumbel_noise, "gumbel_noise", debug)
    
    # Step 2: Add noise to the logits
    M = (log_alpha + noise_factor * gumbel_noise)
    if debug:
        #print("Logits After Adding Gumbel Noise (M):")
        #print_tensor_stats(M, "M")
        check_tensor(M, "M", debug)
    
    # Step 3: Scale by temperature
    M_scaled = M / tau
    if debug:
        #print(f"Logits Scaled by Tau (M_scaled): tau={tau}")
        #print_tensor_stats(M_scaled, "M_scaled")
        check_tensor(M_scaled, "M_scaled", debug)
    
    # Step 4: Subtract max per row for numerical stability
    row_max, _ = M_scaled.max(dim=1, keepdim=True)
    M_scaled = M_scaled - row_max
    if debug:
        #print("Logits After Subtracting Row Max (M_scaled):")
        #print_tensor_stats(M_scaled, "M_scaled - max")
        check_tensor(M_scaled, "M_scaled - max", debug)
    
    # Step 5: Exponentiate to get S
    S = torch.exp(M_scaled)
    if debug:
        #print("Exponentiated Scaled Logits (S):")
        #print_tensor_stats(S, "S")
        check_tensor(S, "S", debug)
    
    # Step 6: Sinkhorn iterations
    for iteration in range(n_iters):
        # Row normalization
        row_sum = S.sum(dim=1, keepdim=True)
        S = S / (row_sum + epsilon)
        if debug:
            #print(f"After Row Normalization Iteration {iteration+1}:")
            #print_tensor_stats(S, f"S_row_norm_iter_{iteration+1}")
            check_tensor(S, f"S_row_norm_iter_{iteration+1}", debug)
        
        # Column normalization
        col_sum = S.sum(dim=0, keepdim=True)
        S = S / (col_sum + epsilon * 100)  # Increased epsilon for column normalization
        if debug:
            # print(f"After Column Normalization Iteration {iteration+1}:")
            # print_tensor_stats(S, f"S_col_norm_iter_{iteration+1}")
            check_tensor(S, f"S_col_norm_iter_{iteration+1}", debug)
    
    
    
    return S

def print_tensor_stats(tensor, tensor_name="Tensor"):
    """
    Prints statistics of a tensor.

    Args:
        tensor (torch.Tensor): The tensor to inspect.
        tensor_name (str): Name of the tensor for identification.
    """
    print(f"{tensor_name} Stats:")
    # print(f"  Shape: {tensor.shape}")
    # print(f"  Dtype: {tensor.dtype}")
    # print(f"  Min: {tensor.min().item()}")
    # print(f"  Max: {tensor.max().item()}")
    # print(f"  Mean: {tensor.mean().item()}")
    # print(f"  Std: {tensor.std().item()}")
    # print(f"  NaN Count: {torch.isnan(tensor).sum().item()}")
    # print(f"  Inf Count: {torch.isinf(tensor).sum().item()}")

def check_tensor(tensor, tensor_name="Tensor", debug=False):
    """
    Checks for NaNs and Infs in a tensor and optionally prints a warning.

    Args:
        tensor (torch.Tensor): The tensor to check.
        tensor_name (str): Name of the tensor for identification.
        debug (bool): If True, prints warnings.
    """
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        if debug:
            # print(f"WARNING: {tensor_name} contains NaNs or Infs!")
            # print(f"  NaN Count: {nan_count}")
            # print(f"  Inf Count: {inf_count}")
            raise ValueError(f"{tensor_name} has NaNs or Infs!")

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    module_name = args.module_name
    permutate_mode = args.permutate_mode
    layer_id = args.layer_id

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        
        
    
    layers = model.model.layers
    # -----------------------------------------------------------
    # Permutation parameters
    num_epochs = 100  # Set the number of training epochs
    learning_rate = 0.5  # Learning rate for optimizer
    tau = 3  # Temperature parameter for Gumbel-Sinkhorn
    sinkhorn_iterations = 30  # Number of Sinkhorn iterations
    epsilon = 1e-8  # Small epsilon to prevent numerical issues
    
    if permutate_mode == 'eval':num_epochs = 1
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        print(subset)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)


        wrapped_layers = {}
        for name in subset:
            if permutate_mode in ('lora', 'full'):
                if module_name in name:
                    wrapped_layers[name] = WrappedGPT(subset[name])
            if permutate_mode == 'eval':
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        
        # Example of correct computation
        outputs_before_pruning = []
        for j in range(args.nsamples):
            output = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            outputs_before_pruning.append(output)

        outputs_before_pruning = torch.stack(outputs_before_pruning)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                #outputs_before_pruning[j] = layer(inps[j].unsqueeze(0),attention_mask=attention_mask,position_ids=position_ids,)[0]
        for h in handles:
            h.remove()


        # Store the original weights to restore later
        original_weights = {}
        for name in subset:
            if permutate_mode in ('lora', 'full'):
                if module_name in name:
                    original_weights[name] = subset[name].weight.data.clone()
            if permutate_mode == 'eval':
                original_weights[name] = subset[name].weight.data.clone()
            
            # Initialize learnable logits M for each module in the subset
        if i == layer_id or permutate_mode == 'eval':
            rank = 1
            M_dict = {}
            U_dict = {}
            V_dict = {}
            original_weight = {}
            for name in subset:
                if permutate_mode in ('lora', 'full'):
                    if module_name in name:
                        print(name)
                        W = subset[name].weight.data  # Shape: [output_dim, input_dim]
                        original_weight[name] = W
                        num_cols = W.shape[1]
                        if permutate_mode == 'full':
                            M_dict[name] = nn.Parameter(torch.zeros(num_cols, num_cols, device=device))
                        if permutate_mode == 'lora':
                            U_dict[name] = nn.Parameter(torch.randn(num_cols, rank, device=device) * 0.0001)
                            V_dict[name] = nn.Parameter(torch.randn(rank, num_cols, device=device) * 0.0001)
                if permutate_mode == 'eval':
                    print(name)
                    W = subset[name].weight.data  # Shape: [output_dim, input_dim]
                    original_weight[name] = W
                    num_cols = W.shape[1]
            if permutate_mode == 'full':
                optimizer = optim.Adam(M_dict.values(), lr=learning_rate)
            # Assuming U_dict, V_dict, and M_dict all contain nn.Parameter objects
            if permutate_mode == 'lora':
                params_to_optimize = list(U_dict.values()) + list(V_dict.values())
                optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

            # output_without_pruned_weights = compute_attention_output_with_pruned_weights(
            #         inps, attention_mask, position_ids, original_weight, num_attention_heads=32, hidden_dim=4096
            #     )
            best_loss = float('inf')
            best_permutations = {}
            
            for epoch in range(num_epochs):
                if permutate_mode in ('lora', 'full'):
                    optimizer.zero_grad()
                original_weights = {}
                current_permutations = {}
                for name in subset:
                    original_weights[name] = subset[name].weight.data.clone()
                # Dictionary to store permuted and pruned weights and masks
                W_perm_dict = {}
                W_pruned_dict = {}
                W_mask_dict = {}
                #initial_M = M_dict['mlp.up_proj'].clone().detach()
                
                total_preserved_metric = 0.0
                total_preserved_weight = 0.0
                # Step 1: Compute permutations and apply pruning for each module
                pruned_weights_dict = {}
                for name in subset:
                    if module_name in name or permutate_mode == 'eval':
                        #print(name)
                        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                        
                        W = subset[name].weight.data
                        #print(W.shape)
                        if permutate_mode in ('lora', 'full'):
                            num_cols = W.shape[1]
                            if permutate_mode == 'lora':
                                U = U_dict[name]
                                V = V_dict[name]
                                M = torch.matmul(U.half(), V.half()).half()
                                M = torch.clamp(M, min=-10.0, max=10.0)
                            if permutate_mode == 'full':
                                M = M_dict[name].half()
                                M = torch.clamp(M, min=-10.0, max=10.0)
                            
                            S_soft = gumbel_sinkhorn(M, sinkhorn_iterations, tau, epsilon=epsilon)
                            S_soft = torch.clamp(S_soft, min=1e-8, max=1 - 1e-8)
                            # Step 2: Compute the hard permutation matrix P_hard
                            with torch.no_grad():
                                S_cpu = S_soft.detach().cpu().numpy()
                                if not np.isfinite(S_cpu).all():
                                    print("Invalid values in S_cpu")
                                    print(S_cpu)
                                    exit()
                                row_ind, col_ind = linear_sum_assignment(-S_cpu)
                                P_hard = torch.zeros_like(S_soft)
                                P_hard[row_ind, col_ind] = 1.0

                            # Modify P_hard to allow gradient flow
                            P_hard = (P_hard - S_soft).detach() + S_soft

                            P_hard = P_hard.to(W.dtype)
                            
                            current_permutations[name] = P_hard.detach().cpu()
                        
                        # # Load the best permutation matrices
                        if permutate_mode == 'eval':
                            #print(name)
                            module = None
                            if name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'): module = 'self'
                            if name == 'mlp.gate_proj': module = 'gate'
                            if name == 'mlp.up_proj': module = 'up'
                            if name == 'mlp.down_proj': module = 'down'
                            
                            if module in ('self', 'gate', 'up'):
                                best_permutations = torch.load(f'./p_weight/Layer{i}_{module}_P_matrix.pt')
                                
                                P_hard = best_permutations[name]
                                P_hard = P_hard.to(W.dtype)
                            if module in ('down'):
                                sorted_idx = torch.sort(torch.sum(W_metric, dim=0))[1]

                                # Channel reallocation (permutation)
                                index = torch.zeros_like(sorted_idx)
                                for ii in range(1, prune_m + 1):
                                    if ii % 2 == 1:
                                        index[
                                            ii - 1 :: prune_m
                                        ] = sorted_idx[
                                            int(W_metric.shape[1] * (ii - 1) / prune_m) : int(
                                                W_metric.shape[1] * ii / prune_m
                                            )
                                        ]
                                    else:
                                        index[
                                            ii - 1 :: prune_m
                                        ] = sorted_idx[
                                            int(W_metric.shape[1] * (ii - 1) / prune_m) : int(
                                                W_metric.shape[1] * ii / prune_m
                                            )
                                        ].flip(0)

                                # Construct P_hard permutation matrix
                                P_hard = construct_permutation_matrix(index, device=W.device)
                                P_hard = P_hard.to(W.dtype)
                                row_sums = P_hard.sum(dim=1)
                                col_sums = P_hard.sum(dim=0)
                                
                        W_device = W.device
                        P_hard = P_hard.to(W_device)
                        # Permute the columns of W using P_hard
                        W_perm = torch.matmul(W, P_hard)
                        W_metric = W_metric.to(torch.float16)
                        W_metric_perm = torch.matmul(W_metric, P_hard)
                        
                        W_mask = (torch.zeros_like(W_metric_perm) == 1)
                        for ii in range(0, W_metric_perm.shape[1], prune_m):
                            tmp = W_metric_perm[:, ii:(ii + prune_m)].float()
                            _, indices = torch.topk(tmp, prune_n, dim=1, largest=False)
                            W_mask[:, ii:(ii + prune_m)].scatter_(1, indices, True)
                        # Apply the pruning mask to the permuted W
                        # Apply pruning mask
                        W_metric_preserved = W_metric_perm.clone()
                        W_metric_preserved[W_mask] = 0
                        W_metric_preserved = W_metric_preserved.to(torch.float32)  # Ensure precision
                        total_preserved_metric += W_metric_preserved.sum()
                        
                        #print(W_metric_preserved)
                        
                        W_pruned_perm = W_perm.clone()
                        W_pruned_perm[W_mask] = 0
                        # Map the permuted and pruned weights back to the original order
                        # Compute the inverse permutation
                        # inv_perm = torch.argsort(torch.tensor(col_ind)).to(device)
                        # Map W_pruned_perm back to original order
                        W_pruned = torch.matmul(W_pruned_perm, P_hard.t())
                        # Store pruned weights
                        W_pruned_dict[name] = W_pruned
                        total_preserved_weight += W_pruned.sum()
                        pruned_weights_dict[name] = W_pruned
                        
                        
                        

                        subset[name].weight.data = W_pruned.clone()

                # output_with_pruned_weights = compute_attention_output_with_pruned_weights(
                #     inps, attention_mask, position_ids, pruned_weights_dict, num_attention_heads=32, hidden_dim=4096
                # )
                # print("Out:")
                # print(output_with_pruned_weights.shape)
                # Forward pass after pruning with permutation
                outputs_with_perm = []  # Use a list instead of a preallocated tensor
                for j in range(args.nsamples):
                    output = layer(inps[j].unsqueeze(0),attention_mask=attention_mask,position_ids=position_ids,)[0]
                    outputs_with_perm.append(output)
                # Stack outputs into a single tensor
                outputs_with_perm = torch.stack(outputs_with_perm)  # Shape: [nsamples, ...]
                
                # Verify gradient through differences_norm_with_perm
                differences_norm_with_perm = []
                for j in range(args.nsamples):
                    diff_norm = torch.norm(outputs_with_perm[j] - outputs_before_pruning[j])
                    differences_norm_with_perm.append(diff_norm)

                # # Stack into a tensor
                differences_norm_with_perm = torch.stack(differences_norm_with_perm)
                differences_norm_with_perm.requires_grad_(True)
                

                # # Compute loss
                loss_diff = torch.mean(differences_norm_with_perm)
                #print(f'Diff: {loss_diff}')
                #loss = -total_preserved_metric/200000 + torch.mean(differences_norm_with_perm)
                loss = -total_preserved_metric / 100
                #print(loss.device)
                #loss = output_with_pruned_weights.sum()
                #diff = output_with_pruned_weights - output_without_pruned_weights
                #print(diff)
                #loss = torch.abs(diff).sum()
                #loss = -total_preserved_weight
                #loss = M.sum()
                #print(f"Loss.requires_grad: {loss.requires_grad}")  # Should be True
                if permutate_mode in ('lora', 'full'):
                    if loss_diff.item() < best_loss:
                        best_loss = loss_diff.item()
                        best_permutations = current_permutations
                        torch.save(best_permutations, f'./p_weight/Layer{i}_{module_name}_P_matrix.pt')  # Save the best permutations to a file
                        print(f"New best loss: {best_loss:.6f}, saving permutation matrices.")
                    
                    print(f'Layer {i}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
                    loss.backward(retain_graph=True)

                    # Check gradients
                    for name in M_dict:
                        if permutate_mode == 'full':
                            grad_norm = M_dict[name].grad.norm().item() if M_dict[name].grad is not None else None
                            print(f'Permutation parameter {name}, Gradient Norm: {grad_norm}')
                        if permutate_mode == 'lora':
                            # For each layer in M_dict, access both U and V
                            U_grad_norm = M_dict[name]['U'].grad.norm().item() if M_dict[name]['U'].grad is not None else None
                            V_grad_norm = M_dict[name]['V'].grad.norm().item() if M_dict[name]['V'].grad is not None else None
                            
                            # Print the gradient norms for both U and V
                            print(f'Layer {name}:')
                            print(f'  U Gradient Norm: {U_grad_norm}')
                            print(f'  V Gradient Norm: {V_grad_norm}')

                    #final_M = M_dict['mlp.up_proj'].clone().detach()
                    #difference = torch.norm(final_M - initial_M)
                    #print(f'Total change in permutation weights: {difference.item()}')
                    optimizer.step()
                    # Zero the gradients
                    optimizer.zero_grad()
                    for name in subset:
                        subset[name].weight.data = original_weights[name].clone()
            
                    del W_metric, W, M, S_soft, P_hard, W_perm, W_metric_perm, W_mask
                    torch.cuda.empty_cache()
                    # for h in handles:
                    #     h.remove()
                    # del handles
        # -------------------------------------------------------------------------------------------------------------------------    
        # Step 1: Pruning without permutation
        # for name in subset:
        #     if "self" in name or permutate_mode == 'eval':
        #         print(f"Pruning layer {i} name {name} without permutation")
        #         W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

        #         W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        #         if prune_n != 0:
        #             # structured n:m sparsity
        #             for ii in range(W_metric.shape[1]):
        #                 if ii % prune_m == 0:
        #                     tmp = W_metric[:,ii:(ii+prune_m)].float()
        #                     W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

        #         subset[name].weight.data[W_mask] = 0  ## set weights to zero 
        # outputs_after_pruning_no_perm = torch.zeros_like(outs)
        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outputs_after_pruning_no_perm[j] = layer(inps[j].unsqueeze(0),attention_mask=attention_mask,position_ids=position_ids,)[0]
        # differences_norm_no_perm = []
        # for j in range(args.nsamples):
        #     diff_norm = torch.norm(outputs_after_pruning_no_perm[j] - outputs_before_pruning[j])
        #     differences_norm_no_perm.append(diff_norm.item())
            
        # # Restore original weights before permutation
        # for name in subset:
        #     if "self" in name:
        #         subset[name].weight.data = original_weights[name].clone()
        # -------------------------------------------------------------------------------------------------------------------------    
        
        # -------------------------------------------------------------------------------------------------------------------------  
        # print(f"Layer {i} difference norms:")
        # for j in range(args.nsamples):
        #     print(
        #         f"Sample {j}: Without Permutation = {differences_norm_no_perm[j]:.6f}, "
        #         f"With Permutation = {differences_norm_with_perm[j]:.6f}"
        #     )
        outputs_after_pruning = torch.zeros_like(outs)
        for j in range(args.nsamples):
            with torch.no_grad():
                outputs_after_pruning[j] = layer(inps[j].unsqueeze(0),attention_mask=attention_mask,position_ids=position_ids,)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        
        # differences_norm = []
        # for j in range(args.nsamples):
        #     # print(outputs_before_pruning[j])
        #     # print(outputs_after_pruning[j])
        #     diff_norm = torch.norm(outputs_after_pruning[j] - outputs_before_pruning[j])
        #     differences_norm.append(diff_norm.item())

        # print(f"Differences for layer {i}:")
        # for j in range(args.nsamples):
        #     print(f"Sample {j}: Difference Norm = {differences_norm[j]}")
            
        #break

    
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()