#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import torch

def block_ctx_for_query_attention_mask(sequence, special_tok_positions, device='cpu'):
    """
    Creates a custom attention mask where:
    - The first j tokens attend to all previous tokens (causal attention).
    - Tokens from j to i can only attend to tokens in the range [j, i].
    - Tokens from i onward attend to all previous tokens (causal attention).
    
    Args:
        sequence_length: Length of the input sequence.
        j: The index where the first region ends.
        i: The index where the second region ends and the third region starts.
        device: Device to place the mask on.
    
    Returns:
        attention_mask: A (sequence_length, sequence_length) attention mask.
    """

    sequence_length = sequence.shape[-1]
    special_tok_positions = special_tok_positions[0]

    start_of_query, start_of_completion = special_tok_positions[-2], special_tok_positions[-1] + 1

    j = start_of_query
    i = start_of_completion

    # Create a standard causal mask for the whole sequence
    causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), device=device))

    # Initialize a zero mask
    custom_mask = torch.zeros((sequence_length, sequence_length), device=device, requires_grad=False)

    # Region 1: First j tokens follow normal causal attention
    custom_mask[:j, :j] = causal_mask[:j, :j]

    # Region 2: j-th to i-th tokens can attend only to themselves
    custom_mask[j:i, j:i] = causal_mask[:i-j, :i-j]  # Offset causal mask for this region

    # Region 3: Tokens after i can attend to all previous tokens (0 to i)
    # We want to copy the entire causal mask up to the i-th position
    custom_mask[i:, :] = causal_mask[i:, :]

    custom_mask = custom_mask.unsqueeze(0).unsqueeze(0) # get back bs dimension

    
    return custom_mask


def blockwise_attention_mask(sequence, special_tok_positions, device):
    """
    Create a blockwise attention mask based on the given indices.
    
    Args:
    - sequence_length (int): The total length of the sequence.
    - indices (list of int): A list of indices defining block boundaries. Each consecutive pair defines a block.
    
    Returns:
    - mask (torch.Tensor): A (sequence_length, sequence_length) mask, where tokens can only attend to previous tokens within their block.
    """
    
    sequence_length = sequence.shape[-1]
    
    # for query ids, we allow it to include the last [/INST]
    indices[-1] = indices[-1] + 1

    # Initialize a full mask with zeros (disallowing all attention initially)
    mask = torch.zeros(sequence_length, sequence_length, requires_grad=False)
    
    # Add a leading 0 and trailing sequence_length to the index list to define blocks
    assert indices[0] == 0
    indices = indices + [sequence_length]
    
    # Loop through each block
    for start, end in zip(indices[:-1], indices[1:]):
        # For each block, create a causal mask (allow only attending to previous tokens in the same block)
        block_size = end - start
        block_mask = torch.tril(torch.ones(block_size, block_size))
        
        # Place the block mask into the appropriate position in the full mask
        mask[start:end, start:end] = block_mask
    
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)

    return mask

# # Example usage
# sequence_length = 10  # Length of the sequence
# indices = [0, 3, 5, 7]  # Block boundaries at index positions [0:3], [3:7], [7:10]

# # Create the block attention mask
# attention_mask = blockwise_attention_mask(sequence_length, indices)

# print(attention_mask)


def hierarchical_attention_mask(sequence, special_tok_positions, device):
    """
    Create a blockwise attention mask based on the given indices.
    
    Args:
    - sequence_length (int): The total length of the sequence.
    - indices (list of int): A list of indices defining block boundaries. Each consecutive pair defines a block.
    
    Returns:
    - mask (torch.Tensor): A (sequence_length, sequence_length) mask, where tokens can only attend to previous tokens within their block.
    """
    
    sequence_length = sequence.shape[-1]
    indices = special_tok_positions[0].tolist()

    # for query ids, we allow it to include the last [/INST]
    indices[-1] = indices[-1] + 1

    # Initialize a full mask with zeros (disallowing all attention initially)
    mask = torch.zeros(sequence_length, sequence_length, requires_grad=False)
    
    # Add a leading 0 and trailing sequence_length to the index list to define blocks
    assert indices[0] == 0
    indices = indices + [sequence_length]
    
    # Loop through each block
    for start, end in zip(indices[:-1], indices[1:]):
        # For each block, create a causal mask (allow only attending to previous tokens in the same block)
        block_size = end - start
        block_mask = torch.tril(torch.ones(block_size, block_size))
        
        # Place the block mask into the appropriate position in the full mask
        mask[start:end, start:end] = block_mask

        if start == indices[-2]: # for query
            continue
        elif start == indices[-1]: # for completion
            continue

    
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)

    return mask

# # # Example usage
# sequence = torch.tensor([[66, 91,  2, 97, 22, 52, 89, 54,  8, 80]])  # Length of the sequence
# indices = [0, 3, 5, 7]  # Block boundaries at index positions INST [0:3), CONTEXT [3:5), QUERY [5:7], COMPLETION (7:10]

# # Create the block attention mask
# attention_mask = hierarchical_attention_mask(sequence, indices, device='cpu')

# print(attention_mask)


import torch

def hierarchical_attention_mask(sequence, special_tok_positions, device='cuda', decoding=False):
    """
    Create a custom attention mask for a sequence with block structure and special attention rules for the last block.
    
    Args:
    - sequence_length (int): Total length of the sequence.
    - indices (list of int): A list of indices defining block boundaries. Each consecutive pair defines a block.
    
    Returns:
    - mask (torch.Tensor): A (sequence_length, sequence_length) mask, where tokens can attend based on block structure.
    """

    sequence_length = sequence.shape[-1]

    if isinstance(special_tok_positions, list):
        indices = special_tok_positions
    else:         
        indices = special_tok_positions[0].tolist()

    indices = indices[:-1] # remove [/INST], for query+completion, allow to attention all previous block's last token

    # for query ids, we allow it to include the last [/INST]
    # indices[-1] = indices[-1] + 1

    # Initialize a full attention mask with zeros (no attention allowed initially)
    mask = torch.zeros(sequence_length, sequence_length, requires_grad=False).to(device)
    
    # Add leading 0 and trailing sequence_length to the indices list to define block boundaries
    indices = [0] + indices + [sequence_length]
    
    # Loop through all blocks except the last one
    for start, end in zip(indices[:-2], indices[1:-1]):
        # For each block, create a causal mask (attend only to previous tokens in the same block)
        block_size = end - start
        block_mask = torch.tril(torch.ones(block_size, block_size, requires_grad=False)).to(device)
        mask[start:end, start:end] = block_mask
    
    # Handle the last block separately
    last_block_start = indices[-2]
    last_block_end = indices[-1]
    last_block_size = last_block_end - last_block_start
    
    # Create a causal mask for the last block itself
    last_block_mask = torch.tril(torch.ones(last_block_size, last_block_size, requires_grad=False))
    mask[last_block_start:last_block_end, last_block_start:last_block_end] = last_block_mask
    
    # Allow tokens in the last block to attend the last token of all previous blocks
    for prev_block_end in indices[1:-1]:
        mask[last_block_start:last_block_end, prev_block_end - 1] = 1
    
    mask = mask.to(device).requires_grad_(False)

    if decoding:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Set 0s to -inf
        mask = mask.masked_fill(mask == 1, 0.0)  # Set 1s to 0
        return mask
    else:
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    
    
# Example usage
# sequence_length = 15  # Length of the sequence
# indices = [3, 5, 7, 10, 12]  # Block boundaries at index positions [0:3], [3:5], [5:7], [7:10], [10:15]

# # Create the custom causal attention mask
# attention_mask = hierarchical_attention_mask(sequence_length, indices)

# print(attention_mask)
