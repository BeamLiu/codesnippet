import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(123)

def manual_calculate_attention_scores(inputs: torch.Tensor, query_idx: int) -> torch.Tensor:
    # the query_idx input token is the query
    query = inputs[query_idx]  

    #create an empty tensor to store the attention scores
    attn_scores = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores[i] = torch.dot(x_i, query)

    attn_weights = torch.softmax(attn_scores, dim=0)

    return attn_scores, attn_weights

attn_scores_2, attn_weights_2 = manual_calculate_attention_scores(inputs, 1)
print(f"Query second input token): {inputs[1]}")
print(f"The second input attention scores for the query with all inputs: {attn_scores_2}")
print("The second input normalized attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

print('-' * 50)

def manual_all_attention_scores(input: torch.Tensor) -> torch.Tensor:
    attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_scores, attn_weights
all_attn_scores, all_attn_weights = manual_all_attention_scores(inputs)
print(f"All attention scores:", all_attn_scores)
print(f"All attention weights:", all_attn_weights)
print(f"All attention weights sum:", all_attn_weights.sum(dim=-1))
print(f"Shape of inputs: {inputs.shape}")
print(f"Shape of all_attn_scores: {all_attn_scores.shape}")
print(f"Shape of all_attn_weights: {all_attn_weights.shape}")

all_context_vecs = all_attn_weights @ inputs
print("All context vectors:", all_context_vecs)


def trainable_self_attention(d_in: int, d_out: int, input_idx: int):
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    keys = inputs @ W_key 
    values = inputs @ W_value

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)

    x_n = inputs[input_idx]
    query_n = x_n @ W_query
    keys_n = x_n @ W_key
    attn_score_nn = query_n.dot(keys_n)
    print(f"The {input_idx+1}th input attention score for the query with itself: query_n dot keys_n = {attn_score_nn}")

    attn_scores_n = query_n @ keys.T # All attention scores for given query
    print(f"The {input_idx+1}th input attention scores for the query with all inputs: query_n @ keys.T = {attn_scores_n}")

    d_k = keys.shape[1]
    attn_weights_n = torch.softmax(attn_scores_n / d_k**0.5, dim=-1)
    print(f"The {input_idx+1}th input attention weights for the query with all inputs: {attn_weights_n}")

    # context is q * k * v
    context_vec_n = attn_weights_n @ values

    return context_vec_n

d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
context_vec_2 = trainable_self_attention(d_in, d_out, 1)
print(f"The 2nd input context vector: {context_vec_2}") 