import torch
from tokenization import tokenizer_sample

def embedding_sample(batch_size=8, max_length=4):
    inputs, targets = tokenizer_sample(batch_size=batch_size,max_length=max_length)

    torch.manual_seed(123)

    vocab_size = 650257
    output_dim = 256
    # to be trained, random initialized
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print("-" * 50)
    print(f"token_embedding_layer.weight: {token_embedding_layer.weight}")
    print(f"token_embedding_layer.weight.shape: {token_embedding_layer.weight.shape}")
    print("-" * 50)
    # look up the embeddings for the input tokens
    token_embeddings = token_embedding_layer(inputs)
    print(f"token_embeddings: {token_embeddings}")
    print(f"token_embeddings.shape: {token_embeddings.shape}")

    context_length = max_length
    position_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    position_embeddings = position_embedding_layer(torch.arange(context_length))
    print("-" * 50)
    print(f"position_embeddings: {position_embeddings}")
    print(f"position_embeddings.shape: {position_embeddings.shape}")

    # input embeddings = token embeddings + position embeddings
    input_embeddings = token_embeddings + position_embeddings
    print("-" * 50)
    print(f"input_embeddings: {input_embeddings}")
    print(f"input_embeddings.shape: {input_embeddings.shape}")

if __name__ == "__main__":
    embedding_sample()
