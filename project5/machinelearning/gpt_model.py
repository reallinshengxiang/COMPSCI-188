"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


References:
1) "minGPT" implemented by Andrej Karpathy
https://github.com/karpathy/minGPT
2) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
3) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
from models import Attention


class Transformer_Block(nn.Module):
    """
    This class builds the basic transformer block.
    """
    def __init__(self, n_embd, block_size):
        super().__init__()

        self.attn_block = Attention(n_embd, block_size)
        self.norm_1 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        """YOUR CODE HERE"""

       

class Character_GPT(nn.Module):
   
    def __init__(self, block_size, n_embd, n_layer, vocab_size):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd) #Embedding layer, think of this as similar to a linear layer

        
        self.transformer_blocks = nn.ModuleList([Transformer_Block(n_embd, block_size) for _ in range(n_layer)]) #You can treat this as a python list
        self.norm = nn.LayerNorm(n_embd) #Normalization Layer
        self.output_layer = nn.Linear(n_embd, vocab_size, bias=False)



    def get_loss(self, input, target):
        output = self(input)
        return F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=-1)
        
    def forward(self, input):
        """
        This function should take in an input representing a sequence of characters, and output
        an array representing the likelihood of any character appearing next.

        All necessary layers have been initialized for you in the __init__() function, you should pay special
        attention to the self.transformer_blocks variable. Since we have multiple transformer blocks in our
        final model, you will have to pass the input through every object in this list.
        """
        b, t = input.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        """YOUR CODE HERE"""


    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # optionally crop the logits to only the top k options

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
