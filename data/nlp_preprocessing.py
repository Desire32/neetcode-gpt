import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        # get unique dicitonary and finding max len
        const_len = len(positive)
        uniques = set()
        max_len = 0

        for st in positive + negative:
            length = len(st.split())
            max_len = max(length, max_len)
            uniques.update(st.split())

        print(f"max len is: {max_len}")

        # convert every word to its lexicographic rank (starting at 1)
        sorted_words = sorted(uniques)
        word_2_rank = {word: rank + 1 for rank, word in enumerate(sorted_words)}

        # pack the results into a single integer tensor of shape 2N×T
        all_seq = []
        for st in positive + negative:
            ranks = [word_2_rank[word] for word in st.split()]
            all_seq.append(torch.tensor(ranks))

        res = torch.nn.utils.rnn.pad_sequence(all_seq, padding_value=0, batch_first=True)

        return res
        
