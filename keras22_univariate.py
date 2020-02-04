import numpy as np

dataset = np.arange(10) + 1
print(dataset)

def split_sequence(sequence, n_steps):
    x, y = list(), list()   
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
        
    return np.array(x), np.array(y)

print(split_sequence('something', 5))