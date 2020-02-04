import numpy as np

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

if __name__=='__main__':
    print(split_sequence('something', 5))

    dataset = np.arange(10, 101, 10)
    print(dataset)

    x, y = split_sequence(dataset, 3)

    for i in range(len(x)):
        print(x[i], y[i])
    

