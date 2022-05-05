def pairing(data, seq_len=6):
    x = [] 
    y = []
    
    for i in range(0, (data.shape[0] - (seq_len+1)), seq_len+1):
        
        seq =np.zeros( (seq_len, data.shape[1]) )
        for j in range(seq_len):
            seq[j] = data.values[i+j]

            x.append(seq.flatten())
            y.append( data['T (degC)'][i+seq_len] )
    
    return np.array(x), np.array(y)