import numpy as np


def distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T)
    # dist = np.sqrt(dist)
    
    return dist


def online_hard_mining(batch, batch_size):
    # devo applicare le trasformazioni sulla batch per poi resituirla alla fine
    A, P, N = batch

    for i in range(batch_size): 
        ap_distance = 0
        an_distance = 0
        for j in range(batch_size): 
            temp_ap_dist = distance(A[i], P[j])
            temp_an_dist = distance(A[i], N[j])
            if  temp_ap_dist > ap_distance: 
                print("hit")

    return A, P, N
        

if __name__ == "__main__": 
    batch = None
    
    online_hard_mining(batch)
