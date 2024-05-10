import numpy as np


def distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T)
    # dist = np.sqrt(dist)
    
    return dist


# si formano dei triplet seguendo questa relazione: smallest anchor-negative distance, largest anchor-positive distance 
def online_hard_mining(batch, batch_size):
    A, P, N = batch

    for i in range(batch_size): 
        ap_distance = distance(A[i], P[i])
        an_distance = distance(A[i], N[i])

        for j in range(batch_size): 
            temp_ap_dist = distance(A[i], P[j])
            temp_an_dist = distance(A[i], N[j])

            if  temp_ap_dist > ap_distance: 
                ap_distance = temp_an_dist
                temp = P[i]
                P[i] = P[j]
                P[j] = P[i]
            
            if temp_an_dist < an_distance: 
                temp = N[i]
                N[i] = N[j]
                N[j] = temp

    return A, P, N


if __name__ == "__main__": 
    batch = None
    
    online_hard_mining(batch)
