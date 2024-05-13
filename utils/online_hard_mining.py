import numpy as np


def distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T)
    # dist = np.sqrt(dist)
    
    return dist


# si formano dei triplet seguendo questa relazione: smallest anchor-negative distance, largest anchor-positive distance 
def online_hard_mining(A, P, N, batch_size):
    for i in range(batch_size): 
        ap_distance = distance(A[i].cpu().detach().numpy(), P[i].cpu().detach().numpy())
        an_distance = distance(A[i].cpu().detach().numpy(), N[i].cpu().detach().numpy())

        for j in range(batch_size): 
            temp_ap_dist = distance(A[i].cpu().detach().numpy(), P[j].cpu().detach().numpy())
            temp_an_dist = distance(A[i].cpu().detach().numpy(), N[j].cpu().detach().numpy())

            if  temp_ap_dist > ap_distance: 
                ap_distance = temp_an_dist
                temp = P[i]
                P[i] = P[j]
                P[j] = temp
            
            if temp_an_dist < an_distance: 
                temp = N[i]
                N[i] = N[j]
                N[j] = temp

    return A, P, N
