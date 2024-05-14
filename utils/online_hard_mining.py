import numpy as np


def distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T)
    # dist = np.sqrt(dist)
    
    return dist


# si formano dei triplet seguendo questa relazione: smallest anchor-negative distance, largest anchor-positive distance 
def online_hard_mining(A, P, N, batch_size):
    for i in range(batch_size): 
        ap_distance = distance(A[i].detach().cpu().numpy(), P[i].detach().cpu().numpy())
        an_distance = distance(A[i].detach().cpu().numpy(), N[i].detach().cpu().numpy())

        for j in range(batch_size): 
            temp_ap_dist = distance(A[i].detach().cpu().numpy(), P[j].detach().cpu().numpy())
            temp_an_dist = distance(A[i].detach().cpu().numpy(), N[j].detach().cpu().numpy())

            if i != j and ap_distance == temp_ap_dist:
                print("temp_ap_dist equal...")

            if i != j and an_distance == temp_an_dist:
                print("temp_an_dist equal...")

            if  temp_ap_dist > ap_distance: 
                ap_distance = temp_ap_dist
                temp = P[i]
                P[i] = P[j]
                P[j] = temp
            
            if temp_an_dist < an_distance: 
                an_distance = temp_an_dist
                temp = N[i]
                N[i] = N[j]
                N[j] = temp

    return A, P, N
