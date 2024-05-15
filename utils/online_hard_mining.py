import numpy as np


def distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T)
    # dist = np.sqrt(dist)
    
    return dist


# si formano dei triplet seguendo questa relazione: smallest anchor-negative distance, largest anchor-positive distance 
def online_hard_mining(A, P, N, model, batch_size):
    print(A.shape)
    print(P.shape) 
    print(N.shape) 

    # f(n) = O(n^2)
    for i in range(batch_size): 
        ai, pi, ni = A[i].detach().cpu().numpy(), P[i].detach().cpu().numpy(), N[i].detach().cpu().numpy()
        ai, pi, ni = np.array(ai), np.array(pi), np.array(ni)
        
        ap_distance = distance(ai, pi)
        an_distance = distance(ai, ni)

        for j in range(batch_size): 
            pj, nj = P[j].detach().cpu().numpy(), N[j].detach().cpu().numpy()
            pj, nj = np.array(pj), np.array(nj)

            temp_ap_dist = distance(ai, pj)
            temp_an_dist = distance(ai, nj)

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
