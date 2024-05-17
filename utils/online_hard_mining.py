import torch


def distance(x, y):
    dist = torch.norm(x - y)
    # dist = np.sqrt(dist)
    
    return dist


# si formano dei triplet seguendo questa relazione: smallest anchor-negative distance, largest anchor-positive distance 
def online_hard_mining(A, P, N, batch_size, device):
    A, P, N = A.cpu(), P.cpu(), N.cpu()
    
    for i in range(batch_size):
        ap_distance = distance(A[i], P[i]).detach().numpy()
        an_distance = distance(A[i], N[i]).detach().numpy()
        
        for j in range(batch_size):
            if j < i: 
                continue 
            
            temp_ap_dist = distance(A[i], P[j]).detach().numpy()
            temp_an_dist = distance(A[i], N[j]).detach().numpy()

            if temp_ap_dist > ap_distance:
                ap_distance = temp_ap_dist

                temp = P[i].clone()
                P[i].copy_(P[j])
                P[j].copy_(temp)
            
            if temp_an_dist < an_distance:
                an_distance = temp_an_dist

                temp = N[i].clone()
                N[i].copy_(N[j])
                N[j].copy_(temp)

    A, P, N = A.to(device), A.to(device), A.to(device)

    return A, P, N
