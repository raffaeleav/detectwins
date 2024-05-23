import ast
import torch
import numpy as np


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


# si selezionano i triplet che rispettano questa relazione: f(ap) < f(an) and f(an) < f(ap) + a
def filter(x, margin):
    # si deserializzano gli array memorizzati precedentemente come stringhe
    a, p, n = np.array(ast.literal_eval(x["Anchor_embs"])), np.array(ast.literal_eval(x["Positive_embs"])), np.array(ast.literal_eval(x["Negative_embs"]))
    a, p, n = torch.tensor(a), torch.tensor(p), torch.tensor(n)

    ap_distance = distance(a, p).detach().numpy()
    an_distance = distance(a, n).detach().numpy()

    return ap_distance < an_distance and an_distance < (ap_distance + margin)


# applico il filtro al df
def offline_semi_hard_mining(df, margin):
    df = df[df.apply(filter, args=(margin,), axis=1)]

    return df 
