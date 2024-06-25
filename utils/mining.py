import os
import ast
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import io


# distanza euclidea per tensori
def tensor_distance(x, y):
    dist = torch.norm(x - y)
    # dist = np.sqrt(dist)
    
    return dist


# distanza euclidea per array np
def array_distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T)
    # dist = np.sqrt(dist)

    return dist


# funzione per generare i vettori di encoding
def get_encoding_csv(mode, fake_data_dir, real_data_dir, model, anc_img_names, device):
    anc_img_names_arr = np.array(anc_img_names)
    encodings = []

    model.eval()

    with torch.no_grad():
        for i in tqdm(anc_img_names_arr, desc="creating encodings..."):
            if mode == "rgb":
                if str(i).startswith("coco"):
                    dir_folder = real_data_dir
                else: 
                    dir_folder = fake_data_dir
                
                a = io.imread(os.path.join(dir_folder, i))
                a = torch.from_numpy(a).permute(2, 0, 1) / 255.0
        
            if mode == "fourier":
                if "real" in str(i):
                    dir_folder = real_data_dir
                else: 
                    dir_folder = fake_data_dir

                a = io.imread(os.path.join(dir_folder, i))
                a = np.expand_dims(a, 0)
                a = torch.from_numpy(a.astype(np.int32)) / 255.0
            
            a = a.to(device)
            a_enc = model(a.unsqueeze(0))
            encodings.append(a_enc.squeeze().cpu().detach().numpy())

        encodings = np.array(encodings)
        encodings = pd.DataFrame(encodings)
        anc_img_names_df = pd.DataFrame(anc_img_names_arr, columns=['Anchor'])
        df_enc = pd.concat([anc_img_names_df, encodings], axis=1)

        return df_enc


# funzione per fare online hard mining delle triplette (non utilizzata)
def online_hard_mining(A, P, N, batch_size, device):
    A, P, N = A.cpu(), P.cpu(), N.cpu()
    
    # si formano dei triplet seguendo questa relazione: d(a, n) < d(a, p)
    for i in range(batch_size):
        # si calcolano le distanze tra anchor-positive e anchor-negative
        ap_distance = tensor_distance(A[i], P[i]).detach().numpy()
        an_distance = tensor_distance(A[i], N[i]).detach().numpy()
        
        for j in range(batch_size):
            # si saltano le immagini già scelte
            if j < i: 
                continue 
            
            temp_ap_dist = tensor_distance(A[i], P[j]).detach().numpy()
            temp_an_dist = tensor_distance(A[i], N[j]).detach().numpy()

            # si confrontano le distanze per trovare distanze anchor-positive maggiori ...
            if temp_ap_dist > ap_distance:
                ap_distance = temp_ap_dist

                temp = P[i].clone()
                P[i].copy_(P[j])
                P[j].copy_(temp)
            
            # ... e anchor-negative minori
            if temp_an_dist < an_distance:
                an_distance = temp_an_dist

                temp = N[i].clone()
                N[i].copy_(N[j])
                N[j].copy_(temp)

    # si assegnano i tensori alla gpu perchè la funzione viene richiamata durante l'addestramento
    A, P, N = A.to(device), A.to(device), A.to(device)

    return A, P, N


# si selezionano i triplet che rispettano questa relazione: d(a, p) < d(a, n) < d(a, p) + alpha
def filter(x, margin):
    # si deserializzano gli array memorizzati precedentemente come stringhe
    a, p, n = np.array(ast.literal_eval(x["Anchor_embs"])), np.array(ast.literal_eval(x["Positive_embs"])), np.array(ast.literal_eval(x["Negative_embs"]))
    a, p, n = torch.tensor(a), torch.tensor(p), torch.tensor(n)

    ap_distance = tensor_distance(a, p).detach().numpy()
    an_distance = tensor_distance(a, n).detach().numpy()

    return ap_distance < an_distance < (ap_distance + margin)


# funzione per fare offline semi-hard mining delle triplette 
def offline_semi_hard_mining_alpha(df, margin):
    df = df[df.apply(filter, args=(margin,), axis=1)]

    return df 


# implementazione alternativa semi-hard mining
def offline_semi_hard_mining_beta(model, margin, fake_dir_path, real_dir_path, fake_metadata, real_metadata, output_dir_path):
    fake_df = pd.read_csv(fake_metadata, usecols=['image_path'], nrows=100000)
    real_df = pd.read_csv(real_metadata, usecols=['image_path'], nrows=100000)

    fake_df = pd.DataFrame({'fake': fake_df['image_path']})
    real_df = pd.DataFrame({'real': real_df['image_path']})

    df_enc_fake = get_encoding_csv(model, fake_df["fake"], fake_dir_path)
    df_enc_real = get_encoding_csv(model, real_df["real"], real_dir_path)

    enc_arr_fake = df_enc_fake.iloc[:, 1:].to_numpy()
    enc_arr_real = df_enc_real.iloc[:, 1:].to_numpy()

    dataset_semi_hard_fake = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])
    dataset_semi_hard_real = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    fake_index = 0
    real_index = 0

    # si considerano al più una volta come anchor le immagini fake
    for i in tqdm(range(len(df_enc_fake))):
        A_embs = enc_arr_fake[i, :]
        path_anchor = df_enc_fake.iloc[i]['fake']

        # si genera casualmente l'indice per selezionare l'immagine positive
        idp = random.randint(0, len(df_enc_fake) - 1)

        # si verifica che l'immagine anchor sia diversa dall'immagine positive appena selezionata
        while torch.equal(torch.tensor(A_embs), torch.tensor(enc_arr_fake[idp, :])):
            idp = random.randint(1, len(df_enc_fake) - 1)

        # si genera casualmente l'indice per selezionare l'immagine negative
        idn = random.randint(0, len(df_enc_real) - 1)

        path_positive = df_enc_fake.iloc[idp]["fake"]
        path_negative = df_enc_real.iloc[idn]["real"]

        # si calcolano le distanze euclidee anchor-positive e anchor-negative
        dist_positive = array_distance(enc_arr_fake[i, :], enc_arr_fake[idp, :])
        dist_negative = array_distance(enc_arr_fake[i, :], enc_arr_real[idn, :])

        # la tripletta verrà selezionata solo se soddisfa il criterio semi-hard
        if (dist_positive < dist_negative) and (dist_negative < (dist_positive + margin)):
            dataset_semi_hard_fake.loc[fake_index] = [
                path_anchor,
                path_positive,
                path_negative,
            ]

        fake_index = fake_index + 1

    # si usa lo stesso procedimento con le immagini real
    for i in tqdm(range(len(df_enc_real))):
        A_embs = enc_arr_real[i, :]
        path_anchor = df_enc_real.iloc[i]["real"]

        idp = random.randint(0, len(df_enc_real) - 1)

        while torch.equal(torch.tensor(A_embs), torch.tensor(enc_arr_real[idp, :])):
            idp = random.randint(1, len(df_enc_real) - 1)

        idn = random.randint(0, len(df_enc_fake) - 1)

        path_positive = df_enc_real.iloc[idp]["real"]
        path_negative = df_enc_fake.iloc[idn]["fake"]

        dist_positive = array_distance(enc_arr_real[i, :], enc_arr_real[idp, :])
        dist_negative = array_distance(enc_arr_real[i, :], enc_arr_fake[idn, :])

        if (dist_positive < dist_negative) and (dist_negative < (dist_positive + margin)):
            dataset_semi_hard_real.loc[real_index] = [
                path_anchor,
                path_positive,
                path_negative,
            ]

        real_index = real_index + 1

    df = pd.concat([dataset_semi_hard_fake, dataset_semi_hard_real], ignore_index=True)
    df.to_csv(output_dir_path, index=False)

    return df


# funzione per fare offline hard mining delle triplette 
def offline_hard_mining(model, fake_dir_path, real_dir_path, fake_metadata, real_metadata, output_dir_path): 
    fake_df = pd.read_csv(fake_metadata, usecols=['image_path'], nrows=15000)
    real_df = pd.read_csv(real_metadata, usecols=['image_path'], nrows=15000)

    fake_df = pd.DataFrame({'fake': fake_df['image_path']})
    real_df = pd.DataFrame({'real': real_df['image_path']})

    df = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    df_enc_real = get_encoding_csv(model, real_df["real"], real_dir_path)
    df_enc_fake = get_encoding_csv(model, fake_df["fake"], fake_dir_path)

    enc_arr_real = df_enc_real.iloc[:, 1:].to_numpy()
    enc_arr_fake = df_enc_fake.iloc[:, 1:].to_numpy()

    last_idx = 0

    for i in tqdm(range(enc_arr_real.shape[0]), desc="scorro gli anchor real"):
        max_dist = float('-inf')
        max_idx = 0
        min_dist = float('inf')
        min_idx = 0

        # si cerca il positive con distanza massima
        for j in range(enc_arr_real.shape[0]):
            if i == j:
                continue

            dist = array_distance(enc_arr_real[i : i+1, :], enc_arr_real[j : j+1, :])

            if dist > max_dist:
                max_dist = dist
                max_idx = j

        # si cerca il negative con distanza minima
        for k in range(enc_arr_fake.shape[0]):
            dist = array_distance(enc_arr_real[i : i+1, :], enc_arr_fake[k : k+1, :])

            if dist < min_dist:
                min_dist = dist
                min_idx = k

        # una volta trovate le immagini positive e negative, si costruisce la tripletta
        df.loc[i] = [
            df_enc_real['real'].iloc[i],
            df_enc_real['real'].iloc[max_idx],
            df_enc_fake['fake'].iloc[min_idx]
        ]

        last_idx = i

    # la stessa cosa viene fatta usando come anchor le immagini fake
    for i in tqdm(range(enc_arr_fake.shape[0]), desc="scorro gli anchor fake"):
        max_dist = float('-inf')
        max_idx = 0
        min_dist = float('inf')
        min_idx = 0

        # cerco come immagine positive, quella con la distanza massima
        for j in range(enc_arr_fake.shape[0]):
            if i == j:
                continue

            dist = array_distance(enc_arr_fake[i : i+1, :], enc_arr_fake[j : j+1, :])

            if dist > max_dist:
                max_dist = dist
                max_idx = j

        for k in range(enc_arr_real.shape[0]):
            dist = array_distance(enc_arr_fake[i : i+1, :], enc_arr_real[k : k+1, :])

            if dist < min_dist:
                min_dist = dist
                min_idx = k

        # si costruisce la tripletta e la si inserisce come riga nel dataframe
        df.loc[last_idx + i] = [
            df_enc_fake['fake'].iloc[i], 
            df_enc_fake['fake'].iloc[max_idx], 
            df_enc_real['real'].iloc[min_idx]
        ]
       
    df.to_csv(output_dir_path,index=False)
