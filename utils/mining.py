import os
import ast
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import io


def distance(x, y):
    dist = torch.norm(x - y)
    # dist = np.sqrt(dist)
    
    return dist


def euclidean_dist(img_enc, anc_enc_arr):
    # dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T))
    dist = np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T)
    # dist = np.sqrt(dist)
    return dist


# funzione per generare i vettori di encoding
def get_encoding_csv(model, anc_img_names, dirFolder, mode, fake_dir_path, real_dir_path, DEVICE):
    anc_img_names_arr = np.array(anc_img_names)
    encodings = []

    model.eval()

    with torch.no_grad():
        for i in tqdm(anc_img_names_arr):

            if mode == "rgb":
                # serve per trovare correttamente l'immagine
                if "tt" in i:
                    dirFolder = fake_dir_path
                    A = io.imread(os.path.join(dirFolder, i))
                else:
                    dirFolder = real_dir_path
                    A = io.imread(os.path.join(dirFolder, i))

                A = torch.from_numpy(A).permute(2, 0, 1) / 255.0

            if mode == "grey_scale":
                A = io.imread(os.path.join(dirFolder, i))

                A = np.expand_dims(A, 0)
                A = torch.from_numpy(A.astype(np.int32)) / 255.0

            A = A.to(DEVICE)
            A_enc = model(A.unsqueeze(0))
            encodings.append(A_enc.squeeze().cpu().detach().numpy())

        encodings = np.array(encodings)
        encodings = pd.DataFrame(encodings)
        df_enc = pd.concat([anc_img_names, encodings], axis=1)

        return df_enc


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

    return ap_distance < an_distance < (ap_distance + margin)


# applico il filtro al df
def offline_semi_hard_mining(df, margin):
    df = df[df.apply(filter, args=(margin,), axis=1)]

    return df 


# funzione per effettuare hard mining 
def offline_hard_mining(model, fake_dir_path, real_dir_path, fake_metadata, real_metadata, output_dir_path): 
    taming_transformer = pd.read_csv(fake_metadata, usecols=['image_path'], nrows=15000)
    coco = pd.read_csv(real_metadata, usecols=['image_path'], nrows=15000)

    real_df = pd.DataFrame({'real': coco['image_path']})
    real_df.to_csv("real.csv", index=False)


    fake_df = pd.DataFrame({'fake': taming_transformer['image_path']})
    fake_df.to_csv("fake.csv", index=False)

    df = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    df_enc_real = get_encoding_csv(model, real_df["real"], real_dir_path)
    df_enc_fake = get_encoding_csv(model, fake_df["fake"], fake_dir_path)

    anc_enc_arr_real = df_enc_real.iloc[:, 1:].to_numpy()
    anc_enc_arr_fake = df_enc_fake.iloc[:, 1:].to_numpy()

    last_idx = 0

    for i in tqdm(range(anc_enc_arr_real.shape[0]), desc="scorro gli anchor real"):
        max_dist = float('-inf')
        max_idx = 0
        min_dist = float('inf')
        min_idx = 0

        # cerco il positive con distanza massima
        for j in range(anc_enc_arr_real.shape[0]):
            if i == j:
                continue
            dist = euclidean_dist(anc_enc_arr_real[i : i+1, :], anc_enc_arr_real[j : j+1, :])
            if dist > max_dist:
                max_dist = dist
                max_idx = j
                #print("l'indice massimo è", max_idx)

        # cerco il negative con distanza minima
        for k in range(anc_enc_arr_fake.shape[0]):
            dist = euclidean_dist(anc_enc_arr_real[i : i+1, :], anc_enc_arr_fake[k : k+1, :])
            if dist < min_dist:
                min_dist = dist
                min_idx = k
                #print("l'indice minimo è", min_idx)

        # una volta effettuato lo scorrimento dei real e i fake e trovato l'immagine positive e negative, costruisco la tripletta
        df.loc[i] = [
            df_enc_real['real'].iloc[i],
            df_enc_real['real'].iloc[max_idx],
            df_enc_fake['fake'].iloc[min_idx]
        ]

        last_idx = i

    # la stessa cosa viene effettuata usando come anchor le immagini fake
    for i in tqdm(range(anc_enc_arr_fake.shape[0]), desc="scorro gli anchor fake"):
        max_dist = float('-inf')
        max_idx = 0
        min_dist = float('inf')
        min_idx = 0

        # cerco come immagine positive, quella con la distanza massima
        for j in range(anc_enc_arr_fake.shape[0]):
            if i == j:
                continue
            dist = euclidean_dist(anc_enc_arr_fake[i : i+1, :], anc_enc_arr_fake[j : j+1, :])
            if dist > max_dist:
                max_dist = dist
                max_idx = j

        # cerco come immagine negative, quella con distanza minima
        for k in range(anc_enc_arr_real.shape[0]):
            dist = euclidean_dist(anc_enc_arr_fake[i : i+1, :], anc_enc_arr_real[k : k+1, :])
            if dist < min_dist:
                min_dist = dist
                min_idx = k

        # costruisco quindi la tripletta, aggiungendola al dataframe
        df.loc[last_idx + i] = [
            df_enc_fake['fake'].iloc[i], 
            df_enc_fake['fake'].iloc[max_idx], 
            df_enc_real['real'].iloc[min_idx]
        ]
       
    df.to_csv(output_dir_path,index=False)


# funzione per creare
def semi_hard_mining(model, fake_dir_path, real_dir_path, real_metadata, fake_metadata, margin, output_dir_path):
    taming_transformer = pd.read_csv(fake_metadata, usecols=['image_path'], nrows=100000)
    coco = pd.read_csv(real_metadata, usecols=['image_path'], nrows=100000)

    real_df = pd.DataFrame({'real': coco['image_path']})
    real_df.to_csv("real.csv", index=False)

    print("ho creato real df")

    fake_df = pd.DataFrame({'fake': taming_transformer['image_path']})
    fake_df.to_csv("fake.csv", index=False)

    df = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    df_enc_real = get_encoding_csv(model, real_df["real"], real_dir_path)
    df_enc_fake = get_encoding_csv(model, fake_df["fake"], fake_dir_path)
   # df_enc_real.to_csv("df_enc_real.csv",index=False)
    #df_enc_fake.to_csv("df_enc_fake.csv", index=False)

    anc_enc_arr_real = df_enc_real.iloc[:, 1:].to_numpy()

    anc_enc_arr_fake = df_enc_fake.iloc[:, 1:].to_numpy()

    dataset_semi_hard_fake = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])
    dataset_semi_hard_real = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    fake_index = 0
    real_index = 0

    print(len(df_enc_fake))
    for i in tqdm(range(len(df_enc_fake))):
        A_embs = anc_enc_arr_fake[i, :]
        print(A_embs.shape)
        path_anchor = df_enc_fake.iloc[i]['fake']
        idp = random.randint(0, len(df_enc_fake) - 1)
        while torch.equal(torch.tensor(A_embs), torch.tensor(anc_enc_arr_fake[idp, :])):
            idp = random.randint(1, len(df_enc_fake) - 1)

        #P_embs = anc_enc_arr_fake[idp : idp +1, :]
        path_positive = df_enc_fake.iloc[idp]["fake"]


        idn = random.randint(0, len(df_enc_real) - 1)
        #N_embs = anc_enc_arr_real[idn : idn+1, :]
        path_negative = df_enc_real.iloc[idn]["real"]

        dist_positive = euclidean_dist(anc_enc_arr_fake[i, :], anc_enc_arr_fake[idp, :])
        dist_negative = euclidean_dist(anc_enc_arr_fake[i, :], anc_enc_arr_real[idn, :])

        #print("dist_positive  :", dist_positive,"  dist_negative  :", dist_negative, "  dist_positive + margin   :", dist_positive+margin)
        if (dist_positive < dist_negative) and (dist_negative < (dist_positive + margin)):
            #print("FAKE -> sono dentro l'if")
            dataset_semi_hard_fake.loc[fake_index] = [
                path_anchor,
                path_positive,
                path_negative,
            ]


        fake_index = fake_index + 1

           # print ("Ciclo Fake:  IDP:  ",idp,"  IDN: ",idn)

    for i in tqdm(range(len(df_enc_real))):
        A_embs = anc_enc_arr_real[i, :]
        path_anchor = df_enc_real.iloc[i]["real"]

        idp = random.randint(0, len(df_enc_real) - 1)
        while torch.equal(torch.tensor(A_embs), torch.tensor(anc_enc_arr_real[idp, :])):
            idp = random.randint(1, len(df_enc_real) - 1)

        #P_embs = anc_enc_arr_real[idp : idp+1, :]
        path_positive = df_enc_real.iloc[idp]["real"]

        idn = random.randint(0, len(df_enc_fake) - 1)
        #N_embs = anc_enc_arr_fake[idn : idn+1, :]
        path_negative = df_enc_fake.iloc[idn]["fake"]

        dist_positive = euclidean_dist(anc_enc_arr_real[i, :], anc_enc_arr_real[idp, :])
        dist_negative = euclidean_dist(anc_enc_arr_real[i, :], anc_enc_arr_fake[idn, :])

        #print("dist_positive  :", dist_positive,"  dist_negative  :", dist_negative, "  dist_positive + margin   :", dist_positive+margin)
        if (dist_positive < dist_negative) and (dist_negative < (dist_positive + margin)):
            #print("REAL -> sono dentro l'if")
            dataset_semi_hard_real.loc[real_index] = [
                path_anchor,
                path_positive,
                path_negative,
            ]


        real_index = real_index + 1

    dataset_semi_hard = pd.concat([dataset_semi_hard_fake, dataset_semi_hard_real], ignore_index=True)
    dataset_semi_hard = dataset_semi_hard.sample(frac=1)
    dataset_semi_hard.to_csv(output_dir_path ,index=False)
    return dataset_semi_hard
