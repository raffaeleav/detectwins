import ast
<<<<<<< HEAD
import random
import cv2
import os
import sys
import timm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch import nn
from tqdm import tqdm
from skimage import io
from pathlib import Path

# per far funzionare il modello su immagini rgb o in scala di grigi (per usare fourier)
mode="rgb"

BATCH_SIZE = 32

DEVICE = "cpu"

path = Path(__file__).parent.parent.parent
project_path = Path(__file__).parent.parent

artifact_path = os.path.join(path, "artifact")
fake_dir_path = os.path.join(artifact_path, "taming_transformer")
real_dir_path = os.path.join(artifact_path, "coco")

fake_metadata = os.path.join(artifact_path, "taming_transformer", "metadata.csv")
real_metadata = os.path.join(artifact_path, "coco", "metadata.csv")

output_dir_path = os.path.join(project_path, "datasets", "super_hard_mining_database.csv") 


# classe del modello che genera gli embedding per applicare il semi-hard mining
class EmbModel(nn.Module):

    # size del vettore di embedding
    def __init__(self, emb_size = 512):
        super(EmbModel, self).__init__()

        # gli embedding vengono creati con un modello preallenato (risultato più efficace in test precedenti)
        self.efficientnet = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings
    
model = EmbModel()

# per processare le immagini in scala di grigi per fare fourier serve una CNN 2D
if mode == "grey_scale":
    model.efficientnet.conv_stem = nn.Conv2d(1, 32, 3, 2, 1, bias=False)

model.to(DEVICE)


# funzione per generare i vettori di encoding
def get_encoding_csv(model, anc_img_names, dirFolder):
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

def string_to_tensor(tensor_str):
    # Rimuovi 'tensor(' all'inizio e ')' alla fine
    tensor_str = tensor_str.replace('tensor(', '').rstrip(')')
    # Converti la stringa in una lista
    tensor_list = ast.literal_eval(tensor_str)
    # Converti la lista in un tensor di PyTorch
    return nn.tensor(tensor_list)

def euclidean_dist(img_enc, anc_enc_arr):
    # dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T))
    dist = np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T)
    # dist = np.sqrt(dist)
    return dist

# funzione per effettuare hard mining 
def hard_mining(fake_metadata, real_metadata, output_dir_path): 
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

# funzione per creare il dataset di test
def build_test_df(fake_dataset_path, real_dataset_path, output_dir):
    df_real = pd.read_csv(real_dataset_path)
    df_fake = pd.read_csv(fake_dataset_path)

    #'mescolo' i dataframe
    df_real = (df_real[df_real.target == 0]).sample(frac=1)
    df_fake = (df_fake[df_fake.target != 0]).sample(frac=1)
    
    df_out = pd.read_csv(output_dir_path)

    df_test = pd.DataFrame(columns=["real", "fake"])
    df_test_size = len(df_out) / 100 * 20
    df_test_size = int(df_test_size)

    # f(n) = O(n x m), m numero di celle in Anchor e Positive
    for i in tqdm(range(df_test_size), desc="building (real column) test dataframe..."):
        # scelgo un elemento casuale dal dataset real
        idx = random.randint(0, len(df_real))
        item = df_real.iloc[idx]["image_path"]

        # controllo che non sia già presente nel dataset da usare per l'allenamento, altrimenti ne scelgo un altro
        while item in df_out["real"].to_list():
            idx = random.randint(0, len(df_real))
            item = df_real.iloc[idx]["image_path"]

        df_test.loc[i, "real"] = item

    # f(n) = O(n x m), m numero di celle in Negative
    for i in tqdm(range(df_test_size), desc="building (fake column) test dataframe..."):
        idx = random.randint(0, len(df_fake))
        item = df_fake.iloc[idx]["image_path"]

        while item in df_out["fake"].to_list():
            idx = random.randint(0, len(df_fake))
            item = df_fake.iloc[idx]["image_path"]

        df_test.loc[i, "fake"] = item

    df_test.to_csv(output_dir, index=False)    

def main():
    
    hard_mining(fake_metadata, real_metadata, output_dir_path)

    output_dir = os.path.join(project_path, "datasets", "testList.csv")

    build_test_df(fake_metadata, real_metadata, output_dir)

if __name__ == "__main__": 
    main()
=======
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

    return ap_distance < an_distance < (ap_distance + margin)


# applico il filtro al df
def offline_semi_hard_mining(df, margin):
    df = df[df.apply(filter, args=(margin,), axis=1)]

    return df 
>>>>>>> feature/semi-hard-not-pretrained
