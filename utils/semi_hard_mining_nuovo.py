import ast
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

DEVICE = "cuda"

path = Path(__file__).parent.parent.parent
project_path = Path(__file__).parent.parent

artifact_path = os.path.join(path, "artifact")
fake_dir_path = os.path.join(artifact_path, "taming_transformer")
real_dir_path = os.path.join(artifact_path, "coco")

fake_metadata = os.path.join(artifact_path, "taming_transformer", "metadata.csv")
real_metadata = os.path.join(artifact_path, "coco", "metadata.csv")

output_dir_path = os.path.join(project_path, "datasets", "semi_hard_mining_database.csv") 

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

def euclidean_dist(img_enc, anc_enc_arr):
    # dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T))
    dist = np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T)
    # dist = np.sqrt(dist)
    return dist


def string_to_tensor(tensor_str):
    # Rimuovi 'tensor(' all'inizio e ')' alla fine
    tensor_str = tensor_str.replace('tensor(', '').rstrip(')')
    # Converti la stringa in una lista
    tensor_list = ast.literal_eval(tensor_str)
    # Converti la lista in un tensor di PyTorch
    return torch.tensor(tensor_list)


# funzione per creare
def semi_hard_mining(real_metadata, fake_metadata, margin, output_dir_path):
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

    anc_enc_arr_real = df_enc_real.iloc[:, 1:].to_numpy()
    anc_enc_arr_fake = df_enc_fake.iloc[:, 1:].to_numpy()

    dataset_semi_hard_fake = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])
    dataset_semi_hard_real = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    fake_index = 0
    real_index = 0

    print(len(df_enc_fake))
    for i in tqdm(range(len(df_enc_fake))):
        A_embs = anc_enc_arr_fake[i : i+1, :]
        path_anchor = df_enc_fake.iloc[i]['fake']
        idp = random.randint(0, len(df_enc_fake) - 1)
        while torch.equal(torch.tensor(A_embs), torch.tensor(anc_enc_arr_fake[idp:idp+1, :])):
            idp = random.randint(1, len(df_enc_fake) - 1)

        P_embs = anc_enc_arr_fake[idp : idp +1, :]
        path_positive = df_enc_fake.iloc[idp]["fake"]


        idn = random.randint(0, len(df_enc_real) - 1)
        N_embs = anc_enc_arr_real[idn : idn+1, :]
        path_negative = df_enc_real.iloc[idn]["real"]

        dist_positive = euclidean_dist(anc_enc_arr_fake[i : i+1, :], anc_enc_arr_fake[idp : idp+1, :])
        dist_negative = euclidean_dist(anc_enc_arr_fake[i : i+1, :], anc_enc_arr_real[idn : idn+1, :])

        print("dist_positive  :", dist_positive,"  dist_negative  :", dist_negative, "  dist_positive + margin   :", dist_positive+margin)
        if (dist_positive < dist_negative) and (dist_negative < (dist_positive + margin)):
            print("sono dentro l'if")
            dataset_semi_hard_fake.loc[fake_index] = [
                path_anchor,
                path_positive,
                path_negative,
            ]
            fake_index = fake_index + 1

           # print ("Ciclo Fake:  IDP:  ",idp,"  IDN: ",idn)

    for i in tqdm(range(len(df_enc_real))):
        A_embs = anc_enc_arr_real[i : i+1, :]
        path_anchor = df_enc_real.iloc[i]["real"]

        idp = random.randint(0, len(df_enc_real) - 1)
        while torch.equal(torch.tensor(A_embs), torch.tensor(anc_enc_arr_real[idp:idp+1, :])):
            idp = random.randint(1, len(df_enc_real) - 1)

        P_embs = anc_enc_arr_real[idp : idp+1, :]
        path_positive = df_enc_real.iloc[idp]["real"]

        idn = random.randint(0, len(df_enc_fake) - 1)
        N_embs = anc_enc_arr_fake[idn : idn+1, :]
        path_negative = df_enc_fake.iloc[idn]["fake"]

        dist_positive = euclidean_dist(anc_enc_arr_real[i : i+1, :], anc_enc_arr_real[idp : idp+1, :])
        dist_negative = euclidean_dist(anc_enc_arr_real[i : i+1, :], anc_enc_arr_fake[idn : idn+1, :])

        print("dist_positive  :", dist_positive,"  dist_negative  :", dist_negative, "  dist_positive + margin   :", dist_positive+margin)
        if (dist_positive < dist_negative) and (dist_negative < (dist_positive + margin)):
            #print("sono dentro l'if")
            dataset_semi_hard_real.loc[real_index] = [
                path_anchor,
                path_positive,
                path_negative,
            ]
            real_index = real_index + 1
            # print("Ciclo Real:  IDP:  ", idp, "  IDN: ", idn)


    dataset_semi_hard = pd.concat([dataset_semi_hard_fake, dataset_semi_hard_real], ignore_index=True)
    dataset_semi_hard = dataset_semi_hard.sample(frac=1)
    dataset_semi_hard.to_csv(output_dir_path ,index=False)
    return dataset_semi_hard


# funzione per creare il dataset di test
def build_test_df(fake_dataset_path, real_dataset_path, output_dir):
    project_path = Path(__file__).parent.parent
    df_real = pd.read_csv(real_dataset_path)
    df_fake = pd.read_csv(fake_dataset_path)

    # 'mescolo' i dataframe
    df_real = (df_real[df_real.target == 0]).sample(frac=1)
    df_fake = (df_fake[df_fake.target != 0]).sample(frac=1)

    df_out = pd.read_csv(os.path.join(project_path, "datasets", "out.csv"))

    df_test = pd.DataFrame(columns=["real", "fake"])
    df_test_size = len(df_out) / 100 * 20
    df_test_size = int(df_test_size)

    # f(n) = O(n x m), m numero di celle in Anchor e Positive
    for i in tqdm(range(df_test_size), desc="building (real column) test dataframe..."):
        # scelgo un elemento casuale dal dataset real
        idx = random.randint(0, len(df_real))
        item = df_real.iloc[idx]["image_path"]

        # controllo che non sia già presente nel dataset da usare per l'allenamento, altrimenti ne scelgo un altro
        while item in df_out["Anchor"].to_list() or item in df_out["Positive"].to_list() or item in df_out[
            "Negative"].to_list():
            idx = random.randint(1, len(df_real) - 1)
            item = df_real.iloc[idx]["image_path"]

        df_test.loc[i, "real"] = item

    # f(n) = O(n x m), m numero di celle in Negative
    for i in tqdm(range(df_test_size), desc="building (fake column) test dataframe..."):
        idx = random.randint(0, len(df_fake))
        item = df_fake.iloc[idx]["image_path"]

        while item in df_out["Anchor"].to_list() or item in df_out["Positive"].to_list() or item in df_out[
            "Negative"].to_list():
            idx = random.randint(1, len(df_fake) - 1)
            item = df_fake.iloc[idx]["image_path"]

        df_test.loc[i, "fake"] = item

    df_test.to_csv(output_dir, index=False)


def main():
    margin = 0.2
    semi_hard_mining(fake_metadata, real_metadata, margin, output_dir_path)
    output_dir = os.path.join(project_path, "datasets", "testList.csv")
    build_test_df(fake_metadata,real_metadata,output_dir)

if __name__ == "__main__": 
    main()

