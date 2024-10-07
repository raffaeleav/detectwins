import os
import timm
import torch
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from skimage import io


# classe per caricare il modello di rete neurale 
class ApnModel(nn.Module):

    # size del vettore di embedding
    def __init__(self, emb_size=512):
        super(ApnModel, self).__init__()

        # caricamento del modello, in questo caso efficientnet b0 (architettura più leggera della famiglia)
        self.efficientnet = timm.create_model("tf_efficientnetv2_b0", pretrained=False)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings
  

# funzione per generare i vettori di encoding
def get_encoding_csv(model, anc_img_names, fake_data_dir, real_data_dir, device):
    anc_img_names_arr = np.array(anc_img_names)
    encodings = []

    model.eval()

    with torch.no_grad():
        for i in tqdm(anc_img_names_arr, desc="creating encodings..."):
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
    

# funzione che genera embeddings di una singola immagine
def get_image_embeddings(img, model, device):
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img) / 255
    
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        img_enc = model(img.unsqueeze(0))
        img_enc = img_enc.detach().cpu().numpy()
        img_enc = np.array(img_enc)

    return img_enc


# distanza euclidea per array np
def array_distance(img_enc, anc_enc_arr):
    dist = np.dot(img_enc-anc_enc_arr, (img_enc- anc_enc_arr).T)
    # dist = np.sqrt(dist)

    return dist


# funzione che cerca nel database l'immagine più simile a quella data in input
def search_in_database(img_enc, database):
    anc_enc_arr = database.iloc[:, 1:].to_numpy()

    distance = []
    for i in range(anc_enc_arr.shape[0]):
        dist = array_distance(img_enc, anc_enc_arr[i : i+1, :])
        distance = np.append(distance, dist)

    closest_idx = np.argsort(distance)

    return database["Anchor"][closest_idx[0]]


# funzione per ottenere i path di tutti i file in una cartella (per creare dataset di test)
def get_file_paths(directory): 
    file_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            # path completo del file
            file_path = os.path.join(root, file)  
            file_paths.append(file_path)
            
    return file_paths
