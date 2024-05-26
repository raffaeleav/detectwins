import itertools
import os
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

size = 32
def combine(anchor_images, positive_images, negative_images):
    combinations = list(itertools.product(anchor_images, positive_images, negative_images))
    df = pd.DataFrame(combinations, columns=["Anchor", "Positive", "Negative"])

    return df


# funzione per costruire il dataset di tutti i possibili triplet
def build_combinations_df(fake_dataset_path, real_dataset_path, output_dir, dataset_size):   
    # dataframe dei metadati del dataset di immagini
    df_fake = pd.read_csv(fake_dataset_path)
    # il dataset delle immagini real è coco
    df_real = pd.read_csv(real_dataset_path)

    # 'mescolo' i dataframe
    df_real = (df_real[df_real.target == 0]).sample(frac=1)
    df_fake = (df_fake[df_fake.target != 0]).sample(frac=1)

    dataset_size = int(dataset_size / 2)

    # si crea la parte del dataset in cui le Anchor sono real
    df_real1 = df_real.sample(dataset_size * 2)
    df_fake1 = df_fake.sample(dataset_size)

    # si ottengono le immagini anchor, positive e fake
    df_anchor = df_real1.head(dataset_size)
    df_positive = df_real1.tail(dataset_size)
    df_negative = df_fake1

    df_out1 = combine(df_anchor["image_path"].tolist(), df_positive["image_path"].tolist(), df_negative["image_path"].tolist())

    # si crea la parte del dataset in cui le Anchor sono fake (le Anchor possono essere anche immagini che nella 
    # prima parte del dataset erano presenti nella colonna Negative)
    df_fake2 = df_fake.sample(dataset_size * 2)
    df_real2 = df_real.sample(dataset_size)

    df_anchor = df_fake2.head(dataset_size)
    df_positive = df_fake2.tail(dataset_size)
    df_negative = df_real2

    df_out2 = combine(df_anchor["image_path"].tolist(), df_positive["image_path"].tolist(), df_negative["image_path"].tolist())

    df_out = pd.concat([df_out1, df_out2], axis=0)
    df_out.to_csv(output_dir, index=False)


# funzione per creare 
def semi_hard_mining(embeddings_a, embeddings_p, embeddings_n, margin):

    dataset = pd.DataFrame(columns=["Anchor", "Positive", "Negative", "A_emb", "P_emb", "N_emb"])

    for i in range (len(embeddings_a)):
        for j in range(len(embeddings_p)):
            dist_positive = torch.norm(embeddings_a.iloc[i]["A_emb"] - embeddings_p.iloc[j]["P_emb"])
            for k in range(len(embeddings_n)):
                dist_negative = torch.norm(embeddings_a.iloc[i]["A_emb"] - embeddings_n.iloc[k]["N_emb"])
                if dist_positive < dist_negative < dist_positive + margin:
                    dataset.loc[i] = [
                    embeddings_a.iloc[i]["Anchor"],
                    embeddings_p.iloc[j]["Positive"],
                    embeddings_n.iloc[k]["Negative"],
                    embeddings_a.iloc[i]["A_emb"],
                    embeddings_p.iloc[j]["P_emb"],
                    embeddings_n.iloc[k]["N_emb"]
                    ]
    dataset.to_csv("prova.csv",index=False)                
    return dataset

    