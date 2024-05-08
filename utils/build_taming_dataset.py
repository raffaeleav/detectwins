import os
import cv2
import random
import pandas as pd
import random

from tqdm import tqdm
from pathlib import Path


# funzione per costruire il dataset
def build_df_tt(fake_dataset_path, real_dataset_path, output_dir):   
    # immagini fake - latent transformer
    df_fake = pd.read_csv(fake_dataset_path)
    # il dataset delle immagini real è coco
    df_real = pd.read_csv(real_dataset_path)

    # ottengo le immagini anchor, positive e fake
    df_real = (df_real[df_real.target == 0]).sample(8000)
    df_anchor = df_real.head(4000)
    df_positive = df_real.tail(4000)
    df_negative = (df_fake[df_fake.target != 0]).sample(4000)


    df_real_tt = (df_fake[df_fake.target != 0]).sample(8000)
    df_anchor_tt = df_real_tt.head(4000)
    df_positive_tt = df_real_tt.tail(4000)
    df_negative_coco = (df_real[df_real.target == 0]).sample(4000)

    df_out_coco = pd.DataFrame(columns=['Anchor', 'Positive', 'Negative'])
    df_out_tt = pd.DataFrame(columns=['Anchor', 'Positive', 'Negative'])

    # dataframe finale
    df_out = pd.DataFrame(columns=['Anchor', 'Positive', 'Negative'])

    # f(n) = O(n)
    for i in tqdm(range(4000), desc="anchor positive"):
        df_out_coco.loc[i] = [
            df_anchor.iloc[i]['image_path'], 
            df_positive.iloc[i]['image_path'], 
            df_negative.iloc[i]['image_path']
        ]

    for i in tqdm(range(4000), desc="anchor negative"):
        df_out_tt.loc[i] = [
            df_anchor_tt.iloc[i]['image_path'], 
            df_positive_tt.iloc[i]['image_path'], 
            df_negative_coco.iloc[i]['image_path']
        ]    

    df_out = pd.concat([df_out_coco, df_out_tt])
    df_out.to_csv(output_dir, index=False)


#funzione per creare il dataset di test
def build_test_df_tt(fake_dataset_path, real_dataset_path, output_dir):
    project_path = Path(__file__).parent.parent
    df_real = pd.read_csv(real_dataset_path)
    df_fake = pd.read_csv(fake_dataset_path)

    #'mescolo' i dataframe
    df_real = (df_real[df_real.target == 0]).sample(frac=1)
    df_fake = (df_fake[df_fake.target != 0]).sample(frac=1)
    
    df_out = pd.read_csv(os.path.join(project_path, "datasets", "out_tt.csv"))

    df_test = pd.DataFrame(columns=["real", "fake"])
    df_test_size = len(df_out) / 100 * 20
    df_test_size = int(df_test_size)

    # f(n) = O(n x m), m numero di celle in Anchor e Positive
    for i in tqdm(range(df_test_size), desc="building (real column) test dataframe..."):
        # scelgo un elemento casuale dal dataset real
        idx = random.randint(0, len(df_real))
        item = df_real.iloc[idx]["image_path"]

        # controllo che non sia già presente nel dataset da usare per l'allenamento, altrimenti ne scelgo un altro
        while item in df_out["Anchor"].to_list() or item in df_out["Positive"].to_list():
            idx = random.randint(0, len(df_real))
            item = df_real.iloc[idx]["image_path"]

        df_test.loc[i, "real"] = item

    # f(n) = O(n x m), m numero di celle in Negative
    for i in tqdm(range(df_test_size), desc="building (fake column) test dataframe..."):
        idx = random.randint(0, len(df_fake))
        item = df_fake.iloc[idx]["image_path"]

        while item in df_out["Negative"].to_list():
            idx = random.randint(0, len(df_fake))
            item = df_fake.iloc[idx]["image_path"]

        df_test.loc[i, "fake"] = item

    df_test.to_csv(output_dir, index=False)


# attenzione ai path inseriti
def main():
    path = Path(__file__).parent.parent.parent
    project_path = Path(__file__).parent.parent

    artifact_path = os.path.join(path, "artifact")
    fake_dataset_path = os.path.join(artifact_path, "taming_transformer", "metadata.csv")
    real_dataset_path = os.path.join(artifact_path, "coco", "metadata.csv")

    #output_dir = os.path.join(project_path, "datasets", "out_tt.csv")

    #build_df_tt(fake_dataset_path, real_dataset_path, output_dir)

    output_dir = os.path.join(project_path, "datasets", "testList_tt.csv")

    build_test_df_tt(fake_dataset_path, real_dataset_path, output_dir)


if __name__ == "__main__":
    main()
   