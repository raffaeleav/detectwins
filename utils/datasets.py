import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import io, color, img_as_ubyte
from pathlib import Path


# funzione per costruire il dataset
def train(fake_dataset_path, real_dataset_path, dataset_size, output_dir):   
    # dataframe dei metadati del dataset di immagini
    df_fake = pd.read_csv(fake_dataset_path)
    # il dataset delle immagini real è coco
    df_real = pd.read_csv(real_dataset_path)

    # 'mescolo' i dataframe
    df_real = (df_real[df_real.target == 0]).sample(frac=1)
    df_fake = (df_fake[df_fake.target != 0]).sample(frac=1)

    # si crea la parte del dataset in cui le Anchor sono real
    df_real1 = df_real.sample(dataset_size * 2)
    df_fake1 = df_fake.sample(dataset_size)

    # si ottengono le immagini anchor, positive e fake
    df_anchor = df_real1.head(dataset_size)
    df_positive = df_real1.tail(dataset_size)
    df_negative = df_fake1

    # dataframe con anchor positive
    df_out1 = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])
    dataset_size = int(dataset_size / 2)

    # f(n) = O(n)
    for i in tqdm(range(dataset_size), desc="building (positive anchor) dataframe..."):
        df_out1.loc[i] = [
            df_anchor.iloc[i]["image_path"], 
            df_positive.iloc[i]["image_path"], 
            df_negative.iloc[i]["image_path"]
        ]

    # dataframe con anchor negative
    df_out2 = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    # si crea la parte del dataset in cui le Anchor sono fake (le Anchor possono essere anche immagini che nella 
    # prima parte del dataset erano presenti nella colonna Negative)
    df_fake2 = df_fake.sample(dataset_size * 2)
    df_real2 = df_real.sample(dataset_size)

    df_anchor = df_fake2.head(dataset_size)
    df_positive = df_fake2.tail(dataset_size)
    df_negative = df_real2

    # f(n) = O(n)
    for i in tqdm(range(dataset_size), desc="building (negative anchor) dataframe..."):
        df_out2.loc[i] = [
            df_anchor.iloc[i]["image_path"], 
            df_positive.iloc[i]["image_path"], 
            df_negative.iloc[i]["image_path"]
        ]

    df_out = pd.concat([df_out1, df_out2], axis=0)
    df_out.to_csv(output_dir, index=False)


# funzione per creare il dataset di test
def test(fake_dataset_path, real_dataset_path, df_out, output_dir):
    df_real = pd.read_csv(real_dataset_path)
    df_fake = pd.read_csv(fake_dataset_path)

    # 'mescolo' i dataframe
    df_real = (df_real[df_real.target == 0]).sample(frac=1)
    df_fake = (df_fake[df_fake.target != 0]).sample(frac=1)

    df_test = pd.DataFrame(columns=["real", "fake"])
    df_test_size = len(df_out) / 100 * 20
    df_test_size = int(df_test_size / 2)

    # si ottengono le immagini già presenti nel dataset di training
    used_images = set(df_out["Anchor"].to_list() + df_out["Positive"].to_list() + df_out["Negative"].to_list())
    real_images = []
    fake_images = []

    # f(n) = O(n), i set python sono hash table
    for i in tqdm(df_real["image_path"], desc="building (real column) test dataframe..."):
        if i not in used_images: 
            real_images.append(i)

            if len(real_images) >= df_test_size:
                break

    for i in tqdm(df_fake["image_path"], desc="building (fake column) test dataframe..."):
        if i not in used_images: 
            fake_images.append(i)

            if len(fake_images) >= df_test_size:
                break

    df_test["real"] = real_images
    df_test["fake"] = fake_images
    df_test.to_csv(output_dir, index=False)


# funzione per convertire le immagini rgb nel dataset di training nello spettro di fourier
def convert_train(fake_dataset_path, real_dataset_path, df_out): 
    project_path = Path(__file__).parent.parent
    fake_dataset = fake_dataset_path.split("\\")[-1]
    real_dataset = real_dataset_path.split("\\")[-1]
    temp_path = os.path.join(project_path, "temp")
    output_path = os.path.join(temp_path, fake_dataset + "+" + real_dataset)

    df_fourier = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])

    # controllo che non esistano le cartelle dove salvare le immagini, altrimenti le si creano
    if not os.path.isdir(temp_path): 
        os.mkdir(temp_path)
    
    if not os.path.isdir(output_path): 
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "train"))
        os.mkdir(os.path.join(output_path, "train", fake_dataset))
        os.mkdir(os.path.join(output_path, "train", real_dataset))

    output_path = os.path.join(output_path, "train")

    for index, row in tqdm(df_out.iterrows(), desc="converting rgb images to fourier spectrum..."):
        a, p, n = row["Anchor"], row["Positive"], row["Negative"]

        # si ottengono le immagini 
        if str(row.Anchor).startswith("coco"):
            a_img = io.imread(os.path.join(real_dataset_path, row.Anchor))
            p_img = io.imread(os.path.join(real_dataset_path, row.Positive))
            n_img = io.imread(os.path.join(fake_dataset_path, row.Negative))

        else:
            a_img = io.imread(os.path.join(fake_dataset_path, row.Anchor))
            p_img = io.imread(os.path.join(fake_dataset_path, row.Positive))
            n_img = io.imread(os.path.join(real_dataset_path, row.Negative))

        # si convertono le immagini rgb in scala di grigi
        a_img = color.rgb2gray(a_img)
        p_img = color.rgb2gray(p_img)
        n_img = color.rgb2gray(n_img)

        # si effettua la trasformata di fourier
        a_img = np.fft.fft2(a_img)
        p_img = np.fft.fft2(p_img)
        n_img = np.fft.fft2(n_img)

        # si applica il logaritmo (ln) al valore assoluto dell'immagine per normalizzarla 
        try: 
            a_img = np.log(np.abs(a_img))
            p_img = np.log(np.abs(p_img))
            n_img = np.log(np.abs(n_img))
        except Warning: 
            # nelle immagini a basso contrasto il valore assoluto potrebbe essere zero
            df_fourier.loc[index] = [
                None,
                None,
                None
            ]

            pass 
            continue
                
        # si considerano le frequenze necessarie
        a_min = np.min(a_img)
        a_max = np.max(a_img)
        p_min = np.min(p_img)
        p_max = np.max(p_img)
        n_min = np.min(n_img)
        n_max = np.max(n_img)

        # controllo per evitare errori con le immagini con poco contrasto
        if (a_max - a_min) <= 0:
            a_img = (a_img - a_min) / ((a_max - a_min) + np.finfo(float).eps)
        else: 
            a_img = (a_img - a_min) / (a_max - a_min)

        if (p_max - p_min) <= 0:
            p_img = (p_img - p_min) / ((p_max - p_min) + np.finfo(float).eps)
        else: 
            p_img = (p_img - p_min) / (p_max - p_min)

        if (n_max - n_min) <= 0:
            n_img = (n_img - n_min) / ((n_max - n_min) + np.finfo(float).eps)
        else: 
            n_img = (n_img - n_min) / (n_max - n_min)

        # si convertono le immagini in formato unsigned byte (al posto di fare / 255.0)
        a_img = img_as_ubyte(a_img)
        p_img = img_as_ubyte(p_img)
        n_img = img_as_ubyte(n_img)

        # si memorizzano le immagini fake in una cartella e quelle real in un'altra
        if str(row.Anchor).startswith("coco"):
            a_path = os.path.join(output_path, real_dataset, str(index) + "_anchor_real" + ".png")
            p_path = os.path.join(output_path, real_dataset, str(index) + "_positive_real" + ".png")
            n_path = os.path.join(output_path, fake_dataset, str(index) + "_negative_fake" + ".png")

            # path da memorizzare nel df per caricare le immagini successivamente
            df_a_path = str(index) + "_anchor_real" + ".png"
            df_p_path = str(index) + "_positive_real" + ".png"
            df_n_path = str(index) + "_negative_fake" + ".png"

        else: 
            a_path = os.path.join(output_path, fake_dataset, str(index) + "_anchor_fake" + ".png")
            p_path = os.path.join(output_path, fake_dataset, str(index) + "_positive_fake" + ".png")
            n_path = os.path.join(output_path, real_dataset, str(index) + "_negative_real" + ".png")

            df_a_path = str(index) + "_anchor_fake" + ".png"
            df_p_path = str(index) + "_positive_fake" + ".png"
            df_n_path = str(index) + "_negative_real" + ".png"

        io.imsave(a_path, a_img)
        io.imsave(p_path, p_img)
        io.imsave(n_path, n_img)

        df_fourier.loc[index] = [
            df_a_path,
            df_p_path,
            df_n_path
        ]

    df_fourier = df_fourier.dropna(how='any',axis=0) 
    
    output_dir = os.path.join(project_path, "datasets", "fourier_out.csv")
    df_fourier.to_csv(output_dir, index=False)


# attenzione ai path inseriti
def build(fake_dataset, real_dataset, dataset_size):
    path = Path(__file__).parent.parent.parent
    project_path = Path(__file__).parent.parent
    artifact_path = os.path.join(path, "artifact")
    fake_dataset_path = os.path.join(artifact_path, fake_dataset, "metadata.csv")
    real_dataset_path = os.path.join(artifact_path, real_dataset, "metadata.csv")

    output_dir = os.path.join(project_path, "datasets", "out.csv")
    train(fake_dataset_path, real_dataset_path, dataset_size, output_dir)

    # df_out = pd.read_csv(output_dir)

    # output_dir = os.path.join(project_path, "datasets", "testList.csv")
    # test(fake_dataset_path, real_dataset_path, df_out, output_dir)


def fourier(fake_dataset, real_dataset): 
    path = Path(__file__).parent.parent.parent
    project_path = Path(__file__).parent.parent
    artifact_path = os.path.join(path, "artifact")
    fake_dataset_path = os.path.join(artifact_path, fake_dataset)
    real_dataset_path = os.path.join(artifact_path, real_dataset)
    
    df_out = pd.read_csv(os.path.join(project_path, "datasets", "out.csv"))
    convert_train(fake_dataset_path, real_dataset_path, df_out)

    # df_fourier = pd.read_csv(os.path.join(project_path, "datasets", "fourier.csv"))


if __name__ == "__main__":
    fake_dataset = "taming_transformer"
    real_dataset = "coco"

    # build(fake_dataset, real_dataset, 50000)

    fourier(fake_dataset, real_dataset)
