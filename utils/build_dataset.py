import os
import cv2
import random
import pandas as pd

from tqdm import tqdm
from pathlib import Path


# funzione per costruire il dataset
def build_df(fake_dataset_path, real_dataset_path, output_dir, dataset_size):   
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

    # dataframe finale
    df_out1 = pd.DataFrame(columns=["Anchor", "Positive", "Negative"])
    dataset_size = int(dataset_size / 2)

    # f(n) = O(n)
    for i in tqdm(range(dataset_size), desc="building (positive anchor) dataframe..."):
        df_out1.loc[i] = [
            df_anchor.iloc[i]["image_path"], 
            df_positive.iloc[i]["image_path"], 
            df_negative.iloc[i]["image_path"]
        ]

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


# funzione per controllare che le immagini nel dataset si possano reperire
def check_image_path():
    # dir dove si trova il dataset artifact
    path = Path(__file__).parent.parent.parent
    dataset_path = os.path.join(path, "artifact", "cycle_gan")
    img_path = os.path.join("st", "horse2zebra", "img003759.jpg")
    path = os.path.join(dataset_path, img_path)

    image = cv2.imread(path)
    cv2.imshow("dataset-image", image)
    cv2.waitKey(0)


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
        while item in df_out["Anchor"].to_list() or item in df_out["Positive"].to_list() or item in df_out["Negative"].to_list():
            idx = random.randint(1, len(df_real) - 1)
            item = df_real.iloc[idx]["image_path"]

        df_test.loc[i, "real"] = item

    # f(n) = O(n x m), m numero di celle in Negative
    for i in tqdm(range(df_test_size), desc="building (fake column) test dataframe..."):
        idx = random.randint(0, len(df_fake))
        item = df_fake.iloc[idx]["image_path"]

        while item in df_out["Anchor"].to_list() or item in df_out["Positive"].to_list() or item in df_out["Negative"].to_list():
            idx = random.randint(1, len(df_fake) - 1)
            item = df_fake.iloc[idx]["image_path"]

        df_test.loc[i, "fake"] = item

    df_test.to_csv(output_dir, index=False)


# attenzione ai path inseriti
def main():
    path = Path(__file__).parent.parent.parent
    project_path = Path(__file__).parent.parent

    artifact_path = os.path.join(path, "artifact")
    fake_dataset_path = os.path.join(artifact_path, "big_gan", "metadata.csv")
    real_dataset_path = os.path.join(artifact_path, "coco", "metadata.csv")

    output_dir = os.path.join(project_path, "datasets", "out.csv")
    # size di 8000 perchè big_gan ha solo 10000 fake
    dataset_size = 8000
    build_df(fake_dataset_path, real_dataset_path, output_dir, dataset_size)

    output_dir = os.path.join(project_path, "datasets", "testList.csv")
    build_test_df(fake_dataset_path, real_dataset_path, output_dir)


if __name__ == "__main__":
    main()
    
    # check_image_path()
