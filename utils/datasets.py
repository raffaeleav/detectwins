import os
import pandas as pd

from tqdm import tqdm
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

    used_images = set(df_out["Anchor"].to_list() + df_out["Positive"].to_list() + df_out["Negative"].to_list())
    real_images = []
    fake_images = []

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


# attenzione ai path inseriti
def build(dataset_size, fake_dataset, real_dataset):
    path = Path(__file__).parent.parent.parent
    project_path = Path(__file__).parent.parent
    artifact_path = os.path.join(path, "artifact")
    fake_dataset_path = os.path.join(artifact_path, fake_dataset, "metadata.csv")
    real_dataset_path = os.path.join(artifact_path, real_dataset, "metadata.csv")

    output_dir = os.path.join(project_path, "datasets", "out.csv")
    # train(fake_dataset_path, real_dataset_path, dataset_size, output_dir)

    df_out = pd.read_csv(output_dir)

    output_dir = os.path.join(project_path, "datasets", "testList.csv")
    test(fake_dataset_path, real_dataset_path, df_out, output_dir)


if __name__ == "__main__":
    build(50000, "taming_transformer", "coco")
