import os
import cv2
import pandas as pd

from tqdm import tqdm
from pathlib import Path


# funzione per costruire il dataset
def build_df(fake_dataset_path, real_dataset_path, output_dir):   
    # dataframe dei metadati del dataset di immagini
    df_fake = pd.read_csv(fake_dataset_path)
    # il dataset delle immagini real è coco
    df_real = pd.read_csv(real_dataset_path)

    # ottengo le immagini anchor, positive e fake
    df_real = (df_real[df_real.target == 0]).sample(2000)

    df_anchor = df_real.head(1000)
    df_positive = df_real.tail(1000)
    df_negative = (df_fake[df_fake.target != 0]).sample(1000)

    # dataframe finale
    df_out = pd.DataFrame(columns=['Anchor', 'Positive', 'Negative'])

    # f(n) = O(n)
    for i in tqdm(range(1000), desc="building dataframe..."):
        df_out.loc[i] = [
            df_anchor.iloc[i]['image_path'], 
            df_positive.iloc[i]['image_path'], 
            df_negative.iloc[i]['image_path']
        ]

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


# attenzione ai path inseriti
def main():
    path = Path(__file__).parent.parent.parent
    project_path = Path(__file__).parent.parent

    artifact_path = os.path.join(path, "artifact")
    fake_dataset_path = os.path.join(artifact_path, "cycle_gan", "metadata.csv")
    real_dataset_path = os.path.join(artifact_path, "coco", "metadata.csv")

    output_dir = os.path.join(project_path, "datasets", "out.csv")

    build_df(fake_dataset_path, real_dataset_path, output_dir)
    


if __name__ == "__main__":
    main()
    
    # check_image_path()
