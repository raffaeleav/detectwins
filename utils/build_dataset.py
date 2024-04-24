import cv2
import pandas as pd

from tqdm import tqdm
from pathlib import Path


# funzione per costruire il dataset
def build_df(dataset_path, output_dir):   
    # dataframe dei metadati del dataset di immagini
    df = pd.read_csv(dataset_path)

    # ottengo le immagini anchor, positive e fake
    df_real = (df[df.target == 0]).sample(2000)

    df_anchor = df_real.head(1000)
    df_positive = df_real.tail(1000)
    df_negative = (df[df.target != 0]).sample(1000)

    # dataframe finale
    df_out = pd.DataFrame(columns=['Anchor', 'Positive', 'Negative'])

    # f(n) = O(n)
    # non abbastanza info sulla velocita' di .iloc ...
    for i in tqdm(range(1000), desc="building dataframe..."):
        df_out.loc[i] = [
            df_anchor.iloc[i]['image_path'], 
            df_positive.iloc[i]['image_path'], 
            df_negative.iloc[i]['image_path']
        ]

    df_out.to_csv(output_dir, index=False)

# funzione per controllare che le immagini nel dataset si possano reperire
def check_image_path():
    dir_path = r'D:\sviluppo\jupyter-notebooks\fvab\dataset\cycle_gan'
    img_path = r'st/horse2zebra/img003759.jpg'
    path = dir_path + r'\\' + img_path

    image = cv2.imread(path)
    cv2.imshow("dataset-image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    dataset_path = r"D:\sviluppo\jupyter-notebooks\fvab\dataset\cycle_gan\metadata.csv"
    output_dir = r"D:\sviluppo\python-workspace\fvab\results\out.csv"

    build_df(dataset_path, output_dir)

    # check_image_path()
