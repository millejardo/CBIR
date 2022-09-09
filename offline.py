from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        # Ekstrak Fitur
        feature = fe.extract(img=Image.open(img_path))
        # Menyimpan array NumPy (.npy) pada folder feature
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")
        np.save(feature_path, feature)
        print(img_path) # e.g., ./static/img/xxx.jpg
        print(feature_path) # e.g., ./static/feature/xxx.npy
        print(feature) # [x. x. x. ... x. x. x.]