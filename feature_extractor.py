from keras.utils import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        
        # Mengubah ukuran gambar. Model VGG harus menggunakan gambar dengan skala 224x224 sebagai input
        img = img.resize((224, 224))
        # mengubah gambar menjadi berwarna
        img = img.convert('RGB')
        # Mengubah menjadi gambar menjadi data 3D array sebagai Height(H) x Width(W) x Channel(C). dtype=float32
        x = img_to_array(img)
        # (H, W, C)->(1, H, W, C), menambahkan data dimensi array keempat pada elemen pertama adalah angka index dari img
        x = np.expand_dims(x, axis=0)
        # mengurangi nilai rata-rata untuk setiap piksel
        x = preprocess_input(x)
        # (1, 4096) -> (4096, ), membuat model fitur
        feature = self.model.predict(x)[0]
        # Menormalisasikan
        return feature / np.linalg.norm(feature)
