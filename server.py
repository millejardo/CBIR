import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Membaca dan menghitung jarak kesamaan fitur gambar
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Menyimpan query gambar
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + \
            datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Menjalankan pencarian
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # Mengukur jarak L2 ke features
        ids = np.argsort(dists)[:20]  # Top 20 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        idf = np.argsort(dists)[:1] #Top 1

        # Batas Nilai
        per = np.linalg.norm(1.0)

        # Mengecek gambar yang diinput
        print(scores)
        print(dists)
        print(query)
        if dists[idf] < per:
            ket = "Benar"
            return render_template('index.html',
                                query_path=uploaded_img_path,
                                scores=scores,
                                ket=ket)
        else:
            ket = "Bukan"
            return render_template('index.html',
                                query_path=uploaded_img_path,
                                ket=ket)
   
    else:
        return render_template('index.html') # Menampilkan halaman pencarian

# Menjalankan aplikasi
if __name__=="__main__":
    app.run("127.0.0.1")
