
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "perbaikankontras"
nama_foto = "ayam.jpg"

img = cv2.imread(nama_foto)

if img is None:
    raise FileNotFoundError(
        "Foto ayam.jpg tidak ditemukan. "
        "Simpan foto di folder perbaikankontras."
    )

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

citra_asli = img.copy()

kontras_rendah = cv2.convertScaleAbs(
    img, alpha=0.5, beta=0
)

kontras_normal = cv2.normalize(
    img, None, 0, 255, cv2.NORM_MINMAX
)

kontras_tinggi = cv2.convertScaleAbs(
    img, alpha=1.5, beta=-60
)


gamma = 1.8

tabel_gamma = np.array([
    ((i / 255.0) ** gamma) * 255
    for i in range(256)
]).astype(np.uint8)

gamma_correction = cv2.LUT(img, tabel_gamma)

lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l_baru = clahe.apply(l)
lab_baru = cv2.merge((l_baru, a, b))

hasil_clahe = cv2.cvtColor(
    lab_baru, cv2.COLOR_LAB2RGB
)


hasil = [
    ("Asli", citra_asli),
    ("Kontras Rendah", kontras_rendah),
    ("Normalisasi", kontras_normal),
    ("Kontras Tinggi", kontras_tinggi),
    ("Gamma Correction", gamma_correction),
    ("CLAHE", hasil_clahe)
]

plt.figure(figsize=(16, 9))

for i, (nama, gambar) in enumerate(hasil):

    # Menampilkan citra
    plt.subplot(2, 6, i + 1)
    plt.imshow(gambar)
    plt.title(nama)
    plt.axis("off")

    # Mengubah citra menjadi grayscale
    gray = cv2.cvtColor(
        gambar, cv2.COLOR_RGB2GRAY
    )

    # Menampilkan histogram
    plt.subplot(2, 6, i + 7)
    plt.hist(
        gray.ravel(),
        bins=256,
        range=(0, 256)
    )

    plt.title("Histogram " + nama)
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")

plt.tight_layout()
plt.show()


folder_hasil = "hasil_kontras"
os.makedirs(folder_hasil, exist_ok=True)

for nama, gambar in hasil:
    nama_file = nama.lower().replace(" ", "_") + ".jpg"
    lokasi = os.path.join(folder_hasil, nama_file)

    cv2.imwrite(
        lokasi,
        cv2.cvtColor(gambar, cv2.COLOR_RGB2BGR)
    )

print("================================")
print("PROSES PERBAIKAN KONTRAS SELESAI")
print("Hasil tersimpan di folder:", folder_hasil)
print("================================")
