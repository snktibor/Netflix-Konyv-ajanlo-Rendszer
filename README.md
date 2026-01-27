# 🎬 Netflix ↔ Könyv Ajánló Rendszer

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)
![SpaCy](https://img.shields.io/badge/spaCy-NLP-blue?logo=spacy)
![License](https://img.shields.io/badge/License-MIT-green)
![GloVe](https://img.shields.io/badge/Embeddings-GloVe-orange)

> [!NOTE]
> **Egyetemi Projekt**
>
> Ez a repository a **Budapesti Műszaki és Gazdaságtudományi Egyetem** *Természetes nyelvi és szemantikus technológiák* (BMEVIMIAC22) tárgyának 2025/26/1 féléves beadandó feladatát tartalmazza.

---

Ez a projekt egy **szemantikus hasonlóságon alapuló ajánlórendszer**, amely Netflix filmek és sorozatok címei alapján javasol olvasnivalót. A feldolgozás **GloVe szóbeágyazásokat (word embeddings)** és műfaji szűrést használ a releváns könyvek megtalálásához.

A rendszer célja, hogy összekösse a vizuális média fogyasztását az irodalommal, demonstrálva a szöveges hasonlóság-keresés és az NLP technikák gyakorlati alkalmazását.

---

## 📖 Projekt áttekintése

A program a felhasználó által megadott Netflix címeket (filmeket vagy sorozatokat) elemzi, és keres hozzájuk tartalmilag és hangulatilag hasonló könyveket. A működés fő lépései:

* **Adatfeldolgozás:** A leírások tisztítása és előkészítése (spaCy + NLTK stop-szavak).
* **Vektorizálás:** A szövegek matematikai reprezentációja GloVe vektorok segítségével.
* **Hasonlóság keresés:** Koszinusz-hasonlóság (Cosine Similarity) és műfaji átfedések vizsgálata.
* **Gyorsítótárazás:** A kiszámított vektorok mentése a `.cache` mappába a későbbi gyors futtatás érdekében.

### 🛠 Technológiák

* **Nyelv:** Python 3.11.9
* **NLP & ML:** `spaCy`, `nltk`, `scikit-learn`, `GloVe`
* **Adatkezelés:** `pandas`, `numpy`
* **String hasonlóság:** `thefuzz`

---

## 📊 Használt Adatkészletek

A projekt nyilvánosan elérhető, Kaggle-en közzétett adatkészleteket használ.

1.  **[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/bansodesandeep/netflix-movies-and-tv-shows/data)**
    * Tartalom: Filmek és sorozatok metaadatai (cím, leírás, műfaj).
2.  **[Best Books 10k](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data)**
    * Tartalom: Többműfajú könyvadatbázis (Goodreads-szerű metaadatok és leírások).

> *A fájlok `datasets/` mappában találhatóak. Kérlek, a felhasználás során tartsd tiszteletben a források licencfeltételeit.*

---

## 🚀 Telepítés és Futtatás

### Előfeltételek

1.  **Python 3.11** (vagy újabb) telepítve legyen.
2.  **Internetkapcsolat**: Az első futtatáshoz szükséges a GloVe modell és a spaCy nyelvi csomag letöltéséhez.

### Telepítés lépésről lépésre (Windows PowerShell) ⚡

A projekt futtatásához ajánlott virtuális környezetet (venv) használni.

```powershell
# 1. Virtuális környezet létrehozása
python -m venv .venv

# 2. Környezet aktiválása
.\.venv\Scripts\Activate.ps1

# 3. Pip frissítése és függőségek telepítése
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn tqdm nltk spacy thefuzz

# 4. SpaCy nyelvi modell letöltése
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

---

## 🖥 Használat

### Indítás

A programot a `Main.py` futtatásával indíthatod el.

```powershell
# Ha a venv aktív:
python Main.py

# Vagy közvetlenül a venv elérési útjával:
.\.venv\Scripts\python.exe Main.py
```

> ⚠️ **Fontos:** Az **első indítás** (a GloVe modell letöltése és a dokumentumok vektorizálása) rendszerint **körülbelül 5–10 percet vesz igénybe**, a hálózati sebességtől és a számítógép teljesítményétől függően. A későbbiekben a `.cache` mappa segítségével az indulás pillanatok alatt megtörténik.

### Példa bemenet

A program futása közben add meg a kedvenc címeidet vesszővel elválasztva:

```text
Stranger Things, Breaking Bad
```

### Mappastruktúra
* `datasets/`: A forrás CSV fájlok helye.
* `.cache/`: A kiszámított vektorok és parquet fájlok (automatikus gyorsítótár).
* `.glove/`: A letöltött GloVe modellek helye.

---

## ⚖️ Licenc

Ez a projekt az **MIT License** alatt áll. Szabadon felhasználható, módosítható és terjeszthető.

Copyright (c) 2025 **Sinka Tibor**

További részletek a [LICENSE](LICENSE) fájlban.
