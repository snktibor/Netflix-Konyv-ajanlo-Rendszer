# Netflix ↔ Könyv Ajánló — Használati útmutató

Rövid leírás
- Ez a projekt egy egyszerű ajánlórendszer, ami Netflix filmcímek alapján próbál könyveket javasolni. A feldolgozás GloVe szóbeágyazásokat és egyszerű műfaj-szűrést használ.

Előkészületek / követelmények
- Python 3.11.9 verzió alatt készítve, és tesztelve.
- Ajánlott virtuális környezet (venv):
  - Windows például:
```powershell
python -m venv .venv
```
  - Aktiválás (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
- Szükséges Python csomagok:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn tqdm nltk spacy thefuzz
```
- spaCy nyelvi modell:
```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```
- Internetkapcsolat szükséges az első futtatáshoz (GloVe letöltés, spaCy modell).

Fájlok és adatforrások
- Datasets mappa (elvárt helyek):
  - datasets/Netflix_movies_and_tv_shows.csv
  - datasets/goodreads_data.csv
- Gyorsítótár és letöltött modellek:
  - .cache/ — a kiszámított vektorok és parquet fájlok tárolása (gyors betöltéshez).
  - .glove/ — a letöltött GloVe ZIP és kicsomagolt txt fájlok. (Ez mappa javasolt, hogy a .gitignore-ban szerepeljen.)

<!-- Beillesztendő: Használt adatkészletek (szakszerű, rövid leírás a forrásokról) -->
## Használt adatkészletek

A projekt során nyilvánosan elérhető, Kaggle-en közzétett adatkészleteket használtam fel a modell építéséhez és értékeléséhez. A források rövid, szakszerű megnevezése és hivatkozása alább található:

- Netflix filmek és TV-műsorok (Netflix Movies and TV Shows) — forrás és adatelemes leírás:
  https://www.kaggle.com/datasets/bansodesandeep/netflix-movies-and-tv-shows/data

- Best Books 10k — többműfajú könyvadatbázis (Goodreads-szerű metaadatok és leírások):
  https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data

Ezek az adatkészletek szolgáltak a bemeneti adatok (filmcímek, leírások, műfajok) és a könyvek metaadatainak forrásául. A README-ben és a kódban található fájlnevek a fenti forrásokból származó CSV-fájlokra mutatnak. Kérlek, a felhasználás során tartsd tiszteletben a források licencfeltételeit.

Futtatás
1. Aktiváld a virtuális környezetet vagy használd közvetlenül a projekt venv Python-ját.
2. Ajánlott telepítési parancsok (Windows PowerShell példa):
```powershell
# Frissítsd a pip-et, telepítsd a szükséges csomagokat és a spaCy modellt
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pandas numpy scikit-learn tqdm nltk spacy thefuzz
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Megjegyzés: **az első indítás** (a GloVe modell letöltése és a dokumentumok vektorizálása) rendszerint **körülbelül 5–10 percet vesz igénybe**, a hálózati sebességtől és a számítógép teljesítményétől függően.

3. Indítás:
```powershell
# a projekt gyökérkönyvtárában
.\.venv\Scripts\python.exe Main.py
# vagy ha aktiváltad a venv-et
python Main.py
```
4. Példa bemenet a futó programhoz:
```text
Stranger Things, Breaking Bad
```

Működés röviden
- A program először betölti az adatokat, előfeldolgozza a leírásokat (spaCy + NLTK stop-szavak), majd GloVe vektorokat használ a dokumentumok vektorizálásához.
- A számítások eredményét a .cache mappába menti, így ismételt futtatás gyorsabb lesz.
- Ajánlás: a megadott filmcímekhez hasonló könyveket választ a műfaj-átfedés és koszinusz-hasonlóság alapján.

Gyakori problémák és megoldások
- "can't open file '... pip'": Ne futtasd így: python pip install spacy. Helyette használd:
```powershell
python -m pip install spacy
# vagy a venv Pythonjával
.\.venv\Scripts\python.exe -m pip install spacy
```
- "No module named spacy": A csomag nincs telepítve a használt Python környezetbe — telepítsd a fent említett módon.
- Ha a spaCy modell hiányzik:
```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```
- GloVe letöltése: az első futtatáskor a script letölti a glove.6B.zip fájlt a .glove mappába (internet szükséges). Ha megszakad, töröld a részben letöltött fájlt és futtasd újra.

További megjegyzések
- A projekt egyszerűsített demonstrációs célú; a pontosság és robusztusság javítható (további normalizálás, lemmatizálás, finomabb súlyozás stb.).

