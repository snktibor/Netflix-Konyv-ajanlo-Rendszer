import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import time
import ast
from tqdm import tqdm
import zipfile
import urllib.request
import spacy
from thefuzz import process

# NLTK adatok letöltése (stopwords, tokenizer)
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK 'stopwords' letöltése...")
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' letöltése...")
        nltk.download('punkt')

stop_words = None
spacy_nlp = None

def get_document_vector(tokens, embeddings_dict, dim=100):
    vectors = []
    for token in tokens:
        if token in embeddings_dict:
            vectors.append(embeddings_dict[token])
    
    if not vectors:
        return np.zeros(dim)
    
    return np.array(vectors).mean(axis=0)

#A 'dim' lehet 50, 100, 200, 300.
def load_glove_model(dim=100):
    glove_zip_url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
    glove_zip_path = ".glove/glove.6B.zip"
    glove_dir = ".glove/glove.6B"
    modelfile_name = f"glove.6B.{dim}d.txt"
    modelfile_path = os.path.join(glove_dir, modelfile_name)
    
    #400k sor van
    total_lines = 400000 

    #Letöltés (ha a ZIP hiányzik)
    if not os.path.exists(glove_zip_path) and not os.path.exists(modelfile_path):
        print(f"GloVe modell letöltése ({glove_zip_url})...")
        # tqdm progress bar a letöltéshez
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=glove_zip_url.split('/')[-1]) as t:
            urllib.request.urlretrieve(glove_zip_url, glove_zip_path, 
                                       reporthook=lambda b, bsize, tsize: t.update(bsize))
    
    #Kicsomagolás (ha a TXT fájl hiányzik, de a ZIP megvan)
    if not os.path.exists(modelfile_path):
        print(f"GloVe modell kicsomagolása ({glove_zip_path})...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            print(f"'{modelfile_name}' kicsomagolása...")
            zip_ref.extract(modelfile_name, path=glove_dir)
            print("Kicsomagolás kész.")
            
    #Betöltés
    print(f"Szóbeágyazás-modell betöltése: {modelfile_path}")
    print("Ez eltarthat 1-2 percig...")
    
    embeddings = {}
    with open(modelfile_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Modell betöltése"):
            parts = line.split()
            word = parts[0]
            try:
                vector = np.array(parts[1:]).astype('float')
                embeddings[word] = vector
            except ValueError:
                #hibás sor, átugorjuk
                pass
                
    print(f"Modell betöltve, {len(embeddings)} szóvektorral.")
    return embeddings, dim

def load_datasets():
    print("Adathalmazok betöltése a helyi mappákból...")

    netflix_full_path = os.path.join('datasets', 'Netflix_movies_and_tv_shows.csv')
    books_full_path = os.path.join('datasets', 'goodreads_data.csv')

    try:
        netflix_df = pd.read_csv(netflix_full_path)
        print(f"Sikeresen betöltve: {netflix_full_path} ({len(netflix_df)} sor)")
    except FileNotFoundError:
        print(f"Hiba: A {netflix_full_path} fájl nem található.")
        return None, None

    try:
        books_df = pd.read_csv(books_full_path)
        print(f"Sikeresen betöltve: {books_full_path} ({len(books_df)} sor)")
    except FileNotFoundError:
        print(f"Hiba: A '{books_full_path}' fájl nem található.")

    print("Adathalmazok sikeresen betöltve.")
    return netflix_df, books_df


def preprocess_text(text):
    global stop_words, spacy_nlp
    
    #NLTK stop_words betöltése (ha még nem történt meg)
    if stop_words is None:
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
        
    #SpaCy modell betöltése (ha még nem történt meg)
    if spacy_nlp is None:
        print("SpaCy 'en_core_web_sm' modell betöltése (csak egyszer)...")
        spacy_nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "lemmatizer"])
        print("Modell betöltve.")

    if not isinstance(text, str):
        return []

    doc = spacy_nlp(text)
    processed_tokens = [
        token.text.lower() #Kisbetűsítés
        for token in doc
        if token.is_alpha and #Csak betűk
            token.text.lower() not in stop_words #Stop-szavak szűrése
    ]
    return processed_tokens


def parse_netflix_genres(genres_str):
    #Kisbetűsítés és normalizálás
    cleaned_str = genres_str.lower()
    cleaned_str = re.sub(r'sci-fi|sci fi', 'science fiction', cleaned_str)
    cleaned_str = cleaned_str.replace('stand-up comedy', 'comedy')

    #Zavaró szavak
    noise_words = [
        'international', 
        'tv shows', 
        'tv series', 
        'tv dramas',
        'tv', 
        'movies', 
        'series',
        'dramas'
    ]
    
    for word in noise_words:
        cleaned_str = cleaned_str.replace(word, '')

    #Darabolás az összes elválasztó mentén (vessző, &, kötőjel)
    potential_genres = re.split(r'[,\-&]', cleaned_str)
    
    genres_set = set()
    
    #Végső tisztítás és hozzáadás a halmazhoz
    for genre in potential_genres:
        final_genre = genre.strip()
        if final_genre:
            genres_set.add(final_genre)
            
    return genres_set

def safe_genre_parse(genres_str):
    if not genres_str or not isinstance(genres_str, str):
        return set()

    unified_str = ""
    try:
        parsed_list = ast.literal_eval(genres_str)
        
        if isinstance(parsed_list, list):
            unified_str = ", ".join(str(g) for g in parsed_list)
        else:
            #Ha pl. csak egy számot vagy szimpla stringet olvasott be
            unified_str = str(parsed_list)
            
    except (ValueError, SyntaxError, TypeError):
        unified_str = genres_str
    return parse_netflix_genres(unified_str)

def build_recommendation_models(netflix_df, books_df):
    print("\nSzöveges adatok előfeldolgozása...")
    start_time = time.time()
    
    #Hiányzó adatok kezelése
    netflix_df['description'] = netflix_df['description'].fillna('')
    netflix_df['listed_in'] = netflix_df['listed_in'].fillna('')
    books_df['Description'] = books_df['Description'].fillna('')
    books_df['Genres'] = books_df['Genres'].fillna("[]")
    
    #Műfaj modellezés
    netflix_df['genres_list'] = netflix_df['listed_in'].apply(parse_netflix_genres)
    books_df['genres_list'] = books_df['Genres'].apply(safe_genre_parse)

    tqdm.pandas(desc="Szövegfeldolgozás")
    #Tartalom modellezés
    netflix_df['cleaned_tokens'] = netflix_df['description'].progress_apply(preprocess_text)
    books_df['cleaned_tokens'] = books_df['Description'].progress_apply(preprocess_text)
    
    print(f"Szöveg előfeldolgozás kész. Időtartam: {time.time() - start_time:.2f} mp.")
    
    #Modellépítés
    print("Szóbeágyazás modell építése...")
    start_time = time.time()
    
    #GloVe modell betöltése
    embeddings_dict, EMBEDDING_DIM = load_glove_model(dim=300) 
    
    #Dokumentum vektorok kiszámítása
    print("Netflix leírások vektorizálása...")
    netflix_vectors_list = [get_document_vector(tokens, embeddings_dict, EMBEDDING_DIM) 
                            for tokens in tqdm(netflix_df['cleaned_tokens'], desc="Netflix")]
    
    print("Könyv leírások vektorizálása...")
    books_vectors_list = [get_document_vector(tokens, embeddings_dict, EMBEDDING_DIM) 
                          for tokens in tqdm(books_df['cleaned_tokens'], desc="Könyvek")]

    #Átalakítás NumPy mátrixszá
    netflix_doc_vectors = np.array(netflix_vectors_list)
    books_doc_vectors = np.array(books_vectors_list)
    
    print(f"Ajánlórendszer felépítve (GloVe alapján). Időtartam: {time.time() - start_time:.2f} mp.")
    
    print("Vektorok és adatok mentése a lemezre...")
    np.save('.cache/netflix_vectors.npy', netflix_doc_vectors)
    np.save('.cache/books_vectors.npy', books_doc_vectors)

    netflix_df.to_parquet('.cache/netflix_df.parquet')
    books_df.to_parquet('.cache/books_df.parquet')

    return netflix_doc_vectors, books_doc_vectors, netflix_df, books_df

def recommend_books(user_movie_titles, netflix_df, books_df, netflix_vectors, books_vectors, top_n=5):
    movie_indices = []
    user_movie_genres = set() #filmek műfajai

    for title in user_movie_titles:
        #Cím alapján index keresése
        all_movie_titles = netflix_df['title'].tolist()
        matches = process.extract(title, all_movie_titles, limit=3)
        if matches and matches[0][1] > 80: #Csak ha a legjobb egyezés > 80%
            best_match_title = matches[0][0]
            idx = netflix_df.index[netflix_df['title'] == best_match_title].tolist()[0]
            movie_indices.append(idx)
            user_movie_genres.update(netflix_df.iloc[idx]['genres_list'])
        else:
            print(f"Figyelem: A '{title}' című film nem található...")
            
    if not movie_indices:
        print("Nem található egyező film, az ajánlás nem lehetséges.")
        return

    recommendation_from_movies = [netflix_df['title'].iloc[i] for i in movie_indices]
    print(f"\nAjánlások keresése a következő filmek alapján: {recommendation_from_movies}")

    print(f"A felhasználó hangulata (normalizált műfajok): {user_movie_genres}")

    #Hangulat/Műfaj Szűrés: Olyan könyveket keresünk, amiknek van legalább egy közös műfaja
    def has_common_genre(book_genres_list):
        return not user_movie_genres.isdisjoint(book_genres_list)

    filtered_book_indices = books_df[books_df['genres_list'].apply(has_common_genre)].index
    
    recommendation_title = "" #Dinamikus cím
    
    if filtered_book_indices.empty:
        print("Figyelem: Nem található a hangulatnak (műfajnak) megfelelő könyv.")
        print("Keresés kiterjesztése a teljes adatbázisra (csak téma alapján)...")
        
        #Fallback: A teljes adatbázist használjuk
        filtered_book_indices = books_df.index
        filtered_books_vectors = books_vectors
        recommendation_title = f"Top {top_n} könyvajánlat (Csak Téma alapján):"
        
    else:
        print(f"Hangulat alapján {len(books_df)} könyvből {len(filtered_book_indices)} releváns találat szűrve.")
        
        #A szűrt listát használjuk
        filtered_books_vectors = books_vectors[filtered_book_indices]
        recommendation_title = f"Top {top_n} könyvajánlat (Hangulat + Téma alapján):"

    #Téma Hasonlóság (TF-IDF) a szűrt listán:
    
    user_movie_vectors = netflix_vectors[movie_indices]
    user_profile_vector = np.array(user_movie_vectors.mean(axis=0)).reshape(1, -1)
    
    #Koszinusz-hasonlóság számítása
    similarity_scores = cosine_similarity(user_profile_vector, filtered_books_vectors)
    
    top_n_filtered_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    
    #Visszaalakítjuk az indexeket az eredeti 'books_df' indexeire
    top_book_indices = [filtered_book_indices[i] for i in top_n_filtered_indices]


    top_scores = [similarity_scores[0][i] for i in top_n_filtered_indices]

    print("-" * 30)
    print(recommendation_title)
    
    results = []
    for i, (idx, score) in enumerate(zip(top_book_indices, top_scores)):
        book = books_df.iloc[idx]
        
        #A leírás néha hiányozhat
        description_snippet = "Nincs leírás."
        if pd.notna(book['Description']):
            description_snippet = book['Description'][:300] + "..."
            
        #A szerző néha hiányozhat
        author_name = book.get('Author', 'Ismeretlen szerző')
        if pd.isna(author_name):
            author_name = 'Ismeretlen szerző'

        result = {
            "Cím": book['Book'],
            "Szerző": author_name,
            "Hasonlóság": f"{score:.4f}",
            "Műfajok": book['Genres'],
            "Leírás": description_snippet
        }
        results.append(result)
        
    for res in results:
        print(f"\nCím: {res['Cím']} (Szerző: {res['Szerző']})")
        print(f"  Hasonlóság: {res['Hasonlóság']}")
        print(f"  Műfajok: {res['Műfajok']}")
        print(f"  Leírás: {res['Leírás']}")
    
    print("-" * 30)
    return results, recommendation_title, recommendation_from_movies