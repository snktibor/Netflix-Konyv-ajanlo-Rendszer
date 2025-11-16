import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import time
import ast
from tqdm import tqdm
from thefuzz import process
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


def load_datasets():
    print("Adathalmazok betöltése a helyi mappákból...")

    netflix_full_path = os.path.join('datasets', 'netflix_titles.csv')
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

def join_genres_for_encoding(genre_set):
    if not genre_set:
        return ""
    return " ".join(sorted(list(genre_set)))

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
    
    # Hiányzó adatok kezelése
    netflix_df['description'] = netflix_df['description'].fillna('')
    netflix_df['listed_in'] = netflix_df['listed_in'].fillna('')
    books_df['Description'] = books_df['Description'].fillna('')
    books_df['Genres'] = books_df['Genres'].fillna("[]")
    
    # Műfaj modellezés
    netflix_df['genres_list'] = netflix_df['listed_in'].apply(parse_netflix_genres)
    books_df['genres_list'] = books_df['Genres'].apply(safe_genre_parse)

    print(f"Műfajok feldolgozva. Időtartam: {time.time() - start_time:.2f} mp.")
    
    # MODELLÉPÍTÉS (Súlyozott)
    print("Sentence-Transformer modell építése...")
    start_time = time.time()
    
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # Leírás vektorok
    print("Netflix leírások kódolása...")
    netflix_desc_vectors = model.encode(
        netflix_df['description'].tolist(), 
        show_progress_bar=True
    )
    print("Könyv leírások kódolása...")
    books_desc_vectors = model.encode(
        books_df['Description'].tolist(), 
        show_progress_bar=True
    )

    # Műfaj vektorok
    print("Netflix műfajok kódolása...")
    netflix_df['genre_text'] = netflix_df['genres_list'].apply(join_genres_for_encoding)
    netflix_genre_vectors = model.encode(
        netflix_df['genre_text'].tolist(), 
        show_progress_bar=True
    )
    
    print("Könyv műfajok kódolása...")
    books_df['genre_text'] = books_df['genres_list'].apply(join_genres_for_encoding)
    books_genre_vectors = model.encode(
        books_df['genre_text'].tolist(), 
        show_progress_bar=True
    )

    # Súlyozott átlagolás
    print("Vektorok súlyozása (70% leírás, 30% műfaj)...")
    
    DESC_WEIGHT = 0.7 
    GENRE_WEIGHT = 0.3

    # szuper-vektort
    netflix_doc_vectors = (netflix_desc_vectors * DESC_WEIGHT) + (netflix_genre_vectors * GENRE_WEIGHT)
    books_doc_vectors = (books_desc_vectors * DESC_WEIGHT) + (books_genre_vectors * GENRE_WEIGHT)

    # Normalizálás
    netflix_doc_vectors = normalize(netflix_doc_vectors)
    books_doc_vectors = normalize(books_doc_vectors)
    
    print(f"Ajánlórendszer felépítve (Súlyozott). Időtartam: {time.time() - start_time:.2f} mp.")
    
    # Mentés
    print("Vektorok és adatok mentése a lemezre...")
    os.makedirs('.cache', exist_ok=True) 
    np.save('.cache/netflix_vectors.npy', netflix_doc_vectors)
    np.save('.cache/books_vectors.npy', books_doc_vectors)

    # Töröljük a felesleges segédoszlopokat mentés előtt
    if 'genre_text' in netflix_df.columns:
        netflix_df = netflix_df.drop(columns=['genre_text'])
    if 'genre_text' in books_df.columns:
        books_df = books_df.drop(columns=['genre_text'])

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