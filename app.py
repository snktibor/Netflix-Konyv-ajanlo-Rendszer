import streamlit as st
import os
import pandas as pd
import numpy as np
from recommender_logic import (
    load_datasets,
    build_recommendation_models,
    recommend_books
)

# --- KONFIGURÁCIÓ ---
RENDER_DISK_PATH = os.environ.get('DISK_PATH') 
GLOVE_DIR = os.path.join('.glove')
CACHE_DIR = os.path.join('.cache')

import recommender_logic
recommender_logic.glove_dir = GLOVE_DIR
recommender_logic.cache_dir = CACHE_DIR

#MODELL BETÖLTÉSE
@st.cache_resource
def load_all_models():
    """
    Betölti a modelleket a Render Persistent Disk-ről, vagy felépíti őket, 
    ha még nem léteznek.
    """
    os.makedirs(GLOVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    #download_nltk_data() # NLTK letöltés (a Rendernek is kell)

    n_vec_path = os.path.join(CACHE_DIR, 'netflix_vectors.npy')
    b_vec_path = os.path.join(CACHE_DIR, 'books_vectors.npy')
    n_df_path = os.path.join(CACHE_DIR, 'netflix_df.parquet')
    b_df_path = os.path.join(CACHE_DIR, 'books_df.parquet')

    # placeholder az állapotüzenetekhez — ezt kiürítjük a folyamat végén
    msg_ph = st.empty()

    if (os.path.exists(n_vec_path) and os.path.exists(b_vec_path) and
        os.path.exists(n_df_path) and os.path.exists(b_df_path)):

        msg_ph.info("Elő-számított modellek (cache) betöltése...")
        n_vectors = np.load(n_vec_path, allow_pickle=True)
        b_vectors = np.load(b_vec_path, allow_pickle=True)
        n_df = pd.read_parquet(n_df_path)
        b_df = pd.read_parquet(b_df_path)
        # eltávolítjuk az előzetes üzenetet
        msg_ph.empty()
        msg_ph.success("Modellek betöltve a gyorsítótárból.")
    
    else:
        msg_ph.warning("Gyorsítótár nem található, modellek felépítése nulláról...")
        msg_ph.info("Ez az első indításkor 10-15 percet is igénybe vehet. Kérlek várj...")
        
        # Adatok betöltése
        netflix_data, book_data = load_datasets() 
        if netflix_data is None or book_data is None:
            msg_ph.empty()
            st.error("Adatfájlok (CSV) nem találhatóak!")
            return None
            
        # Modellek felépítése (és mentése a CACHE_DIR-be)
        n_vectors, b_vectors, n_df, b_df = build_recommendation_models(netflix_data, book_data)
        # felépítés után eltávolítjuk az info/warning üzeneteket
        msg_ph.empty()
        st.success("Modellek felépítve és gyorsítótárazva.")
    
    return n_vectors, b_vectors, n_df, b_df

# --- WEB FRONT-END (Streamlit UI) ---
st.set_page_config(page_title="Film -> Könyv Ajánló", layout="centered", page_icon="🎬", initial_sidebar_state="collapsed",     menu_items={
        'About': 'https://github.com/snktibor/Netflix-Konyv-ajanlo-Rendszer',
        'Get Help': None,
        'Report a Bug': None
    })
st.title("🎬 Film-Könyv Ajánló 🍿")
st.write("Adj meg egy Netflix filmet (vagy többet, vesszővel elválasztva), és mi ajánlunk hozzá hasonló könyveket!")

st.set_option("client.showErrorDetails", False)
st.set_option("client.showSidebarNavigation", False)
st.set_option("client.toolbarMode", False)

# Modellek betöltése
model_data = load_all_models()

if model_data:
    n_vectors, b_vectors, n_df, b_df = model_data

    with st.form(key='recommendation_form', border=False):
        user_input = st.text_input(
            "Filmcím(ek):",
            "Stranger Things, The Queen's Gambit"
        )
        submit_button = st.form_submit_button("Ajánlások Keresése")

    # Ajánlás gomb
    if submit_button:
        movie_titles = [title.strip() for title in user_input.split(',') if title.strip()]
        
        if not movie_titles:
            st.error("Kérem, adjon meg legalább egy filmcímet.")
        else:
            with st.spinner("Keresés... A fuzzy matching és a számítás eltarthat pár másodpercig..."):
                # Futtatjuk a logikát
                results, title, movie_titles = recommend_books(
                    movie_titles,
                    n_df, b_df, n_vectors, b_vectors,
                    top_n=5
                )
            st.info(f"Ajánlások keresése a következő filmek alapján: {', '.join(movie_titles)}")
            st.success(f"✔️ {title}")
            
            # Eredmények megjelenítése
            for res in results:
                st.subheader(f"{res['Cím']} (Szerző: {res['Szerző']})")
                st.caption(f"Hasonlóság: {res['Hasonlóság']} | Műfajok: {res['Műfajok']}")
                st.write(res['Leírás'])
                st.divider()
else:
    st.error("A modellek betöltése nem sikerült. Kérjük, ellenőrizze a logokat.")