import streamlit as st
from io import StringIO
import pickle
import streamlit as st
import requests
import pandas as pd
import pathlib
from pathlib import Path
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import io
import base64
import csv
from csv import writer
import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(
     page_title="Anime Recommendation System",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<center><img src='data:image/png;base64,{}' style='width:230px;height:200px;'></center>".format(
    img_to_bytes("anime.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)


varname = st.sidebar.selectbox("Select plug-in", ("Home", "Recommendations", "EDA"))

if varname == "Home":
    st.markdown("<h1 style='text-align: center; color: #00ced1;'>Welcome to Anime Recommendation System</h1><br><br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: ; color: #C3B091;'>Created by:</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: ; color: white;'>Student Name: Muhammad Aliff bin Md Norman</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: ; color: white;'>Student id: 2020971927</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: ; color: white;'>Supervised by: Madam Nurzeatul Hamimah binti Abdul Hamid</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: ; color: green;'>Purposed this system is to help users find an anime based on their preferences from selected anime.</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: ; color: red;'>*Notes : Kindly remind this only prototype and not all anime cover in the dataset. Thank you :)</h4>", unsafe_allow_html=True)


elif varname == "Recommendations":

    st.markdown("<h1 style='text-align: center; color: #00ced1;'>Anime Recommender System Item-Based</h1><br><br>", unsafe_allow_html=True)
    def get_recommendation(anime_name):

        sim_score = similarity.sort_values(by=anime_name, ascending=False).loc[:, anime_name].tolist()[1:]
        sim_animes = similarity.sort_values(by=anime_name, ascending=False).index[1:]

        return sim_animes, sim_score


    animes = pickle.load(open('anime.pkl','rb'))
    test = pickle.load(open('test.pkl','rb'))
    #pivot_norm = pickle.load(open('pivot_norm.pkl','rb'))
    similarity = pickle.load(open('anime_itembased.pkl','rb'))

    pivot = pd.pivot_table(test, index='name', columns='user_id', values='rating')
    pivot.dropna(axis=1, how='all', inplace=True)
    pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
    pivot_norm.fillna(0, inplace=True)


    anime_list = animes['name'].values
    selected_anime = st.selectbox("Type or select a anime from the dropdown",  anime_list)

    animeurl = "https://myanimelist.net/anime/"

    if st.button('Show Recommendation'):
        
        if selected_anime not in pivot_norm.index:
            st.markdown("<h2 style='text-align: center; color: #FF0000;'>NO ANIME IN THIS DATASET...PLS TRY ANOTHER ANIME</h2><br><br>", unsafe_allow_html=True)
        else:
            anime, score = get_recommendation(selected_anime)
            col1, col2, col3, col4, col5 = st.columns(5)

            name = [] 
            genre = [] 
            image = []
            animeid = []

            for x,y in zip(anime[:10], score[:10]):
            
                name.append(x)
                genre.append(animes[animes['name'] == x].iloc[0].genre_x)
                animeid.append(animes[animes['name'] == x].iloc[0].anime_id)
                response = requests.get(animes[animes['name'] == x].iloc[0].image_url)
                m = response.status_code

                if m == 404:
                    img = Image.open('picture4.png')
                    image.append(img)
        
                else:
                    img = Image.open(io.BytesIO(response.content))
                    image.append(img)
          
            with col1:
                real = animeurl + str(animeid[0])
                st.image(image[0])
                st.text(name[0])
                st.text("Genre: " + genre[0])
                st.write("[More info](%s)" % real)

            with col2:
                real = animeurl + str(animeid[1])
                st.image(image[1])
                st.text(name[1])
                st.text("Genre: " + genre[1])
                st.write("[More info](%s)" % real)

            with col3:
                real = animeurl + str(animeid[2])
                st.image(image[2])
                st.text(name[2])
                st.text("Genre: " + genre[2])
                st.write("[More info](%s)" % real)

            with col4:
                real = animeurl + str(animeid[3])
                st.image(image[3])
                st.text(name[3])
                st.text("Genre: " + genre[3])
                st.write("[More info](%s)" % real)

            with col5:
                real = animeurl + str(animeid[4])
                st.image(image[4])
                st.text(name[4])
                st.text("Genre: " + genre[4])
                st.write("[More info](%s)" % real)

            col6, col7, col8, col9, col10 = st.columns(5)
            
            with col6:
                real = animeurl + str(animeid[5])
                st.image(image[5])
                st.text(name[5])
                st.text("Genre: " + genre[5])
                st.write("[More info](%s)" % real)

            with col7:
                real = animeurl + str(animeid[6])
                st.image(image[6])
                st.text(name[6])
                st.text("Genre: " + genre[6])
                st.write("[More info](%s)" % real)

            with col8:
                real = animeurl + str(animeid[7])
                st.image(image[7])
                st.text(name[7])
                st.text("Genre: " + genre[7])
                st.write("[More info](%s)" % real)

            with col9:
                real = animeurl + str(animeid[8])
                st.image(image[8])
                st.text(name[8])
                st.text("Genre: " + genre[8])
                st.write("[More info](%s)" % real)

            with col10:
                real = animeurl + str(animeid[9])
                st.image(image[9])
                st.text(name[9])
                st.text("Genre: " + genre[9])
                st.write("[More info](%s)" % real)


elif varname == "EDA":

    @st.cache
    def load_csv():
        csv = pd.read_csv("AnimeDataset.csv")
        return csv

    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.markdown("<h1 style='text-align: center; color: #00ced1;'>Anime DataFrame</h1><br><br>", unsafe_allow_html=True)
    st.write(df)
    st.write('---')

    st.markdown("<h1 style='text-align: center; color: #00ced1;'>Anime Profiling Report</h1>", unsafe_allow_html=True)
    st_profile_report(pr)
    


   