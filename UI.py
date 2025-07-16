import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import html
import os

# ----------------------------
# Load Saved Data
# ----------------------------

# Check if files exist
required_files = [
    "movies_df.pkl",
    "tfidf_vectorizer.pkl",
    "tfidf_matrix.pkl"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ Missing required files: {', '.join(missing_files)}")
    st.stop()

# Load Data
df = pd.read_pickle("movies_df.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

# ----------------------------
# Recommendation Function
# ----------------------------

def get_recommendations(title, df, tfidf_matrix):
    try:
        # Exact match
        idx_list = df.index[
            df["title"].str.lower().str.strip() == title.lower().strip()
        ].tolist()

        if not idx_list:
            # Partial match (contains)
            idx_list = df.index[
                df["title"].str.lower().str.contains(title.lower().strip(), na=False)
            ].tolist()

        if not idx_list:
            return None, f"❌ No movie found matching: '{title}'"

        # لو لقي أكتر من فيلم بنفس الاسم → ياخد أول واحد وخلاص
        idx = idx_list[0]

        # Safety check
        if idx >= len(df):
            return None, f"❌ Index {idx} out of range. Dataset size: {len(df)}."

        movie_row = df.iloc[idx]
        movie_vec = tfidf_matrix[idx]
        sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()
        sim_scores[idx] = 0
        top_indices = sim_scores.argsort()[::-1][:10]

        recommendations = df.iloc[top_indices][[
            "title",
            "poster_path",
            "release_date",
            "vote_average",
            "tagline",
            "homepage",
            "imdb_id"
        ]].copy()
        recommendations["similarity"] = sim_scores[top_indices]

        return movie_row, recommendations

    except Exception as e:
        return None, f"❌ Error occurred: {str(e)}"

# ----------------------------
# Helper Function
# ----------------------------

def format_date(value):
    if pd.notna(value) and value != "":
        try:
            return pd.to_datetime(value).strftime("%d-%m-%Y")
        except:
            return str(value)
    return 'N/A'

# ----------------------------
# Streamlit UI
# ----------------------------

base_image_url = "https://image.tmdb.org/t/p/w500"

# Inject CSS
st.markdown("""
    <style>
        body {
            background-color: #1C1C1E;
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .main-movie {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            align-items: flex-start;
        }
        .main-movie img {
            width: 300px;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(229, 9, 20, 0.5);
        }
        .movie-box {
            background-color: #2C2C2E;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 15px rgba(255, 0, 0, 0.4);
            flex: 1;
            min-width: 300px;
            max-height: 500px;
            overflow-y: auto;
        }
        .movie-title {
            color: #E50914;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .movie-info {
            color: #DDDDDD;
            font-size: 16px;
            margin-top: 10px;
            line-height: 1.5;
        }
        .button-container {
            margin-top: 15px;
        }
        .link-btn {
            display: inline-block;
            padding: 6px 12px;
            margin-right: 8px;
            background-color: #E50914;
            color: #fff;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            font-size: 13px;
            transition: background-color 0.3s;
        }
        .link-btn:hover {
            background-color: #B00610;
        }
        .rec-grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
            gap: 20px;
        }
        .rec-card {
            background-color: #2C2C2E;
            border-radius: 8px;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.5);
            width: 160px;
            height: 400px;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .rec-card:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px #E50914;
        }
        .rec-card img {
            width: 100%;
            height: 240px;
            object-fit: cover;
            border-radius: 5px 5px 0 0;
        }
        .rec-title {
            font-size: 16px;
            font-weight: bold;
            color: #E50914;
            margin: 10px 8px 0 8px;
            text-align: center;
        }
        .rec-info {
            font-size: 13px;
            color: #CCCCCC;
            margin: 5px 8px 10px 8px;
            text-align: center;
            max-height: 80px;
            overflow-y: auto;
        }
        .rec-links {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie title:")

if movie_name:
    movie_row, result = get_recommendations(movie_name, df, tfidf_matrix)

    if movie_row is None:
        st.warning(result)
    else:
        recommendations = result

        st.markdown("---")

        # Poster
        poster_path = movie_row.get('poster_path', '')
        if pd.notna(poster_path) and poster_path != '':
            poster_url = base_image_url + poster_path
            poster_html = f"<img src='{poster_url}'>"
        else:
            poster_html = "<div style='width:300px; height:450px; background:#444; display:flex; justify-content:center; align-items:center; color:#aaa;'>No Poster</div>"

        # Movie Info (escaped safely)
        tagline = html.escape(movie_row['tagline']) if pd.notna(movie_row.get('tagline')) and movie_row.get('tagline') else ''
        overview = html.escape(movie_row['overview']).replace('\n', '<br>') if pd.notna(movie_row.get('overview')) and movie_row.get('overview') else ''
        release_date = format_date(movie_row.get('release_date'))
        runtime = f"{int(movie_row['runtime'])} min" if pd.notna(movie_row.get('runtime')) else 'N/A'
        rating = f"{movie_row['vote_average']}/10" if pd.notna(movie_row.get('vote_average')) else 'N/A'
        genres = html.escape(movie_row['genres']) if pd.notna(movie_row.get('genres')) else 'N/A'

        homepage = movie_row.get("homepage", "")
        imdb_id = movie_row.get("imdb_id", "")
        imdb_url = f"https://www.imdb.com/title/{imdb_id}" if pd.notna(imdb_id) and imdb_id != '' else ''

        buttons_html = ""
        if pd.notna(homepage) and homepage != "":
            buttons_html += f"<a href='{homepage}' target='_blank' class='link-btn'>Official Website</a>"
        if imdb_url:
            buttons_html += f"<a href='{imdb_url}' target='_blank' class='link-btn'>IMDb</a>"

        details_html = f"""
            <div class="movie-box">
                <div class="movie-title">{movie_row['title']}</div>
                <div class="movie-info">
                    {f"<em>{tagline}</em><br>" if tagline else ""}
                    {overview if overview else ""}<br><br>
                    <b>Release Date:</b> {release_date}<br>
                    <b>Runtime:</b> {runtime}<br>
                    <b>Rating:</b> {rating}<br>
                    <b>Genres:</b> {genres}<br>
                    <div class="button-container">
                        {buttons_html}
                    </div>
                </div>
            </div>
        """

        final_html = f"""
            <div class="main-movie">
                {poster_html}
                {details_html}
            </div>
        """

        st.markdown(final_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### Because you watched **{movie_row['title']}**")

        # Recommended Movies Grid
        grid_html = "<div class='rec-grid'>"

        for _, row in recommendations.iterrows():
            card_html = "<div class='rec-card'>"

            poster_path = row.get('poster_path', '')
            if pd.notna(poster_path) and poster_path != '':
                poster_url = base_image_url + poster_path
                card_html += f"<img src='{poster_url}'>"
            else:
                card_html += "<div style='height: 240px; background-color: #444444; display:flex; align-items:center; justify-content:center;'><em>No poster</em></div>"

            card_html += f"<div class='rec-title'>{html.escape(row['title'])}</div>"

            info_text = ""
            tagline = html.escape(row['tagline']) if pd.notna(row.get('tagline')) and row.get('tagline') else ''
            release_date = format_date(row.get('release_date'))
            rating = f"{row['vote_average']}/10" if pd.notna(row.get('vote_average')) else ''

            if tagline:
                info_text += f"<em>{tagline}</em><br>"
            if release_date and release_date != 'N/A':
                info_text += f"Release: {release_date}<br>"
            if rating:
                info_text += f"Rating: {rating}"

            card_html += f"<div class='rec-info'>{info_text}</div>"

            homepage = row.get("homepage", "")
            imdb_id = row.get("imdb_id", "")
            imdb_url = f"https://www.imdb.com/title/{imdb_id}" if pd.notna(imdb_id) and imdb_id != '' else ''

            links_html = ""
            if pd.notna(homepage) and homepage != "":
                links_html += f"<a href='{homepage}' target='_blank' class='link-btn'>Website</a>"
            if imdb_url:
                links_html += f"<a href='{imdb_url}' target='_blank' class='link-btn'>IMDb</a>"

            if links_html:
                card_html += f"<div class='rec-links'>{links_html}</div>"

            card_html += "</div>"
            grid_html += card_html

        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)
