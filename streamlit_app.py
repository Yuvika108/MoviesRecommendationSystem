import os
import pickle
import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Cinematix | AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium typography, shadows, card layout, and hover zoom effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');

    /* Global Typography overrides */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main glassmorphism header */
    .hero-container {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.03) 0%, rgba(147, 51, 234, 0.05) 100%);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.01);
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #E11D48, #BE185D, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Space Grotesk', sans-serif;
        color: #4B5563;
        font-size: 1.15rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Movie Details Section */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.7rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        color: #111827;
        border-left: 4px solid #E11D48;
        padding-left: 12px;
    }

    /* Selected Movie Details Styles */
    .movie-tagline {
        font-style: italic;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    .movie-meta-row {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .genre-container {
        margin-bottom: 1.5rem;
    }
    
    .genre-pill {
        background: #F3F4F6;
        color: #374151;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    /* Grid elements and Hover zoom for poster cards */
    .poster-container {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07);
        transition: transform 0.4s cubic-bezier(0.165, 0.84, 0.44, 1), box-shadow 0.4s ease;
    }
    .poster-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(225, 29, 72, 0.12);
    }

    /* Customized Recommendation Cards */
    .rec-card {
        background: #F9FAFB;
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.01);
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        background: #FFFFFF;
        border-color: rgba(225, 29, 72, 0.2);
        box-shadow: 0 8px 16px rgba(225, 29, 72, 0.06);
    }
    
    .rec-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1F2937;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        line-height: 1.3;
    }
    
    .rec-score {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        color: #E11D48;
        font-weight: 600;
        margin-top: 0.4rem;
        background: rgba(225, 29, 72, 0.05);
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
        align-self: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_similarity():
    """Load the preprocessed dataset and compute the similarity matrix."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(current_dir, "movies.pkl")
    dict_path = os.path.join(current_dir, "movies_dict.pkl")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            new_df = pickle.load(f)
    elif os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            data_dict = pickle.load(f)
            new_df = pd.DataFrame(data_dict)
    else:
        st.error("Error: Could not find `movies.pkl` or `movies_dict.pkl` in dataset path.")
        st.stop()
        
    # Compute similarity matrix
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tag']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    
    return new_df, similarity_matrix

def get_movie_poster(movie_id):
    """Fetch movie poster URL from TMDB API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&auto=format&fit=crop&q=60"

def get_movie_details(movie_id):
    """Fetch detailed metadata of the selected movie from TMDB."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'overview': data.get('overview'),
                'year': data.get('release_date', '')[:4],
                'rating': data.get('vote_average', 0.0),
                'tagline': data.get('tagline'),
                'runtime': data.get('runtime'),
                'genres': [g['name'] for g in data.get('genres', [])]
            }
    except Exception:
        pass
    return None

def main():
    # Hero Title Section
    st.markdown("""
    <div class="hero-container">
        <div class="main-title">CINEMATIX</div>
        <div class="subtitle">An advanced content-based movie discovery engine powered by Cosine Similarity</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load dataset & similarity matrix
    with st.spinner("Initializing cinematic engine..."):
        new_df, similarity_matrix = load_data_and_similarity()
        
    # Search layout
    st.markdown('<div class="section-header">Find Movie Matches</div>', unsafe_allow_html=True)
    col_input, col_slider = st.columns([2, 1])
    
    movie_list = sorted(new_df['title'].tolist())
    
    with col_input:
        selected_movie = st.selectbox(
            "Select or type a movie title:",
            movie_list,
            index=movie_list.index("Avatar") if "Avatar" in movie_list else 0
        )
        
    with col_slider:
        num_recs = st.slider("Number of recommendations:", min_value=4, max_value=12, value=6, step=2)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Selected movie info
    selected_idx = new_df[new_df['title'] == selected_movie].index[0]
    selected_id = new_df.iloc[selected_idx]['movie_id']
    
    # Fetch details and poster for the selected movie
    selected_poster = get_movie_poster(selected_id)
    details = get_movie_details(selected_id)
    
    # Main content split
    col_poster, col_info = st.columns([1, 2.2])
    
    with col_poster:
        st.markdown(f'<div class="poster-container"><img src="{selected_poster}" style="width:100%; border-radius:14px; display:block;"></div>', unsafe_allow_html=True)
        
    with col_info:
        st.markdown(f"<h2 style='margin-top:0;'>{selected_movie}</h2>", unsafe_allow_html=True)
        
        if details:
            # Display tagline if available
            if details['tagline']:
                st.markdown(f'<div class="movie-tagline">"{details["tagline"]}"</div>', unsafe_allow_html=True)
            
            # Display metadata row
            meta_info = []
            if details['year']:
                meta_info.append(details['year'])
            if details['runtime']:
                meta_info.append(f"{details['runtime']} min")
            if details['rating']:
                meta_info.append(f"⭐ {details['rating']:.1f}/10")
            
            if meta_info:
                st.markdown(f'<div class="movie-meta-row">{" &nbsp;•&nbsp; ".join(meta_info)}</div>', unsafe_allow_html=True)
            
            # Display genres pills
            if details['genres']:
                pills = "".join([f'<span class="genre-pill">{g}</span>' for g in details['genres']])
                st.markdown(f'<div class="genre-container">{pills}</div>', unsafe_allow_html=True)
            
            # Display overview
            if details['overview']:
                st.markdown(f"<p style='font-size:1.05rem; line-height:1.6; color:#374151;'>{details['overview']}</p>", unsafe_allow_html=True)
        else:
            # Fallback to local tag
            tag_clean = new_df.iloc[selected_idx]['tag']
            st.write(f"**Local tags:** {tag_clean[:200]}...")
            
        st.markdown("---")
        st.write("Ready to discover matching cinema? Click the button below.")
        trigger = st.button("Generate Recommendations", type="primary", use_container_width=True)

    if trigger:
        distances = similarity_matrix[selected_idx]
        # Sort recommendations descending (excluding itself)
        movie_list_recs = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recs+1]
        
        st.markdown('<div class="section-header">Recommended For You</div>', unsafe_allow_html=True)
        
        # Display recommendations in a grid of cards
        cols_per_row = 4 if num_recs % 4 == 0 else 3
        
        # Split recommendations into chunks based on columns per row
        chunks = [movie_list_recs[i:i + cols_per_row] for i in range(0, len(movie_list_recs), cols_per_row)]
        
        for chunk in chunks:
            cols = st.columns(cols_per_row)
            for col, (idx, score) in zip(cols, chunk):
                rec_title = new_df.iloc[idx]['title']
                rec_id = new_df.iloc[idx]['movie_id']
                rec_poster = get_movie_poster(rec_id)
                
                with col:
                    # Render poster inside poster-container for hover animation
                    st.markdown(f'<div class="poster-container"><img src="{rec_poster}" style="width:100%; border-radius:14px; display:block;"></div>', unsafe_allow_html=True)
                    
                    # Render Card Info
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-title">{rec_title}</div>
                        <div class="rec-score">Match Strength: {score:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

