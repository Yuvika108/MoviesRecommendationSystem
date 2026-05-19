import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Cinematic Match | Movie Recommendations",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for high-quality, premium visual aesthetics
st.markdown("""
<style>
    /* Dark glassmorphism styled headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #FF4B4B, #85203b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card style for recommendations */
    .movie-card {
        background-color: #1e1e24;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.15);
    }
    .movie-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .movie-meta {
        font-size: 0.9rem;
        color: #FF4B4B;
        font-weight: 600;
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
        st.error("Error: Could not find `movies.pkl` or `movies_dict.pkl`. Please run `MovieRecommendationSystem.py` first to process data.")
        st.stop()
        
    # Compute similarity matrix
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tag']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    
    return new_df, similarity_matrix

def main():
    st.markdown('<div class="main-title">🎬 Cinematic Match</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Discover your next favorite movie using advanced content-based filtering</div>', unsafe_allow_html=True)
    
    # Load dataset & similarity matrix
    with st.spinner("Initializing cinematic engine..."):
        new_df, similarity_matrix = load_data_and_similarity()
        
    # Search controls
    st.markdown("### Search for a Movie")
    movie_list = sorted(new_df['title'].tolist())
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        movie_list,
        index=movie_list.index("Avatar") if "Avatar" in movie_list else 0,
        placeholder="Type or select a movie...",
    )
    
    num_recs = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)
    
    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        try:
            movie_index = new_df[new_df['title'] == selected_movie].index[0]
            distances = similarity_matrix[movie_index]
            
            # Sort distances (similarities) descending, exclude index 0 (the movie itself)
            movie_list_recs = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recs+1]
            
            st.markdown("---")
            st.markdown(f"### Movies Similar to **{selected_movie}**:")
            
            for i, (idx, score) in enumerate(movie_list_recs, 1):
                rec_title = new_df.iloc[idx]['title']
                
                # HTML Card Layout for premium look
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{i}. {rec_title}</div>
                    <div class="movie-meta">Similarity Score: {score:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar to show match percentage visually
                st.progress(float(score), text=f"Match strength: {score:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred while fetching recommendations: {e}")

if __name__ == "__main__":
    main()
