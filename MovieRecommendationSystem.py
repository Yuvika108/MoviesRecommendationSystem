import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load the processed movie dataframe from movies.pkl or movies_dict.pkl."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(current_dir, "movies.pkl")
    dict_path = os.path.join(current_dir, "movies_dict.pkl")

    if os.path.exists(pkl_path):
        print(f"Loading dataset from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    elif os.path.exists(dict_path):
        print(f"Loading dataset from: {dict_path}")
        with open(dict_path, "rb") as f:
            data_dict = pickle.load(f)
            return pd.DataFrame(data_dict)
    else:
        # Check if CSV files are available in current directory or parent directory
        csv_path = os.path.join(current_dir, "tmdb_5000_movies.csv")
        credits_path = os.path.join(current_dir, "tmdb_5000_credits.csv")
        
        if os.path.exists(csv_path) and os.path.exists(credits_path):
            print("CSV files found. Processing raw data...")
            movies = pd.read_csv(csv_path)
            credits = pd.read_csv(credits_path)
            # Basic preprocessing similar to the notebook
            movies = movies.merge(credits, on="title")
            movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            movies.dropna(inplace=True)
            
            # Simple string cleaning helper
            def clean_list(x):
                if isinstance(x, str):
                    import ast
                    try:
                        lst = ast.literal_eval(x)
                        return " ".join([i['name'].replace(" ", "") for i in lst])
                    except:
                        return ""
                return ""
            
            movies['genres'] = movies['genres'].apply(clean_list)
            movies['keywords'] = movies['keywords'].apply(clean_list)
            
            def clean_cast(x):
                import ast
                try:
                    lst = ast.literal_eval(x)
                    return " ".join([i['name'].replace(" ", "") for i in lst[:3]])
                except:
                    return ""
            movies['cast'] = movies['cast'].apply(clean_cast)
            
            def get_director(x):
                import ast
                try:
                    lst = ast.literal_eval(x)
                    for i in lst:
                        if i['job'] == 'Director':
                            return i['name'].replace(" ", "")
                except:
                    return ""
                return ""
            movies['crew'] = movies['crew'].apply(get_director)
            
            movies['tag'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['crew']
            new_df = movies[['movie_id', 'title', 'tag']].copy()
            new_df['tag'] = new_df['tag'].apply(lambda x: str(x).lower())
            
            # Save the processed data for future runs
            with open(pkl_path, "wb") as f:
                pickle.dump(new_df, f)
            return new_df
        else:
            print("Error: Could not find movies.pkl, movies_dict.pkl, or raw tmdb csv files.")
            sys.exit(1)

def get_recommendations(new_df, similarity_matrix, movie_title, num_recommendations=5):
    """Recommend top N similar movies using similarity matrix."""
    try:
        # Perform a case-insensitive match for the movie title
        match = new_df[new_df['title'].str.lower() == movie_title.lower()]
        if match.empty:
            # Try a partial match
            match = new_df[new_df['title'].str.lower().str.contains(movie_title.lower())]
            if match.empty:
                return None, f"Movie '{movie_title}' not found in the database."
            else:
                movie_title = match.iloc[0]['title']
        
        movie_index = match.index[0]
        distances = similarity_matrix[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        recommendations = []
        for idx, score in movie_list:
            recommendations.append({
                'title': new_df.iloc[idx]['title'],
                'score': float(score)
            })
        return movie_title, recommendations
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def main():
    print("=" * 50)
    print("       MOVIE RECOMMENDATION SYSTEM ENGINE       ")
    print("=" * 50)
    
    new_df = load_data()
    print(f"Loaded {len(new_df)} movies successfully.")
    
    print("Computing feature vectors and cosine similarity matrix...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tag']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    print("Similarity matrix computed successfully!\n")
    
    # Check if a movie argument was passed from command line
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
        matched_title, recs = get_recommendations(new_df, similarity_matrix, search_query)
        if matched_title:
            print(f"\nTop recommendations for '{matched_title}':")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec['title']} (similarity: {rec['score']:.4f})")
        else:
            print(recs)
        return

    # Default demo runs
    demo_movies = ["Avatar", "Batman Begins", "Spider-Man 3"]
    print("=== RUNNING DEFAULT DEMO RECOMMENDATIONS ===")
    for movie in demo_movies:
        matched_title, recs = get_recommendations(new_df, similarity_matrix, movie)
        if matched_title:
            print(f"\nTop recommendations for '{matched_title}':")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec['title']} (similarity: {rec['score']:.4f})")
        else:
            print(recs)
    
    print("\n" + "=" * 50)
    print("=== INTERACTIVE MODE ===")
    print("Type a movie title to get recommendations (or 'exit' to quit):")
    try:
        while True:
            user_input = input("\nEnter movie name: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting Movie Recommendation System. Goodbye!")
                break
            
            matched_title, recs = get_recommendations(new_df, similarity_matrix, user_input)
            if matched_title:
                print(f"\nTop recommendations for '{matched_title}':")
                for i, rec in enumerate(recs, 1):
                    print(f"  {i}. {rec['title']} (similarity: {rec['score']:.4f})")
            else:
                print(recs)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Movie Recommendation System. Goodbye!")

if __name__ == "__main__":
    main()

