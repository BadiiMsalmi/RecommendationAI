from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import ratio
import mysql.connector

app = Flask(__name__)

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",  # Use "localhost" without the port
        port=3306,         # Specify port separately
        user="root",
        password="",
        database="pfa-api-db"
    )

# Fetch data from database
def fetch_data(query):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

# Calculate location similarity based on string matching
def calculate_location_similarity(candidate_location, job_location):
    if not candidate_location or not job_location:
        return 0  # No similarity if one of the locations is missing
    # Use Levenshtein ratio for string similarity (closer to 1 means higher similarity)
    return ratio(candidate_location.lower(), job_location.lower())

# Recommendation logic using AI
def recommend_offres(candidat_id):
    # Fetch candidat details
    candidat_query = f"SELECT * FROM candidat WHERE id = {candidat_id}"
    candidat = fetch_data(candidat_query)[0]
    candidat_location = candidat.get("location", "")

    # Fetch candidat competences
    competences_query = f"""
        SELECT competence.name
        FROM candidat_competences
        JOIN competence ON candidat_competences.competence_id = competence.id
        WHERE candidat_competences.candidat_id = {candidat_id}
    """
    candidat_competences = " ".join([row['name'] for row in fetch_data(competences_query)])

    # Fetch job offers
    offres_query = """
        SELECT id, titre, description, experience, localisation, salaire
        FROM offre_emploi
        WHERE status = 'OPEN'
    """
    offres = fetch_data(offres_query)
    offres_df = pd.DataFrame(offres)

    # Create a content column for job offers
    offres_df['content'] = offres_df.apply(
        lambda row: f"{row['titre']} {row['description']} {row['localisation']}", axis=1
    )

    # Use TF-IDF Vectorizer for textual similarity
    vectorizer = TfidfVectorizer()
    all_texts = offres_df['content'].tolist() + [candidat_competences]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compute cosine similarity between candidate and job offers
    candidate_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]
    skill_similarities = cosine_similarity(candidate_vector, job_vectors).flatten()

    # Calculate location similarity for each job offer
    offres_df['location_similarity'] = offres_df['localisation'].apply(
        lambda loc: calculate_location_similarity(candidat_location, loc)
    )

    # Combine skill similarity and location similarity (weighted sum)
    skill_weight = 0.7
    location_weight = 0.3
    offres_df['combined_score'] = (
        skill_weight * skill_similarities + location_weight * offres_df['location_similarity']
    )

    # Sort and return top 5 recommendations
    top_offres = offres_df.sort_values(by='combined_score', ascending=False).head(5)
    return top_offres[['id', 'titre', 'description', 'combined_score']].to_dict(orient='records')

@app.route('/recommendations/<int:candidat_id>', methods=['GET'])
def get_recommendations(candidat_id):
    try:
        print("Attempting to call recommend_offres...")
        recommendations = recommend_offres(candidat_id)
        print("recommend_offres executed successfully.")
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
