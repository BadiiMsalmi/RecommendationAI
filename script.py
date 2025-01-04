from flask import Flask, jsonify, request
import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio

app = Flask(__name__)

# ----------------------------------------
# Database Connection and Utilities
# ----------------------------------------

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="pfa-api-db"
    )

def fetch_data(query):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return []

# ----------------------------------------
# Formations Recommendation Logic
# ----------------------------------------

def recommend_formations(candidat_id):
    # Fetch candidat competencies
    candidat_competencies_query = f"""
        SELECT competence.name
        FROM candidat_competences
        JOIN competence ON candidat_competences.competence_id = competence.id
        WHERE candidat_competences.candidat_id = {candidat_id}
    """
    candidat_competencies = [row['name'] for row in fetch_data(candidat_competencies_query)]

    # Fetch all job offer competencies
    job_offer_competencies_query = """
        SELECT DISTINCT competence.name
        FROM offre_emploi_competence
        JOIN competence ON offre_emploi_competence.competence_id = competence.id
    """
    job_offer_competencies = [row['name'] for row in fetch_data(job_offer_competencies_query)]

    # Identify missing competencies
    missing_competencies = list(set(job_offer_competencies) - set(candidat_competencies))

    if not missing_competencies:
        return []

    # Fetch formations that cover missing competencies
    formation_query = """
        SELECT formation.id, formation.titre, formation.description, formation.plateforme, competence.name AS competence
        FROM formation
        JOIN formation_competence ON formation.id = formation_competence.formation_id
        JOIN competence ON formation_competence.competence_id = competence.id
    """
    formations = fetch_data(formation_query)
    formation_df = pd.DataFrame(formations)

    # Filter formations covering missing competencies
    relevant_formations = formation_df[formation_df['competence'].isin(missing_competencies)]

    if relevant_formations.empty:
        return []

    # Group formations by id and combine competencies into a list
    grouped_formations = relevant_formations.groupby(
        ['id', 'titre', 'description', 'plateforme']
    )['competence'].apply(list).reset_index()

    # Add a combined text column for similarity scoring
    grouped_formations['combined_text'] = grouped_formations.apply(
        lambda row: f"{row['titre']} {row['description']} {row['competence']}", axis=1
    )

    # Use TF-IDF to calculate formation relevance
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        grouped_formations['combined_text'].tolist() + [" ".join(job_offer_competencies)]
    )

    # Compute cosine similarity
    job_offer_vector = tfidf_matrix[-1]
    formation_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(job_offer_vector, formation_vectors).flatten()

    # Add similarity scores to formations
    grouped_formations['relevance_score'] = similarities

    # Sort formations by relevance score
    sorted_formations = grouped_formations.sort_values(by='relevance_score', ascending=False)

    return sorted_formations[['id', 'titre', 'description', 'plateforme', 'competence', 'relevance_score']].to_dict(orient='records')

@app.route('/api/formations/recommendations/<int:candidat_id>', methods=['GET'])
def get_formation_recommendations(candidat_id):
    try:
        recommendations = recommend_formations(candidat_id)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# Offres Recommendation Logic
# ----------------------------------------

def calculate_location_similarity(candidate_location, job_location):
    if not candidate_location or not job_location:
        return 0
    return ratio(candidate_location.lower(), job_location.lower())

def recommend_offres(candidat_id):
    candidat_query = f"SELECT * FROM candidat WHERE id = {candidat_id}"
    candidat = fetch_data(candidat_query)[0]
    candidat_location = candidat.get("location", "")

    competences_query = f"""
        SELECT competence.name
        FROM candidat_competences
        JOIN competence ON candidat_competences.competence_id = competence.id
        WHERE candidat_competences.candidat_id = {candidat_id}
    """
    candidat_competences = " ".join([row['name'] for row in fetch_data(competences_query)])

    offres_query = """
        SELECT id, titre, description, experience, localisation, salaire
        FROM offre_emploi
        WHERE status = 'OPEN'
    """
    offres = fetch_data(offres_query)
    offres_df = pd.DataFrame(offres)

    offres_df['content'] = offres_df.apply(
        lambda row: f"{row['titre']} {row['description']} {row['localisation']}", axis=1
    )

    vectorizer = TfidfVectorizer()
    all_texts = offres_df['content'].tolist() + [candidat_competences]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    candidate_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]
    skill_similarities = cosine_similarity(candidate_vector, job_vectors).flatten()

    offres_df['location_similarity'] = offres_df['localisation'].apply(
        lambda loc: calculate_location_similarity(candidat_location, loc)
    )

    skill_weight = 0.7
    location_weight = 0.3
    offres_df['combined_score'] = (
        skill_weight * skill_similarities + location_weight * offres_df['location_similarity']
    )

    top_offres = offres_df.sort_values(by='combined_score', ascending=False).head(5)
    return top_offres[['id', 'titre', 'description', 'combined_score']].to_dict(orient='records')

@app.route('/api/offres/recommendations/<int:candidat_id>', methods=['GET'])
def get_offre_recommendations(candidat_id):
    try:
        recommendations = recommend_offres(candidat_id)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------
# Run the Application
# ----------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=5000)
