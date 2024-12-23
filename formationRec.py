from flask import Flask, jsonify, request
import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="pfa-api-db"
    )

# Fetch data from the database
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

# Recommendation logic for formations
def recommend_formations(candidat_id):
    # Fetch candidat competencies
    candidat_competencies_query = f"""
        SELECT competence.name
        FROM candidat_competences
        JOIN competence ON candidat_competences.competence_id = competence.id
        WHERE candidat_competences.candidat_id = {candidat_id}
    """
    candidat_competencies = [row['name'] for row in fetch_data(candidat_competencies_query)]
    print(f"Candidat Competencies: {candidat_competencies}")

    # Fetch all job offer competencies
    job_offer_competencies_query = """
        SELECT DISTINCT competence.name
        FROM offre_emploi_competence
        JOIN competence ON offre_emploi_competence.competence_id = competence.id
    """
    job_offer_competencies = [row['name'] for row in fetch_data(job_offer_competencies_query)]
    print(f"Job Offer Competencies: {job_offer_competencies}")

    # Identify missing competencies
    missing_competencies = list(set(job_offer_competencies) - set(candidat_competencies))
    print(f"Missing Competencies: {missing_competencies}")

    if not missing_competencies:
        print("No missing competencies identified.")
        return []

    # Fetch formations that cover missing competencies
    formation_query = """
        SELECT formation.id, formation.titre, formation.description, formation.plateforme, competence.name AS competence
        FROM formation
        JOIN formation_competence ON formation.id = formation_competence.formation_id
        JOIN competence ON formation_competence.competence_id = competence.id
    """
    formations = fetch_data(formation_query)
    print(f"Formations Retrieved: {formations}")

    if not formations:
        print("No formations found.")
        return []

    formation_df = pd.DataFrame(formations)
    print(f"Formation DataFrame:\n{formation_df}")

    # Filter formations covering missing competencies
    relevant_formations = formation_df[formation_df['competence'].isin(missing_competencies)]
    print(f"Relevant Formations:\n{relevant_formations}")

    if relevant_formations.empty:
        print("No relevant formations found.")
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

    # Compute cosine similarity between formations and job offer competencies
    job_offer_vector = tfidf_matrix[-1]
    formation_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(job_offer_vector, formation_vectors).flatten()

    # Add similarity scores to formations
    grouped_formations['relevance_score'] = similarities

    # Sort formations by relevance score
    sorted_formations = grouped_formations.sort_values(by='relevance_score', ascending=False)

    print(f"Sorted Formations:\n{sorted_formations}")

    # Convert to dict for JSON response
    return sorted_formations[['id', 'titre', 'description', 'plateforme', 'competence', 'relevance_score']].to_dict(orient='records')

@app.route('/recommendations/formations/<int:candidat_id>', methods=['GET'])
def get_formation_recommendations(candidat_id):
    try:
        print(f"Fetching recommendations for candidat_id: {candidat_id}")
        recommendations = recommend_formations(candidat_id)
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in recommendation logic: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '_main_':
    app.run(debug=True)