import streamlit as st
from pypdf import PdfReader
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page configuration
st.set_page_config(page_title="PhD Supervisor Scout", layout="wide")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def search_openalex_authors(query_text, limit=15):
    """Fetch relevant authors from OpenAlex based on research concepts."""
    url = f"https://api.openalex.org/authors?search={requests.utils.quote(query_text)}&per-page={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    return []

# User Interface
st.title("🎓 PhD Supervisor Scout (Local)")
st.caption("Match your research interests or CV against academic profiles.")

with st.sidebar:
    st.header("Your Profile")
    cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
    research_summary = st.text_area(
        "Or paste research interests / thesis proposal:", 
        height=180,
        placeholder="e.g., Graph neural networks for drug discovery, molecular dynamics simulation..."
    )
    search_btn = st.button("Find Supervisors", type="primary")

# Main Execution
if search_btn:
    combined_query = research_summary
    
    if cv_file:
        parsed_cv = extract_pdf_text(cv_file)
        # Use top 1000 characters from CV to prevent token overflow
        combined_query += " " + parsed_cv[:1000]

    if not combined_query.strip():
        st.warning("Please provide a text summary or upload a CV.")
    else:
        with st.spinner("Searching global academic databases..."):
            # 1. Query OpenAlex API
            authors = search_openalex_authors(combined_query[:200])
            
            if not authors:
                st.info("No matching researchers found. Try broadening your keywords.")
            else:
                # 2. Semantic re-ranking using sentence embeddings
                user_vec = embedder.encode([combined_query])
                
                results = []
                for author in authors:
                    # Compile researcher topic fingerprint
                    topics = [c.get("display_name", "") for c in author.get("x_concepts", [])]
                    inst = author.get("last_known_institution")
                    inst_name = inst.get("display_name", "Unknown Institution") if inst else "Unknown Institution"
                    
                    profile_text = f"{author.get('display_name')} {inst_name} " + " ".join(topics)
                    profile_vec = embedder.encode([profile_text])
                    
                    sim = cosine_similarity(user_vec, profile_vec)[0][0]
                    
                    results.append({
                        "name": author.get("display_name"),
                        "institution": inst_name,
                        "works_count": author.get("works_count", 0),
                        "h_index": author.get("summary_stats", {}).get("h_index", "N/A"),
                        "topics": ", ".join(topics[:5]),
                        "profile_url": author.get("id"),
                        "similarity": round(float(sim) * 100, 1)
                    })
                
                # Sort descending by similarity score
                results = sorted(results, key=lambda x: x["similarity"], reverse=True)

                # 3. Display Results
                st.subheader(f"Top Matches ({len(results)})")
                
                for r in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### [{r['name']}]({r['profile_url']})")
                            st.write(f"🏛️ **Institution:** {r['institution']}")
                            st.write(f"🔬 **Core Topics:** {r['topics']}")
                        with col2:
                            st.metric("Match Score", f"{r['similarity']}%")
                            st.write(f"📚 Works: {r['works_count']} | H-Index: {r['h_index']}")
