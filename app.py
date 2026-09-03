import streamlit as st
import pdfplumber
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PhD Supervisor Scout", layout="wide")

def extract_text_from_pdf(uploaded_file):
    """Extract clean text handling multi-column academic CV formats."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()

def extract_phd_keywords(text, top_n=5):
    """Extract niche technical n-grams rather than noisy raw CV text."""
    # Clean non-alphanumeric noise
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    
    # Custom stop words common in CVs that dilute search
    academic_stop_words = [
        'cv', 'curriculum', 'vitae', 'resume', 'experience', 'education', 
        'university', 'department', 'skills', 'projects', 'gpa', 'email', 
        'phone', 'github', 'prof', 'professor', 'dr', 'student', 'bachelor', 
        'master', 'phd', 'candidate', 'research', 'work', 'present'
    ]
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=1
    )
    
    try:
        tfidf = vectorizer.fit_transform([clean_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        
        # Filter out generic CV terms
        ranked_terms = [
            term for term, score in sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
            if not any(w in term for w in academic_stop_words) and len(term) > 3
        ]
        return ranked_terms[:top_n]
    except Exception:
        return [word for word in clean_text.split() if len(word) > 4][:top_n]

def search_semantic_scholar(query, limit=20):
    """
    Find relevant recent papers via Semantic Scholar API, 
    then extract active primary investigators/supervisors.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,citationCount,venue,openAccessPdf"
    }
    
    response = requests.get(url, params=params, timeout=12)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

# App Header
st.title("🎓 PhD Supervisor Scout")
st.caption("Find potential PhD advisors by matching your CV or niche research topic against the Semantic Scholar graph.")

with st.sidebar:
    st.subheader("1. Your Background")
    cv_file = st.file_uploader("Upload Academic CV (PDF)", type=["pdf"])
    manual_interest = st.text_area(
        "Or specify target research topic:",
        placeholder="e.g., Physics-informed neural networks for fluid dynamics"
    )
    
    year_filter = st.slider("Only consider papers published after:", min_value=2018, max_value=2026, value=2021)
    submit_btn = st.button("Find Matching Supervisors", type="primary")

if submit_btn:
    raw_cv_text = ""
    extracted_keywords = []

    if cv_file:
        raw_cv_text = extract_text_from_pdf(cv_file)
        if raw_cv_text:
            st.success(f"✓ Parsed {len(raw_cv_text.split())} words from CV.")
            extracted_keywords = extract_phd_keywords(raw_cv_text, top_n=4)
        else:
            st.warning("Could not parse text from this PDF (it might be an image scan).")

    # Determine core search terms
    search_query = ""
    if manual_interest.strip():
        search_query = manual_interest.strip()
    elif extracted_keywords:
        search_query = " ".join(extracted_keywords)
    
    if not search_query:
        st.error("Please enter a research topic or upload a readable text PDF.")
    else:
        st.info(f"Targeting search terms: **{search_query}**")
        
        with st.spinner("Querying peer-reviewed literature and identifying authors..."):
            papers = search_semantic_scholar(search_query, limit=25)
            
            if not papers:
                st.warning("No papers found matching this exact query. Try refining or shortening your keywords.")
            else:
                supervisors = {}
                
                # Aggregate authors from relevant recent publications
                for paper in papers:
                    year = paper.get("year") or 0
                    if year < year_filter:
                        continue
                    
                    paper_title = paper.get("title", "Untitled")
                    abstract = paper.get("abstract") or ""
                    authors = paper.get("authors", [])
                    
                    if not authors:
                        continue
                    
                    # In academic publishing, supervisors/PIs are almost always the last author (or first)
                    candidate_authors = [authors[0]]
                    if len(authors) > 1:
                        candidate_authors.append(authors[-1])
                        
                    for auth in candidate_authors:
                        auth_id = auth.get("authorId")
                        auth_name = auth.get("name")
                        
                        if not auth_id or not auth_name:
                            continue
                            
                        if auth_id not in supervisors:
                            supervisors[auth_id] = {
                                "name": auth_name,
                                "profile_url": f"https://www.semanticscholar.org/author/{auth_id}",
                                "papers": [],
                                "citations": paper.get("citationCount", 0),
                                "corpus": ""
                            }
                        
                        supervisors[auth_id]["papers"].append({
                            "title": paper_title,
                            "year": year,
                            "venue": paper.get("venue", "Conference/Journal"),
                            "abstract": abstract
                        })
                        supervisors[auth_id]["corpus"] += f" {paper_title} {abstract}"

                # Rank supervisors based on cosine similarity against user query/CV
                comparison_text = raw_cv_text if raw_cv_text else search_query
                all_profiles = list(supervisors.values())
                
                if not all_profiles:
                    st.warning("No active authors found for the selected year window.")
                else:
                    texts_to_vectorize = [comparison_text] + [p["corpus"] for p in all_profiles]
                    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(texts_to_vectorize)
                    user_vec = tfidf_matrix[0]
                    profile_vecs = tfidf_matrix[1:]
                    
                    sims = cosine_similarity(user_vec, profile_vecs)[0]
                    
                    for idx, profile in enumerate(all_profiles):
                        profile["similarity"] = round(float(sims[idx]) * 100, 1)

                    ranked_supervisors = sorted(all_profiles, key=lambda x: x["similarity"], reverse=True)
                    
                    st.subheader(f"Found {len(ranked_supervisors)} Potential Supervisors")
                    
                    for prof in ranked_supervisors[:10]:
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"### [{prof['name']}]({prof['profile_url']})")
                                st.markdown("**Relevant Recent Publication:**")
                                top_paper = prof["papers"][0]
                                st.write(f"📄 *{top_paper['title']}* ({top_paper['year']})")
                                if top_paper['abstract']:
                                    st.caption(top_paper['abstract'][:300] + "...")
                            with c2:
                                st.metric("Relevance Match", f"{prof['similarity']}%")
                                st.write(f"Matched Papers: {len(prof['papers'])}")
