import streamlit as st
from pypdf import PdfReader
import requests
import urllib.parse

st.set_page_config(page_title="PhD Supervisor Scout", layout="wide")

def read_pdf(file_obj):
    """Extract and sanitize text directly from the in-memory uploaded PDF."""
    try:
        reader = PdfReader(file_obj)
        extracted = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted.append(text)
        return "\n".join(extracted).strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def search_openalex_works(query, min_year=2021, limit=15):
    """
    Queries OpenAlex Works (papers).
    Works reliably for both broad terms (e.g. '5G', 'Internet') and niche PhD topics.
    """
    encoded_query = urllib.parse.quote(query)
    # polite pool with user-agent avoids throttling
    url = f"https://api.openalex.org/works?search={encoded_query}&filter=publication_year:>{min_year}&sort=cited_by_count:desc&per-page={limit}&mailto=phdscout_local@example.com"
    
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            return res.json().get("results", [])
        else:
            st.error(f"OpenAlex responded with status {res.status_code}")
            return []
    except Exception as e:
        st.error(f"Network error contacting academic database: {e}")
        return []

# App Header
st.title("🎓 PhD Supervisor Scout")
st.caption("Extracts lead researchers and active lab heads from global publications.")

# 1. Inputs (Two-Column Layout)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Option A: Upload CV (PDF)")
    uploaded_pdf = st.file_uploader("Upload your CV / Resume", type=["pdf"], key="cv_uploader")

with col_right:
    st.subheader("Option B: Enter Research Topic")
    manual_query = st.text_area(
        "Keywords or Abstract",
        placeholder="e.g., 5G beamforming, federated learning, or IoT security",
        height=100
    )

min_year = st.slider("Only show researchers active since year:", min_value=2018, max_value=2026, value=2021)

# Inspect extracted PDF content immediately
parsed_text = ""
if uploaded_pdf is not None:
    parsed_text = read_pdf(uploaded_pdf)
    if parsed_text:
        with st.expander("📄 Click to preview extracted CV text", expanded=False):
            st.text(parsed_text[:1200] + ("..." if len(parsed_text) > 1200 else ""))
    else:
        st.warning("⚠️ No text detected in this PDF. If this is an exported image/scan, please type your topic on the right.")

# 2. Search Execution
if st.button("Find Supervisors", type="primary"):
    # Determine the query string
    active_query = ""
    if manual_query.strip():
        active_query = manual_query.strip()
    elif parsed_text:
        # Take the first ~150 characters of substantive text from the CV
        lines = [line.strip() for line in parsed_text.split("\n") if len(line.strip()) > 20]
        active_query = " ".join(lines[:3])[:150]

    if not active_query:
        st.warning("Please enter a research topic or upload a readable text PDF.")
    else:
        st.info(f"Searching academic publications for: **{active_query}**")
        
        with st.spinner("Searching peer-reviewed literature and identifying professors..."):
            works = search_openalex_works(active_query, min_year=min_year, limit=20)
            
            if not works:
                st.warning(f"No recent papers found for '{active_query}'. Try a more standard keyword.")
            else:
                professors = {}
                
                for work in works:
                    paper_title = work.get("display_name", "Untitled")
                    paper_url = work.get("doi") or work.get("id")
                    authorships = work.get("authorships", [])
                    
                    if not authorships:
                        continue
                    
                    # In academic STEM/PhD publishing, the Last Author is typically the Lab Head / Principal Investigator (PI),
                    # and the First Author is typically the Lead Researcher.
                    key_authors = []
                    key_authors.append(authorships[0])
                    if len(authorships) > 1:
                        key_authors.append(authorships[-1])
                        
                    for auth_entry in key_authors:
                        author = auth_entry.get("author", {})
                        author_id = author.get("id")
                        author_name = author.get("display_name")
                        
                        if not author_id or not author_name:
                            continue
                            
                        # Get institution affiliation
                        institutions = auth_entry.get("institutions", [])
                        inst_name = institutions[0].get("display_name") if institutions else "Independent / Unlisted"
                        
                        if author_id not in professors:
                            professors[author_id] = {
                                "name": author_name,
                                "institution": inst_name,
                                "profile_url": author_id,
                                "papers": []
                            }
                            
                        professors[author_id]["papers"].append({
                            "title": paper_title,
                            "year": work.get("publication_year"),
                            "citations": work.get("cited_by_count", 0),
                            "link": paper_url
                        })

                # Display Results
                st.success(f"Found {len(professors)} potential advisors actively publishing on this topic.")
                
                for prof_id, prof in professors.items():
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"### [{prof['name']}]({prof['profile_url']})")
                            st.markdown(f"🏛️ **Institution:** {prof['institution']}")
                            
                            st.markdown("**Recent Matching Work:**")
                            for p in prof["papers"][:2]:
                                st.write(f"- [{p['title']}]({p['link']}) ({p['year']}) — *{p['citations']} citations*")
                        
                        with c2:
                            st.metric("Matching Papers", len(prof["papers"]))
