import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("Chat with your PDF 📄💬")
st.write("1) Upload a PDF.  2) Ask a question about it.  3) Get an answer.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file is not None:
    st.success("PDF uploaded! Next step: extract text.")
    st.write(f"File name: {uploaded_file.name}")

    # Extract text from all pages in the PDF
    pdf_reader = PdfReader(uploaded_file)
    pages_text = []
    for page in pdf_reader.pages:
        pages_text.append(page.extract_text())

    pdf_text = "\n".join(pages_text)
    if pdf_text.strip():
        st.text_area("Extracted PDF Text (preview)", pdf_text[:1000], height=250)
        
        # Function to split text into chunks (~500-1000 characters per chunk)
        def chunk_text(text, chunk_size=500):
            paragraphs = text.split('\n')
            chunks = []
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n"
                else:
                    chunks.append(current_chunk)
                    current_chunk = para + "\n"
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        chunks = chunk_text(pdf_text)
        st.write(f"Total chunks created: {len(chunks)}")
        st.text_area("Text chunk preview", chunks[0] if chunks else "No chunks", height=100)

        question = st.text_input("Ask a question about your PDF:")

        if question and chunks:
            # Use TF-IDF for simple text similarity (no heavy dependencies)
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            
            # Combine chunks and question for vectorization
            all_texts = chunks + [question]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between question and all chunks
            question_vector = tfidf_matrix[-1]  # Last vector is the question
            chunk_vectors = tfidf_matrix[:-1]   # All before are chunks
            
            similarities = cosine_similarity(question_vector, chunk_vectors)[0]
            
            # Get the most similar chunk
            top_idx = similarities.argmax()
            best_match_score = similarities[top_idx]
            
            st.subheader("Best-matching answer from your PDF:")
            st.info(f"Match confidence: {best_match_score:.1%}")
            st.write(chunks[top_idx])
