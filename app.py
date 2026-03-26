import streamlit as st
import ollama
import chromadb
import fitz
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Chatbot", page_icon="📄")
st.title("📄 chat with your PDF")
st.caption("Requirements document")

@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection("my_pdf")
    return model, client, collection

@st.cache_resource
def index_pdf(_model,_collection):
    if _collection.count()>0:
        return   #already indexed
    
    with st.spinner("Indexing PDF.... please wait"):
        doc = fitz.open(r"C:\Users\hp\Desktop\PythonThesis\RAG testing.pdf")  # fix path
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if not text or len(text)<50:
                continue
            text =text.replace('\xa0', ' ')
            text = ' '.join(text.split())

            lines = text.split('.')
            avg_len = sum(len(l) for l in lines)/ max(len(lines),1)
            if avg_len < 40:
                continue

            sentences = text.split('. ')
            current_chunk = " "
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 300:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append({"text": current_chunk.strip(), "page": page_num + 1})
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                chunks.append({"text": current_chunk.strip(), "page": page_num + 1})

        texts = [c["text"] for c in chunks]
        embeddings = _model.encode(texts).tolist()
        _collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"page": c["page"]} for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

def ask(question, model, collection):
    q_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=5)

    context = "\n\n".join(results["documents"][0])
    pages = sorted(set([m["page"] for m in results["metadatas"][0]]))

    prompt = f"""You are a technical assistant for a CLAAS EMC document.
Use ONLY the context below to answer. Be specific and complete.
If the answer is not in the context, say exactly: "This information is not in the document."
Do not guess or add information from outside the context.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], pages


model, client, collection = load_models()
index_pdf(model, collection)

with st.sidebar:
    st.title(" About")
    st.markdown("**Model:** llama2 (local)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown(f"**Chunks indexed:** {collection.count()}")
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
        
 
# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "pages" in msg:
            st.caption(f"📖 Sources: pages {msg['pages']}")

# ---- Chat input ----
if question := st.chat_input("Ask something about the document..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, pages = ask(question, model, collection)
        st.markdown(answer)
        st.caption(f"📖 Sources: pages {pages}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pages": pages
    })

