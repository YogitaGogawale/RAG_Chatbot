import fitz  #PyMuPDF

def load_pdf (path):
    doc = fitz.open(path)
    chunks = []
    #for i, page in enumerate(doc):
    #    text = page.get_text()
    #    print(f"page{i+1}:{len(text)} characters | preview: {text[:100]!r}")
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        #simple chunking, spliting into around 500 char blocks
        if not text or len(text)<50:
            continue
        
        # Clean up the text
        text = text.replace('\xa0', ' ')
        text = ' '.join(text.split())

        lines = text.split('.')
        avg_len = sum(len(l) for l in lines)/max(len(lines),1)
        if avg_len <40:
            print(f"Skipping page {page_num+1} (looks like a table)")
            continue

        #split into sentence based chunks
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) +len(sentence)<600:
                current_chunk += sentence +". "
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_num+1
                    })
                current_chunk = sentence +". "

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page": page_num + 1
            })
    
    return chunks


chunks = load_pdf(r"C:\Users\hp\Desktop\PythonThesis\RAG testing.docx")
print(f"Created {len(chunks)} chunks")


from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("my_pdf")

texts =[c["text"] for c in chunks] 
embeddings = model.encode(texts).tolist()

collection.add(
    documents = texts,
    embeddings = embeddings,
    metadatas=[{"page": c["page"]} for c in chunks],
    ids = [f"chunk_{i}" for i in range(len(chunks))]
)

print("PDF indexed!")


def ask(question):
    q_embeddings = model.encode([question]).tolist()

    results = collection.query(query_embeddings =q_embeddings, n_results = 3)
    context = "\n\n".join(results["documents"][0])


    prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question : {question}
Answer:"""
    
    import ollama
    response = ollama.chat(
        model = "llama2",
        messages = [{"role":"user", "content": prompt}]
    )

    return response["message"]["content"]

answer = ask("what is the main topic of this document?")
print(answer)