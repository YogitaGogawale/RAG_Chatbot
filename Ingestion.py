import fitz  #PyMuPDF
import ollama
from sentence_transformers import SentenceTransformer
import chromadb

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
            if len(current_chunk) +len(sentence)<300:
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


chunks = load_pdf(r"C:\Users\hp\Desktop\PythonThesis\RAG testing.pdf")
print(f"Created {len(chunks)} chunks")


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

    results = collection.query(query_embeddings =q_embeddings, n_results = 5)
    context = "\n\n".join(results["documents"][0])
    pages =[m["page"] for m in results["metadatas"][0]]


    prompt = f"""Answer the question based only on the context below.
    Use ONLY the context below to answer. Be specific and complete.
    If the answer is not in the context, say exactly: "This information is not in the document."
    Do not guess or add information from outside the context.   

Context:
{context}

Question : {question}
Answer:"""
    
    response = ollama.chat(
        model = "llama2",
        messages = [{"role":"user", "content": prompt}]
    )

    answer = response["message"]["content"]
    print(f"Pages used:{pages}")
    return answer

"""
Questions = [
    "What is the main purpose of this norm?",
    "What types of machines does this document cover?",
    "What is functional status B?",
    "What is the price of the machine?",
    "Who is the CEO of CLAAS?",
]

for q in Questions:
    print(f"\n{'='*50}")
    print(f"Q:{q}")
    print(f"A:{ask(q)}")
"""