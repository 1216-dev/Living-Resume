import sys
import asyncio
from pathlib import Path

# Add project root to sys path
sys.path.insert(0, '/Users/devshree/Documents/Pro/living-resume')

from backend.config import DATA_DIR, GRAPH_DB_PATH, CHROMA_DB_PATH
from backend.ingestion.document import ingest_file, get_all_chunks
from backend.agents.graph_agent import extract_and_populate_graph
import chromadb
from backend.knowledge.graph import get_graph
import networkx as nx
from backend.knowledge.bm25_index import rebuild_bm25_from_chroma

async def main():
    print("Resetting database...")
    # Clear ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection("living_resume")
        print("ChromaDB cleared.")
    except Exception as e:
        print(f"ChromaDB clear error: {e}")

    # Clear graph
    try:
        graph = get_graph()
        graph.g = nx.DiGraph()
        graph._community_map = {}
        graph._community_summaries = {}
        graph.save()
        print("Graph cleared.")
    except Exception as e:
        print(f"Graph clear error: {e}")

    # Rebuild BM25 (will be empty)
    try:
        rebuild_bm25_from_chroma()
        print("BM25 rebuilt.")
    except Exception as e:
        print(f"BM25 clear error: {e}")
        
    print("Ingesting Resume...")
    resume_path = "/Users/devshree/Documents/Pro/living-resume/data/uploads/ML_Devshree_Resume_.pdf"
    person_name = "Devshree Jadeja"
    
    try:
        result = ingest_file(resume_path, person_name)
        print("Chunks ingested:", result)
        
        chunks = get_all_chunks()
        print(f"Total chunks: {len(chunks)}")
        
        print("Running Entity Extractor...")
        extraction = await extract_and_populate_graph(chunks, person_name)
        print("Extraction Result:", extraction)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
