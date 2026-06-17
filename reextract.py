import sys
import asyncio

sys.path.insert(0, '/Users/devshree/Documents/Pro/living-resume')

from backend.ingestion.document import get_all_chunks
from backend.agents.graph_agent import extract_and_populate_graph

async def main():
    person_name = "Devshree Jadeja"
    chunks = get_all_chunks()
    print(f"Found {len(chunks)} chunks. Running extraction...")
    extraction = await extract_and_populate_graph(chunks, person_name)
    print("Extraction Result:", extraction)

if __name__ == "__main__":
    asyncio.run(main())
