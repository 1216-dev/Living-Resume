import sys
sys.path.insert(0, '/Users/devshree/Documents/Pro/living-resume')
from backend.ingestion.document import ingest_file
try:
    with open('/tmp/test.txt', 'w') as f:
        f.write("Hello world")
    ingest_file('/tmp/test.txt', "Devshree")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
