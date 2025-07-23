from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params.document import Document as RAGDocument
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger as AgentEventLogger
import os

# Initialize the client
client = LlamaStackClient(base_url="http://localhost:8321")

vector_db_id = "tekton_docs"

EMBED_MODEL = "text-embedding-004"  # Vertex AI embedding model (768-dim)
EMBED_DIM = 768

response = client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=EMBED_MODEL,
    embedding_dimension=EMBED_DIM,
    provider_id="faiss",
)


# ---------------------------------------------------------------------------
# Local helper: walk docs directory on the developer machine
# ---------------------------------------------------------------------------


# Directories to ingest
DOC_DIRS = [
    "/home/zashaikh/go-projects/pipeline/docs",
    "/home/zashaikh/go-projects/pipelines-as-code/docs",
]


def list_local_files(root_dir: str):
    """Yield absolute file paths under `root_dir`."""

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def extension_to_mime(ext: str) -> str:
    """Return a rough mime-type for the given extension."""

    import mimetypes

    mime, _ = mimetypes.guess_type(f"dummy{ext}")
    return mime or "application/octet-stream"


all_files = []
for root in DOC_DIRS:
    print(f"[INFO] Scanning docs directory: {root}")
    root_files = list(list_local_files(root))
    print(f"[INFO]  └─ found {len(root_files)} files")
    all_files.extend((root, f) for f in root_files)

print(f"[INFO] Grand total files discovered: {len(all_files)}")

# Build RAGDocument list with the actual file contents (important for good retrieval!)
documents = []

TEXT_FORMATS = {".md", ".txt", ".html", ".csv", ".json", ".yaml", ".yml"}

for root_dir, abs_path in all_files:
    file_ext = os.path.splitext(abs_path)[1].lower()
    if file_ext not in TEXT_FORMATS:
        continue

    try:
        with open(abs_path, "rb") as fp:
            raw = fp.read()
    except Exception as err:
        print(f"[WARN] Could not read {abs_path}: {err}")
        continue

    if file_ext in TEXT_FORMATS:
        try:
            content_str = raw.decode("utf-8")
        except UnicodeDecodeError:
            content_str = raw.decode("latin-1", errors="ignore")
    else:
        content_str = "BINARY_FILE_PLACEHOLDER " + abs_path

    rel_path = os.path.relpath(abs_path, root_dir)

    documents.append(
        RAGDocument(
            document_id=rel_path,
            content=content_str,
            mime_type=extension_to_mime(file_ext),
            metadata={},
        )
    )
    print(f"[INFO] Added document: {rel_path} (size={len(content_str)} chars)")

if not documents:
    raise RuntimeError("No documents collected from docs directories – check paths and SUPPORTED_FORMATS.")


# ------------------------------------------------------------------
# Insert documents one-by-one to stay well under VertexAI 100-chunk limit
# ------------------------------------------------------------------

for idx, doc in enumerate(documents, 1):
    try:
        client.tool_runtime.rag_tool.insert(
            documents=[doc],
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=2048,  # larger chunks → fewer pieces per request
        )
        if idx % 10 == 0 or idx == len(documents):
            print(f"[INFO] Inserted {idx}/{len(documents)} documents")
    except Exception as e:
        print(f"[ERROR] Failed to insert {doc.document_id}: {e}")

# ------------------------------ AGENT SETUP ---------------------------------

# Enhanced system prompt guiding the assistant on how to leverage the RAG tool
SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant specialised in Tekton CI/CD and its official documentation. "
    "Whenever you answer a user, you MUST first identify the key Tekton topics or resources in the request "
    "and call the built-in knowledge_search tool with those terms. "
    "If the first search yields no useful context, reformulate and try again up to two times. "
    "Only after collecting relevant passages should you craft the final answer. "
    "Always cite the documents (with their file names) you used."
)


rag_agent = Agent(
    client,
    model=os.environ["INFERENCE_MODEL"],
    # Define instructions for the agent (system prompt)
    instructions=SYSTEM_INSTRUCTIONS,
    enable_session_persistence=False,
    # Define tools available to the agent
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {
                "vector_db_ids": [vector_db_id],
            },
        }
    ],
)

session_id = rag_agent.create_session("test-session")

# ---------------------------------------------------------------------------
# Prompt engineering: ask for an inlined PipelineRun only (no separate Tasks)
# ---------------------------------------------------------------------------

user_prompts = [
    (
        "Create a Tekton PipelineRun (ONLY the PipelineRun resource, no separate Task or Pipeline objects) that:\n"
        "• Clones the repository https://github.com/zakisk/pac-demo.\n"
        "• Runs the project’s tests.\n"
        "• Builds an OCI image from a Dockerfile in the repo.\n"
        "Hard requirements:\n"
        "1. Use taskSpec blocks embedded directly inside the PipelineRun (do not reference external Tasks or Pipelines).\n"
        "2. Use a single workspace shared across steps.\n"
        "3. Accept two Params: ‘git-url’ and ‘image-url’.\n"
        "4. Use catalog images (e.g. alpine/git, golang, buildah) that do not require private pulls.\n"
        "5. Output must be valid YAML starting with apiVersion.\n\n"
        "6. Output only the PipelineRun YAML, no other text or comments.\n"
        "Retrieve any necessary details from the docs via knowledge_search before answering."
    )
]

for prompt in user_prompts:
    print(f"User> {prompt}")
    response = rag_agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
        log.print()

