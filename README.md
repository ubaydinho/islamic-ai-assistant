# Islamic AI Assistant

A RAG-based (Retrieval-Augmented Generation) Islamic Q&A assistant with built-in content filtering, hallucination detection, and citation generation. Built with FastAPI, ChromaDB, and the Groq LLM API.

## Features

- **RAG-powered answers** — retrieves relevant passages from Qur'an, Hadith, Fiqh, Fatwa, and Scholar Opinion sources before generating a response, rather than relying purely on the LLM's own knowledge.
- **Islamic content filter** — screens both incoming queries and outgoing responses for content that doesn't align with Islamic teaching principles.
- **Intent classification** — routes each query into categories (`quran_tafsir`, `hadith_lookup`, `fiqh_question`, `ibadah_guide`, `dawa_content`, `aqeedah`, `general_islamic`, `out_of_scope`), so different types of questions get different handling.
- **Model routing by complexity** — simple/procedural questions use a fast model; questions needing deeper reasoning (fiqh, aqeedah, tafsir, hadith) are routed to a larger model.
- **Hallucination detection** — validates LLM output against retrieved source chunks using semantic similarity (embedding cosine similarity) rather than naive keyword matching, and returns a confidence score with each answer.
- **Citation generation** — every answer includes source references back to the original Qur'an/Hadith/Fiqh text used.
- **Session management** — conversation history is persisted per session (SQLite), so follow-up questions retain context.
- **Rate limiting** — token-bucket rate limiter to stay within Groq API limits.
- **Audit logging** — flagged content and query history are logged for review.

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM provider | Groq API (Llama 3.1/3.3, Gemma2) |
| Vector database | ChromaDB |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Session storage | SQLite |
| Containerization | Docker + Docker Compose |

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│ 1. Islamic Filter    │  ← blocks non-halal queries
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 2. Intent Classifier │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 3. RAG Retrieval     │  ← searches relevant ChromaDB collections
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 4. Prompt Builder    │  ← combines query + history + retrieved chunks
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 5. LLM Orchestrator  │  ← model selection + retry/fallback logic
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 6. Response Validator│  ← hallucination detection via semantic similarity
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 7. Response Filter   │  ← screens generated content
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 8. Citation Generator│
└─────────┬────────────┘
          ▼
   Final Response
   (answer, citations, confidence, intent)
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- A [Groq API key](https://console.groq.com) (free tier available)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ubaydinho/islamic-ai-assistant.git
   cd islamic-ai-assistant
   ```

2. Copy the environment template and fill in your values:
   ```bash
   cp .env.example .env
   ```
   At minimum, set `GROQ_API_KEY` in `.env`.

3. Start the services:
   ```bash
   docker compose up -d --build
   ```

4. Wait for the API to become healthy, then check:
   ```bash
   curl http://localhost:8000/health
   ```

### Usage

**Interactive API docs (Swagger UI):**
```
http://localhost:8000/docs
```

**Or use the included browser-based chat tester** — open `islamic_ai_chat.html` directly in your browser.

**Or via curl:**
```bash
# Create a session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"language": "id", "madhab_preference": "shafii"}'

# Send a message (use the session_id from above)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Jelaskan ayat kursi", "session_id": "<your-session-id>"}'
```

## Project Structure

```
├── src/
│   ├── api/            # FastAPI routes, auth, rate limiting
│   ├── core/           # Pipeline components (filter, RAG, LLM, validator, etc.)
│   ├── models/          # Pydantic schemas
│   ├── utils/           # Logging, caching
│   └── main.py          # App entry point
├── data/
│   └── scripts/         # Data ingestion scripts
├── tests/               # Unit, integration, property tests
├── docker-compose.yml
├── Dockerfile
└── islamic_ai_chat.html # Standalone browser chat client
```

## Known Limitations

This is a work-in-progress project. Notably:
- The knowledge base currently contains a small sample dataset — coverage across Fiqh, Fatwa, and Scholar Opinion sources is still limited and needs expansion for production use.
- Response confidence scores depend heavily on retrieval quality; sparse topics may return low-confidence or "no relevant information found" responses by design, to avoid hallucination.

## License

This project is for educational/portfolio purposes.
