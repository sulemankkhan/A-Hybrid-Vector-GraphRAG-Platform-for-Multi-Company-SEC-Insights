from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import jwt
from datetime import datetime, timedelta
import os
from pydantic import BaseModel
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import vector_search_rrf, graph_search
from src.query_router import route_query
from openai import OpenAI

app = FastAPI(title="Hybrid RAG API")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Auth Configuration
SECRET_KEY = os.getenv("JWT_SECRET", "super-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/api/main/token")
@app.post("/api/token")
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Mock user verification for demo deployment
    expected_username = os.getenv("API_USERNAME", "admin")
    expected_password = os.getenv("API_PASSWORD", "password123")
    
    if form_data.username != expected_username or form_data.password != expected_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    vector_context: list
    graph_context: list

@app.post("/api/main/chat", response_model=QueryResponse)
@app.post("/api/chat", response_model=QueryResponse)
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest, current_user: str = Depends(get_current_user)):
    try:
        # 1. NER Extraction
        entities = route_query(request.query)
        
        # 2. Vector Search (RRF via Pinecone)
        vector_results = vector_search_rrf(request.query, top_k=5)
        
        # 3. Graph Search (Neo4j)
        graph_results = graph_search(entities)
        
        # 4. Fuse Context
        fusion_parts = []
        if graph_results:
            fusion_parts.append("### Structured Knowledge Graph Facts ###\n" + "\n".join(graph_results))
        if vector_results:
            fusion_parts.append("### Relevant Document Excerpts ###\n" + "\n\n---\n\n".join(vector_results))
            
        hybrid_context = "\n\n".join(fusion_parts) if fusion_parts else "No relevant context found."
        
        # 5. LLM Generation
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        
        system_prompt = (
            "You are a senior hedge fund market researcher. Answer the following question based on the context.\n\n"
            f"Context:\n{hybrid_context}\n\n"
            f"Question:\n{request.query}"
        )
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
            
        formatted_vectors = [{"text": v, "source": "Pinecone RRF"} for v in vector_results]
        
        return QueryResponse(
            answer=answer,
            vector_context=formatted_vectors,
            graph_context=graph_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import Request
@app.api_route("/api/debug/{path_name:path}", methods=["GET", "POST"])
async def debug_route(request: Request, path_name: str):
    return {
        "request_url": str(request.url),
        "path_info": request.scope.get("path"),
        "raw_path": request.scope.get("raw_path").decode() if request.scope.get("raw_path") else None
    }
