"""
FastAPI application for MLSearch web interface.
"""
import logging
import os
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from .routes import search, papers
from .models.responses import SearchProgress

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MLSearch",
    description="AI-powered academic paper search using multi-agent orchestration",
    version="1.0.0"
)

# Get the web directory path
web_dir = Path(__file__).parent

# Mount static files
app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

# Set up templates
templates = Jinja2Templates(directory=web_dir / "templates")

# Include routers
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main search interface."""
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/results/{search_id}", response_class=HTMLResponse)
async def results(request: Request, search_id: str):
    """Serve the results page for a specific search."""
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "search_id": search_id
    })

@app.websocket("/ws/{search_id}")
async def websocket_endpoint(websocket: WebSocket, search_id: str):
    """WebSocket endpoint for real-time search progress updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages (keep connection alive)
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message for search {search_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for search {search_id}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mlsearch-web"}

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Run the application
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "mlsearch.web.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )