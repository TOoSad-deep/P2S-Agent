#!/bin/bash
# P2S-Agent Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[P2S-Agent]${NC} $1"
}

print_error() {
    echo -e "${RED}[P2S-Agent Error]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[P2S-Agent Warning]${NC} $1"
}

# Check if .env exists
check_env() {
    if [ ! -f "$BACKEND_DIR/.env" ]; then
        print_warning ".env file not found. Copying from .env.example..."
        cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
        print_warning "Please edit $BACKEND_DIR/.env with your API keys"
    fi
}

# Install backend dependencies
install_backend() {
    print_status "Installing backend dependencies..."
    cd "$BACKEND_DIR"
    pip install -r requirements.txt
}

# Install frontend dependencies
install_frontend() {
    print_status "Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm install
}

# Start backend
start_backend() {
    print_status "Starting backend on port 8001..."
    cd "$BACKEND_DIR"
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$SCRIPT_DIR/.backend.pid"
    print_status "Backend started (PID: $BACKEND_PID)"
}

# Start frontend
start_frontend() {
    print_status "Starting frontend..."
    cd "$FRONTEND_DIR"
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$SCRIPT_DIR/.frontend.pid"
    print_status "Frontend started (PID: $FRONTEND_PID)"
}

# Stop all services
stop_all() {
    print_status "Stopping services..."
    if [ -f "$SCRIPT_DIR/.backend.pid" ]; then
        kill $(cat "$SCRIPT_DIR/.backend.pid") 2>/dev/null || true
        rm "$SCRIPT_DIR/.backend.pid"
    fi
    if [ -f "$SCRIPT_DIR/.frontend.pid" ]; then
        kill $(cat "$SCRIPT_DIR/.frontend.pid") 2>/dev/null || true
        rm "$SCRIPT_DIR/.frontend.pid"
    fi
    print_status "All services stopped"
}

# Show status
show_status() {
    echo "=== P2S-Agent Status ==="
    if [ -f "$SCRIPT_DIR/.backend.pid" ] && kill -0 $(cat "$SCRIPT_DIR/.backend.pid") 2>/dev/null; then
        echo -e "Backend:  ${GREEN}Running${NC} (PID: $(cat "$SCRIPT_DIR/.backend.pid"))"
    else
        echo -e "Backend:  ${RED}Stopped${NC}"
    fi
    if [ -f "$SCRIPT_DIR/.frontend.pid" ] && kill -0 $(cat "$SCRIPT_DIR/.frontend.pid") 2>/dev/null; then
        echo -e "Frontend: ${GREEN}Running${NC} (PID: $(cat "$SCRIPT_DIR/.frontend.pid"))"
    else
        echo -e "Frontend: ${RED}Stopped${NC}"
    fi
}

# Main command handler
case "$1" in
    start)
        check_env
        start_backend
        start_frontend
        print_status "P2S-Agent started. Frontend: http://localhost:5174, Backend: http://localhost:8001"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    install)
        check_env
        install_backend
        install_frontend
        ;;
    backend)
        check_env
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    *)
        echo "Usage: $0 {start|stop|status|install|backend|frontend}"
        echo ""
        echo "Commands:"
        echo "  start    - Start both backend and frontend"
        echo "  stop     - Stop all services"
        echo "  status   - Show service status"
        echo "  install  - Install all dependencies"
        echo "  backend  - Start only backend"
        echo "  frontend - Start only frontend"
        exit 1
        ;;
esac
