# Makefile for Modern Text Summarization System

.PHONY: help install test run clean docker-build docker-run setup

# Default target
help:
	@echo "🧠 Modern Text Summarization System"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies and setup"
	@echo "  make test        - Run test suite"
	@echo "  make run         - Run Streamlit web interface"
	@echo "  make cli         - Run CLI demo"
	@echo "  make clean       - Clean temporary files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make setup       - Run automated setup"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
	@echo "✅ Installation complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest test_summarizer.py -v --cov=modern_summarizer --cov-report=html
	@echo "✅ Tests completed!"

# Run Streamlit app
run:
	@echo "🚀 Starting Streamlit app..."
	streamlit run app.py

# Run CLI demo
cli:
	@echo "💻 Running CLI demo..."
	python cli.py demo

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "✅ Cleanup complete!"

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t modern-summarizer .
	@echo "✅ Docker image built!"

# Run with Docker Compose
docker-run:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up --build
	@echo "✅ Docker Compose started!"

# Automated setup
setup:
	@echo "🔧 Running automated setup..."
	python setup.py
	@echo "✅ Setup complete!"

# Development setup
dev-setup: install
	@echo "🔧 Setting up development environment..."
	pip install pytest black flake8
	@echo "✅ Development setup complete!"

# Format code
format:
	@echo "🎨 Formatting code..."
	black modern_summarizer.py app.py test_summarizer.py cli.py
	@echo "✅ Code formatted!"

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 modern_summarizer.py app.py test_summarizer.py cli.py
	@echo "✅ Linting complete!"

# Full development workflow
dev: dev-setup format lint test
	@echo "🎉 Development workflow complete!"

# Production deployment
deploy: docker-build
	@echo "🚀 Production deployment ready!"
	@echo "Run 'make docker-run' to start the application"
