# HYPIR COG Project Makefile

.PHONY: build test push clean help

# Configuration
MODEL_NAME = hypir
REGISTRY = r8.im/youruser/hypir
TEST_IMAGE = test_input.jpg

build:
	@echo "🔨 Building HYPIR COG image..."
	cog build -t $(MODEL_NAME)

test: build
	@echo "🧪 Testing HYPIR model..."
	@if [ ! -f $(TEST_IMAGE) ]; then \
		echo "❌ Test image not found: $(TEST_IMAGE)"; \
		echo "💡 Add a test image or update TEST_IMAGE in Makefile"; \
		exit 1; \
	fi
	cog predict -i image=@$(TEST_IMAGE) -i prompt="high quality photo" -i upscale_factor=2.0

push: build
	@echo "🚀 Pushing to Replicate..."
	cog login
	cog push $(REGISTRY)

clean:
	@echo "🧹 Cleaning up..."
	docker rmi $(MODEL_NAME) 2>/dev/null || true
	rm -rf HYPIR/
	rm -f *.pth *.png *.jpg
	rm -rf cache/

setup:
	@echo "⚙️ Setting up development environment..."
	pip install cog
	@echo "✅ Development environment ready"

help:
	@echo "HYPIR COG Commands:"
	@echo ""
	@echo "  build    - Build the COG Docker image"
	@echo "  test     - Test the model locally (requires $(TEST_IMAGE))"
	@echo "  push     - Push to Replicate (update REGISTRY first)"
	@echo "  clean    - Remove built images and downloaded files"
	@echo "  setup    - Install COG for development"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL_NAME: $(MODEL_NAME)"
	@echo "  REGISTRY:   $(REGISTRY)"
	@echo "  TEST_IMAGE: $(TEST_IMAGE)"
