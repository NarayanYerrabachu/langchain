# Test Suite for RAG Document Q&A API

This directory contains comprehensive tests for the RAG Document Q&A API. The tests are designed to work both with and without external dependencies (database, OpenAI API) using mock implementations.

## Test Structure

```
tests/
├── __init__.py                     # Tests package initialization
├── conftest.py                     # Pytest configuration and fixtures
├── test_app.py                     # FastAPI application tests
├── test_config.py                  # Configuration tests
├── test_database.py                # Database and vector store tests
├── test_document_loaders.py        # Document processing tests
├── test_models.py                  # Pydantic models tests
├── test_qa_chain.py                # Question-answering chain tests
├── test_routes_collections.py      # Collections endpoints tests
├── test_routes_documents.py        # Document endpoints tests
├── test_routes_health.py           # Health endpoints tests
└── README.md                       # This file
```

## Running Tests

### Option 1: Using the Test Runner Script

The easiest way to run tests is using the provided test runner script:

```bash
# Run all tests
python run_tests.py

# Run tests for a specific module
python run_tests.py --module health
python run_tests.py --module documents
python run_tests.py --module database

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# Stop on first failure
python run_tests.py --failfast

# List available test modules
python run_tests.py --list-modules
```

### Option 2: Using unittest directly

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_health

# Run specific test class
python -m unittest tests.test_health.TestHealthRoutes

# Run specific test method
python -m unittest tests.test_health.TestHealthRoutes.test_root_endpoint

# Run with verbose output
python -m unittest discover tests -v
```

### Option 3: Using pytest

If you have pytest installed:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_health.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run with specific markers
pytest -m unit
pytest -m integration

# Run with verbose output
pytest -v
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Use mocks to avoid external dependencies
- Fast execution, suitable for development

### Integration Tests  
- Test interactions between components
- May use real databases or APIs in CI/CD
- Slower execution but more comprehensive

## Test Configuration

### Environment Variables

Tests use these environment variables (automatically set by test fixtures):

```bash
OPENAI_API_KEY=dummy-key-for-testing
DATABASE_URL=postgresql://test:test@localhost:5432/test_db
COLLECTION_NAME=test_collection
ENABLE_DATABASE=false  # Uses mock database by default
MAX_FILE_SIZE=10       # 10MB for tests
LOG_LEVEL=WARNING      # Reduce log noise
```

### Fixtures

Common test fixtures are defined in `conftest.py`:

- `setup_test_environment`: Sets up test environment variables
- `mock_database`: Provides mock database components
- `mock_openai`: Provides mock OpenAI API
- `sample_text_content`: Sample text for testing
- `sample_html_content`: Sample HTML for testing
- `sample_metadata`: Sample metadata for testing

## Test Coverage

### Current Test Coverage

- **Health Endpoints**: ✅ Complete coverage
- **Document Endpoints**: ✅ Complete coverage including file uploads
- **Collection Endpoints**: ✅ Complete coverage
- **Document Processing**: ✅ Complete coverage for all formats
- **Database Layer**: ✅ Complete coverage including mocks
- **QA Chain**: ✅ Complete coverage including mocks
- **Models**: ✅ Complete coverage including validation
- **Configuration**: ✅ Complete coverage
- **Application Setup**: ✅ Complete coverage

### Coverage Reporting

Generate coverage reports using:

```bash
# Generate coverage report
python run_tests.py --coverage

# Or with pytest
pytest --cov=. --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html
```

## Mock Strategy

Tests use comprehensive mocking to avoid external dependencies:

### Database Mocking
- `MockVectorStore`: Simulates PGVector functionality
- `MockRetriever`: Simulates document retrieval
- `MockEmbeddings`: Simulates OpenAI embeddings

### API Mocking
- `MockLLM`: Simulates OpenAI language model
- `MockQAChain`: Simulates question-answering chain
- HTTP requests mocked for web scraping tests

### File System Mocking
- Temporary files for document processing tests
- Mock file uploads for API endpoint tests

## Test Data

### Sample Documents
Tests include sample content for various document types:
- Plain text
- HTML with various elements
- Simulated PDF/DOCX content
- CSV and Excel data
- Markdown content

### Metadata Samples
Common metadata patterns used across tests:
- Document identification
- Source tracking
- Author information
- Timestamps
- Custom fields

## Writing New Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<description>`

### Example Test Structure

```python
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app

class TestMyModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_something_success(self):
        """Test successful operation"""
        # Arrange
        # Act
        # Assert
        pass
    
    def test_something_error(self):
        """Test error handling"""
        # Test error conditions
        pass
    
    @patch('module.dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependencies"""
        # Configure mocks
        # Run test
        # Verify interactions
        pass
```

### Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Test both success and error cases**
4. **Use appropriate mocks** to isolate units under test
5. **Keep tests independent** - each test should be able to run alone
6. **Use fixtures** for common test data and setup
7. **Test edge cases** and boundary conditions
8. **Keep tests fast** - use mocks for expensive operations

## Continuous Integration

### GitHub Actions Integration

Tests are designed to work in CI/CD environments:

```yaml
# Example .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python run_tests.py --coverage
```

### Docker Testing

Run tests in Docker environment:

```bash
# Build test image
docker build -t rag-api-test .

# Run tests
docker run --rm rag-api-test python run_tests.py
```

## Debugging Tests

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Mock Issues**: Verify mock paths match actual import paths
3. **Environment Variables**: Check test environment setup
4. **Async Issues**: Use appropriate test clients for FastAPI

### Debug Commands

```bash
# Run single test with detailed output
python -m unittest tests.test_health.TestHealthRoutes.test_root_endpoint -v

# Run with Python debugger
python -m pdb -m unittest tests.test_health

# Run with coverage and keep temporary files
python run_tests.py --coverage --verbose
```

## Performance Testing

While not included in the current test suite, consider adding:

- Load testing for API endpoints
- Memory usage testing for document processing
- Database query performance testing
- File upload size limit testing

## Security Testing

Current security test coverage:

- Input validation testing
- File upload security (size limits, type validation)
- SQL injection prevention (through ORM)
- XSS prevention (through proper encoding)

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Ensure good coverage** (aim for >90%)
3. **Test error conditions** as well as success cases
4. **Update this documentation** if adding new test categories
5. **Use existing patterns** for consistency

## Troubleshooting

### Common Test Failures

1. **Database connection errors**: Make sure `ENABLE_DATABASE=false` for tests
2. **OpenAI API errors**: Use mock implementations in tests
3. **File permission errors**: Ensure test user has write permissions for temp files
4. **Import path errors**: Check PYTHONPATH includes project root

### Getting Help

- Check test output for specific error messages
- Review mock configurations in `conftest.py`
- Look at similar existing tests for patterns
- Verify environment variable setup