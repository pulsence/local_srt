# Tests for Local SRT

This directory contains comprehensive test coverage for the `local_srt` package.

## Test Structure

The test suite is organized to mirror the source code structure, with each module having its own test file:

- `test_models.py` - Tests for data models (ResolvedConfig, SubtitleBlock, WordItem)
- `test_text_processing.py` - Tests for text normalization, wrapping, splitting, and timing
- `test_config.py` - Tests for configuration loading and preset management
- `test_logging_utils.py` - Tests for logging and progress display utilities
- `test_system.py` - Tests for system utilities and dependency checks
- `test_batch.py` - Tests for batch processing and file discovery
- `test_output_writers.py` - Tests for SRT/VTT/ASS/TXT/JSON output formats
- `test_subtitle_generation.py` - Tests for subtitle chunking and timing logic
- `test_audio.py` - Tests for audio processing and silence detection

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

This installs the package in editable mode along with test dependencies:
- pytest
- pytest-cov
- pytest-mock

### Run All Tests

```bash
pytest
```

This automatically generates:
- **Coverage report** in `htmlcov/index.html` - Shows which lines of code are covered by tests
- **Test results report** in `report.html` - Shows pass/fail status of all tests

### View Reports

```bash
# Open coverage report
start htmlcov/index.html

# Open test results report
start report.html
```

### Run Specific Test File

```bash
pytest tests/test_models.py
```

### Run Specific Test Class or Function

```bash
pytest tests/test_models.py::TestResolvedConfig
pytest tests/test_models.py::TestResolvedConfig::test_default_values
```

### Run with Verbose Output

```bash
pytest -v
```

### Run Tests in Parallel (faster)

```bash
pip install pytest-xdist
pytest -n auto
```

## Test Coverage

The test suite aims for comprehensive coverage of:

1. **Unit Tests**: Individual functions and methods
2. **Integration Points**: Module interactions
3. **Edge Cases**: Empty inputs, boundary conditions, error handling
4. **Format Compliance**: SRT/VTT/ASS format specifications
5. **Timing Logic**: Subtitle duration, gaps, and alignment

### Coverage Goals

- Core logic modules: >90% coverage
- Utility modules: >80% coverage
- CLI and integration code: >70% coverage

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` or `Test<FunctionName>`
- Test methods: `test_<specific_behavior>`

### Example Test Structure

```python
"""Tests for the example module."""
import pytest
from local_srt.example import function_to_test


class TestFunctionToTest:
    """Tests for function_to_test."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = function_to_test("input")
        assert result == "expected_output"

    def test_edge_case(self):
        """Test edge case handling."""
        result = function_to_test("")
        assert result == ""

    def test_error_handling(self):
        """Test that errors are handled properly."""
        with pytest.raises(ValueError):
            function_to_test(None)
```

### Using Fixtures

```python
@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return ResolvedConfig(max_chars=50, max_lines=2)


def test_with_fixture(sample_config):
    """Test using the fixture."""
    assert sample_config.max_chars == 50
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('local_srt.system.run_cmd_text')
def test_with_mock(mock_run_cmd):
    """Test with mocked subprocess call."""
    mock_run_cmd.return_value = (0, "output", "")
    # Test code that calls run_cmd_text
```

## Continuous Integration

Tests are designed to run in CI environments without requiring:
- ffmpeg installed
- Whisper models downloaded
- GPU/CUDA availability

External dependencies are mocked in tests to ensure fast, reliable execution.

## Test Philosophy

1. **Fast**: Tests should run quickly (< 5 seconds total)
2. **Isolated**: Each test is independent and can run in any order
3. **Deterministic**: Tests always produce the same result
4. **Readable**: Test names and structure clearly communicate intent
5. **Maintainable**: Tests are easy to update as code evolves

## Troubleshooting

### Import Errors

If you get import errors, make sure the package is installed:

```bash
pip install -e .
```

### Missing Test Dependencies

```bash
pip install -e ".[test]"
```

### pytest Command Not Found

```bash
python -m pytest
```

## Contributing

When adding new features to `local_srt`:

1. Write tests first (TDD) or alongside the implementation
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov=local_srt`
4. Add new test files for new modules
5. Update this README if adding new test patterns
