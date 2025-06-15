# Connect Chain
An enterprise-grade, Generative AI framework and utilities for AI-enabled applications. `connectchain` is designed to bridge the gap between enterprise needs and what is available in existing frameworks.

Primary objectives include:
* A login utility for API-based LLM services that integrates with Enterprise Auth Service (EAS). Simplified generation of the JWT token, which is then passed to the modeling service provider.
* Support for configuration-based outbound proxy support at the model level to allow integration with enterprise-level security requirements.
* A set of tools to provide greater control over generated prompts. This is done by adding hooks to the existing langchain packages.

## Installation

### Using pip
```bash 
pip install connectchain
```

### Using uv (recommended for development)
```bash
# Install uv first (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install connectchain
uv pip install connectchain
```

## Usage

Connectchain works with a combination of environmental variables and a configuration `.yml` file. Environmental variables are defined in the `config.yml` and their corresponding values set in the `.env` file. The path to the `config.yml` is defined as an variable in the `.env` file. *You MUST create both a `config.yml` and `.env` file to use the module.* The [example config file](./connectchain/config/example.config.yml) can be found at [`./connectchain/config/example.config.yml`](./connectchain/config/example.config.yml). See the [example env file](example.env) for more details. You can copy and rename both files; replacing the required values with your ids and secrets and adding additional supported options as needed.

### `connectchain.lcel`: For the simplest Use Cases
[LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) supports adding a model() method. Now one can execute a chain by following the LCEL syntax with a minor tweak: 
* when you add the () to the model, it gets instantiated on the fly:

```python
from connectchain.lcel import model
...
prompt = PromptTemplate(
    input_variables=["music_genre"],
    template="Tell me about {music_genre} music."
)

# using langchain directly this would be:
# chain = prompt | model
chain = prompt | model()

out = chain.invoke({"music_genre": "classical"})
print(out)
```
You can have multiple model configurations defined in the `config.yml`. These are accessed via connectchain's LCEL support by passing the model configuration index (the key under which the model's configuration is defined in the `config.yml`) to the `model` method of connectchain. The following is an example using `2` as the model configuraiton index assuming it is defined in the `config.yml > models` section:

```python
chain = prompt | model('2')
```

_Optionally_ `eas`, `proxy` and `cert` sections of the `config.yml` can be overriden by model definitions. To do this, simply define those sections in a model config (again in the `config.yml`) and re-define any values you want to override. For example, if you want to override all three options for a model, you can define it as follows:

```yaml
models:
  foo:
    eas:
      id_key: ... # Env key for id
      secret_key: ... # Env key for secret
      scope: [
        # ...
      ]
    cert:
        cert_path: /path/to/cert
        cert_name: model_specific_cert.crt
        cert_size: 2048
    proxy:
        host: proxy.foo.com
        port: 8080
    # ... continue the model configuration
```

Add logging or auditing to the chain:
```python
from connectchain.lcel import Logger
... 
class PrintLogger(Logger):
    def print(self, payload):
        print(payload)
...
logger = PrintLogger()
chain = prompt | logger.log() | model() | logger.log()
```

There is a portable solution for the "regular" prompt template-based requests. It is portable, i.e. no need to directly import a model provider package (e.g. `openai`). Additionally, prompts can be validated before being sent to the LLM.

```python
from connectchain.orchestrators import PortableOrchestrator

orchestrator = PortableOrchestrator.from_prompt_template(
    prompt_template="Tell me about the climate in {area_of_interest}.", input_variables=["area_of_interest"])
output = orchestrator.run('Peru')
```

Again, you can have multiple models defined in the `config.yml`. For example, a second model could be defined as '2' in the config which configures a different model, a different API and even a different EAS and would look like this:

```python
orchestrator = PortableOrchestrator.from_prompt_template(
    prompt_template="Tell me about the climate in {area_of_interest}.", input_variables=["area_of_interest"], index='2')
```

### `connectchain.utils`: For direct use with a model provider (e.g. OpenAI)

```python  
from connectchain.utils import get_token_from_env

...
auth_token = get_token_from_env()
...
openai.api_key = auth_token
```

Same token can be used in lieu of the OPENAI_API_KEY:

```python
my_api_base = "<insert_your_api_base_here>"
llm = AzureOpenAI(
        engine='gpt-35',
        model_name='gpt-35-turbo',
        openai_api_key=auth_token,
        openai_api_base=my_api_base)

chain = LLMChain(llm=llm, prompt=prompt)
```

### `connectchain.prompts`: A package to provide greater control over generated prompts before they are passed to the LLM by providing an entrypoint for sanitizer implementations.

```python
from connectchain.prompts import ValidPromptTemplate
from connectchain.utils.exceptions import OperationNotPermittedException

def my_sanitizer(query: str) -> str:
    """IMPORTANT: This is a simplified example designed to showcase concepts and should not used
    as a reference for production code. The features are experimental and may not be suitable for
    use in sensitive environments or without additional safeguards and testing.

    Any use of this code is at your own risk."""
    pattern = r'BADWORD'

    if re.search(pattern, query):
        print("BADWORD found!")
        raise OperationNotPermittedException("Illegal execution detected: {}".format(query))
    else:
        return query
...
prompt_template = "Tell me about {food_type} production.}"
prompt = ValidPromptTemplate(
    output_sanitizer=my_sanitizer,
    input_variables=["food_type"],
    template=prompt_template
)


chain = LLMChain(llm=llm, prompt=prompt)
# the following will throw an exception
output = chain.run('BADWORD')
print(output)

```
### `connectchain.chains`: An extension of the langchain chains.
We add hooks to improve control over code that is executed by providing an entrypoint for sanitizer implementations.

```python   
from connectchain.chains import ValidLLMChain

def my_sanitizer(query: str) -> str:
    """IMPORTANT: This is a simplified example designed to showcase concepts and should not used
    as a reference for production code. The features are experimental and may not be suitable for
    use in sensitive environments or without additional safeguards and testing.

    Any use of this code is at your own risk."""
    # define your own logic here.
    # for example, can call an API to verify the content of the code
    pass

chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=my_sanitizer)

output = chain.run('drought resistant wheat')
print(output)

try:
    output = chain.run('BADWORD')
except OperationNotPermittedException as e:
    print(e)

```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management and packaging.

### Prerequisites

Install `uv`:
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/americanexpress/connectchain.git
cd connectchain

# Install dependencies (including dev dependencies)
uv sync --dev

# Activate the virtual environment (optional, uv run handles this automatically)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Running Tests

The project uses [pytest](https://pytest.org/) for testing. All tests are located in `connectchain/test/`.

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage report
uv run pytest --cov=connectchain

# Run tests with coverage and show missing lines
uv run pytest --cov=connectchain --cov-report=term-missing

# Run specific test file
uv run pytest connectchain/test/test_model.py

# Run tests matching a pattern
uv run pytest -k "test_model"

# Run tests and stop on first failure
uv run pytest -x
```

### Test Coverage

Current test coverage: 36 tests covering the core functionality including:
- Configuration management
- Model initialization and proxy support
- Token utilities and authentication
- Prompt validation and sanitization
- Chain execution and logging
- Orchestrator functionality

### Project Structure

```
connectchain/
├── chains/          # Extended LangChain chains with validation
├── examples/        # Usage examples and demonstrations  
├── lcel/           # LangChain Expression Language extensions
├── orchestrators/   # High-level orchestration utilities
├── prompts/        # Enhanced prompt templates
├── test/           # Test suite
├── tools/          # Extended LangChain tools
└── utils/          # Core utilities (auth, config, proxy)
```

### Code Quality & Linting

The project uses multiple linting tools to maintain code quality. All tools are configured in `pyproject.toml` and can be run via the Makefile:

```bash
# Run all checks (recommended)
make check                                  # Run linting + tests
make lint                                   # Run all linting checks (includes mypy)
make lint-quick                             # Run linting checks (skip mypy type checking)
make test                                   # Run all tests

# Individual linting tools
make lint-black                             # Code formatting check
make lint-isort                             # Import sorting check  
make lint-pylint                            # Static code analysis
make lint-mypy                              # Type checking

# Auto-fix formatting issues
make format                                 # Auto-format code and sort imports

# Utility commands
make clean                                  # Clean up cache files
make help                                   # Show all available targets
```

You can also run the tools directly if needed:
```bash
uv run black --check connectchain/          # Code formatting check
uv run isort --check-only connectchain/     # Import sorting check  
uv run pylint connectchain/                 # Static code analysis
uv run mypy connectchain/                   # Type checking
```

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and add tests for new functionality
3. Run linting checks: `make lint`
4. Run tests to ensure everything passes: `make test`
5. Update documentation if needed
6. Submit a pull request
## Contributing

We welcome Your interest in the American Express Open Source Community on GitHub. Any Contributor to
any Open Source Project managed by the American Express Open Source Community must accept and sign
an Agreement indicating agreement to the terms below. Except for the rights granted in this 
Agreement to American Express and to recipients of software distributed by American Express, You
reserve all right, title, and interest, if any, in and to Your Contributions. Please
[fill out the Agreement](https://cla-assistant.io/americanexpress/connectchain).

## License

Any contributions made under this project will be governed by the
[Apache License 2.0](./LICENSE.txt).

## Code of Conduct

This project adheres to the [American Express Community Guidelines](./CODE_OF_CONDUCT.md). By
participating, you are expected to honor these guidelines.
