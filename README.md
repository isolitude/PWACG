# LLMPWA - LLM-Powered Partial Wave Analysis Code Generator

LLMPWA is an intelligent code generation tool that leverages Large Language Models (LLM) to automate the creation of optimized Python code for partial wave analysis:

- **Resonance Configuration**: Define resonance states in a structured TOML configuration file
- **Automatic Code Generation**: Uses LLM to intelligently generate optimized Python code
- **Smart Caching**: Efficiently caches generated code to avoid redundant API calls
- **Template-Based Output**: Produces consistent, production-ready code

## Installation

### Prerequisites
- Python 3.8 or higher
- API Key for LLM service (EasyTrans)

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or manually install the dependencies:

```bash
pip install -U \
    requests \
    toml \
    jinja2
```

### Configure API Access

Set the `EASYTRANS_API_KEY` environment variable for LLM access:

```bash
export EASYTRANS_API_KEY="your_api_key_here"
```

You can also pass it as a command-line argument when running the code generator.

## Usage

### Basic Usage

The LLMPWA code generator automates the creation of partial wave analysis code based on resonance configurations.

#### 1. Configure Resonances

Edit the resonance configuration file `agent/resonances_config.toml` to define your resonance states:

```toml
[resonances.phif0_980.propagators.A_propagator]
propagator_type = "BW"
mass = { value = 1.02, fixed = true}
width = { value = 0.004, fixed = true}

[resonances.phif0_980.Amplitude]
AMP = "phif0_kk"
const1 = { value = 0.1, fixed = true }
```

#### 2. Generate Code

Run the LLM code generator:

```bash
cd agent
python llm_code_generator.py --config resonances_config.toml --api_key your_api_key
```

Or with environment variable:

```bash
export EASYTRANS_API_KEY="your_api_key"
python llm_code_generator.py --config resonances_config.toml
```

#### 3. Output

The generator will create optimized Python code for:
- Resonance propagators
- Amplitude calculations
- Data loading functions
- Likelihood computations
- Parameter fitting routines

### Advanced Options

#### Cache Management

The generator includes intelligent caching to avoid redundant API calls. Generated code is cached in the `agent/cache/` directory.

#### Code Compression

Generated code is automatically optimized and compressed for better performance using the built-in code compressor.

### File Structure

```
agent/
├── llm_code_generator.py    # Main code generation script
├── resonances_config.toml   # Resonance configuration file
├── easytrans_client.py      # LLM API client
├── code_compressor.py       # Code optimization utilities
├── common_template.py       # Code templates
└── cache/                   # Generated code cache
```

## Key Features

- **LLM-Powered Code Generation**: Leverages state-of-the-art language models to generate production-ready code
- **Configuration-Driven**: Define physics models in simple TOML configuration files
- **Intelligent Caching**: Avoids redundant API calls with smart caching mechanisms
- **Code Optimization**: Automatically compresses and optimizes generated code for better performance
- **Template-Based**: Uses proven templates to ensure consistent, high-quality output

## Contributing

We welcome contributions! Please feel free to submit Pull Requests or open Issues for bug reports and feature requests.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- GitHub Issues: https://github.com/caihao/PWACG/issues
