# Contributing to Spiking-FPGA-Toolchain

Thank you for your interest in contributing to the Spiking-FPGA-Toolchain! This project aims to democratize neuromorphic computing research by providing an open-source toolchain for deploying spiking neural networks on affordable FPGA hardware.

## Project Status

**🚧 Early Development Phase**: This project is currently in the documentation and planning phase. The implementation will begin in Q2 2025 according to our roadmap. Contributions are welcome across documentation, architecture design, and preparation for the upcoming implementation phases.

## Ways to Contribute

### 📚 Documentation & Research
- Improve technical documentation and tutorials
- Add references to relevant research papers
- Create benchmark network definitions
- Develop educational materials for neuromorphic computing

### 🏗️ Architecture & Design
- Review and improve system architecture documents
- Contribute to ADR (Architecture Decision Records)
- Design optimization algorithms for SNN-to-HDL compilation
- Propose new FPGA platform support

### 🧪 Research & Validation
- Develop benchmark datasets and networks
- Create reference implementations for validation
- Contribute performance analysis and modeling
- Share domain expertise in neuromorphic computing

### 💻 Implementation (Starting Q2 2025)
- Core compiler implementation (Python)
- HDL template development (Verilog/VHDL)
- FPGA toolchain integration
- Testing and validation frameworks

## Getting Started

### Prerequisites
- Python 3.10+ for development
- Git for version control
- FPGA development boards (optional, for hardware testing)
- Xilinx Vivado or Intel Quartus (for synthesis validation)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/Spiking-FPGA-Toolchain.git
   cd Spiking-FPGA-Toolchain
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Setup**
   ```bash
   spiking-fpga --help
   pytest tests/
   ```

## Contribution Guidelines

### Code Standards
- **Python**: Follow PEP 8, use type hints, minimum Python 3.10
- **HDL**: Adhere to industry coding standards, prefer Verilog for portability
- **Documentation**: Write clear docstrings, update README for major changes
- **Testing**: Include tests for new functionality, maintain >80% coverage

### Commit Message Format
Use conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
Scopes: `compiler`, `hdl`, `runtime`, `docs`, `cli`

Examples:
- `feat(compiler): add PyNN network parser`
- `docs(architecture): update memory hierarchy design`
- `fix(hdl): correct LIF neuron timing constraints`

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/
   black --check src/
   ruff check src/
   mypy src/
   ```

4. **Submit Pull Request**
   - Use descriptive title and description
   - Reference related issues
   - Request review from maintainers
   - Ensure CI passes

### Review Criteria
- Code quality and style compliance
- Test coverage and documentation
- Alignment with project architecture
- Performance considerations for FPGA constraints

## Research Contribution Guidelines

### Academic Contributions
- **Citations**: Use proper academic citation format in documentation
- **Reproducibility**: Provide detailed methodology for experiments
- **Data Sharing**: Follow open science principles where possible
- **Ethics**: Ensure compliance with research ethics guidelines

### Benchmark Networks
When contributing benchmark networks:
- Provide network description in standardized format
- Include expected performance metrics
- Document biological or computational motivation
- Ensure compatibility with target FPGA platforms

### Algorithm Contributions
For optimization algorithms and compilation techniques:
- Provide theoretical foundation and references
- Include complexity analysis
- Demonstrate effectiveness on target platforms
- Consider hardware resource constraints

## Community Guidelines

### Code of Conduct
We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, inclusive, and constructive in all interactions.

### Communication Channels
- **GitHub Discussions**: Technical discussions and Q&A
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code review and implementation discussions
- **Project Board**: Track development progress

### Recognition
Contributors will be recognized in:
- README.md contributor list
- Academic publications using the toolchain
- Release notes for significant contributions
- Conference presentations and workshops

## Technical Areas Needing Help

### High Priority
- [ ] PyNN/Brian2 network parsers
- [ ] Resource-aware placement algorithms
- [ ] HDL generation templates
- [ ] Vivado/Quartus integration scripts

### Medium Priority
- [ ] Optimization pass framework
- [ ] Power analysis tools
- [ ] Multi-FPGA scaling architecture
- [ ] Hardware-in-the-loop testing

### Research Opportunities
- [ ] Novel SNN compression techniques
- [ ] Hardware-aware learning algorithms
- [ ] Cross-platform performance modeling
- [ ] Neuromorphic benchmarking standards

## Questions?

- Check existing [GitHub Discussions](../../discussions)
- Review [Architecture Documentation](ARCHITECTURE.md)
- Read the [Project Charter](PROJECT_CHARTER.md)
- Contact maintainers through GitHub issues

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

*We appreciate your interest in advancing open-source neuromorphic computing tools!*