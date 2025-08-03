# Changelog

All notable changes to the Spiking-FPGA-Toolchain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project foundation infrastructure and community files
- Core functionality implementation for SNN compilation
- Development environment setup and tooling
- Comprehensive documentation and architecture design

### Changed
- Migrated from placeholder implementation to functional core

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- Added security policy and vulnerability reporting process

## [0.1.0-dev] - 2025-08-03

### Added
- Initial project structure and Python package setup
- Basic CLI interface with placeholder commands
- FPGA target enumeration and resource definitions
- Comprehensive project documentation
- Architecture design and technical specifications
- Project charter and roadmap
- Apache 2.0 licensing

### Documentation
- README.md with project overview and quick start
- ARCHITECTURE.md with system design details
- PROJECT_CHARTER.md with scope and success criteria
- CONTRIBUTING.md with development guidelines
- docs/ROADMAP.md with release planning
- Architecture Decision Records (ADR) structure

### Development Infrastructure
- Python package configuration (pyproject.toml)
- Development dependencies and tooling setup
- Code quality tools (black, ruff, mypy)
- Testing framework configuration (pytest)
- Pre-commit hooks for code quality

### FPGA Platform Support
- Xilinx Artix-7 series target definitions
- Intel Cyclone V series target definitions
- Resource constraint specifications
- Vendor toolchain mapping

---

## Release Planning

### Version 1.0.0 - Foundation (Q2 2025)
- Complete HDL generation pipeline
- PyNN/Brian2 frontend parsers
- Vivado and Quartus integration
- Basic optimization passes
- Hardware validation on target platforms

### Version 1.1.0 - Optimization (Q3 2025)
- Advanced optimization pipeline
- Resource-aware placement algorithms
- Power optimization features
- Performance benchmarking suite

### Version 1.2.0 - Learning (Q4 2025)
- On-chip STDP learning implementation
- Dynamic weight updates
- Plasticity mechanism support
- Real-time adaptation capabilities

### Version 2.0.0 - Scale (Q1 2026)
- Multi-FPGA scaling support
- Distributed spike routing
- OpenCL host interface
- Advanced neuron models

---

## Notes

- This project follows semantic versioning
- Pre-1.0 versions may include breaking changes
- Hardware compatibility maintained across minor versions
- Academic citations updated with each release

For detailed technical changes, see individual commit messages and pull requests.