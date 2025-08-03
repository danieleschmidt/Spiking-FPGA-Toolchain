# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core SNN compilation pipeline
- Network parser for YAML/JSON configurations
- HDL generation system with Verilog templates
- Resource estimation and mapping algorithms
- FPGA toolchain integration (Vivado/Quartus)
- Performance optimization passes
- CLI interface for network compilation
- Comprehensive test suite
- Development environment setup
- Documentation and community files

### Changed
- Replaced placeholder implementations with functional code
- Enhanced error handling across all modules
- Improved resource utilization algorithms

### Fixed
- Network parsing edge cases
- HDL template parameter binding
- Resource constraint validation

## [1.0.0] - TBD

### Added
- Initial release of Spiking-FPGA-Toolchain
- Support for Loihi-style spiking neural networks
- FPGA deployment for Xilinx Artix-7 and Intel Cyclone V
- Address-Event Representation (AER) spike routing
- Power optimization with DVFS
- Multi-platform HDL generation
- Comprehensive benchmarking suite

[Unreleased]: https://github.com/danieleschmidt/Spiking-FPGA-Toolchain/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/danieleschmidt/Spiking-FPGA-Toolchain/releases/tag/v1.0.0