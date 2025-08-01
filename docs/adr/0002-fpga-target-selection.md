# ADR-0002: FPGA Target Platform Selection

## Status
Accepted

## Context
The toolchain needs to support multiple FPGA platforms to maximize accessibility and performance across different use cases. We need to decide which FPGA families to prioritize for initial development and long-term support.

## Decision
We will primarily target Xilinx Artix-7 and Intel Cyclone V FPGA families as our initial platforms, with extensible architecture for future platform support.

### Primary Targets:
- **Xilinx Artix-7 Series** (35T, 100T variants)
- **Intel Cyclone V Series** (GX, GT variants)

### Secondary Targets (Future):
- Xilinx Zynq-7000 (ARM+FPGA integration)
- Intel Arria 10 (Higher performance tier)

## Alternatives Considered

### Option 1: Xilinx-Only
- **Pros**: Simplified toolchain, mature Vivado ecosystem
- **Cons**: Vendor lock-in, limited accessibility due to pricing

### Option 2: Intel-Only  
- **Pros**: Competitive pricing, good open-source tool support
- **Cons**: Smaller ecosystem, less adoption in research

### Option 3: Multi-Vendor Support
- **Pros**: Maximum flexibility, competitive benchmarking
- **Cons**: Complex abstraction layer, higher maintenance burden

## Consequences

### Positive:
- Broad platform coverage balances cost and performance
- Competitive options for users with different budgets
- Cross-platform benchmarking validates design decisions
- Reduced vendor lock-in risk

### Negative:
- Increased complexity in HDL generation
- Need for platform-specific optimization passes
- Higher testing and validation overhead
- More complex build system requirements

## Implementation Notes

### Hardware Abstraction Layer
```python
class FPGATarget(Enum):
    ARTIX7_35T = "artix7_35t"
    ARTIX7_100T = "artix7_100t" 
    CYCLONE_V_GX = "cyclone5_gx"
    CYCLONE_V_GT = "cyclone5_gt"
```

### Platform-Specific Features
- Resource constraint templates per platform
- Vendor-specific HDL dialect generation
- Platform-optimized memory controllers
- Timing constraint templates

### Build System Integration
- CMake platform selection: `-DTARGET_FPGA=ARTIX7`
- Automated resource utilization reporting
- Platform-specific synthesis script generation

## Related Decisions
- [ADR-0003: HDL Generation Strategy](0003-hdl-generation-strategy.md)
- [ADR-0004: Memory Architecture](0004-memory-architecture.md)