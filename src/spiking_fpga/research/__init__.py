"""
Research module for novel neuromorphic algorithms and experimental features.

This module contains cutting-edge research implementations including:
- Bio-inspired adaptive spike encoding/decoding
- Hardware-optimized STDP with meta-plasticity  
- Quantum-inspired optimization algorithms
- Federated learning with differential privacy
- Real-time adaptive learning with reinforcement signals
"""

from .adaptive_encoding import AdaptiveSpikeCoder, MultiModalEncoder
from .meta_plasticity import MetaPlasticSTDP, BitstiftSTDP, HomeostasticRegulator
from .quantum_optimization import (
    QuantumOptimizationSuite,
    QuantumAnnealer,
    SuperpositionWeightOptimizer,
    QuantumGradientDescent,
    quantum_annealing_optimize,
    superposition_weight_optimize,
    quantum_gradient_optimize
)
from .federated_neuromorphic import (
    FederatedNeuromorphicServer,
    FederatedNeuromorphicClient,
    FederatedNeuromorphicOrchestrator,
    create_federated_config,
    run_federated_neuromorphic_learning
)
from .adaptive_realtime_learning import (
    AdaptiveRealTimeLearningSystem,
    AdaptationParameters,
    ReinforcementModulator,
    HomeostasticController,
    MetaLearningSystem,
    create_adaptive_learning_system
)

__all__ = [
    # Adaptive encoding
    "AdaptiveSpikeCoder",
    "MultiModalEncoder",
    
    # Meta-plasticity
    "MetaPlasticSTDP",
    "BitstiftSTDP", 
    "HomeostasticRegulator",
    
    # Quantum optimization
    "QuantumOptimizationSuite",
    "QuantumAnnealer",
    "SuperpositionWeightOptimizer",
    "QuantumGradientDescent",
    "quantum_annealing_optimize",
    "superposition_weight_optimize",
    "quantum_gradient_optimize",
    
    # Federated learning
    "FederatedNeuromorphicServer",
    "FederatedNeuromorphicClient", 
    "FederatedNeuromorphicOrchestrator",
    "create_federated_config",
    "run_federated_neuromorphic_learning",
    
    # Adaptive real-time learning
    "AdaptiveRealTimeLearningSystem",
    "AdaptationParameters",
    "ReinforcementModulator",
    "HomeostasticController",
    "MetaLearningSystem",
    "create_adaptive_learning_system",
]