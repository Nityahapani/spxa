"""
Shared Computational Graph Engine for Stochastic Modeling, Analytics, Autodiff & Risk.
"""

from __future__ import annotations

from spxa.graph.adapters import (
    CalculusGeneratorAdapter,
    CopulaAdapter,
    DifferentiableSDEAdapter,
    GraphAdapterRegistry,
    RiskInstrumentAdapter,
    UniversalProcessNode,
    adapt_portfolio,
    as_graph_node,
    install_graph_extensions,
    to_graph,
)
from spxa.graph.autodiff.dual import DualNumber
from spxa.graph.autodiff.hessian import compute_graph_hessian
from spxa.graph.autodiff.smoothers import SmoothIndicatorNode, SoftplusNode
from spxa.graph.autodiff.tape import GradientTape, value_and_grad
from spxa.graph.backends import (
    AcceleratedBackend,
    AnalyticalBackend,
    Backend,
    ExecutionContext,
    ExecutionRouter,
    NumericalBackend,
    SimulationBackend,
    get_backend,
)
from spxa.graph.boundary import (
    FellerBoundaryClassifierNode,
    ScaleDensityNode,
    SpeedDensityNode,
)
from spxa.graph.calibration.constraints import FellerConstraintNode
from spxa.graph.calibration.loss_nodes import ArbitragePenaltyNode, MSELossNode
from spxa.graph.calibration.optimizer import GraphCalibrator
from spxa.graph.char_nodes import (
    AffineCharacteristicNode,
    CharacteristicExponentNode,
    CharacteristicFunctionNode,
    CumulantGeneratingFunctionNode,
    GeneratorApplicationNode,
    ProcessMomentNode,
)
from spxa.graph.compiler import EvaluationCache, GraphCompiler
from spxa.graph.duality import (
    DensityFacet,
    FourierFacet,
    GeneratorFacet,
    ProcessDualityCore,
    SDEFacet,
    compile_levy_khintchine,
    triplet_to_fourier_facet,
)
from spxa.graph.generator import (
    InfinitesimalGeneratorNode,
    MartingaleVerificationNode,
)
from spxa.graph.graph import ComputationalGraph
from spxa.graph.introspection import GraphProfiler, graph_summary, to_dot
from spxa.graph.ito import ItoTransformNode
from spxa.graph.malliavin import MalliavinBismutDeltaNode
from spxa.graph.math_ops import (
    AbsNode,
    AddNode,
    CholeskyNode,
    CosNode,
    CoshNode,
    DivNode,
    ErfNode,
    ExpNode,
    GammaNode,
    LogNode,
    MatMulNode,
    MaximumNode,
    MeanNode,
    MinimumNode,
    MulNode,
    NegNode,
    NormalCDFNode,
    NormalPDFNode,
    PowNode,
    SinNode,
    SinhNode,
    SqrtNode,
    SubNode,
    SumNode,
    TanhNode,
    TransposeNode,
)
from spxa.graph.measure_change import EsscherTransformNode, GirsanovTransformNode
from spxa.graph.morph import MorphNode, QuadraticVarianceMorphNode
from spxa.graph.multiasset import CorrelatedDiffusionNode, MultiAssetBasketNode
from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, OutputNode, ParameterNode
from spxa.graph.pricing.greeks import GreeksResult, compute_graph_greeks, graph_greeks
from spxa.graph.pricing.payoff_nodes import (
    AsianPayoffNode,
    DigitalPayoffNode,
    EuropeanCallPayoffNode,
    EuropeanPutPayoffNode,
)
from spxa.graph.pricing.pricer_graph import (
    BlackScholesPriceNode,
    build_black_scholes_pricing_graph,
    monte_carlo_graph_price,
)
from spxa.graph.risk.risk_graph import PortfolioRiskGraph
from spxa.graph.risk.scenario_graph import ScenarioShockNode, apply_macro_scenario
from spxa.graph.risk.xva_graph import GraphCVANode, NettingExposureNode
from spxa.graph.semimartingale import (
    QuadraticCovariationNode,
    SemimartingaleComponents,
    SemimartingaleWire,
    StochasticIntegralNode,
    StochasticProductNode,
)
from spxa.graph.geom import (
    FisherInformationMetricNode,
    NaturalGradientCalibratorNode,
)
from spxa.graph.rough import (
    FractionalRiccatiNode,
    RoughBergomiNode,
    RoughKernelNode,
)
from spxa.graph.serialization import GraphSerializer
from spxa.graph.signature import (
    ExpectedSignatureNode,
    PathSignatureNode,
    SignatureKernelNode,
    SignatureLinearPricerNode,
)
from spxa.graph.stochastic_nodes import (
    BrownianMotionNode,
    CEVNode,
    CIRNode,
    GeometricBrownianMotionNode,
    HestonNode,
    KouJumpDiffusionNode,
    MertonJumpDiffusionNode,
    OrnsteinUhlenbeckNode,
    SABRNode,
    StochasticProcessNode,
    SubordinatedProcessNode,
    VasicekNode,
)
from spxa.graph.subgraph import SubgraphNode
from spxa.graph.subordination import AutomatedCumulantNode, BochnerSubordinationNode
from spxa.graph.surgery import (
    DoleansExponentialNode,
    DoobHTransformNode,
    ItoToStratonovichNode,
    StochasticLogarithmNode,
    StratonovichToItoNode,
    TanakaDecompositionNode,
)
from spxa.graph.types import (
    BackendType,
    DType,
    DomainType,
    ParameterBounds,
    PortDescriptor,
)

__all__ = [
    # Types
    "DType",
    "NodeType",
    "BackendType",
    "DomainType",
    "PortDescriptor",
    "ParameterBounds",
    # Nodes
    "Node",
    "ConstantNode",
    "ParameterNode",
    "OperatorNode",
    "OutputNode",
    # Math Ops
    "AddNode",
    "SubNode",
    "MulNode",
    "DivNode",
    "PowNode",
    "NegNode",
    "ExpNode",
    "LogNode",
    "SqrtNode",
    "AbsNode",
    "SinNode",
    "CosNode",
    "SinhNode",
    "CoshNode",
    "TanhNode",
    "NormalCDFNode",
    "NormalPDFNode",
    "ErfNode",
    "GammaNode",
    "SumNode",
    "MeanNode",
    "MaximumNode",
    "MinimumNode",
    "MatMulNode",
    "TransposeNode",
    "CholeskyNode",
    # DAG & Compiler
    "ComputationalGraph",
    "GraphCompiler",
    "EvaluationCache",
    # Stochastic Nodes
    "StochasticProcessNode",
    "BrownianMotionNode",
    "GeometricBrownianMotionNode",
    "OrnsteinUhlenbeckNode",
    "VasicekNode",
    "CIRNode",
    "CEVNode",
    "HestonNode",
    "SABRNode",
    "MertonJumpDiffusionNode",
    "KouJumpDiffusionNode",
    "SubordinatedProcessNode",
    # Characteristic & Moments
    "CharacteristicExponentNode",
    "CharacteristicFunctionNode",
    "AffineCharacteristicNode",
    "CumulantGeneratingFunctionNode",
    "GeneratorApplicationNode",
    "ProcessMomentNode",
    # Backends
    "Backend",
    "ExecutionContext",
    "AnalyticalBackend",
    "NumericalBackend",
    "SimulationBackend",
    "AcceleratedBackend",
    "get_backend",
    "ExecutionRouter",
    # Autodiff
    "DualNumber",
    "GradientTape",
    "value_and_grad",
    "SmoothIndicatorNode",
    "SoftplusNode",
    "compute_graph_hessian",
    # Pricing
    "EuropeanCallPayoffNode",
    "EuropeanPutPayoffNode",
    "DigitalPayoffNode",
    "AsianPayoffNode",
    "build_black_scholes_pricing_graph",
    "monte_carlo_graph_price",
    "compute_graph_greeks",
    "graph_greeks",
    "GreeksResult",
    # Calibration
    "MSELossNode",
    "ArbitragePenaltyNode",
    "GraphCalibrator",
    "FellerConstraintNode",
    # Risk
    "PortfolioRiskGraph",
    "ScenarioShockNode",
    "apply_macro_scenario",
    "NettingExposureNode",
    "GraphCVANode",
    # Universal Adapters & Implementation Layer
    "GraphAdapterRegistry",
    "to_graph",
    "as_graph_node",
    "UniversalProcessNode",
    "DifferentiableSDEAdapter",
    "RiskInstrumentAdapter",
    "adapt_portfolio",
    "CalculusGeneratorAdapter",
    "CopulaAdapter",
    "install_graph_extensions",
    # Advanced Depth: Serialization, Profiling & Subgraphs
    "GraphSerializer",
    "graph_summary",
    "to_dot",
    "GraphProfiler",
    "SubgraphNode",
    "BlackScholesPriceNode",
    "MultiAssetBasketNode",
    "CorrelatedDiffusionNode",
    # Deep Stochastic Calculus & Operators
    "ItoTransformNode",
    "GirsanovTransformNode",
    "EsscherTransformNode",
    "MorphNode",
    "QuadraticVarianceMorphNode",
    "BochnerSubordinationNode",
    "AutomatedCumulantNode",
    "InfinitesimalGeneratorNode",
    "MartingaleVerificationNode",
    "ScaleDensityNode",
    "SpeedDensityNode",
    "FellerBoundaryClassifierNode",
    "MalliavinBismutDeltaNode",
    # Semimartingale Decomposition Wires & Algebra
    "SemimartingaleWire",
    "SemimartingaleComponents",
    "QuadraticCovariationNode",
    "StochasticProductNode",
    "StochasticIntegralNode",
    # Four-Facet Process Duality Engine
    "SDEFacet",
    "GeneratorFacet",
    "FourierFacet",
    "DensityFacet",
    "ProcessDualityCore",
    "compile_levy_khintchine",
    "triplet_to_fourier_facet",
    # Algebraic Process Surgery Operators
    "DoleansExponentialNode",
    "StochasticLogarithmNode",
    "TanakaDecompositionNode",
    "DoobHTransformNode",
    "StratonovichToItoNode",
    "ItoToStratonovichNode",
    # Information Geometry & Manifolds
    "FisherInformationMetricNode",
    "NaturalGradientCalibratorNode",
    # Rough Paths & Tensor Signatures
    "PathSignatureNode",
    "SignatureKernelNode",
    "ExpectedSignatureNode",
    "SignatureLinearPricerNode",
    # Rough Volatility & Volterra SDEs
    "RoughKernelNode",
    "RoughBergomiNode",
    "FractionalRiccatiNode",
]
