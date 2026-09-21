""" """


from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

METHOD_ADAM = "adam"
METHOD_ADAM_VARIABLE_LR = "adam_variable_lr"
METHOD_LBFGS = "lbfgs"
METHOD_RPROP = "rprop"
METHOD_ADAM_AURA = "adam_aura"
METHOD_AURA_LIGHT = "aura_light"
METHOD_ASTRA = "astra"
METHOD_ECLIPSE = "eclipse"
METHOD_PULSAR = "pulsar"
METHOD_NADAMW = "nadamw"
METHOD_NADAM = "nadam"
METHOD_HCSCGM = "hcscgm"
METHOD_CvAMSGrad = "cvamsgrad"
METHOD_MUON = "muon"
METHOD_MUON_AURA = "muon_aura"
METHOD_ADAM_AURA_S = "adam_aura_s"
METHOD_MUON_AURA_S = "muon_aura_s"
METHOD_ADAM_AURA_SN = "adam_aura_sn"
METHOD_MUON_AURA_SN = "muon_aura_sn"
AURA_S_FAMILY = (METHOD_ADAM_AURA_S, METHOD_MUON_AURA_S, METHOD_ADAM_AURA_SN, METHOD_MUON_AURA_SN)
METHOD_ADAM_AURA_SIGN = "adam_aura_sign"
METHOD_MUON_AURA_SIGN = "muon_aura_sign"
AURA_SIGN_FAMILY = (METHOD_ADAM_AURA_SIGN, METHOD_MUON_AURA_SIGN)
METHOD_ADAM_AURA_SNR = "adam_aura_snr"
METHOD_MUON_AURA_SNR = "muon_aura_snr"
METHOD_ADAM_AURA_SNR_ABLATION = "adam_aura_snr_ablation"
AURA_SNR_FAMILY = (METHOD_ADAM_AURA_SNR, METHOD_MUON_AURA_SNR, METHOD_ADAM_AURA_SNR_ABLATION)
METHOD_ADAM_AURA_COSINE_ABLATION = "adam_aura_cosine_ablation"
METHOD_ADAM_AURA_SPRING_ABLATION = "adam_aura_spring_ablation"
METHOD_MUON_AURA_SPRING_ABLATION = "muon_aura_spring_ablation"
METHOD_ADAM_AURA_SPRING_EB = "adam_aura_spring_eb"
METHOD_SGD = "sgd"
METHOD_ADAMAXW = "adamaxw"
METHOD_CLBFGS = "cl_bfgs"
METHOD_NEU_OPTEB = "method_NEU_optEB"

METHODS = (
    METHOD_RPROP,
    METHOD_ADAM,
    METHOD_ADAM_VARIABLE_LR,
    METHOD_NADAMW,
    METHOD_CvAMSGrad,
    METHOD_MUON,
    METHOD_ADAM_AURA,
    METHOD_MUON_AURA,
)
    

TIMING_BASELINE_METHODS = (METHOD_SGD,)

EXPERIMENTAL_METHODS = (
    METHOD_LBFGS,
    METHOD_ASTRA,
    METHOD_ECLIPSE,
    METHOD_PULSAR,
    METHOD_AURA_LIGHT,
    METHOD_ADAM_AURA_S,
    METHOD_MUON_AURA_S,
    METHOD_ADAM_AURA_SN,
    METHOD_MUON_AURA_SN,
    METHOD_ADAM_AURA_SNR_ABLATION,
    METHOD_ADAM_AURA_COSINE_ABLATION,
    METHOD_ADAM_AURA_SNR,
    METHOD_MUON_AURA_SNR,
    METHOD_ADAM_AURA_SIGN,
    METHOD_MUON_AURA_SIGN,
    METHOD_ADAM_AURA_SPRING_EB,
    METHOD_ADAM_AURA_SPRING_ABLATION,
    METHOD_MUON_AURA_SPRING_ABLATION,
)

METHOD_COLORS = {
    METHOD_ADAM: "#332288",
    METHOD_ADAM_VARIABLE_LR: "#88CCEE",
    METHOD_LBFGS: "#DDCC77",
    METHOD_RPROP: "#AA4499",
    METHOD_ADAM_AURA: "#EE7733",
    METHOD_ASTRA: "#CC3311",
    METHOD_ECLIPSE: "#009988",
    METHOD_PULSAR: "#BBCC33",
    METHOD_AURA_LIGHT: "#EE3377",
    METHOD_NEU_OPTEB: "#000000",
    METHOD_NADAMW: "#CC6677",
    METHOD_CvAMSGrad: "#0077BB",
    METHOD_MUON: "#DDAA33",
    METHOD_MUON_AURA: "#117733",
    METHOD_ADAM_AURA_S: "#CC3311",
    METHOD_MUON_AURA_S: "#009988",
    METHOD_ADAM_AURA_SN: "#EE3377",
    METHOD_MUON_AURA_SN: "#33BBEE",
    METHOD_ADAM_AURA_SIGN: "#AA3377",
    METHOD_MUON_AURA_SIGN: "#228833",
    METHOD_ADAM_AURA_SNR: "#994455",
    METHOD_MUON_AURA_SNR: "#6699CC",
    METHOD_ADAM_AURA_SNR_ABLATION: "#EE99AA",
    METHOD_ADAM_AURA_COSINE_ABLATION: "#997700",
    METHOD_ADAM_AURA_SPRING_ABLATION: "#44BB99",
    METHOD_MUON_AURA_SPRING_ABLATION: "#004488",
    METHOD_ADAM_AURA_SPRING_EB: "#AAAA00",
    METHOD_SGD: "#BBBBBB",

    METHOD_ADAMAXW: "#999933",
    METHOD_NADAM: "#882255",
    METHOD_CLBFGS: "#44AA99",
    METHOD_HCSCGM: "#666666",
}

ADAM_VARIABLE_LR_DROP_AT_FRACTION = 0.5
ADAM_VARIABLE_LR_DROP_FACTOR = 10.0


@dataclass(frozen=True)
class LBFGSConfig:
    """ """

    memory_size: int = 10
    scale_init_precond: bool = True
    adam_warmup_epochs: int = 1000


@dataclass(frozen=True)
class AdamConfig:
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8


@dataclass(frozen=True)
class SgdConfig:
    """ """

    momentum: float | None = None
    nesterov: bool = False


@dataclass(frozen=True)
class NadamWConfig:
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class NadamConfig:
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8


@dataclass(frozen=True)
class CvAMSGradConfig:
    """ """

    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    bias_correction: bool = True


@dataclass(frozen=True)
class MuonConfig:
    """ """

    beta: float = 0.95
    ns_steps: int = 5
    nesterov: bool = True
    weight_decay: float = 0.0
    learning_rate_scale: float = 10.0
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    adam_weight_decay: float = 0.0


@dataclass(frozen=True)
class HCSCGMConfig:
    """ """

    theta_max: float = 1e4
    delta_1: float = 0.5
    sigma_1: float = 1e-4
    sigma_2: float = 0.9
    max_linesearch_steps: int = 20


@dataclass(frozen=True)
class RpropConfig:
    """ """

    eta_minus: float = 0.5
    eta_plus: float = 1.2
    min_step_size: float = 1e-6
    max_step_size: float = 50.0


@dataclass(frozen=True)
class AdamAuraConfig:
    """ """
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    # # Hyperparams 1
    # epsilon_e = 1e-6
    # beta_zeta = 0.95
    # chi_alignment = 0.7
    # chi_opposition = 0.4
    # psi_alignment = 0.02
    # psi_opposition = 0.3

    # Hyperparams 1: # Good on analytical cases
    epsilon_e: float = 1e-6
    beta_zeta: float = 0.95
    chi_alignment: float = 0.7
    chi_opposition: float = 0.4
    psi_alignment: float = 0.015
    psi_opposition: float = 0.3

    # # Hyperparams 2:  # more robust than hyperparms 1
    # epsilon_e: float = 1e-4 
    # beta_zeta: float = 0.95
    # chi_alignment: float = 0.7 | gamma grows only if chi >= this; higher -> rarer, more conservative growth
    # chi_opposition: float = 0.4 | gamma shrinks if chi <= this; higher -> shrinks more readily
    # psi_alignment: float = 0.015 | gamma grows only if |psi| <= this (complex params only); lower -> stricter growth
    # psi_opposition: float = 0.25 | gamma shrinks if |psi| >= this (complex params only); lower -> shrinks more on rotation

    # # Hyperparams 3: # optimized on CIFAR-10 simplified
    # epsilon_e = 1e-4 
    # beta_zeta = 0.85
    # chi_alignment = 0.85
    # chi_opposition = 0.45
    # psi_alignment = 0.0075
    # psi_opposition = 0.25
    
    eta_minus: float = 0.99 
    eta_plus: float = 1.01 
    gamma_min: float = 1e-3
    gamma_max: float =  1e3
    weight_decay: float = 1e-4 
    learning_rate_scale: float =  1.0
    gamma_init: float =  1.0  


@dataclass(frozen=True)
class MuonAuraConfig:
    """ """

    beta: float = 0.95 
    ns_steps: int = 5 # | Newton-Schulz iterations
    nesterov: bool = True  
    beta_1: float = 0.9  
    beta_2: float = 0.999  
    epsilon: float = 1e-8 

    # Hyperparams 1:
    epsilon_e: float = 1e-6 
    beta_zeta: float = 0.95 
    chi_alignment: float = 0.75 # | gamma grows only if chi >= this; higher -> rarer, more conservative growth
    chi_opposition: float = 0.4 # | gamma shrinks if chi <= this; higher -> shrinks more readily
    psi_alignment: float = 0.01 # | gamma grows only if |psi| <= this (complex params only); lower -> stricter growth
    psi_opposition: float = 0.2 # | gamma shrinks if |psi| >= this (complex params only); lower -> shrinks more on rotation

    # # Hyperparams 2: (identical to 1)
    # epsilon_e: float = 1e-6
    # beta_zeta: float = 0.95
    # chi_alignment: float = 0.75
    # chi_opposition: float = 0.4
    # psi_alignment: float = 0.01
    # psi_opposition: float = 0.2

    eta_minus: float = 0.99 
    eta_plus: float = 1.01 
    gamma_min: float = 1e-3
    gamma_max: float = 1e3 
    weight_decay: float = 1e-4 
    learning_rate_scale: float = 10.0
    gamma_init: float = 1.0 


@dataclass(frozen=True)
class AdamAuraSConfig:
    """ """

    beta_zeta: float = 0.95
    epsilon_e: float = 1e-12
    kappa_plus: float = 0.03
    kappa_minus: float = 0.005
    rotation_penalty: float = 1.0
    leak: float = 0.9998
    chi_opposition: float = 0.4
    psi_opposition: float = 0.3
    kappa_brake: float = 0.02
    gamma_min: float = 1e-3
    gamma_max: float = 1e3
    gamma_init: float = 1.0
    weight_decay: float = 0.0
    learning_rate_scale: float = 1.0


@dataclass(frozen=True)
class MuonAuraSConfig:
    """ """

    beta: float = 0.95
    ns_steps: int = 5
    nesterov: bool = True
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    learning_rate_scale: float = 10.0
    beta_zeta: float = 0.95
    epsilon_e: float = 1e-12
    kappa_plus: float = 0.01
    kappa_minus: float = 0.005
    rotation_penalty: float = 1.0
    leak: float = 0.9998
    chi_opposition: float = 0.4
    psi_opposition: float = 0.3
    kappa_brake: float = 0.02
    gamma_min: float = 1e-3
    gamma_max: float = 30.0
    gamma_init: float = 1.0
    weight_decay: float = 0.0


@dataclass(frozen=True)
class AdamAuraSNConfig(AdamAuraSConfig):
    """ """

    kappa_plus: float = 0.06
    noise_band: float = 2.0


@dataclass(frozen=True)
class MuonAuraSNConfig(MuonAuraSConfig):
    """ """

    kappa_plus: float = 0.02
    noise_band: float = 2.0


@dataclass(frozen=True)
class AdamAuraSignConfig:
    """ """

    beta: float = 0.95
    kappa_plus: float = 0.02
    kappa_minus: float = 0.0035
    brake_threshold: float = 0.5
    leak: float = 0.9998
    gamma_min: float = 1e-3
    gamma_max: float = 1e3
    weight_decay: float = 0.0
    learning_rate_scale: float = 1.0


@dataclass(frozen=True)
class MuonAuraSignConfig:
    """ """

    beta: float = 0.95
    ns_steps: int = 5
    nesterov: bool = True
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    learning_rate_scale: float = 10.0
    sign_beta: float = 0.95
    kappa_plus: float = 0.02
    kappa_minus: float = 0.0035
    brake_threshold: float = 0.5
    leak: float = 0.9998
    gamma_min: float = 1e-3
    gamma_max: float = 30.0
    weight_decay: float = 0.0
    momentum_reference: bool = True


@dataclass(frozen=True)
class AdamAuraSnrConfig:
    """ """

    beta_zeta: float = 0.95
    epsilon_e: float = 1e-30
    kappa_plus: float = 0.03
    kappa_minus: float = 0.005
    opposition_threshold: float = 0.35
    leak: float = 0.9998
    gamma_min: float = 1e-3
    gamma_max: float = 1e3
    weight_decay: float = 0.0
    learning_rate_scale: float = 1.0


@dataclass(frozen=True)
class MuonAuraSnrConfig:
    """ """

    beta: float = 0.95
    ns_steps: int = 5
    nesterov: bool = True
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    learning_rate_scale: float = 10.0
    beta_zeta: float = 0.95
    epsilon_e: float = 1e-12
    kappa_plus: float = 0.03
    kappa_minus: float = 0.005
    opposition_threshold: float = 0.5
    leak: float = 0.9998
    gamma_min: float = 1e-3
    gamma_max: float = 30.0
    weight_decay: float = 0.0
    momentum_reference: bool = True


@dataclass(frozen=True)
class AdamAuraSnrAblationConfig:
    """ """

    beta_zeta: float = AdamAuraSnrConfig.beta_zeta
    epsilon_e: float = AdamAuraSnrConfig.epsilon_e
    opposition_threshold: float = AdamAuraSnrConfig.opposition_threshold
    eta_plus: float = AdamAuraConfig.eta_plus
    eta_minus: float = AdamAuraConfig.eta_minus
    gamma_min: float = AdamAuraSnrConfig.gamma_min
    gamma_max: float = AdamAuraSnrConfig.gamma_max
    weight_decay: float = AdamAuraSnrConfig.weight_decay
    learning_rate_scale: float = AdamAuraSnrConfig.learning_rate_scale


@dataclass(frozen=True)
class AdamAuraSpringAblationConfig:
    """ """

    beta_zeta: float = AdamAuraConfig.beta_zeta
    epsilon_e: float = AdamAuraConfig.epsilon_e
    chi_opposition: float = AdamAuraConfig.chi_opposition
    psi_opposition: float = AdamAuraConfig.psi_opposition
    kappa_plus: float = AdamAuraSnrConfig.kappa_plus
    kappa_minus: float = AdamAuraSnrConfig.kappa_minus
    leak: float = AdamAuraSnrConfig.leak
    gamma_min: float = AdamAuraConfig.gamma_min
    gamma_max: float = AdamAuraConfig.gamma_max
    gamma_init: float = AdamAuraConfig.gamma_init
    weight_decay: float = AdamAuraConfig.weight_decay
    learning_rate_scale: float = AdamAuraConfig.learning_rate_scale


@dataclass(frozen=True)
class MuonAuraSpringAblationConfig:
    """ """

    beta: float = MuonAuraConfig.beta
    ns_steps: int = MuonAuraConfig.ns_steps
    nesterov: bool = MuonAuraConfig.nesterov
    beta_1: float = MuonAuraConfig.beta_1
    beta_2: float = MuonAuraConfig.beta_2
    epsilon: float = MuonAuraConfig.epsilon
    learning_rate_scale: float = MuonAuraConfig.learning_rate_scale
    beta_zeta: float = MuonAuraConfig.beta_zeta
    epsilon_e: float = MuonAuraConfig.epsilon_e
    chi_opposition: float = MuonAuraConfig.chi_opposition
    psi_opposition: float = MuonAuraConfig.psi_opposition
    kappa_plus: float = MuonAuraSnrConfig.kappa_plus
    kappa_minus: float = MuonAuraSnrConfig.kappa_minus
    leak: float = MuonAuraSnrConfig.leak
    gamma_min: float = MuonAuraConfig.gamma_min
    gamma_max: float = MuonAuraConfig.gamma_max
    gamma_init: float = MuonAuraConfig.gamma_init
    weight_decay: float = MuonAuraConfig.weight_decay


@dataclass(frozen=True)
class AdamAuraSpringEbConfig:
    """ """

    beta_zeta: float = AdamAuraConfig.beta_zeta
    epsilon_e: float = AdamAuraConfig.epsilon_e
    chi_alignment: float = AdamAuraConfig.chi_alignment
    chi_opposition: float = AdamAuraConfig.chi_opposition
    psi_alignment: float = AdamAuraConfig.psi_alignment
    psi_opposition: float = AdamAuraConfig.psi_opposition
    eta_minus: float = AdamAuraConfig.eta_minus
    eta_plus: float = AdamAuraConfig.eta_plus
    leak: float = 0.9998
    gamma_min: float = AdamAuraConfig.gamma_min
    gamma_max: float = AdamAuraConfig.gamma_max
    gamma_init: float = AdamAuraConfig.gamma_init
    weight_decay: float = AdamAuraConfig.weight_decay
    learning_rate_scale: float = AdamAuraConfig.learning_rate_scale


@dataclass(frozen=True)
class AstraConfig:
    """ """

    beta_zeta: float = 0.95
    epsilon_e: float = 1e-12
    chi_alignment: float = 0.15
    chi_opposition: float = -0.15
    psi_alignment: float = 0.25
    psi_opposition: float = 0.5
    eta_minus: float = 0.99
    eta_plus: float = 1.01
    gamma_min: float = 1e-3
    gamma_max: float = 1e3
    leak_rate: float = 1.0
    leak_exponent: float = 0.999
    weight_decay: float = 1e-4
    gamma_init: float = 1.0


@dataclass(frozen=True)
class EclipseConfig:
    """ """

    epsilon_e: float = 1e-12
    chi_alignment: float = 0.65
    chi_opposition: float = 0.0
    psi_alignment: float = 0.015
    psi_opposition: float = 0.06
    eta_minus: float = 0.99
    eta_plus: float = 1.01
    gamma_min: float = 1e-3
    gamma_max: float = 1e3
    leak_rate: float = 0.999
    weight_decay: float = 1e-4
    gamma_init: float = 1.0


@dataclass(frozen=True)
class PulsarConfig:
    """ """

    kappa: float = 0.5
    epsilon_e: float = 1e-12
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class AuraLightConfig:
    """ """

    beta_1: float = 0.9
    beta_zeta: float = 0.95
    epsilon_e: float = 1e-12
    chi_alignment: float = 0.7
    chi_opposition: float = 0.4
    psi_alignment: float = 0.02
    psi_opposition: float = 0.3
    eta_minus: float = 0.99
    eta_plus: float = 1.01
    gamma_min: float = 1e-3
    gamma_max: float = 1e2
    weight_decay: float = 1e-4
    learning_rate_scale: float = 0.001
    gamma_init: float = 1.0


ADAM_CONFIG = AdamConfig()
SGD_CONFIG = SgdConfig()
LBFGS_CONFIG = LBFGSConfig()
RPROP_CONFIG = RpropConfig()
NADAMW_CONFIG = NadamWConfig()
NADAM_CONFIG = NadamConfig()
CVAMSGRAD_CONFIG = CvAMSGradConfig()
MUON_CONFIG = MuonConfig()
MUON_AURA_CONFIG = MuonAuraConfig()
AURA_S_CONFIG = AdamAuraSConfig()
MUON_AURA_S_CONFIG = MuonAuraSConfig()
AURA_SN_CONFIG = AdamAuraSNConfig()
MUON_AURA_SN_CONFIG = MuonAuraSNConfig()
AURA_SIGN_CONFIG = AdamAuraSignConfig()
MUON_AURA_SIGN_CONFIG = MuonAuraSignConfig()
AURA_SNR_CONFIG = AdamAuraSnrConfig()
MUON_AURA_SNR_CONFIG = MuonAuraSnrConfig()
AURA_SNR_ABLATION_CONFIG = AdamAuraSnrAblationConfig()
AURA_SPRING_CONFIG = AdamAuraSpringAblationConfig()
MUON_AURA_SPRING_CONFIG = MuonAuraSpringAblationConfig()
AURA_SPRING_EB_CONFIG = AdamAuraSpringEbConfig()
HCSCGM_CONFIG = HCSCGMConfig()
AURA_CONFIG = AdamAuraConfig()
ASTRA_CONFIG = AstraConfig()
ECLIPSE_CONFIG = EclipseConfig()
PULSAR_CONFIG = PulsarConfig()
AURA_LIGHT_CONFIG = AuraLightConfig()


def hyperparameters_for_method(
    method: str,
    *,
    adam_config: AdamConfig = ADAM_CONFIG,
    sgd_config: SgdConfig = SGD_CONFIG,
    rprop_config: RpropConfig = RPROP_CONFIG,
    aura_config: AdamAuraConfig = AURA_CONFIG,
    astra_config: AstraConfig = ASTRA_CONFIG,
    eclipse_config: EclipseConfig = ECLIPSE_CONFIG,
    pulsar_config: PulsarConfig = PULSAR_CONFIG,
    aura_light_config: AuraLightConfig = AURA_LIGHT_CONFIG,
    nadamw_config: NadamWConfig = NADAMW_CONFIG,
    nadam_config: NadamConfig = NADAM_CONFIG,
    cvamsgrad_config: CvAMSGradConfig = CVAMSGRAD_CONFIG,
    muon_config: MuonConfig = MUON_CONFIG,
    muon_aura_config: MuonAuraConfig = MUON_AURA_CONFIG,
    aura_s_config: AdamAuraSConfig = AURA_S_CONFIG,
    muon_aura_s_config: MuonAuraSConfig = MUON_AURA_S_CONFIG,
    aura_sn_config: AdamAuraSNConfig = AURA_SN_CONFIG,
    muon_aura_sn_config: MuonAuraSNConfig = MUON_AURA_SN_CONFIG,
    aura_sign_config: AdamAuraSignConfig = AURA_SIGN_CONFIG,
    muon_aura_sign_config: MuonAuraSignConfig = MUON_AURA_SIGN_CONFIG,
    aura_snr_config: AdamAuraSnrConfig = AURA_SNR_CONFIG,
    muon_aura_snr_config: MuonAuraSnrConfig = MUON_AURA_SNR_CONFIG,
    aura_snr_ablation_config: AdamAuraSnrAblationConfig = AURA_SNR_ABLATION_CONFIG,
    aura_spring_config: AdamAuraSpringAblationConfig = AURA_SPRING_CONFIG,
    muon_aura_spring_config: MuonAuraSpringAblationConfig = MUON_AURA_SPRING_CONFIG,
    aura_spring_eb_config: AdamAuraSpringEbConfig = AURA_SPRING_EB_CONFIG,
    hcscgm_config: HCSCGMConfig = HCSCGM_CONFIG,
    lbfgs_config: LBFGSConfig = LBFGS_CONFIG,
) -> dict[str, Any]:
    """ """

    if method in (METHOD_ADAM, METHOD_ADAM_VARIABLE_LR):
        return {"adam": asdict(adam_config)}
    if method == METHOD_SGD:
        return {"sgd": asdict(sgd_config)}
    if method == METHOD_NADAMW:
        return {"nadamw": asdict(nadamw_config)}
    if method == METHOD_NADAM:
        return {"nadam": asdict(nadam_config)}
    if method == METHOD_CvAMSGrad:
        return {"cvamsgrad": asdict(cvamsgrad_config)}
    if method == METHOD_MUON:
        return {"muon": asdict(muon_config)}
    if method == METHOD_MUON_AURA:
        return {"muon_aura": asdict(muon_aura_config)}
    if method == METHOD_ADAM_AURA_S:
        return {"adam": asdict(adam_config), "aura_s": asdict(aura_s_config)}
    if method == METHOD_MUON_AURA_S:
        return {"muon_aura_s": asdict(muon_aura_s_config)}
    if method == METHOD_ADAM_AURA_SN:
        return {"adam": asdict(adam_config), "aura_sn": asdict(aura_sn_config)}
    if method == METHOD_MUON_AURA_SN:
        return {"muon_aura_sn": asdict(muon_aura_sn_config)}
    if method == METHOD_ADAM_AURA_SIGN:
        return {"adam": asdict(adam_config), "aura_sign": asdict(aura_sign_config)}
    if method == METHOD_MUON_AURA_SIGN:
        return {"muon_aura_sign": asdict(muon_aura_sign_config)}
    if method == METHOD_ADAM_AURA_SNR:
        return {"adam": asdict(adam_config), "aura_snr": asdict(aura_snr_config)}
    if method == METHOD_MUON_AURA_SNR:
        return {"muon_aura_snr": asdict(muon_aura_snr_config)}
    if method == METHOD_ADAM_AURA_SNR_ABLATION:
        return {"adam": asdict(adam_config), "aura_snr_ablation": asdict(aura_snr_ablation_config)}
    if method == METHOD_ADAM_AURA_COSINE_ABLATION:
        return {"aura": asdict(aura_config)}
    if method == METHOD_ADAM_AURA_SPRING_ABLATION:
        return {"adam": asdict(adam_config), "aura_spring": asdict(aura_spring_config)}
    if method == METHOD_MUON_AURA_SPRING_ABLATION:
        return {"muon_aura_spring": asdict(muon_aura_spring_config)}
    if method == METHOD_ADAM_AURA_SPRING_EB:
        return {"adam": asdict(adam_config), "aura_spring_eb": asdict(aura_spring_eb_config)}
    if method == METHOD_RPROP:
        return {"rprop": asdict(rprop_config)}
    if method == METHOD_HCSCGM:
        return {"hcscgm": asdict(hcscgm_config)}
    if method == METHOD_LBFGS:
        return {"adam": asdict(adam_config), "lbfgs": asdict(lbfgs_config)}
    if method == METHOD_ADAM_AURA:
        return {"aura": asdict(aura_config)}
    if method == METHOD_ASTRA:
        return {"adam": asdict(adam_config), "astra": asdict(astra_config)}
    if method == METHOD_ECLIPSE:
        return {"adam": asdict(adam_config), "eclipse": asdict(eclipse_config)}
    if method == METHOD_PULSAR:
        return {"adam": asdict(adam_config), "pulsar": asdict(pulsar_config)}
    if method == METHOD_AURA_LIGHT:
        return {"aura_light": asdict(aura_light_config)}
    raise ValueError(f"Unknown method {method!r}")


class CachedRun(NamedTuple):
    """ """

    config: dict[str, Any]
    result: Any
    run_id: str = ""


def _hyperparameter_leaves(hyperparameters: dict[str, Any]) -> set[tuple[str, str]]:
    leaves = set()
    for name, value in hyperparameters.items():
        if isinstance(value, dict):
            leaves |= _hyperparameter_leaves(value)
        else:
            leaves.add((name, repr(value)))
    return leaves


def same_hyperparameters(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """Whether two ``hyperparameters_for_method`` dicts hold the same name/value leaves,
    regardless of how they are grouped (a regrouping must not invalidate cached runs)."""

    return _hyperparameter_leaves(saved) == _hyperparameter_leaves(current)
