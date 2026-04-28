import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.spec_decode import SpecConfig
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class SpecArgs:
    """Speculative decoding options (mirrors spec.py's ``OpenPi0Config`` fields).

    Enable via ``--spec.enable-speculative``. Supports both JAX and PyTorch
    backends.
    """

    # master switch
    enable_speculative: bool = False
    # number of parallel drafts
    spec_batch_size: int = 8
    # action horizon used for spec (None -> model.action_horizon)
    spec_action_horizon: int | None = None
    # diffusion steps for spec (None -> 10)
    spec_diffusion_num_steps: int | None = None
    # confidence selector hyperparameters
    spec_conf_alpha: float = 0.8
    spec_conf_eps: float = 1e-6
    # per-dim delta threshold in *exec* (un-normalized) action space
    spec_delta_threshold: float = 0.1
    # optional per-dim delta thresholds (takes precedence over scalar)
    spec_delta_thresholds: tuple[float, ...] = ()
    # extra switches
    spec_debug: bool = False
    spec_log_conf_stats: bool = True
    spec_verify_conf: bool = True
    spec_verify_seq: bool = True
    # environment action dim (controls the comparison heuristic)
    action_env_dim: int = 7
    # optional file path for spec debug log lines
    spec_log_path: str | None = None


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Speculative decoding options.
    spec: SpecArgs = dataclasses.field(default_factory=SpecArgs)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _build_spec_config(args: SpecArgs) -> SpecConfig | None:
    if not args.enable_speculative:
        return None
    return SpecConfig(
        enable_speculative=True,
        spec_batch_size=args.spec_batch_size,
        spec_action_horizon=args.spec_action_horizon,
        spec_diffusion_num_steps=args.spec_diffusion_num_steps,
        spec_conf_alpha=args.spec_conf_alpha,
        spec_conf_eps=args.spec_conf_eps,
        spec_delta_threshold=args.spec_delta_threshold,
        spec_delta_thresholds=tuple(args.spec_delta_thresholds) if args.spec_delta_thresholds else None,
        spec_debug=args.spec_debug,
        spec_log_conf_stats=args.spec_log_conf_stats,
        spec_verify_conf=args.spec_verify_conf,
        spec_verify_seq=args.spec_verify_seq,
        action_env_dim=args.action_env_dim,
        spec_log_path=args.spec_log_path,
    )


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    spec_config: SpecConfig | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            spec_config=spec_config,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    spec_config = _build_spec_config(args.spec)
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                spec_config=spec_config,
            )
        case Default():
            return create_default_policy(
                args.env, default_prompt=args.default_prompt, spec_config=spec_config
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
