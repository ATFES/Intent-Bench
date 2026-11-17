"""Public requests shared by CLI, Python and MCP."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from .model import Environment, Measurement

@dataclass(frozen=True)
class RunRequest:
    workload: str
    implementations: tuple[str, ...]
    environment: Environment = field(default_factory=Environment)
    measurement: Measurement = field(default_factory=Measurement)
    baseline: str | None = None
    cases: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    seed: int = 0
    output: Path | None = None
    kind: str = "workload"
    stage: str = "run"
    timeout: float = 300
    oracle_environment: Environment | None = None

    def __post_init__(self):
        if not self.implementations:
            raise ValueError("select implementations explicitly")
        if self.baseline is not None and self.baseline not in self.implementations:
            raise ValueError("baseline must be a selected implementation")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.kind != "workload" and self.oracle_environment is not None:
            raise ValueError("oracle_environment requires a numerical workload request; "
                             "use workload='program/<name>' with kind='workload' for an author program")

    def record(self):
        value = asdict(self)
        for environment in (value["environment"], value["oracle_environment"]):
            if environment is not None:
                for name in ("intent_root", "output_root"):
                    if environment[name] is not None:
                        environment[name] = str(environment[name])
        value["output"] = str(self.output) if self.output is not None else None
        return value

    @classmethod
    def read(cls, value):
        data = dict(value)
        for field in ("environment", "oracle_environment"):
            if data.get(field) is not None:
                env = dict(data[field])
                for name in ("intent_root", "output_root"):
                    if env.get(name) is not None:
                        env[name] = Path(env[name])
                data[field] = Environment(**env)
        data["measurement"] = Measurement(**data["measurement"])
        data["implementations"] = tuple(data["implementations"])
        data["cases"] = tuple(data["cases"])
        if data["output"] is not None:
            data["output"] = Path(data["output"])
        return cls(**data)
