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

    def __post_init__(self):
        if not self.implementations:
            raise ValueError("select implementations explicitly")
        if self.baseline is not None and self.baseline not in self.implementations:
            raise ValueError("baseline must be a selected implementation")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def record(self):
        value = asdict(self)
        for name in ("intent_root", "output_root"):
            item = value["environment"][name]
            if item is not None:
                value["environment"][name] = str(item)
        value["output"] = str(self.output) if self.output is not None else None
        return value

    @classmethod
    def read(cls, value):
        data = dict(value)
        env = dict(data.pop("environment"))
        for name in ("intent_root", "output_root"):
            if env.get(name) is not None:
                env[name] = Path(env[name])
        data["environment"] = Environment(**env)
        data["measurement"] = Measurement(**data["measurement"])
        data["implementations"] = tuple(data["implementations"])
        data["cases"] = tuple(data["cases"])
        if data["output"] is not None:
            data["output"] = Path(data["output"])
        return cls(**data)
