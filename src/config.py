import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class SKABDataConfig:
    raw_path: str
    folders: List[str]
    target_col: str
    datetime_col: str
    exclude_cols: List[str]
    separator: str


@dataclass
class BATADALDataConfig:
    raw_path: str
    file: str
    label_col: str
    normal_value: int
    time_cols: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class DataConfig:
    skab: SKABDataConfig
    batadal: BATADALDataConfig


@dataclass
class PCAConfig:
    enabled: bool
    n_components: int


@dataclass
class PreprocessingConfig:
    normalize: bool
    handle_missing: bool
    pca: PCAConfig


@dataclass
class EarlyStoppingConfig:
    patience: int
    monitor: str


@dataclass
class LSTMConfig:
    units: List[int]
    dropout: float


@dataclass
class GRUConfig:
    units: List[int]
    dropout: float


@dataclass
class CNNConfig:
    filters: List[int]
    kernel_size: int
    dropout: float


@dataclass
class ModelConfig:
    seeds: List[int]
    epochs: int
    batch_size: int
    sequence_length: int
    early_stopping: EarlyStoppingConfig
    lstm: LSTMConfig
    gru: GRUConfig
    cnn: CNNConfig


@dataclass
class AutomataParamSearch:
    window_sizes: List[int]
    alphabet_sizes: List[int]


@dataclass
class AutomataConfig:
    window_size: int
    alphabet_size: int
    param_search: AutomataParamSearch


@dataclass
class NoiseConfig:
    gaussian_std: float


@dataclass
class ExperimentConfig:
    noise: NoiseConfig
    scenarios: List[str]


@dataclass
class SKABEvalConfig:
    method: str
    n_splits: int
    group_col: str


@dataclass
class BATADALEvalConfig:
    method: str


@dataclass
class EvaluationConfig:
    skab: SKABEvalConfig
    batadal: BATADALEvalConfig


@dataclass
class ResultsConfig:
    output_dir: str
    log_dir: str


@dataclass
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    automata: AutomataConfig
    experiment: ExperimentConfig
    evaluation: EvaluationConfig
    results: ResultsConfig

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls._build(raw)

    @classmethod
    def _build(cls, raw: dict) -> "Config":
        d = raw["data"]
        skab = SKABDataConfig(**d["skab"])
        batadal = BATADALDataConfig(**d["batadal"])

        p = raw["preprocessing"]
        preprocessing = PreprocessingConfig(
            normalize=p["normalize"],
            handle_missing=p["handle_missing"],
            pca=PCAConfig(**p["pca"]),
        )

        m = raw["model"]
        model = ModelConfig(
            seeds=m["seeds"],
            epochs=m["epochs"],
            batch_size=m["batch_size"],
            sequence_length=m["sequence_length"],
            early_stopping=EarlyStoppingConfig(**m["early_stopping"]),
            lstm=LSTMConfig(**m["lstm"]),
            gru=GRUConfig(**m["gru"]),
            cnn=CNNConfig(**m["cnn"]),
        )

        a = raw["automata"]
        automata = AutomataConfig(
            window_size=a["window_size"],
            alphabet_size=a["alphabet_size"],
            param_search=AutomataParamSearch(**a["param_search"]),
        )

        e = raw["experiment"]
        experiment = ExperimentConfig(
            noise=NoiseConfig(**e["noise"]),
            scenarios=e["scenarios"],
        )

        ev = raw["evaluation"]
        evaluation = EvaluationConfig(
            skab=SKABEvalConfig(**ev["skab"]),
            batadal=BATADALEvalConfig(**ev["batadal"]),
        )

        return cls(
            data=DataConfig(skab=skab, batadal=batadal),
            preprocessing=preprocessing,
            model=model,
            automata=automata,
            experiment=experiment,
            evaluation=evaluation,
            results=ResultsConfig(**raw["results"]),
        )
