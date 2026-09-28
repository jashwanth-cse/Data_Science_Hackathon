from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path = Path("outputs")
    figures_dir: Path = Path("figures")
    model_dir: Path = Path("models")
    processed_dir: Path = Path("data/processed")
    report_dir: Path = Path("reports")
    random_state: int = 42
    n_jobs: int = 2
    bootstrap_resamples: int = 1000
    bootstrap_pairs: int = 20

    def create_directories(self) -> None:
        for path in (
            self.output_dir,
            self.figures_dir,
            self.model_dir,
            self.processed_dir,
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

