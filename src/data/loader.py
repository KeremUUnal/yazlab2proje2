import pandas as pd
from pathlib import Path
from src.config import Config


class SKABLoader:
    def __init__(self, config: Config):
        self.config = config

    def load(self) -> pd.DataFrame:
        cfg = self.config.data.skab
        base = Path(cfg.raw_path)
        dfs = []
        for folder in cfg.folders:
            folder_path = base / folder
            csv_files = sorted(folder_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"CSV bulunamadı: {folder_path}")
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, sep=cfg.separator)
                df["source_group"] = folder
                df["source_file"] = csv_file.name
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"SKAB verisi bulunamadı: {base}")
        combined = pd.concat(dfs, ignore_index=True)
        return combined


class BATADALLoader:
    def __init__(self, config: Config):
        self.config = config

    def load(self) -> pd.DataFrame:
        cfg = self.config.data.batadal
        path = Path(cfg.raw_path) / cfg.file
        if not path.exists():
            raise FileNotFoundError(f"BATADAL verisi bulunamadı: {path}")
        # skipinitialspace: sütun adlarındaki baştaki boşlukları temizler
        df = pd.read_csv(path, skipinitialspace=True)
        if cfg.label_col not in df.columns:
            raise ValueError(
                f"Etiket sütunu '{cfg.label_col}' bulunamadı. "
                f"Mevcut sütunlar: {list(df.columns)}"
            )
        # ATT_FLAG: -999 = normal → 0, diğer = anomali → 1
        df[cfg.label_col] = (df[cfg.label_col] != cfg.normal_value).astype(int)
        return df
