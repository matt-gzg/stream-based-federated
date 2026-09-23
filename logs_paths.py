from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent / 'logs'

def output_path(model_name, filename):
    directory = OUTPUT_ROOT / model_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename