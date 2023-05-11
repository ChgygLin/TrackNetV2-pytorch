import yaml
from pathlib import Path



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


def yaml_load(file):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

def check_dataset(data):
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary
    
    path = Path(data.get('path'))  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path  # download scripts
    
    for k in 'train', 'val':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    
    return data