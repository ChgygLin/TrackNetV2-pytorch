import torch
import pnnx

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.tracknet import TrackNet


model = TrackNet().to("cpu")
model.load_state_dict(torch.load("./last.pt"))
model = model.eval()

x = torch.rand(1, 9, 288, 512)

opt_model = pnnx.export(model, "./last_opt.pt", x, fp16 = False)
