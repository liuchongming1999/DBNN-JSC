import resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import numpy as np
import torch
from pathlib import Path
from cpu_backend import CPUModel
root=Path(__file__).resolve().parents[1].parent
state=torch.load(root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
torch.set_num_threads(1)
model=CPUModel(state,np.arange(64),1001)
x=np.zeros((500,1001),np.float32)
print(model.run(x,1,0,0,1)[:3],flush=True)
