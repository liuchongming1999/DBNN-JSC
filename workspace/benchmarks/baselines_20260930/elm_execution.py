"""Author ELM equations expressed with native ops for CUDA graph capture.

Uses the original module's parameters directly. This changes execution only,
never architecture, loss, samples, optimizer, schedule, or inference code.
"""
import torch
from torch.nn import functional as F


def plain_forward(model, inputs):
    batch,steps,_=inputs.shape
    w_s=torch.relu(model._proto_w_s)
    tau_m=(model.memory_tau_max-model.memory_tau_min)*torch.sigmoid(model._proto_tau_m)+model.memory_tau_min
    kappa_s=torch.exp(-model.delta_t/torch.clamp(model.tau_s,min=1e-6))
    kappa_m=torch.exp(-model.delta_t/torch.clamp(tau_m,min=1e-6))
    routed=torch.index_select(inputs,2,model.input_to_synapse_indices)*model.valid_indices_mask
    synapse=torch.zeros(batch,model.num_synapse,device=inputs.device)
    memory=torch.zeros(batch,model.num_memory,device=inputs.device)
    first=model.mlp.network[0];second=model.mlp.network[2];outputs=[]
    assert model.mlp_num_layers==1 and model.mlp_activation=='relu'
    for t in range(steps):
        synapse=kappa_s*synapse+w_s*routed[:,t]
        branch=synapse.view(batch,model.num_branch,-1).sum(dim=-1)
        hidden=F.relu(F.linear(torch.cat([branch,kappa_m*memory],dim=-1),first.weight,first.bias))
        delta=torch.tanh(F.linear(hidden,second.weight,second.bias)*2/3)*1.7159
        memory=kappa_m*memory+model.lambda_value*(1-kappa_m)*delta
        outputs.append(F.linear(memory,model.w_y.weight,model.w_y.bias))
    return torch.stack(outputs,dim=-2)
