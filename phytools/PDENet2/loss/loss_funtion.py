import torch


def _sparse_loss(model):
    """
    SymNet regularization
    """
    loss = 0
    s = 1e-3
    for p in model.expr_params():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss
def _moment_loss(model):
    """
    Moment regularization
    """
    loss = 0
    s = 1e-2
    for p in model.diff_params():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss
def loss(model, u_obs, globalnames, block, layerweight=None):
    if layerweight is None:
        layerweight = [0,]*stepnum
        layerweight[-1] = 1
    dt = globalnames['dt']
    stableloss = 0
    dataloss = 0
    sparseloss = _sparse_loss(model)
    momentloss = _moment_loss(model)
    stepnum = block if block>=1 else 1
    ut = u_obs[0]
    for steps in range(1,stepnum+1):
        uttmp = model(ut, T=dt)
        upad = model.fd00.pad(ut)
        stableloss = stableloss+(model.relu(uttmp-model.maxpool(upad))**2+
                model.relu(-uttmp-model.maxpool(-upad))**2).sum()
        dataloss = dataloss+\
                layerweight[steps-1]*torch.mean(((uttmp-u_obs[steps])/dt)**2)
                # layerweight[steps-1]*torch.mean(((uttmp-u_obs[steps])/(steps*dt))**2)
        ut = uttmp
    return stableloss, dataloss, sparseloss, momentloss

