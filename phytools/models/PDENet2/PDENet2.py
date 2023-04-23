import torch
import timeout_decorator
import numpy as np
from phytools.optim import NumpyFunctionInterface
from scipy.optimize import fmin_bfgs as bfgs
from phytools.models.PDENet2 import setenv,initparameters


class PDENet2():

    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):

        cfg, callback, model, data_model, sampling, addnoise = setenv.setenv(self.cfg)

        torch.cuda.manual_seed_all(cfg["torchseed"])
        torch.manual_seed(cfg["torchseed"])
        np.random.seed(cfg["npseed"])

        # initialization of parameters
        if cfg["start_from"]<0:
            initparameters.initkernels(model, scheme=cfg["scheme"])
            # initparameters.renormalize(model, u0)
            initparameters.initexpr(model, viscosity=cfg["viscosity"], pattern='random')
        else: # load checkpoint of layer-$start_from
            callback.load(cfg["start_from"], iternum='final')

        # train
        for block in cfg["blocks"]:
            if block<=cfg["start_from"]:
                continue
            print('block: ', block)
            print('name: ', cfg["name"])
            r = np.random.randn()+torch.randn(1,dtype=torch.float64,device=cfg["device"]).item()
            with callback.open() as output:
                print('device: ', cfg["device"], file=output)
                print('generate a random number to check random seed: ', r, file=output)
            # print('block: ', block)
            if block == 0:
                callback.stage = 'warmup'
                isfrozen = (False if cfg["constraint"] == 'free' else True)
            else:
                callback.stage = 'block-'+str(block)
                isfrozen = False
                if cfg["constraint"] == 'frozen':
                    isfrozen = True
            stepnum = block if block>=1 else 1
            layerweight = [1,]*stepnum
            # layerweight = list(1/(stepnum+1-i)**2 for i in range(1,stepnum+1))
            # generate data
            u_obs,u_true,u = setenv.data(model, data_model, cfg, sampling, addnoise, block, cfg["data_start_time"])
            print("u_obs shape: batchsize x channelNum x xgridsize x ygridsize")
            print(u_obs[0].shape)
            print("u_obs.abs().max()")
            print(u_obs[0].abs().max())
            print("u_obs variance")
            print(initparameters.trainvar(model.UInputs(u_obs[0])))
            # set NumpyFunctionInterface
            def forward():
                stableloss, dataloss, sparseloss, momentloss = setenv.loss(model, u_obs, cfg, block, layerweight)
                if block == 0:
                    # for stage='warmup', no regularization term used
                    stableloss = 0
                    sparseloss = 0
                    momentloss = 0
                if cfg["constraint"] == 'frozen':
                    momentloss = 0
                loss = cfg["stablize"] * stableloss + dataloss \
                       + stepnum * cfg["sparsity"] * sparseloss \
                       + stepnum * cfg["momentsparsity"] * momentloss
                if torch.isnan(loss):
                    loss = (torch.ones(1,requires_grad=True)/torch.zeros(1)).to(loss)
                return loss
            nfi = NumpyFunctionInterface([
                dict(params=model.diff_params(), isfrozen=isfrozen,
                    x_proj=model.diff_x_proj, grad_proj=model.diff_grad_proj),
                dict(params=model.expr_params(),
                    isfrozen=False)
                ], forward=forward, always_refresh=False)
            callback.nfi = nfi
            def callbackhook(_callback, *args):
                # global model,block,u0_obs,T,stable_loss,data_loss,sparse_loss
                stableloss, dataloss, sparseloss, momentloss = setenv.loss(model, u_obs, cfg, block, layerweight)
                stableloss, dataloss, sparseloss, momentloss = \
                    stableloss.item(), dataloss.item(), sparseloss.item(), momentloss.item()
                with _callback.open() as output:
                    print("stableloss: {:.2e}".format(stableloss), "  dataloss: {:.2e}".format(dataloss),
                            "  sparseloss: {:.2e}".format(sparseloss), "momentloss: {:.2e}".format(momentloss),
                            file=output)
                return None
            callbackhookhandle = callback.register_hook(callbackhook)
            if block == 0:
                callback.save(nfi.flat_param, 'start')
            try:
                # optimize
                xopt = bfgs(nfi.f, nfi.flat_param, nfi.fprime, gtol=2e-16, maxiter=cfg["maxiter"], callback=callback)
                # xopt,f,d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=maxiter, callback=callback, factr=1e7, pgtol=1e-8,maxiter=maxiter,iprint=0)
                np.set_printoptions(precision=2, linewidth=90)
                print("convolution moment and kernels")
                for k in range(cfg["max_order"] + 1):
                    for j in range(k+1):
                        print((model.__getattr__('fd'+str(j)+str(k-j)).moment).data.cpu().numpy())
                        print((model.__getattr__('fd'+str(j)+str(k-j)).kernel).data.cpu().numpy())
                for p in model.expr_params():
                    print("SymNet parameters")
                    print(p.data.cpu().numpy())
            except RuntimeError as Argument:
                with callback.open() as output:
                    print(Argument, file=output) # if overflow then just print and continue
            finally:
                # save parameters
                nfi.flat_param = xopt
                callback.save(xopt, 'final')
                with callback.open() as output:
                    print('finally, finish this stage', file=output)
                callback.record(xopt, callback.ITERNUM)
                callbackhookhandle.remove()
                @timeout_decorator.timeout(20)
                def printcoeffs():
                    with callback.open() as output:
                        print('current expression:', file=output)
                        for poly in model.polys:
                            tsym, csym = poly.coeffs()
                            print(tsym[:20], file=output)
                            print(csym[:20], file=output)
                try:
                    printcoeffs()
                except timeout_decorator.TimeoutError:
                    with callback.open() as output:
                        print('Time out', file=output)

        u_obs, u_true,u = setenv.data(model, data_model, cfg, sampling, addnoise, block=1, data_start_time=0)
        with callback.open() as output:
            print("u_obs.abs().max()", file=output)
            print(u_obs[0].abs().max(), file=output)
        with torch.no_grad():
            with callback.open() as output:
                print("model(u_obs[0],T=50*dt).abs().max()", file=output)
                print(model(u_obs[0], T=50*cfg["dt"]).abs().max(), file=output)
                print("model(u_obs[0],T=100*dt).abs().max()", file=output)
                print(model(u_obs[0], T=300*cfg["dt"]).abs().max(), file=output)
