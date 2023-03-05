import torch
import time
import sys
import warnings
from os.path import join as ospj
warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
sys.path.insert(1, '../..')
from src.deepymod_torch.output import Tensorboard, progress
from src.deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from src.deepymod_torch.sparsity import scaling, threshold

def train_rf(model, data, target, optimizer, max_iterations, loss_func_args={'l1':1e-5}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    # 不完备的v值，因此需要更改prediction和target的mse_loss计算形式
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list)

        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        # mse只算u，不算v，target中u在前v在后
        # print("taeget_u ", target[:, 0:1].shape)
        # # [10000, 1]
        # print("prediction_u ", prediction[:, 0:1].shape)
        # # [10000, 1]
        loss_mse = mse_loss(prediction[:, 0:1], target[:, 0:1])
        loss_l1 = l1_loss(coeff_vector_scaled_list, loss_func_args['l1'])
        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1)

        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item())
            board.write(iteration, loss, loss_mse, loss_reg, loss_l1, coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train(model, data, target, optimizer, max_iterations, loss_func_args={'l1':1e-5}):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_mse = mse_loss(prediction, target)
        loss_l1 = l1_loss(coeff_vector_scaled_list, loss_func_args['l1'])
        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1)
        
        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item())
            board.write(iteration, loss, loss_mse, loss_reg, loss_l1, coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_mse(model, data, target, optimizer, max_iterations, loss_func_args={}):
    '''Trains the deepmod model only on the MSE. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 

        # Calculating loss
        loss_mse = mse_loss(prediction, target)
        loss = torch.sum(loss_mse)

        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), 0, 0)
            board.write(iteration, loss, loss_mse, [0], [0], coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

# def train_deepmod(model, data, target, optimizer, max_iterations, loss_func_args):
#     '''Performs full deepmod cycle: trains model, thresholds and trains again for unbiased estimate. Updates model in-place.'''
#     # Train first cycle and get prediction
#     train(model, data, target, optimizer, max_iterations, loss_func_args)
#     prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
#     # prediction, time_deriv_list, sparse_theta_list, coeff_vector_list
#     print()
#     # print("prediction")
#     # print(prediction.shape)
#     # # (10000,2)
#     # print("time_deriv_list")
#     # print(len(time_deriv_list))
#     # # 2
#     # print(time_deriv_list[0].shape)
#     # # (10000,1)
#     # print("sparse_theta_list")
#     # print(len(sparse_theta_list))
#     # # 2
#     # print(sparse_theta_list[0].shape)
#     # # (10000, 111)
#     # print("coeff_vector_list")
#     # print(len(coeff_vector_list))
#     # # 2
#     # print(coeff_vector_list[0].shape)
#     # # (111,1)
#     # Threshold, set sparsity mask and coeff vector
#     sparse_coeff_vector_list, sparsity_mask_list = threshold(coeff_vector_list, sparse_theta_list, time_deriv_list)
#     # print("sparse_coeff_vector_list")
#     # print(len(sparse_coeff_vector_list))
#     # # 2
#     # print(sparse_coeff_vector_list[0].shape)
#     # # (2,1)
#     # print("sparsity_mask_list")
#     # print(len(sparsity_mask_list))
#     # # 2
#     # print(sparsity_mask_list[0].shape)
#     # # (2)
#     model.fit.sparsity_mask = sparsity_mask_list
#     model.fit.coeff_vector = torch.nn.ParameterList(sparse_coeff_vector_list)
#
#     # TODO 所有的项系数都输出了
#     # print()
#     # print("*Primitive*:", coeff_vector_list[0])
#     print()
#     print(sparse_coeff_vector_list)
#     # [[0.1062],[-0.8332]] 每次都一样
#     print(sparsity_mask_list)
#     # [2,4] (u_xx, uu_x)
#     # print("fit.mask")
#     # print(model.fit.sparsity_mask)
#     # print("fit.vector")
#     # print(model.fit.coeff_vector)
#
#     #Resetting optimizer for different shapes, train without l1
#     optimizer.param_groups[0]['params'] = model.parameters()
#     print() #empty line for correct printing
#     train(model, data, target, optimizer, max_iterations, dict(loss_func_args, **{'l1': 0.0}))

# TODO num_burgers
def train_deepmod(model, data, target, optimizer, max_iterations, loss_func_args):
    '''Performs full deepmod cycle: trains model, thresholds and trains again for unbiased estimate. Updates model in-place.'''
    # Train first cycle and get prediction
    train(model, data, target, optimizer, max_iterations, loss_func_args)
    prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
    # prediction, time_deriv_list, sparse_theta_list, coeff_vector_list
    # print()
    # print("prediction")
    # print(prediction.shape)
    # # (1000,1)
    # print("time_deriv_list")
    # print(len(time_deriv_list))
    # # 1
    # print(time_deriv_list[0].shape)
    # # (1000,1)
    # print("sparse_theta_list")
    # print(len(sparse_theta_list))
    # # 1
    # print(sparse_theta_list[0].shape)
    # # (1000,9)
    # print("coeff_vector_list")
    # print(len(coeff_vector_list))
    # # 1
    # print(coeff_vector_list[0].shape)
    # # (9,1)
    # Threshold, set sparsity mask and coeff vector


    sparse_coeff_vector_list, sparsity_mask_list = threshold(coeff_vector_list, sparse_theta_list, time_deriv_list)
    # print("sparse_coeff_vector_list")
    # print(len(sparse_coeff_vector_list))
    # # 1
    # print(sparse_coeff_vector_list[0].shape)
    # # (2,1)
    # print("sparsity_mask_list")
    # print(len(sparsity_mask_list))
    # # 1
    # print(sparsity_mask_list[0].shape)
    # # (2)
    model.fit.sparsity_mask = sparsity_mask_list
    model.fit.coeff_vector = torch.nn.ParameterList(sparse_coeff_vector_list)

    print()
    print("*Primitive*:", coeff_vector_list[0])

    print(sparse_coeff_vector_list)
    # [[0.1062],[-0.8332]] 每次都一样
    print(sparsity_mask_list)
    # [2,4] (u_xx, uu_x)
    # print("fit.mask")
    # print(model.fit.sparsity_mask)
    # print("fit.vector")
    # print(model.fit.coeff_vector)

    #Resetting optimizer for different shapes, train without l1
    optimizer.param_groups[0]['params'] = model.parameters()
    print() #empty line for correct printing
    train(model, data, target, optimizer, max_iterations, dict(loss_func_args, **{'l1': 0.0}))

    return coeff_vector_list[0], sparsity_mask_list
