import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations, product
from functools import reduce

def library_poly(prediction, max_order):
    # Calculate the polynomes of u
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    return u


def library_deriv(data, prediction, max_order):
    # 返回predication 对 data 的梯度
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    # t在先 x在后
    time_deriv = dy[:, 0:1]
    
    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)
        if max_order >1:
            for order in np.arange(1, max_order):
                du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)

    return time_deriv, du


def library_1D_in(input, poly_order, diff_order):
    prediction, data = input
    poly_list = []
    deriv_list = []
    time_deriv_list = []

    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        # du/dt,du/dx (dv/dt,dv/dx)
        time_deriv, du = library_deriv(data, prediction[:, output:output+1], diff_order)
        u = library_poly(prediction[:, output:output+1], poly_order)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(time_deriv)
    # du/dt的个数为data数量（5000）
    # print("poly_list", len(poly_list), poly_list[0].shape)
    # print("deriv_list", len(deriv_list), deriv_list[0].shape)
    # print("time_deriv_list", len(time_deriv_list))

    samples = time_deriv_list[0].shape[0]
    # poly=3，deriv=3，总为9
    total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]
    
    # Calculating theta
    if len(poly_list) == 1:
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms) # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:

        theta_uv = reduce((lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)), poly_list)
        # print("theta_uv", theta_uv.shape)
        # ks [1,u][1,v]——>1,v,u,uv、4
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1) # calculate all unique combinations of derivatives
        # print("theta_dudv", theta_dudv.shape)
        # ks [1,ux,uxx][1,vx,vxx]——>去掉1, vx,vxx,ux,uxvx,uxvxx,uxx,uxxvx,uxxvxx、8
        theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, dv in product(poly_list, deriv_list)], 1)  # calculate all unique products of polynomials and derivatives
        # print("theta_udu", theta_udu.shape)
        # ks [u][ux,uxx] [v][ux,uxx] [v][vx,vxx] [u][vx,vxx]、8
        theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)
        
    return time_deriv_list, theta

def library_2Din_1Dout(input, poly_order, diff_order):
        '''
        Constructs a library graph in 1D. Library config is dictionary with required terms.
        '''
        prediction, data = input
        # Polynomial
        
        u = torch.ones_like(prediction)
        for order in np.arange(1, poly_order+1):
            u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

        # Gradients
        # 返回predication 对 data 的梯度
        du = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 2:3]
 
        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        # print("du ", du.shape)
        # (1000,6)
        samples= du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples,-1)
        # print("theta ", theta.shape)
        # (1000,12)
        
        return [u_t], theta


def library_2Din_2Dout(input, poly_order, diff_order):
    prediction, data = input
    poly_list = []
    deriv_list = []
    time_deriv_list = []

    # creating lists for u and v
    for output in torch.arange(prediction.shape[1]):
        u = torch.ones_like(prediction[:, output:output+1])
        for order in np.arange(1, poly_order+1):
            u = torch.cat((u, u[:, order-1:order] * prediction[:, output:output+1]), dim=1)
        du = grad(prediction[:, output:output+1], data, grad_outputs=torch.ones_like(prediction[:, output:output+1]), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction[:, output:output+1]), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction[:, output:output+1]), create_graph=True)[0][:, 2:3]
        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(u_t)

    # print("poly_list", len(poly_list), poly_list[0].shape)
    # # 2 (10000,4)
    # print("deriv_list", len(deriv_list), deriv_list[0].shape)
    # # 2 (10000,6)
    # print("time_deriv_list", len(time_deriv_list))
    # # 2

    samples = time_deriv_list[0].shape[0]

    theta_uv = reduce((lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)), poly_list)
    # print("theta_uv", theta_uv.shape)
    # # [1,16]
    theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1)
    # print("theta_dudv", theta_dudv.shape)
    # # [1,35]
    theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, du in product(poly_list, deriv_list)], 1)
    # print("theta_udu", theta_udu.shape)
    # # [1,60]
    theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)
    # print("theta", theta.shape)
    # # [1,111]

    return time_deriv_list, theta
