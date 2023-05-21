import torch
import torch.nn as nn
from utils.utils import K2M
from dataset.moving_mnist import MovingMNIST

# Create PDENet Block
class PhyCell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))


    def forward(self, x): # x [N, W, H] or [N, 1, W, H]
        x_dims = len(x.shape)
        if x_dims == 3:
            n = x.shape[0]
            w = x.shape[1]
            h = x.shape[2]
            # x [N, W, H] -> x [N, 1, W, H]
            x = x.view(n, 1, w, h)
            x_r = self.F(x)
            next_hidden = x_r.view(n, w, h)

        if x_dims == 4:
            n = x.shape[0]
            w = x.shape[2]
            h = x.shape[3]
            x_r = self.F(x)
            next_hidden = x_r.view(n, 1, w, h)

        return next_hidden

class PDENet():
  def __init__(self, cfg):
      self.cfg = cfg
      self.device = torch.device(self.cfg["device"])
    
  def train(self):
      mm = MovingMNIST(root=self.cfg["root"], is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
      train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0)
      PhyModel = PhyCell().to(self.device)
        
      # Moment regularization
      constraints = torch.zeros((49,7,7)).to(self.device)
      ind = 0
      for i in range(0,7):
          for j in range(0,7):
              constraints[ind,i,j] = 1
              ind +=1
            
      train_losses = []
      encoder_optimizer = torch.optim.Adam(PhyModel.parameters(),lr=0.001)
      scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2,factor=0.1,verbose=True)
      criterion = nn.MSELoss()
    
      for epoch in range(0, self.cfg["epoch"]):
          t0 = time.time()
          loss_epoch = 0
          teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003)
          for i, out in enumerate(train_loader, 0):
              input_tensor = out[1].to(device)
              target_tensor = out[2].to(device)
              input_length  = input_tensor.size(1)
              target_length = target_tensor.size(1)
              encoder_optimizer.zero_grad()
              loss = 0
              for ei in range(input_length-1): 
                  encoder_output, encoder_hidden, output_image,_,_ = encoder(input_tensor[:,ei,:,:,:], (ei==0) )
                  loss += criterion(output_image,input_tensor[:,ei+1,:,:,:])

              decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
              use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
              for di in range(target_length):
                  decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
                  target = target_tensor[:,di,:,:,:]
                  loss += criterion(output_image,target)
                  if use_teacher_forcing:
                      decoder_input = target # Teacher forcing    
                  else:
                      decoder_input = output_image
             
              k2m = K2M([7, 7])
              for b in range(0, PhyModel.input_dim):
                  filters = PhyModel.F.conv1.weight[:, b, :, :] # (nb_filters,7,7)
                  m = k2m(filters.double())
                  m = m.float()
                  # constrains is a precomputed matrix
                  loss += criterion(m, constraints)
              loss.backward()
              encoder_optimizer.step()
              loss_epoch += (loss.item() / target_length)
          train_losses.append(loss_epoch) 
          if (epoch+1) % print_every == 0:
              print('epoch ',epoch,  ' loss ',loss_epoch, ' time epoch ',time.time()-t0)
            
          if (epoch+1) % eval_every == 0:
              mse, mae,ssim = evaluate(encoder) 
              scheduler.step(mse)                   
              torch.save(encoder.state_dict(),'save/encoder_{}.pth'.format(name))                           
      return train_losses

  def evaluate(encoder):
      mm = MovingMNIST(root=args.root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
      test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=False, num_workers=0)
      total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
      t0 = time.time()
      with torch.no_grad():
          for i, out in enumerate(test_loader, 0):
              input_tensor = out[1].to(device)
              target_tensor = out[2].to(device)
              input_length = input_tensor.size()[1]
              target_length = target_tensor.size()[1]
                
              for ei in range(input_length-1):
                  encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))
              
              decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
              predictions = []
              
              for di in range(target_length):
                  decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input, False, False)
                  decoder_input = output_image
                  predictions.append(output_image.cpu())
                    
              input = input_tensor.cpu().numpy()
              target = target_tensor.cpu().numpy()
              predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
              predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)

              mse_batch = np.mean((predictions-target)**2 , axis=(0,1,2)).sum()
              mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1,2)).sum() 
              total_mse += mse_batch
              total_mae += mae_batch
            
              for a in range(0,target.shape[0]):
                  for b in range(0,target.shape[1]):
                      total_ssim += ssim(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1]) 
      
              cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
              cross_entropy = cross_entropy.sum()
              cross_entropy = cross_entropy / (args.batch_size*target_length)
              total_bce +=  cross_entropy
        
      print('eval mse ', total_mse/len(test_loader),  ' eval mae ', total_mae/len(test_loader),' eval ssim ',total_ssim/len(test_loader), ' time= ', time.time()-t0)        
      return total_mse/len(test_loader),  total_mae/len(test_loader), total_ssim/len(test_loader)
