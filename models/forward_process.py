# Forward Process
import torch

def beta(timesteps, start = 0.0001, end = 0.02):
    return torch.linspace(start, end, timesteps)

class ForwardProcess():
    def __init__(self, T):
        self.T = T
        alpha = torch.cumprod(1. - beta(T), axis = 0) 
        self.minus_alphas = torch.sqrt(1 - alpha)
        self.alphas = torch.sqrt(alpha)
        
    def get_all_alphas(self, t, shape, minus = True):
        if minus:
            out = self.minus_alphas.gather(-1, t)
        else:
            out = self.alphas.gather(-1, t)
        return out.reshape(t.shape[0], *((1, ) * (len(shape) - 1)))
    
    def NoisePredictor(self, x, timesteps):
        noise = torch.randn_like(x).to(x.device)
        c_x = self.get_all_alphas(timesteps, x.shape, False).to(x.device)
        #print(c_x.shape)
        c_n = self.get_all_alphas(timesteps, x.shape, True).to(x.device)
        return (c_x * x + c_n * noise).to(x.device)