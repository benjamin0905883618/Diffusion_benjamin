# Forward Process
import torch
import torch.nn.functional as F

def beta(timesteps, start = 0.0001, end = 0.02):
    return torch.linspace(start, end, timesteps)

class ForwardProcess():
    def __init__(self, T):
        self.T = T
        self.betas = beta(T)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, axis = 0)
        self.cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value = 1.0)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_alpha_cumprod_minus = torch.sqrt(1. - self.alpha_cumprod)
        self.posterior_variance = self.betas * (1. - self.cumprod_prev) / (1. - self.alpha_cumprod)
        
    def get_all_alphas(self, t, shape, minus = True):
        if minus:
            out = self.sqrt_alpha_cumprod_minus.gather(-1, t)
        else:
            out = self.sqrt_alpha_cumprod.gather(-1, t)
        return out.reshape(t.shape[0], *((1, ) * (len(shape) - 1)))
    
    def NoisePredictor(self, x, timesteps):
        noise = torch.randn_like(x).to(x.device)
        c_x = self.get_all_alphas(timesteps, x.shape, False).to(x.device)
        #print(c_x.shape)
        c_n = self.get_all_alphas(timesteps, x.shape, True).to(x.device)
        return (c_x * x + c_n * noise).to(x.device), noise.to(x.device)
    
    def output_sample(self, model, x, t):
        betas_t = self.betas.gather(-1, t).reshape(t.shape[0], *((1, ) * (len(x.shape) - 1))).to(x.device)
        cn = self.get_all_alphas(t, x.shape).to(x.device)
        cx = self.sqrt_recip_alpha.gather(-1, t).reshape(t.shape[0], *((1, ) * (len(x.shape) - 1))).to(x.device)
        model_mean = cx * (x - betas_t * model(x, t.to(x.device)) / cn)
        posterior = self.posterior_variance.gather(-1, t).reshape(t.shape[0], *((1, ) * (len(x.shape) - 1))).to(x.device)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            #print(torch.sqrt(self.posterior_variance).shape)
            return model_mean + torch.sqrt(posterior) * noise
    
    
