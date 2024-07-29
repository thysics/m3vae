import torch
from torch.nn.functional import softplus
import torch.distributions as dist
import numpy as np
import pandas as pd


class DGP:
    def __init__(self, lambda_1 = 1.5, lambda_2 = 1.0):
        self.a_i = torch.distributions.Uniform(0., 1.)
        self.pi_i = torch.distributions.Categorical(torch.tensor([1/2, 1/2]))
        
        
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        

        
    def generate_samples(self, N, random_state = 13, p_censor = 0.5):
        torch.manual_seed(random_state)
        samples = torch.zeros((N, 6))

        for _ in range(N):
            a = self.a_i.sample((1, ))
            pi = self.pi_i.sample((1,))
            
            idx = int(pi.detach().item())
            x_id = torch.distributions.Bernoulli(torch.tensor([0.3])).sample((1,))
            c_ij = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0).sample((1,))

            scale = (self.lambda_1 - a) * (self.lambda_2 + pi).sum() 
    
#             print(scale)
            T_i = torch.distributions.weibull.Weibull(scale = scale, concentration = 3.0).sample()
                                                
            # event indicator
            s_i = torch.tensor([np.random.binomial(1, 1 - p_censor)])
            
            if s_i > 0:
                t_i = T_i
            else:
                t_i = torch.tensor(np.random.uniform(0, T_i))
#             print(T_i.shape, x_id.shape, c_ij.shape)

            samples[_, :] = torch.hstack([t_i[:, None], s_i[:, None], x_id, c_ij[:, None], a[:, None], pi[:, None]])
            
        col_names = ['durations', 'events', 'x', 'c', 'a', 'b']
        
        return pd.DataFrame(samples, columns = col_names)
    

