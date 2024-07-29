#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np

from pycox.models import CoxPH, DeepHitSingle, CoxTime, cox
import torchtuples as tt
from pycox.evaluation import EvalSurv

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import KFold

from model.utils import normalize_log_probs, KL_normal, d_calibration, calculate_silverman_prob, Baseline, calculate_diptest
from model.M4VAE import Encoder, SurvivalDecoder, ObservationDecoder, M4VAE
from model.DeSurv import ODESurvSingle

from sumo import SuMo

# Suppress warning
import warnings
warnings.filterwarnings('ignore', message='To copy construct from a tensor')

class Experiment():
    def __init__(self, df, t_cols, s_cols, c_cols, x_cols,  directory, b_cols = None, a_cols = None):
        if a_cols is not None and b_cols is not None:
            self.df = df.loc[:, t_cols + s_cols + x_cols + c_cols + a_cols + b_cols]
        elif b_cols is not None:
            self.df = df.loc[:, t_cols + s_cols + x_cols + c_cols + b_cols]
        elif a_cols is not None:
            self.df = df.loc[:, t_cols + s_cols + x_cols + c_cols + a_cols]
        
        self.events = s_cols[0]
        self.durations = t_cols[0]
        
        # Save columns indices for future reference
        self.tidx = self.get_column_indices(t_cols)
        self.sidx = self.get_column_indices(s_cols)
        self.cidx = self.get_column_indices(c_cols)
        self.xidx = self.get_column_indices(x_cols)
        
        if a_cols is not None:
            self.aidx = self.get_column_indices(a_cols)
        if b_cols is not None:
            self.bidx = self.get_column_indices(b_cols)
        
        self.c_dim = len(self.cidx)
        self.x_dim = len(self.xidx)
        # u = [a, b]
        
        if a_cols is not None and b_cols is not None:
            self.u_idx = self.aidx + self.bidx
        elif b_cols is not None:
            self.u_idx = self.bidx
        elif a_cols is not None:
            self.u_idx = self.aidx
            
        self.u_dim = len(self.u_idx)
        
        self.encoder_idx = self.xidx + self.cidx + self.u_idx
        self.y_idx = torch.arange(2 + self.x_dim + self.c_dim)
        self.directory = directory
        
        if a_cols is not None and b_cols is not None:
            self.indexs = {"x":self.xidx, "c":self.cidx, "t":self.tidx, "s":self.sidx, "b":self.bidx, "a":self.aidx}
        elif b_cols is not None:
            self.indexs = {"x":self.xidx, "c":self.cidx, "t":self.tidx, "s":self.sidx, "b":self.bidx}
        elif a_cols is not None:
            self.indexs = {"x":self.xidx, "c":self.cidx, "t":self.tidx, "s":self.sidx, "a":self.aidx}
        torch.save(self.indexs, "result/" + self.directory + "indexs")
        torch.save(df, "result/" + self.directory + "df_original")
        
    def get_column_indices(self, col_names):
        indices = []
        for col_name in col_names:
            indices.append(self.df.columns.get_loc(col_name))
        return indices

    def split_df(self, random_state):
        N = self.df.shape[0]

        torch.manual_seed(random_state)
        idx = torch.randperm(N)

        train_idx = idx[: int(N * .8)]
        val_idx = idx[int(N * .8):int(N * .9)]
        test_idx = idx[int(N * .9): ]

        train = self.df.iloc[train_idx, :]
        val = self.df.iloc[val_idx, :]
        test = self.df.iloc[test_idx, :]

        return train, val, test
    
    def set_scaler( self, df, cols_to_std = None ):
        # create a StandardScaler object
        scaler = StandardScaler()

        # fit the scaler
        scaler.fit(df.loc[:, cols_to_std])

        return scaler
    
    def prepare_df(self, cols_to_std = None, random_state = 13):
        
        train, val, test = self.split_df(random_state = random_state)
        self.scaler = None
        
        if cols_to_std != None:
            
            # set-up scaler based on train set
            scaler_ = self.set_scaler(df = train, cols_to_std = cols_to_std)
            
            self.scaler = scaler_
            
            def standardize_columns(df, scaler, cols_to_std):

                # fit and transform the selected columns
                df.loc[:, cols_to_std] = scaler.transform(df.loc[:, cols_to_std])

                return df

            df_train = standardize_columns(train.copy(), scaler_, cols_to_std)
            df_val = standardize_columns(val.copy(), scaler_, cols_to_std)
            df_test = standardize_columns(test.copy(), scaler_, cols_to_std)
            
            torch.save(scaler_, "result/" + self.directory + "scaler")
        
            return df_train, df_val, df_test
        
        return train, val, test
    
    def configure(self, z_dim, hidden_dim, lr, alpha, beta, 
                  device, gamma = 1.0, net_z = None, net_zx = None, net_zc = None, verbose = True):
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        encoder = Encoder(self.encoder_idx, hidden_dim, z_dim, net_z = net_z, device=device, verbose = verbose)

        obs_decoder = ObservationDecoder(self.c_dim, self.x_dim, hidden_dim, z_dim, net_zx = net_zx, net_zc = net_zc,
                         nonlinearity=nn.ReLU,
                         device=device, verbose = verbose)

        surv_decoder = SurvivalDecoder(self.x_dim, hidden_dim, z_dim, 
                         nonlinearity=nn.ReLU,
                         device=device, n = 15, verbose = verbose)

        vae = M4VAE(encoder, obs_decoder, surv_decoder, cov_dim = self.u_dim, lr=lr, alpha = alpha, beta = beta, gamma = gamma)
        
        base_config = {"encoder_idx": self.encoder_idx, "c_dim":self.c_dim, "x_dim": self.x_dim, 
                       "hidden_dim":hidden_dim, "cov_dim":self.u_dim, "lr":lr, "device":device}
        
#         torch.save(base_config, f"result/" + self.directory +"base_config")
        
        return vae
    
    def configure_baseline_nets(self):
    
        # of features: c, x, a, b
        self.in_features = self.df.shape[1] - 2 
        
        # deepsurv network
        ds_net = self.baseline_net(self.in_features, self.hidden_dim, 1)
        
        # time-cox network
        ct_net = CoxTimeNet(self.in_features, self.hidden_dim)
        
        # deephit network
        num_durations = 50
        dh_net = self.baseline_net(self.in_features, self.hidden_dim, num_durations)
        
        baseline_net_dict = {"in_features": self.in_features, "hidden_dim": self.hidden_dim, 
                             "num_durations": num_durations}
        
#         torch.save(baseline_net_dict, "result/" + self.directory + "baseline_net_dict")
        
        return ds_net, ct_net, dh_net
    
    def configure_baseline_models(self, lr, optim = 'Adam', device = 'cpu'):
        
        if optim == 'Adam':
            optimiser = tt.optim.Adam
    
        ds_net, ct_net, dh_net = self.configure_baseline_nets()
        
        # Cox
        cox = CoxPH(torch.nn.Linear(self.in_features, 1), optimiser)
        
        # DeepSurv
        deepsurv = CoxPH(ds_net, optimiser)

        # Time-dependent Cox
        coxtime = CoxTime(ct_net, optimiser)

        # DeepHit
        deephit = DeepHitSingle(dh_net, optimiser, alpha=0.2, sigma=0.1)

        # DeSurv
        desurv = ODESurvSingle(lr, self.in_features, self.hidden_dim, device = device)
        
        # SuMo
        sumo = SuMo(layers = [self.hidden_dim], layers_surv = [self.hidden_dim])
        
        return cox, deepsurv, coxtime, deephit, desurv, sumo
            
        
        
    def baseline_net(self, in_features, hidden_dim, out_features, nonlinearity = nn.ReLU):
    
        net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, out_features)
                )

#         nn.init.kaiming_normal_(net[0].weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(net[2].weight, mode='fan_out', nonlinearity='relu')
        
        return net
    

    def get_t_eval(self, df_test, n_eval = 100):
        t_test = df_test[self.durations]
        t_eval = np.linspace(np.amin(t_test), np.amax(t_test), n_eval)
        
        return t_eval

    def predict(self, model, Y, t_eval, k_peaks):
        # Make predictions across test_time
        
        with torch.no_grad():        
            mu_z, sigma_z = model.encoder(Y[:, torch.tensor(model.encoder_idx)])
#             print(mu_z.shape)
            silverman = calculate_silverman_prob(mu_z.detach().numpy(), k_peaks)
            dip, pval = calculate_diptest(mu_z.detach().numpy())
        
            t_ = torch.tensor(np.concatenate([t_eval]*Y.shape[0], 0),dtype=torch.float32)
            x_ = torch.tensor(np.repeat(mu_z, [t_eval.size]*mu_z.shape[0], axis=0), dtype=torch.float32)
            x = torch.cat([t_[:, None], x_], 1)
            
            pred = pd.DataFrame(np.transpose((1 - model.surv_decoder.forward(x).reshape((mu_z.shape[0],t_eval.size))).detach().numpy()),
                            index=t_eval)
            
        
        duration = Y[:, 0].detach().numpy() 
        event = Y[:, 1].detach().numpy()

        # Evaluate the performance across equally-spaced test_time
        ev = EvalSurv(pred, duration, event, 'km')
        c_idx = ev.concordance_td()
        brier = ev.integrated_brier_score(t_eval)
        nbll = ev.integrated_nbll(t_eval)
        lik = (model.calculate_survival_lik(Y) / Y.shape[0]).item()


        return c_idx, brier, nbll, lik, pred, silverman, dip, pval
    
    def compute_d_cal(self, model, Y, df_test):
        # retrive durations and events
        t_test = df_test[self.durations].values
        s_test = df_test[self.events].values
        
        # Make predictions across test_time
        with torch.no_grad():        
            mu_z, sigma_z = model.encoder(Y[:, torch.tensor(model.encoder_idx)])
            x = torch.cat([torch.tensor(t_test, dtype = torch.float32)[:, None], mu_z], 1)
            pred = model.surv_decoder.forward(x)
        
        return d_calibration(points = pred.flatten(), is_alive = 1 - torch.tensor(s_test, dtype = torch.float32))
            
            
    def run_evaluations_by_fold(self, cols_to_std, k_peaks, num_folds = 5, v_prop = 0.1, random_state = 13, 
                                hidden_dim = 64, z_dim = 2,  alpha = 0.7, beta = 1.0, gamma = [1.0, 10.0],
                                lr = 1e-3, n_epochs = 500, batch_size = 512, logging_freq = 10, max_wait = 20, 
                                 device = 'cpu', return_baseline = True, verbose = True, cox = False):
        """
            This code runs the M4VAE with given hyperparameter and other baseline methods using K-fold train-test split
            
            Args:
                return_baseline: True -> run K-fold for other baseline methods.
                num_folds: # of data partitions
                random_state: random_seed pertaining to partitions
            
            Returns:
                eval_dict(str:dict(str:list(double)))
                    each key corresponds to evaluation metric.
                    each evaluation metric contains dictionary (model: their values across K iterations)
                surv_dict dict(str: pd.DataFrame)
                    pd.DataFrame (T * N): each column contains estimated survival probailities across time
                    surv_dict['Test']: pd.DataFrame - Test set
                    The above is result from 0th iteration. (we don't save 10 different result for memory issue)
        """
        
        
        def standardize_columns(df, scaler, cols_to_std):

            # fit and transform the selected columns
            df.loc[:, cols_to_std] = scaler.transform(df.loc[:, cols_to_std])

            return df
        
        KF = KFold(n_splits = num_folds, random_state = random_state, shuffle = True)
        
        # Create a container
        surv_dict = {}
        eval_dict = {'c_idx': defaultdict(list), 'brier': defaultdict(list), 
                     'nbll': defaultdict(list), 'loglik': defaultdict(list), 'd_cal':defaultdict(list)
                    ,'silverman': defaultdict(list), 'dip': defaultdict(list), 'dippval': defaultdict(list)}
        

        # Perform K-fold train-test set split.
        j = 0
        for train_idx, test_idx in KF.split(self.df):
            print(f"{j}th train-test partition-based training is under way...")
            
            print(f"Use {v_prop} of training set as a validation set to implement an early-stopping")
            vsize = int(v_prop*len(train_idx))
            
            training_idx, validation_idx = train_idx[:-vsize], train_idx[-vsize:]
   
            train, validation, test = self.df.iloc[training_idx], self.df.iloc[validation_idx], self.df.iloc[test_idx]
            
            # Save only 0th test set
            if j == 0:
                surv_dict['Test'] = test
            
            self.scaler = None
            
            # fit scaler based on train set
            scaler_ = self.set_scaler(df = train, cols_to_std = cols_to_std)

            self.scaler = scaler_

            df_train = standardize_columns(train.copy(), scaler_, cols_to_std)
            df_test = standardize_columns(test.copy(), scaler_, cols_to_std)
            df_val = standardize_columns(validation.copy(), scaler_, cols_to_std)
            
            x_train = torch.tensor(df_train.values, dtype=torch.float32)
            x_test = torch.tensor(df_test.values, dtype=torch.float32)
            x_val = torch.tensor(df_val.values, dtype=torch.float32)

            train_dataset = TensorDataset(x_train)
            val_dataset = TensorDataset(x_val)
            
            data_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
            data_loader_val = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
            
            if cox:
                self.hidden_dim = hidden_dim
                
                cox, deepsurv, coxtime, deephit, desurv, sumo = self.configure_baseline_models(lr = lr, 
                                                                                        optim = 'Adam', device = 'cpu')

                for comparator, comparator_name in zip([cox], ['cox']):

                    baseline = Baseline(comparator, comparator_name, events = self.events, durations = self.durations)
                    print(f"\nTraining: {comparator_name}")
                    baseline.fit(df_train, df_val, epochs = n_epochs, max_wait = max_wait, verbose = verbose)
                    
                    t_eval = self.get_t_eval(df_test)

                    c_idx, brier, nbll, lik_, d_cal, base_surv, silverman, dip, dip_pval = baseline.evaluate(df_test, 
                                                                                              t_eval = t_eval, lik = True, 
                                                                                              training = True, k_peaks = k_peaks)
                    eval_dict['c_idx'][comparator_name].append(c_idx)
                    eval_dict['brier'][comparator_name].append(brier)
                    eval_dict['nbll'][comparator_name].append(nbll)
                    eval_dict['loglik'][comparator_name].append(lik_)
                    eval_dict['d_cal'][comparator_name].append(d_cal.detach().item())

                    if j == 0:
                        surv_dict[comparator_name] = base_surv
            else:
                
                for gam in gamma:
                    # Run the model multiple times
                    print(f"\nTraining: M4VAE")
                    model = self.configure(z_dim = z_dim, hidden_dim = hidden_dim, 
                                           lr = lr, alpha = alpha, beta = beta, gamma = gam, device = device, verbose = verbose)

                    state_dict = model.optimize(data_loader, y_idx = self.y_idx, u_idx = self.u_idx,
                               n_epochs = n_epochs, logging_freq = logging_freq, data_loader_val=data_loader_val,
                               max_wait = max_wait, verbose = verbose)

                    # Save outcome
                    gamma_name = "{:02.0f}".format(gam*10)

                    t_eval = self.get_t_eval(df_test)

                    c_idx, brier, nbll, lik_, m4vae_surv, silverman, dip, dip_pval = self.predict(model, x_test, t_eval, k_peaks)

                    d_cal = self.compute_d_cal(model, x_test, df_test)

                    eval_dict['c_idx'][f'M4VAE_{gamma_name}'].append(c_idx)
                    eval_dict['brier'][f'M4VAE_{gamma_name}'].append(brier)
                    eval_dict['nbll'][f'M4VAE_{gamma_name}'].append(nbll)
                    eval_dict['loglik'][f'M4VAE_{gamma_name}'].append(lik_)
                    eval_dict['d_cal'][f'M4VAE_{gamma_name}'].append(d_cal.detach().item())
                    eval_dict['silverman'][f'M4VAE_{gamma_name}'].append(silverman)
                    eval_dict['dip'][f'M4VAE_{gamma_name}'].append(dip)
                    eval_dict['dippval'][f'M4VAE_{gamma_name}'].append(dip_pval)


                    # Save only 0th test set
                    if j == 0:
                        surv_dict[f'M4VAE_{gamma_name}'] = m4vae_surv


                if return_baseline:
                    cox, deepsurv, coxtime, deephit, desurv, sumo = self.configure_baseline_models(lr = lr, 
                                                                                        optim = 'Adam', device = 'cpu')

                    for comparator, comparator_name in zip([deepsurv, coxtime, deephit, desurv, sumo], 
                                                           ['deepsurv', 'coxtime', 'deephit', 'desurv', 'sumo']):

                        baseline = Baseline(comparator, events = self.events, durations = self.durations)
                        print(f"\nTraining: {comparator_name}")
                        baseline.fit(df_train, df_val, epochs = n_epochs, max_wait = max_wait, verbose = verbose)

                        c_idx, brier, nbll, lik_, d_cal, base_surv, silverman, dip, dip_pval = baseline.evaluate(df_test, 
                                                                                                  t_eval = t_eval, lik = True, 
                                                                                                  training = True, k_peaks = k_peaks)
                        eval_dict['c_idx'][comparator_name].append(c_idx)
                        eval_dict['brier'][comparator_name].append(brier)
                        eval_dict['nbll'][comparator_name].append(nbll)
                        eval_dict['loglik'][comparator_name].append(lik_)
                        eval_dict['d_cal'][comparator_name].append(d_cal.detach().item())
                        eval_dict['silverman'][comparator_name].append(silverman)
                        eval_dict['dip'][comparator_name].append(dip)
                        eval_dict['dippval'][comparator_name].append(dip_pval)

                        if j == 0:
                            surv_dict[comparator_name] = base_surv

                j += 1
        if cox:
            torch.save(eval_dict, "result/" + self.directory + f"eval_dict_{num_folds}_fold_cox")
            torch.save(surv_dict, "result/" + self.directory + f"surv_dict_{num_folds}_fold_cox")
        else:
            torch.save(eval_dict, "result/" + self.directory + f"eval_dict_{num_folds}_fold")
            torch.save(surv_dict, "result/" + self.directory + f"surv_dict_{num_folds}_fold")
    
    def train(self, cols_to_std, z_dim = 10, hidden_dim = 64, alpha = 0.2, beta = 1.0, gamma = 1.0, lr = 1e-3, n_epochs = 1000, batch_size = 512, logging_freq = 10, max_wait = 20, device = 'cpu', k_peaks = 1, verbose = True, return_baseline = False, return_metrics = False):
        assert isinstance(gamma, list) or isinstance(gamma, float), "gamma must be either list or float"
        
        if isinstance(gamma, float):
            gamma_list = [gamma]
        else:
            gamma_list = gamma
        
        df_train, df_val, df_test = self.prepare_df(cols_to_std)
        
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
            
        x_test = torch.tensor(df_test.values, dtype=torch.float32)

        x_train = torch.tensor(df_train.values, dtype=torch.float32)
        x_val = torch.tensor(df_val.values, dtype=torch.float32)


        train_dataset = TensorDataset(x_train)
        val_dataset = TensorDataset(x_val)

        data_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        data_loader_val = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        
        torch.save(df_test, "result/" + self.directory + "df_test")
        torch.save(df_val, "result/" + self.directory + "df_val")
        torch.save(df_train, "result/" + self.directory  + "df_train")
        torch.save(self.indexs, "result/" + self.directory  + "indexs")
        
        if return_metrics:
            # Create a container
            surv_dict = {}
            eval_dict = {'c_idx': defaultdict(list), 'brier': defaultdict(list), 
                         'nbll': defaultdict(list), 'loglik': defaultdict(list), 'd_cal':defaultdict(list)
                        ,'silverman': defaultdict(list), 'dip': defaultdict(list), 'dippval': defaultdict(list)}
        
        for gam in gamma_list:
            model = self.configure(z_dim = z_dim, hidden_dim = hidden_dim, lr = lr, 
                                   alpha = alpha, beta = beta, gamma = gam, device = device)

            state_dict = model.optimize(data_loader, y_idx = self.y_idx, u_idx = self.u_idx,
                       n_epochs = n_epochs, logging_freq = logging_freq, data_loader_val=data_loader_val,
                       max_wait = max_wait)

            # Save outcome
            alpha_name = "{:02.0f}".format(alpha*10)
            gamma_name = "{:02.0f}".format(gam*10)
            
            if return_metrics:

                t_eval = self.get_t_eval(df_test)

                c_idx, brier, nbll, lik_, m4vae_surv, silverman, dip, dip_pval = self.predict(model, x_test, t_eval, k_peaks)

                d_cal = self.compute_d_cal(model, x_test, df_test)

                eval_dict['c_idx'][f'M4VAE_{gamma_name}'].append(c_idx)
                eval_dict['brier'][f'M4VAE_{gamma_name}'].append(brier)
                eval_dict['nbll'][f'M4VAE_{gamma_name}'].append(nbll)
                eval_dict['loglik'][f'M4VAE_{gamma_name}'].append(lik_)
                eval_dict['d_cal'][f'M4VAE_{gamma_name}'].append(d_cal.detach().item())
                eval_dict['silverman'][f'M4VAE_{gamma_name}'].append(silverman)
                eval_dict['dip'][f'M4VAE_{gamma_name}'].append(dip)
                eval_dict['dippval'][f'M4VAE_{gamma_name}'].append(dip_pval)
                
                surv_dict[f'M4VAE_{gamma_name}'] = m4vae_surv

            torch.save(model, "result/" + self.directory + f"z{z_dim}a{alpha_name}g{gamma_name}_m4vae")
            torch.save(state_dict, "result/" + self.directory + f"z{z_dim}a{alpha_name}g{gamma_name}_m4vae_state_dict")
        
        if return_baseline:
            cox, deepsurv, coxtime, deephit, desurv, sumo = self.configure_baseline_models(lr = lr, 
                                                                                optim = 'Adam', device = 'cpu')

            for comparator, comparator_name in zip([deepsurv, coxtime, deephit, desurv, sumo], 
                                                   ['deepsurv', 'coxtime', 'deephit', 'desurv', 'sumo']):

                baseline = Baseline(comparator, comparator_name, events = self.events, durations = self.durations)
                print(f"\nTraining: {comparator_name}")
                baseline.fit(df_train, df_val, epochs = n_epochs, max_wait = max_wait, verbose = verbose)
                
                if return_metrics:
                    
                    c_idx, brier, nbll, lik_, d_cal, base_surv, silverman, dip, dip_pval = baseline.evaluate(df_test, 
                                                                                              t_eval = t_eval, lik = True, 
                                                                                              training = True, k_peaks = k_peaks)
                    eval_dict['c_idx'][comparator_name].append(c_idx)
                    eval_dict['brier'][comparator_name].append(brier)
                    eval_dict['nbll'][comparator_name].append(nbll)
                    eval_dict['loglik'][comparator_name].append(lik_)
                    eval_dict['d_cal'][comparator_name].append(d_cal.detach().item())
                    eval_dict['silverman'][comparator_name].append(silverman)
                    eval_dict['dip'][comparator_name].append(dip)
                    eval_dict['dippval'][comparator_name].append(dip_pval)
                    
                    surv_dict[comparator_name] = base_surv

                    
        if return_metrics:
            torch.save(eval_dict, "result/" + self.directory + f"eval_dict")
            torch.save(surv_dict, "result/" + self.directory + f"surv_dict")
        
class CoxTimeNet(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        
        in_features += 1
        out_features = 1
        
        self.net = self.baseline_net(in_features, hidden_dim, out_features)
    
    def baseline_net(self, in_features, hidden_dim, out_features, nonlinearity = nn.ReLU):
    
        net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, out_features)
                )

        nn.init.kaiming_normal_(net[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(net[2].weight, mode='fan_out', nonlinearity='relu')
        
        return net
    
    def forward(self, inputs, time):
        inputs_ = torch.cat([inputs, time], dim = 1)
        return self.net(inputs_) 

