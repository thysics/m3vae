from model.silverman_test import prob_silverman
from sklearn.decomposition import PCA
import diptest
from scipy.spatial.distance import pdist, squareform

import torch
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from pycox.evaluation import EvalSurv
import torchtuples as tt
import pandas as pd

import numpy as np
import torch

# a dict to store the activations
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def d_calibration(points, is_alive, nbins=10, differentiable=False, gamma=1.0, device='cpu'):
    # each "point" in points is a time for datapoint i mapped through the model CDF
    # each such time_i is a survival time if not censored or a time sampled uniformly in (censor time, max time
    # compute empirical cdf of cdf-mapped-times
    # Move censored points with cdf values greater than 1 - 1e-4 t0 uncensored group
    new_is_alive = is_alive.detach().clone()
    new_is_alive[points > 1. - 1e-4] = 0

    points = points.to(device).view(-1,1)
    # print(points[:200])
    # BIN DEFNITIONS
    bin_width = 1.0/nbins
    bin_indices = torch.arange(nbins).view(1,-1).float().to(device)
    bin_a = bin_indices * bin_width #+ 0.02*torch.rand(size=bin_indices.shape)
    noise = 1e-6/nbins*torch.rand(size=bin_indices.shape).to(device)
    if not differentiable:
        noise = noise * 0.
    cum_noise = torch.cumsum(noise, dim=1)
    bin_width = torch.tensor([bin_width]*nbins).to(device) + cum_noise
    bin_b = bin_a + bin_width

    bin_b_max = bin_b[:,-1]
    bin_b = bin_b/bin_b_max
    bin_a[:,1:] = bin_b[:,:-1]
    bin_width = bin_b - bin_a

    # CENSORED POINTS
    points_cens = points[new_is_alive.long()==1]
    upper_diff_for_soft_cens = bin_b - points_cens
    # To solve optimization issue, we change the first left bin boundary to be -1.;
    # we change the last right bin boundary to be 2.
    bin_b[:,-1] = 2.
    bin_a[:,0] = -1.
    lower_diff_cens = points_cens - bin_a # p - a
    upper_diff_cens = bin_b - points_cens # b - p
    diff_product_cens = lower_diff_cens * upper_diff_cens
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        bin_index_ohe = torch.sigmoid(gamma * diff_product_cens)
        exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)
    else:
        # (p-a)*(b-p)
        bin_index_ohe = (lower_diff_cens >= 0).float() * (upper_diff_cens > 0).float()
        exact_bins_next = (lower_diff_cens <= 0).float()  # all bins after correct bin

    EPS = 1e-13
    right_censored_interval_size = 1 - points_cens + EPS

    # each point's distance from its bin's upper limit
    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)

    # assigns weights to each full bin that is larger than the point
    # full_bin_assigned_weight = exact_bins*bin_width
    # 1 / right_censored_interval_size is the density of the uniform over [F(c),1]
    full_bin_assigned_weight = (exact_bins_next*bin_width.view(1,-1)/right_censored_interval_size.view(-1,1)).sum(0)
    partial_bin_assigned_weight = (upper_diff_within_bin/right_censored_interval_size).sum(0)
    assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

    # NON-CENSORED POINTS
    points_uncens = points[new_is_alive.long() == 0]
    # compute p - a and b - p
    lower_diff = points_uncens - bin_a
    upper_diff = bin_b - points_uncens
    diff_product = lower_diff * upper_diff
    assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
    assert lower_diff.shape == (points_uncens.shape[0], bin_a.shape[1])
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        soft_membership = torch.sigmoid(gamma*diff_product)
        fraction_in_bins = soft_membership.sum(0)
        # print('soft_membership', soft_membership)
    else:
        # (p-a)*(b-p)
        exact_membership = (lower_diff >= 0).float() * (upper_diff > 0).float()
        fraction_in_bins = exact_membership.sum(0)

    assert fraction_in_bins.shape == (nbins, ), fraction_in_bins.shape

    frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) /points.shape[0]
    return torch.pow(frac_in_bins - bin_width, 2).sum()

def KL_standard_normal(mu, sigma):
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    q = Normal(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))

def KL_normal(mu, sigma, mu_prior, sigma_prior):
    p = Normal(mu_prior, sigma_prior)
    q = Normal(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))

def KL_laplace(mu, sigma, mu_prior, sigma_prior):
    p = Laplace(mu_prior, sigma_prior)
    q = Laplace(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))

def normalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized log probabilities or normalized probabilities from unnormalized log probabilities.
    :param log_probs: unnormalized log probabilities
    :return: normalized log probabilities and normalized probabilities
    """
    # use the logsumexp trick to compute the normalizing constant
    log_norm_constant = torch.logsumexp(log_probs, dim = 1, keepdim= True)

    # subtract the normalizing constant from the unnormalized log probabilities to get the normalized log probabilities
    norm_log_probs = log_probs - log_norm_constant

    return norm_log_probs, torch.exp(norm_log_probs)


def expand_grid(a, b):
    nrow_a = a.size()[0]
    nrow_b = b.size()[0]
    ncol_b = b.size()[1]
    x = a.repeat(nrow_b, 1)
    y = b.repeat(1, nrow_a).view(-1, ncol_b)
    return x, y

def calculate_silverman_prob(x_test, k_peaks = 1):
    # Example data: replace with your dataset
    # Each row represents a sample; each column represents a feature

    # Create PCA object to get the first principal component
    pca = PCA(n_components=1)

    # Fit PCA on your data
    pca.fit(x_test)

    # Transform the data
    X_pca = pca.transform(x_test)


    # Array used to search for critical widths
    # It depends on the typical values of the data
    # If it doesn't cover the required values, a message is shown.
    widths = np.linspace(0, X_pca.max()+3 * X_pca.std(), 1000) 

    # First find probabilities with the non-callibrated original Silverman test:
    try:
        P_silverman = prob_silverman(X_pca.flatten(), k_peaks, widths=widths, N_boots=1000)
    except:
        print("Error: silverman test can not be completed.")
        return np.nan
    
    return P_silverman.mean()

def calculate_diptest(x_test, iterations = 100):
    
    pairwise = pdist(x_test)
    
    if pairwise.shape[0] > 72000:
        print("Use sample mean of pval for dip_pval")
        dip_pvals = []
        dip_stats = []
        for i in range(iterations):
            # Select a random subsample of 10% of the data
            idx = np.random.choice(x_test.shape[0], size=int(x_test.shape[0] * 0.1), replace=False)
            
            # Compute pairwise distances
            pair_wise = pdist(x_test[idx, :])
            
            dip_s, dip_p = diptest.diptest(pair_wise)
            dip_pvals.append(dip_p)
            dip_stats.append(dip_s)
            
        dip_stat = np.mean(dip_stats)
        dip_pval = np.mean(dip_pvals)
        
    else:
        dip_stat, dip_pval = diptest.diptest(pairwise)
    
    return dip_stat, dip_pval


class ApproximateLikelihood:
    '''
    Enter a model, covariates, times and events
    model: a pycox_local model
    x: 2d numpy array
    t: 1d numpy array
    d: 1d numpy array - event indicator
    half_width: k >=1 and densities are evaluated using T_{i-k+1} and T_{i+k}
    '''

    def __init__(self, model, x, t, d, half_width = 2):
        self.model = model
        self.t = t
        self.d = d
        self.x = x
        self.n = len(self.t)
        self.half_width = int(half_width)
        self.mask_observed = self.d == 1
        self.densities = None
        self.survival = None
        self.log_likelihood = None

    def drop_outliers(self, min_time, max_time):

        # Select the outliers and reset self.x, t, d and n
        outlier_mask = (self.t > max_time) or (self.t < min_time)
        self.t = self.t[~outlier_mask]
        self.d = self.d[~outlier_mask]
        if isinstance(self.x, tuple):
            self.x = tt.tuplefy((self.x[0][~outlier_mask],self.x[1][~outlier_mask]))
        else:
            self.x = self.x[~outlier_mask]
        self.n = len(self.t)

        return None

    def get_densities(self,surv_df_raw=None):

        # Get the survival dataframe for x_observed, drop duplicate rows
        # if duplicates, it implies that no one went through the event of interest at that time.
        if surv_df_raw is None:
            if isinstance(self.x,tuple):
                input= tt.tuplefy((self.x[0][self.mask_observed], self.x[1][self.mask_observed]))
            else:
                input = self.x[self.mask_observed]
            survival_df_observed = self.model.predict_surv_df(input).drop_duplicates(keep='first') #  T, N
  
        else:
            np_bool = self.mask_observed
            survival_df_observed = surv_df_raw.iloc[np_bool].transpose().drop_duplicates(keep='first')

        assert survival_df_observed.index.is_monotonic
        min_index, max_index = 0, len(survival_df_observed.index.values) - 1

        # Create an Eval object
        eval_observed = EvalSurv(survival_df_observed, self.t[self.mask_observed], self.d[self.mask_observed])
        # Get the indices of the survival_df
        indices = eval_observed.idx_at_times(self.t[self.mask_observed])

        left_index = np.minimum(np.maximum(indices - self.half_width + 1, min_index), max_index - 1).squeeze()
        right_index = np.minimum(indices + self.half_width, max_index).squeeze()

        # Get the survival probabilities and times
        left_survival = np.array([survival_df_observed.iloc[left_index[i], i] for i in range(len(left_index))])
        right_survival = np.array([survival_df_observed.iloc[right_index[i], i] for i in range(len(right_index))])

        
        left_time = np.array(survival_df_observed.index[left_index])
        right_time = np.array(survival_df_observed.index[right_index])

        # Approximate the derivative
        delta_survival = left_survival - right_survival
        delta_t = right_time - left_time
        self.densities = delta_survival / delta_t

        return self.densities

    def get_survival(self,surv_df_raw=None):
        """
            Note that this method works only if there is at least one censored instance
        """
        # Create the survival_df and the Eval object
        if surv_df_raw is None:
            if isinstance(self.x,tuple):
                input= tt.tuplefy((self.x[0][~self.mask_observed], self.x[1][~self.mask_observed]))
            else:
                input = self.x[~self.mask_observed]

            survival_df_censored = self.model.predict_surv_df(input).drop_duplicates(keep='first')
        else:
            np_bool = ~self.mask_observed
            survival_df_censored = surv_df_raw.iloc[:, np_bool].transpose().drop_duplicates(keep='first')
        
        eval_censored = EvalSurv(survival_df_censored, self.t[~self.mask_observed], self.d[~self.mask_observed])
        # Get a list of indices of the censored times
        # Find the closest index to the censored times
        indices = eval_censored.idx_at_times(self.t[~self.mask_observed]).squeeze()

        # Select the survival probabilities
        self.survival = np.array([survival_df_censored.iloc[indices[i], i] for i in range(len(indices))])

        return self.survival

    def estimate_lik(self, surv_df_raw=None, training = True):
        # Get the survival probabilities and the densities
        if training:
            if surv_df_raw is None and self.model.__class__.__name__ != 'DeepHitSingle':
                _ = self.model.compute_baseline_hazards()

        self.get_survival(surv_df_raw)
        self.get_densities(surv_df_raw)

        # Compute the log-likelihood
        self.log_likelihood = np.mean(np.log(np.concatenate((self.survival, self.densities)) + 1e-7))

        return self.log_likelihood




class Baseline():
    """
        params: events (str) - event column name
        params: durations (float) - time-to-event column name  
    """
    def __init__(self, model, model_name, batch_size = 256, events = 'FU_death_status', durations = 'FU_Duration_All'):
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.events = events
        self.durations = durations
        if self.model.__class__.__name__ == "DeepHitSingle":
            self.num_durations = 50
    
    def get_features(self, df):
        return df.loc[:, ~df.columns.isin([self.durations, self.events])].values.astype('float32')
    
    def get_target(self, df):
        func = lambda df: (df[self.durations].values, df[self.events].values)
        return func(df)
    
    def convert_dtypes(self, df):
        df[self.durations] = df[self.durations].astype('float32')
        df[self.events] = df[self.events].astype('int')
        return df
    
    def prepare_dataset(self, df_train, df_val = None):
        
        x_train = self.get_features(df_train)
        
        if df_val is not None:
            x_val = self.get_features(df_val)
        
        
        if self.model.__class__.__name__ in 'CoxPH':
            y_train = self.get_target(df_train)
            if df_val is not None:
                y_val = self.get_target(df_val)
        
        elif self.model.__class__.__name__ == 'SuMo':
            t_train, s_train = self.get_target(df_train)
            
            if df_val is not None:
                t_val, s_val = self.get_target(df_val)
            else:
                x_val = None
                t_val = None
                s_val = None
         
            
            return x_train, t_train, s_train, x_val, t_val, s_val
            
        elif self.model.__class__.__name__ == "DeepHitSingle":
            df_train = self.convert_dtypes(df_train.copy())
            
            if df_val is not None:
                df_val = self.convert_dtypes(df_val.copy())
            
            labtrans = self.model.label_transform(self.num_durations)
            
            y_train = labtrans.fit_transform(*self.get_target(df_train))
            
            if df_val is not None:
                y_val = labtrans.transform(*self.get_target(df_val))
                
            self.model.duration_index = labtrans.cuts

        elif self.model.__class__.__name__ == 'CoxTime':
            labtrans = self.model.label_transform()
            y_train = labtrans.fit_transform(*self.get_target(df_train))
            
            if df_val is not None:
                y_val = labtrans.transform(*self.get_target(df_val))
                
            self.model.labtrans = labtrans
        
        elif self.model.__class__.__name__ == "ODESurvSingle":
            y_train = self.get_target(df_train)
            
            if df_val is not None:
                y_val = self.get_target(df_val)
            
            t_train, s_train = y_train
            
            if df_val is not None:
                t_val, s_val = y_val
            
            dataset_train = TensorDataset(*[torch.tensor(u,dtype=dtype_) for u, dtype_ in [(x_train,torch.float32),
                                                                               (t_train,torch.float32),
                                                                               (s_train,torch.long)]])
            data_loader_train = DataLoader(dataset_train, batch_size=self.batch_size, pin_memory=True, shuffle=True, drop_last=True)
            
            if df_val is not None:
                dataset_val = TensorDataset(*[torch.tensor(u,dtype=dtype_) for u, dtype_ in [(x_val,torch.float32),
                                                                               (t_val,torch.float32),
                                                                               (s_val,torch.long)]])
                data_loader_val = DataLoader(dataset_val, batch_size=self.batch_size, pin_memory=True, shuffle=True)
                
            else:
                data_loader_val = None
            
            return data_loader_train, data_loader_val
        
        if df_val is None:
            x_val = None
            y_val = None
            
        return x_train, y_train, x_val, y_val
    
    def fit(self, df_train, df_val = None, epochs = 56, verbose = True, max_wait = 5):
        
        if self.model.__class__.__name__ == "ODESurvSingle":
            
            data_loader_train, data_loader_val = self.prepare_dataset(df_train, df_val)
            
            self.model.optimize(data_loader_train, n_epochs = epochs, logging_freq = 1,
                                data_loader_val = data_loader_val, max_wait = max_wait, verbose = verbose)
        
        elif self.model.__class__.__name__ == "SuMo":
            x_train, t_train, s_train, x_val, t_val, s_val = self.prepare_dataset(df_train, df_val)
            # The fit method is called to train the model
            
            if df_val is not None:
                self.model.fit(x_train.astype(np.float64), t_train, s_train, n_iter = epochs, bs = 512,
                        lr = 5e-3, val_data = (x_val.astype(np.float64), t_val, s_val))
            else:
                self.model.fit(x_train.astype(np.float64), t_train, s_train, n_iter = epochs, bs = 512,
                        lr = 5e-3, val_data = None)

        else:
            
            x_train, y_train, x_val, y_val = self.prepare_dataset(df_train, df_val)
            
            if self.model.__class__.__name__ in ['CoxPH', 'CoxTime']:
                if df_val is not None:
                    val = tt.tuplefy(x_val, y_val)

            elif self.model.__class__.__name__ == "DeepHitSingle":
                train = (x_train, y_train)
                
                if df_val is not None:
                    val = (x_val, y_val)


            lrfinder = self.model.lr_finder(x_train, y_train, self.batch_size, tolerance = 2)
            self.model.optimizer.set_lr(lrfinder.get_best_lr())
            
            if df_val is not None:
                callbacks = [tt.callbacks.EarlyStopping(patience = max_wait)]

            
                log = self.model.fit(x_train, y_train, self.batch_size, 
                                     epochs, verbose = verbose, callbacks = callbacks, val_data = val)
            else:
                log = self.model.fit(x_train, y_train, self.batch_size, 
                                     epochs, verbose = verbose, callbacks = None, val_data = None)

        
    def get_cif(self, df, t_test):
        """
            Arg:
                df: T * N pd.dataframe - (t,n)th entry refers to nth individual's survival prob at t
                t_test: N-dim np.array - (n)th entry refers to time-to-event of nth individual
            returns cumulative incidence function at t_test times.
            
            return:
            
            Note:
                closet_time_index: find the closest time point with predictive value to the test time.
        """
        T, N = df.shape
        probabilities = []

        for n in range(N):
            
            closest_time_index = abs(df.index - t_test[n]).argsort()[:1][0]
            closest_time = df.index[closest_time_index]
            survival_probability = df.loc[closest_time, n]
            probabilities.append(survival_probability)

        return 1 - torch.tensor(probabilities, dtype = torch.float32)


    def evaluate(self, df_test, k_peaks, t_eval = None, lik = True, training = True):

        y_test = self.get_target(df_test)
        t_test, s_test = y_test
        x_test = self.get_features(df_test)
        
        if t_eval is None:
            n_eval = 100
            t_eval = np.linspace(np.amin(t_test), np.amax(t_test), n_eval)
        
        if self.model.__class__.__name__ in ['CoxPH', 'CoxTime']:
            _ = self.model.compute_baseline_hazards(max_duration = t_eval[-1])
        
        if self.model.__class__.__name__ not in ['ODESurvSingle', 'SuMo']:
            
            
            if self.model.__class__.__name__ in ['CoxTime'] and self.model_name != 'cox':
                h1 = self.model.net.net[-3].register_forward_hook(getActivation(f"{self.model.__class__.__name__}"))
                surv = self.model.predict_surv_df(x_test, max_duration = t_eval[-1])
                h1.remove() 
            elif self.model_name == 'cox':
                surv = self.model.predict_surv_df(x_test) 
            else:
                h1 = self.model.net[-3].register_forward_hook(getActivation(f"{self.model.__class__.__name__}"))
                surv = self.model.predict_surv_df(x_test)   
                h1.remove()            
                
            t_hat = self.get_cif(surv, t_test)
        
        elif self.model.__class__.__name__ == 'ODESurvSingle':
            # Feed dataset with 1000 observations each
            N = df_test.shape[0]
            surv_list = []
            for i in range(int(N / 1000) + 1):
                begin = 1000 * i 
                end = 1000 * (i+1)
                surv_list.append(self.model.predict_surv_df(x_test[begin:end], t_eval))
            surv = pd.concat(surv_list, axis = 1)
            
            h1 = self.model.net.dudt[-3].register_forward_hook(getActivation(f"{self.model.__class__.__name__}"))
            
            t_hat = self.model.predict(torch.tensor(x_test, dtype = torch.float32), torch.tensor(t_test, dtype = torch.float32))
            h1.remove()        

        elif self.model.__class__.__name__ == 'SuMo':
            h1 = self.model.torch_model.outcome[1].register_forward_hook(getActivation(f"{self.model.__class__.__name__}"))
            out_survival = self.model.predict_survival(x_test.astype(np.float64), t_eval.tolist())
            surv = pd.DataFrame(out_survival.T, index = t_eval)
            t_hat = self.get_cif(surv, t_test)
            h1.remove()
            
        ev = EvalSurv(surv, t_test, s_test, censor_surv='km')

        c_idx = ev.concordance_td()
        brier_score = ev.integrated_brier_score(t_eval)
        nbll_score = ev.integrated_nbll(t_eval)
        
        d_cal = d_calibration(points = t_hat, is_alive = 1 - torch.tensor(s_test, dtype = torch.float32).flatten())
        
        if self.model_name != 'cox':
            # Compute cluster-ability by silverman method.
            silverman = calculate_silverman_prob(activation[f"{self.model.__class__.__name__}"], k_peaks)
            dip, dippval = calculate_diptest(activation[f"{self.model.__class__.__name__}"])
        else:
            silverman = None
            dip, dippval = None, None
        
        if lik:
            if self.model.__class__.__name__ not in ['M4VAE', 'ODESurvSingle', 'SuMo']:
                approx = ApproximateLikelihood(self.model, x_test, 
                                      t_test, s_test, half_width = 2)
                lik_mean = approx.estimate_lik(training = training)
                
            elif self.model.__class__.__name__ == 'ODESurvSingle':
                lik_mean = self.model.compute_lik(x_test, t_test, s_test)
            
            elif self.model.__class__.__name__ == 'SuMo':
                lik_mean = -1 * self.model.compute_nll(x_test, t_test, s_test)
            
            return c_idx, brier_score, nbll_score, lik_mean, d_cal, surv, silverman, dip, dippval
                
        return c_idx, brier_score, nbll_score, d_cal, surv, silverman, dip, dippval
    
    def get_t_eval(self, df_test):
        y_test = self.get_target(df_test)
        t_test, s_test = y_test
        
        n_eval = 100
        t_eval = np.linspace(np.amin(t_test), np.amax(t_test), n_eval)
        
        return t_eval
 

