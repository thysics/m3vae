#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import matplotlib.pylab as plt
from matplotlib.cm import ScalarMappable
from kmeans_pytorch import kmeans, kmeans_predict
import seaborn as sns

class present_syn:
    def __init__(self, data_dir, model_dir, directory, durations = "durations", events = "events"):
        self.model = torch.load('result/' + data_dir + model_dir + '_m4vae')
        
        print("Loaded model")
        print(f"z_dim: {self.model.encoder.z_dim}, " + f"alpha: {self.model.alpha}, " + f"beta: {self.model.beta}")
        
        self.df_test = torch.load('result/' + data_dir + 'df_test')
        self.df_train = torch.load('result/' + data_dir + 'df_train')

        self.X_train = torch.tensor(self.df_train.values, dtype = torch.float32)
        self.X_test = torch.tensor(self.df_test.values, dtype = torch.float32)
        self.indexs = torch.load('result/' + data_dir + 'indexs')
        self.scaler = torch.load('result/' + data_dir + 'scaler')
        self.directory = directory
        
        self.durations = durations
        self.events = events
        
        self.clusters = None
    
    def project_via_umap(self, latent_vars, n_neighbors = 30, random_state = 13, metric = 'euclidean', min_dist = 0):
        def train_umap(train_latent_vars,
               n_neighbors=n_neighbors, random_state = random_state, metric = metric, min_dist = min_dist):
    
            reducer = umap.UMAP(n_components= 2, n_neighbors=n_neighbors, 
                                random_state = random_state, min_dist = 0, metric = metric)
            reducer.fit(train_latent_vars)

            return reducer
        
        trained_umap = train_umap(train_latent_vars = self.train_latent_vars.detach().numpy(),
                                  n_neighbors = n_neighbors, random_state = random_state, metric = metric, min_dist = min_dist)
        
        return trained_umap.transform(latent_vars)
    
    def get_latent_vars(self, metric = 'euclidean', n_neighbors = 30):
        
        self.test_latent_vars = self.model.get_latent_vars(torch.tensor(self.df_test.values, dtype = torch.float32), mean = True)
        self.train_latent_vars = self.model.get_latent_vars(torch.tensor(self.df_train.values, dtype = torch.float32), mean = True)
        
        # Umap indicator
        self.UMAP = False
        
        if self.model.encoder.z_dim < 3:
            self.test_latent_space = self.test_latent_vars.detach().numpy()
            self.train_latent_space = self.train_latent_vars.detach().numpy()
            self.notation_x = "z"
        else:
            self.test_latent_space = self.project_via_umap(self.test_latent_vars.detach().numpy(), metric = metric, n_neighbors = n_neighbors)
            self.train_latent_space = self.project_via_umap(self.train_latent_vars.detach().numpy(), metric = metric, n_neighbors = n_neighbors)
            self.notation_x = "u"
            self.UMAP = True
        
        if self.UMAP:
            print(f"UMAP applied: metric = {metric}, n_neighbors = {n_neighbors}")
        else:
            print("UMAP has not been applied.")
    
    def plot_loss(self, dataset = 'validation', start = 0, end = None):
        if dataset == "train":
            losses = self.model.train_loss
        
        elif dataset == "validation":
            losses = self.model.val_loss
        else:
            raise ValueError("dataset (str) must be either validation or train")
            
        N = len(losses.keys())

        fig, axs = plt.subplots(nrows = 1, ncols = N, figsize = (5 * N, 5))
        for i, item in enumerate(losses.keys()):
            component = item[:5]
            IT = len(losses[item]) - 1
            if end is not None:
                IT = end
            axs[i].scatter(np.arange(start, IT), losses[item][start:IT])
            axs[i].set_title(item)
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel("Loss")
        
        plt.tight_layout()
        plt.savefig("figure/" + self.directory + "optimization_step_" + f"{dataset}", bbox_inches='tight', dpi=300)
        plt.show()
    
    def mean_by(self, input_df, by = "b"):
        # Get the column order of the input dataframe
        col_order = input_df.columns

        # Group the input dataframe by column "b" and calculate mean for other columns
        output_df = input_df.groupby("b").mean().reset_index()

        # Reorder the columns of the output dataframe to match the input dataframe column order
        output_df = output_df[col_order]
        
        output_tensor = torch.tensor(output_df.values, dtype = torch.float32)
        return output_tensor
            
    def present_cluster(self, dataset = "test", cls_mean = False, by = "b", K = 2):
        """
            This function is suited for the experiment on synthetic data
        """
        if dataset == "test":
            df = self.df_test.reset_index(drop = True)
            latent_vars = self.test_latent_space
        
        elif dataset == "train":
            df = self.df_train.reset_index(drop = True)
            latent_vars = self.train_latent_space
        else:
            raise ValuerError("dataset (str) must be either test or train")
        
        if cls_mean:
            cluster_mean = self.mean_by(df, by = by)
            print(f"Clustering data by {by}")
            cls_mean_latent_vars = self.model.get_latent_vars(cluster_mean)
            print("Means of (synthetic) clusters had been loaded")
            if self.UMAP:
                print("UMAP applied to (synthetic) cluster mean latent variables")
                cls_mean_latent_vars = torch.tensor(self.project_via_umap(cls_mean_latent_vars), dtype = torch.float32)
        
        title = "Latent Embedding: " + dataset + " set/ " + f"z_dim: {self.model.encoder.z_dim}, " + f"alpha: {self.model.alpha}, " + f"beta: {self.model.beta}"
        
        if self.UMAP:
            title = "(UMAP) " + title
        
        ref_idx = df.iloc[:, self.indexs[by]]

        colors = sns.color_palette('hls', K)
        fig, axs = plt.subplots(nrows = 1, ncols = K+1, figsize = (16, 4), sharex='all', sharey='all')

        for n, ax in enumerate(axs.ravel()):
            if n > 0:
                temp_vars = latent_vars[(ref_idx == n-1).values.flatten(), :]
                ax.scatter(temp_vars[:, 0], temp_vars[:, 1], color = colors[n-1], marker = ".", label = f"{by}={n-1}")
                ax.legend()
#                 mean = temp_vars.mean(axis=0)
#                 ax.scatter(mean[0], mean[1], color = 'black', label = f"Mean: {mean.round(3)}", marker='x')
                if cls_mean:
                    ax.scatter(cls_mean_latent_vars[n-1, 0], cls_mean_latent_vars[n-1, 1], marker = '*', 
                                       color = 'black', s = 90,  
                                       label = f"center: {cls_mean_latent_vars[n-1].numpy().round(3)}")
                    ax.legend(loc = 'lower left')

                axs[0].scatter(temp_vars[:, 0], temp_vars[:, 1], color = colors[n-1], alpha = 0.3, marker = ".")
                if cls_mean:
                    axs[0].scatter(cls_mean_latent_vars[n-1, 0], cls_mean_latent_vars[n-1, 1], marker = '*', 
                                   color = colors[n-1], s = 90,  
                                   label = f"center: {cls_mean_latent_vars[n-1].numpy().round(3)}")
            ax.set_xlabel(self.notation_x + '0')

        axs[0].set_ylabel(self.notation_x + "1")
        
        plt.suptitle(title)
        
        plt.savefig("figure/" + self.directory + "latent_space_by_clusters", bbox_inches='tight', dpi=300)

        plt.show()
    
    def show_latent_space(self, dataset = "test"):
        """
            This function is suited for the experiment on synthetic data
        """
        if dataset == "test":
            df = self.df_test.reset_index(drop = True)
            latent_vars = self.test_latent_space
        
        elif dataset == "train":
            df = self.df_train.reset_index(drop = True)
            latent_vars = self.train_latent_space
        else:
            raise ValuerError("dataset (str) must be either test or train")
        
        plt.figure(figsize = (4, 4))
        plt.scatter(latent_vars[:, 0], latent_vars[:, 1], s = 5 , color = 'grey')
        plt.xlabel(self.notation_x + '0')
        plt.ylabel(self.notation_x + '1')

        title = "Latent Embedding: " + dataset + " set"
        
        if self.UMAP:
            title = "(UMAP) " + title
            
        plt.suptitle(title)
        plt.savefig("figure/" + self.directory + "latent_space", bbox_inches='tight', dpi=300)
        plt.show()

    def prepare_latent_traversal(self, dataset, T, N, fixed_value = 0.):

        if dataset == "test":
            df = self.df_test.reset_index(drop = True)
            latent_vars = self.test_latent_vars
        
        elif dataset == "train":
            df = self.df_train.reset_index(drop = True)
            latent_vars = self.train_latent_vars
        else:
            raise ValuerError("dataset (str) must be either test or train")
        
        t_grid = torch.linspace(df.iloc[:, self.indexs['t']].min()[0], 
                               df.iloc[:, self.indexs['t']].max()[0], T)[:, None]
        
        D = latent_vars.shape[1]
        s_curve = torch.zeros((T, N, D))
        z_grid = torch.zeros((D, N))
        
        for d in range(D):
            z_grid[d, :] = torch.linspace(latent_vars[:, d].min().item(), latent_vars[:, d].max().item(), N)
            z_cov = torch.zeros((T, D), dtype=torch.float32) + fixed_value
            
            for n, value in enumerate(z_grid[d, :]):
                z_cov[:, d] = value.repeat(T)
                x = torch.cat([t_grid, z_cov], 1)
                s_curve[:, n, d] = 1 - self.model.surv_decoder.forward(x).flatten()
    
        return s_curve.detach().numpy(), t_grid, z_grid
    
    def plot_latent_traversal(self, s_curve, t_grid, z_grid, figsize):
        T, N, D = s_curve.shape
        cmap = plt.cm.get_cmap('Blues')  # Choose a color map

        # Set up a ScalarMappable to map values to colors
#         sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_grid.min(), vmax=z_grid.max()))
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N))
        sm.set_array([])  # Set an empty array to create the colorbar

        # Set up figure and axes
        nrow = self.model.encoder.z_dim // 3
        if self.model.encoder.z_dim % 3 != 0:
            nrow += 1
        
        if self.model.encoder.z_dim == 2:
            fig, axs = plt.subplots(figsize=figsize, nrows = nrow, ncols = 2, sharey = 'row', sharex='row')
        else:
            fig, axs = plt.subplots(figsize=figsize, nrows = nrow, ncols = 3, sharey = 'row', sharex='row')
        ax_cbar = fig.add_axes([0.95, 0.15, 0.02, 0.7], anchor='NE')  # Define position of colorbar axis 1

        # Plot data
        for d, ax in enumerate(axs.ravel()):
            if d < D:
                for n in range(N):
                    color = sm.to_rgba(n)  # Get the color from the ScalarMappable
                    ax.plot(t_grid.flatten(), s_curve[:, n, d], color=color)
                    ax.set_title(fr"S(t | $z_{d}$)", fontsize = 10)
                    ax.set_ylim([0., 1.])
                    ax.set_xlabel('Time (in year)')
                    if d == 0:
                        ax.set_ylabel('Survival Probability')
        

        cbar = plt.colorbar(sm, cax=ax_cbar)
#         cbar.set_label(r'$z_{d}$')
        cbar.set_label("percentile")

#         plt.tight_layout()
#         plt.suptitle(r"$\Delta S(t|z)$ w.r.t. $\Delta z_{d}$ $\forall d$")
        plt.savefig("figure/" + self.directory + "lat_traversal_by_z", bbox_inches='tight', dpi=300)
        
        # Show the plot
        plt.show()
        
    def show_lat_traversal(self, dataset, figsize, T = 100, N = 100):
        
        s_curve, t_grid, z_grid = self.prepare_latent_traversal(dataset, T, N)
#         print(s_curve, t_grid, z_grid)
        self.plot_latent_traversal(s_curve, t_grid, z_grid, figsize)
    

    def show_latent_space_by_cont(self, dataset, column_name, figsize = (10, 5), rescale = True, with_event = False):
        """
            arg
                dataset: "test" or "train"
                column_name: list of continuous variables to be coloured
                figsize: figure size
                rescale: True -> axis is re-scaled to original scale (e.g. BMI, Age into original scale)
        """
        if dataset == "test":
            df = self.df_test.reset_index(drop = True)
            if with_event:
                df = df[df[self.events] == 1]
            latent_vars = self.test_latent_space[(df[self.events] == 1).index]
        
        elif dataset == "train":
            df = self.df_train.reset_index(drop = True)
            if with_event:
                df = df[df[self.events] == 1]
            latent_vars = self.train_latent_space[(df[self.events] == 1).index]
        else:
            raise ValuerError("dataset (str) must be either test or train")

        if rescale:
            df = self.inverse_standardize(df)
        
        num_cols = len(column_name)
        fig, axes = plt.subplots(1, num_cols, figsize = figsize)

        for i, col in enumerate(column_name):
            im = axes[i].scatter(latent_vars[:, 0], latent_vars[:, 1], s = 5, c=df.loc[:, col], cmap='coolwarm')
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label(col)
            axes[i].set_xlabel(self.notation_x + "0")
            axes[i].set_ylabel(self.notation_x + "1")

        combined_string = ''.join([s.replace('.', '') for s in column_name])

        fig.suptitle(f"Latent Embedding: {dataset}")

        plt.tight_layout()
        plt.savefig("figure/" + self.directory + f"lat_space_by_{combined_string}", bbox_inches='tight', dpi=300)
        plt.show()

    
    def inverse_standardize(self, df):
        
        scaler = self.scaler
        
        inverse_df = df.copy()
        columns = scaler.feature_names_in_

        inverse_data = scaler.inverse_transform(df.loc[:, columns])
        inverse_df.loc[:, columns] = inverse_data

        return inverse_df

    def show_latent_variations(self, dataset, figsize=(5, 3.5)):

        if dataset == "test":
            latent_vars = self.test_latent_vars
        elif dataset == "train":
            latent_vars = self.train_latent_vars
        else:
            raise ValueError("dataset (str) must be either test or train")
            
        plt.figure(figsize = figsize)

        ax = sns.boxplot(data=latent_vars, orient="h", color="grey")
        ax.set(ylabel="Latent dimension", xlabel="z", title="Variation within each latent dimension")

        plt.savefig("figure/" + self.directory + f"lat_variation_z", bbox_inches='tight', dpi=300)
        plt.show()

    def standardize_with_mean(self, data, mean):
        # Calculate the number of samples
        n = len(data)

        # Calculate the sum of squared differences from the provided mean
        ssd = np.sum(np.square(data - mean))

        # Calculate the standard deviation using the provided mean
        std = np.sqrt(ssd / n)

        # Calculate the standardized data using the provided mean and standard deviation
        standardized_data = (data - mean) / std

        return standardized_data

    def re_standardize(self, df, refer_mean = [120, 80, 21.75]):

        inverse_df = df.copy()
        columns = self.scaler.feature_names_in_

        inverse_data = self.scaler.inverse_transform(df.loc[:, columns])
        for i, (idx, mean) in enumerate(zip(self.indexs['c'], refer_mean)):
            inverse_df.iloc[:, idx] = self.standardize_with_mean(inverse_data[:, i], mean)

        return inverse_df


    def train_KM(self, n_clusters, distance):
        """
            Train KM algorithm with training latent space
            returns
                cluster_centers
        """
        
        torch.manual_seed(13)
        _, cls_centers = kmeans(X = torch.tensor(self.train_latent_space, dtype = torch.float32), 
                                num_clusters = n_clusters, 
                                distance = distance, device = 'cpu')
        
        return cls_centers
    
    def fit_KM(self, latent_vars, cls_centers):
        """
            It returns a dictionary
                key: clusters (e.g. 0, 1, 2, 3)
                values: [indexs]
        """
        # get the labels for each row
        labels = kmeans_predict(torch.tensor(latent_vars, dtype=torch.float32), cls_centers).detach().numpy()
        # initialize a dictionary to store indices for each cluster
        clusters = {i: [] for i in range(cls_centers.shape[0])}
        # iterate over rows and add their indices to the appropriate cluster
        for i, label in enumerate(labels):
            clusters[label].append(i)
        # return the dictionary of indices for each cluster
        return clusters
    
    def present_figure(self, dataset= 'test', z_axis = 0, cls_mean = True, 
                       by = 'b', K = 3, T = 50, N = 100, figsize = (12, 4),
                      fixed_value = 0, xlim = [0, 11]):
        
        if dataset == "test":
            df = self.df_test.reset_index(drop = True)
            latent_vars = self.test_latent_space
        
        elif dataset == "train":
            df = self.df_train.reset_index(drop = True)
            latent_vars = self.train_latent_space
        else:
            raise ValuerError("dataset (str) must be either test or train")
        
        
        ref_idx = df.iloc[:, self.indexs[by]]
        
        if cls_mean:
            cluster_mean = self.mean_by(df, by = by)
            print(f"Clustering data by {by}")
            cls_mean_latent_vars = self.model.get_latent_vars(cluster_mean)
            print("Means of (synthetic) clusters had been loaded")
        
        colors = sns.color_palette('hls', K)
        fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = figsize)

        labels = [ "High risk", "Low risk"]
        for n, ax in enumerate(axs.ravel()):
            # Plot clustering patterns
            if n == 0:
                for k in range(K):
                    temp_vars = latent_vars[(ref_idx == k).values.flatten(), :]
                    ax.scatter(temp_vars[:, 0], temp_vars[:, 1], color = colors[k], alpha = 0.3, marker = ".", label = labels[k])
                    if cls_mean:
                        ax.scatter(cls_mean_latent_vars[k, 0], cls_mean_latent_vars[k, 1], marker = '*', 
                                       color = colors[k], s = 90)
#                         label = f"center: {cls_mean_latent_vars[k].numpy().round(3)}"
                        ax.set_xlabel(r"$z_{0}$")
                        ax.set_ylabel(r"$z_{1}$")
                    ax.legend(loc = 'lower left')
                if z_axis == 0:
                    ax.axhline(fixed_value, color = 'grey', linestyle='dashed')
                else:
                    ax.axvline(fixed_value, color = 'grey', linestyle='dashed')
                
            if n == 1:
                im = ax.scatter(latent_vars[:, 0], latent_vars[:, 1], s = 5, c=df.loc[:, 'a'], cmap='coolwarm')
                cbar = fig.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=5)
                cbar.set_label(r'$b_{cont}$')
                ax.set_xlabel(r"$z_{0}$")
                ax.set_ylabel(r"$z_{1}$")

            
            if n == 2:
                s_curve, t_grid, z_grid = self.prepare_latent_traversal(dataset, T, N, fixed_value)
                cmap = plt.cm.get_cmap('Blues')  # Choose a color map
                
                sm = ScalarMappable(cmap=cmap, 
                                    norm=plt.Normalize(vmin=z_grid[z_axis, :].min(), vmax=z_grid[z_axis, :].max()))
                sm.set_array([])  # Set an empty array to create the colorbar
                
                for j in range(N):
                    color = sm.to_rgba(z_grid[z_axis, j])  # Get the color from the ScalarMappable
                    ax.plot(t_grid.flatten(), s_curve[:, j, z_axis], color=color)
                    ax.set_ylim([0., 1.05])
                    ax.set_xlabel('Time-to-failure')
                    ax.set_ylabel('Survival probability')
                
                if z_axis == 0:
                    z_other = 1
                else:
                    z_other = 0
                    
                ax.plot(t_grid.flatten(), s_curve[:, j, z_axis], color=color, label = fr"S(t | $z_{z_axis}$, $z_{z_other}$ = {fixed_value})")
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label(fr"$z_{z_axis}$")
                ax.legend(loc = 'upper right')
                ax.set_xlim(xlim)
        
        textbox_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '(A)'
        axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=20,
                    verticalalignment='top', bbox=textbox_props)

        plt.tight_layout()
        plt.savefig("figure/" + self.directory + f"FIG_01", bbox_inches='tight', dpi=400)
    

