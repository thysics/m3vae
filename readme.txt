
1. Implementation of M4VAE: For a detailed look into the M3VAE's implementation, refer to Syn01_Train. This section is key for understanding the practical application of the model.

2. Performance Evaluation: To evaluate the M3VAE's performance, we use K-fold cross-validation. This rigorous assessment, especially in comparison to benchmark models, can be found under Syn01_K_CV.

3. Visualization of Outcomes: Check out Syn01_Analysis for insights into the outcomes derived from the Syn01_Train. This part is crucial for visualizing and interpreting the results of the M4VAE implementation.

Furthermore, to delve into the source code of the M4VAE, navigate to model/M4VAE. This directory is divided into four main components:

    Encoder: Transforms inputs x, c into latent space z (Encoder (x, c -> z)).
    
    ObservationDecoder: Reconstructs inputs from the latent space (ObservationDecoder (z -> x, c)).
    
    SurvivalDecoder: Decodes the survival time from the latent space (SurvivalDecoder (z -> t)).
    
    M4VAE Module: This wrapper encapsulates the entire model, including the loss function and training mechanisms.
   

Just to let you know, this repository includes only the simulated dataset (A). If you are interested in accessing real-world data, it is publicly available but requires approval. You can find the real-world dataset at the Golestan Cohort Study in Iran. For more information and to request access, visit their website: Golestan Cohort Study: https://dceg.cancer.gov/research/cancer-types/esophagus/golestan-cohort-study-iran-esophageal-squamous-cell-carcinoma
