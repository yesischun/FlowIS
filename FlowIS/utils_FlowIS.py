import numpy as np
from scipy.special import ndtr
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, laplace, uniform, multivariate_normal

# accept-reject sampling
class RejectAcceptSampling:
    """
    Rejection-acceptance sampling utility class that supports sampling from 1D and 2D conditional distributions.
    Used to obtain samples from the target conditional distribution p(conditional variables|given variables).
    The core idea is to use a proposal distribution q combined with a scaling constant M, accepting or rejecting
    sampled points with a certain probability to approximate the target distribution.
    """
    def __init__(self, dimensionality=1):
        """
        Initialize the rejection-acceptance sampler
        :param dimensionality: Sampling dimensionality, currently supports 1 (one-dimensional) and 2 (two-dimensional)
                               to distinguish sampling logic for different dimensions, default is 1D
        """
        self.dimensionality = dimensionality

    def rejection_sampling_1D(s, p_as, p_s, q, M, max_trials=10000, random_state=None):
        """
        Rejection-acceptance sampling for 1D conditional distribution: sample from p(a|s) = p(a,s)/p(s),
        i.e., sample a that satisfies the condition when s is known
        
        :param s: Given conditional variable value, shape should be compatible with subsequent joint distribution,
                  marginal distribution and other logics, usually 1D array or tensor with appropriate dimensions
        :param p_as: Joint distribution object, must implement logpdf(a, s) method that returns the log probability
                     density of the joint distribution p(a,s)
        :param p_s: Marginal distribution object, must implement logpdf(s) method that returns the log probability
                    density of the marginal distribution p(s)
        :param q: Proposal distribution object, must implement sample() method (generates samples) and logpdf(a)
                  method (returns log probability density of the sample)
        :param M: Scaling constant, must satisfy M*q(a) ≥ p(a|s) for all possible a, used to control acceptance probability
        :param max_trials: Maximum number of sampling attempts, returns None if no success after this number, default 10000
        :param random_state: Random seed for fixing random number generation to ensure reproducibility, default None (not fixed)
        
        :return: Returns tuple (sampled a, log probability density of conditional distribution p(a|s), number of sampling attempts)
                 on successful sampling; returns None on failure
        """
        # Initialize random number generation, fix with specified seed if provided to ensure reproducibility
        np.random.seed(random_state)  
        
        for trial_idx in range(max_trials):
            # Step 1: Sample candidate a from proposal distribution q
            a = q.sample()  
            
            # Step 2: Calculate various log probability densities for subsequent acceptance probability calculation
            log_p_s_val = p_s.logpdf(s)  # Log probability density of marginal distribution p(s)
            # Log probability density of joint distribution p(a,s), concatenate s and a into appropriate shape
            # (adjust according to parameter requirements of p_as.logpdf method)
            log_p_as_val = p_as.logpdf(np.concatenate([s, a], axis=1))  
            log_q_a_val = q.logpdf(a)  # Log probability density of proposal distribution q at sample a
            
            # Step 3: Calculate log probability density of conditional distribution
            # Conditional probability formula: p(a|s) = p(a,s)/p(s), in log form: log(p(a|s)) = log(p(a,s)) - log(p(s))
            log_p_cond_val = log_p_as_val - log_p_s_val  
            
            # Step 4: Calculate log form of acceptance probability
            # Acceptance probability alpha = p(a|s) / (M*q(a)), in log form: log(alpha) = log(p(a|s)) - (log(M) + log(q(a)))
            log_alpha = log_p_cond_val - (np.log(M) + log_q_a_val)  
            
            # Step 5: Determine whether to accept the sample
            # Generate random number from uniform distribution [0,1), compare its log with log_alpha,
            # equivalent to alpha > random number (comparison in linear space)
            if np.log(np.random.rand()) < log_alpha:  
                return a, log_p_cond_val, trial_idx + 1  # Return sampling result, conditional log density, number of attempts (counting from 1)
        
        # Return None if no sample is accepted after maximum attempts
        return None  

    def rejection_sampling_2D(scene, NDE_intersection, 
                              dim=[0, 1, 2, 3, 4, 5, 6, 7], 
                              max_trials=10000, random_state=None, type='sample'):
        """
        Rejection-acceptance sampling for 2D conditional distribution: sample from p(a,b|s) = p(a,b,s)/p(s),
        i.e., sample a and b that satisfy the condition when scene is known
        """
        # Initialize random number generation, fix with specified seed if provided to ensure reproducibility
        np.random.seed(random_state)  
        accept = 0
        for trial_idx in range(max_trials):
            # Step 1: Sample 2D behavior from proposal distribution, must satisfy 0<=behavior<=1 constraint
            while True:
                accelerate = NDE_intersection['proposal_accelerate'].sample()
                if 0 <= accelerate <= 1:
                    log_proposal_accelerate = NDE_intersection['proposal_accelerate'].logpdf(accelerate)
                    break
            while True:
                steering_angle = NDE_intersection['proposal_steering_angle'].sample()
                if 0 <= steering_angle <= 1:
                    log_proposal_steering_angle = NDE_intersection['proposal_steering_angle'].logpdf(steering_angle)
                    break
            scene_behavior = np.concatenate([scene[0], accelerate, steering_angle])
            
            # Step 2: Calculate log probability densities of joint and marginal distributions
            log_p_as_val = NDE_intersection['scene_behavior'].logpdf(scene_behavior.reshape(1,-1))  
            log_p_s_val = NDE_intersection['scene_behavior'].logpdf_marginal(scene, dim)  
            
            # Step 3: Calculate log probability density of conditional distribution
            # Conditional probability formula: p(a,b|s) = p(a,b,s)/p(s), in log form: log(p(a,b|s)) = log(p(a,b,s)) - log(p(s))
            log_p_cond_val = log_p_as_val - log_p_s_val  
            
            # Step 4: Calculate log form of acceptance probability
            # Acceptance probability alpha = p(a,b|s) / (M*q(a)*q(b)), in log form: 
            # log(alpha) = log(p(a,b|s)) - (log(M) + log(q(a)) + log(q(b)))
            log_alpha = log_p_cond_val - (np.log(NDE_intersection['proposal_M']) + log_proposal_accelerate + log_proposal_steering_angle)  
            
            # Step 5: Determine whether to accept the sample, add handling for excessively large log_p_cond_val
            if log_p_cond_val > 4 or np.log(np.random.rand()) < log_alpha:  
                if type == 'sample':
                    return scene_behavior, log_p_cond_val, trial_idx + 1  
                else:
                    accept += 1
                
        if type == 'sample':
            return None
        else:
            return accept / max_trials

# marginal truncated distribution
class MarginalTruncatedGMM:
    """Truncated Gaussian Mixture Model (GMM) that supports manual specification of precomputed dimension combinations"""
    def __init__(self, n_components=1, random_state=None, bounds=None, precompute_dims=None):
        """
        Initialize the truncated Gaussian Mixture Model
        
        Parameters:
            precompute_dims: List of dimension combinations for which truncation factors need to be precomputed
                             Example: [[0, 1], [0, 2], [1, 2, 3]]
        """
        self.n_components = n_components
        self.random_state = random_state
        self.bounds = bounds or [(0, 1)]  # Default truncation range is [0,1]; n tuples required for n-dimensional data
        self.precompute_dims = precompute_dims  # Dimension combinations to precompute
        
        self.weights = None      # Weights of each Gaussian component
        self.means = None        # Means of each Gaussian component
        self.covariances = None  # Covariance matrices of each Gaussian component
        self.data_dim = None     # Dimension of the input data
        
        # Cache normalization constants
        self.truncation_factors = None         # Truncation factors for the full distribution
        self.marginal_truncation_factors = {}  # Truncation factors for marginal distributions
    
    def fit(self, X):
        """Fit data using scikit-learn's GMM and precompute truncation factors for specified dimensions"""
        # Fit the GMM to the data
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='full'  # Use full covariance matrix
        ).fit(X)
        
        # Extract GMM parameters
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covariances_
        self.data_dim = X.shape[1]
        
        # Precompute all necessary truncation factors
        self._precompute_all_truncation_factors()
        
        return self
    
    def _precompute_all_truncation_factors(self):
        """Precompute truncation factors for the full distribution and specified marginal distributions"""
        if self.weights is None:
            return
            
        # Precompute truncation factors for the full distribution
        self.truncation_factors = np.zeros(self.n_components)
        for i in range(self.n_components):
            self.truncation_factors[i] = self._compute_truncation_factor(
                self.means[i], 
                self.covariances[i]
            )
            
        # If precompute dimensions are specified, compute truncation factors for marginal distributions
        if self.precompute_dims is not None:
            self._precompute_custom_marginal_truncation_factors()
    
    def _precompute_custom_marginal_truncation_factors(self):
        """Precompute truncation factors for user-specified dimension combinations of marginal distributions"""
        if not self.precompute_dims:
            return
            
        for dims in self.precompute_dims:
            # Validate the format of dimension combinations
            if not isinstance(dims, (list, tuple)):
                raise ValueError(f"Dimension combination {dims} has invalid format; must be a list or tuple")
                
            # Check for invalid dimension indices
            if any(d < 0 or d >= self.data_dim for d in dims):
                raise ValueError(f"Dimension combination {dims} contains invalid dimension indices")
                
            # Convert to a sorted tuple to ensure unique keys in the dictionary
            dims_tuple = tuple(sorted(dims))
            
            # Extract parameters for the marginal distribution
            marginal_means = self.means[:, dims_tuple]
            marginal_covs = []
            
            for cov in self.covariances:
                # Extract the sub-covariance matrix corresponding to the marginal dimensions
                marginal_cov = cov[np.ix_(dims_tuple, dims_tuple)]
                marginal_covs.append(marginal_cov)
                
            # Compute truncation factors for each component of the marginal distribution
            truncation_factors = np.zeros(self.n_components)
            for i in range(self.n_components):
                truncation_factors[i] = self._compute_truncation_factor(
                    marginal_means[i], 
                    marginal_covs[i],
                    dims=list(range(len(dims_tuple)))  # Indices start from 0 for the marginal distribution
                )
                
            # Store truncation factors in the dictionary
            self.marginal_truncation_factors[dims_tuple] = truncation_factors
    
    def _compute_truncation_factor(self, mean, cov, dims=None):
        """Compute the truncation factor (normalization constant)"""
        if dims is None:
            dims = list(range(len(mean)))
        
        truncation_factor = 1.0
        for i, d in enumerate(dims):
            lower, upper = self.bounds[d]
            std_dev = np.sqrt(cov[i, i])
            
            if std_dev > 1e-10:  # Avoid division by zero
                z_lower = (lower - mean[i]) / std_dev
                z_upper = (upper - mean[i]) / std_dev
                phi_upper = ndtr(z_upper)  # CDF of standard normal distribution at z_upper
                phi_lower = ndtr(z_lower)  # CDF of standard normal distribution at z_lower
                truncation_factor *= (phi_upper - phi_lower)
        
        return truncation_factor
    
    def _truncate_sample(self, sample):
        """Truncate samples to the specified bounds"""
        truncated = np.copy(sample)
        for i, (lower, upper) in enumerate(self.bounds):
            truncated[:, i] = np.clip(truncated[:, i], lower, upper)
        return truncated    
    
    def sample(self, n_samples=1, max_attempts=1000):
        """Sample from the truncated mixture distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        samples = []
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # Calculate remaining samples needed
            remaining = n_samples - len(samples)
            # Determine number of samples to generate from each component
            n_samples_per_component = np.random.multinomial(remaining, self.weights)
            
            # Generate samples for each Gaussian component
            for i in range(self.n_components):
                if n_samples_per_component[i] > 0:
                    # Sample from the i-th Gaussian component
                    component_samples = np.random.multivariate_normal(
                        mean=self.means[i],
                        cov=self.covariances[i],
                        size=n_samples_per_component[i]
                    )
                    
                    # Truncate samples to the specified bounds
                    truncated_samples = self._truncate_sample(component_samples)
                    
                    # Only keep samples that lie entirely within the bounds
                    valid_mask = np.ones(len(truncated_samples), dtype=bool)
                    for j, (lower, upper) in enumerate(self.bounds):
                        valid_mask &= (truncated_samples[:, j] >= lower) & (truncated_samples[:, j] <= upper)
                    
                    valid_samples = truncated_samples[valid_mask]
                    samples.extend(valid_samples)
                    
                    # Stop early if enough valid samples are collected
                    if len(samples) >= n_samples:
                        break
            
            attempts += 1
        
        # Combine samples, limit to n_samples, and shuffle
        if samples:
            samples = np.array(samples[:n_samples])
            np.random.shuffle(samples)
            return samples
        else:
            return np.array([])
    
    def pdf(self, X):
        """Calculate the probability density function (PDF) of the truncated distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # For each sample, compute weighted density sum across components using precomputed truncation factors
        density = np.zeros(X.shape[0])
        for i in range(self.n_components):
            truncation_factor = self.truncation_factors[i]
            if truncation_factor > 1e-10:  # Avoid division by zero
                # Compute PDF of the i-th Gaussian component
                component_density = multivariate_normal.pdf(
                    X, 
                    mean=self.means[i], 
                    cov=self.covariances[i],
                    allow_singular=True
                )
                # Weight by component weight and normalize with truncation factor
                density += self.weights[i] * component_density / truncation_factor
            
        return density
        
    def sample_marginal(self, n_samples=1, dims=[0, 1]):
        """
        Sample from the marginal distribution of specified dimensions
        
        Parameters:
            n_samples: Number of samples to generate
            dims: List of dimension indices to retain for the marginal distribution
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        dims_tuple = tuple(sorted(dims))
        # Check if truncation factors for the specified dimensions are precomputed
        if dims_tuple not in self.marginal_truncation_factors:
            raise ValueError(f"Truncation factors for dimension combination {dims} not precomputed; specify during initialization")
            
        # Extract parameters for the marginal distribution
        marginal_means = self.means[:, dims]
        marginal_covs = []
        for cov in self.covariances:
            # Extract sub-covariance matrix for the marginal dimensions
            marginal_cov = cov[np.ix_(dims, dims)]
            marginal_covs.append(marginal_cov)

        # Create a marginal truncated GMM sampler
        marginal_bounds = [self.bounds[i] for i in dims]        
        marginal_sampler = MarginalTruncatedGMM(
                            n_components=self.n_components,
                            random_state=self.random_state,
                            bounds=marginal_bounds
                            )
        # Assign precomputed parameters to the marginal sampler
        marginal_sampler.weights = self.weights
        marginal_sampler.means = marginal_means
        marginal_sampler.covariances = np.array(marginal_covs)
        marginal_sampler.truncation_factors = self.marginal_truncation_factors[dims_tuple]
        
        # Sample from the marginal distribution
        return marginal_sampler.sample(n_samples)
    
    def logpdf(self, X):
        """Calculate the log-probability density function (log-PDF) of the truncated distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # For each sample, compute log-density of each component using precomputed truncation factors
        log_densities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # Handle zero truncation factor (avoid log(0) by setting to -infinity)
            log_truncation_factor = np.log(self.truncation_factors[i]) if self.truncation_factors[i] > 0 else np.inf
            # Compute log-PDF of the i-th Gaussian component
            log_component_density = multivariate_normal.logpdf(
                X, 
                mean=self.means[i], 
                cov=self.covariances[i],
                allow_singular=True
            )
            # Adjust log-density with truncation factor
            log_densities[:, i] = log_component_density - log_truncation_factor
        
        # Stably compute weighted log-sum using logaddexp to avoid numerical underflow
        log_weights = np.log(self.weights)
        log_pdf = np.logaddexp.reduce(log_weights + log_densities, axis=1)
        
        return log_pdf
    
    def logpdf_marginal(self, X, dims=[0, 1]):
        """
        Calculate the log-probability density function (log-PDF) of the marginal distribution for specified dimensions
        
        Parameters:
            X: Input samples, shape (n_samples, len(dims))
            dims: List of dimension indices corresponding to the marginal distribution
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        dims_tuple = tuple(sorted(dims))
        # Check if truncation factors for the specified dimensions are precomputed
        if dims_tuple not in self.marginal_truncation_factors:
            raise ValueError(f"Truncation factors for dimension combination {dims} not precomputed; specify during initialization")
            
        # Extract parameters for the marginal distribution  
        marginal_truncation_factors = self.marginal_truncation_factors[dims_tuple]    
        marginal_means = self.means[:, dims]
        marginal_covs = [] 
        for cov in self.covariances:
            # Extract sub-covariance matrix for the marginal dimensions
            marginal_cov = cov[np.ix_(dims, dims)]
            marginal_covs.append(marginal_cov)
            
        # For each sample, compute log-density of each marginal component
        log_densities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # Handle zero truncation factor (avoid log(0) by setting to -infinity)
            log_truncation_factor = np.log(marginal_truncation_factors[i]) if marginal_truncation_factors[i] > 0 else np.inf
            # Compute log-PDF of the i-th marginal Gaussian component
            log_component_density = multivariate_normal.logpdf(
                X, 
                mean=marginal_means[i], 
                cov=marginal_covs[i],
                allow_singular=True
            )
            # Adjust log-density with marginal truncation factor
            log_densities[:, i] = log_component_density - log_truncation_factor
        
        # Stably compute weighted log-sum using logaddexp
        log_weights = np.log(self.weights)
        log_pdf = np.logaddexp.reduce(log_weights + log_densities, axis=1)
        
        return log_pdf

class MarginalGaussianMixture:
    """Gaussian Mixture Model (GMM) that supports sampling from marginal distributions of arbitrary dimensions"""
    def __init__(self, n_components=1, random_state=None):
        """Initialize the Gaussian Mixture Model"""
        self.n_components = n_components
        self.random_state = random_state
        self.weights = None      # Weights of each Gaussian component
        self.means = None        # Means of each Gaussian component
        self.covariances = None  # Covariance matrices of each Gaussian component
    
    def fit(self, X):
        """Fit data using scikit-learn's GMM and extract model parameters"""
        # Fit GMM to the input data
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='full'  # Use full covariance matrix
        ).fit(X)
        
        # Extract learned model parameters
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covariances_
        
        return self
    
    def sample(self, n_samples=1):
        """Sample from the full Gaussian mixture distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # Determine the number of samples to generate from each Gaussian component
        n_samples_per_component = np.random.multinomial(n_samples, self.weights)
        
        # Generate samples for each component
        samples = []
        for i in range(self.n_components):
            if n_samples_per_component[i] > 0:
                # Sample from the i-th multivariate Gaussian component
                component_samples = np.random.multivariate_normal(
                    mean=self.means[i],
                    cov=self.covariances[i],
                    size=n_samples_per_component[i]
                )
                samples.append(component_samples)
        
        # Combine all samples and shuffle to avoid component order bias
        if samples:
            samples = np.vstack(samples)  # Vertically stack samples from all components
            np.random.shuffle(samples)
            return samples
        else:
            return np.array([])
    
    def pdf(self, X):
        """Calculate the probability density function (PDF) of the Gaussian mixture distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # Compute weighted sum of PDF values from all Gaussian components for each sample
        density = np.zeros(X.shape[0])
        for i in range(self.n_components):
            # Calculate PDF of the i-th multivariate Gaussian component using scipy
            component_density = multivariate_normal.pdf(
                X, 
                mean=self.means[i], 
                cov=self.covariances[i],
                allow_singular=True  # Allow singular covariance matrices (avoids errors for degenerate distributions)
            )
            # Accumulate weighted density (weight * component PDF)
            density += self.weights[i] * component_density
            
        return density
    
    def sample_marginal(self, n_samples=1, dims=[0, 1]):
        """
        Sample from the marginal distribution of specified dimensions
        
        Parameters:
            n_samples: Number of samples to generate
            dims: List of dimension indices to retain for the marginal distribution, 
                  default is the first two dimensions [0, 1]
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # Extract parameters for the marginal distribution
        # Marginal mean: retain only the specified dimensions from the full mean vector
        marginal_means = self.means[:, dims]
        marginal_covs = []
        
        for cov in self.covariances:
            # Marginal covariance: extract the submatrix corresponding to the specified dimensions
            # np.ix_ ensures we get the correct 2D submatrix (rows and columns matching 'dims')
            marginal_cov = cov[np.ix_(dims, dims)]
            marginal_covs.append(marginal_cov)
            
        # Determine number of samples to generate from each marginal component
        n_samples_per_component = np.random.multinomial(n_samples, self.weights)
        
        # Generate samples for each marginal component
        samples = []
        for i in range(self.n_components):
            if n_samples_per_component[i] > 0:
                # Sample from the marginal distribution of the i-th Gaussian component
                component_samples = np.random.multivariate_normal(
                    mean=marginal_means[i],
                    cov=marginal_covs[i],
                    size=n_samples_per_component[i]
                )
                samples.append(component_samples)
        
        # Combine and shuffle marginal samples
        if samples:
            samples = np.vstack(samples)
            np.random.shuffle(samples)
            return samples
        else:
            return np.array([])
    
    def logpdf(self, X):
        """Calculate the log-probability density function (log-PDF) of the Gaussian mixture distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # Compute log-PDF values for each Gaussian component and each sample
        log_densities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # Calculate log-PDF of the i-th multivariate Gaussian component using scipy
            log_densities[:, i] = multivariate_normal.logpdf(
                X, 
                mean=self.means[i], 
                cov=self.covariances[i],
                allow_singular=True  # Allow singular covariance matrices
            )
        
        # Stably compute the weighted log-sum using np.logaddexp.reduce to avoid numerical underflow
        # Formula: log(sum(w_i * exp(log_p_i))) = log(sum(exp(log(w_i) + log_p_i)))
        log_weights = np.log(self.weights)
        log_pdf = np.logaddexp.reduce(log_weights + log_densities, axis=1)
        
        return log_pdf
    
    def logpdf_marginal(self, X, dims=[0, 1]):
        """
        Calculate the log-probability density function (log-PDF) of the marginal distribution for specified dimensions
        
        Parameters:
            X: Input samples, shape must be (n_samples, len(dims)) (matches the marginal dimension)
            dims: List of dimension indices corresponding to the marginal distribution
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted; call the fit method first")
            
        # Extract parameters for the marginal distribution
        marginal_means = self.means[:, dims]
        marginal_covs = []
        
        for cov in self.covariances:
            # Extract the sub-covariance matrix for the specified marginal dimensions
            marginal_cov = cov[np.ix_(dims, dims)]
            marginal_covs.append(marginal_cov)
            
        # Compute log-PDF values for each marginal component and each sample
        log_densities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # Calculate log-PDF of the i-th marginal Gaussian component
            log_densities[:, i] = multivariate_normal.logpdf(
                X, 
                mean=marginal_means[i], 
                cov=marginal_covs[i],
                allow_singular=True
            )
        
        # Stably compute the weighted log-sum for the marginal distribution
        log_weights = np.log(self.weights)
        log_pdf = np.logaddexp.reduce(log_weights + log_densities, axis=1)
        
        return log_pdf

# proposal distribution
class NormalProposalDistribution:
    """
    Normal distribution proposal distribution that supports log-probability calculation and sampling
    """
    def __init__(self, mean=0.0, std=1.0):
        """
        Initialize the normal distribution proposal distribution
        
        Parameters:
            mean: Mean of the distribution
            std: Standard deviation of the distribution (must be a positive number)
        """
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.var = std ** 2  # Variance
        
        # Precompute constants to improve log-probability calculation efficiency
        self.log_const = -0.5 * np.log(2 * np.pi) - np.log(self.std)
    
    def sample(self, n_samples=1):
        """Generate random samples from the normal distribution"""
        return np.array(np.random.normal(self.mean, self.std, n_samples))
    
    def logpdf(self, x):
        """
        Calculate the log-probability density of sample x
        
        Parameters:
            x: Sample value(s)
            
        Returns:
            Log-probability density value(s)
        """
        # Log-probability density formula of normal distribution: 
        # log(p(x)) = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²
        return np.array(self.log_const - 0.5 * ((x - self.mean) / self.std) ** 2)
    
    def pdf(self, x):
        """
        Calculate the probability density of sample x
        
        Parameters:
            x: Sample value(s)
            
        Returns:
            Probability density value(s)
        """
        return np.exp(self.logpdf(x))

class MixedGaussianLaplace:
    def __init__(self, gaussian_params, laplace_params, mix_ratio):
        """
        Initialize the mixed distribution class

        :param gaussian_params: Parameters for the Gaussian distribution, in dictionary form, 
                               containing 'mu' (mean) and 'sigma' (standard deviation)
        :param laplace_params: Parameters for the Laplace distribution, in dictionary form, 
                               containing 'loc' (location parameter, similar to mean) and 'scale' (scale parameter)
        :param mix_ratio: Mixing ratio, i.e., the proportion of the Laplace distribution in the mixed distribution, 
                          ranging from (0, 1). The proportion of the Gaussian distribution is 1 - mix_ratio
        """
        self.gaussian_mu = gaussian_params['mu']
        self.gaussian_sigma = gaussian_params['sigma']
        self.laplace_loc = laplace_params['loc']
        self.laplace_scale = laplace_params['scale']
        self.mix_ratio = mix_ratio

    def sample(self, n_samples=1):
        """
        Sample from the mixed distribution

        :param n_samples: Number of samples to generate
        :return: Array of sampled values with shape (n_samples,)
        """
        samples = []
        for _ in range(n_samples):
            if np.random.rand() < self.mix_ratio:
                # Sample from the Laplace distribution
                sample = laplace.rvs(loc=self.laplace_loc, scale=self.laplace_scale)
            else:
                # Sample from the Gaussian distribution
                sample = norm.rvs(loc=self.gaussian_mu, scale=self.gaussian_sigma)
            samples.append(sample)
        return np.array(samples)

    def pdf(self, x):
        """
        Calculate the probability density value of the mixed distribution at the given x

        :param x: Input value(s), can be a single number or an array
        :return: Probability density values of the mixed distribution at the corresponding x, 
                 with the same shape as x
        """
        laplace_pdf_val = laplace.pdf(x, loc=self.laplace_loc, scale=self.laplace_scale)
        gaussian_pdf_val = norm.pdf(x, loc=self.gaussian_mu, scale=self.gaussian_sigma)
        return self.mix_ratio * laplace_pdf_val + (1 - self.mix_ratio) * gaussian_pdf_val

    def log_pdf(self, x):
        """
        Calculate the log-probability density value of the mixed distribution at the given x

        :param x: Input value(s), can be a single number or an array
        :return: Log-probability density values of the mixed distribution at the corresponding x, 
                 with the same shape as x
        """
        return np.log(self.pdf(x))

class MixedLaplaceUniform:
    def __init__(self, laplace_params, uniform_params, mix_ratio):
        """
        Initialize a mixed Laplace and uniform distribution
        
        Parameters:
            laplace_params: Parameters for the Laplace distribution, in dictionary form {'loc': location, 'scale': scale}
            uniform_params: Parameters for the uniform distribution, in dictionary form {'low': lower bound, 'high': upper bound}
            mix_ratio: Weight of the Laplace distribution in the mixture (0 < mix_ratio < 1)
        """
        self.laplace_loc = laplace_params['loc']
        self.laplace_scale = laplace_params['scale']
        self.uniform_low = uniform_params['low']
        self.uniform_high = uniform_params['high']
        self.mix_ratio = mix_ratio
        
        # Validate parameter validity
        if not (0 < mix_ratio < 1):
            raise ValueError("mix_ratio must be in the range (0, 1)")
        if self.uniform_high <= self.uniform_low:
            raise ValueError("The upper bound of the uniform distribution must be greater than the lower bound")

    def sample(self, n_samples=1):
        """Sample from the mixed distribution"""
        samples = []
        for _ in range(n_samples):
            # Select distribution based on the mix ratio
            if np.random.rand() < self.mix_ratio:
                # Sample from the Laplace distribution
                samples.append(laplace.rvs(
                    loc=self.laplace_loc, 
                    scale=self.laplace_scale
                ))
            else:
                # Sample from the uniform distribution
                samples.append(uniform.rvs(
                    loc=self.uniform_low, 
                    scale=self.uniform_high - self.uniform_low
                ))
        return np.array(samples)

    def pdf(self, x):
        """Calculate the probability density function value"""
        # Probability density of the Laplace distribution
        laplace_pdf = laplace.pdf(
            x, 
            loc=self.laplace_loc, 
            scale=self.laplace_scale
        )
        
        # Probability density of the uniform distribution
        uniform_pdf = uniform.pdf(
            x, 
            loc=self.uniform_low, 
            scale=self.uniform_high - self.uniform_low
        )
        
        # Probability density of the mixed distribution
        return self.mix_ratio * laplace_pdf + (1 - self.mix_ratio) * uniform_pdf

    def logpdf(self, x):
        """Calculate the log probability density function value"""
        return np.log(self.pdf(x))

class GMMProposalDistribution:
    """
    Gaussian Mixture Model proposal distribution that supports log probability calculation and sampling
    """
    def __init__(self, x, log_max_pdf, n_components):
        # Initialize target distribution
        self.n_components = n_components
        pdf = np.exp(log_max_pdf)
        # dx = x[1] - x[0]
        # pdf_normalized = pdf / (np.sum(pdf) * dx)
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
        samples = np.repeat(x.reshape(-1, 1), repeats=(pdf * 10000).astype(int), axis=0)
        self.gmm.fit(samples)

        # Get GMM parameters
        self.weights = self.gmm.weights_
        self.means = self.gmm.means_.flatten()
        self.covars = self.gmm.covariances_.flatten()

        # Evaluate GMM at the same points
        pdf_gmm = np.zeros_like(x)
        for i in range(n_components):
            pdf_gmm += self.weights[i] * norm.pdf(x, loc=self.means[i], 
                                                     scale=np.sqrt(self.covars[i]))

        # Normalize GMM results and calculate fitting error
        # pdf_gmm_normalized = pdf_gmm / (np.sum(pdf_gmm) * dx)
        gmm_error = np.mean((pdf_gmm - pdf) ** 2)
        print(f"GMM fitting mean squared error: {gmm_error:.6f}")
    
    def sample(self, n_samples=1):
        """Sample from univariate mixture distribution"""
        if self.weights is None:
            raise ValueError("Model has not been fitted, please call fit method first")
            
        # Determine number of samples to generate for each component
        n_samples_per_component = np.random.multinomial(n_samples, self.weights)
        
        # Generate samples for each component
        samples = []
        for i in range(self.n_components):
            if n_samples_per_component[i] > 0:
                # Sample from the i-th Gaussian distribution (using norm)
                component_samples = norm.rvs(
                    loc=self.means[i],
                    scale=np.sqrt(self.covars[i]),  # Correction: use self.covars
                    size=n_samples_per_component[i]
                )
                samples.append(component_samples)
        
        # Combine all samples and shuffle order
        if samples:
            samples = np.hstack(samples)  # Horizontally stack (univariate)
            np.random.shuffle(samples)
            return samples.reshape(-1, 1)  # Maintain 2D array format (n_samples, 1)
        else:
            return np.array([])
    
    def pdf(self, X):
        """Calculate univariate probability density function values"""
        if self.weights is None:
            raise ValueError("Model has not been fitted, please call fit method first")
            
        # Ensure input is a 2D array
        X = np.asarray(X).reshape(-1, 1)
        
        # Initialize probability density array
        pdf_gmm = np.zeros(X.shape[0])
        
        # Calculate weighted probability density for each component
        for i in range(self.n_components):  # Correction: use self.n_components
            pdf_gmm += self.weights[i] * norm.pdf(X[:, 0], loc=self.means[i], 
                                                     scale=np.sqrt(self.covars[i]))
            
        return pdf_gmm
    
    def logpdf(self, X):  # Correction: indent function to make it a class method
        """Calculate log probability density function values (numerically stable version)"""
        if self.weights is None:
            raise ValueError("Model has not been fitted, please call fit method first")
        
        # Ensure input is a 2D array
        X = np.asarray(X).reshape(-1, 1)
        
        n_samples = X.shape[0]
        log_density = np.zeros(n_samples)
        
        for i in range(self.n_components):
            # Calculate log probability density of the i-th component + log weight
            # Use norm.logpdf to get log probability directly, avoiding log(pdf)
            component_logpdf = np.log(self.weights[i]) + norm.logpdf(
                X[:, 0],  # Correction: handle 1D input
                loc=self.means[i], 
                scale=np.sqrt(self.covars[i])  # Correction: use self.covars
            )
            
            # Accumulate along sample dimension (initialize log_density in first iteration)
            if i == 0:
                log_density = component_logpdf
            else:
                # Use logaddexp.reduce to combine log probabilities and avoid underflow
                log_density = np.logaddexp(log_density, component_logpdf)
        
        return log_density

class PiecewiseUniformDistribution:
    """
    Piecewise uniform distribution sampler that supports different probability densities for different intervals
    """
    def __init__(self, breakpoints, densities):
        """
        Initialize the piecewise uniform distribution
        
        Parameters:
            breakpoints: List of interval breakpoints, e.g., [0, 0.3, 0.6, 1.0]
            densities: List of density coefficients for each interval, e.g., [18, 15, 8]
        """
        # Validate input legitimacy
        if len(breakpoints) - 1 != len(densities):
            raise ValueError("Number of breakpoints must be one more than number of density coefficients")
        
        if not np.all(np.diff(breakpoints) > 0):
            raise ValueError("Breakpoints must be strictly increasing")
        
        if any(d <= 0 for d in densities):
            raise ValueError("Density coefficients must be positive numbers")
        
        self.breakpoints = np.array(breakpoints)
        self.densities = np.array(densities)
        
        # Calculate normalization constant a
        intervals = np.diff(self.breakpoints)
        self.a = 1.0 / np.sum(intervals * self.densities)
        
        # Calculate breakpoints for cumulative distribution function (CDF)
        self.cdf_breakpoints = np.zeros(len(breakpoints))
        for i in range(1, len(breakpoints)):
            self.cdf_breakpoints[i] = self.cdf_breakpoints[i-1] + \
                                      intervals[i-1] * densities[i-1] * self.a
    
    def sample(self, size=1):
        """Sample from the distribution"""
        # Sample from uniform distribution U(0,1)
        u = np.random.rand(size)
        
        # Initialize sample array
        samples = np.zeros_like(u)
        
        # For each sample, determine the interval based on u value and apply inverse transformation
        for i in range(len(self.breakpoints) - 1):
            mask = (u >= self.cdf_breakpoints[i]) & (u < self.cdf_breakpoints[i+1])
            
            if np.any(mask):
                # Inverse transformation formula: x = (u - cdf[i]) / (density[i]*a) + breakpoint[i]
                samples[mask] = (u[mask] - self.cdf_breakpoints[i]) / \
                               (self.densities[i] * self.a) + self.breakpoints[i]
        
        return samples
    
    def logpdf(self, x):
        """Calculate log probability density"""
        # Initialize log probability array with default -inf (for points outside the range)
        log_probs = np.full_like(x, -np.inf, dtype=np.float64)
        
        # For each interval, check if x is within it and set corresponding log probability
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i+1])
            
            # Special handling for the right endpoint of the last interval (closed interval)
            if i == len(self.breakpoints) - 2:
                mask = mask | (x == self.breakpoints[i+1])
            
            if np.any(mask):
                log_probs[mask] = np.log(self.densities[i] * self.a)
        
        return log_probs
    
    def prob(self, x):
        """Calculate probability density"""
        return np.exp(self.log_prob(x))

class MixedNormalUniform:
    def __init__(self, normal_params, uniform_params, mix_ratio):
        """
        Initialize a mixed normal and uniform distribution
        
        Parameters:
            normal_params: Parameters for the normal distribution, in dictionary form {'loc': mean, 'scale': standard deviation}
            uniform_params: Parameters for the uniform distribution, in dictionary form {'low': lower bound, 'high': upper bound}
            mix_ratio: Weight of the normal distribution in the mixture (0 < mix_ratio < 1)
        """
        self.normal_loc = normal_params['loc']
        self.normal_scale = normal_params['scale']
        self.uniform_low = uniform_params['low']
        self.uniform_high = uniform_params['high']
        self.mix_ratio = mix_ratio
        
        # Validate parameter validity
        if not (0 < mix_ratio < 1):
            raise ValueError("mix_ratio must be in the range (0, 1)")
        if self.normal_scale <= 0:
            raise ValueError("Standard deviation of the normal distribution must be greater than 0")
        if self.uniform_high <= self.uniform_low:
            raise ValueError("Upper bound of the uniform distribution must be greater than the lower bound")

    def sample(self, n_samples=1):
        """Sample from the mixed distribution"""
        samples = []
        for _ in range(n_samples):
            # Select distribution based on mix ratio
            if np.random.rand() < self.mix_ratio:
                # Sample from normal distribution
                samples.append(norm.rvs(
                    loc=self.normal_loc, 
                    scale=self.normal_scale
                ))
            else:
                # Sample from uniform distribution
                samples.append(uniform.rvs(
                    loc=self.uniform_low, 
                    scale=self.uniform_high - self.uniform_low
                ))
        return np.array(samples)

    def pdf(self, x):
        """Calculate the probability density function value"""
        # Probability density of normal distribution
        normal_pdf = norm.pdf(
            x, 
            loc=self.normal_loc, 
            scale=self.normal_scale
        )
        
        # Probability density of uniform distribution
        uniform_pdf = uniform.pdf(
            x, 
            loc=self.uniform_low, 
            scale=self.uniform_high - self.uniform_low
        )
        
        # Mixed probability density
        return self.mix_ratio * normal_pdf + (1 - self.mix_ratio) * uniform_pdf

    def logpdf(self, x):
        """Calculate the log probability density function value"""
        return np.log(self.pdf(x))
    