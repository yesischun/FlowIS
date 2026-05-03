import numpy as np


class RejectAcceptSampling:
    """Rejection-acceptance samplers for conditional behavior generation."""

    def __init__(self, dimensionality=1):
        self.dimensionality = dimensionality

    @staticmethod
    def rejection_sampling_1D(
        scene,
        NDE_intersection,
        dim=[0, 1, 2, 3, 4, 5, 6, 7],
        max_trials=1000,
        random_state=None,
        type='sample',
    ):
        """Sample a 1D behavior from p(a|s) via rejection sampling."""
        np.random.seed(random_state)
        accept = 0

        for trial_idx in range(max_trials):
            while True:
                accelerate = NDE_intersection['proposal_accelerate'].sample()
                if 0 <= accelerate <= 1:
                    log_proposal_accelerate = NDE_intersection['proposal_accelerate'].logpdf(accelerate)
                    break

            scene_behavior = np.concatenate([scene[0], accelerate])

            log_p_as_val = NDE_intersection['scene_behavior'].logpdf(scene_behavior.reshape(1, -1))
            log_p_s_val = NDE_intersection['scene_behavior'].logpdf_marginal(scene, dim)
            log_p_cond_val = log_p_as_val - log_p_s_val

            log_alpha = log_p_cond_val - (
                np.log(NDE_intersection['proposal_M']) + log_proposal_accelerate
            )

            if np.log(np.random.rand()) < log_alpha:
                if type == 'sample':
                    return scene_behavior, log_p_cond_val, trial_idx + 1
                accept += 1

        if type == 'sample':
            # Keep sampler failure silent; caller handles fallback policy.
            return None, None, max_trials
        return accept / max_trials

    @staticmethod
    def rejection_sampling_2D(
        scene,
        NDE_intersection,
        dim=[0, 1, 2, 3, 4, 5, 6, 7],
        max_trials=1000,
        random_state=None,
        type='sample',
    ):
        """Sample a 2D behavior from p(a,b|s) via rejection sampling."""
        np.random.seed(random_state)
        accept = 0

        for trial_idx in range(max_trials):
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

            log_p_as_val = NDE_intersection['scene_behavior'].logpdf(scene_behavior.reshape(1, -1))
            log_p_s_val = NDE_intersection['scene_behavior'].logpdf_marginal(scene, dim)
            log_p_cond_val = log_p_as_val - log_p_s_val

            log_alpha = log_p_cond_val - (
                np.log(NDE_intersection['proposal_M'])
                + log_proposal_accelerate
                + log_proposal_steering_angle
            )

            if log_p_cond_val > 4 or np.log(np.random.rand()) < log_alpha:
                if type == 'sample':
                    return scene_behavior, log_p_cond_val, trial_idx + 1
                accept += 1

        if type == 'sample':
            # Keep sampler failure silent; caller handles fallback policy.
            return None, None, max_trials
        return accept / max_trials

    @staticmethod
    def _gmm_params(gmm_obj):
        """Extract (weights, means, covariances) from sklearn/custom GMM-like objects."""
        weights = getattr(gmm_obj, "weights_", None)
        means = getattr(gmm_obj, "means_", None)
        covs = getattr(gmm_obj, "covariances_", None)
        if weights is None:
            weights = getattr(gmm_obj, "weights", None)
        if means is None:
            means = getattr(gmm_obj, "means", None)
        if covs is None:
            covs = getattr(gmm_obj, "covariances", None)
        if weights is None or means is None or covs is None:
            return None, None, None
        return np.asarray(weights, dtype=float), np.asarray(means, dtype=float), np.asarray(covs, dtype=float)

    @staticmethod
    def _log_mvn(x, mean, cov):
        """Numerically stable log N(x | mean, cov)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        k = x.shape[0]
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(k)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 1e-6 * np.eye(k)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return -np.inf
        diff = x - mean
        sol = np.linalg.solve(cov, diff)
        quad = float(diff.T @ sol)
        return float(-0.5 * (k * np.log(2.0 * np.pi) + logdet + quad))

    @staticmethod
    def conditional_sample_gmm(scene, distribution, scene_dims, behavior_dims, random_state=None):
        """
        Exact conditional sampling for Gaussian-mixture joint model p(s, a):
        sample a ~ p(a|s) without rejection.
        Returns (scene_behavior_norm, log_q_cond, trials)
        """
        if random_state is not None:
            np.random.seed(random_state)

        gmm = distribution.get("scene_behavior", None) if isinstance(distribution, dict) else None
        if gmm is None:
            return None, None, 1
        weights, means, covs = RejectAcceptSampling._gmm_params(gmm)
        if weights is None:
            return None, None, 1

        scene = np.asarray(scene, dtype=float).reshape(1, -1)
        s = scene[0]
        scene_dims = [int(i) for i in scene_dims]
        behavior_dims = [int(i) for i in behavior_dims]
        if len(behavior_dims) == 0:
            return None, None, 1

        n_comp, full_dim = means.shape
        log_post = np.full(n_comp, -np.inf, dtype=float)
        for k in range(n_comp):
            mu = means[k]
            cov = covs[k]
            mu_s = mu[scene_dims]
            ss = cov[np.ix_(scene_dims, scene_dims)]
            log_post[k] = np.log(max(weights[k], 1e-300)) + RejectAcceptSampling._log_mvn(s, mu_s, ss)

        m = np.max(log_post)
        if not np.isfinite(m):
            return None, None, 1
        probs = np.exp(log_post - m)
        z = np.sum(probs)
        if z <= 0 or (not np.isfinite(z)):
            return None, None, 1
        probs = probs / z
        k = int(np.random.choice(np.arange(n_comp), p=probs))

        mu = means[k]
        cov = covs[k]
        mu_s = mu[scene_dims]
        mu_b = mu[behavior_dims]
        ss = cov[np.ix_(scene_dims, scene_dims)]
        bb = cov[np.ix_(behavior_dims, behavior_dims)]
        bs = cov[np.ix_(behavior_dims, scene_dims)]
        sb = cov[np.ix_(scene_dims, behavior_dims)]

        ss = 0.5 * (ss + ss.T) + 1e-9 * np.eye(len(scene_dims))
        inv_ss = np.linalg.inv(ss)
        cond_mean = mu_b + bs @ inv_ss @ (s - mu_s)
        cond_cov = bb - bs @ inv_ss @ sb
        cond_cov = 0.5 * (cond_cov + cond_cov.T) + 1e-9 * np.eye(len(behavior_dims))

        a = np.random.multivariate_normal(cond_mean, cond_cov, size=1).reshape(-1)

        out = np.zeros((1, full_dim), dtype=float)
        out[0, :] = mu
        out[0, scene_dims] = s
        out[0, behavior_dims] = a

        try:
            log_joint = distribution["scene_behavior"].logpdf(out)
            log_marg = distribution["scene_behavior"].logpdf_marginal(scene, scene_dims)
            log_q_cond = float(np.asarray(log_joint - log_marg).reshape(-1)[0])
        except Exception:
            return None, None, 1
        return out, log_q_cond, 1
