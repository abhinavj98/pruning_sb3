class BetaProbabilityDistribution(Distribution):
     def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.alpha = None
        self.beta = None
        # as per http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
       
    
      def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)
        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        alpha = nn.Linear(latent_dim, self.action_dim)
        alpha_activation = nn.Softplus()
        beta = nn.Linear(latent_dim, self.action_dim)
        beta_activation = nn.Softplus()
        return nn.Sequential(alpha, alpha_activation), nn.Sequential(beta, beta_activation)
        # alpha = 1.0 + nn.Linear(flat, len(flat.shape)-1, activation=th.nn.softplus)
        # beta  = 1.0 + th.layers.dense(flat, len(flat.shape)-1, activation=th.nn.softplus)
        # self.dist = th.distributions.Beta(concentration1=alpha, concentration0=beta, validate_args=True, allow_nan_stats=False)

    def proba_distribution(
        self: SelfDiagGaussianDistribution, alpha: th.Tensor, beta: th.Tensor
    ) -> SelfDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)
        :param mean_actions:
        :param log_std:
        :return:
        """
        alpha = 1 + alpha
        beta = 1 + beta
        self.distribution = th.distributions.beta.Beta(alpha, beta, validate_args=True)
        return self


    def flatparam(self):
        return self.flat

    def mode(self):
        return self.dist.mode()

    def neglogp(self, x):
        return tf.reduce_sum(-self.dist.log_prob(x), axis=-1)

    def kl(self, other):
        assert isinstance(other, BetaProbabilityDistribution)
        return self.dist.kl_divergence(other.dist)

    def entropy(self):
        return self.dist.entropy()

    def sample(self):
        return self.dist.sample()

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)