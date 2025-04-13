function particles = stateTransitionFunction(previousParticles)

sigma_a = 1e-3;
sigma_b = 1e-4;
sigma_c = 1e-5;
sigmaCov = diag([sigma_a, sigma_b, sigma_c] .^ 2);
numParticles = size(previousParticles, 1);
particles = previousParticles + mvnrnd([0, 0, 0], sigmaCov, numParticles);
% multivariate same as univariate since we assume they are uncorrelated

end