function likelihood = measurementAndLikelihoodFunction(predictedParticles, measurement, n)

arguments % initialization, to compute the likelihood for the parameters
    predictedParticles (:, :) {mustBeNumeric, mustBeReal} % It must be a scalar and a real numeric number
    measurement (1, 1) {mustBeNumeric, mustBeReal} % It must be a scalar % This is Q_k in the slides
    n (1, 1) {mustBeNumeric, mustBeReal} % It must be a scalar % THis is the cycle number
end

sigmaNoise = 0.05; % Sigma_Q in the slides
% sigmaNoise = 0.01; % Sigma_Q in the slides

measFnc = @(a, b, c) a + b * (1 - exp(c*n)); % measurement function (as in slides)

likFnc = @(x) 1./(2*pi*sigmaNoise^2)^(1/2) * exp(-1/(2*sigmaNoise^2)*(measurement - x)^2);
% usual gaussian likelihood function, with the measurement

likelihood = nan(size(predictedParticles, 1), 1);

for counter = 1:size(predictedParticles, 1) % For every particle
    a_counter = predictedParticles(counter, 1); % Take a for this particle
    b_counter = predictedParticles(counter, 2); % Take b for this particle
    c_counter = predictedParticles(counter, 3); % Take c for this particle
    likelihood(counter) = likFnc(measFnc(a_counter, b_counter, c_counter));
    % For each sample in the cell, we compute the likelihood function for
    % that counter (i.e. for every particle we have, in predictedParticles)
end


end