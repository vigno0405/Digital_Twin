clc;
clear;
close all;

numParticles = 2500; % TRY TO CHANGE IT (to have estimations)

% This code estimates the state and the RUL; it exploits measurements of
% capacity to compute the likelihood, since it is needed to perform the
% computation (in that moment) of the likelihood! So the real measurement
% is actually needed (comparing it to the measurement equation).

%% Load Measurements

% Here we load some data we are going to use, as a measurement of Q(n)
load("CS2_36_resampled.mat")

% Plot measurements
% figure
% plot(cycles, capacity, 'LineWidth', 2)
% hold on
% grid on
% xlabel('Cycles','Interpreter','latex')
% ylabel('Capacity $[Ah]$','Interpreter','latex')
% title('Battery measurements','Interpreter','latex')
% xlim([cycles(1), cycles(end)])

% End of Life (EOL) definition
% The battery is considered dead when the EOL is 60% of the initial capacity
EOL_th = 0.6*capacity(1);
% End of Life index (when capacity "touches" EOL: first index)
EOL = find(capacity <= EOL_th, 1);
% True Residual Useful Life (RUL)
RUL_true = cycles(EOL) - cycles(1:EOL); % assumed linear
% yline(EOL_th, 'r', 'LineWidth', 2)
% xline(cycles(EOL), 'LineWidth', 2, 'LineStyle', '--');
% legend('Measurements', 'End of Life threshold', 'End of Life',...
%     'Location','southwest','Interpreter','latex')
% Here we have something deterministic, since we just keep the threshold as
% the remaining useful life.

%% Particle Filter Parameters initialization

% Number of time steps in which we update our state:
numMeasurements = length(cycles(1:EOL));
% how many times do we measure? We just consider the battery life to
% estimate the residual useful life, otherwise it is pointless.

% Number of parameters of the state (in this case the parameters of the
% degradation model: a, b, c):
numState = 3; % a, b, c
% Number of particles of the Particle Filter (samples). If:
% - n° of samples too low: very bad estimation, not finding good
%   distribution
% - too many particles: very large computational burden

% We preallocate one cell vector in which each element will contain the
% particles at each time step.
% At each time step, we have a matrix of size <Number of particles> x
% <Number of elements in the state>. This varies each time step.
particles = cell(1, numMeasurements);
weights = cell(1, numMeasurements);

% We preallocate one cell vector in which each element will contain the
% particles weights at each time step: every time we have a measure, we
% allocate particles and weights here


initialStateMean = [1.054, 0.021, 0.004823];

% INITIAL STATE, before we see any data (assumption)
initialCov = (diag(abs(initialStateMean))*0.2).^2;
% Initial state COVARIANCE (assumption: they are uncorrelated, because
% there are no extra-diagonal values)

%% Particle Filter code

% Initialize the particles: first element of the cell. Multivariate random
% samples having mean InitialStateMean and covariance matrix initialCov
% (assumed)
particles{1} = mvnrnd(initialStateMean,  initialCov, numParticles);
% How many samples? numParticles

for k = 1:length(cycles(1:EOL)) % For every measurement till the end of life

    % Evaluate the measurement equation and the likelihoods for every particle
    [lik] = measurementAndLikelihoodFunction(particles{k}, capacity(k), cycles(k));
    % capacity(k) we already have (MEASURE), particles with PROCESS EQ,
    % cycles is the INPUT

    % We NORMALIZE the likelihoods (weights) so that they sum to one
    lik = lik / sum(lik);
    weights{k} = lik;

    % We now RESAMPLE the particles according to their likelihood (WEIGHT):
    % multinomial distribution! Plot one if possible
    pd = makedist("Multinomial", "probabilities", lik);
    particles{k} = particles{k}(pd.random(numParticles, 1), :); % amond the previous ones, I take the most likely
    % Multinomial distribution: we are sampling according to lik as
    % probabilities

    % We now predict the next state according to the PROCESS equation or STATE TRANSITION function
    if k < length(cycles(1:EOL)) % We do this for all the time steps except for the last one
        particles{k + 1} = stateTransitionFunction(particles{k}); % i.e. process equation
    end
end
clearvars lik k

% Obtained: "numMeasurements" particles cells, with all the measurements

%% Plot state estimates for each time step

close all
% We preallocate matrices of size <number of time steps> x <number of
% elements in the state> for the mean, the 5% quantile and the 95% quantile
% Plotting all quantiles to have the boundaries (confidence)
m = nan(length(particles), numState); % Mean
q05 = nan(length(particles), numState); % 5th quantile 
q95 = nan(length(particles), numState); % 95th quantile

for counterCycle = 1:length(particles) % For each step
    % Compute the mean and quantiles
    m(counterCycle, :) =  mean(particles{counterCycle});
    q05(counterCycle, :) = quantile(particles{counterCycle}, 0.05);
    q95(counterCycle, :) = quantile(particles{counterCycle}, 0.95);
end

figure
temp = 'abc';
for state = 1:numState
    subplot(3,1,state)
    plot(cycles(1:EOL), m(:, state),'Color','b','LineWidth',2)
    hold on
    plot(cycles(1:EOL), q05(:, state), 'Color', 'r', 'LineWidth',2)
    plot(cycles(1:EOL), q95(:, state), 'Color', 'r','LineWidth',2)
    title(['State ', temp(state)],'Interpreter','latex')
    grid on
    xlabel('Cycles','Interpreter','latex')
    ylabel('State Estimate','Interpreter','latex')    
    fill([cycles(1:EOL)'; flipud(cycles(1:EOL)')], ...
     [q05(:, state); flipud(q95(:, state))], ...
     'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none') % 'FaceAlpha' controlla la trasparenza
    legend('Mean Estimate', '$q_{0.05}$', '$q_{0.95}$','','Interpreter','latex',...
        'Location','best')
end
% clearvars temp m q05 q95

% We see that increasing the number of cycles, we also increase the
% accuracy. We could CHANGE the number of cycles as a trial.
% However, whene we have high uncertainty it decreases more (it is easier
% to reduce a big error rather than a small one).

%% Plot state correlation

% cycle = cycles(100); % take a measurement every tot cycles (cycle != cycles(100)
% % Select one cycle: increasing the number of cycles, the assumption of
% % having three independent Gaussian distributions
% figure
% plotmatrix(particles{cycle == cycles})
% title("Cycle = " + string(cycle),'Interpreter','latex')
% corr(particles{cycle == cycles})
% clearvars cycle

% We said that there is no correlation between the parameters, but is it
% really like this? If we change cycles(N_TRY) we see that with increasing
% N_TRY the correlation happens, as we have it between the parameters.
% First cycle:
% - clouds of points are uncorrelated
% Then:
% - if we increase the number of cycles performed, we find the shape not
%   circular anymore! Then there is correlation
% So our assumption of uncorrelated Gaussian distribution does not hold
% anymore; TRY to put some extradiagonal elements!
% However, convergence of the algorithm happens anyway; with better
% covariance matrix we could only have better results.
% Correlation is due to the fact that parameters in the phenomenon exploit
% dependencies that we did not assume: this happens because particles tend
% to find regions with the maximum likelihood. If we assume correlation,
% maybe reflecting what is happening later, we could exploit better results
% but this correlation will be found anyway.

%% Plot capacity estimate

clc;
measFnc = @(a, b, c, k) a + b .* (1 - exp(c * k)); % Measurement function

% We preallocate a matrix to store the estimated capacity from the filter for each time step.
% The matrix will be of dimensions, <number of particles> x <number of time
% steps>
% Capacity is estimated using the measurement function we have; this is
% compared to the true capacity, which is an assumed data (provided at the
% beginning).

estimatedCapacity = nan(numParticles, length(cycles(1:EOL)));

for counterCycle = 1:length(cycles(1:EOL))
    estimatedCapacity(:, counterCycle) = measFnc(particles{counterCycle}(:, 1), particles{counterCycle}(:, 2), particles{counterCycle}(:, 3), cycles(counterCycle));
end

% Matlab statistics functions naturally operates along the column, so in
% this case by applying the mean function to the estimatedCapacity matrix,
% we will get a vector of size 1 x <number of time steps>. The same goes
% for the quantile function.

m = mean(estimatedCapacity);
q05 = quantile(estimatedCapacity, 0.05);
q95 = quantile(estimatedCapacity, 0.95);

figure
plot(cycles(1:EOL), m, 'Color','b','LineWidth',2,'LineStyle','--')
hold on
plot(cycles(1:EOL), q05,'LineWidth',2,'Color','r')
plot(cycles(1:EOL), q95,'Color','r','LineWidth',2)
plot(cycles(1:EOL), capacity(1:EOL),'LineWidth',2,...
    'Color','k')
grid on
xlabel('Cycles','Interpreter','latex')
ylabel('Capacity $[Ah]$','Interpreter','latex')
fill([cycles(1:EOL)'; flipud(cycles(1:EOL)')], ...
     [q05(:); flipud(q95(:))], ...
     'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
legend('Mean Estimate', '$q_{0.05}$', '$q_{0.95}$','Measured Capacity','', ...
    'Interpreter','latex')
title('Capacity estimate','Interpreter','latex')

% %% RUL
% 
% % We want to estimate the number of cycles we have left.
% % Compute true RUL from the particle filter. We preallocate a matrix of
% % size <number of particles> x <number of time steps>.
% 
% RUL_predicted = nan(numParticles, length(cycles(1:EOL)));
% 
% % parfor: to parallelize computation
% % For every cycle, we want to know how many cycles we still have (1:EOL)
% % Every time we are providing a new prediction of the remaining useful life
% 
% parfor counterCycle = 1:length(cycles(1:EOL)) % For each measurement until the true EOL
%     fprintf('Cycle #%d out of %d\n', counterCycle, length(cycles(1:EOL))) % Print how many cycles we have completed
%     cycleRUL = cycles(counterCycle); % Store the cycle at which we start predicting
%     for counterParticle = 1:numParticles % For every particle
%         cycle = cycleRUL; % Store the cycle at which we start predicting
%         predCapacity = EOL_th + 1; % This is just to enter the while cycle for the first time    
%         xTemp = particles{counterCycle}(counterParticle, :); % State at the cycle at which we start predicting
%         while predCapacity > EOL_th % While the predicted capacity is above the End of Life threshold
%             cycle = cycle + 1; % Predict one cycle ahead
%             xTemp = stateTransitionFunction(xTemp); % Use the process equation
%             predCapacity = measFnc(xTemp(1), xTemp(2), xTemp(3), cycle); % We predict the capacity
% %           _equivalent_: predCapacity = measFnc(particles{counterCycle}(counterParticle, 1), particles{counterCycle}(counterParticle, 2), particles{counterCycle}(counterParticle, 3), cycle);
%             if cycle > 1500 % If we predict that the remaining cycles are above 1500, then we exit the loop, since this is not realistic according to what we know about batteries
%                 break
%             end
%         end
% 
%         % When we exit the loop, it means that the predicted capacity has
%         % fallen below the threshold: we record the cycle and we count how
%         % many cycles ahead remains till this event happens!
%         % RUL computed as the difference:
% 
%         RUL_predicted(counterParticle, counterCycle) =  cycle - cycleRUL;
%     end
% end
% disp("RUL computed.")
% 
% %% Plot the RUL
% 
% RUL_mean = mean(RUL_predicted);
% RUL_05 = quantile(RUL_predicted, 0.05);
% RUL_95 = quantile(RUL_predicted, 0.95);
% 
% figure
% plot(cycles(1:EOL), RUL_mean, 'LineWidth',2,'Color','b','LineStyle','--')
% hold on
% plot(cycles(1:EOL), RUL_05,'Color','r','LineWidth',2)
% plot(cycles(1:EOL), RUL_95,'Color','r','LineWidth',2)
% plot(cycles(1:EOL), RUL_true,'Color','k','LineWidth',2)   % linear, computed as true RUL
% xlabel("Cycles",'Interpreter','latex')
% ylabel("RUL",'Interpreter','latex')
% fill([cycles(1:EOL)'; flipud(cycles(1:EOL)')], ...
%      [RUL_05(:); flipud(RUL_95(:))], ...
%      'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
% legend('Mean','$q_{0.05}$','$q_{0.95}$','True RUL','','Interpreter','latex')
% grid on
% title('RUL Estimation','Interpreter','latex')
% 
% % Accuracy decreases while the number of cycles decreases in time. The real
% % goal of the particle filter is to decrease this uncertainty when
% % increasing the number of cycles.
% % NOTE: when we underestimate the capacity of the battery, we underestimate
% % also the RUL, because we expect to have less cycles to live.
% 
% 
% %% Understanding what happens when we change the parameters
% 
% % 1. Number of particles
% %    Let's decrease it to a small number (e.g. 100): we have a smaller
% %    uncertainty than before, because so few particles easily fit the
% %    distribution. So by decreasing this number we "concentrate" them on a
% %    specific region: in fact, at each measurement we "cut" some particles
% %    and resample in a different zone, but this way we get narrower
% %    distributions. However, we lose the capacity to describe the regions
% %    around the mean. Obviously, running becomes faster (less particles).
% %    Estimation is not so bad, but the reason is that we have a very simple
% %    example, since the RUL we are estimating is actually linear!
% %    OBS. When is the RUL not linear? When, for instance, we have a
% %    periodic load.
% %    Let's increase the number of particles (e.g 5000, or 20000):
% %    uncertainty is increasing, but this is not going to last forever: from
% %    a certain point on, uncertainty starts decreasing.
% % 2. Initial state mean and mean covariance (a, b, c)
% %    They depend on how much we know about the parameters (in the process
% %    equation): if we are quite sure about the initial value, we can have
% %    low sigma, otherwise not. For example (keeping num_particles=2500):
% %    - mean multiplied by 0.8: converges to the right estimation then (if
% %      number of particles is enough), except when this value is too low
% %      (since it has to "move" too much). So they will adjust if they have
% %      the same order of magnitude.
% %    - covariance multiplied by 5: it converges to the right distribution,
% %      but at the beginning I have more uncertainty.
% %    OBS. Under- or over-estimation depends on the stochastic nature of the
% %    simulation: every time we are filtering we perform a random sampling
% % 3. Measurement noise (in measurement equation)
% %    This is however something we cannot change in the simulation: we can
% %    only change the sensor, by selecting it when designing the sensor
% %    network. The sensor characteristics can be taken into account when
% %    simulating the system -> process not just in the usage of the
% %    component, but also in the design phase.
% %    If it INCREASES, the uncertainty increases, since all measurements are
% %    affected by an higher noise. This is why we need a proper sensor.
% % 4. sigma_a, sigma_b, sigma_c in the process (transition) function
% %    If I increase them, uncertainty increases. Note that in this case I
% %    have not a defined process law, like the Paris law, but simply a
% %    random walk, taking the old sample and adding a certain noise. Look
% %    also at the states graphs: we are propagating with a BIGGER JUMP!
% %    RUL is very bad, with huge uncertainty.
% 
% % OBS. Why filtering? We have a certain set of particles, but we take only
% % the distribution of particles concentrated in the "right" values -> then
% % we resample to have "concentrated" particles. So I decide to filter the
% % particles to concentrate on the right region, and this is done by Bayes
% % theorem (through the use of misurations).
% % In fact, at a given time instant, I have that the capacity uncertainty
% % increases if I propagate up to a certain threshold. How? Through the
% % PROCESS EQUATION, APPLIED TO EACH SINGLE PARTICLE! If I do something like
% % this I clearly see that my distribution has this behavior; particles are
% % close if we are at the end, then uncertainty is not increasing; if we are
% % at the beginning, uncertainty is huge instead and it propagates more!
% 
% %% Plot Predicted Capacity for Specific Cycles
% % Partendo da cycles(50) e cycles(120), calcoliamo i valori predetti della capacità
% % clc
% % close all
% % 
% % start_cycles = [40, 60]; % Cicli da cui partire
% % colors = ['b', 'g']; % Colori per i plot
% % 
% % for i = 1:length(start_cycles)
% %     % Indice del ciclo di partenza
% %     start_idx = start_cycles(i);
% % 
% %     % Predizione della capacità per ogni particella dal ciclo selezionato
% %     predictedCapacity = nan(numParticles, length(cycles(start_idx:end)));
% %     for counterParticle = 1:numParticles
% %         xTemp = particles{start_idx}(counterParticle, :); % Stato iniziale
% %         for k = start_idx:length(cycles)
% %             predictedCapacity(counterParticle, k - start_idx + 1) = ...
% %                 measFnc(xTemp(1), xTemp(2), xTemp(3), cycles(k));
% %             xTemp = stateTransitionFunction(xTemp); % Stato successivo
% %         end
% %     end
% % 
% %     % Calcolo media e quantili
% %     meanPredicted = mean(predictedCapacity, 1);
% %     q05Predicted = quantile(predictedCapacity, 0.05);
% %     q95Predicted = quantile(predictedCapacity, 0.95);
% % 
% %     estimatedCapacity = nan(numParticles, length(cycles(1:start_idx-1)));
% % 
% %     for counterCycle = 1:length(cycles(1:start_idx-1))
% %         estimatedCapacity(:, counterCycle) = measFnc(particles{counterCycle}(:, 1), particles{counterCycle}(:, 2), particles{counterCycle}(:, 3), cycles(counterCycle));
% %     end
% % 
% %     % Combina la capacità stimata precedente con la nuova predizione
% %     meanPredictedCombined = [mean(estimatedCapacity), meanPredicted];
% %     q05Combined = q05Predicted;
% %     q95Combined = q95Predicted;
% % 
% %     % Plot dei risultati
% %     figure
% %     plot(cycles, meanPredictedCombined, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--')
% %     hold on
% %     ylim([0.5,1.5])
% %     yline(EOL_th, 'LineWidth', 2, 'Color', [0, 0.5, 0], 'LineStyle', '--') % Verde scuro
% %     xline(cycles(start_idx), 'LineWidth', 2, 'Color', [0.5, 0, 0.5], 'LineStyle', '--') % Viola
% %     plot(cycles(start_idx:end), q05Combined, 'Color', 'r', 'LineWidth', 1.5)
% %     plot(cycles(start_idx:end), q95Combined, 'Color', 'r', 'LineWidth', 1.5)
% %     plot(cycles, capacity, 'k', 'LineWidth', 2) % Capacita reale
% %     fill([cycles(start_idx:end)'; flipud(cycles(start_idx:end)')], ...
% %          [q05Combined(:); flipud(q95Combined(:))], ...
% %          'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
% %     grid on
% %     xlabel('Cycles', 'Interpreter', 'latex')
% %     ylabel('Capacity $[Ah]$', 'Interpreter', 'latex')
% %     legend('Mean Predicted','','','$q_{0.05}$', '$q_{0.95}$', 'Measured Capacity', '', ...
% %         'Interpreter', 'latex', 'Location', 'best')
% %     title(['Predicted Capacity starting from Cycle ', num2str(cycles(start_idx))], 'Interpreter', 'latex')
% % end
% % 
% % % Fine del codice
