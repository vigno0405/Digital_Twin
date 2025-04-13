clc;
clear;
close all;

% "Create" the experimental data
leakage_diameter = 1.5e-4; % [m]
leakage_area = (leakage_diameter.^2) * pi /4;
leak = leakage_area;
out = sim("Pressure_vessel_model.slx");
temp = find(out.logsout, 'Name', 'LeakCumulativeOutVolume');

%% Adding noise

measurementsWithoutNoise = temp{1}.Values.Data(1:10:end);
simulationTime = out.tout(1:10:end);

noiseSTD = 2.5e-5; 
measurements = measurementsWithoutNoise + noiseSTD * randn(length(measurementsWithoutNoise), 1);

figure
plot(simulationTime, measurementsWithoutNoise)
% Clean output of our simulation, without putting any noise by sensors
hold on
plot(simulationTime, measurements,'LineWidth',1.5)
legend('Ideal measurements','Noisy measurements','Interpreter','latex','Location','best')
title('Measurements','Interpreter','latex')
xlabel('Time [s]','Interpreter','latex')
ylabel('Leakage Volume [$\mathrm{m^3}$]', 'Interpreter', 'latex')
hold off

%% Solve MCMC

% Note: the result is not deterministic, since we are sampling in accepting
% (and the result can be different all the times, also plotting the
% histograms).
rng('shuffle');

% Define the prior on the Leakage Area times 1e9:
priorPd = makedist("Weibull", "a", 40, "b", 1.5); % p

prior = @(x) priorPd.pdf(x * 1e9);

prior(leakage_area)

% Defining q, proposal distribution:
proposalSTD = 3e-9;
proposal = @(x) x +  proposalSTD * randn;

init = 1e-3;
x =  pi * (init) ^ 2 / 4; % Starting Point for the leakage area

D = linspace(0, 0.6, 100000)*1e-3;
area = (D.^2) * pi /4;
prior1 = prior(area);

% % Primo grafico: prior in funzione dell'area
% figure;
% plot(area, prior1, 'LineWidth', 1.5);
% xlabel('Leakage Area [$m^2$]', 'Interpreter', 'latex');
% ylabel('Prior pdf', 'Interpreter', 'latex');
% title('Prior pdf vs Area', 'Interpreter', 'latex');
% grid on;
% 
% % Secondo grafico: prior in funzione del diametro
% figure;
% plot(D * 1e3, prior1, 'LineWidth', 1.5); % D in mm
% xlabel('Leakage Diameter [$mm$]', 'Interpreter', 'latex');
% ylabel('Prior pdf', 'Interpreter', 'latex');
% title('Prior pdf vs Diameter', 'Interpreter', 'latex');
% grid on;

%% x = leak; % We start from the true value for the leakage area

close all;

% Creating the Markov chain:
numberOfSamples = 5000;
chain = nan(numberOfSamples, 1); % We are storing the samples in here
acceptanceRatios = nan(floor(numberOfSamples/50), 1); % Acceptance ratios
chain(1) = x;
prior_x = log(prior(x)); % needed to 

% Use the surrogate model:
load("Pressure_vessel_trainedNetwork.mat")
surrogateOutput = @(x) Y_mean + Y_std * sim(net, ( [simulationTime,...
    repmat(x, length(simulationTime), 1)]' - X_mean') ./ X_std');
% This is done because we normalized values inside the Neural Network, i.e.
% we denormalize values.
%%
% Function to compute the likelihood:
likelihood_x = likelihoodFnc(measurements, surrogateOutput(x), noiseSTD);

acceptedSamples = nan(numberOfSamples, 1);
acceptedSamples(1) = true;

vediamo = nan(numberOfSamples,1)
vediamo(1)=0;

for counter = 2:size(chain, 1)
    xNew = proposal(x); % Proposed state
    prior_xNew = log(prior(xNew)); % prior, given the sampled one
    % New likelihood (always log for numerical stability):
    likelihood_xNew = likelihoodFnc(measurements, surrogateOutput(xNew), noiseSTD);
    % Use alpha to evaluate if accepting or not:
    logAlpha = likelihood_xNew - likelihood_x + prior_xNew - prior_x;
    vediamo(counter) = prior_xNew;
    % log(rand) used to sample from U(0, 1):
    logr = log(rand);
    if logr < logAlpha % We accept
        chain(counter) = xNew; % Store in this case
        x = xNew; % Overwrite with the new value
        % Overwrite also prior and likelihood
        prior_x = prior_xNew;
        likelihood_x = likelihood_xNew;
        acceptedSamples(counter) = true;
    else
        chain(counter) = x; % x, not xNew! We store the previous value
        acceptedSamples(counter) = false;
    end
    if rem(counter, 50) == 0 % Every 50 samples (remainder)
       disp('Acceptance ratio: ')
       disp(min(sum(acceptedSamples(counter-49:counter))/49, 1)); % average
       acceptanceRatios(floor(counter/50)) = min(sum(acceptedSamples(counter-49:counter))/49, 1);
       % here stable, about 0.6
    end
end

%% Plot the chain and the acceptance ratio

close all;
figure
plot(chain)
hold on
xlabel('Samples','Interpreter','latex')
ylabel('Chain samples: leakage area','Interpreter','latex')
yline(leak, "LineWidth", 4)
legend("Chain", "True leakage area",'Interpreter','latex')
title('Markov Chain', 'Interpreter','latex')
hold off

% We observe that the chain stabilizes. True leakage area was imposed at
% the beginning of the simulation; we see that the chain tends to
% overestimate the leakage area, but the error is very small, thus
% reasonable.

figure
plot(acceptanceRatios,'o')
hold on
ylim([0 1])
xlabel('Samples','Interpreter','latex')
ylabel('Acceptance ratio','Interpreter','latex')
yline(mean(acceptanceRatios), "LineWidth", 2, 'LineStyle', '--', 'Color', 'red');
legend('$\alpha$', '$\bar{\alpha}$','Interpreter','latex')
title('$\alpha$', 'Interpreter', 'latex');

%% Erase the burn-in period and apply thinning

chainThinned = chain(1000:10:end); % observing it (not automated)
% Thinning: we take a sample every 10 from the chain (removing
% autocorrelation, i.e. dependances between consecutive samples).
% Burn-in period: we remove the initial part of the path, when we were
% still 'stuck' to the tentative part.
figure
plot(chainThinned)
hold on
xlabel('Thinned samples','Interpreter','latex')
ylabel('Chain samples: leakage area','Interpreter','latex')
title('Thinned Markov Chain','Interpreter','latex')
yline(leak, "LineWidth", 4)
legend("Thinned Chain", "True leakage area",'Interpreter','latex')
hold off

%% Plot the leakage area distribution

% To do so, we plot as histogram the thinned chain, keeping the values'
% occurrence as the metric to evalutate the chain's behavior. This plots
% the posterior distribution. Here we see that, given a certain measurement
% z, this distribution is the leakage are: a little bias is still present.
figure
histogram(chainThinned, 'Normalization','pdf')
xlabel('Leakage Area $[m^2]$','Interpreter','latex')
ylabel('Frequency','Interpreter','latex')
xline(leak, "LineWidth", 4)
legend("Leakage Area PD", "True Leakage Area",'Interpreter','latex')
title('Leakage Area Distribution','Interpreter','latex')

%% Change the variance of the proposal distribution

% Very small variance: 1e-12 (three orders less...) -> this way we accept
% all the samples, with acceptance ratio which is often 1. Accepting
% everything, it is very unlikely that we move to the right region:
% asymptotically we will, but we cannot right now! This happens (less) with
% 1e-10, but converging a bit: however, autocorrelation is too high and all
% the data are repeating the previous!
% Variance too big: lots of samples are rejected, so we get 'stuck' with
% the same values all the time.

%% Boia

close all

% Definizione dei valori iniziali
simulationTime;
x1 = 5e-4; % Primo valore di x (in m)
x2 = 1e-3; % Secondo valore di x (in m)

% Calcolo delle aree di leakage
leakage_area1 = pi * (x1)^2 / 4; 
leakage_area2 = pi * (x2)^2 / 4;

% Output del surrogate model
out1 = surrogateOutput(leakage_area1);
out2 = surrogateOutput(leakage_area2);

% Valori della prior
prior1 = prior(leakage_area1);
prior2 = prior(leakage_area2);

% Creazione della prima figura
figure;
plot(simulationTime, out1, 'LineWidth', 1.5);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Cumulative Leakage Volume', 'Interpreter', 'latex');
title('$Prior: 1.57 \cdot 10^{-6}$', 'Interpreter', 'latex');
grid on;

% Creazione della seconda figura
figure;
plot(simulationTime, out2, 'LineWidth', 1.5);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Cumulative Leakage Volume', 'Interpreter', 'latex');
title('$Prior: 2.72 \cdot 10^{-39}$', 'Interpreter', 'latex');
grid on;
