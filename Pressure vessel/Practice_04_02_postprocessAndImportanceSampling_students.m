clc;
clear;
close all;

%% Find all the files in the folder "Simulations" with the extension ".mat"
simulationList = dir("Simulations\*.mat");

% Preallocate cells to store the output data we want from the simulations
simCracks = cell(1, length(simulationList));
simDefects = cell(1, length(simulationList));

% Putting all the crack samples inside an array
for counter = 1:length(simulationList)
    clearvars defects cracks leakages out % This is just for precaution
    % Load the .mat file
    load([simulationList(counter).folder '\' simulationList(counter).name])
    % Find defects, cracks and leakages
    defects = find(out.logsout, 'Name', 'smallDefects');
    cracks = find(out.logsout, '-regexp', 'BlockPath', '.*/Hold and release');
    leakages = find(out.logsout, 'Name', 'leakages');
    % Store the leakages and the cracks depths in the same cell
    simCracks{counter} = [cracks{1}.Values.a.Data(cracks{1}.Values.a.Time == cracks{1}.Values.a.Time(end));... % the ones at the end of the simulation! Not considering time...
            leakages{1}.Values.a.Data];
    % Stire the defects depths in another cell
    simDefects{counter} = defects{1}.Values.a.Data; 
end

% Put all the crack depths from all the simulations in this vector
crackSamples = vertcat(simCracks{:});

%% Plot the histogram of the crack depths
figure
histogram(crackSamples, 'Normalization','pdf')
hold on
xlabel("Crack depth [mm]")
ylabel("Histogram and pdf")

% This fit is not what it should be done in practice, it is just to show
% the effect of importance sampling!
% In practice we should apply importance sampling on the SimEvents model,
% not on the kernel distribution!

% Fit kernel distribution and plot it
kernPd = fitdist(crackSamples, 'kernel', 'Support', [0.4, 5]);
% Fitting inside a kernel distribution
plot(linspace(0, 5, 100), kernPd.pdf(linspace(0, 5, 100)), 'r', 'LineWidth', 2)
hold on
title('Fitted kernel distribution')
x_vertical = 3.2 * 0.95;
xline(x_vertical, '--', 'Color', [0.5, 0, 0.5], 'LineWidth', 2);
hold off
% To sample 1 sample from a probability distribution object in Matlab you can use
random(kernPd, 1, 1)

%% Without importance sampling

% Evaluate the cost sampling 1000 cracks/defects
numSamples = 1000;
cost_MC = zeros(1, numSamples); % Preallocate the cost

% In reality we should run 1000 times the model and obtain 1000 samples!
for counter = 1:length(cost_MC)
    sample = random(kernPd, 1, 1);
    cost_MC(counter) = costFunction(sample);
end

cost_MC = costFunction(crackSamples);

% Expected cost and variance
[costMean, costMeanMCse] = computeMCstats(cost_MC)

% Expected variance of the cost and MonteCarlo standard error
[costVariance, costVarianceMCse] = computeMCstats((cost_MC - costMean).^2)
% (also from session 2)

%% Evaluation of results

% Number of Monte Carlo runs
numRuns = 100;
numSamples = 1000;

% Preallocate arrays for storing results
allCostMeans = zeros(1, numRuns);
allCostMeanMCse = zeros(1, numRuns);
allCostVariances = zeros(1, numRuns);
allCostVarianceMCse = zeros(1, numRuns);

for runIdx = 1:numRuns
    % Preallocate the cost array for the current run
    rng('shuffle');
    cost_MC = zeros(1, numSamples);
    
    % Generate samples and compute costs
    for counter = 1:length(cost_MC)
        sample = random(kernPd, 1, 1);
        cost_MC(counter) = costFunction(sample);
    end
    
    % Compute statistics for the current run
    [costMean, costMeanMCse] = computeMCstats(cost_MC);
    [costVariance, costVarianceMCse] = computeMCstats((cost_MC - costMean).^2);
    
    % Store results
    allCostMeans(runIdx) = costMean;
    allCostMeanMCse(runIdx) = costMeanMCse;
    allCostVariances(runIdx) = costVariance;
    allCostVarianceMCse(runIdx) = costVarianceMCse;
end

% Histogram of cost means
figure
histogram(allCostMeans); % Usa 30 bin
title('Cost Means','Interpreter','latex');
ylabel('Frequency');

% Histogram of cost mean MC standard errors
figure
histogram(allCostMeanMCse); % Usa 30 bin
title('Cost Mean MCSE', 'Interpreter', 'latex');
ylabel('Frequency');

% Histogram of cost variances
figure
histogram(allCostVariances); % Usa 30 bin
title('Cost Variance','Interpreter','latex');
ylabel('Frequency');

% Histogram of cost variance MC standard errors
figure
histogram(allCostVarianceMCse); % Usa 30 bin
title('Cost Variance MCSE','Interpreter','latex');
ylabel('Frequency');

%% Varying N

% Evaluate the cost sampling with varying numSamples
numSamplesArray = 100:100:10000;
numIterations = length(numSamplesArray);

% Preallocate arrays for results
meanValues = zeros(1, numIterations);
meanMCseValues = zeros(1, numIterations);

for idx = 1:numIterations
    rng('shuffle');
    numSamples = numSamplesArray(idx);
    cost_MC = zeros(1, numSamples); % Preallocate the cost array

    % Generate samples and compute costs
    for counter = 1:numSamples
        sample = random(kernPd, 1, 1);
        cost_MC(counter) = costFunction(sample);
    end

    % Compute expected cost and Monte Carlo standard error
    [costMean, costMeanMCse] = computeMCstats(cost_MC);

    % Store results
    meanValues(idx) = costMean;
    meanMCseValues(idx) = costMeanMCse;
end

% Plot results
figure;

% Plot Mean vs numSamples
subplot(2, 1, 1);
plot(numSamplesArray, meanValues, '-o');
title('Mean of Cost vs Number of Samples');
xlabel('Number of Samples');
ylabel('Mean of Cost');
grid on;

% Plot Mean MCSE vs numSamples
subplot(2, 1, 2);
plot(numSamplesArray, meanMCseValues, '-o');
title('Monte Carlo Standard Error of Mean vs Number of Samples');
xlabel('Number of Samples');
ylabel('Mean MCSE');
grid on;

%% With importance sampling (1000 cracks/defects)

clc
numSamples = 1000;
q = makedist("Uniform", "lower", 0, "upper", 5);
cost_MC_IS = zeros(1, numSamples);

for counter = 1:length(cost_MC_IS)
    sample = random(q, 1, 1);
    cost_MC_IS(counter) = costFunction(sample) * ...
            kernPd.pdf(sample)/q.pdf(sample);
    % kernPd.pdf is a way to take a sample from this distribution
end

% Expected cost and variance
[costMean_IS, costMeanMCse_IS] = computeMCstats(cost_MC_IS)

% Expected variance of the cost and MonteCarlo standard error
[costVariance_IS, costVarianceMCse_IS] = ...
        computeMCstats((cost_MC_IS - costMean_IS).^2)

%% Evaluation of results

q = makedist("Uniform", "lower", 0, "upper", 5);
% Number of Monte Carlo runs
numRuns = 100;
numSamples = 1000;

% Preallocate arrays for storing results
allCostMeans = zeros(1, numRuns);
allCostMeanMCse = zeros(1, numRuns);
allCostVariances = zeros(1, numRuns);
allCostVarianceMCse = zeros(1, numRuns);

for runIdx = 1:numRuns
    % Preallocate the cost array for the current run
    rng('shuffle');
    cost_MC_IS = zeros(1, numSamples);
    
    % Generate samples and compute costs
    for counter = 1:length(cost_MC_IS)
    sample = random(q, 1, 1);
    cost_MC_IS(counter) = costFunction(sample) * ...
            kernPd.pdf(sample)/q.pdf(sample);
    % kernPd.pdf is a way to take a sample from this distribution
    end
    
    % Compute statistics for the current run
    [costMean, costMeanMCse] = computeMCstats(cost_MC_IS);
    [costVariance, costVarianceMCse] = computeMCstats((cost_MC_IS - costMean).^2);
    
    % Store results
    allCostMeans(runIdx) = costMean;
    allCostMeanMCse(runIdx) = costMeanMCse;
    allCostVariances(runIdx) = costVariance;
    allCostVarianceMCse(runIdx) = costVarianceMCse;
end

% Histogram of cost means
figure
histogram(allCostMeans); % Usa 30 bin
title('Cost Means','Interpreter','latex');
ylabel('Frequency');

% Histogram of cost mean MC standard errors
figure
histogram(allCostMeanMCse); % Usa 30 bin
title('Cost Mean MCSE', 'Interpreter', 'latex');
ylabel('Frequency');

% Histogram of cost variances
figure
histogram(allCostVariances); % Usa 30 bin
title('Cost Variance','Interpreter','latex');
ylabel('Frequency');

% Histogram of cost variance MC standard errors
figure
histogram(allCostVarianceMCse); % Usa 30 bin
title('Cost Variance MCSE','Interpreter','latex');
ylabel('Frequency');

%% Varying N

% Evaluate the cost sampling with varying numSamples
numSamplesArray = 100:100:10000;
numIterations = length(numSamplesArray);

% Preallocate arrays for results
meanValues = zeros(1, numIterations);
meanMCseValues = zeros(1, numIterations);

for idx = 1:numIterations
    rng('shuffle');
    numSamples = numSamplesArray(idx);
    cost_MC_IS = zeros(1, numSamples); % Preallocate the cost array

    % Generate samples and compute costs
    % Generate samples and compute costs
    for counter = 1:length(cost_MC_IS)
    sample = random(q, 1, 1);
    cost_MC_IS(counter) = costFunction(sample) * ...
            kernPd.pdf(sample)/q.pdf(sample);
    % kernPd.pdf is a way to take a sample from this distribution
    end

    % Compute expected cost and Monte Carlo standard error
    [costMean, costMeanMCse] = computeMCstats(cost_MC_IS);

    % Store results
    meanValues(idx) = costMean;
    meanMCseValues(idx) = costMeanMCse;
end

% Plot results
figure;

% Plot Mean vs numSamples
subplot(2, 1, 1);
plot(numSamplesArray, meanValues, '-o');
title('Mean of Cost vs Number of Samples');
xlabel('Number of Samples');
ylabel('Mean of Cost');
grid on;

% Plot Mean MCSE vs numSamples
subplot(2, 1, 2);
plot(numSamplesArray, meanMCseValues, '-o');
title('Monte Carlo Standard Error of Mean vs Number of Samples');
xlabel('Number of Samples');
ylabel('Mean MCSE');
grid on;
