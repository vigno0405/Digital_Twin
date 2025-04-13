clc;
clear;
close all;
rng('shuffle')
seed = randi(1000000);
load('Buses.mat')
alpha_mu = 50/10*9;
beta_mu = 50;
stopTime = 24*365*10;

%% Run the simulations

for counter = 1:100
    seed = randi(1000000);
    if ~isempty(gcs) % gcs gives as output the current simulink system
        close_system(gcs)
    end
    tic
    %out = sim("Practice_04_model.slx");
    toc
    if ~isempty(gcs) % gcs gives as output the current simulink system
        close_system(gcs)
    end
    %save(['Simulations\' sprintf('Simulation%d.mat', counter)])
    % Saving all simulations in the folder, with counter Simulation'i'
    %Simulink.sdi.clear
    % Clears all data in Simulink data inspector.
end

%% Load data

simulationList = dir("Simulations\*.mat");
% All simulations in the model: dir() lists files and folders in the
% current folder!
% Preallocate cells to store the output data we want from the simulations
simCracks = cell(1, length(simulationList));
simDefects = cell(1, length(simulationList));

for counter = 1:length(simulationList)
    % This is just for precaution: clears variables from the workspace:
    clearvars defects cracks leakages out
    % Load the .mat file
    load([simulationList(counter).folder '\' simulationList(counter).name])
    % The latter specifies the path inside the computer.
    % Find defects, cracks and leakages:
    defects = find(out.logsout, 'Name', 'smallDefects');
    % Here store small defects.
    cracks = find(out.logsout, '-regexp', 'BlockPath', ...
    '.*/Hold and release');
    % Here store cracks (which haven't lead to leakages).
    leakages = find(out.logsout, 'Name', 'leakages');
    % Here store leakages only.
    % Store the leakages and the cracks depths in the same cell; for the
    % moment we are considering them as cracks.
    simCracks{counter} = [cracks{1}.Values.a.Data( ...
        cracks{1}.Values.a.Time == cracks{1}.Values.a.Time(end));...
            leakages{1}.Values.a.Data];
    % We take in a vector both:
    % - crack values at end time (maximum dimension)
    % - dimension of leakages (obviously always equal to 3.2 * 0.95)
    % In fact we will see a bigger column on that value due to leakages.
    % Store the defects depths in another cell:
    simDefects{counter} = defects{1}.Values.a.Data; 
end
% Put all the crack depths from all the simulations in this vector
crackSamples = vertcat(simCracks{:});
crackDefects = vertcat(simDefects{:});
% Here we have all the cracks from the simulation (without defects).

%% Plot the histogram of the crack depths (not defects):

figure
histogram(crackSamples, 'Normalization','pdf')
hold on
xlabel("Crack depth [mm]")
ylabel("Histogram and pdf")
title('Distribution obtained from the cracks')
hold on
x_vertical = 3.2 * 0.95;
xline(x_vertical, '--', 'Color', [0.5, 0, 0.5], 'LineWidth', 2);
hold off

% We assumed a Weibull distribution, and the shape is actually the same.
% We have a local peak, at value: 3.20*0.95. This is the position where we
% have accumulated the leakages. Similarly, we also could plot crackDefects
% (lengths of defects only, not cracks).