clc;
clear;
close all;

% All possible dimensions of leakages, approximated as holes
leakage_diameter = linspace(0.1, 0.3, 100) * 1e-3; % [m]
leakage_area = (leakage_diameter.^2) * pi /4;

% Run 1 simulation for each leakage area
for counter = 1:length(leakage_area)
    leak = leakage_area(counter);
    %out = sim("Pressure_vessel_model.slx");
    %save("Simulations\" + "Sim_" + num2str(counter) + ".mat")
end

%% VolumeLeakage(t, LeakageArea) = SimscapeModel(t, LeakageArea) = ANN(t, LeakageArea)

% Accessing all the different dimensions we are interested in out from the
% simulation folder
simFiles = dir("Simulations\*.mat"); % all .mat files
inputLeak = nan(1, length(simFiles));
leakageVolume = cell(1, length(simFiles)); % cell to store diffferent data
% Volume is increasing in time, since I am continuously losing water as the
% simulation works

for counter = 1 : length(simFiles)
    load([simFiles(counter).folder '\' simFiles(counter).name])
    simulationTime = out.tout; % Input1
    inputLeak(counter) = leakage_area(counter); % Input2
    temp = find(out.logsout, 'Name', 'LeakCumulativeOutVolume');
    % by name (to find)
    leakageVolume{counter} = temp{1}.Values.Data; % Output
end

% Given the leakage area, given the time, we are finding the leakage
% volume.
% Note that simulation time is equal for all the simulations, since we have
% chosen a fixed time step