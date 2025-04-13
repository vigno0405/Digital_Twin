clc;
clear;
close all;
% unzip the folder if necessary
simulationList = dir("Simulations\*.mat");
if isempty(simulationList)
    try 
        unzip("Simulations.zip")
    catch ME
        error("You should download the file 'Simulations.zip' and extract it in a folder called 'Simulations', so that the folder 'Simulations' contains only the simulation files.")
    end
end % ok
% load the files
simFiles = dir("Simulations\*.mat");
inputLeakArea = nan(1, length(simFiles));
leakageVolume = cell(1, length(simFiles));
for counter = 1:length(simFiles)
    load([simFiles(counter).folder '\' simFiles(counter).name])
    simulationTime = out.tout;
    inputLeakArea(counter) = leak;
    temp = find(out.logsout, 'Name', 'LeakCumulativeOutVolume');
    leakageVolume{counter} = temp{1}.Values.Data;
end
% Simply loads all the files (as in createDatabase)

%% Plot the measurements

% We compare results of simulations 1, 50, 100: smaller leakage area means
% smaller leakage volume, meaning that simulations are performing well. We
% have a LINEAR relation between time and volume.
% Input leak area, instead (which is a constant) is a single input.

close all

figure
simulationCounter = 1; % which simulation are we selecting?
plot(simulationTime, leakageVolume{simulationCounter})
hold on
simulationCounter = 50;
plot(simulationTime, leakageVolume{simulationCounter})
simulationCounter = 100;
plot(simulationTime, leakageVolume{simulationCounter})
xlabel('Time [s]','Interpreter','latex')
ylabel('Leakage Volume [$\mathrm{m^3}$]', 'Interpreter', 'latex')
legend(string(inputLeakArea([1, 50, 100])))
title('Volume vs Time', 'Interpreter','latex')
hold off

% Plotting the final leakage volume vs the leakage area
figure

% Preallocazione dei vettori per i dati finali e a metà
fin_leak_Volume = zeros(1, 100); % Volume di perdita finale
mid_leak_Volume = zeros(1, 100); % Volume di perdita a metà simulazione (t = 500)

% Iterazione sulle simulazioni per estrarre i dati
for simulationCounter = 1:length(leakageVolume)
    temp = leakageVolume{simulationCounter};
    fin_leak_Volume(simulationCounter) = temp(end); % Volume finale
    mid_leak_Volume(simulationCounter) = temp(501); % Volume a t = 500
end

% Plot dei dati finali
plot(inputLeakArea, fin_leak_Volume, 'DisplayName', 'Final Leakage Volume')
hold on

% Plot dei dati a metà simulazione
plot(inputLeakArea, mid_leak_Volume, 'DisplayName', 'Leakage Volume at t = 500')

% Personalizzazione del grafico
xlabel('Leakage Area [$\mathrm{m^2}$]', 'Interpreter', 'latex')
ylabel('Leakage Volume [$\mathrm{m^3}$]', 'Interpreter', 'latex')
title('Leakage Volume vs Area','Interpreter','latex')
legend('Interpreter', 'latex') % Mostra una legenda
hold off


%% Rearrange the data so that the ANN can be trained

close all
% Input data: X = [simulationTime_Simulation1, leakageArea_Simulation1;
%              simulationTime_Simulation2, leakageArea_Simulation2
%              ...]
% Output data: Y = [leakageVolume_Simulation1;
%                   leakageVolume_Simulation2;
%                   ...]

% Here we store all data in a matrix
X = [];
Y = [];
for simulationCounter = 1:length(leakageVolume)
    X = [X; simulationTime, repmat(inputLeakArea(simulationCounter), ...
        length(simulationTime), 1)];
    Y = [Y; leakageVolume{simulationCounter}];
end

% repmat(3, 5, 2): repeat the first value - array - in a matrix
% repmat([3, 2], 5, 2)

%% Normalize the data

% Normalization is needed to avoid vanishing gradient phenomenon, thus we
% rescale everything centered aroung the mean
X_mean = mean(X);
X_std = std(X);
X_norm = (X - X_mean)./X_std;
Y_mean = mean(Y);
Y_std = std(Y);
Y_norm = (Y - Y_mean)./Y_std;

%% Train the network

% We create a feedforward networks (same as nftool). Only wants number of
% neurons since it avoids more complex structures.

% Crea la rete neurale con, ad esempio, 10 neuroni nello strato nascosto
numberOfNeurons = 10;
net = feedforwardnet(numberOfNeurons);

[net,tr] = train(net, X_norm', Y_norm');
% Columns are samples, Rows are inputs/outputs (the opposite of datasets)

save("Pressure_vessel_trainedNetwork.mat", "net", ...
    "X_mean", "X_std", "Y_mean", "Y_std");

% Validation error is very low (also because we have a simple relationship,
% being it linear - as shown - both in time and leakage area)

%% De-normalize

% REMEMBER TO DE NORMALIZE
% X_denorm = X_norm .* X_std + X_mean;
% Applied to the output of the neural networks in inference time, but we
% also have to normalize the provided input data.