clc;
clear;
close all;

%% Battery DT aging parameters

Vnom = 4.2; % Nominal Voltage [V]
R0 = 0.15; % Internal Resistance [Ohm]
AH = 2; % Capacity [Ah]
V1 = 3.3; % Voltage V1 when charge is AH1 [V]
AH1 = 2.03 - 1.86; % Charge AH1 when no-load voltage is V1 [Ah]
R1 = 0.05; % First polarization resistance (time-dynamics) [Ohm]
T1 = 100; % First time constant [s]

% Fade parameters at 100 discharge cycles
N = 1; % Number of charge/discharge cycles at simulation start []
etaDischarge = 0.9; % Ratio between capacity at 0 cycles and capacity at 100 cycles
etaDischargeTrue = etaDischarge;
etaR0 = 1.1;
etaR0True = etaR0; % Ratio between internal resistance at 0 cycles and capacity at 100 cycles
etaV1 = 0.9;
etaV1True = etaV1; % Ratio between voltage V1 when charge is AH1 at 0 cycles and capacity at 100 cycles
noiseSTD = 0.05;

N = 100; % Number of charge/discharge cycles at simulation start []
proposalSTD = 0.5e-1; % one order or magnitude lower than the expected value

%% Change etaDischarge

% N=300;
% etaDischarge = 0.6;
% out_06 = sim("Battery_model_M2022b.slx");
% 
% etaDischarge = 0.8;
% out_08 = sim("Battery_model_M2022b.slx");
% 
% etaDischarge = 0.99;
% out_099 = sim("Battery_model_M2022b.slx");

%% Plotting

% figure
% subplot(2, 1, 1)
% hold on
% plot(out_06.tout,...
%     out_06.logsout.find('Voltage_measured_sim').Values.Data,'LineWidth',1.5)
% plot(out_08.tout,...
%     out_08.logsout.find('Voltage_measured_sim').Values.Data,'LineWidth',1.5)
% plot(out_099.tout,...
%     out_099.logsout.find('Voltage_measured_sim').Values.Data,'LineWidth',1.5)
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Voltage [V]','Interpreter','latex')
% legend('0.6', '0.8', '0.99','Interpreter','latex')
% title('Simulated Voltage', 'Interpreter','latex')
% grid on
% 
% subplot(2,1,2)
% hold on
% plot(out_06.tout,...
%     out_06.logsout.find('Current_load').Values.Data,'LineWidth',2)
% ylim([-3, 3])
% hold on
% plot(out_08.tout,...
%     out_08.logsout.find('Current_load').Values.Data,'LineWidth',2)
% plot(out_099.tout,...
%     out_099.logsout.find('Current_load').Values.Data,'LineWidth',2)
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Current [A]','Interpreter','latex')
% legend('0.6', '0.9', '0.99','Interpreter','latex')
% title('Simulated Current', 'Interpreter','latex')
% grid on


%% Let's see what happens during discharging when changing the number of cycles

% close all
% etaDischarge=0.9;
% 
% % We can change etaDischarge to see different results:
% 
% % N is the N0 we have as starting position (initial target)
% N = 100; % It is changed inside the battery (where we have N)
% out_N0 = sim("Battery_model_M2022b.slx");
% 
% N = 200;
% out_N50 = sim("Battery_model_M2022b.slx");
% 
% N = 300;
% out_N100 = sim("Battery_model_M2022b.slx");
% 
% % Load the index of the current load equal to -2 (for discharge: different
% % values could be used for the other regions
% dischargeInd_N0 = out_N0.logsout.find('Current_load').Values.Data == -2;
% dischargeInd_N100 = out_N100.logsout.find('Current_load').Values.Data == -2;
% dischargeInd_N50 = out_N50.logsout.find('Current_load').Values.Data == -2;

%% Plotting the results

% figure
% subplot(2, 1, 1)
% plot(out_N0.tout,...
%     out_N0.logsout.find('Voltage_measured_sim').Values.Data,'LineWidth',1.5)
% hold on
% plot(out_N50.tout,...
%     out_N50.logsout.find('Voltage_measured_sim').Values.Data,'LineWidth',1.5)
% plot(out_N100.tout,...
%     out_N100.logsout.find('Voltage_measured_sim').Values.Data,'LineWidth',1.5)
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Voltage [V]','Interpreter','latex')
% legend('N100', 'N200', 'N300','Interpreter','latex')
% title('Simulated Voltage', 'Interpreter','latex')
% grid on
% 
% subplot(2,1,2)
% plot(out_N0.tout,...
%     out_N0.logsout.find('Current_load').Values.Data,'LineWidth',2)
% ylim([-3, 3])
% hold on
% plot(out_N50.tout,...
%     out_N50.logsout.find('Current_load').Values.Data,'LineWidth',2)
% plot(out_N100.tout,...
%     out_N100.logsout.find('Current_load').Values.Data,'LineWidth',2)
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Current [A]','Interpreter','latex')
% legend('N100', 'N200', 'N300','Interpreter','latex')
% title('Simulated Current', 'Interpreter','latex')
% grid on

%% Add realistic noise

% % Needed to make the simulation realistic, te other one was not.
% % noiseSTD = 0.005;
% noiseSTD = 0.05;
% 
% figure
% plot(out_N0.logsout.find('Voltage_measured_sim').Values.Data...
%     + noiseSTD*randn(size(out_N0.logsout.find('Voltage_measured_sim').Values.Data)))
% hold on
% plot(out_N50.logsout.find('Voltage_measured_sim').Values.Data...
%     + noiseSTD*randn(size(out_N50.logsout.find('Voltage_measured_sim').Values.Data)))
% plot(out_N100.logsout.find('Voltage_measured_sim').Values.Data...
%     + noiseSTD*randn(size(out_N100.logsout.find('Voltage_measured_sim').Values.Data)))
% legend('N100', 'N200', 'N300', 'Interpreter','latex')
% xlabel('Time [s]','Interpreter','latex')
% ylabel('Voltage [V]','Interpreter','latex')
% title('Measured Voltage', 'Interpreter','latex')
% grid on

%% Sweep the degradation parameters at N = 100, 200, 300 and store the simulations

% for N = [100, 200, 300]
%     etaDischargeVector = 0.8:0.001:0.99;
%     batterySimulation = cell(1, length(etaDischargeVector));
%     for counter = 1:length(batterySimulation)
%         %etaDischarge = etaDischargeVector(counter);
%         %batterySimulation{counter} = sim("Battery_model_M2022b.slx");
%         fprintf('Simulation %d\n', counter)
%     end
%     %save(sprintf('BatterySimulation_N_%d.mat', N), 'batterySimulation')
% end

%% Loading the simulations

% N = 100; % We work at N = 100 for the moment
% etaDischargeVector = 0.8:0.001:0.99;
% load(sprintf('BatterySimulation_N_%d.mat', N), 'batterySimulation')

%% Plot the measurements for the first and the last simulation without noise

% Voltage measured, without noise, for N = 100 (from the runned
% simulations)

% figure
% plot(batterySimulation{1}.logsout.find('Voltage_measured_sim').Values.Time, ...
%     batterySimulation{1}.logsout.find('Voltage_measured_sim').Values.Data)
% hold on
% plot(batterySimulation{end}.logsout.find('Voltage_measured_sim').Values.Time, ...
%     batterySimulation{end}.logsout.find('Voltage_measured_sim').Values.Data)
% legend(sprintf('\\eta_{Discharge} = %.3f', etaDischargeVector(end)),...
%     sprintf('\\eta_{Discharge} = %.3f', etaDischargeVector(1)))
% xlabel('Time [s]')
% ylabel('Voltage Measured [V]')
% grid on
% title(sprintf('N = %d', N))

% Input:
% - etaDischarge
% - time!!!
% Output is the measured voltage.
% Since the measurement contains discontinuities, it's better to make 3
% different ANNs, with each one being the surrogate model for the CHARGING
% phase, the DISCHARGE phase, and the REST phase (where voltage anyway increases).
% We thus discard the measurements near the discontinuities, because they
% are hard to be captured by the surrogate model

%% Build the surrogate models

% Surrogate models are provided in the .mat files
% 'BatterySimulationSurrogate..'. We can create them as simple ANNs
% (to get simulation models).

% Each of those files contains four variables:
% - Net_1: this structure contains the ANN for the charge phase
% - Net_2: this structure contains the ANN for the discharge phase
% - Net_3: this structure contains the ANN for the rest phase
% - ind: this is a 1 x 3 cell array, in which each element contains the
% time indexes for the charge phase, the discharge phase, and the rest
% phase (so as to be coherent, being present discontinuities)
% - t: this variable is the simulation output time

% close all
N = 200;
load(sprintf('BatterySimulationSurrogate_N_%d.mat', N))
% 
% % The ANNs take as input:
% %   1 - the simulation time
% %   2 - the degradation parameter etaDischarge
% % Hence, if we want to simulate the output voltage of the surrogate model, we have
% % to use this syntax:
etaDischarge = 0.93;
xSimMeasurements = [sim(Net_1.Network, [t(ind{1}), repmat(etaDischarge, length(t(ind{1})), 1)]'),...
                   sim(Net_2.Network, [t(ind{2}), repmat(etaDischarge, length(t(ind{2})), 1)]'),...
                   sim(Net_3.Network, [t(ind{3}), repmat(etaDischarge, length(t(ind{3})), 1)]')]';
%We can compare the ANNs output and the simulation output:
temp = sim('Battery_model_M2022b.slx');
figure
plot([ind{:}], xSimMeasurements,'LineWidth',1)
hold on
plot(temp.logsout.find('Voltage_measured_sim').Values.Time, ...
     temp.logsout.find('Voltage_measured_sim').Values.Data);
xlim([0, 6000])
legend('Surrogate Model', 'Simulation Output','Interpreter','latex')
grid on
xlabel('Time [s]','Interpreter','latex')
ylabel('Voltage [V]','Interpreter','latex')
title('Detail, $\eta_{discharge}=0.93$','Interpreter','latex')

%% MCMC

close all
clc

% Value of N
% N = 100;
load(sprintf('BatterySimulationSurrogate_N_%d.mat', N))

rng('shuffle')

% Define the prior:
priorPd = makedist("Beta", "a", 2.5, "b", 1.1);
prior = @(x) priorPd.pdf(x);

x = linspace(0, 1, 100000);
prior1 = prior(x);

% Primo grafico: prior in funzione dell'area
%figure;
%plot(x, prior1, 'LineWidth', 1.5);
%xlabel('$\eta_{discharge}$]', 'Interpreter', 'latex');
%ylabel('Prior pdf', 'Interpreter', 'latex');
%title('Prior pdf vs $\eta_{discharge}$', 'Interpreter', 'latex');
%grid on;

% Initialize parameters
x = 0.85;

% Preallocate the chain
chainLength = 10000;

chain = nan(chainLength, 1);

% Define the transition function (of the proposal, always with a Gaussian):

proposal = @(x) x +  proposalSTD * randn;

sigmaMeasurements = noiseSTD; % [V]: to define the likelihood

% Define the measurements:
etaDischarge = 0.9;
target = sim("Battery_model_M2022b.slx");

expMeasurements = target.logsout.find('Voltage_measured_sim').Values.Data(:);
expMeasurements = expMeasurements + noiseSTD*randn(size(expMeasurements));
expMeasurements = expMeasurements([ind{:}]); % cut based on the values in array
% We just cut off the samples for which we don't have the ANNs output

% Output the simulated measurements of the first iteration
measurementFunction = @(x) [sim(Net_1.Network, [t(ind{1}), repmat(x, length(t(ind{1})), 1)]'),...
                   sim(Net_2.Network, [t(ind{2}), repmat(x, length(t(ind{2})), 1)]'),...
                   sim(Net_3.Network, [t(ind{3}), repmat(x, length(t(ind{3})), 1)]')]';
xSimMeasurements = measurementFunction(x);
acceptedSample = nan(chainLength, 1);
acceptanceRatios = nan(floor(chainLength/50), 1); % Acceptance ratios
i = 1;

while  i <= chainLength
    % Propose a new state:
    xNew = proposal(x);

    % Output the simulated measurements of the proposed state
    xNewSimMeasurements = measurementFunction(xNew);
    
    % OCCHIO!
    [xNewlogLik, xLogLik] = logLikelihood(xNewSimMeasurements(1:20:end), xSimMeasurements(1:20:end), ...
        expMeasurements(1:20:end), sigmaMeasurements(1:20:end));

    % Compute the logarithm of the probability of accepting the sample
    randLog =  log(rand);
    
    logAlpha = xNewlogLik - xLogLik + log(prior(xNew)) - log(prior(x));

    if randLog > logAlpha % Not accepted
        chain(i, :) = x;
        acceptedSample(i) = false;
%         fprintf('Rejected\n')
    else % Accepted
        acceptedSample(i) = true;
        chain(i, :) = xNew;
        x = xNew;
        xSimMeasurements = xNewSimMeasurements;
%         fprintf('Accepted\n')
    end
    % Print the acceptance ratio once every 50 samples
    if rem(i, 50) == 0
        fprintf('Iteration: %d\n', i)
        fprintf('Acceptance ratio: %f\n', min(sum(acceptedSample(i-49:i))/49,1))
        acceptanceRatios(floor(i/50)) = min(sum(acceptedSample(i-49:i))/49, 1);
    end
    i = i + 1;
end
fprintf('MCMC completed\n')

%% Plot the chain and the acceptance ratios

% close all;
% figure
% plot(chain)
% hold on
% xlabel('Samples','Interpreter','latex')
% ylabel('Chain samples: $\eta$','Interpreter','latex')
% yline(0.9, "LineWidth", 4)
% legend("Chain", "True $\eta$",'Interpreter','latex')
% title('Markov Chain', 'Interpreter','latex')
% hold off
% 
% figure
% plot(acceptanceRatios,'o')
% hold on
% ylim([0 1])
% xlabel('Samples','Interpreter','latex')
% ylabel('Acceptance ratio','Interpreter','latex')
% yline(mean(acceptanceRatios), "LineWidth", 2, 'LineStyle', '--', 'Color', 'red');
% legend('$\alpha$', '$\bar{\alpha}$','Interpreter','latex')
% title('$\alpha$', 'Interpreter', 'latex');

%% Thinning

chainThinned = chain(1000:10:end);
% figure
% plot(chainThinned)
% hold on
% xlabel('Thinned samples','Interpreter','latex')
% ylabel('Chain samples: $\eta$','Interpreter','latex')
% title('Thinned Markov Chain','Interpreter','latex')
% yline(0.9, "LineWidth", 4)
% legend("Thinned Chain", "True $\eta$",'Interpreter','latex')
% hold off

figure
histogram(chainThinned, 'Normalization','pdf')
hold on
xline(0.9, "LineWidth", 4)
xlabel('$\eta$','Interpreter','latex')
ylabel('Frequency','Interpreter','latex')
xline(mean(chainThinned), 'LineWidth', 2, 'Color', 'b', 'LineStyle', '--');
ksdensity(chainThinned)
xline(quantile(chainThinned, 0.95), 'r', "LineWidth", 2) % "first" quantile
xline(quantile(chainThinned, 0.05), 'r', "LineWidth", 2) % "last" quantile
legend("$\eta$ PD", "True $\eta$", "Mean", "Kernel Smoothing", "$q_{0.05}$","$q_{0.95}$",'Interpreter','latex',...
    'Location','best')
title(sprintf('$\\eta$ Distribution $@$ $N = %d$', N), 'Interpreter', 'latex');
grid on
hold off

%% Plot the distribution of k1

% We exploit the analytical formula (fixed N) for k1
k1 = (1 - chainThinned)/10;
% figure
% histogram(k1, "Normalization", "pdf")
% hold on
% xline(0.01, "LineWidth", 4)
% xlabel('$k_1$','Interpreter','latex')
% ylabel('Frequency','Interpreter','latex')
% xline(mean(k1), 'LineWidth', 2, 'Color', 'b', 'LineStyle', '--');
% ksdensity(k1)
% xline(quantile(k1, 0.05), 'r', "LineWidth", 2)
% xline(quantile(k1, 0.95), 'r', "LineWidth", 2)
% legend("$k_1$ PD", "True $k_1$", "Mean", "Kernel Smoothing", "$q_{0.05}$","$q_{0.95}$",'Interpreter','latex',...
%     'Location','northeast')
% title(sprintf('$k_1$ Distribution $@$ $N = %d$', N),'Interpreter','latex')
% grid on
% hold off
% Note that its behavior is opposite wrt etaDischarge: that's why we have
% these results.

%% Plot the battery capacity as a function of the number of cycles

n = 0:500;
% Plotting as a function of n, but keeping the stochastic distribution
% (with mean and capacity)

capacity_mean = AH * (1 - mean(k1) * sqrt(n));
capacity_true = AH * (1 - 0.01 * sqrt(n));
capacity_95 = AH * (1 - quantile(k1, 0.05) * sqrt(n));
capacity_05 = AH * (1 - quantile(k1, 0.95) * sqrt(n));

figure
plot(n, capacity_mean, 'LineWidth',2,'Color','k')
hold on
plot(n, capacity_true,'LineWidth',1.5,'Color','b','LineStyle','--')
xlabel("Number of cycles",'Interpreter','latex')
ylabel("Capacity $[Ah]$",'Interpreter','latex')
title(sprintf('Deterioration $@$ $N = %d$', N), 'Interpreter', 'latex');
xline(N, 'c','LineWidth',2)
yline(0.8 * AH, 'LineWidth', 2, 'Color', [0, 0.5, 0]); % Verde scuro (RGB)
plot(n, capacity_05,'r','LineWidth',2)
plot(n, capacity_95,'r','LineWidth',2)
legend("Mean Capacity","True Capacity","Estimation Point","EOL Threshold",...
    "$5\%$", "$95\%$",'Interpreter','latex')
grid on

%% Evaluate the probability of the N_EOL

clc
% Threshold based on the rated battery capacity AH_0
N_EOL = ((1-0.8) ./ k1) .^2;
figure
histogram(N_EOL, "Normalization", "pdf")
hold on
xlabel("$N_{EOL}$ $\mid$ Measurements",'Interpreter','latex')
ylabel('Frequency','Interpreter','latex')
xline(mean(N_EOL), "LineWidth", 2)
ksdensity(N_EOL)
xline(quantile(N_EOL, 0.05), 'r', "LineWidth", 2)
xline(quantile(N_EOL, 0.95), 'r', "LineWidth", 2)
legend("$N_{EOL}$ PD", "Mean", "Kernel Smoothing", "$q_{0.05}$","$q_{0.95}$",'Interpreter','latex',...
    'Location','northeast')
title(sprintf('$N_{EOL}$ Distribution $@$ $N = %d$', N),'Interpreter','latex')
grid on
hold off

%% Residual Useful Life

% RUL = N_EOL - N;
% 
% figure
% histogram(RUL, "Normalization", "pdf")
% hold on
% xlabel("RUL $\mid$ Measurements",'Interpreter','latex')
% ylabel('Frequency','Interpreter','latex')
% xline(mean(RUL), "LineWidth", 2)
% ksdensity(RUL)
% xline(quantile(RUL, 0.05), 'r', "LineWidth", 2)
% xline(quantile(RUL, 0.95), 'r', "LineWidth", 2)
% legend("RUL PD", "Mean", "Kernel Smoothing", "$q_{0.05}$","$q_{0.95}$",'Interpreter','latex',...
%     'Location','northeast')
% title(sprintf('RUL Distribution $@$ $N = %d$', N),'Interpreter','latex')
% grid on
% hold off
% 
