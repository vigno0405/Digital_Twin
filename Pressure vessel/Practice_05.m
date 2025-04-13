clc;
clear all;
close all;

leakage_diameter = linspace(0.1, 0.3, 100) * 1e-3; % [m]
leakage_area = pi/4 * leakage_diameter.^2;

for counter = 1:length(leakage_area)
    leakage_area_i = leakage_area(counter);
    seed = randi(1000000);
    if ~isempty(gcs) % gcs gives as output the current simulink system
        close_system(gcs)
    end
    tic
    out = sim("Practice_05_model.slx");
    toc
    if ~isempty(gcs) % gcs gives as output the current simulink system
        close_system(gcs)
    end
    save("Simulations\" + "Sim_" + num2str(counter) + ".mat")
end
