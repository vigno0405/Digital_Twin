function [m, mcse] = computeMCstats(x)
    m = mean(x); % Sample Mean
    v = var(x); % Sample Variance
    mcse = sqrt(v/length(x)); % Monte Carlo Standard Error
end