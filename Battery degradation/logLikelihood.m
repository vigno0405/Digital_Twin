function [xNewlogLik, xLogLik] = logLikelihood(xNewSimMeasurements, xSimMeasurements, expMeasurements, sigmaMeasurements)
% This function takes as input:
%   1 - xNewSimMeasurements: The vector of the surrogate model output voltage
%   given the proposed degradation parameter
%   2 - xSimMeasurements: The vector of the surrogate model output voltage
%   given the degradation parameter at the previous iteration
%   3 - expMeasurements: the vector of the experimental (i.e. Simulink)
%   output voltage
%   4 - sigmaMeasurements: the measurement noise

% This function outputs:
%   1 - the log likelihood of the data given the proposed degradation
%   parameter
%   2 - the log likelihood of the data given the degradation parameter at
%   the previous iteration

try
xNewlogLik = -normlike([0, sigmaMeasurements], xNewSimMeasurements - expMeasurements);
xLogLik = -normlike([0, sigmaMeasurements], xSimMeasurements - expMeasurements);
% normlike to compute the normal negative loglikelihood

catch ME % to see if they were any error
    warning(ME.message)
end
end