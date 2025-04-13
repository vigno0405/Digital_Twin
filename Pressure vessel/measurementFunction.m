function Y = measurementFunction(leakageArea, timeSteps)

arguments
    leakageArea (1, 1) {mustBeNumeric, mustBeReal} % It must be a scalar and a real numeric number
    timeSteps (:, 1) {mustBeNumeric, mustBeReal} % It must be a vector of real numeric numbers
end

persistent meanX meanY stdX stdY trainedNet
if isempty(meanX)
    load("Practice_06_08_trainedModel.mat") % Load the variables
end

X = [leakageArea*ones(length(timeSteps), 1), timeSteps];

Y = trainedNet(((X - meanX)./ stdX)') .* stdY + meanY;
Y = Y'; % Y must be a column vector

end