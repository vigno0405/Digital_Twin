function lik = likelihoodFnc(measurements, surrogateOutput, noiseSTD)
arguments
    measurements (:, 1) {mustBeNumeric, mustBeReal}
    surrogateOutput (:, 1) {mustBeNumeric, mustBeReal}
    noiseSTD (1, 1) {mustBeNumeric, mustBeReal}
end

lik = sum(log(1/(2*pi*noiseSTD^2).^(0.5))- (measurements - surrogateOutput).^2  / (2*noiseSTD^2));

% Following the formula explained in slides.
% lik = 0;
% for counter = 1:measurements
%     lik = lik + log(1/(2*pi*noiseSTD^2).^(0.5))- (measurements(counter) - surrogateOutput(counter)).^2  / (2*noiseSTD^2);
% end

end
