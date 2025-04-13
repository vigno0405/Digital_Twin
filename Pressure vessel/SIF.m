%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Evaluation of SIF for semi-elliptical cracks according to Newman-Raju
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function K = SIF(t,w,a,c,phi,St,Sb)
%
% K   = SIF [MPa*mm^0.5]
% t   = shell thickness [mm]
% w   = shell width [mm]
% a   = crack depth [mm]
% c   = crack semi-axis [mm]
% phi = angle (0 for point C, pi/2 for point A) [-]
% St  = stress due to axial load [MPa]
% Sb  = stress due to bending moment [MPa]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function K = SIF(t,w,a,c,phi,St,Sb)

ac=a/c;
at=a/t;   

M1=1.13-0.09*ac;
M2=-0.54+0.89/(0.2+ac);
M3=0.5-1/(0.65+ac)+14*(1-ac)^24;

g=1+(0.1+0.35*at^2)*(1-sin(phi))^2;
fp=(ac^2*(cos(phi))^2+(sin(phi))^2)^0.25;
fw=sqrt(sec(pi*c/2/w*sqrt(at)));

F=(M1+M2*at^2+M3*at^4)*g*fp*fw;

Q=1+1.464*ac^1.65;

p=0.2+ac+0.6*at;
H1=1-0.34*at-0.11*ac*at;
G1=-1.22-0.12*ac;
G2=0.55-1.05*ac^0.75+0.47*ac^1.5;
H2=1+G1*at+G2*at^2;
H=H1+(H2-H1)*(sin(phi))^p;

K=(St+H*Sb)*sqrt(pi*a/Q)*F;

return
