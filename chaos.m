
clc, clear all, close all
m=100;n=100;
h=0.001; to=0.001; tf=m*n; % h is the step size, t=[a,b] t-range
t = to:h:tf; % Computes t-array up to t=3


x1 = zeros(1,numel(t)); % Memory preallocation
x2 = zeros(1,numel(t));
x3 = zeros(1,numel(t));
x4 = zeros(1,numel(t));
x5 = zeros(1,numel(t));

x1(1) = 1; % initial condition; in MATLAB indices start at 1
x2(1) = 1;
x3(1) = 1;
x4(1) = 1;
x5(1) = 1;

%Initializing parameters
a=10;b=60;c=20;d=15;e=40;f=1;g=50;p=10;
% the function is the expression after (t,x)
Fx1t = @(t,x1,x2,x3,x4,x5) (-a.*x1)+(x2.*x3) ;  
Fx2t = @(t,x1,x2,x3,x4,x5) (-b.*x2)+(f.*x5) ;
Fx3t = @(t,x1,x2,x3,x4,x5) (-c.*x3)+(g.*x4)+ (x1.*x2) ;
Fx4t = @(t,x1,x2,x3,x4,x5)  (d.*x4)-(h.*x3) ;
Fx5t = @(t,x1,x2,x3,x4,x5)  (e.*x5)-(x2.*x1.*x1) ;

for ii=1:1:numel(t)
 k1 = Fx1t(t(ii),x1(ii),x2(ii),x3(ii),x4(ii),x5(ii));
 k2 = Fx1t(t(ii)+0.5*h,x1(ii)+0.5*h*k1,x2(ii)+0.5*h*k1,x3(ii)+0.5*h*k1,x4(ii)+0.5*h*k1,x5(ii)+0.5*h*k1);
 k3 = Fx1t((t(ii)+0.5*h),(x1(ii)+0.5*h*k2),(x2(ii)+0.5*h*k2),(x3(ii)+0.5*h*k2),(x4(ii)+0.5*h*k2),(x5(ii)+0.5*h*k2));
 k4 = Fx1t((t(ii)+h),(x1(ii)+h*k3),(x2(ii)+h*k3),(x3(ii)+h*k3),(x4(ii)+h*k3),(x5(ii)+h*k3));
 
 m1 = Fx2t(t(ii),x1(ii),x2(ii),x3(ii),x4(ii),x5(ii));
 m2 = Fx2t(t(ii)+0.5*h,x1(ii)+0.5*h*k1,x2(ii)+0.5*h*k1,x3(ii)+0.5*h*k1,x4(ii)+0.5*h*k1,x5(ii)+0.5*h*k1);
 m3 = Fx2t((t(ii)+0.5*h),(x1(ii)+0.5*h*k2),(x2(ii)+0.5*h*k2),(x3(ii)+0.5*h*k2),(x4(ii)+0.5*h*k2),(x5(ii)+0.5*h*k2));
 m4 = Fx2t((t(ii)+h),(x1(ii)+h*k3),(x2(ii)+h*k3),(x3(ii)+h*k3),(x4(ii)+h*k3),(x5(ii)+h*k3));
 
 n1 = Fx3t(t(ii),x1(ii),x2(ii),x3(ii),x4(ii),x5(ii));
 n2 = Fx3t(t(ii)+0.5*h,x1(ii)+0.5*h*k1,x2(ii)+0.5*h*k1,x3(ii)+0.5*h*k1,x4(ii)+0.5*h*k1,x5(ii)+0.5*h*k1);
 n3 = Fx3t((t(ii)+0.5*h),(x1(ii)+0.5*h*k2),(x2(ii)+0.5*h*k2),(x3(ii)+0.5*h*k2),(x4(ii)+0.5*h*k2),(x5(ii)+0.5*h*k2));
 n4 = Fx3t((t(ii)+h),(x1(ii)+h*k3),(x2(ii)+h*k3),(x3(ii)+h*k3),(x4(ii)+h*k3),(x5(ii)+h*k3));
 
 l1 = Fx4t(t(ii),x1(ii),x2(ii),x3(ii),x4(ii),x5(ii));
 l2 = Fx4t(t(ii)+0.5*h,x1(ii)+0.5*h*k1,x2(ii)+0.5*h*k1,x3(ii)+0.5*h*k1,x4(ii)+0.5*h*k1,x5(ii)+0.5*h*k1);
 l3 = Fx4t((t(ii)+0.5*h),(x1(ii)+0.5*h*k2),(x2(ii)+0.5*h*k2),(x3(ii)+0.5*h*k2),(x4(ii)+0.5*h*k2),(x5(ii)+0.5*h*k2));
 l4 = Fx4t((t(ii)+h),(x1(ii)+h*k3),(x2(ii)+h*k3),(x3(ii)+h*k3),(x4(ii)+h*k3),(x5(ii)+h*k3));
 
 q1 = Fx5t(t(ii),x1(ii),x2(ii),x3(ii),x4(ii),x5(ii));
 q2 = Fx5t(t(ii)+0.5*h,x1(ii)+0.5*h*k1,x2(ii)+0.5*h*k1,x3(ii)+0.5*h*k1,x4(ii)+0.5*h*k1,x5(ii)+0.5*h*k1);
 q3 = Fx5t((t(ii)+0.5*h),(x1(ii)+0.5*h*k2),(x2(ii)+0.5*h*k2),(x3(ii)+0.5*h*k2),(x4(ii)+0.5*h*k2),(x5(ii)+0.5*h*k2));
 q4 = Fx5t((t(ii)+h),(x1(ii)+h*k3),(x2(ii)+h*k3),(x3(ii)+h*k3),(x4(ii)+h*k3),(x5(ii)+h*k3));
 
 x1(ii+1) = x1(ii) + (h/6)*(k1+2*k2+2*k3+k4); %
 x2(ii+1) = x2(ii) + (h/6)*(m1+2*m2+2*m3+m4);
 x3(ii+1) = x3(ii) + (h/6)*(n1+2*n2+2*n3+n4);
 x4(ii+1) = x4(ii) + (h/6)*(l1+2*l2+2*l3+l4);
 x5(ii+1) = x5(ii) + (h/6)*(q1+2*q2+2*q3+q4);
end

