function DPV = calcularDPV(tmax,tmin,hrmax,hrmin)


% presion de saturacion de vapor (PSV)

PSV_max =  0.6108*exp((17.27*tmax)./(237.3+tmax));
PSV_min =  0.6108*exp((17.27*tmin)/(237.3+tmin));
% PSV_med = (PSV_max./PSV_min)/2;

% presion de saturacion de vapor actual del dia (PSVA)

PSVA = (((PSV_max.*hrmin)/100) + ((PSV_min.*hrmax)/100))/2;

% DPV diario según Allen

DPV = ((PSV_max + PSV_min)/2) - PSVA;


