% unificar datos climas
formafecha = 'yyyy-MM-dd';
% fecha inicial
t1 = datetime(2020,01,01,'Format',formafecha);
% fecha final
t2 = datetime(2022,08,30,'Format',formafecha);
% serie temporal diaria
FECHA = (t1:t2)';

% variables de datos_clima
TMPMIN=nan(size(FECHA));
TMPMED=nan(size(FECHA));
TMPMAX=nan(size(FECHA));
HRMAX=nan(size(FECHA));
HRMIN=nan(size(FECHA));
PREC=nan(size(FECHA));
RAD=nan(size(FECHA));
ET0=nan(size(FECHA));
COND_HELADAS=nan(size(FECHA));
DUR_HELADAS=nan(size(FECHA));



% crear climahistorico con los datos de historicos 1
run('Meteorology\script_clima_historico');

idx = find(ismember(FECHA,climahistorico.fecha));
TMPMIN(idx) = climahistorico.tmpmin;
TMPMED(idx) = climahistorico.tmpmed;
TMPMAX(idx) = climahistorico.tmpmax;
HRMIN(idx) = climahistorico.hrmin;
HRMAX(idx) = climahistorico.hrmax;
PREC(idx) = climahistorico.precip;
idxtmp = (climahistorico.tmpmin<1);
idxvv = (climahistorico.speedmax<10);
COND_HELADAS(idx) = double(idxtmp.*idxvv);
DUR_HELADAS(idx) = COND_HELADAS(idx);

% crear radhist con datos de radiación historica
run('Meteorology\script_radiacion_2');
t1 = datetime(2021,10,01,'Format',formafecha);
% fecha final
t2 = datetime(2021,11,30,'Format',formafecha);
% serie temporal diaria
fechasrad = (t1:t2)';
idx = find(ismember(FECHA,fechasrad));
RAD(idx) = radhist_2.RADmjm2;

% crear radhist con datos de radiación historica
run('Meteorology\script_radiacion_3');
t1 = datetime(2021,11,30,'Format',formafecha);
% fecha final
t2 = datetime(2022,01,16,'Format',formafecha);
% serie temporal diaria
fechasrad = (t1:t2)';
idx = find(ismember(FECHA,fechasrad));
RAD(idx) = radhist_3.PARMJM2;


idx = find(ismember(FECHA,climahistorico.fecha));
ET0(idx) = 0.0023*(TMPMED(idx)+17.78).*(RAD(idx)/2.45).*((TMPMAX(idx) - TMPMIN(idx)).^0.5);


% crear climahistorico con los datos de historicos 2
run('Meteorology\script_clima_historico_2');

idx = find(ismember(FECHA,datosObservados1.fechas));
TMPMIN(idx) = datosObservados1.tmpmin;
TMPMED(idx) = datosObservados1.tmpmed;
TMPMAX(idx) = datosObservados1.tmpmax;
HRMIN(idx) = datosObservados1.hrmin;
HRMAX(idx) = datosObservados1.hrmax;
RAD(idx) = datosObservados1.RAD;
PREC(idx) = datosObservados1.precip;
COND_HELADAS(idx) = datosObservados1.HELADA;
DUR_HELADAS(idx) = datosObservados1.DUR_HELADA;
ET0(idx) = datosObservados1.evap;

% crear climahistorico con los datos de historicos 3
run('Meteorology\script_clima_historico_3');

idx = find(ismember(FECHA,datosObservados2.fechas));
TMPMIN(idx) = datosObservados2.tmpmin;
TMPMED(idx) = datosObservados2.tmpmed;
TMPMAX(idx) = datosObservados2.tmpmax;
HRMIN(idx) = datosObservados2.hrmin;
HRMAX(idx) = datosObservados2.hrmax;
RAD(idx) = datosObservados2.RAD;
PREC(idx) = datosObservados2.precip;
COND_HELADAS(idx) = datosObservados2.HELADA;
DUR_HELADAS(idx) = datosObservados2.DUR_HELADA;
ET0(idx) = datosObservados2.evap;

% 
% % crear climafuturo con los datos de pronosticos
% run('C:\Users\gabri\Dropbox\AgroModel\Meteorology\script_clima_futuro_nuevo2');
% % agregar a datos_clima los pronosticos
% idx = find(ismember(FECHA,climafuturo.FECHA));
% 
% TMPMIN(idx) = climafuturo.TMINC;
% TMPMED(idx) = climafuturo.TMEDC;
% TMPMAX(idx) = climafuturo.TMAXC;
% HRMIN(idx) = climafuturo.HR_MIN;
% HRMAX(idx) = climafuturo.HR_MAX;
% PREC(idx) = climafuturo.LLUVIAmm;
% RAD(idx) = climafuturo.RADMJM2;
% COND_HELADAS(idx) = climafuturo.COND_HELADAS;

COND_HELADAS(isnan(COND_HELADAS)) = 0;
DUR_HELADAS(isnan(DUR_HELADAS)) = 0;
climatotal_nuevo = table(FECHA,TMPMIN,TMPMED,TMPMAX,HRMIN,HRMAX,PREC,RAD,ET0,COND_HELADAS,DUR_HELADAS);

climatotal_nuevo = fillmissing(climatotal_nuevo,'linear','EndValues','none');