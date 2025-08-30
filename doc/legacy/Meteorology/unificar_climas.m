
% unificar datos climas
formafecha = 'yyyy-MM-dd';
% fecha inicial
t1 = datetime(2020,01,01,'Format',formafecha);
% fecha final
t2 = datetime(2022,06,04,'Format',formafecha);
% serie temporal diaria
FECHA = (t1:t2)';

pause

% variables de datos_clima
TMPMIN_H=nan(size(FECHA));
TMPMED_H=nan(size(FECHA));
TMPMAX_H=nan(size(FECHA));
HRMAX_H=nan(size(FECHA));
HRMIN_H=nan(size(FECHA));
PREC_H=nan(size(FECHA));
RAD_H=nan(size(FECHA));
TMPMIN_F=nan(size(FECHA));
TMPMED_F=nan(size(FECHA));
TMPMAX_F=nan(size(FECHA));
DPV_F=nan(size(FECHA));
PREC_F=nan(size(FECHA));
RAD_F=nan(size(FECHA));


% crear climafuturo con los datos de pronosticos
run('C:\Users\gabri\Dropbox\AgroModel\Meteorology\script_clima_futuro');
% agregar a datos_clima los pronosticos
idx = find(ismember(FECHA,climafuturo.FECHA));
TMPMIN_F(idx) = climafuturo.TMINC;
TMPMED_F(idx) = climafuturo.TMEDC;
TMPMAX_F(idx) = climafuturo.TMAXC;
DPV_F(idx) = climafuturo.TTDC;
PREC_F(idx) = climafuturo.LLUVIAMM;
RAD_F(idx) = climafuturo.RADmjm2;

% crear climahistorico con los datos de pronosticos

run('C:\Users\gabri\Dropbox\AgroModel\Meteorology\script_clima_historico');

idx = find(ismember(FECHA,climahistorico.fecha));
TMPMIN_H(idx) = climahistorico.tmpmin;
TMPMED_H(idx) = climahistorico.tmpmed;
TMPMAX_H(idx) = climahistorico.tmpmax;
HRMIN_H(idx) = climahistorico.hrmin;
HRMAX_H(idx) = climahistorico.hrmax;
PREC_H(idx) = climahistorico.precip;

% crear radhist con datos de radiación historica
run('C:\Users\gabri\Dropbox\AgroModel\Meteorology\script_radiacion');
t1 = datetime(2020,10,01,'Format',formafecha);
% fecha final
t2 = datetime(2021,05,31,'Format',formafecha);
% serie temporal diaria
fechasrad = (t1:t2)';
idx = find(ismember(FECHA,fechasrad));
RAD_H(idx) = radhist.mjm2;

climatotal = table(FECHA,TMPMIN_H,TMPMED_H,TMPMAX_H,HRMIN_H,HRMAX_H,PREC_H,RAD_H,...
    TMPMIN_F,TMPMED_F,TMPMAX_F,DPV_F,PREC_F,RAD_F);


cd ..