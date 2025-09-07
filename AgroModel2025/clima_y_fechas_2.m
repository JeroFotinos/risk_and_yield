%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delta_lat = abs(cultivos_lat(1,1)-cultivos_lat(2,1))/2;
delta_lon = abs(cultivos_lon(1,1)-cultivos_lon(1,2))/2;
coord=zeros(size(lats_lons));
for s=1:size(lats_lons,1)
    coord(s,:)=[find(cultivos_lat(:,1)<lats_lons(s,1)+delta_lat & lats_lons(s,1)-delta_lat<cultivos_lat(:,1)),...
        find(cultivos_lon(1,:)<lats_lons(s,2)+delta_lon & lats_lons(s,2)-delta_lon<cultivos_lon(1,:))];
end
% agrego lotes a conocidos
for i=1:size(coord,1)
    cultivo(coord(i,1),coord(i,2)) = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% datos de climas
unificar_climas_v2
% unificar_climas_nuevo_para2020

idx=[];

for i = 1:length(mat_fs)
    idx = [idx;find(ismember(climatotal_nuevo.FECHA,mat_fs(i)))];
end
fecha_actual = datetime(2022,05,02,'Format','yyyy-MM-dd');

d_ini_min=min(idx);
d_ini_max=max(idx);

d_inicial=d_ini_min;
d_final=d_ini_max+no_days_cultivo;
d_actual = find(ismember(climatotal_nuevo.FECHA,fecha_actual));

% %%% fechas
climatotal_nuevo.FECHA(d_inicial)
climatotal_nuevo.FECHA(d_actual)
climatotal_nuevo.FECHA(d_final)

% d_imagen = find(ismember(climatotal_nuevo.FECHA,datetime(2021,01,05,'Format','yyyy-MM-dd')));
no_days = d_ini_max - d_ini_min + no_days_cultivo;

clock_sow = 1 : no_days;                                                   % contador general par le reloj

dds = zeros(size(dds_est,1),size(dds_est,2),no_days);

% fecha de la imagen de ndvi 2022-01-29
t1 = datetime(2022,01,29,'Format','yyyy-MM-dd');
idx_imagen = find(ismember(climatotal_nuevo.FECHA,t1));

idx_siembra_est = idx_imagen - min(dds_est,75);

dds(:,:,1) = d_ini_min - idx_siembra_est;
for j=1:size(coord,1)
    dds(coord(j,1),coord(j,2),1)=d_ini_min - idx(j);
end



if tipo_cultivo == 2
    dds(:,:,1) = max(dds(:,:,1),-34); 
end


% genero datos de temperaturas medias
tm_media = climatotal_nuevo.TMPMED(d_inicial:d_final);% ARCHIVO DE CLIMA - temp media
aux=-1;
while isnan(tm_media(1))
    tm_media(1) = climatotal_nuevo.TMPMED(d_inicial+aux);
    aux=aux-1;
end
for i = 2:length(tm_media)
    if isnan(tm_media(i))
        tm_media(i)=tm_media(i-1);
    end
end
% genero datos de temperaturas max
tm_max = climatotal_nuevo.TMPMAX(d_inicial:d_final);   %
aux=-1;
while isnan(tm_max(1))
    tm_max(1) = climatotal_nuevo.TMPMAX(d_inicial+aux);
    aux=aux-1;
end
for i = 2:length(tm_max)
    if isnan(tm_max(i))
        tm_max(i)=tm_max(i-1);
    end
end
% genero datos de temperaturas minimas
tm_min = climatotal_nuevo.TMPMIN(d_inicial:d_final);   %
aux=-1;
while isnan(tm_min(1))
    tm_min(1) = climatotal_nuevo.TMPMIN(d_inicial+aux);
    aux=aux-1;
end
for i = 2:length(tm_min)
    if isnan(tm_min(i))
        tm_min(i)=tm_min(i-1);
    end
end
% genero datos humedad relativa maxima
hr_max = climatotal_nuevo.HRMAX(d_inicial:d_final);
aux=-1;
while isnan(hr_max(1))
    hr_max(1) = climatotal_nuevo.HRMAX(d_inicial+aux);
    aux=aux-1;
end
for i = 2:length(hr_max)
    if isnan(hr_max(i))
        hr_max(i)=hr_max(i-1);
    end
end
% genero datos humedad relativa minima
hr_min = climatotal_nuevo.HRMIN(d_inicial:d_final);
aux=-1;
while isnan(hr_min(1))
    hr_min(1) = climatotal_nuevo.HRMIN(d_inicial+aux);
    aux=aux-1;
end
for i = 2:length(hr_min)
    if isnan(hr_min(i))
        hr_min(i)=hr_min(i-1);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(tm_max,1)
    dpv(i) = calcularDPV(tm_max(i),tm_min(i),hr_max(i),hr_min(i));
end
dpv=dpv(:);

pp_cotas = [0;25;50;75;100;125;150];
pp_days = climatotal_nuevo.PREC(d_inicial:d_final);

pp=zeros(size(cultivo));                 % ARCHIVO DE CLIMA - precipitaciones
for i = 1:length(pp_days)
    if isnan(pp_days(i))
        pp_days(i)=0;
    end
    pp(:,:,i)=pp_days(i)*ones(size(cultivo));
end

solar_rad = climatotal_nuevo.RAD(d_inicial:d_final);
par = 0.45*solar_rad;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pp=zeros(size(cultivo));                 % ARCHIVO DE CLIMA - precipitaciones
for i = 1:length(pp_days)
    if isnan(pp_days(i))
        pp_days(i)=0;
    end
    pp(:,:,i)=pp_days(i)*ones(size(cultivo));
end
ddp = zeros(size(cultivo));

ET0 = climatotal_nuevo.ET0(d_inicial:d_final);

d_helada = climatotal_nuevo.DUR_HELADAS(d_inicial:d_final);

if tipo_cultivo == 2
    parametros_maiz
elseif tipo_cultivo ==1
    parametros_soja
else
    fprintf('\nError en el tipo de cultivo \n\n')
    return
end
