clear all
close all

datos_tormenta = readtimetable('datos_tormenta.txt');
datos_superficie = load('datos_superficie');
datos_nodos = load('datos_nodos');

M=table2array(datos_tormenta);

% grafico trayectoria del centro
% geoscatter(M(:,1),M(:,2),1)

% % grafico de los modelos
% lat_winds = datos_superficie{1}(:,:,1);
% lon_winds = datos_superficie{1}(:,:,2);
% spd_winds = datos_superficie{1}(:,:,3);
% 
% for  i = 1:size(lat_winds,2)
%     geodensityplot(lat_winds(:,i),lon_winds(:,i),spd_winds(:,i),'FaceColor','interp')
%     geolimits([10 15],[-87 -76])
%     pause(0.1)
% end

% grafico de velocidad maximas en grilla
lat_nodos = datos_nodos{1};
lon_nodos = datos_nodos{2};
vel_nodos = datos_nodos{3};

max_vel = 0.01*ones(40);
for i = 1:433
    max_vel = max(max_vel,vel_nodos(:,:,i));
end
geoscatter(lat_nodos(:),lon_nodos(:),max_vel(:),max_vel(:),'filled')
