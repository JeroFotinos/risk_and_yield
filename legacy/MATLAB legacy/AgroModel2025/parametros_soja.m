% paramétros maiz

size_datos = size(pp);                                                     % tamaño de los arregles dimensiones espaciales por dimensión temporal
clock_sow = 1 : no_days;                                                   % contador general par le reloj
au2inicial = 0.5;                                                          % coeficiente para estimar el agua en el segundo metro, porcentaje del agua disponible en el primer metro
c_drenaje = 0.75;                                                          % coeficiente de drenaje
prof_suelo = 500;                                                         % profundidad del suelo en mm
cc = 0.27;                                                                 % capacidad campo para agua disponible 0.32 - 0.35 
pmp = 0.12;                                                                % es un % funcion espacial del tipo de suelo (laboratorio/mapas)

au_up = 0.65; % actualizado                                                              % umbral superior de estrés hidrico
au_down = 0.15; % actualizado                                                             % umbral inferior de estrés hidrico

c_forma = 1.2; %cambio 9 de junio % 3;   % actualizado                                                           % coeficiente de forma
au_up_r = 0.5; % actualizado                                                           % umbral superior de estrés hidrico radiacion
au_down_r = 0; % actualizado                                                            % umbral inferior de estrés hidrico radiacion
c_forma_r = 3; % actualizado                                                             % coeficiente de forma radiacion

ef_transpiratoria = 4.5; % actualizado                                                     % eficiencia transpiratoria
c_in = 0.039; % actualizado                                                              % porcentaje de cobertura incial
c_fin = 0.01; % actualizado
c_max = 0.95; % actualizado                                                              % porcentaje de cobertura máxima
dds_max = 60;  % actualizado                                                             % días desde siembra en donde se alcanza c_max, input cultivo depende de practicas como densidad
dds_sen = 120; % actualizado                                                             % días desde siembra en donde empieza a disminuir el porc de cobertura, input cultivo depende de practicas como densidad
dds_in = 7; % actualizado                                                               % input cultivo podria modificarse por T y humedad
dds_fin = no_days_cultivo;                                                         % input cultivo,dato fijo aunque podría ser dinámico
alpha1 = (c_max - c_in)/(dds_max - dds_in);                                % tasa de incremento de cobertura potencial diario

% umbrales de temperaturas
tbr = 10; % actualizado                                                                   % temp base p/ radiacion: input cultivo (C)
tor1 = 20; % actualizado                                                                % temp optima 1 p/ radiac : input cultivo (C)
tor2 = 30; % actualizado                                                                % temp optima 2 p/ radiac : input cultivo (C)
tcr = 40; % actualizado                                                                 % temp critica de radiacion: input cultivo (C)

% parametros suelo para lluvia
CN = 40;            % input suelo
esuelo_pot = 7;     % input suelo
c_esuelo = 0.5;     % input suelo
S = 254*(100/(CN-1)); % input para escorrentia

% parametros rendimiento
ic_pot_t = 0.48; % actualizado
ic_in = 0.001; % actualizado
df = 37; % actualizado
Y = 0.19; % actualizado


% EUR potencial: input cultivo, valor teorico para maiz(g Mj m-2)
eur_pot = 1.36 * ones(size_datos(1:2)); % actualizado

% crecimiento de raiz ( para el caso de crecimiento constante)
crec_raiz = 30;

KC = 1.11;


WP = 15;
coefFWP = 0.6;


c_forma_pc = 1.3;
au_up_pc = 0.60;
au_down_pc = 0.15;