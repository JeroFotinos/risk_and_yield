% clear all
% close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clasificacion de los cultivos
% elegir tipo de cultivo
% 2: maiz
% 1: soja
tipo_cultivo = 2; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parametros de los cultivos
if tipo_cultivo == 2
    % carga de coordenadas del cultivo
    load('mat_maiz_2021_lowres.mat')
    % load('mat_maiz_2020_lowres.mat')
    load('mat_maiz_2021_lat_lowres.mat')
    load('mat_maiz_2021_lon_lowres.mat')
    cultivo = clase_maiz_2021_lowres;
    cultivos_lat = lat_lowres;
    cultivos_lon = lon_lowres;

    load('mat_aguadisp_saocom_maiz_2021-2022_2.mat')
    % load('mat_aguadisp_saocom_maiz_2020-2021.mat')
    porc_agua_inicial = a_disp_campo/100;
    
    cargar_coordenadas_maiz2021
    lats_lons = mat_coord_m;
    mat_fs = mat_fs_m;
    no_days_cultivo = 120;
    
    load('mat_dds_maiz_est_lowres')
    load fechas_agua_m
    load agua_obs_m
    
elseif tipo_cultivo ==1
    % carga de coordenadas del cultivo
    
    load('mat_soja_2021_lowres.mat')
    % load('mat_soja_2020_lowres.mat')
    load('mat_soja_2021_lat_lowres.mat')
    load('mat_soja_2021_lon_lowres.mat')
    cultivo = clase_soja_2021_lowres;
    cultivos_lat = lat_lowres;
    cultivos_lon = lon_lowres;

    load('mat_aguadisp_saocom_soja_2021-2022_2.mat')
    % load('mat_aguadisp_saocom_soja_2020-2021.mat')
    porc_agua_inicial = a_disp_campo/100;
    
    cargar_coordenadas_soja2021
    lats_lons = mat_coord_s;
    mat_fs = mat_fs_s;
    no_days_cultivo = 137;
    
    load('mat_dds_soja_est_lowres')
    load fechas_agua_s
    load agua_obs_s
        
else 
    fprintf('\nError en el tipo de cultivo \n\n')
    return
end
tic;

for sim = 1:1
    
clima_y_fechas_2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% seteo de 0 de variables
var_historicas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cambiar fechas de siembras
load dds_datos_estimacion.mat
for j=1:size(coord,1)-1
    dds(coord(j,1),coord(j,2),1) = dds(coord(j,1),coord(j,2),1) + round(normrnd(0,dfecha(j)));
    dds_estimados(j,sim) = dds(coord(j,1),coord(j,2),1);
end
% dds(:,:,1) = max(dds(:,:,1),-34);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iteraciones
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = 1;
% dias desde siembra
dds(:,:,i) = dds(:,:,i)+ones(size(cultivo));
% profundidad de raiz
prof_raiz(:,:,i) = zeros(size_datos(1:2));
% agua dispinible y agua total 
aut = prof_suelo*(cc-pmp)*cultivo;

aud(:,:,1) = prof_suelo*(porc_agua_inicial-pmp).*cultivo;
aud(:,:,2) = 0.5*aut;
aud(:,:,3) = 0.5*aut;
aud(:,:,4) = 0.5*aut; 

perc_1_h(:,:,i) = max(0,aud(:,:,1)-aut);
perc_2_h(:,:,i) = max(0,aud(:,:,2)-aut);
perc_3_h(:,:,i) = max(0,aud(:,:,3)-aut);
dren_h(:,:,i) = max(0,aud(:,:,4)-aut);

aud(:,:,1) = min(aut,aud(:,:,1));
aud(:,:,2) = min(aut,aud(:,:,2));
aud(:,:,3) = min(aut,aud(:,:,3));
aud(:,:,4) = min(aut,aud(:,:,4));

% porcentaje de agua util
p_au = aud(:,:,1)./aut(:,:,1);
p_au_h(:,:,i)=p_au;
DD90 = ones(size(p_au));
DD90(p_au>0.9)=0;

% estrés hídrico relativo para IC
hrs_pc = (au_up_pc-p_au)./(au_up_pc-au_down_pc);

% coeficiente de estrés hídrico para IC
ceh_pc = 1 - (exp(hrs_pc*c_forma_pc) - 1)/(exp(c_forma_pc) - 1);
ceh_pc(p_au > au_up_pc) = 1;
ceh_pc(p_au < au_down_pc) = 0;

ceh_pc_h(:,:,i) = ceh_pc;

% porcentaje de cobertura total CT
% estres hidrico relativo para CT
hrs = (au_up-p_au)./(au_up-au_down);

% coeficiente de estres hidrico para CT
ceh = 1 - (exp(hrs*c_forma) - 1)/(exp(c_forma) - 1);
ceh(p_au < au_down) =  0;
ceh(au_up < p_au) = 1;
ceh_h(:,:,i)=ceh;
% funcion del porcentaje de cobertura diario
ct_i = ones(size(cultivo));
ct_i(dds(:,:,i) <= dds_in) = 0*ct_i(dds(:,:,i) <= dds_in);
ct_i(dds(:,:,i) == dds_in) = c_in*ct_i(dds(:,:,i) == dds_in);
ct(:,:,i) = ct_i;

% Eficiencia en el uso de la radiacion (EUR)
% estres hidrico relativo para EUR   
hrs_r = (au_up_r - p_au)./(au_up_r - au_down_r);

% coeficiente de estr�s h�drico para EUR
ceh_r = 1 - (exp(hrs_r*c_forma_r) - 1)/(exp(c_forma_r) - 1);
ceh_r(p_au > au_up_r) = 1;
ceh_r(p_au < au_down_r) = 0;
cehr_h(:,:,i)=ceh_r;
% temperatura media 
tm_i = tm_media(i) * ones(size_datos(1:2));

% Coeficiente de estres termico para EUR
t_eur = zeros(size(tm_i));
t_eur(tm_i > tor1 & tm_i < tor2)= 1;
t_eur(tm_i > tbr & tm_i < tor1) = (tm_i(tm_i > tbr & tm_i < tor1) - tbr)/(tor1 - tbr);
t_eur(tm_i > tor2 & tm_i < tcr) = (tcr - tm_i(tm_i > tor2 & tm_i < tcr))/(tcr - tor2);
t_eur_h(:,:,i)=t_eur.*cultivo;     


% EUR real: EUR portencial multiplicado por los estresores
eur_act_i = eur_pot .* t_eur .* ceh_r;   
eur_act_h(:,:,i)=eur_act_i;

ici_i = zeros(size_datos(1:2));
ddf = max(0,dds(:,:,i) - df); % (58 maiz) depende de la fecha de siembra

ic_pot = ic_pot_t*ones(size(dds(:,:,1)));
ici_i(ddf>0) = (ic_in*ic_pot(ddf>0))./(ic_in+(ic_pot(ddf>0)-ic_in).*exp(-Y.*ddf(ddf>0)));
ici(:,:,i) = ici_i;


% estimacion de la biomasa diaria (bi) y la acumulada o total (bt)
bi(:,:,i) = 0*t_eur;
bt(:,:,1) = bi(:,:,i);
rend(:,:,i) = bt(:,:,i).*ici(:,:,i).*ceh_r;

aud_1_h(:,:,i) = aud(:,:,1);
aud_2_h(:,:,i) = aud(:,:,2);
aud_3_h(:,:,i) = aud(:,:,3);
aud_4_h(:,:,i) = aud(:,:,4);

ct_old = ct_i;
ct_max = zeros(size(cultivo));

i_fechas = 1;
aux = 0;
aux2=0;

d_helada(119)=3;
tm_min(119)=-1;

FCCH_h = zeros(size(dds(:,:,1)));

for i=2:no_days+1
%     if isnan(ET0(i))
%         ET0(i) = ET0(i-30);
%     end

    dds(:,:,i)=dds(:,:,i-1)+ones(size(cultivo));
    % profundidad de raiz
    prof_raiz(:,:,i)=min(2000*ones(size_datos(1:2)) , prof_raiz(:,:,i-1) + (crec_raiz*ceh_r).*ones(size_datos(1:2)).*(dds(:,:,i)>0)).*cultivo;         
    % transpiracion
    transp = ceh_r.*(ct_i*KC)*ET0(i);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % precipitacion efectiva
    ind = find(pp_days(i)<pp_cotas,1);
    aux=[0;pp_days(i)*0.95; 23.75 + (pp_days(i)-25)*0.90;...
    46.25 + (pp_days(i)-50)*0.82; 66.75 + (pp_days(i)-75)*0.65;...
    83 + (pp_days(i)-100)*0.45;94.25 + (pp_days(i)-125)*0.25;...
    100.25 + (pp_days(i)-150)*0.05];
    ppef = aux(ind)*ones(size(cultivo));
        
       
    DD90 = DD90 + 1;                          % sumo un dia si hay menos de 90% de agua util
    DD90(p_au_h(:,:,i-1)>0.9) = 0;            % si hay más del 90% de agua util de dias a 0
    
    aux = ct(:,:,i-1);

    esuelo = 1.1*ET0(i)*(1-aux); % cuadno p_au_h(:,:,i-1) > .9
    esuelo(p_au_h(:,:,i-1)<0.9) = 1.1*ET0(i)*(1-aux(p_au_h(:,:,i-1)<0.9)).*(DD90(p_au_h(:,:,i-1)<0.9).^(-0.5));    
    
    % prof de raices
    pcapa = prof_raiz(:,:,i)<500;
    scapa = (prof_raiz(:,:,i)>500)&(prof_raiz(:,:,i)<1000);
    tcapa = (prof_raiz(:,:,i)>1000)&(prof_raiz(:,:,i)<1500);
    ccapa = prof_raiz(:,:,i)>1500;
    
    % agua util primer capa
    aud(:,:,1) = (aud_1_h(:,:,i-1) + ppef - esuelo).*cultivo;
    tmp = zeros(size(transp));
    if max(max(pcapa))>0
        tmp(pcapa) = transp(pcapa);
    end
    if max(max(scapa))>0
        tmp(scapa) = transp(scapa)/2;
    end
    if max(max(tcapa))>0
        tmp(tcapa) = transp(tcapa)/3;
    end
    if max(max(ccapa))>0
        tmp(ccapa) = transp(ccapa)/4;
    end
    tmp(isnan(tmp)) =0;
    aud(:,:,1) = aud(:,:,1) - tmp.*cultivo;

    perc_1_h(:,:,i) = max(0,aud(:,:,1)-aut);
    aud(:,:,1) = min(aut,max(0,aud(:,:,1)));
    % agua util segunda capa
    aud(:,:,2) = (aud_2_h(:,:,i-1) + perc_1_h(:,:,i-1)).*cultivo;
    tmp = zeros(size(transp));
    if max(max(scapa))>0
        tmp(scapa) = transp(scapa)/2;
    end
    if max(max(tcapa))>0
        tmp(tcapa) = transp(tcapa)/3;
    end
    if max(max(ccapa))>0
        tmp(ccapa) = transp(ccapa)/4;
    end
    tmp(isnan(tmp)) =0;
    aud(:,:,2) = aud(:,:,2) - tmp.*cultivo;
    perc_2_h(:,:,i) = max(0,aud(:,:,2)-aut);
    aud(:,:,2) = min(aut,max(0,aud(:,:,2)));
    
    % agua util tercer capa
    aud(:,:,3) = (aud_3_h(:,:,i-1) + perc_2_h(:,:,i-1)).*cultivo;
    tmp = zeros(size(transp));
    if max(max(tcapa))>0
        tmp(tcapa) = transp(tcapa)/3;
    end
    if max(max(ccapa))>0
        tmp(ccapa) = transp(ccapa)/4;
    end
    tmp(isnan(tmp)) =0;
    aud(:,:,3) = aud(:,:,3) - tmp.*cultivo;
    perc_3_h(:,:,i) = max(0,aud(:,:,3)-aut);
    aud(:,:,3) = min(aut,max(0,aud(:,:,3)));
    
    % agua util cuarta capa
    aud(:,:,4) = (aud_4_h(:,:,i-1) + perc_3_h(:,:,i-1)).*cultivo;
    tmp = zeros(size(transp));
    if max(max(ccapa))>0
        tmp(ccapa) = transp(ccapa)/4;
    end
    tmp(isnan(tmp)) =0;
    aud(:,:,4) = aud(:,:,4) - tmp.*cultivo;
    dren_h(:,:,i) = max(0,aud(:,:,4)-aut);
    aud(:,:,4) = min(aut,max(0,aud(:,:,4)));
    
    aud_1_h(:,:,i) = aud(:,:,1);
    aud_2_h(:,:,i) = aud(:,:,2);
    aud_3_h(:,:,i) = aud(:,:,3);
    aud_4_h(:,:,i) = aud(:,:,4);
    
    
    sum2 = zeros(size(prof_raiz(:,:,i)));
    aud2 = aud(:,:,2);
    sum2(prof_raiz(:,:,i)>500) = aud2(prof_raiz(:,:,i)>500);

    sum3 = zeros(size(prof_raiz(:,:,i)));
    aud3 = aud(:,:,3);
    sum3(prof_raiz(:,:,i)>1000) = aud3(prof_raiz(:,:,i)>1000);

    sum4 = zeros(size(prof_raiz(:,:,i)));
    aud4 = aud(:,:,4);
    sum4(prof_raiz(:,:,i)>1500) = aud4(prof_raiz(:,:,i)>1500);

    
    p_au = aud(:,:,1) + sum2 + sum3 + sum4;
    
    p_au = p_au./(aut.*(ones(size(prof_raiz(:,:,i)))+(prof_raiz(:,:,i)>500) +(prof_raiz(:,:,i)>1000)+(prof_raiz(:,:,i)>1500)));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % historial de datos
    ppef_h(:,:,i) = ppef;
    tra_h(:,:,i) = transp;
    ddp_h(:,:,i) = ddp;
    eva_h(:,:,i) = esuelo;
    p_au_h(:,:,i)=p_au;
    
    % estrés hídrico relativo para IC
    hrs_pc = (au_up_pc-p_au)./(au_up_pc-au_down_pc);
    
    % coeficiente de estrés hídrico para IC
    ceh_pc = 1 - (exp(hrs_pc*c_forma_pc) - 1)/(exp(c_forma_pc) - 1);
    ceh_pc(p_au > au_up_pc) = 1;
    ceh_pc(p_au < au_down_pc) = 0;
    
    aux = ones(size(ddf));
    ceh_pc(ddf<-15) = aux(ddf<-15);
    ceh_pc(ddf>15) = aux(ddf>15);
    
    ceh_pc_h(:,:,i) = ceh_pc;

    % estres hidrico relativo para CT
    hrs = (au_up-p_au)./(au_up-au_down);

    % coeficiente de estr�s hidrico para CT
    ceh = 1 - (exp(hrs*c_forma) - 1)/(exp(c_forma) - 1);
    ceh(p_au < au_down) =  0;
    ceh(au_up < p_au) = 1;
    ceh_h(:,:,i)=ceh;
    % funcion del porcentaje de cobertura diario
    ct_i = ones(size(ct(:,:,i)));
    ct_old=ct(:,:,i-1);
    
    ct_i(dds(:,:,i) <= dds_in) = 0*ct_i(dds(:,:,i) <= dds_in);
    ct_i(dds(:,:,i) == dds_in) = c_in*ct_i(dds(:,:,i) == dds_in);
    
    ct_i(dds(:,:,i) > dds_in & dds(:,:,i) < dds_max) = ct_old(dds(:,:,i) > dds_in & dds(:,:,i) < dds_max) + (alpha1*ceh(dds(:,:,i) > dds_in & dds(:,:,i) < dds_max)).*ct_i(dds(:,:,i) > dds_in & dds(:,:,i) < dds_max);
    ct_i(dds(:,:,i) >= dds_max & dds(:,:,i) < dds_sen) = ct_old(dds(:,:,i) >= dds_max & dds(:,:,i) < dds_sen);
    
    ct_max = max(ct_max,ct_old);
    
    beta1 = (ct_max - c_fin)/(dds_fin - dds_sen);                    % tasa de decrecimiento de cobertura potencial diario
     
    ct_i(dds(:,:,i) >= dds_sen) = max(ct_old(dds(:,:,i) >= dds_sen) - beta1(dds(:,:,i) >= dds_sen).*(2-ceh(dds(:,:,i) >= dds_sen)).*ct_i(dds(:,:,i) >= dds_sen),c_fin);
    ct_i(dds(:,:,i) >= dds_sen & ct_old<=c_fin) = ct_old(dds(:,:,i) >= dds_sen & ct_old<=c_fin);
    ct(:,:,i) = ct_i;
    ct_old = ct_i;
    % Eficiencia en el uso de la radiacion (EUR)
    % estres hidrico relativo para EUR   
    hrs_r = (au_up_r - p_au)./(au_up_r-au_down_r);

    % coeficiente de estres hidrico para EUR
    ceh_r = 1 - (exp(hrs_r*c_forma_r) - 1)/(exp(c_forma_r) - 1);
    ceh_r(p_au > au_up_r) = 1;
    ceh_r(p_au < au_down_r) = 0;
    
    
    cehr_h(:,:,i)=ceh_r;
    % temperatura media 
    tm_i = tm_media(i) * ones(size_datos(1:2));

    % Coeficiente de estr�s termico para EUR
    t_eur = zeros(size(tm_i));
    t_eur(tm_i > tor1 & tm_i < tor2)= 1;
    t_eur(tm_i > tbr & tm_i < tor1) = (tm_i(tm_i > tbr & tm_i < tor1) - tbr)/(tor1 - tbr);
    t_eur(tm_i > tor2 & tm_i < tcr) = (tcr - tm_i(tm_i > tor2 & tm_i < tcr))/(tcr - tor2);
    t_eur_h(:,:,i)=t_eur.*cultivo;

    % EUR real: EUR portencial multiplicado por los estresores
    eur_act_i = eur_pot .* ceh_r .* t_eur;    
    eur_act_h(:,:,i)=eur_act_i;
    
    ici_i = zeros(size_datos(1:2));
    ddf = dds(:,:,i) - df;    
    
    FWP = ones(size(ddf));
    FWP(ddf>1) = coefFWP;
    ic_pot = ic_pot_t*ones(size(ddf));
    ind_fecha_flor = zeros(size(ddf));
    for k1 = 1:size(ddf,1)
        for k2 = 1:size(ddf,2)
            
            if ddf(k1,k2)>=15
                delta = ddf(k1,k2) - 15;
                delta2 = max(31,i-delta);
                ic_pot(k1,k2) = ic_pot(k1,k2)*sum(ceh_pc_h(k1,k2,delta2-30:delta2))/30;
            end
            
        end
    end
        

    ici_i(ddf>0) = (ic_in*ic_pot(ddf>0))./(ic_in+(ic_pot(ddf>0)-ic_in).*exp(-Y.*ddf(ddf>0)));
    

    ici(:,:,i) = ici_i;
    
    % estimacion de la biomasa diaria (bi) y la acumulada o total (bt)
    bi(:,:,i) = max(0,t_eur_h(:,:,i).*WP.*FWP.*transp/ET0(i));
    
    % FCCH - factor de daño en biomasa por helada
    FCCH = zeros(size(dds(:,:,i)));
    
    if (tm_min(i)<0) && (d_helada(i)>=4)
        FCCH((20<dds(:,:,i))&(dds(:,:,i)<=80)) = 0;
        FCCH((80<dds(:,:,i))&(dds(:,:,i)<=100)) = 0.5;
        FCCH((100<dds(:,:,i))&(dds(:,:,i)<=120)) = 0.3;
    elseif ((4>tm_min(i))&&(tm_min(i)>=0)) && (d_helada(i)>=1)
        FCCH((20<dds(:,:,i))&(dds(:,:,i)<=80)) = 0.3;
        FCCH((80<dds(:,:,i))&(dds(:,:,i)<=100)) = 0.2;
        FCCH((100<dds(:,:,i))&(dds(:,:,i)<=120)) = 0.1;
    end
    
    FCCH_h = max(FCCH_h,FCCH);
    
    bi(:,:,i) = max(0,ct(:,:,i).* (par(i)*ones(size_datos(1:2))) .* eur_act_i);
    
    bt(:,:,i) = bt(:,:,i-1) + bi(:,:,i);
    
    rend(:,:,i) = bt(:,:,i).*ici(:,:,i).*ceh_r;
end

for i = 1:size(rend,3)
    rend(:,:,i) = rend(:,:,i).*(1-FCCH_h);
end

delta_lat = abs(cultivos_lat(1,1)-cultivos_lat(2,1))/2;
delta_lon = abs(cultivos_lon(1,1)-cultivos_lon(1,2))/2;
coord=zeros(size(lats_lons));

for s=1:size(lats_lons,1)
    coord(s,:)=[find(cultivos_lat(:,1)<lats_lons(s,1)+delta_lat & lats_lons(s,1)-delta_lat<cultivos_lat(:,1)),...
        find(cultivos_lon(1,:)<lats_lons(s,2)+delta_lon & lats_lons(s,2)-delta_lon<cultivos_lon(1,:))];
end

rend_final = rend(:,:,end);

% redistribucion de rendimientos
max_rend=0;
for i = 1:size(coord,1)
    max_rend = max(max_rend,rend_final(coord(i,1),coord(i,2)));
end
min_rend=1e10;
for i = 1:size(coord,1)
    if rend_final(coord(i,1),coord(i,2)) > 0
        min_rend = min(min_rend,rend_final(coord(i,1),coord(i,2)));
    end
end

c0 = - min_rend/(max_rend - min_rend); 
c1 = 1/(max_rend - min_rend); 

grado = 1;

c2 = 0.6;
c3 = 1-c2;
esc_pol = c3*((c1*rend_final+c0).^grado)+c2;
for  i = 1:size(rend,3)
    rend_esc(:,:,i) = rend(:,:,i) .* esc_pol;
end

RR=[];
for s = 1 : length(coord)
    R=zeros(no_days+1,34);
    R(:,1)=1:no_days+1; % num dia
    R(:,2)=s*ones(size(no_days,1)); % num_lote
    R(:,3)=tm_media; % tmp media (datos)
    R(:,4)=par; % par (datos)
    R(:,5)=pp(coord(s,1),coord(s,2),:); % precipitaciones (datos)
    R(:,6)=prof_raiz(coord(s,1),coord(s,2),:); % profRaiz"
    R(:,7)=ddp_h(coord(s,1),coord(s,2),:); %     diasDesdePrec"
    R(:,8)=tra_h(coord(s,1),coord(s,2),:); %     transpiracion"
    R(:,9)=ppef_h(coord(s,1),coord(s,2),:); %     "precipitacion_efectiva"
    R(:,10)=eva_h(coord(s,1),coord(s,2),:); %   "evaporacion"
    R(:,11)=perc_1_h(coord(s,1),coord(s,2),:); %    "percolacion_capa_1"
    R(:,12)=perc_2_h(coord(s,1),coord(s,2),:); %   "percolacion_capa_2"
    R(:,13)=perc_3_h(coord(s,1),coord(s,2),:); %   "percolacion_capa_3"
    R(:,14)=dren_h(coord(s,1),coord(s,2),:);
    R(:,15)=aud_1_h(coord(s,1),coord(s,2),:);
    R(:,16)=aud_2_h(coord(s,1),coord(s,2),:);
    R(:,17)=aud_3_h(coord(s,1),coord(s,2),:);
    R(:,18)=aud_4_h(coord(s,1),coord(s,2),:);
    R(:,19) = p_au_h(coord(s,1),coord(s,2),:);
    R(:,20) = ceh_h(coord(s,1),coord(s,2),:);
    R(:,21) = cehr_h(coord(s,1),coord(s,2),:);
    R(:,22) = ct(coord(s,1),coord(s,2),:);
    R(:,23) = t_eur_h(coord(s,1),coord(s,2),:);
    R(:,24) = eur_act_h(coord(s,1),coord(s,2),:);
    R(:,25) = bi(coord(s,1),coord(s,2),:);
    R(:,26) = bt(coord(s,1),coord(s,2),:);
    R(:,27) = rend(coord(s,1),coord(s,2),:);
    R(:,28) = ici(coord(s,1),coord(s,2),:);
    R(:,29) = dpv(:);
    R(:,30) = dds(coord(s,1),coord(s,2),:);
    R(:,31) = ET0;
    R(:,32) = ceh_pc_h(coord(s,1),coord(s,2),:);
    R(:,33) = rend_esc(coord(s,1),coord(s,2),:);
    R(:,34) = FCCH_h(coord(s,1),coord(s,2),:);
    
    rend_var(s,sim) = 10*R(end,27);
    
    FF = convertTo(climatotal_nuevo.FECHA(d_ini_min:d_ini_min+no_days),'yyyymmdd');
    RR=[RR;FF,R];
end
end

if tipo_cultivo == 2
    name = strcat('datos_maiz.csv');
elseif tipo_cultivo == 1
    name = strcat('datos_soja.csv');
end

var_names = ["fecha",...
    "numero_dia",...
    "num_lote",...
    "temp_media",...
    "PAR",...
    "precipitaciones",...
    "profRaiz",...
    "diasDesdePrec",...
    "transpiracion",...
    "precipitacion_efectiva",...
    "evaporacion",...
    "percolacion_capa_1",...
    "percolacion_capa_2",...
    "percolacion_capa_3",...
    "drenaje",...
    "aguaUtilDisp1",...
    "aguaUtilDisp2",...
    "aguaUtilDisp3",...
    "aguaUtilDisp4",...
    "porcAguaUtil",...
    "cEstresHidrico",...
    "cEstresHidricoRadiacion",...
    "porcCobertura",...
    "cEstrsTerminco",...
    "EUR",...
    "biomasaDiaria",...
    "biomasaAcumulada",...
    "rendimiento",...
    "ici",...
    "DPV",...
    "DDS",...
    "ET0",...
    "che_pc",...
    "rend escalado",...
    "FCCH"];

RR = [var_names;RR];
writematrix(RR,name)

% % % 
% clims_t=[0 max(max(rend(:,:,end)))];
% 
% figure()
% 
% h = figure;
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = 'rendimiento.gif';
% 
% for i=1:no_days+1
%     imagesc(rend(:,:,i),clims_t)
%     title(strcat('Dia: ',datestr(climatotal_nuevo.FECHA(d_ini_min+i))))
%     colormap(hot)
%     colorbar
%     
%     drawnow
%     
%     pause(0.1)
%     
%     
%     % Capture the plot as an image 
%     frame = getframe(h); 
%     im = frame2im(frame); 
%     [imind,cm] = rgb2ind(im,256); 
%     % Write to the GIF File 
%     if i == 1 
%         imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
%     else 
%         imwrite(imind,cm,filename,'gif','WriteMode','append'); 
%     end 
%     
% end