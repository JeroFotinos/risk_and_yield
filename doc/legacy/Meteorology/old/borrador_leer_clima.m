clear all, close all
dirClima = 'C:\Users\gabri\Dropbox\AgroModel\Meteorology\clima_total2.csv';

fileID = fopen(dirClima);
C = textscan(fileID,'%d8 %s %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32 %f32',...
    'Delimiter',',','EmptyValue',NaN);
fclose(fileID);

for i = 3:size(C,2)
    mat_clima(:,i-2)=cell2mat(C(i));
end

x=624:680;
figure()
plot(x,mat_clima(x,1),'r-',x,mat_clima(x,10),'r.-',x,mat_clima(x,2),'g-',x,mat_clima(x,9),'g.-',...
        x,mat_clima(x,3),'b-',x,mat_clima(x,8),'b.-'), legend('T max hist','T max fut','T med hist','T med fut','T min hist','T min fut')
title('Temp maxima')
% 
% figure()
% plot(idx(600:end),mat_clima(600:end,2),idx(600:end),mat_clima(600:end,9))
% title('Temp media')
% 
% figure()
% plot(idx(600:end),mat_clima(600:end,3),idx(600:end),mat_clima(600:end,8))
% title('Temp min')
% 
% figure()
% plot(idx(600:end),mat_clima(600:end,7),idx(600:end),mat_clima(600:end,12))
% title('lluvias')



% ind_1=find(ismember(C{2},'2021/11/19'));
% ind_2=ind_1+100;
% C{2}(ind_1:ind_2)
% 
% Carray(ind_1:ind_2,:)



