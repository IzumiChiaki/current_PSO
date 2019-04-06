clear all
clc
x=xlsread('E_plot.csv','A2:A691');
y=xlsread('E_plot.csv','B2:B691');
z=xlsread('E_plot.csv','C2:C691');
E=xlsread('E_plot.csv','D2:D691');
scatter3(x,y,z,E+1,E,'filled')
colorbar
x1=xlabel('T (^oC)','FontSize',12);        
x2=ylabel('H (%)','FontSize',12);        
x3=zlabel('PM2.5','FontSize',12);        
set(x1,'Rotation',20);    
set(x2,'Rotation',-30)