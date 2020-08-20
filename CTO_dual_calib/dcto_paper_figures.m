% DCTO paper figures

%% Set path string and add paths
clc; clear all; close all;

direc = pwd; 
if direc(1)=='C' 
    dpath = 'C:\Users\Carl\Documents\MATLAB\NSF_DEMS\NSF-DEMS_calibration\';
else
    dpath = 'E:\Carl\Documents\MATLAB\NSF-DEMS_calibration\';
end
clear direc;

% Add paths
addpath(dpath);
addpath([dpath,'stored_data']);
addpath([dpath,'Example']);
addpath([dpath,'dual_calib']);

% Change dir
cd(dpath);

%% Get example computer model output
clc ; clearvars -except dpath ; close all ; 

fig=figure('Position',[10 10 400 400]);

% Define inputs
xmin = .5;
xrange = .5;
x = linspace(0,1);
t1min = 1.5;
t1range = 3;
t1=linspace(0,1);
t2min = 0;
t2range = 5;
t2 = linspace(0,1);

[X,T1,T2] = meshgrid(x,t1,t2) ; 
Y = reshape(dual_calib_example_fn(X(:),xmin,xrange,T1(:),t1min,t1range,...
    T2(:),t2min,t2range,0,1,0,true),length(x),length(t1),length(t2));

% Take a look
xidx=100;
xx=reshape(X(:,xidx,:),100,100);
tt1=reshape(T1(:,xidx,:),100,100);
tt2=reshape(T2(:,xidx,:),100,100);
surfax = ...
    surf(tt1*t1range+t1min,tt2*t2range+t2min,...
    reshape(Y(:,xidx,:),100,100),...
    'EdgeAlpha',.4);
xlabel('t_c');ylabel('t_d');zlabel('f(x,t_c,t_d)');

fig.Children.View = [255 8];
set(fig,'color','w');

figstr = 'FIG_obj_fn';
set(fig,'PaperPositionMode','auto')
% print(fig,figstr,'-depsc','-r600')

%% Compare DCTO and KOH+CTO posteriors for theta1,theta2 (with prior)
clc ; clearvars -except dpath ; close all ;

% Select discrepancy
discrep = 0;

% Load results
locstr=[dpath,'\dual_calib\dual_calib_stored_data\2019-11-20_DCTO_vs_KOHCTO_results'];
load(locstr);

% Define inputs mins and ranges 
xmin = .5;
xrange = .5;
t1min = 1.5;
t1range = 3;
t2min = 0;
t2range = 5;


% Get DCTO  and KOH+CTO results
burn_in = results{1}.settings{1}.burn_in;
dcto_t1 = results{discrep+1,1}.theta1(burn_in:end,10);
dcto_t2 = results{discrep+1,1}.theta2(burn_in:end,10);
khct_t1 = results{discrep+1,2}.theta1(burn_in:end,10);
khct_t2 = results{discrep+1,3}.theta2(burn_in:end,10);


% Help function
fillunder = @(x,y,color,falpha) ...
    fill([x(1) x x(end) fliplr(x) 0],...
        [0 y 0 0*y 0],color,'EdgeColor','none','FaceAlpha',falpha);
    
% First, get prior and posterior theta1
f1 = figure('pos',[10 10 300 200]);
% Plot prior
falpha=0.5; % face alpha for posteriors
fillunder([t1min t1min+t1range],[1/t1range 1/t1range],'g',1);
xlim([t1min t1min + t1range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[pd1,xd1,bwd1] = ksdensity(dcto_t1);
[pk1,xk1,bwk1] = ksdensity(khct_t1);
fillunder(xk1,pk1,'r',falpha);
fillunder(xd1,pd1,'b',falpha);
% Plot true theta1
theta1 = results{discrep+1,1}.true_theta1;
plot([theta1 theta1],get(gca,'YLim'),'--','Color',[.85 .85 0],...
    'LineWidth',1.5);
% Put a legend on it
lg1 = legend('Prior dist.','KOH','DCTO','True value');
title('Prior and posterior distributions of \theta_c');
xlabel('\theta_c');
yticks([]);
set(f1,'color','white');


% First, get prior and posterior theta1
f2 = figure('pos',[320 10 300 200]);
% Plot prior
falpha=0.5; % face alpha for posteriors
fillunder([t2min t2min+t2range],[1/t2range 1/t2range],'g',1);
xlim([t2min t2min + t2range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[pd2,xd2,bwd2] = ksdensity(dcto_t2);
[pk2,xk2,bwk2] = ksdensity(khct_t2);
fillunder(xk2,pk2,'r',falpha);
fillunder(xd2,pd2,'b',falpha);
% Plot true theta1
theta2 = results{discrep+1,1}.true_theta2;
plot([theta2 theta2],get(gca,'YLim'),'--','Color',[.85 .85 0],...
    'LineWidth',1.5);
% Put a legend on it
lg2 = legend('Prior dist.','CTO','DCTO','Optimum');
title('Prior and posterior distributions of \theta_d');
xlabel('\theta_d');
yticks([]);
set(f2,'color','white');


% Save them
figstr1 = 'FIG_dual_calib_post_theta1-d0';
figstr2 = 'FIG_dual_calib_post_theta2-d0';
set(f1,'PaperPositionMode','auto')
set(f2,'PaperPositionMode','auto')
% print(f1,figstr1,'-depsc','-r600')
% print(f2,figstr2,'-depsc','-r600')

%% Compare SDOE and PDOE posteriors for theta1,theta2 (with prior)
clc ; clearvars -except dpath ; close all ;

% Load results
obs_initial_size = 0 ; obs_final_size = 20;
locstr = [dpath,'dual_calib\dual_calib_stored_data\'...
    '2019-10-31_SDOE_results_desvarest_nobs' ...
    int2str(obs_initial_size) '-'...
    int2str(obs_final_size)];
load(locstr,'results');

close all;
% Figure height and width
fh = 120;
fw = 300;

% Select discrepancy and specific calib run
discrep = 0;
for discrep = 0:6
run = 10;

% Define inputs mins and ranges 
xmin = .5  ;
xrange = .5;
t1min = 1.5;
t1range = 3;
t2min = 0  ;
t2range = 5;


% Get DCTO  and KOH+CTO results
burn_in = results{1}.settings{1}.burn_in;
sdoe_t1 = results{discrep+1,1}.theta1(burn_in:end,run);
sdoe_t2 = results{discrep+1,1}.theta2(burn_in:end,run);
pdoe_t1 = results{discrep+1,2}.theta1(burn_in:end,run);
pdoe_t2 = results{discrep+1,2}.theta2(burn_in:end,run);


% Help function
fillunder = @(x,y,color,falpha) ...
    fill([x(1) x x(end) fliplr(x) 0],...
        [0 y 0 0*y 0],color,'EdgeColor','none','FaceAlpha',falpha);
    
% First, get prior and posterior theta1
f1 = figure('pos',[10 10 fw fh]);
% Plot prior
falpha=0.5; % face alpha for posteriors
fillunder([t1min t1min+t1range],[1/t1range 1/t1range],'g',1);
xlim([t1min t1min + t1range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[ps1,xs1,bws1] = ksdensity(sdoe_t1);
[pp1,xp1,bwp1] = ksdensity(pdoe_t1);
fillunder(xp1,pp1,'r',falpha);
fillunder(xs1,ps1,'b',falpha);
% Plot true theta1
theta1 = results{discrep+1,1}.true_theta1;
plot([theta1 theta1],get(gca,'YLim'),'--','Color',[.85 .85 0],...
    'LineWidth',1.5);
% Put a legend on it
lg1 = legend('Prior dist.','SFD','AS','True value');
% title('Prior and posterior distributions of \theta_1');
xlabel('\theta_c');
yticks([]);
set(f1,'color','white');
flushLegend(lg1,'ne');


% First, get prior and posterior theta1
f2 = figure('pos',[fw+20 10 fw fh]);
% Plot prior
falpha=0.5; % face alpha for posteriors
fillunder([t2min t2min+t2range],[1/t2range 1/t2range],'g',1);
xlim([t2min t2min + t2range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[ps2,xs2,bws2] = ksdensity(sdoe_t2);
[pp2,xp2,bwp2] = ksdensity(pdoe_t2);
fillunder(xp2,pp2,'r',falpha);
fillunder(xs2,ps2,'b',falpha);
% Plot true theta1
theta2 = results{discrep+1,1}.true_theta2;
plot([theta2 theta2],get(gca,'YLim'),'--','Color',[.85 .85 0],...
    'LineWidth',1.5);
% Put a legend on it
lg2 = legend('Prior dist.','SFD','AS','Optimum');
% title('Prior and posterior distributions of \theta_2');
xlabel('\theta_d');
yticks([]);
set(f2,'color','white');
flushLegend(lg2,'ne');


% Save them
figstr1 = ['FIG_dual_calib_SDOE_comp_theta1-d',int2str(discrep)];
figstr2 = ['FIG_dual_calib_SDOE_comp_theta2-d',int2str(discrep)];
set(f1,'PaperPositionMode','auto')
set(f2,'PaperPositionMode','auto')
print(f1,figstr1,'-depsc','-r600')
print(f2,figstr2,'-depsc','-r600')
end % end of for loop used to run this section for multiple discreps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New figures for version 2.0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Show data used in KOH+CTO vs DCTO comparison (including added "noise")
clc ; clearvars -except dpath ; close all ;

% Select discrepancy
discrep = 0;

% Load results
locstr=[dpath,...
    '\dual_calib\dual_calib_stored_data\',...
    '2019-11-20_DCTO_vs_KOHCTO_results'];
load(locstr);

% For convenience
% clc ; clearvars -except dpath discrep results ; close all ;

fig=figure('Position',[10 10 400 300]);

% Define inputs
xmin = .5;
xrange = .5;
x = linspace(0,1);
t1min = 1.5;
t1range = 3;
t2min = 0;
t2range = 5;
t2 = linspace(0,1);

[X,T2] = meshgrid(x,t2) ; 
T1 = ones(size(T2)) * (2-t1min)/(t1range) ;
Y = reshape(dual_calib_example_fn(X(:),xmin,xrange,...
    T1(:),t1min,t1range,...
    T2(:),t2min,t2range,...
    0,1,... % rescale output using this mean and sd
    0,... % discrep
    true),... % rescale_inputs
    length(x),length(t2));

% Take a look
xx=reshape(X,100,100);
tt1=reshape(T1,100,100);
tt2=reshape(T2,100,100);
surfax = ...
    surf(xx*xrange+xmin,tt2*t2range+t2min,...
    reshape(Y,100,100),...
    'EdgeAlpha',.4);
xlabel('x');ylabel('t_d');zlabel('f(x,2,t_d)');

% Add "real" data points
hold on;
obs_x = results{discrep+1,1}.settings{1}.obs_x * xrange + xmin;
obs_t2 = results{discrep+1,1}.settings{1}.obs_t2 * t2range + t2min;
ymean = results{discrep+1,1}.settings{1}.mean_y;
ystd = results{discrep+1,1}.settings{1}.std_y;
obs_y = results{discrep+1,1}.settings{1}.obs_y * ystd + ymean;
plot3(obs_x,obs_t2,obs_y,'.','color','red','MarkerSize',25);
% Get "true" values at observation points
true_y = dual_calib_example_fn(obs_x,xmin,xrange,...
    ones(size(obs_x))*2,t1min,t1range,...
    obs_t2,t2min,t2range,...
    0,1,...
    0,...
    false);
for ii = 1 : length(true_y)
   
    line([obs_x(ii),obs_x(ii)],...
        [obs_t2(ii),obs_t2(ii)],...
        [obs_y(ii),true_y(ii)],...
        'color','red',...
        'linewidth',2)
    
end

fig.Children.View = [-40 15];
set(fig,'color','w');

figstr = 'FIG_observed_data';
set(fig,'PaperPositionMode','auto')
print(fig,figstr,'-depsc','-r600')