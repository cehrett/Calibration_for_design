% DCTO paper figures

%% Set path string and add paths
clc; clear all; close all;

direc = pwd; 
if direc(1)=='C' 
    dpath = 'C:\Users\carle\Documents\MATLAB\NSF DEMS\Phase 1\';
else
    dpath = 'E:\Carl\Documents\MATLAB\NSF-DEMS_calibration\';
end
clear direc;

% Add paths
addpath(dpath);
addpath([dpath,'stored_data']);
addpath([dpath,'Example']);
addpath([dpath,'Example\Ex_results']);

%% Compare DCTO and KOH+CTO posteriors for theta1,theta2 (with prior)
clc ; clearvars -except dpath ; close all ;

% Select discrepancy
discrep = 0;

% Load results
locstr = sprintf(['C:\\Users\\carle\\Documents',...
    '\\MATLAB\\NSF DEMS\\Phase 1\\',...
    'dual_calib\\dual_calib_stored_data\\'...
    '2019-11-20_DCTO_vs_KOHCTO_results']);
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
title('Prior and posterior distributions of \theta_1');
xlabel('\theta_1');
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
lg1 = legend('Prior dist.','KOH','DCTO','Optimum');
title('Prior and posterior distributions of \theta_2');
xlabel('\theta_2');
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
locstr = sprintf(['C:\\Users\\carle\\Documents',...
    '\\MATLAB\\NSF DEMS\\Phase 1\\',...
    'dual_calib\\dual_calib_stored_data\\'...
    '2019-10-31_SDOE_results_desvarest_nobs' ...
    int2str(obs_initial_size) '-'...
    int2str(obs_final_size)]);
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
xlabel('\theta_1');
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
xlabel('\theta_2');
yticks([]);
set(f2,'color','white');
flushLegend(lg2,'ne');


% Save them
figstr1 = ['FIG_dual_calib_SDOE_comp_theta1-d',int2str(discrep)];
figstr2 = ['FIG_dual_calib_SDOE_comp_theta2-d',int2str(discrep)];
set(f1,'PaperPositionMode','auto')
set(f2,'PaperPositionMode','auto')
% print(f1,figstr1,'-depsc','-r600')
% print(f2,figstr2,'-depsc','-r600')
end % end of for loop used to run this section for multiple discreps