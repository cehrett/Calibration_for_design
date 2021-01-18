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
% print(f1,figstr1,'-depsc','-r600')
% print(f2,figstr2,'-depsc','-r600')
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

fig=figure('Position',[10 10 325 300]);

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

% Add ticks to t_d axis

yticks([0,1,2,3,4,5]);

fig.Children.View = [-40 15];
set(fig,'color','w');

figstr = 'FIG_observed_data';
set(fig,'PaperPositionMode','auto')
% print(fig,figstr,'-depsc','-r600')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Version 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% See objective function output as f'n of tc, td when tc depends on td
clc ; clearvars -except dpath ; close all ;

% Define inputs mins and ranges 
xmin = .5;
xrange = .5;
t1min = 1.5;
t1range = 3;
t2min = 0;
t2range = 5;

% Set discrepancy
discrep = 0;

% Set base theta1
low_theta1 = 1.5
high_theta1 = 2.25

% The function
t1_fn = @(x) high_theta1 - ...
    (high_theta1-low_theta1) * ...
    exp(40*((x-t2min)/t2range)-20)./...
    (1+exp(40*((x-t2min)/t2range)-20));

% Let's take a look at the objective function values for set x, using true
% t1 function as well as just base theta1
x = 1;
t2 = linspace(t2min,t2min+t2range,100)';
t1 = t1_fn(t2);
all_t1 = linspace(t1min,t1min+t1range,100)';
% Get optimal theta2 for each t1
fmfn_t = @(z,t) dual_calib_example_fn(x,xmin,xrange,...
    t,t2min,t1range,...
    z,t2min,t2range,...
    0,1,...
    0,...
    false);
theta2 = nan(size(all_t1));
for ii = 1:length(all_t1)
    fmfn = @(z) fmfn_t(z,all_t1(ii));
    theta2(ii) = fmincon(fmfn,2,[],[],[],[],t2min,t2range);
end
y_all = dual_calib_example_fn(x,xmin,xrange,all_t1,t1min,t1range,...
    theta2,t2min,t2range,0,1,discrep,false);

y = dual_calib_example_fn(x,xmin,xrange,t1,t1min,t1range,...
    t2,t2min,t2range,0,1,discrep,false);

f = figure('pos',[20 20 300 450]);
[m,i] = min(y) ; 
t2opt = t2(i)
t1opt = t1(i)

subplot(2,1,1);
plot(all_t1,y_all,'-','LineWidth',2);
xlabel('t_c');
ylabel('f(1, t_c, \theta_d(t_c))');
xline(t1opt,'r','LineWidth',2);
xlim([min(all_t1),max(all_t1)]);
label = 'True \theta_c value at t_d = \theta_d ';
text(t1opt+.1,0.075,label,'Interpreter','tex');

subplot(2,1,2);
plot(t2,y,'LineWidth',2);
xlabel('t_d');
ylabel('f(1, \theta_c(t_d), t_d)');
hold on;
% plot(t2,y_wrong);
% plot(t2,y_wrong2);
xline(t2opt,'r','LineWidth',2);
label = sprintf('Optimal\n\\theta_d value');
text(t2opt+.1,.9,label);

% Save it
set(f,'Color','w');
savestr = 'FIG_true_optimal_theta1_theta2';
set(f,'PaperPositionMode','auto')
print(f,savestr,'-depsc','-r600')

%% Compare SDOE and PDOE posteriors for theta1,theta2 (with prior), B&W ver
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


% Get SDOE and PDOE results
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
falpha=0.1; % face alpha for posteriors
fillunder([t1min t1min+t1range],[1/t1range 1/t1range],'g',1);
xlim([t1min t1min + t1range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[ps1,xs1,bws1] = ksdensity(sdoe_t1);
[pp1,xp1,bwp1] = ksdensity(pdoe_t1);
plot(xp1,pp1,'color','r','LineStyle',':','linewidth',2);
plot(xs1,ps1,'color','b','linewidth',2);
% Plot true theta1
theta1 = results{discrep+1,1}.true_theta1;
plot([theta1 theta1],get(gca,'YLim'),'--','Color','k',...
    'LineWidth',1.5);
fillunder(xp1,pp1,'r',falpha);
fillunder(xs1,ps1,'b',falpha);
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
fillunder([t2min t2min+t2range],[1/t2range 1/t2range],'g',1);
xlim([t2min t2min + t2range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[ps2,xs2,bws2] = ksdensity(sdoe_t2);
[pp2,xp2,bwp2] = ksdensity(pdoe_t2);
plot(xp2,pp2,'color','r','linestyle',':','linewidth',2);
plot(xs2,ps2,'color','b','linewidth',2);
% Plot true theta1
theta2 = results{discrep+1,1}.true_theta2;
plot([theta2 theta2],get(gca,'YLim'),'--','Color','k',...
    'LineWidth',1.5);
fillunder(xp2,pp2,'r',falpha);
fillunder(xs2,ps2,'b',falpha);
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
% print(f1,figstr1,'-depsc','-r600')
% print(f2,figstr2,'-depsc','-r600')
end % end of for loop used to run this section for multiple discreps

%% Show distribution of obj fn observations under AS and SFD
clc ; clearvars -except dpath ; close all ;

% Load results
obs_initial_size = 0 ; obs_final_size = 20;
locstr = [dpath,'dual_calib\dual_calib_stored_data\'...
    '2020-10-13_SDOE_results_desvarest_nobs' ...
    int2str(obs_initial_size) '-'...
    int2str(obs_final_size)];
load(locstr,'results');

% Pull out the discrep=0 results for AS and for SFD
results_as = results{2,1};
results_sfd = results{2,2};

% Pull out mins, ranges
t2min = results_as.settings{1}.min_t2;
t2range = results_as.settings{1}.range_t2;
xmin = results_as.settings{1}.min_x;
xrange = results_as.settings{1}.range_x;

% Pull out the observation locations
obs_t2_as = results_as.settings{1}.obs_t2 * t2range + t2min;
obs_t2_sfd = results_sfd.settings{1}.obs_t2 * t2range + t2min;
obs_x_as = results_as.settings{1}.obs_x * xrange + xmin;
obs_x_sfd = results_sfd.settings{1}.obs_x * xrange + xmin;

% Plot
f = figure('pos',[50 50 275 250]);
fa=.2 ; % facealpha
histogram(obs_t2_as,'BinWidth',1,'DisplayStyle','stairs','linewidth',3,'EdgeColor','b');
hold on;
histogram(obs_t2_sfd,'BinWidth',1,'DisplayStyle','stairs','linewidth',3,'linestyle',':','EdgeColor','r');
xline(results_as.true_theta2,'color','g','linewidth',2);
histogram(obs_t2_as,'BinWidth',1,'FaceAlpha',fa,'FaceColor','b');
histogram(obs_t2_sfd,'BinWidth',1,'FaceAlpha',fa,'FaceColor','r','EdgeColor','none');
ylim([0,8]);
leg = legend('AS','SFD','\theta_d');
xlabel('t_d');
ylabel('Observations');
legend boxoff
title('Observation locations');
flushLegend(leg,'ne');
set(f,'color','white');

% Save
figstr = ['FIG_AS_vs_SFD_obs_locs'];
set(f,'PaperPositionMode','auto')
print(f,figstr,'-depsc','-r600')

%% Compare DCTO, KOH+CTO posteriors for theta1,theta2 (with prior), B&W ver
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
falpha=0.1; % face alpha for posteriors
lalpha=0.65; % line alpha for posteriors
fillunder([t1min t1min+t1range],[1/t1range 1/t1range],'g',1);
xlim([t1min t1min + t1range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[pd1,xd1,bwd1] = ksdensity(dcto_t1);
[pk1,xk1,bwk1] = ksdensity(khct_t1);
plot1 = plot(xk1,pk1,'LineWidth',2,'color','r','LineStyle',':');
plot1.Color(4)=lalpha;
plot2 = plot(xd1,pd1,'LineWidth',2,'color','b');
plot2.Color(4)=lalpha;
% Plot true theta1
theta1 = results{discrep+1,1}.true_theta1;
plot([theta1 theta1],get(gca,'YLim'),'--','Color','k',...
    'LineWidth',1.5);
fillunder(xk1,pk1,'r',falpha);
fillunder(xd1,pd1,'b',falpha);
% Put a legend on it
lg1 = legend('Prior dist.','KOH','DCTO','True value');
title('Prior and posterior distributions of \theta_c');
xlabel('\theta_c');
yticks([]);
set(f1,'color','white');


% First, get prior and posterior theta1
f2 = figure('pos',[320 10 300 200]);
% Plot prior
fillunder([t2min t2min+t2range],[1/t2range 1/t2range],'g',1);
xlim([t2min t2min + t2range]);
hold on;
% Get kernel estimate of theta1 with true value marked
[pd2,xd2,bwd2] = ksdensity(dcto_t2);
[pk2,xk2,bwk2] = ksdensity(khct_t2);
plot1 = plot(xk2,pk2,'LineWidth',2,'color','r','LineStyle',':');
plot2 = plot(xd2,pd2,'LineWidth',2,'color','b');
plot1.Color(4)=lalpha;
plot2.Color(4)=lalpha;
% Plot true theta2
theta2 = results{discrep+1,1}.true_theta2;
plot([theta2 theta2],get(gca,'YLim'),'--','Color','k' ,...
    'LineWidth',1.5);
fillunder(xk2,pk2,'r',falpha);
fillunder(xd2,pd2,'b',falpha);
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

%% Show the functional relationship between theta1 and theta2
clc ; clearvars -except dpath ; close all ;

% Define inputs mins and ranges 
xmin = .5;
xrange = .5;
t1min = 1.5;
t1range = 3;
t2min = 0;
t2range = 5;

% Set discrepancy
discrep = 0;

% Set base theta1
low_theta1 = 1.5
high_theta1 = 2.25

% The function
% t1_fn = @(t2) t1min + t1range - t1range / t2range * (t2 - t2min) ;
% t1_fn = @(t2) t1min * 7/6 + (t1range - t1min/2) *...
%     exp(-2*(t2-(t2min+t2range*7/10)).^10) ;
t1_fn = @(x) low_theta1 + ...
    (high_theta1-low_theta1) * ...
    exp(40*(x-t2min)/t2range-20)./(1+exp(40*(x-t2min)/t2range-20));
% t1_fn = @(x) base_theta1 - ...
%     2.25 *exp(40*(x-t2min)/t2range-20)./(1+exp(40*(x-t2min)/t2range-20));
% t1_fn = @(x) high_theta1 - ...
%     (high_theta1 - low_theta1) * ...
%     exp(40*(x-t2min)/t2range-20)./(1+exp(40*(x-t2min)/t2range-20));
% t1_fn = @(x) low_theta1 + ...
%     (high_theta1-low_theta1) * ...
%     exp(20*((x-t2min)/t2range).^2-10)./...
%     (1+exp(20*((x-t2min)/t2range).^2-10));
% t1_fn = @(x) low_theta1 + ...
%     (high_theta1-low_theta1) * ...
%     exp(80*((x-t2min)/t2range)-40)./...
%     (1+exp(80*((x-t2min)/t2range)-40));
t1_fn = @(x) low_theta1 + ...
    (high_theta1-low_theta1) * ...
    exp(40*((x-t2min)/t2range)-20)./...
    (1+exp(40*((x-t2min)/t2range)-20));
t1_fn = @(x) high_theta1 - ...
    (high_theta1-low_theta1) * ...
    exp(40*((x-t2min)/t2range)-20)./...
    (1+exp(40*((x-t2min)/t2range)-20));
% t1_fn=@(x)2.5 * ones(size(x))

% Let's take a look at the objective function values for set x, using true
% t1 function as well as just base theta1
x = 1;
t2 = linspace(t2min,t2min+t2range,10000)';
t1 = t1_fn(t2);
y = dual_calib_example_fn(x,xmin,xrange,t1,t1min,t1range,...
    t2,t2min,t2range,0,1,discrep,false);
y_wrong = dual_calib_example_fn(x,xmin,xrange,low_theta1,t1min,t1range,...
    t2,t2min,t2range,0,1,discrep,false);
y_wrong2= dual_calib_example_fn(x,xmin,xrange,high_theta1,t1min,t1range,...
    t2,t2min,t2range,0,1,discrep,false);
f=figure('Position',[10 10 300 200]);
% subplot(1,2,1);
plot(t2,t1,'LineWidth',2);
ylim([low_theta1-.25,high_theta1+.25]);
xlabel('t_d');ylabel('\theta_c(t_d)');
title('Dependence of \theta_c on t_d');

% subplot(1,2,2);
% plot(t2,y,'LineWidth',2);
% xlabel('t2');ylabel('y');
% hold on;
% plot(t2,y_wrong,'--','LineWidth',2);
% plot(t2,y_wrong2,':','LineWidth',2);
% [m,i] = min(y) ; t2opt = t2(i)
% t1opt = t1(i)

% Save it:
set(f,'Color','w');
savestr = ...
sprintf(['FIG_theta_1_dependence_on_t2']);
set(f,'PaperPositionMode','auto')
% print(f,savestr,'-depsc','-r600')

%% Examine six different discrepancy versions
clc ; clearvars -except dpath ; close all ;

f=figure('pos',[640 5 540 800]);
set(f,'color','white');

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

discrep_title_content = [ {1;'a = 1.5'}, {1;'a = 3.5'}, ...
    {2;'a = 0.15, b = 0.075'}, {2;'a = 0.65, b = 0.075'}, ...
    {3;'a = 0.055, b=0'}, {3;'a = 0.055, b = 0.1'} ] ;

%%% Loop through all discrepancies and plot each
for ii=1:6
    subplot(3,2,ii);
%     if mod(ii,2)==1 figure('pos',[10 + ii*20, 5, 540, 250],'color','w');end
%     subplot(1,2,mod(ii-1,2)+1);
    discrep = ii ; % Select which discrepancy
    Y = reshape(...
        dual_calib_example_fn(X(:),xmin,xrange,T1(:),t1min,t1range,...
        T2(:),t2min,t2range,0,1,0,true),length(x),length(t1),length(t2));
    Yd= reshape(...
        dual_calib_example_fn(X(:),xmin,xrange,T1(:),t1min,t1range,...
        T2(:),t2min,t2range,0,1,discrep,true),length(x),length(t1),length(t2));

    % Take a look
    xidx=50;
    xx=reshape(X(:,xidx,:),100,100);
    % Get value of x
    xval = xx(1)*xrange + xmin;
    tt1=reshape(T1(:,xidx,:),100,100);
    tt2=reshape(T2(:,xidx,:),100,100);
    Discrep = Yd-Y;
    ea=.25;
    surf(tt1*t1range+t1min,tt2*t2range+t2min,...
        reshape(Y(:,xidx,:),100,100),...
        'EdgeAlpha',ea);
    hold on;
    surf(tt1*t1range+t1min,tt2*t2range+t2min,...
        reshape(Yd(:,xidx,:),100,100),...
        'EdgeAlpha',ea);
    surf(tt1*t1range+t1min,tt2*t2range+t2min,...
        reshape(Discrep(:,xidx,:),100,100),...
        'EdgeAlpha',ea);
    zlim([0,1.33]);
    
    % Sort out title and labels
    dtc = discrep_title_content(:,ii);
    dtc_lab=dtc{1};
    title(sprintf('f_%d, %s',dtc{:}));
    xlabel('t_c');ylabel('t_d');
    zlabel(sprintf('f_%d(x,t_c,t_d)',dtc_lab));
    axis vis3d;
    view([-110.4000    6.5334]);
    
    % Save
    savestr = sprintf(['FIG_obj_fn_g',int2str(ceil(ii/2))]);
    if mod(ii,2)==0 export_fig(savestr,'-png','-m2'); end
end

%%% Fix sizing
f.Children(6).Position = [0.0 0.685 0.575 0.32];
f.Children(5).Position = [0.5 0.685 0.575 0.32];
f.Children(4).Position = [0.0 0.355 0.575 0.32];
f.Children(3).Position = [0.5 0.355 0.575 0.32];
f.Children(2).Position = [0.0 0.025 0.575 0.32];
f.Children(1).Position = [0.5 0.025 0.575 0.32];

% % Get a rotating gif
% viewpt = [-27.2667 10.4000];
% view(viewpt);
% % gif('FIG_dual_calib_obf_fn.gif','frame',gcf);
% nfms = 120;
% for ii = 1:nfms
%     viewpt = viewpt + [ 360/nfms 0 ];
%     view(viewpt);
%     pause(.01);
% %     gif
% end

% Save it:
% saveas(f,'FIG_six_discrepancies.png');
figstr = 'FIG_six_discrepancies';
set(f,'PaperPositionMode','auto')
% print(f,figstr,'-depsc','-r600')