% CDO paper figures

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
addpath([dpath,'Example\Ex_results']);

% Change dir
cd(dpath);

%% Toy sim Figure describing the problem
clc ; clearvars -except dpath ; close all; 
% Take a look at surfaces of example function output
theta1=linspace(0,3);
theta2=linspace(0,6);
% We need to convert these to a two-col matrix of all possible combinations
[ T1 , T2 ] = meshgrid(theta1,theta2); 
ths = cat(2,T1',T2');
Theta = reshape(ths,[],2);

% Set other parameters
c = repmat(2.5,size(Theta,1),1);

% Now get output values
opc = Ex_sim([c Theta]);

% Normalize the outputs
oscl_n = (opc(:,1)-min(opc(:,1))) / range(opc(:,1)) ;
perf_n = (opc(:,2)-min(opc(:,2))) / range(opc(:,2)) ;
cost_n = (opc(:,3)-min(opc(:,3))) / range(opc(:,3)) ;

% Now take a look at the surfaces
oscls = reshape(oscl_n,[],length(theta1));
perfs = reshape(perf_n,[],length(theta1));
costs = reshape(cost_n,[],length(theta1));


e1 = [  0   0   0 ] ;  % output 1 color
e2 = [.4 .4 .4 ] ;  % output 2 color
e3 = [.8 .8 .8 ] ;  % output 3 color
g1 = [ 0 0 .6 ] ; 
g2 = [ 1 .3 0 ] ;
g3 = [ .4 1 .8 ] ; 
f1 = 'r'; f2='g'; f3='b'; % for color version
ea = 1      ;  % edge alpha
fa = 0      ;  % face alpha


f=figure('pos',[10 10 325 200]);
sp1 = subplot(1,2,1);
sp2 = subplot(1,2,2);
set(sp1,'position',[.1 .225 .65 .825]);
set(sp2,'position',[.4 .225 .6 .825]);
set(f,'currentaxes',sp1);
surf(theta2,theta1,oscls,'FaceColor',g1,'EdgeColor',g1,...
    'EdgeAlpha',ea,'FaceAlpha',fa);
% axis vis3d;

hold on;
surf(theta2,theta1,perfs,'FaceColor',g2,'EdgeColor',g2,...
    'EdgeAlpha',ea,'FaceAlpha',fa); 
% axis vis3d;

surf(theta2,theta1,costs,'FaceColor',g3,'EdgeColor',g3,...
    'EdgeAlpha',ea,'FaceAlpha',fa); 
% axis vis3d;
xlabel('\theta_2'); ylabel('\theta_1'); zlabel('Outcomes');

h=gca;
h.View = [13.1667 11.3333] ; % Sets the perspective
set(h,'ztick',[]);

% title('Model outcomes on normalized scale');

% hh = legend('y_1','y_2','y_3','Orientation','horizontal',...
%     'Location','south');
% hh.Position = hh.Position + [ 0 -.115 0 0 ];
% set(gcf,'unit','inches');
% fig_size = get(gcf,'position');
set(f,'currentaxes',sp2);
s1=patch(nan*[0 1 1 0],[0 0 1 1],g1);
hold on;
s2=patch(nan*[1 2 2 1],[1 1 2 2],g2);
s3=patch(nan*[2 3 3 2],[2 2 3 3],g3);
set(sp2,'Visible','off');
% set(s1,'Visible','off');set(s2,'Visible','off');set(s3,'Visible','off');

hh = legend(sp2,'y_1','y_2','y_3','Orientation','vertical',...
    'Location','east');
% set(hh,'unit','inches');
% leg_size = get(hh,'position');
% fig_size(3) = fig_size(3) + leg_size(3);
% set(gcf,'position',fig_size)

% %%% Code to turn it into the version used on SCSC poster
% hh.Orientation = 'vertical';
% hh.Location = 'east';
% f.Position = [360.3333  197.6667  452.0000  314.6667];
% set(f,'Color',[251/255 244/255 245/255]);
% set(h,'Color',[251/255 244/255 245/255]);
% set(hh,'Color',[251/255 244/255 245/255]);

%saveas(f,'FIG_toy_sim_model_outputs.png');
set(f,'Color','w');
% export_fig FIG_toy_sim_model_outputs -eps -m3 -painters

%% Toy sim results using set discrep marginal precision for var vals
%%% COLOR VERSION

clc ; clearvars -except dpath ; close all ;

% Set true optimum: this is for desired observation [0 0 0] and for desired
% observations derived from that one
optim = [ 0.924924924924925   3.141141141141141 ] ;

% Now do the version with desired observation set to be one normalized unit
% away from the pareto front, and marginal var set accordingly
load([dpath,'Example\Ex_results\'...
    '2018-07-11_discrepancy_true_fn_set_lambda_delta_1'],...
    'results');

samps = results.samples_os(results.settings.burn_in:end,:);
des_obs = results.settings.desired_obs;
h3=calib_heatmap(des_obs,samps,0.9,0.9,10);
xlabel('\theta_1'); ylabel('\theta_2');
hold on;
p=plot(optim(1),optim(2),'ok','MarkerSize',7,'MarkerFaceColor','m',...
    'LineWidth',2);
title(['Posterior \theta samples: desired obs. '...
    '[0.71 0.71 17.92], \lambda_\delta = 1']);

% % Now record it all
% saveas(h1,'FIG_post_theta_heatmap_desobs0_lambdadelta256i.png');
% saveas(h2,'FIG_post_theta_heatmap_desobs0_lambdadelta64i.png');
% saveas(h3,'FIG_post_theta_heatmap_desobs0_lambdadelta1.png');

%%% Turn last one into version for SCSC poster:
title(['Posterior \theta samples']);
% h3.Children(1).Position = h3.Children(1).Position + [0 0.11 0 -0.08];
% h3.Children(2).Position = h3.Children(2).Position + [0.14 0 -0.1 0];
%set(h3,'Color','none');
posc = h3.Children(4).Position;
h3.Children(4).Position = posc + [-0.09 -0.07 0.09 0.07];
%export_fig FIG_post_theta_heatmap_desobs0_lambdadelta1 -png -m3 -painters;

%% Toy sim Results using set discrep marginal precision for var vals
%%% GRAYSCALE VERSION

clc ; clearvars -except dpath ; close all ;
hh=figure();

% Set desired observation
des_obs_os = [ 0 0 0];

% Set true optimum: this is for desired observation [0 0 0] and for desired
% observations derived from that one
optim = [ 0.924924924924925   3.141141141141141 ] ;

% Now do the version with desired observation set to be one normalized unit
% away from the pareto front, and marginal var set accordingly
load([dpath,'Example\Ex_results\'...
    '2018-07-11_discrepancy_true_fn_set_lambda_delta_1'],...
    'results');

samps = results.samples_os(results.settings.burn_in:end,:);
[theta1,theta2] = meshgrid(linspace(0,3,1000),linspace(0,6,1000));
% Load true samples;
load([dpath,'Example\Ex_results\'...
    '2018-05-29_true_ctheta-output'],...
    'ctheta_output');

%%% Put the outputs and the desired observation on the standardized scale
meanout = mean(results.settings.output_means');
sdout   = mean(results.settings.output_sds'  );
cost_std = (ctheta_output(:,6) - meanout(3))/...
    sdout(3);
defl_std = (ctheta_output(:,4) - meanout(1))/...
    sdout(1);
rotn_std = (ctheta_output(:,5) - meanout(2))/...
    sdout(2);
outputs_std = [defl_std rotn_std cost_std];
des_obs = (des_obs_os-meanout)./...
    sdout;

% Now get Euclidean norms of each standardized output
dists = sqrt ( sum ( (outputs_std-des_obs).^2 , 2 ) ) ;
redists = reshape(dists,1000,1000);

%%% Make scatterhist of posterior samples
colormap autumn
sc=scatterhist(samps(:,1),samps(:,2),'Marker','.','Color','b',...
    'Markersize',1); 
hold on;
title(['Posterior \theta samples: desired obs. '...
    '[0.71 0.71 17.92], \lambda_\delta = 1']);

%%% Now add contour plot 
%[C,h]= contour(theta1,theta2,redists,[1 2 3 4 5 ],'LineWidth',3);
[C,h]= contour(theta1,theta2,redists,[16 17 18 19 20 ],'LineWidth',3);
clabel(C,h,'fontsize',12);
xlabel('\theta_1'); ylabel('\theta_2');

%%% Add true optimum
p=plot(optim(1),optim(2),'ok','MarkerSize',7,'MarkerFaceColor','m',...
    'LineWidth',2);


% % Now record it all
% saveas(h1,'FIG_post_theta_heatmap_desobs0_lambdadelta256i.png');
% saveas(h2,'FIG_post_theta_heatmap_desobs0_lambdadelta64i.png');
% saveas(h3,'FIG_post_theta_heatmap_desobs0_lambdadelta1.png');

%%% Turn last one into version for SCSC poster:
title(['Posterior \theta samples']);
hh.Children(3).Position = hh.Children(3).Position +[-.075 -.075 .075 .075];
pause(0.5);
% hh.Children(1).Position = hh.Children(1).Position + [0 0 0 0.035];
% hh.Children(2).Position = hh.Children(2).Position + [0 0 0.035 0];
set(hh,'Color','w');
%export_fig FIG_post_theta_contour_desobs0_lambdadelta1 -png -m3 -painters;
% export_fig FIG_post_theta_contour_desobs0_lambdadelta1 ...
%     -m3 -eps -painters;

%% TS Heatmpar results, bad des_obs and good des_obs side by side comparson
clc ; clearvars -except dpath ; close all ;

% Load results for bad calibration
load([dpath,'Example\Ex_results\'...
    '2018-12-20_preliminary_cdo_truefn_discrep_do0_ldgam10p1'],...
    'results');
br = results ;

% Load results for good calibration
load([dpath,'Example\Ex_results\'...
    '2018-07-11_discrepancy_true_fn_set_lambda_delta_1'],...
    'results');
gr = results;
clear results;

% Set true optimum: this is for desired observation [0 0 0] and for desired
% observations derived from that one
optim = [ 0.924924924924925   3.141141141141141 ] ;

% Load true samples;
load([dpath,'Example\Ex_results\'...
    '2018-05-29_true_ctheta-output'],...
    'ctheta_output');
meanout = mean(br.settings.output_means');
sdout   = mean(br.settings.output_sds'  );
cost_std = (ctheta_output(:,6) - meanout(3))/...
    sdout(3);
defl_std = (ctheta_output(:,4) - meanout(1))/...
    sdout(1);
rotn_std = (ctheta_output(:,5) - meanout(2))/...
    sdout(2);
outputs_std = [defl_std rotn_std cost_std];

des_obs_os = [ 0 0 0];
des_obs = (des_obs_os-meanout)./...
    sdout;

% Now get Euclidean norms of each standardized output from origin (orig sc)
origin_os = [ 0 0 0];
origin = (origin_os-meanout)./...
    sdout;
dists = sqrt ( sum ( (outputs_std-origin).^2 , 2 ) ) ;
redists = reshape(dists,1000,1000);

% Make grid for plotting
[theta1,theta2] = meshgrid(linspace(0,3,1000),linspace(0,6,1000));

%%%%%%%
%%% Make first plot, of bad calibration
h1=figure();
samps = br.samples_os(br.settings.burn_in:end,:);

% Make scatterhist of posterior samples
sc=scatterhist(samps(:,1),samps(:,2),'Marker','.','Color','b',...
    'Markersize',1); 
hold on;
ttl1=title(sprintf(['Posterior \\theta samples:\ntarget '...
    '[0 0 0], \\lambda_\\delta\\simGam(10,10)']));

% Now add contour plot 
%[C,h]= contour(theta1,theta2,redists,[1 2 3 4 5 ],'LineWidth',3);
[C,cntr1]= contour(theta1,theta2,redists,[16 17 18 19 20 ],'LineWidth',3);
clabel(C,cntr1,'fontsize',12);
xlab = xlabel('\theta_1'); ylabel('\theta_2');
xlab.Position = xlab.Position + [0 .2 0];

% Add true optimum
p=plot(optim(1),optim(2),'ok','MarkerSize',7,'MarkerFaceColor','m',...
    'LineWidth',2);

%%%%%%%
%%% Make second plot, of good calibration
h2=figure();
subplot(1,2,2);
samps = gr.samples_os(gr.settings.burn_in:end,:);

% Make scatterhist of posterior samples
sc=scatterhist(samps(:,1),samps(:,2),'Marker','.','Color','b',...
    'Markersize',1); 
hold on;
ttl2=title(sprintf(['Posterior \\theta samples:\ntarget '...
    '[0.71 0.71 17.92], \\lambda_\\delta=1']));

% Now add contour plot 
[C,cntr2]= contour(theta1,theta2,redists,[16 17 18 19 20 ],'LineWidth',3);
clabel(C,cntr2,'fontsize',12);
xlab= xlabel('\theta_1'); ylabel('\theta_2');
xlab.Position = xlab.Position + [0 .2 0];

% Add true optimum
p=plot(optim(1),optim(2),'ok','MarkerSize',7,'MarkerFaceColor','m',...
    'LineWidth',2);

% create third figure split into two uipanels
% h3 = figure('pos',[10 10 780 260]);
% u1 = uipanel('position',[0,0,0.5,1]);
% u2 = uipanel('position',[0.5,0,0.5,1]);
% 
% % get all children from each figure and move to the uipanels
% set(get(h1,'Children'),'parent',u1);
% set(get(h2,'Children'),'parent',u2);
% 
% % Set colormap
% colormap(cntr1.Parent,'autumn');
% colormap(cntr2.Parent,'autumn');
% 
% % Make room for titles
% spos1=ttl1.Parent.Position;
% spos2=ttl2.Parent.Position;
% ttl1.Parent.Position = spos1 + [ 0 0 0 -.05 ];
% ttl2.Parent.Position = spos2 + [ 0 0 0 -.05 ];
% % ttl2.Position = pos2 + [-.7 0 0];
% 
% % Close unneeded figures
% close(h1,h2);
% 
% % Move things a bit
% movedist = -.0869/2;
% u1.Children(1).Position = u1.Children(1).Position + [0 movedist 0 0];
% u2.Children(1).Position = u2.Children(1).Position + [0 movedist 0 0];
% u1.Children(3).Position = u1.Children(3).Position + [0 0 0 movedist];
% u2.Children(3).Position = u2.Children(3).Position + [0 0 0 movedist];
% 
% % Save figure
% set(u1,'BackgroundColor','white');
% set(u2,'BackgroundColor','white');
% set(u1,'ShadowColor','w');
% set(u2,'ShadowColor','w');
% figstr = sprintf('FIG_preliminary_CDO_comparison');
% export_fig(figstr,'-eps','-q0','-painters',h3);

%% WTA estimate of pareto front, with resulting choice of des_obs

clc ; clearvars -except dpath ; close all ;

%%% Load the preliminary CDO
load([dpath,'stored_data\'...
    '2018-07-25_discrepancy_d0'],...
    'results');
n=700;
burn_in = results.settings.burn_in;
eouts = results.model_output.by_sample_est(burn_in:burn_in+n,:);

%%% Estimate the PF
[PF_os, PFidx] = nondominated(eouts) ; 
eouts=setdiff(eouts,PF_os,'rows'); % Remove the Pareto front from eouts so 
                   % they can be plotted with different marker
PFidx = PFidx + results.settings.burn_in; % get idx in original full set

%%% Put PF on standardized scale
omeans = mean(results.settings.output_means');
osds   = mean(results.settings.output_sds'  );
PF     = (PF_os - omeans)./ osds             ;

%%% Find closet point to des_obs
%orig_des_obs = results.settings.desired_obs  ;
orig_des_obs = [.74 .089 100 ] ; % This pt chosen to get at observed elbow
des_obs = (orig_des_obs - omeans)./osds      ;
[m,i] = min( sum( ( PF - des_obs ).^2, 2 ) ) ;
PF_optim = PF(i,:)                           ;

%%% Get new desired obs specified distance from PF in same dir as original
spec_dist = .2                               ;
dirvec_nonnormed = PF_optim - des_obs        ;
dirvec = dirvec_nonnormed/norm(dirvec_nonnormed) ;
des_obs_new = PF_optim - spec_dist * dirvec  ;
des_obs_new_os = des_obs_new .* osds + omeans; 

%%% Take a look
h=figure('pos',[10 10 400 300]);
sc=scatter3(eouts(:,1),eouts(:,2),eouts(:,3),90,'g',...
    'MarkerEdgeAlpha',.5,'MarkerFaceAlpha',1,...
    'MarkerFaceColor','g','Marker','x','LineWidth',2);
hold on;
scatter3(PF_os(:,1),PF_os(:,2),PF_os(:,3),'b','MarkerFaceColor','b',...
    'MarkerEdgeAlpha',1,'MarkerFaceAlpha',1)   ;
% scatter3(orig_des_obs(1),orig_des_obs(2),orig_des_obs(3))          ;
% line([orig_des_obs(1) PF_os(i,1)], [orig_des_obs(2) PF_os(i,2)], ...
%     [orig_des_obs(3) PF_os(i,3)])                                  ;
scatter3(des_obs_new_os(1),des_obs_new_os(2),des_obs_new_os(3),90,'r',...
    'MarkerFaceColor','r','Marker','+','LineWidth',2);
ylim([0.075 0.1])
h.CurrentAxes.View = [55.8000    9.4666] ; %[-3.9333   10.5333] ; 
% [-5.0000    5.2000];% [ 63 10] ;%[-8.4333 17.7333] ; 
title('Estimated Pareto front with target outcome');
xlabel('Deflection');ylabel('Rotation');zlabel('Cost');
set(h,'Color','w');
% export_fig 'FIG_est_PF_with_des_obs' -eps -m3 -painters
% saveas(h,'FIG_est_PF_with_des_obs.png');

%% WTA Pareto bands
clc ; clearvars -except dpath ; close all ;

%%% Load the results
load([dpath,'stored_data\'...
    '2018-07-27_discrepancy_d-elbow_d-p2'],...
    'results');
samps = results.samples_os(results.settings.burn_in+2:end,:) ;

%%% Get the marginal plots
h1 = figure('rend','painters','pos',[10 10 720 190]) ; 
subplot(1,2,1);
histogram(samps(:,1), 'Normalization','pdf','EdgeColor','none') ;
xlim([0.2 0.6]);
unifval = 1/.4;
hold on;
plot([0.2 0.6], [unifval unifval],'--k','LineWidth',2);
title('Volume fraction');

subplot(1,2,2);
histogram(samps(:,2), 'Normalization','pdf','EdgeColor','none') ;
xlim([10 25]);
unifval = 1/15;
hold on;
plot([10 25], [unifval unifval],'--k','LineWidth',2);
title('Thickness (mm)');

%%% Save
set(h1,'Color','white');
%export_fig FIG_posterior_marginals_with_priors -png -m3;

clc ; clearvars -except dpath ; close all ; 

%%% Load the cost_grid results
load([dpath,'stored_data\'...
    '2018-08-03_cost_grid_discrepancy_results'],...
    'results');

% Collect Cost_lambdas, and posterior mean and sds for costs, defl, rot, as
% well as upper and lower .05 quantiles
m=size(results,1); % Store number of target cost_lambdas
cost_lambda = zeros(m,1); % This will store cost_lambdas
cred_level = 90; % Set desired level for credible bands (in %)
alpha = (100-cred_level)/100; % Convert cred_level to alpha level
pmo = zeros(m,3); % This will store posterior mean output of emulator
pdo = zeros(m,3); % ``'' median output
pso = zeros(m,3); % ``'' appropriate multiple of standard deviations
plo = zeros(m,3); % ``'' lower (alpha/2) quantile
puo = zeros(m,3); % ``'' upper (alpha/2) quantile
for ii = 1:m % This loop populates the above arrays
    pmo(ii,:) = results{ii}.post_mean_out;
    pdo(ii,:) = quantile(results{ii}.model_output.by_sample_est,0.5);
    pso(ii,:) = norminv(1-alpha/2) * ...
        mean(results{ii}.model_output.by_sample_sds);
    plo(ii,:) = quantile(results{ii}.model_output.by_sample_est,alpha/2);
    puo(ii,:) = quantile(results{ii}.model_output.by_sample_est,1-alpha/2);
    cost(ii) = results{ii}.desired_obs(3);
end
% Now we break the arrays up each into 3 vectors, one for each output
post_cost_mean = pmo(:,3);
post_defl_mean = pmo(:,1);
post_rotn_mean = pmo(:,2);
post_cost_median = pdo(:,3);
post_defl_median = pdo(:,1);
post_rotn_median = pdo(:,2);
post_cost_sd = pso(:,3);
post_defl_sd = pso(:,1);
post_rotn_sd = pso(:,2);
post_cost_lq = plo(:,3);
post_cost_uq = puo(:,3);
post_defl_lq = plo(:,1);
post_defl_uq = puo(:,1);
post_rotn_lq = plo(:,2);
post_rotn_uq = puo(:,2);
% Get quantiles plus code uncertainty
post_cost_uq_cu = post_cost_uq + post_cost_sd;
post_cost_lq_cu = post_cost_lq - post_cost_sd;
post_defl_uq_cu = post_defl_uq + post_defl_sd;
post_defl_lq_cu = post_defl_lq - post_defl_sd;
post_rotn_uq_cu = post_rotn_uq + post_rotn_sd;
post_rotn_lq_cu = post_rotn_lq - post_rotn_sd;
% Get ylims for the two sets of plots
ylimrat=1.01;
ylim_cost = [min(post_cost_lq_cu)/ylimrat max(post_cost_uq_cu)*ylimrat];
ylim_defl = [min(post_defl_lq_cu)/ylimrat max(post_defl_uq_cu)*ylimrat];
ylim_rotn = [min(post_rotn_lq_cu)/ylimrat max(post_rotn_uq_cu)*ylimrat];

%%% Begin figures
% Set alphas for two types of uncertainty
alpha_wcu = 0.5;  %with code uncertainty
alpha_wocu= 0.15; %without
h=figure('rend','painters','pos',[10 10 640 320]);
x = 96:1:350; % x fills the cost domain

% Now begin plot 2/2
subplot(1,2,2)
% Get main curve
pdefl = pchip(cost,post_defl_mean,x);
% Get upper and lower 0.05 quantiles curves
pdefluq = pchip(cost,post_defl_uq,x);
pdefllq = pchip(cost,post_defl_lq,x);
f=fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(f,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
plot(x,pdefl,'-r','LineWidth',1.5); % Mean
% plot(x,pdefluq,':k',...
%      x,pdefllq,':k');
xl2=xlabel('Target cost');
ylabel('Deflection');
xlim([96,350]);
ylim(ylim_defl);


% Here's plot 1/2
subplot(1,2,1)
% Get main curve
pcost = pchip(cost,post_cost_mean,x);
% Get upper and lower 0.05 quantiles curves
pcostuq = pchip(cost,post_cost_uq,x);
pcostlq = pchip(cost,post_cost_lq,x);
go_fill_unc=fill([ x , fliplr(x) ], [pcostuq, fliplr(pcostlq)],'k');
set(go_fill_unc,'facealpha',alpha_wocu,'edgealpha',alpha_wocu);
hold on;
go_plot_mean=plot(x,pcost,'-r','LineWidth',1.5);
% plot(...%cost,post_cost_mean,'or',...%x,pcost,'-r',...
%     x,pcostuq,':k',...
%     x,pcostlq,':k');
% Plot 2sd errbar
% errorbar(cost_lambda,post_cost_mean,post_cost_sd,'ob'); 
xl1=xlabel('Target cost');
ylabel('Observed cost');
xlim([96,350]);
ylim(ylim_cost);
%plot(x,x,'-k','LineWidth',2);

% % Save the figure temporarily so we can mess with it later, because
% % suptitle seems to mess things up somhow for making change after calling
% % it
% savefig(h,'tempfig');
% 
% % Now add a main title and fix any infelicities
% suptitle(['Deflection vs. (known) target cost,',...
%     ' with ',num2str(cred_level),'% credible interval']); 
% p = get(xl1,'position');
% set(xl1,'position',p + [0 2.75 0]);
% p = get(xl2,'position');
% set(xl2,'position',p + [0 0.00125 0])
% % p = get(xl3,'position');
% % set(xl3,'position',p + [0 0.0002 0])
figpos = get(h,'pos');

% saveas(h,'FIG_cost_grid_pareto.png');

% Now add in code uncertainty. That is, the above assumes that the GP
% emulator nails the FE code precisely. But of course the GP emulator has
% nonnegligible variance. That's the code uncertainty. So our confidence
% bands should reflect it. So we add it in here, by dropping the
% appropriate multiple of the sd from each lower quantile and adding it to
% each upper quantile.
% First, open the figure prior to calling suptitle.
% h=openfig('tempfig');
subplot(1,2,2);
pdefluq_code_uncert = pchip(cost,post_defl_uq_cu,x);
pdefllq_code_uncert = pchip(cost,post_defl_lq_cu,x);
f=fill([ x , fliplr(x) ], [pdefluq_code_uncert,...
    fliplr(pdefluq)],'k');
ff=fill([ x , fliplr(x) ], [pdefllq_code_uncert,...
    fliplr(pdefllq)],'k');
set(f,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
set(ff,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
xl2=xlabel('Target cost');
ylim(ylim_defl);

subplot(1,2,1);
pcostuq_code_uncert = pchip(cost ,post_cost_uq_cu,x);
pcostlq_code_uncert = pchip(cost ,post_cost_lq_cu,x);
go_fill_cunc_up=fill([ x , fliplr(x) ], [pcostuq_code_uncert,...
    fliplr(pcostuq)],'k');
go_fill_cunc_dn=fill([ x , fliplr(x) ], [pcostlq_code_uncert,...
    fliplr(pcostlq)],'k');
set(go_fill_cunc_up,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
set(go_fill_cunc_dn,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
xl1=xlabel('Target cost');
ylim(ylim_cost);
go_plot_diag=plot(ylim_cost,ylim_cost,'--b','LineWidth',2);

% Now add a main title and fix any infelicities
% suptitle(['Posterior estimate vs. target cost,',...
%     ' with ',num2str(cred_level),'% credible interval ']); 
set(h,'pos',figpos); % Just so we can reuse the positioning code from above
p = get(xl1,'position');
% set(xl1,'position',p + [0 2.75 0]);
p = get(xl2,'position');
% set(xl2,'position',p + [0 0.00125 0])
% p = get(xl3,'position');
% set(xl3,'position',p + [0 0.0002 0])

% Now add a legend.
leg_gos = [go_plot_mean go_fill_unc go_fill_cunc_up go_plot_diag];
lg=legend(leg_gos,'Posterior predictive mean',...
    'C.I. w/o code uncertainty',...
    sprintf('C.I. with code uncertainty'),...
    'Diagonal for reference',...
    'Location','northwest');
lg.Position(1:2)=[.623 .725];
% flushLegend(lg,'northeast');

    
%%% Save
set(h,'Color','white');
% export_fig FIG_cost_grid_pareto_bands -png -m3 -painters
% saveas(h,'FIG_cost_grid_pareto_with_code_uncert.png');

%% WTA prior predictive distribution vs posterior predictive distribution
clc ; clearvars -except dpath ; close all ;

%%% Load prior predictive results
load([dpath,'stored_data\'...
    '2019-11-06_prior_predictive_distributions']);
prsamps = prior_model_output.means;
clear prior_pred_dist;

%%% Load calib results
load([dpath,'stored_data\'...
    '2018-07-27_discrepancy_d-elbow_d-p2'],...
    'results');
posamps = results.model_output.by_sample_est;
des_obs = results.settings.desired_obs;

%%% Rescale prior samps
outsds = mean(results.settings.output_sds,2);
outmeans = mean(results.settings.output_means,2);
prsamps = prsamps .* outsds' + outmeans';

%%% Make figure using histograms
f=figure('pos',[10 10  780.0000  200]);
% Deflection
subplot(1,3,1);
[p,x]=ksdensity(posamps(:,1));
plot(x,p,'LineWidth',2);
%histogram(posamps(:,1),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,1));
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,1),'Normalization','pdf','Edgecolor','none');
text(0.05,.9,...
    'Deflection','VerticalAlignment','bottom','Units','normalized');
text(1.715,102,'Rotation','VerticalAlignment','bottom');
xlim([0.6 0.85]);
ylim([0 110]);
line([des_obs(1) des_obs(1)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
% Rotation
subplot(1,3,2);
[p,x]=ksdensity(posamps(:,2));
plot(x,p,'LineWidth',2);
%histogram(posamps(:,2),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,2));
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,2),'Normalization','pdf','Edgecolor','none');
text(0.05,.9,...
    'Rotation','VerticalAlignment','bottom','Units','normalized');
xlim([0.075,0.105])
ylim([0 700]);
line([des_obs(2) des_obs(2)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
% Cost
subplot(1,3,3);
[p,x]=ksdensity(posamps(:,3));
plot(x,p,'LineWidth',2);
%histogram(posamps(:,3),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,3));
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,3),'Normalization','pdf','Edgecolor','none');
text(0.05,.9,...
    'Cost','VerticalAlignment','bottom','Units','normalized');
ylim([0 .0700]);
line([des_obs(3) des_obs(3)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
% Add suptitle
% st=suptitle('Prior and posterior predictive distributions');
% st.Position=[0.5 -.1 0];
lg=legend('Posterior','Prior','Target','Location','northeast');
pos=lg.Position; lg.Position = pos + [0.007 0.035 0 0];
% flushLegend(lg,'northeast');

%%% Save
set(f, 'Color','white');
% export_fig FIG_prior_vs_posterior_dist -eps -m3 -painters

%% WTA Posterior scatterhist from calibration
clc ; clearvars -except dpath ; close all ;

%%% Load the calibration results
load([dpath,'stored_data\'...
    '2018-07-27_discrepancy_d-elbow_d-p2'],...
    'results');
samps = results.samples_os(results.settings.burn_in+2:end,:) ;

h=figure('pos',[10 10 410 310]);
sc=scatterhist(samps(:,1),samps(:,2),'Marker','.','MarkerSize',3);
xlim([0.2 0.6]); ylim([10 25]);
xlabel('Volume fraction');ylabel('Thickness');
title('Posterior distribution on \theta');
set(h,'Color','w');

% saveas(h,'FIG_post_dist_scatterhist.png');
% export_fig 'FIG_post_dist_scatterhist' -png -m3;

%% WTA Contour plot of highest density regions of posterior distribution 
clc ; clearvars -except dpath ; close all ;

%%% Load the calibration results
load([dpath,'stored_data\'...
    '2018-07-27_discrepancy_d-elbow_d-p2'],...
    'results');
samps = results.samples_os(results.settings.burn_in+2:end,:) ;

ff=figure('pos',[10 10  320.0000  240]);

% Get the density
[f,x,bw]=ksdensity(samps);

% Correct it by enforcing the boundary conditions:
f(x(:,1)>0.6 | x(:,2)<10) = 0 ;

% Reshape for the contour plot
X = reshape(x(:,1),30,30); Y = reshape(x(:,2),30,30); Z = reshape(f,30,30);

% Convert Z to quantiles in posterior dist
[f,x,bw]=ksdensity(samps,samps);
Z = reshape(sum(f<Z(:)')/size(f,1),30,30);

% Get contour plot and labels
colormap HSV
[C9,h9] = contour(X,Y,Z, [ .9 .9 ]); hold on;
%[C,h] = contour(X,Y,Z, [ .75 .75 ], 'k--');
[C5,h5] = contour(X,Y,Z, [ .5 .5 ], '--');
%[C,h] = contour(X,Y,Z, [ .25 .25 ], 'k-.');
[C1,h1] = contour(X,Y,Z, [ .1 .1 ], '-.');
[C01,h01] = contour(X,Y,Z, [ .01 .01 ], ':');
w=1.25; h9.LineWidth=w; h5.LineWidth=w; h1.LineWidth=w; h01.LineWidth=w;
xlim([0,0.6]);ylim([10,25]);

lg=legend(...
    '0.1 HDR','0.5 HDR', '0.9 HDR','0.99 HDR','Location','northwest');
title('Posterior distribution of \theta');
xlabel('Volume fraction'); ylabel('Thickness (mm)');

% Save
set(ff,'Color','w');
% export_fig 'FIG_post_dist_contourplot' -eps -m3;

%% Toy case marginal posteriors with true optimum

clc ; clearvars -except dpath ; close all ;

% Load results
load([dpath,'Example\Ex_results\'...
    '2018-07-11_discrepancy_true_fn_set_lambda_delta_1'],...
    'results');
samps = results.samples_os(results.settings.burn_in+2:end,:);

% Make figures
h = figure('rend','painters','pos',[10 10 610 220]) ;

% Load true optimum
load([dpath,'Example\Ex_results\'...
    '2018-09-11_true_optimum_wrt_0']);

%%% Get the marginal plots
subplot(1,2,1);
histogram(samps(:,1), 'Normalization','pdf') ;
xlim([0 3]);
unifval = 1/3;
hold on;
%xlabel('\theta_1');
ylims=[0 3.3];%ylim;
plot([0 3], [unifval unifval],'--r','LineWidth',1);
plot([optim(1) optim(1)], [ylims],':g','LineWidth',1.5);
ylim(ylims);
xlabel('\theta_1');


subplot(1,2,2);
histogram(samps(:,2), 'Normalization','pdf') ;
xlim([0 6]);
unifval = 1/6;
hold on;
%xlabel('\theta_2');
ylims=[0 12];%ylim;
plot([0 6], [unifval unifval],'--r','LineWidth',1);
plot([optim(2) optim(2)], [ylims],':g','LineWidth',1.5);
ylim(ylims);
xlabel('\theta_2');

suptitle('Marginal posterior distributions with priors and optima');

%     %%% Save
set(h,'Color','white');
figstr = sprintf('FIG_iter_post_marginals');
% export_fig(figstr,'-eps','-m3','-painters',h);

%% Example of selecting performance target, v1
clc ; clearvars -except dpath ; close all ;
fig=figure();

% Define Pareto front
x = linspace(1,2);
xt = linspace(1,2,10);
ytl = [1 .6 .4 .15 .09 .0675 .0525 .04 .025 0.015];
yl = spline(xt,ytl,x);

% Define upper model bound
ytu = [1 1.1 1.12 1.14 1.1 .9 .7 .5 .4 .015];
yu = spline(xt,ytu,fliplr(x));

% Plot the model range
p1 = fill([x fliplr(x)],[yl yu],[1 .175 .175]);
axis equal;
%hold on; plot(xt,ytl,'o'); plot(xt,ytu,'o');
xlim([0,2.1]);ylim([0,1.2]);

% Get closest point to [0,0], and connect it to [0,0]
dists = sqrt(sum([x(:) yl(:) ].^2,2)) ;
[mnm,idx]=min(dists);
hold on; 
p2 = plot(x(idx),yl(idx),'ok','LineWidth',2,'MarkerSize',5);
p3 = plot(0,0,'xk','LineWidth',2,'MarkerSize',7);
p4 = plot([0 x(idx)],[0 yl(idx)],'-k','LineWidth',1);

% Also plot elbow and nearby desired observation
lbloc = 37 ;
plot(x(lbloc),yl(lbloc),'ok','LineWidth',2,'MarkerSize',5);
des_obs = [1.32 0.065];
plot(des_obs(1),des_obs(2),'xk','LineWidth',2,'MarkerSize',7);
plot([des_obs(1) x(lbloc)],[des_obs(2) yl(lbloc)],'-k','LineWidth',1);
dodist = norm([des_obs(1)-x(lbloc), des_obs(2)-yl(lbloc)]);

% Get max distance from [0 0]
dists_up = sqrt(sum([x(:) yu(:)].^2,2)) ;
[mxm,idx] = max(dists_up);

% Add partial circle showing region within 1.59 * dist from des_obs
dist_ratio = mxm/mnm ; 
x_idx = abs(x-des_obs(1))<=dist_ratio*dodist ;
xr = x(x_idx) ; 
yr = sin(acos((xr-des_obs(1))/dist_ratio/dodist)) ; 
yrt=yr*dist_ratio*dodist+des_obs(2); % Similarly for y-vals
in_mod_range_idx = yl(x_idx) <= yrt;
xrr = xr( in_mod_range_idx );
yrr = yrt( in_mod_range_idx );
p5 = plot(xrr,yrr,':k','LineWidth',2);

% Add labels and legend
ylabel('y_2'); xlabel('y_1');
lg = legend([p1 p3 p2 p5],{'Model range','Possible target outcome',...
    'Nearest point to target','1.78\timesdiscrepancy distance'},...
    'Location','Northwest');
pos = [0.1300    0.6539    0.3661    0.1591];
lg.Position = pos ;

% Save
set(fig,'color','white');
figstr = sprintf('FIG_des_obs_selection_example');
% export_fig(figstr,'-eps','-m3','-painters',fig);

%% Example of selecting performance target, v2
clc ; clearvars -except dpath ; close all ;
fig=figure();

% Define Pareto front
x = linspace(1.03,2.39);
xt = linspace(1.03,2.39,10);
ytl = [1 .6 .4 .15 .09 .0675 .0525 .04 .025 0.015];
yl = spline(xt,ytl,x);

% Define upper model bound
ytu = [1 1.1 1.12 1.14 1.1 .9 .7 .5 .4 .015];
yu = spline(xt,ytu,fliplr(x));

% Plot the model range
p1 = fill([x fliplr(x)],[yl yu],[1 .175 .175]);
% axis equal;
%hold on; plot(xt,ytl,'o'); plot(xt,ytu,'o');
xlim([0,2.4]);ylim([0,1.2]);

% Get closest point to [0,0], and connect it to [0,0]
dists = sqrt(sum([x(:) yl(:) ].^2,2)) ;
[mnm,idx]=min(dists);
fdists = sqrt(sum([x(:) yu(:) ].^2,2));
hold on; 
p2 = plot(x(idx),yl(idx),'ok','LineWidth',2,'MarkerSize',5);
p3 = plot(0,0,'xk','LineWidth',2,'MarkerSize',7);
p4 = plot([0 x(idx)],[0 yl(idx)],'-k','LineWidth',1);

% Also plot elbow and nearby desired observation
lbloc = 37 ; % elbow location index
plot(x(lbloc),yl(lbloc),'ok','LineWidth',2,'MarkerSize',5);
des_obs = [1.4825 0.065];
plot(des_obs(1),des_obs(2),'xk','LineWidth',2,'MarkerSize',7);
plot([des_obs(1) x(lbloc)],[des_obs(2) yl(lbloc)],'-k','LineWidth',1);
dodist = norm([des_obs(1)-x(lbloc), des_obs(2)-yl(lbloc)]);

% Get max distance from [0 0]
dists_up = sqrt(sum([x(:) yu(:)].^2,2)) ;
[mxm,idx] = max(dists_up);

% Add partial circle showing region within 2 * dist from des_obs
dist_ratio = mxm/mnm ; 
x_idx = abs(x-des_obs(1))<=dist_ratio*dodist ;
xr = x(x_idx) ; 
yr = sin(acos((xr-des_obs(1))/dist_ratio/dodist)) ; 
yrt=yr*dist_ratio*dodist+des_obs(2); % Similarly for y-vals
in_mod_range_idx = yl(x_idx) <= yrt;
xrr = xr( in_mod_range_idx );
yrr = yrt( in_mod_range_idx );
p5 = plot(xrr,yrr,':k','LineWidth',2);

% Resize
fig.Position = fig.Position + [ 0 0 0 -175];


% Add labels and legend
ylabel('y_2'); xlb = xlabel('y_1');
lg = legend([p1 p3 p2 p5],{'Feasible design space',...
    'Possible target outcome',...
    'Nearest point to target','2\timesdiscrepancy distance'},...
    'Location','Northwest');
flushLegend(lg,'northwest');
pos = lg.Position;
% lg.Position = 
% ax=gca; ax.Position = ax.Position + [0 0.01 0 0];
% xlb.Position = xlb.Position + [0 .08 0];


% Save
set(fig,'color','white');
figstr = sprintf('FIG_des_obs_selection_example2');
% export_fig(figstr,'-eps','-q0','-painters',fig);

set(fig,'PaperPositionMode','auto')
% print(fig,figstr,'-depsc','-r600')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figures for revised version (post-Technometrics) %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TSA Results (output on contour plot) with PCTO-chosen target
clc ; clearvars -except dpath ; close all ;

% Initialize figure
figposition = [10 10 390 260];
fig1 = figure('Position',figposition);
ttlfontsize = 11;

% Load results
locstr = [dpath,'Example\Ex_results\'...
    '2019-11-04_CTO_noemulator_varest_afterPCTO'];
load(locstr);

% Set desired observation as the utopia point of the system
des_obs_os = [0.7311 0.6675 15];
des_obs = (des_obs_os - res.settings.mean_y)./res.settings.std_y;

% Set true optimum: this is for desired observation set as the utopia point
% of the system, and for desired observations derived from that one.
optim = [ 0.669808002211120   3.058374201339808 ] ;

burn_in = res.settings.burn_in;
samps = [res.theta1(burn_in:end,1) res.theta1(burn_in:end,2)];
[theta1,theta2] = meshgrid(linspace(0,3,1000),linspace(0,6,1000));
% Load true samples;
load([dpath,'Example\Ex_results\'...
    '2018-05-29_true_ctheta-output'],...
    'ctheta_output');

%%% Put the outputs and the desired observation on the standardized scale
mean_y = res.settings.mean_y;
std_y   = res.settings.std_y;
outputs_std = (ctheta_output(:,4:6) - mean_y)./std_y;

% Now get Euclidean norms of each standardized output
dists = sqrt ( sum ( (outputs_std-des_obs).^2 , 2 ) ) ;
redists = reshape(dists,1000,1000);

%%% Make scatterhist of posterior samples
colormap autumn
subplot(1,2,1);
sc1=scatterhist(samps(:,1),samps(:,2),'Marker','.','Color','b',...
    'Markersize',1,'Kernel','on'); 
% sc(2).Children.EdgeColor = 'none';
% sc(2).Children.FaceAlpha = 1;
% sc(3).Children.EdgeColor = 'none';
% sc(3).Children.FaceAlpha = 1;
hold on;
ttl1 = title({'Posterior \theta samples:' ...
    'target [0.75 0.73 17.56]'});
ttl1.FontSize = ttlfontsize;

%%% Now add contour plot 
[C1,h1]= contour(theta1,theta2,redists,[2 3 4 5 6],'LineWidth',3);
% [C,h]= contour(theta1,theta2,redists,[16 17 18 19 20 ],'LineWidth',3);
clabel(C1,h1,'fontsize',12);
xlabel('\theta_1'); ylabel('\theta_2');
xlim([0 3]) ; ylim([0 6]);

%%% Add true optimum
p=plot(optim(1),optim(2),'ok','MarkerSize',7,'MarkerFaceColor','m',...
    'LineWidth',2);




% Start second figure
fig2 = figure('Position',figposition);

% Load results
locstr = [dpath,'Example\Ex_results\'...
    '2019-11-06_CTO_noemulator_varest_noPCTO'];
load(locstr);

% Set desired observation as the utopia point of the system
des_obs_os = [0.7311 0.6675 15];
des_obs = (des_obs_os - res.settings.mean_y)./res.settings.std_y;

% Set true optimum: this is for desired observation set as the utopia point
% of the system, and for desired observations derived from that one.
optim = [ 0.669808002211120   3.058374201339808 ] ;

burn_in = res.settings.burn_in;
samps = [res.theta1(burn_in:end,1) res.theta1(burn_in:end,2)];
[theta1,theta2] = meshgrid(linspace(0,3,1000),linspace(0,6,1000));
% Load true samples;
load([dpath,'Example\Ex_results\'...
    '2018-05-29_true_ctheta-output'],...
    'ctheta_output');

%%% Put the outputs and the desired observation on the standardized scale
mean_y = res.settings.mean_y;
std_y   = res.settings.std_y;
outputs_std = (ctheta_output(:,4:6) - mean_y)./std_y;

% Now get Euclidean norms of each standardized output
dists = sqrt ( sum ( (outputs_std-des_obs).^2 , 2 ) ) ;
redists = reshape(dists,1000,1000);

%%% Make scatterhist of posterior samples
colormap autumn
sc=scatterhist(samps(:,1),samps(:,2),'Marker','.','Color','b',...
    'Markersize',1,'Kernel','on'); 
% sc(2).Children.EdgeColor = 'none';
% sc(2).Children.FaceAlpha = 1;
% sc(3).Children.EdgeColor = 'none';
% sc(3).Children.FaceAlpha = 1;
hold on;
ttl2 = title({'Posterior \theta samples:', ...
    'target [0.73 0.67 15]'});
ttl2.FontSize = ttlfontsize;

%%% Now add contour plot 
[C,h]= contour(theta1,theta2,redists,[2 3 4 5 6],'LineWidth',3);
% [C,h]= contour(theta1,theta2,redists,[16 17 18 19 20 ],'LineWidth',3);
clabel(C,h,'fontsize',12);
xlabel('\theta_1'); ylabel('\theta_2');
xlim([0 3]) ; ylim([0 6]);

%%% Add true optimum
p=plot(optim(1),optim(2),'ok','MarkerSize',7,'MarkerFaceColor','m',...
    'LineWidth',2);

% pause
pause(.5);

% Mess with color and position
fig1.Children(2).Children(1).Visible='off';
fig1.Children(3).Children(1).Visible='off';
fig2.Children(2).Children(1).Visible='off';
fig2.Children(3).Children(1).Visible='off';
set(fig1,'Color','white');
sppos1 = fig1.Children(4).Position;
fig1.Children(4).Position = sppos1 + [-0.025 -0.035 .025 0];
set(fig2,'Color','white');
sppos2 = fig2.Children(4).Position;
fig2.Children(4).Position = sppos2 + [-0.025 -0.035 .025 0];
pause(1); % Seems to be necessary
xpdfpos = fig1.Children(2).Position;
fig1.Children(2).Position = xpdfpos + [0 0 0 .07];
xpdfpos = fig2.Children(2).Position;
fig2.Children(2).Position = xpdfpos + [0 0 0 .07];


% Save figures as .eps
figstr1 = sprintf('FIG_TS_results_withPCTO');
figstr2 = sprintf('FIG_TS_results_noPCTO');
% export_fig(figstr1,'-eps','-q0','-painters',fig1);
% export_fig(figstr2,'-eps','-q0','-painters',fig2);


%% WTA Histogram2 of highest density regions of posterior distribution 
clc ; clearvars -except dpath ; close all ;

%%% Load the calibration results
locstr = [dpath,'stored_data\'...
    '2019-11-05_CTO'];
load(locstr);
samps = res.theta1(res.settings.burn_in:end,:) ;

fighist = figure('pos',[10 10  360.0000  240]);
restricted_samps = samps(samps(:,1) > .59994 & samps(:,2) < 10.023,:);
binwidth = 1.5e-3*[0.002 1];
h2=histogram2(restricted_samps(:,1),restricted_samps(:,2),...
    'BinWidth',binwidth,'Normalization','pdf');
set(gca,'View',[225 30]);
set(gca,'ZTick',[]);
grid off;
ttlhist = title('Posterior distribution of \theta');
ttlhist.FontSize = 11;
xlbl = xlabel('Vol. fraction'); ylbl = ylabel('Thickness (mm)');
set(fighist,'Color','w');
% pause(1.5);

% Save it
figstr = 'FIG_post_dist_hist2';
set(fighist,'PaperPositionMode','auto')
ylbl.Units = 'pixels'; xlbl.Units='pixels';
ylbl.Position = ylbl.Position + [-8.0 2 0];
xlbl.Position = xlbl.Position + [8.0 2 0];
% print(fighist,figstr,'-depsc','-r600')

%% WTA prior predictive distribution vs posterior predictive distribution
clc ; clearvars -except dpath ; close all ;

%%% Load calib results
% locstr = [dpath,'stored_data\'...
%     '2019-11-05_CTO'];
locstr = [dpath,'stored_data\'...
    '2020-04-25_CTO_size500'];
load(locstr);
mean_y = res.settings.mean_y ; std_y = res.settings.std_y;
posamps = res.model_output.means .* std_y + mean_y;
des_obs = res.settings.obs_y(1,:).* std_y + mean_y;

%%% Load prior predictive results
locstr2 = [dpath,'stored_data\'...
    '2019-11-06_prior_predictive_distributions'];
load(locstr2);
prsamps = prior_model_output.means.* std_y + mean_y;


%%% Make figure using histograms
f=figure('pos',[10 10  360.0000  200]);
[subplts,pos] = tight_subplot(1,3,0.02,[ 0.08 0.01],0.03);
% Deflection
axes(subplts(1));
[p,x,bw]=ksdensity(posamps(:,1),'Bandwidth',1e-03);
plot(x,p/17,'LineWidth',2);
set(gca,'YTick',[]);
%histogram(posamps(:,1),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,1));
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,1),'Normalization','pdf','Edgecolor','none');
text(0.005,.9,...
    'Deflection','VerticalAlignment','bottom','Units','normalized');
text(1.715,102,'Rotation','VerticalAlignment','bottom');
% xlim([0.6 0.85]);
% ylim([0 110]);
line([des_obs(1) des_obs(1)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
% Rotation
axes(subplts(2));
[p,x,bw]=ksdensity(posamps(:,2),'Bandwidth',2.6e-05);
plot(x,p/7.4,'LineWidth',2);
set(gca,'YTick',[]);
%histogram(posamps(:,2),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,2));
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,2),'Normalization','pdf','Edgecolor','none');
text(0.005,.9,...
    'Rotation','VerticalAlignment','bottom','Units','normalized');
% xlim([0.075,0.105])
% ylim([0 700]);
ylim = get(gca,'ylim');
line([des_obs(2) des_obs(2)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
set(gca,'ylim',ylim);
% Cost
axes(subplts(3));
[p,x,bw]=ksdensity(posamps(:,3),'Bandwidth',2.080754589271729e-02);
plot(x,p/10,'LineWidth',2);
set(gca,'YTick',[]);
%histogram(posamps(:,3),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,3));
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,3),'Normalization','pdf','Edgecolor','none');
text(0.05,.9,...
    'Cost','VerticalAlignment','bottom','Units','normalized');
% ylim([0 .0700]);
xlim([60 400]);
ylim = [0 0.0145];%get(gca,'ylim');ylim
line([des_obs(3) des_obs(3)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
set(gca,'ylim',ylim);
% Add suptitle
% st=suptitle('Prior and posterior predictive distributions');
% st.Position=[0.5 -.1 0];
[lg,icons,~,~]=legend('Posterior','Prior','Target','Location','northeast');
% flushLegend(lg,'northeast');
resizeLegend();
pos=lg.Position; 
lg.Position = pos + [.0915 0.055 0 0];
% 
% %%% Save
set(f, 'Color','white');
% % export_fig FIG_prior_vs_posterior_dist -eps -m3 -painters
figstr = 'FIG_prior_vs_posterior_dist';
set(f,'PaperPositionMode','auto')
print(f,figstr,'-depsc','-r600')

%% WTA Pareto bands
clc ; clearvars -except dpath ; close all ;

%%% Load the results
locstr = [dpath,'stored_data\'...
    '2019-11-05_CTO_costgrid'];
load(locstr);
mean_y = results{1}.settings.mean_y ; std_y = results{1}.settings.std_y ;

% Collect Cost_lambdas, and posterior mean and sds for costs, defl, rot, as
% well as upper and lower .05 quantiles
m=length(results); % Store number of target cost_lambdas
cost_lambda = zeros(m,1); % This will store cost_lambdas
cred_level = 90; % Set desired level for credible bands (in %)
alpha = (100-cred_level)/100; % Convert cred_level to alpha level
pmo = zeros(m,2); % This will store posterior mean output of emulator
pdo = zeros(m,2); % ``'' median output
pso = zeros(m,2); % ``'' appropriate multiple of standard deviations
plo = zeros(m,2); % ``'' lower (alpha/2) quantile
puo = zeros(m,2); % ``'' upper (alpha/2) quantile
for ii = 1:m % This loop populates the above arrays
    output_means = results{ii}.model_output.means .* std_y + mean_y;
    output_sds = sqrt(results{ii}.model_output.vars) .* std_y;
    pmo(ii,:) = mean(output_means);
    pdo(ii,:) = quantile(output_means,0.5);
    pso(ii,:) = norminv(1-alpha/2) * ...
        mean(output_sds);
    plo(ii,:) = quantile(output_means,alpha/2);
    puo(ii,:) = quantile(output_means,1-alpha/2);
    cost(ii) = results{ii}.settings.obs_y(1,end) * std_y(2) + mean_y(2);
end
% Now we break the arrays up each into 2 vectors, one for each output
post_cost_mean = pmo(:,2);
post_defl_mean = pmo(:,1);
post_cost_median = pdo(:,2);
post_defl_median = pdo(:,1);
post_cost_sd = pso(:,2);
post_defl_sd = pso(:,1);
post_cost_lq = plo(:,2);
post_cost_uq = puo(:,2);
post_defl_lq = plo(:,1);
post_defl_uq = puo(:,1);
% Get quantiles plus code uncertainty
post_cost_uq_cu = post_cost_uq + post_cost_sd;
post_cost_lq_cu = post_cost_lq - post_cost_sd;
post_defl_uq_cu = post_defl_uq + post_defl_sd;
post_defl_lq_cu = post_defl_lq - post_defl_sd;
% Get ylims for the two sets of plots
ylimrat=1.01;
ylim_cost = [min(post_cost_lq_cu)/ylimrat max(post_cost_uq_cu)*ylimrat];
ylim_defl = [min(post_defl_lq_cu)/ylimrat max(post_defl_uq_cu)*ylimrat];

%%% Begin figures
% Set alphas for two types of uncertainty
alpha_wcu = 0.5;  %with code uncertainty
alpha_wocu= 0.15; %without
h=figure('rend','painters','pos',[10 10 360 240]);
x = 96:1:350; % x fills the cost domain
[subplts , pos] = tight_subplot(1,2,0.175,[0.15 0.02],[0.11 0.01]);

% Now begin plot 1/2
axes(subplts(1));
% Get main curve
pdefl = pchip(cost,post_defl_median,x);
% Get upper and lower 0.05 quantiles curves
pdefluq = pchip(cost,post_defl_uq,x);
pdefllq = pchip(cost,post_defl_lq,x);
unc_wo_cu = fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(unc_wo_cu,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
median_line = plot(x,pdefl,'-r','LineWidth',1.5); % Mean
% plot(x,pdefluq,':k',...
%      x,pdefllq,':k');
xl2=xlabel('Target cost');
ylabel('Deflection');
xlim([96,350]);
ylim(ylim_defl);

figpos = get(h,'pos');

% Add NSGA-II results
locstr = [dpath,'stored_data\'...
    '2020-04-23_NSGA2_results'];
load(locstr);
nsga2_res = plot(result.final_obj(:,2),result.final_obj(:,1),'*');


% Now add a legend.
ylim(ylim+[0 0.03]);
leg_gos = [nsga2_res median_line ];% go_plot_diag];
lg=legend(leg_gos,sprintf('NSGA-II results'),...
    sprintf('Posterior\npredictive median'),...
    'Location','northeast');
% lg.Position(1:2)=[.623 .725];
flushLegend(lg,'northeast');
lg.Box='off';
% lgpos = lg.Position;
% lg.Position = lgpos + [-.004 -.002 -.004 -.002];


% saveas(h,'FIG_cost_grid_pareto.png');

% Now add in code uncertainty. That is, the above assumes that the GP
% emulator nails the FE code precisely. But of course the GP emulator has
% nonnegligible variance. That's the code uncertainty. So our confidence
% bands should reflect it. So we add it in here, by dropping the
% appropriate multiple of the sd from each lower quantile and adding it to
% each upper quantile.
% First, open the figure prior to calling suptitle.
% h=openfig('tempfig');
axes(subplts(2));
pdefluq_code_uncert = pchip(cost,post_defl_uq_cu,x);
pdefllq_code_uncert = pchip(cost,post_defl_lq_cu,x);
f=fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(f,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
f=fill([ x , fliplr(x) ], [pdefluq_code_uncert,...
    fliplr(pdefluq)],'k');
unc_w_cu=fill([ x , fliplr(x) ], [pdefllq_code_uncert,...
    fliplr(pdefllq)],'k');
hold on;
median_line = plot(x,pdefl,'-r','LineWidth',1.5); % Mean
set(f,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
set(unc_w_cu,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
ylabel('Deflection');
xl2=xlabel('Target cost');
xlim([183.7013 186.1534]);
ylim([0.7255 0.7273]);

% Add NSGA-II results
locstr = [dpath,'stored_data\'...
    '2020-04-23_NSGA2_results'];
load(locstr);
plot(result.final_obj(:,2),result.final_obj(:,1),'*');

% Now add a main title and fix any infelicities
% suptitle(['Posterior estimate vs. target cost,',...
%     ' with ',num2str(cred_level),'% credible interval ']); 
set(h,'pos',figpos); % Just so we can reuse the positioning code from above
% p = get(xl1,'position');
% set(xl1,'position',p + [0 2.75 0]);
% p = get(xl2,'position');
% set(xl2,'position',p + [0 0.00125 0])
% p = get(xl3,'position');
% set(xl3,'position',p + [0 0.0002 0])

% Now add a legend.
ylim(ylim+[0 0.0007]);
leg_gos = [median_line unc_wo_cu unc_w_cu];% go_plot_diag];
lg=legend(leg_gos,sprintf('Posterior\npredictive median'),...
    sprintf('C.I. w/o code\nuncertainty'),...
    sprintf('C.I. with code\nuncertainty'),...
    'Location','northeast');
% lg.Position(1:2)=[.623 .725];
flushLegend(lg,'northeast');
lg.Box='off';
% lgpos = lg.Position;
% lg.Position = lgpos + [-.004 -.002 -.004 -.002];

    
%%% Save
set(h,'Color','white');
% export_fig 	 -png -m3 -painters
% saveas(h,'FIG_cost_grid_pareto_with_code_uncert.png');
figstr = 'FIG_cost_grid_pareto_bands';
set(h,'PaperPositionMode','auto')
% print(h,figstr,'-depsc','-r600')

%% WTA Pareto bands (new 2020-04-23)
clc ; clearvars -except dpath ; close all ;

%%% Load the results
locstr = [dpath,'stored_data\'...
    '2019-11-05_CTO_costgrid'];
load(locstr);
mean_y = results{1}.settings.mean_y ; std_y = results{1}.settings.std_y ;

% Collect Cost_lambdas, and posterior mean and sds for costs, defl, rot, as
% well as upper and lower .05 quantiles
m=length(results); % Store number of target cost_lambdas
cost_lambda = zeros(m,1); % This will store cost_lambdas
cred_level = 90; % Set desired level for credible bands (in %)
alpha = (100-cred_level)/100; % Convert cred_level to alpha level
pmo = zeros(m,2); % This will store posterior mean output of emulator
pdo = zeros(m,2); % ``'' median output
pso = zeros(m,2); % ``'' appropriate multiple of standard deviations
plo = zeros(m,2); % ``'' lower (alpha/2) quantile
puo = zeros(m,2); % ``'' upper (alpha/2) quantile
for ii = 1:m % This loop populates the above arrays
    output_means = results{ii}.model_output.means .* std_y + mean_y;
    output_sds = sqrt(results{ii}.model_output.vars) .* std_y;
    pmo(ii,:) = mean(output_means);
    pdo(ii,:) = quantile(output_means,0.5);
    pso(ii,:) = norminv(1-alpha/2) * ...
        mean(output_sds);
    plo(ii,:) = quantile(output_means,alpha/2);
    puo(ii,:) = quantile(output_means,1-alpha/2);
    cost(ii) = results{ii}.settings.obs_y(1,end) * std_y(2) + mean_y(2);
end
% Now we break the arrays up each into 2 vectors, one for each output
post_cost_mean = pmo(:,2);
post_defl_mean = pmo(:,1);
post_cost_median = pdo(:,2);
post_defl_median = pdo(:,1);
post_cost_sd = pso(:,2);
post_defl_sd = pso(:,1);
post_cost_lq = plo(:,2);
post_cost_uq = puo(:,2);
post_defl_lq = plo(:,1);
post_defl_uq = puo(:,1);
% Get quantiles plus code uncertainty
post_cost_uq_cu = post_cost_uq + post_cost_sd;
post_cost_lq_cu = post_cost_lq - post_cost_sd;
post_defl_uq_cu = post_defl_uq + post_defl_sd;
post_defl_lq_cu = post_defl_lq - post_defl_sd;
% Get ylims for the two sets of plots
ylimrat=1.01;
ylim_cost = [min(post_cost_lq_cu)/ylimrat max(post_cost_uq_cu)*ylimrat];
ylim_defl = [min(post_defl_lq_cu)/ylimrat max(post_defl_uq_cu)*ylimrat];

%%% Begin figures
% Set alphas for two types of uncertainty
alpha_wcu = 0.5;  %with code uncertainty
alpha_wocu= 0.15; %without
h=figure('rend','painters','pos',[10 10 360 240]);
x = 96:1:350; % x fills the cost domain
[subplts , pos] = tight_subplot(1,2,0.175,[0.15 0.02],[0.11 0.01]);

% Now begin plot 1/2
axes(subplts(1));
% Get main curve
pdefl = pchip(cost,post_defl_median,x);
% Get upper and lower 0.05 quantiles curves
pdefluq = pchip(cost,post_defl_uq,x);
pdefllq = pchip(cost,post_defl_lq,x);
unc_wo_cu = fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(unc_wo_cu,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
median_line = plot(x,pdefl,'-r','LineWidth',1.5); % Mean
% plot(x,pdefluq,':k',...
%      x,pdefllq,':k');
xl2=xlabel('Target cost');
ylabel('Deflection');
xlim([96,350]);
ylim(ylim_defl);

figpos = get(h,'pos');

% Add NSGA-II results
locstr = [dpath,'stored_data\'...
    '2020-04-23_NSGA2_results'];
load(locstr);
nsga2_res = plot(result.final_obj(:,2),result.final_obj(:,1),'*');


% Now add a legend.
ylim(ylim+[0 0.03]);
leg_gos = [nsga2_res median_line ];% go_plot_diag];
lg=legend(leg_gos,sprintf('NSGA-II results'),...
    sprintf('Posterior\npredictive median'),...
    'Location','northeast');
% lg.Position(1:2)=[.623 .725];
flushLegend(lg,'northeast');
lg.Box='off';
% lgpos = lg.Position;
% lg.Position = lgpos + [-.004 -.002 -.004 -.002];


% saveas(h,'FIG_cost_grid_pareto.png');

% Now add in code uncertainty. That is, the above assumes that the GP
% emulator nails the FE code precisely. But of course the GP emulator has
% nonnegligible variance. That's the code uncertainty. So our confidence
% bands should reflect it. So we add it in here, by dropping the
% appropriate multiple of the sd from each lower quantile and adding it to
% each upper quantile.
% First, open the figure prior to calling suptitle.
% h=openfig('tempfig');
axes(subplts(2));
pdefluq_code_uncert = pchip(cost,post_defl_uq_cu,x);
pdefllq_code_uncert = pchip(cost,post_defl_lq_cu,x);
f=fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(f,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
f=fill([ x , fliplr(x) ], [pdefluq_code_uncert,...
    fliplr(pdefluq)],'k');
unc_w_cu=fill([ x , fliplr(x) ], [pdefllq_code_uncert,...
    fliplr(pdefllq)],'k');
hold on;
median_line = plot(x,pdefl,'-r','LineWidth',1.5); % Mean
set(f,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
set(unc_w_cu,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
ylabel('Deflection');
xl2=xlabel('Target cost');
xlim([183.7013 186.1534]);
ylim([0.7255 0.7273]);

% Add NSGA-II results
locstr = [dpath,'stored_data\'...
    '2020-04-23_NSGA2_results'];
load(locstr);
plot(result.final_obj(:,2),result.final_obj(:,1),'*');

% Now add a main title and fix any infelicities
% suptitle(['Posterior estimate vs. target cost,',...
%     ' with ',num2str(cred_level),'% credible interval ']); 
set(h,'pos',figpos); % Just so we can reuse the positioning code from above
% p = get(xl1,'position');
% set(xl1,'position',p + [0 2.75 0]);
% p = get(xl2,'position');
% set(xl2,'position',p + [0 0.00125 0])
% p = get(xl3,'position');
% set(xl3,'position',p + [0 0.0002 0])

% Now add a legend.
ylim(ylim+[0 0.0007]);
leg_gos = [median_line unc_wo_cu unc_w_cu];% go_plot_diag];
lg=legend(leg_gos,sprintf('Posterior\npredictive median'),...
    sprintf('C.I. w/o code\nuncertainty'),...
    sprintf('C.I. with code\nuncertainty'),...
    'Location','northeast');
% lg.Position(1:2)=[.623 .725];
flushLegend(lg,'northeast');
lg.Box='off';
% lgpos = lg.Position;
% lg.Position = lgpos + [-.004 -.002 -.004 -.002];

    
%%% Save
set(h,'Color','white');
% export_fig 	 -png -m3 -painters
% saveas(h,'FIG_cost_grid_pareto_with_code_uncert.png');
figstr = 'FIG_cost_grid_pareto_bands';
set(h,'PaperPositionMode','auto')
% print(h,figstr,'-depsc','-r600')


%% Example of selecting performance target, v3
clc ; clearvars -except dpath ; close all ;
fig=figure('Position',[10 10 390 240]);

% Define Pareto front
x = linspace(1.03,2.39);
xt = linspace(1.03,2.39,10);
ytl = [1 .6 .4 .15 .09 .0675 .0525 .04 .025 0.015];
yl = spline(xt,ytl,x);

% Define upper model bound
ytu = [1 1.1 1.12 1.14 1.1 .9 .7 .5 .4 .015];
yu = spline(xt,ytu,fliplr(x));

% Plot the model range
p1 = fill([x fliplr(x)],[yl yu],[1 .175 .175]);
% axis equal;
%hold on; plot(xt,ytl,'o'); plot(xt,ytu,'o');
xlim([0,2.4]);ylim([0,1.2]);

% Get closest point to [0,0], and connect it to [0,0]
dists = sqrt(sum([x(:) yl(:) ].^2,2)) ;
[mnm,idx]=min(dists);
fdists = sqrt(sum([x(:) yu(:) ].^2,2));
hold on; 
p2 = plot(x(idx),yl(idx),'ok','LineWidth',2,'MarkerSize',5);
p3 = plot(0,0,'xk','LineWidth',2,'MarkerSize',7);
p4 = plot([0 x(idx)],[0 yl(idx)],'-k','LineWidth',1);

% Also plot elbow and nearby desired observation
lbloc = 37 ; % elbow location index
plot(x(lbloc),yl(lbloc),'ok','LineWidth',2,'MarkerSize',5);
des_obs = [1.4825 0.065];
plot(des_obs(1),des_obs(2),'xk','LineWidth',2,'MarkerSize',7);
plot([des_obs(1) x(lbloc)],[des_obs(2) yl(lbloc)],'-k','LineWidth',1);
dodist = norm([des_obs(1)-x(lbloc), des_obs(2)-yl(lbloc)]);

% Get max distance from [0 0]
dists_up = sqrt(sum([x(:) yu(:)].^2,2)) ;
[mxm,idx] = max(dists_up);

% Add partial circle showing region within 2 * dist from des_obs
dist_ratio = mxm/mnm ; 
x_idx = abs(x-des_obs(1))<=dist_ratio*dodist ;
xr = x(x_idx) ; 
yr = sin(acos((xr-des_obs(1))/dist_ratio/dodist)) ; 
yrt=yr*dist_ratio*dodist+des_obs(2); % Similarly for y-vals
in_mod_range_idx = yl(x_idx) <= yrt;
xrr = xr( in_mod_range_idx );
yrr = yrt( in_mod_range_idx );
% p5 = plot(xrr,yrr,':k','LineWidth',2);

% Resize
% fig.Position = fig.Position + [ 0 0 0 -175];


% Add labels and legend
ylabel('y_2'); xlb = xlabel('y_1');
lg = legend([p1 p3 p2],{sprintf('Feasible\ndesign space'),...
    sprintf('Possible target\noutcome'),...
    sprintf('Nearest point\nto target')},...
    'Location','Northwest');
flushLegend(lg,'northwest');
pos = lg.Position;
% lg.Position = 
% ax=gca; ax.Position = ax.Position + [0 0.01 0 0];
% xlb.Position = xlb.Position + [0 .08 0];


% Save
set(fig,'color','white');
figstr = sprintf('FIG_des_obs_selection_example2');
% export_fig(figstr,'-eps','-q0','-painters',fig);

set(fig,'PaperPositionMode','auto')
% print(fig,figstr,'-depsc','-r600')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figures for R&R to Journal of Mechanical Design %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% WTA Contour plot of highest density regions of posterior distribution 
clc ; clearvars -except dpath ; close all ;

%%% Load the calibration results
% clearvars -except dpath res ; 
samps = res.theta1;
% load([dpath,'stored_data\'...
%     '2018-07-27_discrepancy_d-elbow_d-p2'],...
%     'results');
% samps = results.samples_os(results.settings.burn_in+2:end,:) ;

ff=figure('pos',[10 10  320.0000  240]);

% Get the density
[f,x,bw]=ksdensity(samps);

% Correct it by enforcing the boundary conditions:
f(x(:,1)>0.6 | x(:,2)<10) = 0 ;

% Reshape for the contour plot
X = reshape(x(:,1),30,30); Y = reshape(x(:,2),30,30); Z = reshape(f,30,30);

% Convert Z to quantiles in posterior dist
[f,x,bw]=ksdensity(samps,samps);
Z = reshape(sum(f<Z(:)')/size(f,1),30,30);

% Get contour plot and labels
colormap HSV
[C9,h9] = contour(X,Y,Z, [ .9 .9 ]); hold on;
[C,h] = contour(X,Y,Z, [ .75 .75 ], 'k--');
[C5,h5] = contour(X,Y,Z, [ .5 .5 ], '--');
[C,h] = contour(X,Y,Z, [ .25 .25 ], 'k-.');
[C1,h1] = contour(X,Y,Z, [ .1 .1 ], '-.');
[C01,h01] = contour(X,Y,Z, [ .01 .01 ], ':');
w=1.25; h9.LineWidth=w; h5.LineWidth=w; h1.LineWidth=w; h01.LineWidth=w;
xlim([0,0.6]);ylim([10,25]);

lg=legend(...
    '0.1 HDR','0.5 HDR', '0.9 HDR','0.99 HDR','Location','northwest');
title('Posterior distribution of \theta');
xlabel('Volume fraction'); ylabel('Thickness (mm)');

% Save
set(ff,'Color','w');
% export_fig 'FIG_post_dist_contourplot' -eps -m3;

%% WTA Histogram2 of highest density regions of posterior distribution 
clc ; clearvars -except dpath ; close all ;

%%% Load the calibration results
% clearvars -except dpath res ; 
locstr = [dpath,'stored_data\'...
    '2019-11-05_CTO'];
load(locstr);
samps = res.theta1(res.settings.burn_in:end,:) ;

fighist = figure('pos',[10 10  360.0000  240]);
binwidth = 1.5e-3*[0.002 1];
h2=histogram2(samps(:,1),samps(:,2),...
    'Normalization','pdf');
set(gca,'View',[225 30]);
set(gca,'ZTick',[]);
grid off;
ttlhist = title('Posterior distribution of \theta');
ttlhist.FontSize = 11;
xlbl = xlabel('Vol. fraction'); ylbl = ylabel('Thickness (mm)');
set(fighist,'Color','w');
xlim([0 0.6]);
ylim([10 25]);
% pause(1.5);

% Save it
figstr = 'FIG_post_dist_hist2';
set(fighist,'PaperPositionMode','auto')
ylbl.Units = 'pixels'; xlbl.Units='pixels';
ylbl.Position = ylbl.Position + [-8.0 2 0];
xlbl.Position = xlbl.Position + [8.0 2 0];
% print(fighist,figstr,'-depsc','-r600')

%% WTA prior predictive distribution vs posterior predictive distribution
clc ; clearvars -except dpath ; close all ;

%%% Load prior predictive results
load([dpath,'stored_data\'...
    '2019-11-06_prior_predictive_distributions']);
prsamps = prior_model_output.means;
clear prior_pred_dist;

%%% Load calib results
load([dpath,'stored_data\'...
    '2018-07-27_discrepancy_d-elbow_d-p2'],...
    'results');
posamps = results.model_output.by_sample_est;
des_obs = results.settings.desired_obs;

%%% Rescale prior samps
outsds = mean(results.settings.output_sds,2);
outmeans = mean(results.settings.output_means,2);
prsamps = prsamps .* outsds' + outmeans';

%%% Make figure using histograms
f=figure('pos',[10 10  360.0000  200]);
[subplts,pos] = tight_subplot(1,3,0.02,[ 0.08 0.01],0.03);
% Deflection
axes(subplts(1));
[p,x,bw]=ksdensity(posamps(:,1));
max_lim = max(p);
plot(x,p,'LineWidth',2);
set(gca,'YTick',[]);
%histogram(posamps(:,1),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,1));
max_lim = max([p(:);max_lim]);
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,1),'Normalization','pdf','Edgecolor','none');
text(0.025,.9,...
    'Deflection','VerticalAlignment','bottom','Units','normalized');
% text(1.715,102,'Rotation','VerticalAlignment','bottom');
% xlim([0.6 0.85]);
% ylim([0 110]);
line([des_obs(1) des_obs(1)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);


% Rotation
axes(subplts(2));
[p,x,bw]=ksdensity(posamps(:,2));
max_lim = max(p);
plot(x,p,'LineWidth',2);
set(gca,'YTick',[]);
%histogram(posamps(:,2),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,2));
max_lim = max([p(:);max_lim]);
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,2),'Normalization','pdf','Edgecolor','none');
text(0.025,.9,...
    'Rotation','VerticalAlignment','bottom','Units','normalized');
% xlim([0.075,0.105])
% ylim([0 700]);
ylim = get(gca,'ylim');
line([des_obs(2) des_obs(2)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
set(gca,'ylim',ylim);


% Cost
axes(subplts(3));
[p,x,bw]=ksdensity(posamps(:,3));
max_lim = max(p);
plot(x,p,'LineWidth',2);
set(gca,'YTick',[]);
%histogram(posamps(:,3),'Normalization','pdf','Edgecolor','none'); 
hold on;
[p,x]=ksdensity(prsamps(:,3));
max_lim = max([p(:);max_lim]);
plot(x,p,'--','LineWidth',2);
%histogram(prsamps(:,3),'Normalization','pdf','Edgecolor','none');
text(0.25,.9,...
    'Cost','VerticalAlignment','bottom','Units','normalized');
% ylim([0 .0700]);
xlim([60 400]);
ylim = [0 max_lim*1.15];%get(gca,'ylim');ylim
line([des_obs(3) des_obs(3)],ylim,'Color','black','Linestyle',':',...
    'linewidth',2);
set(gca,'ylim',ylim);


% Add suptitle
% st=suptitle('Prior and posterior predictive distributions');
% st.Position=[0.5 -.1 0];
[lg,icons,~,~]=legend('Posterior','Prior','Target','Location','east');
% flushLegend(lg,'east');
resizeLegend();
pos=lg.Position; 
lg.Position = pos + [.091 0.055 0 0];
% 
% %%% Save
set(f, 'Color','white');
% % export_fig FIG_prior_vs_posterior_dist -eps -m3 -painters
figstr = 'FIG_prior_vs_posterior_dist';
set(f,'PaperPositionMode','auto')
print(f,figstr,'-depsc','-r600')

%% WTA Pareto bands (new 2020-04-23)
clc ; clearvars -except dpath ; close all ;

%%% Load the results
locstr = [dpath,'stored_data\'...
    '2019-11-05_CTO_costgrid'];
load(locstr);
mean_y = results{1}.settings.mean_y ; std_y = results{1}.settings.std_y ;

% Collect Cost_lambdas, and posterior mean and sds for costs, defl, rot, as
% well as upper and lower .05 quantiles
m=length(results); % Store number of target cost_lambdas
cost_lambda = zeros(m,1); % This will store cost_lambdas
cred_level = 90; % Set desired level for credible bands (in %)
alpha = (100-cred_level)/100; % Convert cred_level to alpha level
pmo = zeros(m,2); % This will store posterior mean output of emulator
pdo = zeros(m,2); % ``'' median output
pso = zeros(m,2); % ``'' appropriate multiple of standard deviations
plo = zeros(m,2); % ``'' lower (alpha/2) quantile
puo = zeros(m,2); % ``'' upper (alpha/2) quantile
for ii = 1:m % This loop populates the above arrays
    output_means = results{ii}.model_output.means .* std_y + mean_y;
    output_sds = sqrt(results{ii}.model_output.vars) .* std_y;
    pmo(ii,:) = mean(output_means);
    pdo(ii,:) = quantile(output_means,0.5);
    pso(ii,:) = norminv(1-alpha/2) * ...
        mean(output_sds);
    plo(ii,:) = quantile(output_means,alpha/2);
    puo(ii,:) = quantile(output_means,1-alpha/2);
    cost(ii) = results{ii}.settings.obs_y(1,end) * std_y(2) + mean_y(2);
end
% Now we break the arrays up each into 2 vectors, one for each output
post_cost_mean = pmo(:,2);
post_defl_mean = pmo(:,1);
post_cost_median = pdo(:,2);
post_defl_median = pdo(:,1);
post_cost_sd = pso(:,2);
post_defl_sd = pso(:,1);
post_cost_lq = plo(:,2);
post_cost_uq = puo(:,2);
post_defl_lq = plo(:,1);
post_defl_uq = puo(:,1);
% Get quantiles plus code uncertainty
post_cost_uq_cu = post_cost_uq + post_cost_sd;
post_cost_lq_cu = post_cost_lq - post_cost_sd;
post_defl_uq_cu = post_defl_uq + post_defl_sd;
post_defl_lq_cu = post_defl_lq - post_defl_sd;
% Get ylims for the two sets of plots
ylimrat=1.01;
ylim_cost = [min(post_cost_lq_cu)/ylimrat max(post_cost_uq_cu)*ylimrat];
ylim_defl = [min(post_defl_lq_cu)/ylimrat max(post_defl_uq_cu)*ylimrat];

%%% Begin figures
% Set alphas for two types of uncertainty
alpha_wcu = 0.5;  %with code uncertainty
alpha_wocu= 0.15; %without
h=figure('rend','painters','pos',[10 10 360 240]);
x = 96:1:350; % x fills the cost domain
[subplts , pos] = tight_subplot(1,2,0.175,[0.15 0.02],[0.11 0.01]);

% Now begin plot 1/2
axes(subplts(1));
% Get main curve
pdefl = pchip(cost,post_defl_median,x);
% Get upper and lower 0.05 quantiles curves
pdefluq = pchip(cost,post_defl_uq,x);
pdefllq = pchip(cost,post_defl_lq,x);
unc_wo_cu = fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(unc_wo_cu,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
median_line = plot(x,pdefl,'-r','LineWidth',1.5); % Mean
% plot(x,pdefluq,':k',...
%      x,pdefllq,':k');
xl2=xlabel('Target cost');
ylabel('Deflection');
xlim([96,350]);
ylim(ylim_defl);

figpos = get(h,'pos');

% Add NSGA-II results
locstr = [dpath,'stored_data\'...
    '2020-04-23_NSGA2_results'];
load(locstr);
nsga2_res = plot(result.final_obj(:,2),result.final_obj(:,1),'*');


% Now add a legend.
ylim(ylim+[0 0.03]);
leg_gos = [nsga2_res median_line ];% go_plot_diag];
lg=legend(leg_gos,sprintf('NSGA-II results'),...
    sprintf('Posterior\npredictive median'),...
    'Location','northeast');
% lg.Position(1:2)=[.623 .725];
flushLegend(lg,'northeast');
lg.Box='off';
% lgpos = lg.Position;
% lg.Position = lgpos + [-.004 -.002 -.004 -.002];


% saveas(h,'FIG_cost_grid_pareto.png');

% Now add in code uncertainty. That is, the above assumes that the GP
% emulator nails the FE code precisely. But of course the GP emulator has
% nonnegligible variance. That's the code uncertainty. So our confidence
% bands should reflect it. So we add it in here, by dropping the
% appropriate multiple of the sd from each lower quantile and adding it to
% each upper quantile.
% First, open the figure prior to calling suptitle.
% h=openfig('tempfig');
axes(subplts(2));
pdefluq_code_uncert = pchip(cost,post_defl_uq_cu,x);
pdefllq_code_uncert = pchip(cost,post_defl_lq_cu,x);
f=fill([ x , fliplr(x) ], [pdefluq, fliplr(pdefllq)],'k');
set(f,'facealpha',alpha_wocu,'EdgeAlpha',alpha_wocu);
hold on;
f=fill([ x , fliplr(x) ], [pdefluq_code_uncert,...
    fliplr(pdefluq)],'k');
unc_w_cu=fill([ x , fliplr(x) ], [pdefllq_code_uncert,...
    fliplr(pdefllq)],'k');
hold on;
median_line = plot(x,pdefl,'-r','LineWidth',1.5); % Mean
set(f,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
set(unc_w_cu,'facealpha',alpha_wcu,'EdgeAlpha',alpha_wcu);
ylabel('Deflection');
xl2=xlabel('Target cost');
xlim([183.7013 186.1534]);
ylim([0.7255 0.7273]);

% Add NSGA-II results
locstr = [dpath,'stored_data\'...
    '2020-04-23_NSGA2_results'];
load(locstr);
plot(result.final_obj(:,2),result.final_obj(:,1),'*');

% Now add a main title and fix any infelicities
% suptitle(['Posterior estimate vs. target cost,',...
%     ' with ',num2str(cred_level),'% credible interval ']); 
set(h,'pos',figpos); % Just so we can reuse the positioning code from above
% p = get(xl1,'position');
% set(xl1,'position',p + [0 2.75 0]);
% p = get(xl2,'position');
% set(xl2,'position',p + [0 0.00125 0])
% p = get(xl3,'position');
% set(xl3,'position',p + [0 0.0002 0])

% Now add a legend.
ylim(ylim+[0 0.0007]);
leg_gos = [median_line unc_wo_cu unc_w_cu];% go_plot_diag];
lg=legend(leg_gos,sprintf('Posterior\npredictive median'),...
    sprintf('C.I. w/o code\nuncertainty'),...
    sprintf('C.I. with code\nuncertainty'),...
    'Location','northeast');
% lg.Position(1:2)=[.623 .725];
flushLegend(lg,'northeast');
lg.Box='off';
% lgpos = lg.Position;
% lg.Position = lgpos + [-.004 -.002 -.004 -.002];

    
%%% Save
set(h,'Color','white');
% export_fig 	 -png -m3 -painters
% saveas(h,'FIG_cost_grid_pareto_with_code_uncert.png');
figstr = 'FIG_cost_grid_pareto_bands';
set(h,'PaperPositionMode','auto')
% print(h,figstr,'-depsc','-r600')



