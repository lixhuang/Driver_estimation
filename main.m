% load data
clear
load('proc_data.mat');
B2 = [eye(2);zeros(2)];

x = x(1:2);
%calculate prior
xr_cov = 1;
vr_cov = 1;
R = diag([xr_cov, vr_cov]);
u = [];
Ob = [];
for ii = 1:length(x)
    u = [u;diff(x{ii}(8,:)'/0.1)];
    x1 = x{ii}(7:8,1:end-1) - x{ii}(1:2,1:end-1);
    x2 = B2*x{ii}(7:8,1:end-1) - x{ii}(3:6,1:end-1);
    gamma = [x1;x2]';
    Ob = [Ob;ones([size(gamma,1),1]),gamma];
end
Q_inv = 1/(2*vr_cov)*eye(size(Ob,1));
prior_mean = (Ob'*Q_inv*Ob)^-1*Ob'*Q_inv*u;
prior_cov = (Ob'*Q_inv*Ob)^-1;
%prior_guess = [prior_mean;100];
prior_guess = [prior_mean; 1];

%S = blkdiag(prior_cov, 1);
alpha = 3;
beta = 1;
%S = eye(8);
%log_prior = @(x)-1/2*(x(1:7)-prior_guess)'*S^-1*(x(1:7)-prior_guess)-log(sqrt(2*pi)*det(S));
log_prior = @(x)log(mvnpdf(x(1:7),prior_mean, prior_cov))+log(gampdf(x(8),alpha,beta));

%estimate
[samples, accepted_rat, warm_param] = generate_samples_DRAM(prior_guess, 9000, x, R, log_prior, 0, 0);
figure;
plot(samples(1,:))