close all; clear all; 

%% input dictionary and vector
if ~exist('Ay.mat','file')
    m = 4000; n = 1e5;
    A = rand(m,n);
    y = rand(m,1);
else
    load('Ay.mat','A','y');    
    [m,n] = size(A);
    A = double(A);
    y = double(y); 
    
    % get the outer matrix ready for the dual ADMM
    tic;
    AAt = A*A';
    [U,D] = eig(AAt);
    toc;
end
    
lambda = 1;

%% solves LASSO in the primal: use mexLasso
% + min_x 0.5*\|[A ones(m,1)]x-b\|_2^2+lambda*\|x\|_1

mex_params = struct('mode',2,'lambda',lambda);
t1 = tic; 
mex_x = mexLasso(y,A,mex_params);
mex_time = toc(t1);
[compute_LASSO_cost(A,b,mex_x,lambda,0.5) mex_time]

%% solves LASSO in the dual: use LASSO_Dual_ADMM
% + min_x 0.5*\|[A ones(m,1)]x-b\|_2^2+lambda*\|x\|_1
tic;
all_params = struct('threshold',5*1e-4,'initial_rho',1e-6,...
    'initial_gamma',zeros(n,1),'gamma_val',1.0,'is_verbose',true,...
    'learning_fact',1.0);
[dual_ADMM_x,final_cost,diff_value,dual_LASSO_time,iter] = LASSO_Dual_ADMM(A,y,...
    lambda,all_params,AAt,U,diag(D));
total_dual_time = toc;

[final_cost dual_LASSO_time total_dual_time]