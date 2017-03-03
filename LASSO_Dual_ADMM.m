%% This function minimizes the following optimization in the dual form:
%% min_x 0.5*\|x\|_2^2+lambda*\|x\|_1

% e.g. 
% >> all_params = struct('threshold',0.01,'initial_rho',1);
% >> [gamma,diff_value,time,iter] = LASSO_Dual_ADMM(A,b,lambda,all_params);

function [gamma,final_cost,diff_value,time,iter] = LASSO_Dual_ADMM(A,b,lambda,all_params,AAt,U,d)

initial_params = struct('lambda',lambda,'opt',1,'is_verbose',false,...
        'rho_upper_limit',100, 'history_size',5,'max_iters',50000,...
        'threshold',0.05,'initial_rho',1,'learning_fact',1,'gamma_val',1,...
        'gamma_factor',1,'inc_iter',5,'Q',50,'initial_gamma',...
        randn(size(A,2),1,'double'),'accelerated',true,'AAt',[],'U',[],'D',[]);

if ~exist('all_params','var')
    all_params = [];
end
if ~exist('AAt','var')
    AAt=A*A';
end


param_names = fieldnames(initial_params);
for p = 1:numel(param_names)    
    if ~isfield(all_params,param_names(p))
        all_params.(param_names{p}) = initial_params.(param_names{p});
    end
end


num_rows        = size(A,1);
num_col         = size(A,2);
max_iters       = all_params.max_iters;
% history_size    = all_params.history_size;
threshold       = all_params.threshold;
initial_rho     = all_params.initial_rho;
gamma_val       = all_params.gamma_val;
inc_iter        = all_params.inc_iter;
learning_fact   = all_params.learning_fact;
gamma_factor    = all_params.gamma_factor;

%% Initialization
psi=zeros(num_rows,1,'double');
gamma = all_params.initial_gamma;
rho = double(initial_rho);
zeta = (A'*psi)+(gamma/rho);

iter=1;
factor=double(0.5);
diff_value=double(inf);

% obj_list=[];
% obj_list(1)=compute_LASSO_cost(A,b,gamma,lambda,factor);
qfac=single((sqrt(all_params.Q)-1)/(sqrt(all_params.Q)+1));


if all_params.accelerated
    if isempty(U) || isempty(d)
        disp('got here');
        [U,D] = eig(AAt);
        d = diag(D);
        clear D;        
    end
end

tic;
while(diff_value>threshold && iter<max_iters)
    
    %% Update Psi
    rhs = A*(zeta-gamma/rho)+ b/rho;
    if ~all_params.accelerated
        lhs=AAt+2*factor*eye(num_rows)/rho;        
        psi=inv(lhs)*rhs;
    else
        psi = U*((U'*rhs)./(d+2*factor/rho));
    end
    
    %% Update zeta
    zeta=(A'*psi)+(gamma/rho);
    idx=abs(zeta)>lambda;
    zeta(idx)=sign(zeta(idx))*lambda;
    
    diff_vec = A'*psi-zeta;
    if true
        %% Update gamma No Nestrov        
        gamma = gamma + (gamma_val*rho)*diff_vec;
    else        
        %% Update gamma with Nestrov
        gamma0=gamma;
        gamma = gamma + (gamma_val*rho)*diff_vec;
        gamma=(1+qfac)*gamma-qfac*gamma0;        
    end
    
    
    %% Cost and Standard Deviation
    %obj_list=[obj_list;compute_LASSO_cost(A,b,gamma,lambda,factor)];
    
    %% Stopping Criterion
    diff_value=norm(diff_vec)/norm(zeta);
    %diff_value=norm(gamma-x)/norm(x);
%     diff_value=norm((gamma-gamma_out0))/norm(gamma);

    %% Adjusting Parameters
    if mod(iter,inc_iter)==0
        rho = min(all_params.rho_upper_limit,learning_fact*rho);
        gamma_val= max(gamma_val*gamma_factor,1);
        
        if all_params.is_verbose
            [diff_value compute_LASSO_cost(A,b,gamma,lambda,factor)]
        end
    end
    iter=iter+1;
end
time=toc;
final_cost = compute_LASSO_cost(A,b,gamma,lambda,factor);
end
