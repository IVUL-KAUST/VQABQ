%% cost function for LASSO
function cost = compute_LASSO_cost(A,b,c,lambda,factor)
cost=factor*sum((A*c-b).^2)+lambda*sum(abs(c));
end