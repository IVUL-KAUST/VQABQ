close all; clear; clc;

chop = 0.5; %solve `chop`% of the data

for o=1:ceil(10000/chop)
    % parameters
    lambda = 1e-5;
    input_file = './AB.mat';
    output_file = strcat(strcat('./X', num2str(o)),'.mat');

	% loading
    disp('loding data...');
    load(input_file);

    [~, q] = size(B);
    chunk = ceil(q*chop/100);
    if chunk*(o+1) <= q
        B = B(:, chunk*o+1:chunk*(o+1));
    else
        B = B(:, chunk*o+1:q);
    end

    [m, n] = size(A);
    disp(['size of A is ', num2str(m), 'x', num2str(n)]);
    [m, q] = size(B);
    disp(['size of B is ', num2str(m), 'x', num2str(q)]);

	% preprocessing
    disp('preprocess B...');
    b = cell(size(B,2),1);
    for i=1:size(b,1)
       b{i} = B(:,i); 
    end
    clear B;

    % solving
    disp(['starting solver...']);
    x = cell(q,1);
    parfor i=1:q
        disp(['[', datestr(now), ']', ' question #', num2str(i), '...']);
        x{i} = lasso(A,b{i}, 'Lambda', lambda);
    end
    disp('done! ~AlhamduLeAllah');
    clear A b;

	% postprocessing
    disp('postprocessing x...');
    X = zeros(n, q);
    for i=1:q
       X(:,i) = x{i}; 
    end
    clear x;

	% saving
    disp(['saving x to file ', output_file]);
    save(output_file, 'X');
    clear X;
end