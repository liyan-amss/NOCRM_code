function f =recovery_ability(dataset)

% Generates synthetic data, runs DPAMAL feature selection multiple times,
% and prints mean of selected features.

%% Setup
nKmeans = 20; 
alfaCandi = 1e-6;
betaCandi = 1e-6;
nuCandi   = 100;
maxIter   = 20;
nRepeats  = 20;  % number of repetitions for stability analysis

n_samples  =200;
n_features =600;
n_true     =200;
n_noise     = n_features - n_true;
nClass     = 4;

cluster_std = 1;  % standard deviation of clusters
feaNum = n_true;  % number of features to select

% cluster centers for true features
cluster_centers = [2;-2;4;-4] * ones(1,n_true);
% cluster_centers =[2;-2;4;-4;6] * ones(1,n_true);

%% Preallocate results
true_counts = zeros(nRepeats,1);
noise_counts = zeros(nRepeats,1);

%% Repeat experiments
for r = 1:nRepeats
   rng(100+100*r,'twister');  % different random seed

    % Generate synthetic data
fea_true = [];
samples_per_class = floor(n_samples / nClass); 
for i = 1:nClass
    if i < nClass
        n_i = samples_per_class;                    
    else
        n_i = n_samples - (nClass-1)*samples_per_class; 
    end
    X_cluster = cluster_std * randn(n_i, n_true) + cluster_centers(i,:);
    fea_true = [fea_true; X_cluster];
end


   
% fea_noise = randn(n_samples, n_noise);


fea_noise = randn(n_samples, n_noise);

% Make 10% of them linearly correlated with the first 10% of the true features.
k = min(n_true,round(0.1*n_noise));
idx = randperm(n_noise, k);
fea_noise(:,idx) = fea_true(:,1:k) + randn(n_samples,k);
    fea = [fea_true, fea_noise];
    
    % Randomize feature order
    col_indices = randperm(n_features);
    fea = fea(:, col_indices);
    
    % Normalize
    fea = normalization(fea', 2)';
    
    %% Construct Laplacian
    S = constructW(fea);
    D = diag(sum(S,2));
    L = D^(-1/2) * (D - S) * D^(-1/2);
    
 
    eY = eigY(L, nClass);
    
    % Initialize cluster indicator
    label = litekmeans(eY, nClass, 'Replicates', nKmeans);
    Y = zeros(size(fea,1), nClass);
for i = 1:size(fea,1)
    Y(i, label(i)) = 1;
end

    
    %% DPAMAL feature selection
    W = DPAMAL(fea, L, Y, alfaCandi, betaCandi, nuCandi, maxIter);
    [~, idx] = sort(sum(W.*W,2), 'descend');
    selected_features = idx(1:feaNum);
    
    %% Count true and noise features
    true_features  = 1:n_true;
    noise_features = n_true+1:n_features;
    true_counts(r)  = sum(ismember(col_indices(selected_features), true_features));
    noise_counts(r) = sum(ismember(col_indices(selected_features), noise_features));
end


%---- Compute the ratio ----
success_ratio = true_counts / n_true;     % Recovery ratio for each experiment

mean_ratio = mean(success_ratio);         % Average ratio

fprintf(['n_samples=%d, n_true=%d, n_noise=%d, nClass=%d,' ...
         'Success ratio of recovering relevant features: %.2f%%\n'], ...
        n_samples, n_true, n_noise, nClass, mean_ratio*100);



f = 1;
end


