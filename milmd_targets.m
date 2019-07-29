function [results] = milmd_targets(data, parameters)
% Determines target signatures using multiple instance learning for 
% multiple diverse (MIL MD) algorithm.
% Susan Meerdink
% June 2019
% Algorithm development found in:
% P. Zhong, Z. Gong, and J. Shan, “Multiple Instance Learning for Multiple 
% Diverse Hyperspectral Target Characterizations,” IEEE Trans. Neural Networks
%
% INPUTS:
% data.dataBags: cell list with positive and negative bags [1, n_Bags]
%                Each cell contains a single bag in the form a [n_samples, n_Bands]
% 
% data.labels: labels for dataBags [1, n_Bags]
%              * the labels should be a row vector with labels corresponding to the 
%              * parameters.posLabel and parameters.negLabel where a posLabel corresponds
%              * to a positive bag and a negLabel corresponds to a negative bag.
%              * The index of the label should match the index of the bag in dataBags 
% parameters:
%     K: number of target types in training bags
%     methodFlag: (boolean) Use ACE (1) or SMF (0) as similarity measure    
%     globalBackgroundFlag: (boolean) estimate the background mean and covariance from all data (1) or just negative bags (0)
%     posLabel: what denotes a positive bag's label. ex) 1
%     negLabel: what denotes a negative bag's label. ex) 0
%     alpha: Uniqueness term weight in objective function, set to 0 if you do not want to use the term
%
% OUTPUTS:
% results: a structure containing the following variables:
%             1) b_mu: background mean [1, n_dim]
%             2) b_cov: background covariance [n_dim, n_dim]
%             3) sig_inv_half: inverse background covariance, [n_dim, n_dim]
%             4) initTargets: the initial target signatures [n_targets, n_dim]
%             5) methodFlag: value designating which method was used for similarity measure
%             6) numTargets: the number of target signatures found
%             7) optTargets: the optimized target signatures [n_opttargets,
%                n_dim] Might have fewer targets then initTargets
% ------------------------------------------------------------------------

% 1) Whiten Data 
addpath('Multi-Target-MI-ACE_SMF/algorithm')
[dataBagsWhitened, dataInfo] = whitenData(data, parameters);
pDataBags = dataBagsWhitened.dataBags(data.labels ==  parameters.posLabel);
nDataBags = dataBagsWhitened.dataBags(data.labels == parameters.negLabel);

% 2) Initialize target signatures and maximize objective function
if parameters.initType == 1
    % Initialize by searching all positive instances and greedily selects
    % instances that maximizes objective function. 
    [initTargets, initTargetLocation, originalPDataBagNumbers, initObjectiveValue] = init1(pDataBags, nDataBags, parameters);
elseif parameters.initType == 2
    % Initialize by K-means cluster centers and greedily selecting cluster
    % center that maximizes objective function. 
    [initTargets, objectiveValues, C] = init2(pDataBags, nDataBags, parameters);
elseif parameters.initType == 3
    % Initalize by k-means cluster centers and selecting instance closest
    % to cluster center. Finds the combination of instances that maximizes objective function. 
    [initTargets, initObjectiveValue, clustCenters] = init_3(pDataBags, nDataBags, parameters);
else
    disp('Invalid initType parameter. Options are 1, 2, or 3.')
    return
end

% 3) Optimize target concepts 
if parameters.optimize == 0
    % Do not optimize targets - will return the initialized targets
    results = nonOptTargets(initTargets, parameters, dataInfo);
elseif parameters.optimize == 1
    % optmize targets using MT MI methodology
    results = optimizeTargets(data, initTargets, parameters);
elseif parameters.optimize == 2
    % optimize targets using MIL MD methodology
    results = milmd_ObjFuncOpt(initTargets, pDataBags, nDataBags, parameters);
else
    disp('Invalid optimize parameter. Options are 0, 1, or 2.')
    return
end

end

function [initTargets, objectiveValues, Ctargets] = init_3(pDataBags, nDataBags, parameters)
% Function that initializes target signatures using K Means and maximize 
% MIL MD objective function. In this function, you need to start with more
% clusters than desired target signatures.
% Inputs:
% 1) pDataBags: a structure containing the positive bags (already whitened) 
% 2) nDataBags: a structure containing the negative bags (already whitened)
% 3) parameters: parameters variable where the following parameters are
%                used - numClusters, maxIter, numTargets,  
% OUTPUTS:
% 1) initTargets: matrix containing the initial targets [n_targets, n_dim]
% 2) objectiveValues: 
% 3) Ctargets: the instances closest to the K-Means cluster centers 
% ------------------------------------------------------------------------
disp('Clustering Data');

% Get K-Means cluster centers (C)
pData = vertcat(pDataBags{:});
[~, ~, ~, Cdist] = kmeans(pData, min(size(pData, 1), parameters.numClusters), 'MaxIter', parameters.maxIter);

% Get instance closest to cluster center, use as representative of the cluster
[~,idx] = min(Cdist);
Ctargets = pData(idx,:);

% Get targets that maximize objective function
[objectiveValues, initTargets] = milmd_ObjFuncInit(Ctargets, pDataBags, nDataBags, parameters);

end


