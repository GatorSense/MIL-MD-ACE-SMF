function [objValues, targetSigInit] = milmd_ObjFuncInit(Ctargets, pDataBags, nDataBags, parameters)
% Function that evaluates the objective function for multiple instance learning for 
% multiple diverse (MIL MD) algorithm. Uses Equation 15 on page 4. 
% INPUTS:
% 1) Ctargets: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) nDataBags: a cell array containing the negative bags
% 3) parameters: a structure containing the parameter variables 
% OUTPUTS:
% 1) objValues: The calculated objective values for each combination of
%               KMean cluster representatives.
% 2) targetSigInit: the targets selected that maximize the objective
%                   function. 
% -------------------------------------------------------------------------

% Find all combinations of KCluster target signatures
index = 1:size(Ctargets,1);
combo = combnk(index, parameters.numTargets);
objValues = zeros(size(combo,1),1);

% Loop through all combinations of targets
for c = 1:size(combo,1)
    % Set up current set of target signature combinations
    targetSignatures = Ctargets(combo(c,:),:);
    
    % Calculate the positive bags term for this combination of targets
    cPos = evalPos(targetSignatures, pDataBags, parameters);
    
    % Calculate the negative bags term for this combination of targets
    cNeg = evalNeg(targetSignatures, nDataBags);
    
    % Calculate the unique or diversity promoting term for this combination of targets
    cUni = evalUnique(targetSignatures, parameters);
    
    % Calculate the constraint term for this combination of targets
    cCon = evalConstraint(targetSignatures, parameters);
    
    % Calculate Objective Function for this combination of targets
    objValues(c) = cPos - cNeg - cUni - cCon;
end

% Find the combination of targets that maximizes the objective function
[~,targetIndex] = max(objValues);
targetSigInit = Ctargets(combo(targetIndex,:),:);

end

function [cPos] = evalPos(targetSignatures, pDataBags, parameters)
% Average of the instances with maximum detection characteristics using 
% the learned target signatures
% First term in Equation 15 on page 4.
% INPUTS:
% 1) targetSignatures: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) pDataBags: a cell array containing the positive bags
% 3) parameters: a structure containing the parameter variables. variables
%                used in this function are number of Targets
% OUTPUTS:
% 1) cPos: the positive bag term for the objective function.

% Set up variables
numPBags = size(pDataBags, 2);
pBagSum = zeros(numPBags,1);

% Loop through positive bags
for bag = 1:numPBags
    
    %Get data from specific bag
    pData = pDataBags{bag};
    pBagMax = zeros(parameters.numTargets,1);
    
    % Loop through Targets
    for k = 1:parameters.numTargets
        %Confidences (dot product) of a sample across all other samples in pData, data has already been whitened
        pConf = sum(pData.*targetSignatures(k,:), 2);

        %Get max confidence for this bag
        pBagMax(k) = max(pConf)/parameters.numTargets;
    end
    
    % Calculate the positive bags confidence sum
    pBagSum(bag) = sum(pBagMax);
end

%Calculate actual objectiveValue
cPos = mean(pBagSum(:));

end

function [cNeg] = evalNeg(targetSignatures, nDataBags)
% Function that calculates the average of the max negative instances.
% Get average max confidence of each negative bag.
% Second term in Equation 15 on page 4.
% INPUTS:
% 1) targetSignatures: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) nDataBags: a cell array containing the negative bags
% OUTPUTS:
% 1) cNeg: the negative bag term in the objective function.

% Set up Variables
numNBags = length(nDataBags);
nBagMean = zeros(1, numNBags);

% Loop through negative bags
for bag = 1:numNBags
    
    %Get data from specific bag
    nData = nDataBags{bag};
    
    % Find instance/pixel with the max confidence
    maxDotProd = max(targetSignatures*nData'); %max{k} s^^(k)T*x^^n
    nBagMean(bag) = mean(maxDotProd);
end

% Calculate objective value by calculating the mean from all negative bags
cNeg = mean(nBagMean);

end

function [cU] = evalUnique(targetSignatures, parameters)
% Function that calculates the uniqueness or diversity of selected target signatures
% with alpha is configurable to allow for more different or similar target
% signatures.
% Third term in Equation 15 on page 4.
% INPUTS:
% 1) targetSignatures: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - alpha, numTargets
% OUTPUTS:
% 1) cU: the diversity promoting term in the objective function.

% Get all combinations of targets in order to determine similarity 
numTargets = parameters.numTargets;
allSpectraComb = combnk(1:numTargets, 2);
numCombinations = size(allSpectraComb,1);
cosSimilarity = 0;

% Loop through combinations calculating the target signature similarities
for k = 1:numCombinations
     cosSimilarity = cosSimilarity + targetSignatures(allSpectraComb(k,1),:)*targetSignatures(allSpectraComb(k,2),:)';    
end
coeff = (2*parameters.alpha)/(numTargets*(numTargets-1));
cU = coeff * cosSimilarity;

end

function [cCon] = evalConstraint(targetSignatures, parameters)
% Function that calculates the constraint term of the objective function.
% The introduction of this constraint into the maximization framework aids
% in the prevention of target signatures being arbitrarily large. 
% Fourth term in Equation 15 on page 4.
% INPUTS:
% 1)targetSignature: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - lambda, numTargets
% OUTPUTS:
% 1) cCon: the computed term for the fourth term in the objective function.

targetSigSquared = diag(targetSignatures*targetSignatures');
lagrangeTerm = abs(targetSigSquared - 1);
cCon = (parameters.lambda / parameters.numTargets) * sum(lagrangeTerm);

end

