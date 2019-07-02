function [objValues, pBagMaxConfSig, pBagMaxConf, pBagMaxConfIndex] = milmd_ObjFuncInit(Ctargets, pDataBags, nDataBags, parameters)
% Function that evaluates the objective function for multiple instance learning for 
% multiple diverse (MIL MD) algorithm. Uses Equation 15 on page 4. 
% INPUTS:
% 1) Ctargets: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) nDataBags: a cell array containing the negative bags
% 3) parameters: a structure containing the parameter variables 
% OUTPUTS:
% 1)
% -------------------------------------------------------------------------

% Find all combinations of KCluster target signatures
index = 1:size(Ctargets,1);
combo = combnk(index, parameters.numTargets);
objValues = zeros(size(combo,1),1);

% Loop through all combinations
for c = 1:size(combo,1)
    targetSignature = Ctargets(combo(c,:),:);
    
    % Calculate Objective Function
    cPos = evalC1(pDataBags, targetSignature, parameters);
    cNeg = evalC2(targetSignature, nDataBags);
    cUni = evalUnique(targetSignature, parameters);
    cCon = evalConstraint(targetSignature, parameters);
    objValues(c) = cPos - cNeg - cUni - cCon;
end

end

function [cPos] = evalC1(pDataBags, targetSignatures, parameters)
% Average of the instances with maximum detection characteristics using 
% the learned target signatures
% INPUTS:
% 1) pDataBags: a cell array containing the positive bags
% 2) targetSignature: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 3) parameters: a structure containing the parameter variables. variables
%                used in this function 
% OUTPUTS:
% 1) cPos: the computed term for the first term in the objective function.

% Loop through positive bags
numPBags = size(pDataBags, 2);
pBagSum = zeros(numPBags,1);
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
    
    pBagSum(bag) = sum(pBagMax);
end

%Calculate actual objectiveValue
cPos = mean(pBagSum(:));
end

function [cNeg] = evalC2(targetSignature, nDataBags)
% Function that calculates the average of the max negative instances
% INPUTS:
% 1) targetSignature: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) nDataBags: a cell array containing the negative bags
% OUTPUTS:
% 1) cNeg: the computed term for the second term in the objective function.

%Get average confidence of each negative bag
numNBags = length(nDataBags);
nBagMaxConf = zeros(1, numNBags);
for bag = 1:numNBags
    
    %Get data from specific bag
    nData = nDataBags{bag};
    
    for k = 1:size(targetSignature,1)
        %Confidence (dot product) of a sample across all other samples in nData, data has already been whitened
        nBagMaxConf(bag) = max(nData.*targetSignature(k,:),2);
    end
    
end

% Calculate objective value
cNeg = mean(nBagMaxConf(:));

end

function [cU] = evalUnique(targetSignatures, parameters)
% Function that calculates the uniqueness of selected target signatures
% with alpha is configurable to allow for more different or similar target
% signatures.
% INPUTS:
% 1) targetSignature: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - alpha, numTargets
% OUTPUTS:
% 1) cU: the computed term for the third term in the objective function.

similarities = sum(targetSignatures.*targetSignatures, 2);
top = (2*parameters.alpha)/(parameter.numTargets*(parameter.numTargets-1));
cU = top * similarities;

end

function [cCon] = evalConstraint(targetSignature, parameters)
% Function that calculates the constraint term of the objective function.
% The introduction of this constraint into the maximization framework aids
% in the prevention of target signatures being arbitrarily large. 
% INPUTS:
% 1)targetSignature: the instances closest to the K-Means cluster centers [n_targets, n_dims]
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - lambda, numTargets
% OUTPUTS:
% 1) cCon: the computed term for the fourth term in the objective function.

const = zeros(parameters.numTargets,1);
for k = 1:parameters.numTargets
    const(k) = abs((targetSignature(k)'.*targetSignature(k)) - 1);
end
cCon = (parameters.lamba / parameter.numTargets) * sum(const);

end

