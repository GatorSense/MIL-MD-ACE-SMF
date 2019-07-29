function [results] = milmd_ObjFuncOpt(initTargets, pDataBags, nDataBags, parameters)
% Function that evaluates the objective function for multiple instance learning for 
% multiple diverse (MIL MD) algorithm to optimize target signatures.
%Uses Equation 16 on page 4. 
% INPUTS:
% 1) inittargets: the initalized target signatures [n_targets, n_dims]
% 2) nDataBags: a cell array containing the negative bags
% 3) parameters: a structure containing the parameter variables 
% OUTPUTS:
% 1) objValues: The calculated objective values for each combination of
%               KMean cluster representatives.
% 2) targetSigInit: the targets selected that maximize the objective
%                   function. 
% -------------------------------------------------------------------------

% Continue optimizing signatures until max iteration is reached or the
% difference between the obtained target signatures is reached. 
iter = 0;
optTargets = zeros(parameters.maxIter, size(initTargets,2));

% For each target, optimize signature
for k = 1:parameters.numTargets
    continueFlag = 1;
    while (continueFlag == 1 && iter < parameters.maxIter)
        % Display Iteration
        disp(['Iteration: ' num2str(iter)]);
        iter = iter + 1;
        
        % Calculate Objective Function
        cPos = evalPos(pDataBags, initTargets(k,:), parameters);
        cNeg = evalNeg(nDataBags, initTargets(k,:), parameters);
        if k == parameters.numTargets
            cDiv = evalDiv(initTargets(1,:), parameters);
        else
            cDiv = evalDiv(initTargets(k+1,:), parameters);
        end
        cCon = evalCon(initTargets(k,:), parameters, cPos, cNeg, cDiv);
        optTargets(iter) = cPos - cNeg - cDiv - cCon;
        
        %If the target signature is similar enough to previous target signature, then it is done optimizing.
        if(iter ~= 1)
            if sum(optTargets(iter,:) - optTargets(iter-1,:)) < 0.01
                continueFlag = 0;
                disp(['Stopping at Iteration: ', num2str(iter-1)]);
            end
        end
    end
end
% Save out Results
results.optTargets = optTargets;
results.initTargets = initTargets;
end

function [cPos] = evalPos(pDataBags, targetSignature, parameters)
% The gradient-based optimization algorithm for positive bags. This is the
% average of the instances with maximum detetion characterisitics using the
% learned target signatures.
% Equation 17 on page 4
% INPUTS:
% 1) pDataBags: a cell array containing the positive bags
% 2) targetSignature: the initalized target signatures [n_targets, n_dims]
% 3) parameters: a structure containing the parameter variables. variables
%                used in this function number of targets
% OUTPUTS:
% 1) cPos: the gradient value for the positive bags.  

numPBags = size(pDataBags, 2);
pBagSum = zeros(numPBags,size(cell2mat(pDataBags(1)),2));
for bag = 1:numPBags
    
    %Get data from specific bag
    pData = pDataBags{bag};
    pBagMax = zeros(parameters.numTargets, size(pData,2));
    
    % Loop through Targets
    for k = 1:parameters.numTargets
        
        % Get instance with max confidence using a target signature
        if parameters.methodFlag == 0
            % Use SMF as detector
            [~, ~, pBagMaxID, ~, ~] = smf_det(pData', targetSignature(parameters.numTargets,:)',[],[],0);
            pBagMax(k,:) = pData(pBagMaxID(1),:);
        else
            % Use ACE as detector
            [~, ~, pBagMaxID, ~, ~] = ace_det(pData', targetSignature(parameters.numTargets,:)',[],[],0);
            pBagMax(k,:) = pData(pBagMaxID(1),:);
        end
    end
    
    % Sum the instances found with max confidence for each target signature
    pBagSum(bag,:) = sum(pBagMax);   
end

% Get the average max confidence instances
cPos = mean(pBagSum);

end

function [cNeg] = evalNeg(nDataBags, targetSignature, parameters )
% The gradient based optimization algorith for the negative bags. Equation
% 23 on Page 5.
% INPUTS:
% 1) nDataBags: a cell array containing the negative bags
% 2) targetSignature: the initalized target signatures [n_targets, n_dims]
% 3) parameters: a structure containing the parameter variables. variables
%                used in this function - numTargets
% OUTPUTS:
% 1) cNeg: the gradient value for negative bags

numNBags = length(nDataBags);
nBagMean = zeros(1, numNBags);
for bag = 1:numNBags
    
    %Get data from specific bag
    nData = nDataBags{bag};
    nBagVal = zeros(parameters.numTargets,1);
    
    for k = 1:size(targetSignature,1)
        nBagVal(k) = nData * prod(((1 - targetSignature(k,:)'*nData)/2));
    end
    
    nBagMean(bag) = mean(nBagVal);
end

cNeg = mean(nBagMean);
end

function [cDiv] = evalDiv(targetSignature, parameters)
% The gradient-based optimization algorithm for the diversity promoting
% term. Equation 18 on Page 5.
% INPUTS:
% 1) targetSignature: the initalized target signatures [n_targets, n_dims]
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - numTargets
% OUTPUTS:
% 1) cDiv: the gradient of the diversity promoting term

cDiv = -(2/(parameters.numTargets*(parameters.numTargets-1))) * sum(targetSignature);

end

function [cCon] = evalCon(targetSignature, parameters, cPos, cNeg, cDiv)
% Function that calculates the gradient-based optimization for the
% normalization constraint. Equation 24 page 5.
% INPUTS:
% 1) targetSignature: the initalized target signatures [n_targets, n_dims]
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - numTargets, lambda
% 3) cPos: the gradient base value for positive bags
% 4) cNeg: the gradient based value for negative bags
% 5) cDiv: the gradient based value for diversity promoting term
% OUTPUTS:
% 1) cCon: the gradient based value for normalization constraint

% Calculates delta T for the normalization constraint - Equation 25 page 5.
cT = (cPos + cNeg + cDiv) - (((2 * parameters.lambda)/parameters.numTargets)*targetSignature);

% Calculate the signature value
cS = targetSignature'*targetSignature;

if cS >= 1 && cT < 0
    cCon = (2/parameters.numTargets)*targetSignature;
elseif cS <= 1 && cT > 0
    cCon = -(2/parameters.numTargets)*targetSignature;
else
    cCon = 0;
end
end