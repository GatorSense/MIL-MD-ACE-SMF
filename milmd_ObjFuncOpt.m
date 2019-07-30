function [results] = milmd_ObjFuncOpt(initTargets, pDataBags, nDataBags, parameters, dataInfo)
% Function that evaluates the objective function for multiple instance learning for 
% multiple diverse (MIL MD) algorithm to optimize target signatures.
% Uses Equation 16 on page 4. 
% INPUTS:
% 1) initTargets: the initalized target signatures [n_targets, n_dims]
% 2) nDataBags: a cell array containing the negative bags
% 3) parameters: a structure containing the parameter variables 
% 4) dataInfo: a structure that contains information about the background.
%              Not used in the function, but save variables out to results
%              1) mu: background mean [1, n_dim]
%              2) cov: background covariance [n_dim, n_dim]
%              2) invcov: inverse background covariance, [n_dim, n_dim]
%              3) D: a singular value decomposition of matrix A, such that A = U*D*V'.
%              4) V: a singular value decomposition of matrix A, such that A = U*D*V'.
%              5) U: a singular value decomposition of matrix A, such that A = U*D*V'.
% OUTPUTS:
% 1) objValues: The calculated objective values for each combination of
%               KMean cluster representatives.
% 2) targetSigInit: the targets selected that maximize the objective
%                   function. 
% -------------------------------------------------------------------------

% Set up variable to hold the final optimized target signatures
finalOptTargets = zeros(parameters.numTargets, size(initTargets,2));

% For each target, optimize signature
for k = 1:parameters.numTargets
    continueFlag = 1;
    iter = 0;
    inTarget = initTargets(k,:); % Set the first signature to the initialized target
    optTargets = zeros(parameters.maxIter, size(initTargets,2)); % Store each iterations optimized signatures
    
    % Continue optimizing signatures until max iteration is reached or the
    % difference between the obtained target signatures is reached. 
    while (continueFlag && iter <= parameters.maxIter)
        % Display Iteration
        disp(['Iteration: ' num2str(iter)]);
        iter = iter + 1;
        
        % Calculate the gradient value for positive bags - Equation 18 Page 5
        cPos = evalPos(pDataBags, inTarget, parameters);
        
        % Calculate the gradient value for negative bags - Equation 23 Page 5
        cNeg = evalNeg(nDataBags, inTarget);
        
        % Calcualte the gradient value for diversity promoting term - Equation 18 on Page 5.
        range = 1:parameters.numTargets;
        range(k) = [];
        cDiv = evalDiv(initTargets(range,:), parameters);
        
        % Calcualte the gradient value for constraint term - Equation 24 & 25 Page 5
        cCon = evalCon(inTarget, parameters, cPos, cNeg, cDiv);
        
        % Calculate Graident-based Objective Function - Equation 16 Page 4
        optTargets(iter,:) = cPos - cNeg - parameters.alpha*cDiv - parameters.lambda*cCon; 
        
        % Update the target with the current optimized target
        inTarget = optTargets(iter,:);
        
        %If the target signature is similar enough to previous target signature, then it is done optimizing.
        %the paper does not specify what value or method they used for a stopping criterion 
        if(iter ~= 1)
            if abs(mean(optTargets(iter,:) - optTargets(iter-1,:))) < 0.001
                continueFlag = 0;
                finalOptTargets(k,:) = optTargets(iter-1,:); % Set the final signature for this target
                disp(['For Target ', num2str(k), ' Stopping at Iteration: ', num2str(iter-1)]);
            end
        end
        if iter == parameters.maxIter
            disp(['For Target ', num2str(k), ' max iteration reached.']);
            finalOptTargets(k,:) = optTargets(iter,:); % Set the final signature for this target
        end
    end
end

% Undo whitening
finalOptTargetsUW = (finalOptTargets*dataInfo.D^(1/2)*dataInfo.V');
for k = 1:parameters.numTargets
    finalOptTargetsUW(k,:) = finalOptTargetsUW(k,:)/norm(finalOptTargetsUW(k,:));
end
initTargetsUW = (initTargets*dataInfo.D^(1/2)*dataInfo.V');
for k = 1:parameters.numTargets
    initTargetsUW(k,:) = initTargetsUW(k,:)/norm(initTargetsUW(k,:));
end

% Save out Results
results.optTargets = finalOptTargetsUW;
results.initTargets = initTargetsUW;
results.b_mu = dataInfo.mu;
results.sig_inv_half = dataInfo.invcov;
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
             
    % Get instance with max confidence using a target signature
    if parameters.methodFlag == 0
        % Use SMF as detector
        [~, ~, pBagMaxID, ~, ~] = smf_det(pData', targetSignature',[],[],0);
        pBagMax = pData(pBagMaxID(1),:);
    else
        % Use ACE as detector
        [~, ~, pBagMaxID, ~, ~] = ace_det(pData', targetSignature',[],[],0);
        pBagMax = pData(pBagMaxID(1),:);
    end
    
    % Sum the instances found with max confidence for each target signature
    pBagSum(bag,:) = pBagMax/parameters.numTargets;   
end

% Get the average max confidence instances
cPos = mean(pBagSum);

end

function [cNeg] = evalNeg(nDataBags, targetSignature)
% The gradient based optimization algorith for the negative bags. Equation
% 23 on Page 5.
% INPUTS:
% 1) nDataBags: a cell array containing the negative bags
% 2) targetSignature: the initalized target signatures [n_targets, n_dims]
% OUTPUTS:
% 1) cNeg: the gradient value for negative bags

numNBags = length(nDataBags);
nBagMean = zeros(numNBags,size(cell2mat(nDataBags(1)),2));
for bag = 1:numNBags
    
    %Get data from specific bag
    nData = nDataBags{bag};
    
    nBagVal = nData.* prod(((1 - targetSignature.*nData)/2));
    
    nBagMean(bag,:) = mean(nBagVal);
end

cNeg = mean(nBagMean);
end

function [cDiv] = evalDiv(targetSignature, parameters)
% The gradient-based optimization algorithm for the diversity promoting
% term. Equation 18 on Page 5.
% INPUTS:
% 1) targetSignature: the initalized target signatures [n_targets, n_dims]
%    NOT the signature currently being evaluated ALL Others
% 2) parameters: a structure containing the parameter variables. variables
%                used in this function - numTargets
% OUTPUTS:
% 1) cDiv: the gradient of the diversity promoting term

if size(targetSignature,1) > 1
    cDiv = -(2/(parameters.numTargets*(parameters.numTargets-1))) * sum(targetSignature);
else
    cDiv = -(2/(parameters.numTargets*(parameters.numTargets-1))) * targetSignature;
end

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

if sum(sum(cS >= 1)) == 0 && sum(cT) < 0
    cCon = (2/parameters.numTargets)*targetSignature;
elseif sum(sum(cS <= 1)) == 0 && sum(cT) > 0
    cCon = -(2/parameters.numTargets)*targetSignature;
else
    cCon = 0;
end
end