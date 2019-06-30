function [parameters] = setParams()
% Parameters function for Multiple Instance Learning for Multiple Diverse (MIL MD)
% hyperspectral target characteriziations with 
% Adaptive Cosine Estimator and Spectral Match Filter
% This function builds off the Multi-Target Multiple Instance
% setParameters function. 

addpath('Multi-Target-MI-ACE_SMF/algorithm')
parameters = setParameters();

parameters.initType = 3;   % Set Initialization method used to obtain initalized targets 
                           % 1: searches all positive instances and greedily selects instances that maximize objective. 
                           % 2: K-Means clusters positive instances and greedily selects cluster centers that maximize MT MI objective function.
                           % 3: K-Means clusters positive instances that maximizes MIL MD objective function.

parameters.optimize = 2;   % Determine whether to optimize target signatures
                           % 0: Do not optimize target signatures
                           % 1: Optimize target signatures using MT MI  
                           % 2: Optmize target signatures using MIL MD  
                           
parameters.lambda = 1;     % Constant (greater than 0) that weights the constraint in the objective function
end