function [all_J, all_logL, modelname] = ising_neurons_L1reg(datafile, model, nchunks)
%ISING_NEURONS_L1REG Implements minimum probability flow learning for
% an Ising model fit to neural data with the connectivity matrix of 
% your choice.
%
%  [all_J, all_logL, modelname] = ISING_NEURONS_L1REG(datafile), 
%  where datafile is a mat file containing a binary matrix of
%  spike events, a binary matrix of stimulus conditions, and the 
%  time bin size, returns the Ising model coupling matrices 
%  all_J, the log-likelihood all_logL of a held-out validation 
%  set given these couplings, and the modelname (e.g. "fully 
%  connected").
%
%  [all_J, all_logL, modelname] = ISING_NEURONS_L1REG(datafile, model, nchunks)
%  allows you to specify a model number that describes which couplings
%  should be set to zero, if you are assuming a set connectivity
%  matrix. By default, model=0, which assumes a fully connected model
%  where all possible pairwise couplings are fitted.  See select_zero_nodes.m
%  for details on other available models.  nchunks is the number of cross-validation
%  iterations so that you can determine reasonable boundaries on the log-likelihoods
%  (nchunks = 10 by default).
%
%   EXAMPLE:
%       datafile = 'sample_data.mat';
%       model = 0;
%       nchunks = 10; 
%       [all_J, all_logL, modelname] = ISING_NEURONS_L1REG(datafile, model, nchunks);
%       imagesc(all_J(:,:,1), [-2 2]); % plot coupling from 1st cross validation iteration
%
%  See also SELECT_ZERO_NODES, K_DK_ISING_L1REG
%

%  References:
%    Hamilton LS, Sohl-Dickstein J, Huth AG, Carels VM, Bao S (2013). Optogenetic
%       Activation of an Inhibitory Network Enhances Functional Connectivity in
%       Auditory Cortex.  Neuron (in press).
%    Sohl-Dickstein J, Battaglino P, DeWeese M (2011).  New method for parameter
%       estimation in probabilistic models: minimum probability flow.  Physical
%       Review Letters, 107:22 p220601.
%
%  Uses 3rd party code minFunc developed by Mark Schmidt at the Computer Science
%  Department at the Ecole Normale Superieure, see his website with code at
%  http://www.di.ens.fr/~mschmidt/Software/minFunc.html
%
%
% Copyright ©2013 [See notes below]. The Regents of the University of
% California (Regents).  All Rights Reserved.  Permission to use, copy
% and distribute this software and its documentation for educational,
% research, and not-for-profit purposes, without fee and without a 
% signed licensing agreement, is hereby granted, provided that the above 
% copyright notice, this paragraph and the following two paragraphs appear in 
% all copies, modifications, and distributions.  Contact
%    The Office of Technology Licensing, UC Berkeley,
%    2150 Shattuck Avenue, Suite 510, 
%    Berkeley, CA 94720-1620,
%    (510) 643-7201,
% for commercial licensing opportunities.

% Created by Liberty S. Hamilton, Jascha Sohl-Dickstein, and Alexander Huth, 
% at the University of California, Berkeley.
% Based on code written by Jascha Sohl-Dickstein (2009) available at 
% https://github.com/Sohl-Dickstein/Minimum-Probability-Flow-Learning 
%
% IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
% SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
% ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
% IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
% FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION,
% IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
% TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
%
   
    if nargin < 2
        model = 0; % assume fully connected model 
        nchunks = 10; % 10 cross-validation iterations
    end

    % load the data matrices. [datafile] should contain three variables with the
    % following names and contents: 
    %   spk:        an nneurons x t matrix (nneurons = number of neurons/electrode sites, t = number of time bins)
    %   stim:       an nstims x t matrix (nstims = number of stimulus features, t = number of time bins)
    %   bin_size:   the size of each time bin in seconds (in our sample dataset, bin_size = 0.005 (5 ms))
    mydata = load(datafile);
    nneurons = size(mydata.spk, 1); % number of neurons or electrode/recording sites
    nstims = size(mydata.stim, 1);  % number of stimuli or number of stimulus features
    t = size(mydata.spk, 2);        % number of time bins
    X = vertcat(mydata.stim, mydata.spk); % concatenate stimulus and spike matrices
   
    % Initialize variables for the cross-validated log-likelihood measurements
    % and the cross-validated coupling matrices.
    all_logL = zeros(nchunks,1);
    all_J = zeros(nneurons, nneurons+nstims, nchunks);
    
    % Set the length of the validation set to be 10% of the total length
    val_len = floor(t/10);

    % When we randomize, we want to shuffle trials if possible to make sure
    % that the noise level does not affect our model.  To do this, we choose
    % a [spanlen] as the segment that will be randomized from the data matrices.
    % A good choice here would be a multiple of your trial length (for our sample
    % data, the trial length is 500 ms, and the bin size is 5 ms, so we can shuffle
    % the trials using a span length = 200. 
    spanlen = 200; % 200 samples @ 5ms/sample = 1s/span
    nspans = floor(floor(t/spanlen)/nchunks)*nchunks; % truncate data if not exactly divisible

    % randomize which data validation is performed on, so it's not all from
    % the same time period, since behavior may change with time and we want
    % the behavior on the validation set to be qualitatively the same as
    % the behavior on the training set
    span_inds = reshape(1:(spanlen*nspans), spanlen, [])';
   
    % randomly permute the order of each of these segments
    chunk_span_inds = reshape(randperm(nspans), nchunks, []);
    chunk_inds = {};
    for ii = 1:nchunks
        smat = span_inds(chunk_span_inds(ii,:),:)';
        chunk_inds{ii} = smat(:);
    end

    % set options for minFunc function minimization
    T = nneurons+nstims; % number of couplings to fit
    nsamples = size(X,2); % number of training samples
    maxlinesearch = 1000; % this number is excessive just to be safe! Learning works fine if this is just a few hundred
    independent_steps = 10*nneurons; % the number of Gibbs sampling steps to take between samples
    minf_options = [];
    minf_options.display = 'iter';  % set to 'none' to make this faster
    minf_options.maxFunEvals = maxlinesearch;
    minf_options.maxIter = maxlinesearch;
        
    run_checkgrad = 0; % set to 1 to check that the derivative being calculated in K_dK_ising.m 
                       % is correct -- usually you will want this off, as it is mainly for debugging purposes.
    if run_checkgrad
        nsamples = 2; 
        minf_options.DerivativeCheck = 'on';
    end

    % Perform cross validation
    for k = 1:nchunks
        fprintf(1,'\n.............Running cross validation iteration %d of 10..........\n',k);
        smat = span_inds(chunk_span_inds(k,:),:)';
        cinds = smat(:); % validation data indices
        
        X_val = X(:,cinds); % validation data
        X_train = X;
        X_train(:,cinds) = []; % training data
        
        description = sprintf('d=%d, %d samples, %d learning steps',T,nsamples,maxlinesearch);
        fprintf(1,'%s\n',description);

        % randomly initialize the parameter matrix we're going to try to learn
        % note that the bias units lie on the diagonal of J
        Jnew_n = randn(nneurons, nneurons) / sqrt(nneurons) / 100; % site-to-site coupling
        Jnew_n = (Jnew_n + Jnew_n')/2; % make weight matrix symmetric
        Jnew_s = randn(nneurons, nstims)/sqrt(nneurons)/100; % stimulus-to-site coupling

        % choose model connectivity (see select_zero_nodes.m or make your own function here)
        [Jnew_all, zeronodes, modelname] = select_zero_nodes(model, Jnew_s, Jnew_n);

        % choose lambda (ideally with another round of cross-validation on the training set)
        % but for simplicity we choose a value here.
        lambda = 5.9948e-05;
        
        % perform parameter estimation using L1-regularized Ising model fit with MPF
        fprintf(1, '\nRunning minFunc for up to %d learning steps...\n', maxlinesearch );
        
        time_elapsed = tic();
       
        Jnew_all_fit = minFunc( @K_dK_ising_L1reg, Jnew_all(:), minf_options, X_train, nneurons, nstims, lambda, zeronodes);
        Jnew_all_fit(zeronodes) = 0; % force couplings to zero based on connectivity provided
        Jnew_all_fit = reshape(Jnew_all_fit, size(Jnew_all));

        time_elapsed = toc(time_elapsed);

        fprintf(1, 'parameter estimation in %f seconds \n', time_elapsed );
        
        % add the new coupling matrix to the 3D matrix of Js for each cross validation iteration
        all_J(:,:,k) = -Jnew_all_fit; % flip the sign for plotting so that positive couplings indicate sites fire together

        % compute log likelihood of the validation data given the model
        [loglik] = L_dL_ising(Jnew_all_fit, X_val, nneurons, nstims, zeronodes);
        
        % set log-likelihood for this cross validation iteration
        all_logL(k) = loglik; 

        fprintf(1,'Log-likelihood for %s: %d\n', modelname, loglik);

    end
