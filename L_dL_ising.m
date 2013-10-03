function [logL, dlogL] = L_dL_ising( J, X, nneurons, nstims, zeronodes )
%L_DL_ISING calculates the log-likelihood of your data, X, given Ising
% model couplings, J.
%
% [logL, dlogL] = L_DL_ISING(J, X, nneurons, nstims, zeronodes)
% where J is the horizontally concatenated stimulus-to-site and site-to-site
% coupling matrix, X is the concatenated stimulus and spike conditions over
% time (nneurons+nstims x time points), nneurons is the number of electrode
% channels or recording sites, nstims is the number of stimuli, and
% zeronodes are the indices of couplings that should be set to zero based
% on your connectivity model. Returns the log-likelihood and the gradient
% of the log-likelihood of your data given the model.
%
%  References:
%    Hamilton LS, Sohl-Dickstein J, Huth AG, Carels VM, Bao S (2013). Optogenetic
%       Activation of an Inhibitory Network Enhances Functional Connectivity in
%       Auditory Cortex.  Neuron (in press).
%  See also ISING_NEURONS_L1REG, SELECT_ZERO_NODES, K_DK_ISING_L1REG

% Copyright ©2013 Liberty S. Hamilton and Jascha Sohl-Dickstein. The Regents 
% of the University of California (Regents).  All Rights Reserved.  Permission
% to use, copy, modify, and distribute this software and its documentation for
% educational, research, and not-for-profit purposes, without fee and without a 
% signed licensing agreement, is hereby granted, provided that the above 
% copyright notice, this paragraph and the following two paragraphs appear in 
% all copies, modifications, and distributions.  Contact
%    The Office of Technology Licensing, UC Berkeley,
%    2150 Shattuck Avenue, Suite 510, 
%    Berkeley, CA 94720-1620,
%    (510) 643-7201,
% for commercial licensing opportunities.
%
% Created by Liberty S. Hamilton and Jascha Sohl-Dickstein, University of California, Berkeley.
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


    J(zeronodes) = 0;        
        
    [ndims, Treal] = size( X ); % number of coupling dimensions (nstims+nneurons) x number of time points
    
    S = X(1:nstims,:);      % sound condition at each time point
    Xn = X((nstims+1):end,:);   % spiking activity at each time point
    
    J = reshape(J,[nneurons,ndims]); % the full coupling matrix
    
    Js = J(:,1:nstims); % the stimulus-to-site coupling matrix
    Jn = J(:,(nstims+1):end); % the site-to-site coupling matrix 
    Jn = (Jn + Jn')/2; % make neuron couplings symmetric
    
    % get the energy for all possible states
    E_data = sum(Xn.*(Jn*Xn)) + sum(S.*(Js'*Xn));

    % this will hold the log partition function for each observation.  Note that because the distribution is conditioned on the light or sound case, the normalization constant is a function of the light or sound case
    logZ_data = zeros( 1, Treal );
        
    % fill in all possible binary patterns for neurons
    X_all = zeros( nneurons, 2^nneurons );
    for d = 1:nneurons
        X_all(d,:) = bitget( 0:2^nneurons-1, d );
    end

    % initialize the log likelihood and derivatives
    logL = 0;
    dJn = 0;
    dJs = 0;
    
    % fill in all possible stimulus conditions
    soundcond = [eye(nstims)]; 
    soundcond = [zeros(size(soundcond,1),1),soundcond]; % add a "no stimulus" condition also
   
    % pre-allocate some matrix multiplications we'll need later
    Xsum = sum(X_all.*(Jn*X_all));
    % step through all the possible stimulus conditions
    for si=1:size(soundcond,2)
        St = soundcond(:,si);
        % find all the data states that had this condition
        sound_diff = bsxfun( @plus, S,  - St );
        matching_inds = find( sum(abs(sound_diff) ) == 0 );
        % don't bother calculating logZ if we won't use it anywhere
        if length(matching_inds)>0
            % get the energy for all possible states for this light and sound condition
            E_all = sum(X_all.*(Jn*X_all)) + St(:)'*(Js'*X_all);

            potential_all = exp( -E_all ); % caclulate the potential function for all patterns    
            Z = sum( potential_all ); % calculate the partition function

            % increment the log likelihood for this case
            logL = logL - sum(E_data(matching_inds)) - length(matching_inds)*log(Z);

            % and similarly increment the gradients
            Xnmatch = Xn(:,matching_inds);
            Smatch = S(:,matching_inds);

            % neuron to neuron gradient
            dJn = dJn - Xnmatch*Xnmatch' + length(matching_inds)/Z * bsxfun( @times, X_all, potential_all ) * X_all';
            % sound to neuron gradient
            dJs = dJs - Xnmatch*Smatch' + length(matching_inds)/Z * (X_all * potential_all(:)) * St';

            % we could compute many of the matrix multiplications above outside this loop, or only once in the loop, and make it go faster
        end

    end    
    
    logL = logL / Treal; % average log likelihood
    
    dJn = (dJn + dJn') / 2; 
    dlogL = [dJs(:); dJn(:)] / Treal;

    dlogL(zeronodes) = 0;    
    
    % We want the NEGATIVE log likelihood, since maximizing
    % the likelihood is akin to minimizing the negative 
    % log-likelihood
    logL = -logL;
    dlogL = -dlogL;

