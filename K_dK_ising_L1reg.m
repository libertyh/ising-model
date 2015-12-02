function [K, dK] = K_dK_ising_L1reg( J, X, nneurons, nstims, lambda, zeronodes)
%K_DK_ISING_L1REG Fits the minimum probability flow learning objective
% function for an Ising model fit to neural data with the connectivity
% matrix of your choice. 
%
%  [K, dK] = K_DK_ISING_L1REG(J, X, nneurons, nstims, lambda, zeronodes), 
%  where J is the horizontally concatenated stimulus-to-site and site-to-site
%  coupling matrix, X is the concatenated stimulus and spike conditions over
%  time (nneurons+nstims x time points), nneurons is the number of electrode
%  channels or recording sites, nstims is the number of stimuli, lambda is 
%  the L1 regularization parameter, and zeronodes are the indices of couplings
%  that should be set to zero based on your connectivity model.
%  Returns the MPF objective function value K and the gradient of the function,
%  dK.  This function is minimized using 3rd party code minFunc by Mark Schmidt
%  (see details in ising_neurons_L1reg.m)
%
%  References:
%    Hamilton LS, Sohl-Dickstein J, Huth AG, Carels VM, Bao S (2013). Optogenetic
%       Activation of an Inhibitory Network Enhances Functional Connectivity in
%       Auditory Cortex.  Neuron. 2013 Nov 20;80(4):1066-76. 
%       doi: 10.1016/j.neuron.2013.08.017.
%    Sohl-Dickstein J, Battaglino P, DeWeese M (2011).  New method for parameter
%       estimation in probabilistic models: minimum probability flow.  Physical
%       Review Letters, 107:22 p220601.
%
% This code is written under the greatly accelerating assumption that 
% samples in the data vector differ from each other by more than one 
% bit flip (see http://arxiv.org/abs/0906.4779). This is nearly always 
% true for large systems, but means you'll get funny answers if you do
% a test run with a system of only a few units with lots of data.
%
%  See also ISING_NEURONS_L1REG, SELECT_ZERO_NODES

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

    if nargin<6
       zeronodes=[];
    end
    
    J(zeronodes) = 0;
    
    [ndims, Treal] = size( X ); % ndims should be equal to nneurons + nstims, Treal is the # of time points
    
    S = X( 1:nstims, : );  % stimulus condition
    Xn = X( (nstims+1):end, : ); % spike data

    J = reshape(J,[nneurons,ndims]); % the full coupling matrix
    
    Js = J(:, 1:nstims); % the stimulus-to-site coupling matrix
    Jn = J(:, (nstims+1):end); % the site-to-site coupling matrix 
    
    Jn = (Jn + Jn')/2; % make couplings symmetric
    
    Y = Jn*Xn; % sum over n of x_i * Jn_{in}
    diagJn = diag(Jn);
    
    % XnotX contains (X - [bit flipped X])
    XnotX = 2*Xn-1;
    
    % Kfull is a [ndims, Treal] matrix containing the contribution 
    % to the objective function from flipping each bit in the rows, 
    % for each datapoint in the columns
    Kfull = exp( XnotX .* Y - (1/2)*diagJn(:,ones(1,Treal)) + (1/2)*Js*S.*XnotX );
    K = sum(Kfull(:));
   
    % gradient from site-to-site
    dJ_ab = Kfull.*XnotX * Xn' - (1/2)*diag( sum(Kfull, 2) );
    dJ_ab = (dJ_ab + dJ_ab')/2;

    % gradient from stimulus-to-site
    dJ_cb = (1/2)*Kfull.*XnotX*S';
    
    % concatenated full gradient
    dK = [dJ_cb(:); dJ_ab(:)];
    
    %% add all bit flip comparison case
    % calculate the energies for the data states
    EX = sum(Xn.*(Jn*Xn)) + sum(S.*(Js'*Xn));
    % calculate the energies for the states where all bits are flipped relative to data states
    notXn = 1-Xn;
    EnotX = sum(notXn.*(Jn*notXn)) + sum(S.*(Js'*notXn));    
    % calculate the contribution to the MPF objective function from all-bit-flipped states
    K2full = exp( (EX - EnotX)/2 );
    K2 = sum(K2full);
    % calculate the gradient contribution from all-bit-flipped states
    % site-to-site gradient
    dJn = bsxfun( @times, Xn, K2full ) * Xn'/2 - bsxfun( @times, notXn, K2full ) * notXn'/2;
    % stimulus-to-site gradient
    dJs = bsxfun( @times, Xn, K2full ) * S'/2 - bsxfun( @times, notXn, K2full ) * S'/2;

    % add all-bit-flipped contributions on to full objective
    K = K + K2;
    dK = dK + [dJs(:); dJn(:)];
    
    K  = K  / Treal;
    dK = dK / Treal;
    
    % Add L1 regularization term (sum of absolute values of all the weights)
    wsum = sum(abs(Jn(:))) + sum(abs(Js(:)));
    K = K + lambda*wsum;
    
    dsum = sign([Js(:); Jn(:)]);
    dK = dK + lambda*dsum;
    
    dK(zeronodes) = 0;
