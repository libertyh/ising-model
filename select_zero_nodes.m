function [Jnew_all, zeronodes, modelname] = select_zero_nodes(model, Js, Jn)
%SELECT_ZERO_NODES(model, Js, Jn) 
% sets values of the stimulus-to-site coupling matrices Js and the 
% site-to-site coupling matrices Jn to be zero given a particular 
% hypothesis about the connectivity matrix. 
%
%    [Jnew_all, zeronodes, modelname] = SELECT_ZERO_NODES(model, Js, Jn)
%    where model is a number between 0 and 7, Js is the n x s stimulus-to-site
%    coupling matrix, and Jn is the n x n site-to-site coupling matrix.  Returns
%    the zeroed Jnew_all coupling matrix (concatenated new Js and Jn) as well 
%    as zeronodes, which are the numbers of the nodes that should not be updated 
%    when calculating the derivative of the objective function in K_dK_ising_L1reg.m.
%
%    IMPORTANT: This code assumes a 4 x 4 electrode grid or polytrode configuration.
%    If you are using something different, you will need to write your own function.
%    Setting model = 0 (fully connected model) will work for any configuration, since
%    zeronodes is just an empty set.
%
%  Author: Liberty Hamilton (2013)
%
%  See also ISING_NEURONS_L1REG, K_DK_ISING_L1REG
%
%  References:
%    Hamilton LS, Sohl-Dickstein J, Huth AG, Carels VM, Bao S (2013). Optogenetic
%       Activation of an Inhibitory Network Enhances Functional Connectivity in
%       Auditory Cortex.  Neuron.
%
%  See also ISING_NEURONS_L1REG, SELECT_ZERO_NODES, K_DK_ISING_L1REG, L_DL_ISING

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

    b = size(Jn, 1); % number of neurons / electrode channels
    
    % Initialize new output (concatenated Js and Jn)
    Jnew_all = horzcat(Js, Jn);

    % initialize matrices to indicate which variables
    % should be set to zero
    zeroJ_s = zeros(size(Js));
    zeroJ_n = zeros(size(Jn));
   
    % Choose which model of connectivity to use (all assume 4 x 4 
    % except for case 0, case 3, and case 4.)
    switch model
        case 0
            % For all connections:
            modelname = 'Fully connected graph';
        
        case 1
            % For only fully connected columns:
            modelname = 'Fully connected columns';
            diags = [-12, -8, -4, 0, 4, 8, 12];
            takefirst = @(m) m(1:b,1:b);
            zeroJ_n=ones(size(Jn));
            for d = diags
                zeroJ_n = zeroJ_n - takefirst(diag(ones(1,b), d));
            end

        case 2
            % For fully connected layers:
            modelname = 'Fully connected layers';
            zeroJ_n=ones(size(Jn));
            for i = [1 5 9 13]
                zeroJ_n(i:i+3,i:i+3)=0;
                zeroJ_n(i:i+1,i:i+3)=0;
            end 
            size(zeroJ_s)

        case 3
            % For independent neurons:
            modelname = 'independent neurons';
            % note here: we don't want to set the diagonal of Jn
            % to zero because this represents the bias term, or 
            % intrinsic firing of each site, so we make sure not
            % to include those diagonal indices in the nodes that
            % should be zeroed out.
            zeroJ_n=ones(size(Jn))-eye(b); 
            fname='independent_neurons.eps';

        case 4
            % For no dependence on sound:
            modelname = 'No dependence on sound';
            zeroJ_s = ones(size(Js));

        case 5
            % For neighbor connected layers:
            modelname = 'Neighbor connected layers';
            diags = [-1, 0, 1];
            takefirst = @(m) m(1:b,1:b);
            zeroJ_n=ones(b,b);
            for d = diags
                zeroJ_n = zeroJ_n - takefirst(diag(ones(1,b), d));
            end
            for i = [4 8 12]
                zeroJ_n(i,i+1)=1;
                zeroJ_n(i+1,i)=1;
            end 

        case 6
            % Fully connected layers and columns
            modelname='Fully connected layers and columns';
            diags = [-12, -8, -4, 0, 4, 8, 12];
            takefirst = @(m) m(1:b,1:b);
            zeroJ_n=ones(b,b);
            for d = diags
                zeroJ_n = zeroJ_n - takefirst(diag(ones(1,b), d));
            end
            for i = [1 5 9 13]
                zeroJ_n(i:i+3,i:i+3)=0;
                zeroJ_n(i:i+1,i:i+3)=0;
            end 

        case 7
            % For neighbor connected layers and columns:
            modelname = 'neighbor connected layers and columns';
            diags = [-1, 0, 1];
            takefirst = @(m) m(1:b,1:b);
            zeroJ_n=ones(b,b);
            for d = diags
                zeroJ_n = zeroJ_n - takefirst(diag(ones(1,b), d));
            end
            for i = [4 8 12]
                zeroJ_n(i,i+1)=1;
                zeroJ_n(i+1,i)=1;
            end 
            for i = 1:12
                zeroJ_n(i,i+4)=0;
                zeroJ_n(i+4,i)=0; 
            end

        otherwise
            % For all connections:
            modelname = 'Fully connected graph';

    end
    
    % concatenate the matrices showing which nodes should be set to zero
    zeroJ_all = horzcat(zeroJ_s, zeroJ_n);

    % find the indices of the zero nodes
    zeronodes = find(zeroJ_all);
    
    % set the coupling matrix to zero at the appropriate nodes
    Jnew_all(zeronodes) = 0;
    %imagesc(Jnew_all);
end
