% run_ising_demo.m
% Runs example Ising model fitting on sample_data.mat
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

% [Created by Liberty S. Hamilton, Jascha Sohl-Dickstein, and Alexander Huth, 
% at the University of California, Berkeley.
% Based on code written by Jascha Sohl-Dickstein (2009) available at 
% https://github.com/Sohl-Dickstein/Minimum-Probability-Flow-Learning]
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

datafile = 'sample_data.mat';
model = 0;
nchunks = 10; 
[all_J, all_logL, modelname] = ising_neurons_L1reg(datafile, model, nchunks);

imagesc(all_J(:,:,1), [-2 2]); % plot coupling from 1st cross validation iteration

