function ans = rpade_no_norm(A, q)
% RPADE_NO_NORM - Computes the matrix exponential using Pade approximation.
%
%   This is a compiled rust function. The documentation is extracted from the rust code.
%
%   ans = rpade_no_norm(A, q)
%
%   Computes the matrix exponential of a square matrix A using the Pade
%   approximation. The scaling factor is computed internally.
%
%   Input:
%       A: (n x n) square matrix
%       q: scalar, degree of the Pade approximation
%
%   Output:
%       ans: (n x n) matrix, the matrix exponential of A.
