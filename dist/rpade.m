function ans = rpade(A, q, s)
% RPADE - Computes the matrix exponential using Pade approximation.
%
%   This is a compiled rust function. The documentation is extracted from the rust code.
%
%   ans = rpade(A, q, s)
%
%   Computes the matrix exponential of a square matrix A using the Pade
%   approximation. This is equivalent to MATLAB's expm(A).
%
%   Input:
%       A: (n x n) square matrix
%       q: scalar, degree of the Pade approximation
%       s: scalar, scaling factor
%
%   Output:
%       ans: (n x n) matrix, the matrix exponential of A.
