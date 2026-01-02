function ans = rpade_triangular_no_norm(A, q)
% RPADE_TRIANGULAR_NO_NORM - Computes the matrix exponential of a triangular matrix using Pade approximation.
%
%   This is a compiled rust function. The documentation is extracted from the rust code.
%
%   ans = rpade_triangular_no_norm(A, q)
%
%   Computes the matrix exponential of a square triangular matrix A using the Pade
%   approximation. The scaling factor is computed internally.
%
%   Input:
%       A: (n x n) square triangular matrix
%       q: scalar, degree of the Pade approximation
%
%   Output:
%       ans: (n x n) matrix, the matrix exponential of A.
