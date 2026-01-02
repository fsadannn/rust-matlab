function ans = gemv3d(A, x, y)
% GEMV3D - 3D matrix-matrix or matrix-vector multiplication.
%
%   This is a compiled rust function. The documentation is extracted from the rust code.
%
%   ans = gemv3d(A, x, y)
%
%   Computes matrix-vector or matrix-matrix products of 3D matrices.
%   A is a 3D matrix of size M x N x P.
%   x is a vector or a 2D/3D matrix.
%   y is a vector or a 2D/3D matrix that is added to the result.
%
%   The behavior depends on the dimensions of x and y.
%
%   Input:
%       A: 3D matrix (M x N x P)
%       x: vector, 2D matrix or 3D matrix
%       y: vector, 2D matrix or 3D matrix
%
%   Output:
%       ans: result of the computation.
