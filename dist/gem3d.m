function ans = gem3d(A, B, y)
% GEMV3D - 3D matrix-matrix or matrix-vector multiplication.
%
%   This is a compiled rust function. The documentation is extracted from the rust code.
%
%   ans = gem3d(A, B, y)
%
%   Computes matrix-vector or matrix-matrix products of 3D matrices.
%   A is a 3D matrix of size M x N x P.
%   B is a vector or a 2D/3D matrix.
%   y is a vector or a 2D/3D matrix that is added to the result.
%
%   The behavior depends on the dimensions of x and y.
%
% The options for A and B be compatible with A: A1xA2xA3 and B: B1xB2xB3
% - A2 == B1 meaning we can multiply A and B across pages
% - A3 == B3 have the same pages
% - OUT: A1xB2xA3
% - A3 == B2 and B3 == 1, B is a vector across pages
% - OUT: A1x1xA3 -> simplified to a matrix of A1xA3
% - A2 != B1 and B1 == 1 and B2 == A3 and B3 == 1, meaning B is a numeric constant across pages
% - OUT: A1xA2xA3

% The options for C=A*B and Y be compatible with C: C1xC2xC3 and B: Y1xY2xY3
% - C1 == Y1 and C2 == Y2 meaning we can add C and Y across pages
% - C3 == Y3 have the same pages
% - C3 != Y3 broadcast Y across pages
% - OUT: A1x1xA3 -> simplified to a matrix of A1xA3
% - Y1==Y2==Y3==1 Y=0 meaning ignore the additive part
