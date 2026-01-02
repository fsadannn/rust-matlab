function ans = lin_taylor_2_1(A, a, B, b, x0, t, dW)
% LIN_TAYLOR_2_1 - Solves a linear stochastic differential equation using a 2.1 Taylor method.
%
%   This is a compiled rust function. The documentation is extracted from the rust code.
%
%   ans = lin_taylor_2_1(A, a, B, b, x0, t, dW)
%
%   Solves a system of linear stochastic differential equations of the form:
%   dX = (A*X + a)*dt + (B*X + b)*dW
%
%   using a strong Taylor method of order 2.1.
%
%   Input:
%       A:  (d x d) matrix
%       a:  (d x 1) vector
%       B:  (d x d x m) matrix or (d x d) matrix
%       b:  (d x m) matrix
%       x0: (d x 1) initial condition vector
%       t:  (1 x n) time vector
%       dW: (m x n) Wiener process increments
%
%   Output:
%       ans: (d x n) matrix of the solution
