function [x,y] = get_batch(X,Y)
%GET_BATCH Summary of this function goes here
%   Detailed explanation goes here
N = size(X, 1);
n = randi(N);
x = cell2mat(X(n, :));
y = cell2mat(Y(n, :));
end

