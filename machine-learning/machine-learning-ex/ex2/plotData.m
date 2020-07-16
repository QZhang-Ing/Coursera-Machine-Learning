function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


ones = find(y==1);
zeros = find(y==0);
% plot

plot(X(ones, 1), X(ones, 2), 'k+', 'MarkerSize', 5)
plot(X(zeros, 1), X(zeros, 2), 'ro', 'MarkerSize', 5)








% =========================================================================



hold off;

end
