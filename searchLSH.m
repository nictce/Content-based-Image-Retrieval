function [ranks] = searchLSH(input_file, featureSet, ref, lshStruct)

cd('vLSH')

% tune table searching hyperparameters 
K = 32;   % # of nearest neighbors to search for each query
T = 10;   % # of additional probing bins


% extract the SIFT features of the query image for searching
I = im2gray(imread(input_file));
% I = imresize(I, [256 256]); % can try using this
sift = detectSIFTFeatures(I).selectStrongest(25);
[queryFeatures, ~] = extractFeatures(I, sift);


% query each of the SIFT features of the image to search for the nearest K
% features matching it in the hashtables (in lshStruct)
[idsMULTIPROBE, ~, ~] = lshSearch(queryFeatures.', featureSet, lshStruct, K, T);


% convert the returned feature IDs into their original image IDs
oriImages = zeros(size(idsMULTIPROBE));
for i = 1 : size(idsMULTIPROBE, 1)
    for j = 1 : size(idsMULTIPROBE, 2)
        if idsMULTIPROBE(i, j)>0
          oriImages(i,j) = ref(:,idsMULTIPROBE(i, j));
        end
    end
end


% aggregate (and weight) each of the image IDs into ranks, according to
% the number of times an image's features were returned by the query
uniqImageIds = unique(oriImages);
N = numel(uniqImageIds);
weights = zeros(N,1);
% fib = zeros(1, K); fib(K-1) = 1; % creates the fibonacci sequence for ranked weighting of NNs
% for i = K-2:-1:1
%    fib(i) = fib(i+1)+fib(i+2);
% end

for k = 1 : N
    weights(k) = sum(sum(oriImages == uniqImageIds(k))); % unranked weighting of NNs
    % ranked weighting of NNs
    % weights(k) = sum((K + (0 : K - 1) * -1) * (oriImages == uniqImageIds(k)));  % linear
    % weights(k) = sum(flip(exp(1 : K)) * (oriImages == uniqImageIds(k))); % exponential
    % weights(k) = sum(fib * (oriImages == uniqImageIds(k))); % fibonacci
end


% sort each of the image IDs in descending order accordint to their frequency
ranks = sortrows([weights uniqImageIds], 1, 'descend');

cd("..\")
end

