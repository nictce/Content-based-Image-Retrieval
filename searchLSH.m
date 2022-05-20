function [ranks] = searchLSH(input_file, featureSet, ref, lshStruct)

cd('vLSH')
K = 32;   % # of nearest neighbors to search for each query
T = 10;   % # of additional probing bins

% query an input image
I = imread(input_file);
I = imresize(im2gray(I), [256 256]);
sift = detectSIFTFeatures(I).selectStrongest(25);
[queryFeatures, valid_points] = extractFeatures(I, sift);

[idsMULTIPROBE, cand_sizeMULTIPROBE, ~] = lshSearch(queryFeatures.', featureSet, lshStruct, K, T);


% convert the feature IDs into their original image IDs
oriImages = zeros(size(idsMULTIPROBE));
for i = 1 : size(idsMULTIPROBE, 1)
    for j = 1 : size(idsMULTIPROBE, 2)
        if idsMULTIPROBE(i, j)>0
          oriImages(i,j) = ref(:,idsMULTIPROBE(i, j));
        end
    end
end

% aggregate and weight the image features by their ranks 
uniqImageIds = unique(oriImages);
N = numel(uniqImageIds);
weights = zeros(N,1);
fib = zeros(1, K); fib(K-1) = 1;
for i = K-2:-1:1
   fib(i) = fib(i+1)+fib(i+2);
end

for k = 1 : N
    weights(k) = sum(sum(oriImages == uniqImageIds(k))); % unranked weighting of NNs
    % ranked weighting of NNs
    % weights(k) = sum((K + (0 : K - 1) * -1) * (oriImages == uniqImageIds(k)));  % linear
    % weights(k) = sum(flip(exp(1 : K)) * (oriImages == uniqImageIds(k))); % exponential
    % weights(k) = sum(fib * (oriImages == uniqImageIds(k))); % fibonacci
end

ranks = sortrows([weights uniqImageIds], 1, 'descend');

cd("..\")
end

