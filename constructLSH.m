function [lshStruct, featureSet, ref, trainingSet, testingSet] = constructLSH()

% cd('C:\Users\PC\Desktop\3081\Content-based-Image-Retrieval')

%CIFAR100
% rootFolder = 'cifar-100-matlab\CIFAR-100\TEST';
% categories = {'cup','dinosaur','forest','hamster'};

%COREL80
rootFolder = 'CorelDB';
categories = {'art_antiques','obj_car','sc_sunset','wl_buttrfly','bld_sculpt'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imagesPerClass = 100;
[trainingSet, testingSet] = splitEachLabel(imds, imagesPerClass, 'randomize'); 

featureSet = [];
ref = [];
numImages = numel(trainingSet.Files);

% preprocess the images in the dataset and collect:
% - their SIFT features into the featureSet matrix
% - their original image ID into the ref vector
for i = 1 : numImages
    I = im2gray(readimage(trainingSet,i));
%     I = imresize(I, [256 256]); % can try without this
    sift = detectSIFTFeatures(I).selectStrongest(25);
    [features, ~] = extractFeatures(I, sift);
    featureSet = cat(2, featureSet, features.');
    ref = cat(2, ref, repmat([i], 1, size(features,1)));
end


% construct LSH hash tables for featureSet
cd('vLSH')

% Tune hyperparameters
% Q = 144;  % # of queries
L = 10;   % # of tables
M = 25;   % # of dimensions at projection space
W = 1000; % bucket width

% Construct index tables
lshStruct = lshConstruct(featureSet, L, M, W);

cd("..\")
end

