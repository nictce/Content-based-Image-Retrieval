function [lshStruct, featureSet, ref, trainingSet, testingSet] = constructLSH()

%CIFAR10
rootFolder = 'cifar10Train';
categories = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};

%CIFAR100
% rootFolder = 'cifar100Train';
% categories = {'keyboard', 'tulip', 'oak_tree', 'turtle', 'mountain', 'pickup_truck', 'clock', 'otter', 'whale', 'lawn_mower', 'girl', 'sunflower', 'elephant', 'tiger', 'plate', 'crocodile', 'butterfly', 'hamster', 'tank', 'orchid', 'snake', 'squirrel', 'lion', 'camel', 'bottle', 'lizard', 'bee', 'maple_tree', 'shark', 'mushroom', 'television', 'rabbit', 'sea', 'house', 'streetcar', 'couch', 'skyscraper', 'shrew', 'beetle', 'pine_tree', 'aquarium_fish', 'leopard', 'lobster', 'telephone', 'raccoon', 'road', 'cattle', 'cloud', 'kangaroo', 'crab', 'castle', 'lamp', 'man', 'bed', 'forest', 'poppy', 'dolphin', 'rocket', 'orange', 'mouse', 'flatfish', 'sweet_pepper', 'baby', 'pear', 'skunk', 'wardrobe', 'porcupine', 'bridge', 'chair', 'can', 'cockroach', 'wolf', 'willow_tree', 'motorcycle', 'snail', 'plain', 'tractor', 'seal', 'trout', 'palm_tree', 'possum', 'worm', 'cup', 'dinosaur', 'fox', 'ray', 'rose', 'bicycle', 'bus', 'table', 'caterpillar', 'boy', 'chimpanzee', 'train', 'bowl', 'woman', 'beaver', 'spider', 'apple', 'bear'};

%COREL80
% rootFolder = 'CorelDB';
% categories = {'art_dino', 'bld_sculpt', 'obj_aviation', 'obj_car', 'obj_door', 'obj_moleculr', 'pet_cat', 'sc_', 'sc_iceburg', 'sc_rural', 'texture_1', 'texture_6', 'wl_eagle', 'wl_horse', 'wl_owls', 'wl_wolf', 'art_mural', 'eat_drinks', 'obj_balloon', 'obj_cards', 'obj_eastregg', 'obj_orbits', 'pet_dog', 'sc_autumn', 'sc_indoor', 'sc_sunset', 'texture_2', 'wl_buttrfly', 'wl_elephant', 'wl_lepoad', 'wl_porp', 'woman', 'art_1', 'bld_castle', 'eat_feasts', 'obj_bob', 'obj_decoys', 'obj_flags', 'obj_ship', 'pl_flower', 'sc_cloud', 'sc_mountain', 'sc_waterfal', 'texture_3', 'wl_cat', 'wl_fish', 'wl_lion', 'wl_primates', 'art_antiques', 'bld_lighthse', 'fitness', 'obj_bonsai', 'obj_dish', 'obj_mask', 'obj_steameng', 'pl_foliage', 'sc_firewrk', 'sc_night', 'sc_waves', 'texture_4', 'wl_cougr', 'wl_fox', 'wl_lizard', 'wl_roho', 'art_cybr', 'bld_modern', 'obj_234000', 'obj_bus', 'obj_doll', 'obj_mineral', 'obj_train', 'pl_mashroom', 'sc_forests', 'sc_rockform', 'sp_ski', 'texture_5', 'wl_deer', 'wl_goat', 'wl_nests', 'wl_tiger'};

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
    % I = imresize(I, [256 256]); % can try with this
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

