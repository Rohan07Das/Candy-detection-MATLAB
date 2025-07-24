%% STEP 1: Load Dataset
imds = imageDatastore('C:/yolo/croppedCandies', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Filter out images smaller than 40x40
validIdx = false(numel(imds.Files),1);
for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    if size(img,1) > 40 && size(img,2) > 40
        validIdx(i) = true;
    end
end
imds = subset(imds, find(validIdx));

% Check how many classes
numClasses = numel(categories(imds.Labels));
fprintf('Detected %d classes with %d images\n', numClasses, numel(imds.Files));

% Split into train/val sets
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

%% STEP 2: Load Pretrained SqueezeNet and Modify
net = squeezenet;
lgraph = layerGraph(net);

% Remove original classification layers
lgraph = removeLayers(lgraph, {'conv10','relu_conv10','pool10','prob','ClassificationLayer_predictions'});

% Define new classification head
newLayers = [
    convolution2dLayer(1, numClasses, 'Name', 'new_conv', ...
        'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
    reluLayer('Name','new_relu')
    globalAveragePooling2dLayer('Name','new_gap')
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_output')];

% Add and connect layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'drop9', 'new_conv');

%% STEP 3: Prepare Data
inputSize = net.Layers(1).InputSize;

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing','gray2rgb');

augimdsVal = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing','gray2rgb');

%% STEP 4: Define Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',16, ...
    'ValidationData',augimdsVal, ...
    'Shuffle','every-epoch', ...
    'ValidationPatience',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% STEP 5: Load the Classifier

load('C:/yolo/candyClassifier.mat', 'candyClassifier');

inputSize = candyClassifier.Layers(1).InputSize;
sqzLabels = strings(size(bboxes,1),1);
results = table;

% Set confidence threshold if available
confidenceThreshold = 0.5;

for i = 1:size(bboxes,1)
    % Optional: Skip low-confidence detections
    if exist('scores', 'var') && scores(i) < confidenceThreshold
        continue;
    end

    % Crop and resize
    crop = imcrop(testImage, bboxes(i,:));
    crop = imresize(crop, inputSize(1:2));

    % Fix grayscale
    if size(crop,3) == 1
        crop = repmat(crop, [1 1 3]);
    end

    % Classify with SqueezeNet
    sqzLabel = classify(candyClassifier, crop);
    sqzLabels(i) = string(sqzLabel);

    % Store info
    results = [results; 
        table(bboxes(i,:), string(labels(i)), sqzLabels(i), ...
        'VariableNames', {'BBox', 'YOLOLabel', 'SqueezeNetLabel'})];

    % Optional display
    figure; imshow(crop);
   yoloLabel = strrep(string(labels(i)), "_", "\_");
sqzLabel  = strrep(sqzLabels(i), "_", "\_");
    title(sprintf("YOLO: %s | SqueezeNet: %s", yoloLabel, sqzLabel));
end

disp(results);

%% STEP 6: Confusion Matrix and Accuracy
validRows = sqzLabels ~= "";
trueLabels = string(results.YOLOLabel(validRows));
predLabels = string(results.SqueezeNetLabel(validRows));

figure;
confusionchart(trueLabels, predLabels, ...
    'Title', 'YOLO vs SqueezeNet Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

matchAccuracy = sum(trueLabels == predLabels) / numel(trueLabels);
fprintf("SqueezeNet vs YOLO Match Accuracy: %.2f%%\n", matchAccuracy * 100);

