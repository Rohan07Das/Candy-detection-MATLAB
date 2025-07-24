% PART 1: Load Dataset and Class Names
% =========================================
load('C:/yolo/candyDataset.mat'); % Must contain 'candyDataset'

classNames = readlines("C:/yolo/candyimages/classes.txt");
classNames = strtrim(classNames);  % Remove any whitespace

% Split data
rng(0);
shuffledIdx = randperm(height(candyDataset));
n = height(candyDataset);
idxTrain = 1:round(0.6*n);
idxVal   = round(0.6*n)+1 : round(0.8*n);
idxTest  = round(0.8*n)+1 : n;

trainingDataTbl   = candyDataset(shuffledIdx(idxTrain), :);
validationDataTbl = candyDataset(shuffledIdx(idxVal), :);
testDataTbl       = candyDataset(shuffledIdx(idxTest), :);

imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
bldsTrain = boxLabelDatastore(trainingDataTbl(:, classNames));
trainingData = combine(imdsTrain, bldsTrain);

imdsVal = imageDatastore(validationDataTbl.imageFilename);
bldsVal = boxLabelDatastore(validationDataTbl(:, classNames));
validationData = combine(imdsVal, bldsVal);

imdsTest = imageDatastore(testDataTbl.imageFilename);
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));  % assuming 2:end are class columns
testData = combine(imdsTest, bldsTest);  % ✅ this is what you're missing

save('C:/yolo/testData.mat', 'testData');
save('C:/yolo/trainingData.mat', 'trainingData');
save('C:/yolo/validationData.mat', 'validationData');

inputSize = [320 320 3];
augmentedTrainingData = transform(trainingData, @(data) preprocessData(data, inputSize));

% Estimate anchor boxes
[anchors, meanIoU] = estimateAnchorBoxes(augmentedTrainingData, 9);

% Sort anchors by area (descending)
areas = anchors(:,1) .* anchors(:,2);
[~, idx] = sort(areas, 'descend');
anchors = anchors(idx, :);

% Group into 3 anchor box groups (YOLOv4 expects 3 levels)
anchorBoxes = {anchors(1:3,:); anchors(4:6,:); anchors(7:9,:)};

detector = yolov4ObjectDetector("csp-darknet53-coco", classNames, anchorBoxes, InputSize=inputSize);

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=30, ...
    MiniBatchSize=2, ...
    ValidationData=validationData, ...
    Shuffle="every-epoch", ...
    VerboseFrequency=20, ...
    CheckpointPath="C:/yolo/checkpoints", ...
    Plots="training-progress", ...
    BatchNormalizationStatistics="moving", ...
    ResetInputNormalization=false);  % ✅ Required

load('C:/yolo/multiCandyYOLOv4.mat'); % Loads 'detector'

% Use any test image path from your dataset
testImage = imread("C:\yolo\candyimages\images\7e893c2d-candy_38.jpg");

% Run detection
[bboxes, scores, labels] = detect(detector, testImage);

% Annotate image
Iout = insertObjectAnnotation(testImage, "rectangle", bboxes, ...
    cellstr(labels) + " " + string(round(scores * 100)) + "%");

% Display
figure;
imshow(Iout);
title("Candy Detection with Confidence Scores");

% Load trained detector and test data
load('C:/yolo/multiCandyYOLOv4.mat', 'detector');
load('C:/yolo/testData.mat', 'testData');

% Run detection
detectionResults = detect(detector, testData);

% Evaluate detection precision
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, testData);

% Load and format class names
classNames = readlines("C:/yolo/candyimages/classes.txt");
classNames = strtrim(classNames);
classNames = strrep(classNames, "_", "\_");  % 👈 Escaping underscores

% Plot all PR curves
figure;
hold on;
legendEntries = cell(numel(ap), 1);

for i = 1:numel(ap)
    plot(recall{i}, precision{i}, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    legendEntries{i} = sprintf("%s (AP = %.2f)", classNames(i), ap(i));
end

xlabel("Recall");
ylabel("Precision");
title("Precision-Recall Curves for All Classes");
legend(legendEntries, 'Location', 'eastoutside');
xlim([0 1]);
ylim([0 1]);
grid on;
hold off;

% Display Mean AP
meanAP = mean(ap);
disp("Mean AP across all classes: " + meanAP);

% Ensure classNames is in string format
classNames = readlines("C:/yolo/candyimages/classes.txt");
classNames = strtrim(classNames);

% Convert cell arrays (recall and precision) to scalar values (last point or max)
recallVals = cellfun(@(r) r(end), recall);
precisionVals = cellfun(@(p) p(end), precision);

% Create table
metricsTable = table(classNames, ap(:), recallVals(:), precisionVals(:), ...
    'VariableNames', {'Class', 'AveragePrecision', 'Recall', 'Precision'});

% Display the table
disp(metricsTable);