%% STEP 1: Load Detector and Classifier
load('C:/yolo/multiCandyYOLOv4.mat', 'detector');
load('C:/yolo/candyClassifier.mat', 'candyClassifier');

classNames = readlines("C:/yolo/candyimages/classes.txt");
classNames = strtrim(classNames);

%% STEP 2: Load Test Image
testImage = imread("C:\yolo\candyimages\images\36b46b7d-candy_98.jpg");

%% STEP 3: Run YOLOv4 Detection
[bboxes, scores, labels] = detect(detector, testImage);

%% STEP 4: Classify Each Detection with SqueezeNet
inputSize = candyClassifier.Layers(1).InputSize;
confidenceThreshold = 0.5;
sqzLabels = strings(size(bboxes,1),1);
results = table;

for i = 1:size(bboxes,1)
    if scores(i) < confidenceThreshold
        continue;
    end

    crop = imcrop(testImage, bboxes(i,:));
    crop = imresize(crop, inputSize(1:2));

    if size(crop,3) == 1
        crop = repmat(crop, [1 1 3]);
    end

    sqzLabel = classify(candyClassifier, crop);
    sqzLabels(i) = string(sqzLabel);

    results = [results;
        table(bboxes(i,:), string(labels(i)), sqzLabels(i), ...
        'VariableNames', {'BBox', 'YOLOLabel', 'SqueezeNetLabel'})];
end

%% STEP 5: Display Annotated Image Once
if ~isempty(results)
    annotations = sqzLabels + " (" + string(round(scores(1:height(results))*100)) + "%)";
    Iout = insertObjectAnnotation(testImage, "rectangle", results.BBox, annotations);
    figure; imshow(Iout);
    title("YOLO + SqueezeNet Classification Results");
else
    warning("No objects detected with confidence above threshold.");
end

%% STEP 6: Accuracy & Confusion Matrix
validRows = sqzLabels ~= "";
trueLabels = string(results.YOLOLabel(validRows));
predLabels = string(results.SqueezeNetLabel(validRows));

if ~isempty(trueLabels)
    figure;
    confusionchart(trueLabels, predLabels, ...
        'Title', 'YOLO vs SqueezeNet Confusion Matrix', ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    matchAccuracy = sum(trueLabels == predLabels) / numel(trueLabels);
    fprintf("SqueezeNet vs YOLO Match Accuracy: %.2f%%\n", matchAccuracy * 100);
else
    warning("No valid comparisons for confusion matrix.");
end
