% Path setup
imageFolder = 'C:\yolo\candyimages\images';
labelFolder = 'C:\yolo\candyimages\labels';

% Get all image files
imgFiles = dir(fullfile(imageFolder, '*.jpg')); % change to .png if needed

% Initialize cell arrays for table
imagePaths = {};
bboxData = {};

for i = 1:length(imgFiles)
    % Get image name
    imgName = imgFiles(i).name;
    [~, nameNoExt, ~] = fileparts(imgName);
    
    % Full path to image
    imgFullPath = fullfile(imageFolder, imgName);
    imagePaths{end+1} = imgFullPath;
    
    % Full path to corresponding label file
    labelPath = fullfile(labelFolder, [nameNoExt '.txt']);
    
    % Read image to get its size
    I = imread(imgFullPath);
    [imgH, imgW, ~] = size(I);

    % Read YOLO label (if exists)
    if exist(labelPath, 'file')
        yoloData = readmatrix(labelPath);  % [class x_center y_center width height]
        
        % Make sure it's not empty
        if ~isempty(yoloData)
            if size(yoloData,2) == 5
                % Convert each bbox from normalized to [x y width height]
                boxes = zeros(size(yoloData,1), 4);
                for j = 1:size(yoloData,1)
                    xc = yoloData(j,2) * imgW;
                    yc = yoloData(j,3) * imgH;
                    w  = yoloData(j,4) * imgW;
                    h  = yoloData(j,5) * imgH;

                    x = xc - w/2;
                    y = yc - h/2;

                    boxes(j,:) = [x y w h];
                end
                bboxData{end+1} = boxes;
            else
                bboxData{end+1} = zeros(0,4);  % no box
            end
        else
            bboxData{end+1} = zeros(0,4);
        end
    else
        bboxData{end+1} = zeros(0,4);  % No label file
    end
end

% Create table
candyDataset = table(imagePaths', bboxData', 'VariableNames', {'imageFilename', 'candy'});

% Save it
save('C:\yolo\candyDataset.mat', 'candyDataset');

disp('✅ Dataset prepared and saved as C:\yolo\candyDataset.mat');
