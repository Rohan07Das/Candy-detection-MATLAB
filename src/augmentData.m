% Supporting Functions Section
% =============================

function data = augmentData(A)
    data = cell(size(A));
    for ii = 1:size(A,1)
        I = A{ii,1};
        bboxes = A{ii,2};
        labels = A{ii,3};
        sz = size(I);

        if numel(sz) == 3 && sz(3) == 3
            I = jitterColorHSV(I, Contrast=0.0, Hue=0.1, Saturation=0.2, Brightness=0.2);
        end

        tform = randomAffine2d(XReflection=true, Scale=[1 1.1]);
        rout = affineOutputView(sz, tform, BoundsStyle="centerOutput");
        I = imwarp(I, tform, OutputView=rout);

        [bboxes, indices] = bboxwarp(bboxes, tform, rout, OverlapThreshold=0.25);
        labels = labels(indices);

        if isempty(indices)
            data(ii,:) = A(ii,:);
        else
            data(ii,:) = {I, bboxes, labels};
        end
    end
end

