% Demo macro to very, very simple color detection in RGB color space.
% by ImageAnalyst


	
	% Read in image into an array.
	rgbImage = imread('train06.jpg');
 	rgbImage1  = im2double(rgbImage);
    [mask,~] = createMaskR(rgbImage1);
    
    
	[row,col,~] = size(rgbImage);
	% Display the original image.
    figure(1)
	subplot(2, 3, 1);
	imshow(rgbImage);
	redBand = rgbImage(:, :, 1);
	redObjectsMask = uint8(mask);
    %redObjectsMask = imopen(redObjectsMask, structuringElement);
	subplot(2, 3, 2);
	imshow(redObjectsMask, []);
	caption = sprintf('Mask of Only\nThe Red Objects');
	title(caption, 'FontSize', fontSize);
	
	% filter out small objects.
	smallestAcceptableArea = 8000; % Keep areas only if they're bigger than this.

	
	% Get rid of small objects.  Note: bwareaopen returns a logical.
	redObjectsMask = uint8(bwareaopen(redObjectsMask, smallestAcceptableArea));
	subplot(2, 3, 3);
	imshow(redObjectsMask, []);
	fontSize = 13;
	caption = sprintf('bwareaopen() removed objects\nsmaller than %d pixels', smallestAcceptableArea);
	title(caption, 'FontSize', fontSize);
	
	% Smooth the border using a morphological closing operation, imclose().
	structuringElement = strel('disk', 6);
	redObjectsMask = imclose(redObjectsMask, structuringElement);
    %redObjectsMask = imopen(redObjectsMask, structuringElement);
	subplot(2, 3, 4);
	imshow(redObjectsMask, []);
	fontSize = 16;
	title('Border smoothed', 'FontSize', fontSize);
    
    whiteside_row = ones(row,1);
    redObjectsMask = [redObjectsMask,whiteside_row];
    % Fill in any holes in the regions, since they are most likely red also.
	redObjectsMask = uint8(imfill(redObjectsMask, 'holes'));
    redObjectsMask = redObjectsMask(:,1:end-1);

    whiteside_col = ones(1,col);
	redObjectsMask = [whiteside_row,redObjectsMask];
	% Fill in any holes in the regions, since they are most likely red also.
	redObjectsMask = uint8(imfill(redObjectsMask, 'holes'));
    redObjectsMask = redObjectsMask(:,2:end);
    
    
    redObjectsMask = [whiteside_col;redObjectsMask];
    redObjectsMask = uint8(imfill(redObjectsMask, 'holes'));
    redObjectsMask = redObjectsMask(2:end,:);
    
    redObjectsMask = [redObjectsMask;whiteside_col];
    redObjectsMask = uint8(imfill(redObjectsMask, 'holes'));
    redObjectsMask = redObjectsMask(1:end-1,:);
    
	subplot(2, 3, 5);
	imshow(redObjectsMask, []);
	title('Regions Filled', 'FontSize', fontSize);
    
    




	redObjectsMask = watershedSegment(redObjectsMask);
	subplot(2, 3, 6);
	imshow(redObjectsMask, []);
	fontSize = 16;
	title('water', 'FontSize', fontSize);
    
    
axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
hold on;
[boundaries,l] = bwboundaries(redObjectsMask);
numberOfBoundaries = size(boundaries, 1);
for k = 1 : numberOfBoundaries
	thisBoundary = boundaries{k};
	plot(thisBoundary(:,2), thisBoundary(:,1), 'g', 'LineWidth', 2);
end

maxferetprop = regionprops(l,'MaxFeretProperties');
labelarea = size(maxferetprop,1);
a = maxferetprop(1).MaxFeretCoordinates;
for k = 1:labelarea
    feret_coor = maxferetprop(k).MaxFeretCoordinates;
    plot(feret_coor(:,1),feret_coor(:,2),'rx');
end

ratio_list = feret_ratio(l);
minferetprop = regionprops(l,'MinFeretProperties');
labelarea = size(minferetprop,1);
b = minferetprop(1).MinFeretCoordinates;
b = b.';
c = ProjPoint(a(1,:),b);
c2 = ProjPoint(a(2,:),b);
distance = norm(a(1,:)-c);
distance2 = norm(a(2,:)-c2);
ratio = (distance+distance2)/norm(b(:,1)-b(:,2));

for k = 1:labelarea
    feret_coor = minferetprop(k).MinFeretCoordinates;
    plot(feret_coor(:,1),feret_coor(:,2),'rx');
    plot(c(1),c(2),'bx','markersize',50);
    plot(c2(1),c2(2),'bx','markersize',50);
end
hold off;
	% You can only multiply integers if they are of the same type.
	% (redObjectsMask is a logical array.)
	% We need to convert the type of redObjectsMask to the same data type as redBand.
	redObjectsMask = cast(redObjectsMask, class(redBand));
	
	% Use the red object mask to mask out the red-only portions of the rgb image.
	maskedImageR = redObjectsMask .* redBand;
	
    
	% Show the masked off red image.
    figure(2)
	subplot(3, 3, 1);
	imshow(maskedImageR);
	title('Masked Red Image', 'FontSize', fontSize);
    hold on;
    
box=regionprops(l, 'BoundingBox');
boxnum = size(box,1);
for k = 1 : boxnum
	thisBoundary = box(k).BoundingBox;
	rectangle('position', [thisBoundary(1),thisBoundary(2),thisBoundary(3),thisBoundary(4)],'edgecolor','r');
end
hold off;

    
    subplot(3,3,2);
    imagesc(l);
    colorbar
	% Show the masked off, original image.
	
	fontSize = 13;
	caption = sprintf('Masked Original Image\nShowing Only the Red Objects');
	title(caption, 'FontSize', fontSize);
    
    subplot(3,3,4);
    imshow(redBand);
	% Show the original image next to it.
	subplot(3, 3, 5);
	imshow(rgbImage);
	title('The Original Image (Again)', 'FontSize', fontSize);
    
    %find the circle
    %rgbImage1 = rgb2gray(rgbImage1);
    maskedImageR = im2double(maskedImageR);
    filt = imfilter(maskedImageR, fspecial('log', 28, 0.525));
    % only keep the lowest values
    % they highlight the circles on the legos
    Ibin = filt < 0;
    % remove noise and close holes
    Ibin = imopen(Ibin, strel('disk', 2));
    Ibin = imclose(Ibin, strel('disk', 2));
    Ibin = ~bwareaopen(~Ibin, 300);
    subplot(3,3,6);
    imshow(Ibin);
    
    % this will sometimes remove more pixel bridges, without damaging the circles
    Ibin = imclose(Ibin, strel('disk', 2));
    %Ibin = imdilate(Ibin, strel('disk', 1));
    Ibin = bwareaopen(Ibin, 200);
    subplot(3,3,7);
    imshow(Ibin);
    
    % label all regions
    [B, L] = bwboundaries(Ibin);
    
    
    subplot(3,3,8); imagesc(Ibin);
    % https://uk.mathworks.com/matlabcentral/answers/16033-detect-rounding-objects-only-and-remove-all-other-objects
    % https://uk.mathworks.com/help/images/examples/identifying-round-objects.html
    circularityThresh = 0.65;
    stats = regionprops(L, 'Area', 'Centroid');
    keeperList = [];
    % iterate through the regions in the original image
    % only keep those that are round enough
    % loop over the boundaries
    for k = 1:length(B)
      % obtain (X,Y) boundary coordinates corresponding to label 'k'
      boundary = B{k};
      % compute a simple estimate of the object's perimeter
      delta_sq = diff(boundary).^2;
      perimeter = sum(sqrt(sum(delta_sq,2)));
      % obtain the area corresponding to label 'k'
      area = stats(k).Area;
      % compute the roundness metric
      metric = 4*pi*area/perimeter^2;
      if metric > circularityThresh
        keeperList = [keeperList k];
      end
    end
    % only keep round regions
    % https://uk.mathworks.com/matlabcentral/fileexchange/25157-image-segmentation-tutorial
    I = ismember(L, keeperList);
    subplot(3,3,9); imagesc(I);
    
    
    function bw = watershedSegment(bwmask)
    % bwmask = binary image where regions will be split using watershed
    % watershed will separate regions that are joined by small bridges
    % https://uk.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html
  
    d = -bwdist(~bwmask);
    mask = imextendedmin(d, 8);
    d2 = imimposemin(d, mask);
    ld2 = watershed(d2);
    bwmask(ld2 == 0) = 0;
    
    bw = imerode(bwmask, strel('disk', 8));
    end
	
function [BW,maskedRGBImage] = createMaskR(RGB)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 06-Dec-2019
%------------------------------------------------------


I = rgb2hsv(RGB);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.969;
channel1Max = 0.027;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.471;
channel2Max = 1.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.330;
channel3Max = 0.843;

% Create mask based on chosen histogram thresholds
sliderBW = ( (I(:,:,1) >= channel1Min) | (I(:,:,1) <= channel1Max) ) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;



% Initialize output masked image based on input image.
maskedRGBImage = RGB;

% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;
end

%点到线投影
function proj_point = ProjPoint( point,line_p )
x1 = line_p(1);
y1 = line_p(2);
x2 = line_p(3);
y2 = line_p(4);

x3 = point(1);
y3 = point(2);

yk = ((x3-x2)*(x1-x2)*(y1-y2) + y3*(y1-y2)^2 + y2*(x1-x2)^2) / (norm([x1-x2,y1-y2])^2);
xk = ((x1-x2)*x2*(y1-y2) + (x1-x2)*(x1-x2)*(yk-y2)) / ((x1-x2)*(y1-y2));


if x1 == x2
    xk = x1;
end

if y1 == y2
    xk = x3;
end

proj_point = [xk,yk];

end





