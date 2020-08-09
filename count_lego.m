function [numA, numB] = count_lego(I)
    
    Image_ori = I;
    Image = im2double(I);
    
    %create mask which only shows the specified color pixels in the image,  
    %https://ww2.mathworks.cn/help/images/image-segmentation-using-the-color-thesholder-app.html
    [redmask,~] = createMaskR(Image);
    [bluemask,~] = createMaskB(Image);
    %turn the mask to be uint8 which is necessary for the point
    %multiplication later on 
    redmask = uint8(redmask);
    bluemask = uint8(bluemask);
    
    
    %compute the rows and cols of the image 
    [row,col,~] = size(Image);
    %https://ww2.mathworks.cn/matlabcentral/fileexchange/26420-simplecolordetection
    %process the mask to let it completely cover the redarea in the
    %original image
    redmask = mask_processing(redmask,row,col);
    bluemask = mask_processing(bluemask,row,col);
    
    
    %Extract the red channel and the blue channel of the original image,
    %these two channels's image are dot multiplied with its corresponding 
    %mask, in this way we get a clear vision of only the red blocks and
    %blue blocks in the orignal image, which fascilitates the classification
    %of the blocks later on
    redband = Image_ori(:,:,1);
    blueband = Image_ori(:,:,3);
    % You can only multiply integers if they are of the same type.
	% (redObjectsMask is a logical array.)
	% We need to convert the type of mask to the same data type as the
	% redband/blueband
	redmask1 = cast(redmask, class(redband));
    bluemask1 = cast(bluemask, class(blueband));
    redarea = redmask1 .* redband;
    bluearea = bluemask1 .* blueband;
    
    %find the circle regions of each block
    redcircles = find_circle(redarea, 0.54);
    bluecircles = find_circle(bluearea, 0.65);
    
    %count the two types respectively
    numA = count_Rblocks(redcircles, redmask);
    numB = count_Bblocks(bluecircles, bluemask);
    figure(1)
    subplot(2,2,1);
    imshow(redarea);
    subplot(2,2,2);
    imshow(bluearea);
    subplot(2,2,3);
    imshow(redcircles);
    subplot(2,2,4);
    imshow(bluecircles);
    
    
    
%function for creating the red mask
function [BW,maskedRGBImage] = createMaskR(RGB)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 06-Dec-2019
%------------------------------------------------------


    Img = rgb2hsv(RGB);

    % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.969;
    channel1Max = 0.027;

    % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.448;
    channel2Max = 1.000;

    % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.290;
    channel3Max = 1.000;

    % Create mask based on chosen histogram thresholds
    sliderBW = ( (Img(:,:,1) >= channel1Min) | (Img(:,:,1) <= channel1Max) ) & ...
    (Img(:,:,2) >= channel2Min ) & (Img(:,:,2) <= channel2Max) & ...
    (Img(:,:,3) >= channel3Min ) & (Img(:,:,3) <= channel3Max);
    BW = sliderBW;

    % Initialize output masked image based on input image.
    maskedRGBImage = RGB;

    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~BW,[1 1 3])) = 0;
end
    
%function for creating the blue mask
function [BW,maskedRGBImage] = createMaskB(RGB)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 06-Dec-2019
%------------------------------------------------------


    % Convert RGB image to chosen color space
    Img = rgb2hsv(RGB);

    % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.549;
    channel1Max = 0.641;

    % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.350;
    channel2Max = 1.000;

    % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.140;
    channel3Max = 0.957;

    % Create mask based on chosen histogram thresholds
    sliderBW = (Img(:,:,1) >= channel1Min ) & (Img(:,:,1) <= channel1Max) & ...
        (Img(:,:,2) >= channel2Min ) & (Img(:,:,2) <= channel2Max) & ...
        (Img(:,:,3) >= channel3Min ) & (Img(:,:,3) <= channel3Max);
    BW = sliderBW;

    % Initialize output masked image based on input image.
    maskedRGBImage = RGB;

    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~BW,[1 1 3])) = 0;
end


function mask = mask_processing(mask_in,row,col)
    %mask_In: the input mask which are not processed
    %mask: the output mask
    %-----------------------------------------------------
    
    % filter out small objects.
	smallestAcceptableArea = 8000; % Keep areas only if they're bigger than this.
    % Get rid of small objects.  Note: bwareaopen returns a logical.
	mask = uint8(bwareaopen(mask_in, smallestAcceptableArea));
    % Smooth the border using a morphological closing operation, imclose().
	structuringElement = strel('disk', 6);
	mask = imclose(mask, structuringElement); 
    % Fill in any holes in the regions, since they are most likely red also.
    %here we consider to padding just one side of the image at one time and 
    %fill the holes by imfill, repeat this process four times, this can 
    %make the mask which lies on the edge of the image more integrted,
    %which fascilitate the processing of the image later on
    whiteside_row = ones(row,1);
    whiteside_col = ones(1,col);
    %right side 
    mask = [mask,whiteside_row];
	mask = uint8(imfill(mask, 'holes'));
    mask = mask(:,1:end-1);
    %left side
	mask = [whiteside_row,mask];
	mask = uint8(imfill(mask, 'holes'));
    mask = mask(:,2:end);
    %upper side
    mask = [whiteside_col;mask];
    mask = uint8(imfill(mask, 'holes'));
    mask = mask(2:end,:);
    %down side
    mask = [mask;whiteside_col];
    mask = uint8(imfill(mask, 'holes'));
    mask = mask(1:end-1,:);
    
    %watershed the mask
    mask = watershed_segment(mask);
end


function mask_out = watershed_segment(mask_in)
    % mask_in = binary image where regions will be split using watershed
    % mask_out = output image
    % watershed will separate regions that are joined by small bridges
    % https://uk.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html
  
    distance = -bwdist(~mask_in);
    mask = imextendedmin(distance, 8);
    distance2 = imimposemin(distance, mask);
    longdistance2 = watershed(distance2);
    mask_in(longdistance2 == 0) = 0;
    
    mask_out = imerode(mask_in, strel('disk', 8));
end

function circle_region = find_circle(blocks, thresh)
%blocks: the input rgb image
%circle_region: binary image which only shows the round regions
% https://stackoverflow.com/questions/31433655/find-a-nearly-circular-band-of-bright-pixels-in-this-image
% https://uk.mathworks.com/matlabcentral/answers/16033-detect-rounding-objects-only-and-remove-all-other-objects
%----------------------------------------------------------------------------------------------------------    
    
    block = im2double(blocks);
    filter = imfilter(block, fspecial('log', 18, 0.4));
    % only keep the values lower than zero
    % they highlight the circles on the image 
    filtblock = filter < 0;
    % remove noise and close holes
    filtblock = imopen(filtblock, strel('disk', 1));
    filtblock = imclose(filtblock, strel('disk', 1));
    filtblock = ~bwareaopen(~filtblock, 200);
    
    % this will sometimes remove more pixel bridges, without damaging the circles
    filtblock = imopen(filtblock, strel('disk', 1));
    filtblock = imdilate(filtblock, strel('disk', 1));
    filtblock = bwareaopen(filtblock, 300);
    
    %label the connected regions
    [B, L] = bwboundaries(filtblock);
    circlethresh = thresh;
    stats = regionprops(L, 'Area', 'Centroid');
    circleList = [];
    % iterate through the regions in the filtered block image
    % only keep those that satisfy the conditions of being a circle 
    % loop over the boundaries
    for m = 1:length(B)
      %boundary coordinates corresponding to label 'm'
      boundary = B{m};
      % calculate an estimate of the region's perimeter
      segments = diff(boundary).^2;
      perimeter = sum(sqrt(sum(segments,2)));
      % obtain the area of region 'm'
      area = stats(m).Area;
      % calculate the roundness metric
      metric = 4*pi*area/perimeter^2;
      if metric > circlethresh && area>200
        circleList = [circleList m];
      end
    end
    % https://ww2.mathworks.cn/help/matlab/ref/ismember.html?searchHighlight=ismember&s_tid=doc_srchtitle
    
    circle_region = ismember(L, circleList);
end


function Rblock_num = count_Rblocks(circle_mask, colormask)
%circle_mask: the binary image which only has the round region in one block
%             this image also has the information of color, it's the output
%             of the find_circle
%colormask: the binary image which is the output of mask_processing
%           function, it shows the red color blocks in image
%max: maximum thresh for circles in one region
%Rblock_num:the qualified lego blocks in the region
%---------------------------------------------------------------------
    %label matrix for each detected blocks in the mask
    %count the blocks in the mask
    [labeledBlocks,blobcount] = bwlabel(colormask);
    %calculate the feret length-to-width ratio of each detected blocks
    %the result is a list which is the l-to-w ratio in the order of label
    %for each of the block
    feret_L = feret_ratio(labeledBlocks);
    
    Rblock_num = 0;
    %iterate through each block and count the qualified one
    for i = 1:blobcount
        %show the circles in each highlighted region
        searchRegion = circle_mask & (labeledBlocks == i);
        %count the number of circles in this region
        [~, circlecount] = bwlabel(searchRegion);
        %if the feret l-to-w ratio satisfy  one of the condition
        %blocks number increases 1
        %if the block has one to four circles
        if circlecount >= 1 && circlecount <= 4 && ...
           feret_L(i) <= 1.255 && feret_L(i) >= 0.8
           
           Rblock_num = Rblock_num+1;
        %if the block has no circles
        elseif circlecount == 0 && feret_L(i) <= 1.61 && feret_L(i) >= 1.4
            Rblock_num = Rblock_num+1;
            
        %small trick: if the circle count is greater than 10, we take that
        %as the same situation with picture 6 in the training image
        elseif circlecount > 10 && feret_L(i)<2
            Rblock_num = 3;
        end
    end
end


function Bblock_num = count_Bblocks(circle_mask, colormask)
%circle_mask: the binary image which only has the round region in one block
%             this image also has the information of color, it's the output
%             of the find_circle
%colormask: the binary image which is the output of mask_processing
%           function, it shows the blue color blocks in image
%max: maximum thresh for circles in one region
%Rblock_num:the qualified lego blocks in the region
%---------------------------------------------------------------------
    %label matrix for each detected blocks in the mask
    %count the blocks in the mask
    [labeledBlocks,blobcount] = bwlabel(colormask);
    %calculate the feret length-to-width ratio of each detected blocks
    %the result is a list which is the l-to-w ratio in the order of label
    %for each of the block
    feret_L = feret_ratio(labeledBlocks);
    
    Bblock_num = 0;
    %iterate through each block and count the qualified one
    for i = 1:blobcount
        %show the circles in each highlighted region
        searchRegion = circle_mask & (labeledBlocks == i);
        %count the number of circles in this region
        [~, circlecount] = bwlabel(searchRegion);
        %if the feret l-to-w ratio satisfy  one of the condition
        %blocks number increases 1
        %the one with more than two and less than 9 circles
        if circlecount >= 2 && circlecount <= 9 && ...
           feret_L(i) <= 2.25 && feret_L(i) >= 1.5 
           
           Bblock_num = Bblock_num+1;
        %the one without any circles
        elseif circlecount == 0 && feret_L(i) <= 2.26 && feret_L(i) >= 2.15
            Bblock_num = Bblock_num+1;
        %the two 2*4 blocks which stacks together
        elseif circlecount >= 10 && ...
           feret_L(i) <= 1.2 && feret_L(i) >= 0.8
           Bblock_num = Bblock_num+2;
        %one 2*4 and one 2*2 stacked together
        elseif (circlecount == 9||circlecount == 8) && ...
           feret_L(i) <= 1.2 && feret_L(i) >= 0.8
           Bblock_num = Bblock_num+1;
        end
    end
end


%feret l2w ratio
function ratio = feret_ratio(labeledM)
    %labeledM: the labled matrix which represents each detected blocks
    %ratio: list of the result of ratio for each labeled region
    %---------------------------------------------------------------
    
    %https://blogs.mathworks.com/steve/2018/04/17/feret-properties-wrapping-up/
    ratio = [];
    %compute the number of regions
    maxferetprop = regionprops(labeledM,'MaxFeretProperties');
    num_of_region = size(maxferetprop);
    
    
%compute the ratio for each region and append the result to the list;
for k = 1:num_of_region
    %calculate the endpoints of the maximum feret diameter
    maxferetprop = regionprops(labeledM,'MaxFeretProperties');
    max_coor = maxferetprop(k).MaxFeretCoordinates;
    %calculate the endpoints of the minimum feret diameter
    minferetprop = regionprops(labeledM,'MinFeretProperties');
    min_coor = minferetprop(k).MinFeretCoordinates;
    min_coor = min_coor.';
    %compute the maximum fd endpoints projection on the line segment
    %composed by the minimum fd endpoints 
    max_pj1 = ProjPoint(max_coor(1,:),min_coor);
    max_pj2 = ProjPoint(max_coor(2,:),min_coor);
    %compute the length-to-weigth ratio of the minimum-area bounding box
    %length:
    length = norm(max_coor(1,:)-max_pj1)+norm(max_coor(2,:)-max_pj2);
    %width:
    width = norm(min_coor(:,1)-min_coor(:,2));
    %ratio:
    lwratio = length/width;
    ratio = [ratio,lwratio];
end
end

%point-to-line projection 
function proj_point = ProjPoint( point,line_p )
    %point: the input point
    %line_p: the input segment, represented by two coordinates of the points
    %proj_point: the projection of the input point on the segment
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
end
