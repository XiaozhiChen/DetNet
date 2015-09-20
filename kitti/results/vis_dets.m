close all;

data_set = 'testing';
% res_dir = 'kitti_car_test_1295_770771/data';
res_dir = 'all_test_1295_770771_550551/data';
image_ids = 1 : 7518;

kitti_dir = '/w/datasets/kitti';
image_dir = fullfile(kitti_dir,['/object/' data_set '/image_2']);

for i = 100 : length(image_ids)
    id = sprintf('%06d', image_ids(i) - 1);
    
    % parse input file
    fid = fopen(sprintf('%s/%s.txt',res_dir,id),'r');
    C   = textscan(fid,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
    fclose(fid);

    im = imread(sprintf('%s/%s.png', image_dir, id));
    figure(1); 
    imshow(im);
    hold on;
    
    bbs = cat(2, C{5:8});
    scores = C{end};
    bbs = bbs(scores > 0, :);
    for j = 1 : size(bbs, 1)
        bb = bbs(j, :);
        plot(bb([1 3 3 1 1]), bb([2 2 4 4 2]), 'g', 'linewidth', 3);
        fprintf('Press any key to continue\n');
        pause;
    end
end