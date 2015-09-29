% Proposal Detection set for COCO Images

%% set up the enviornment
clear;
img_dir = '/w/datasets/coco/train2014/';
coco_dir = '../coco/';
addpath([coco_dir,'MatlabAPI']);
addpath(genpath('../toolbox/'));
coco_type = 'train2014';
annFile = sprintf('%s/annotations/instances_%s.json',coco_dir,coco_type);
if (~exist('coco','var')), coco=CocoApi(annFile); end

%% load pre-trained edge detection model and set opts
model = load('models/forest/modelBsds'); 
model = model.model;
model.opts.multiscale = 0; 
model.opts.sharpen = 2; 
model.opts.nThreads = 4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

%% load coco images and detect Edge Box bounding box proposals
imgIds = coco.getImgIds();
for imgId = imgIds.'
    img = coco.loadImgs(imgId);
    img_filename = img.file_name;
    I = imread(sprintf('%s/images/%s/%s',coco_dir,coco_type,img.file_name));
    if size(I, 3) ~= 3
        continue;
    end
    bbs = edgeBoxes(I,model,opts);
    box_filename = ['COCO_',coco_type,'_',num2str(imgId, '%012d'),'.mat'];
    box_dir = sprintf('%s/proposals/edge_boxes_AR/mat/%s/%s', ...
        coco_dir, box_filename(1:14), box_filename(1:22));
    box_file = [box_dir,'/',box_filename];
    proposal = struct('boxes', bbs(:,1:4), 'scores', bbs(:,5), ...
        'num_candidates', size(bbs, 1));
    
    if ~exist(box_dir, 'file')
        mkdir(box_dir);
    end
    save(box_file, 'proposal');
end
