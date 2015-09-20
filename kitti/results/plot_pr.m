close all;
colors = [ ...
255, 0, 0; ...
228, 229, 97 ; ...
163, 163, 163 ; ...
218, 71, 56 ; ...
219, 135, 45 ; ...
145, 92, 146 ; ...
83, 136, 173 ; ...
135, 130, 174 ; ...
225, 119, 174 ; ...
142, 195, 129 ; ...
138, 180, 66 ; ...
223, 200, 51 ; ...
92, 172, 158 ; ...
177,89,40; ...
255, 255, 90; ...
188, 128, 189;
% 255, 255, 0; ...
% 0, 0, 255; ...
] ./ 256;

addpath(genpath('~/Documents/cxz/legendflex'));
addpath('~/Documents/cxz/export_fig-master/');

cls_id = 3;
metric = 'detection';
metric = 'orientation';

classes = {'Car', 'Pedestrian', 'Cyclist'};

methods = cell(3,1);
% 'DenseBox', 'DeepInsight',
methods{1} = {'Ours',  'Regionlets', '3DVP', 'SubCat', 'AOG', ...
              'OC-DPM', 'DPM-VOC+VP', 'MDPM-un-BB', 'DPM-C8B1', ...
              'ACF-SC', 'LSVM-MDPM-sv', 'LSVM-MDPM-us', 'ACF', 'mBoW'};
          
methods{2} = {'Ours', 'Regionlets', 'MV-RGBD-RF', 'pAUCEnsT', ...
              'FilteredICF', 'Fusion-DPM', 'DA-DPM', 'DPM-VOC+VP', ...
              'ACF-SC', 'SquaresICF', 'ACF', 'LSVM-MDPM-sv', ...
              'LSVM-MDPM-us', 'mBoW', 'DPM-C8B1'};
          
methods{3} = {'Ours', 'Regionlets', 'MV-RGBD-RF', 'pAUCEnsT', ...
              'DPM-VOC+VP', 'LSVM-MDPM-us', 'DPM-C8B1', 'LSVM-MDPM-sv', ...
              'mBoW'};
          
methods = methods{cls_id};
methods = methods(end:-1:1);
classes = classes(cls_id);

%%
aps = cell(length(classes), 1);
for k = 1 : length(classes)
    cls = classes{k};

    pr = cell(length(methods), 1);
    sel = false(length(methods), 1);
    idx = 0;
    for i = 1 : length(methods)
        fname = sprintf('%s/%s_%s.txt', methods{i}, cls, metric);
        if exist(fname, 'file')
            pr{i} = load(fname);   % before reg.
            idx = idx + 1;
            aps{k}(idx,:) = kittiAP(pr{i});
            sel(i) = true;
        end
    end
    methods = methods(sel);
    pr = pr(sel);
    colors(1, :) = colors(length(methods),:);
    colors(length(methods),:) = [1, 0, 0];

    levels = {'Easy', 'Moderate', 'Hard'};
    for i = 1:3
        labels = cell(length(pr), 1);
        fig_idx = (k-1) * length(classes) + i;
        figure(fig_idx); hold on;
        for j = 1 : length(pr)
            plot(pr{j}(:, 1), pr{j}(:, i+1), 'color', colors(j,:), 'linewidth', 2);
            labels{j} = sprintf('%s %.2f', methods{j}, aps{k}(j,i)*100);
        end

        lg_location = 'SouthWest';
        if (strcmpi(cls, 'pedestrian') && strcmpi(levels{i}, 'hard')) || ...
           (strcmpi(cls, 'cyclist') && (strcmpi(levels{i}, 'moderate') || ...
           strcmpi(levels{i}, 'hard')))
            lg_location = 'NorthEast';
        end
        xlabel('Recall');
        if strcmpi(metric, 'detection')
            ylabel('Precision');
        else
            ylabel('Orientation Similarity');
        end
        %title(sprintf('%s (%s)', cls, levels{i}));
        setlegend( labels, lg_location );
        %legend(labels, 'location', lg_location);
        %legendshrink(0.5);
        %legend boxoff;
        % save to file
        grid on;
        hei = 12;
        wid = 12;
        set(gcf, 'Units','centimeters', 'Position',[0 0 wid hei]);
        set(gcf, 'PaperPositionMode','auto');
        printpdf(sprintf('%s_%s_%s.pdf', metric, cls, levels{i}));
    end
end

fprintf('Easy   Moderate  Hard\n');
for k = 1 : length(classes)
    fprintf('%s:\n', classes{k});
    for i = 1 : size(aps{k}, 1)
        fprintf('%.2f  %.2f  %.2f\n', aps{k}(i,:)*100);
    end
end