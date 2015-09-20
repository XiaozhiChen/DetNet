cache_name = 'kitti_test';

car_dir = 'kitti_car_test_1295_770771/data';
pc_dir = 'kitti_ped_cyc_test_1295_550551/data';
final_dir = 'all_test_1295_770771_550551/data';

if ~exist(final_dir, 'dir')
    mkdir(final_dir);
end

car_ids = dir(car_dir);
car_ids = setdiff({car_ids.name}, {'.', '..'});
pc_ids = dir(pc_dir);
pc_ids = setdiff({pc_ids.name}, {'.', '..'});
assert(all(ismember(car_ids, pc_ids)) == 1 && ...
    length(car_ids) == length(pc_ids));

for i = 1 : length(car_ids)
    fprintf('%d/%d\n', i, length(car_ids));
    id = car_ids{i};
    final_file = sprintf('%s/%s', final_dir, id);
    fid = fopen(final_file, 'w');
    fclose(fid);
    car_file = sprintf('%s/%s', car_dir, id);
    pc_file = sprintf('%s/%s', pc_dir, id);
    
    system(['cat ', car_file, ' >> ', final_file]);
    system(['cat ', pc_file, ' >> ', final_file]);
    
end
