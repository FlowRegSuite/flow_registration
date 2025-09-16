% Author   : Philipp Flotho
% Copyright 2021 by Philipp Flotho, All rights reserved.
% Example that downloads the jupiter demo data and demonstrates the minimal motion compensation config:

clear;
run('../set_path.m');

% preparing the data:
output_folder = 'jupiter_demo';
if ~isfolder(output_folder)
    mkdir(output_folder)
end
input_file = fullfile(output_folder, 'jupiter.tiff');
if (~exist(input_file, 'file'))
    websave(input_file, ...
        'https://drive.usercontent.google.com/download?id=12lEhahzKtOZsFgxLzwxnT8JsVBErvzJH&export=download&authuser=0');
end

options = OF_options(...
    'input_file', input_file, ...
    'output_path', fullfile(output_folder, 'hdf5_comp_minimal'), ... 
    'output_format', 'HDF5', ...
    'alpha', 4, ... 
    'quality_setting', 'balanced', ...
    'output_typename', '', ...
    'n_references', 2, ...
    'min_frames_per_reference', 20, ...
    'reference_frames', 1:250 ...
    );
compensate_recording(options);

vid = get_video_file_reader('jupiter_demo/hdf5_comp_minimal/compensated.HDF5');

% implay(mat2gray(vid));
load('jupiter_demo/hdf5_comp_minimal/statistics.mat')
load('jupiter_demo/hdf5_comp_minimal/reference_frames.mat')
ref1 = energy;
ref2 = energy;
ref2(idx == 1) = nan;
ref1(idx == 2) = nan;
figure;
subplot(3, 1, 1);
hold on
plot(ref1);
plot(ref2);
title("Energy");

subplot(3, 1, 2);
hold on
ref1 = max_disp;
ref2 = max_disp;
ref2(idx == 1) = nan;
ref1(idx == 2) = nan;
plot(ref1);
plot(ref2);
legend({"Max displacements, ref1", "Max displacements, ref2"});
title("Max displacements");

subplot(3, 1, 3);
imshowpair(reference_frames{1}, reference_frames{2}, 'montage');
title("Reference frames:")

%% postprocessing to get multiple videos:
n_videos = sum(diff(idx) ~= 0) + 1;
for i = 1:n_videos
    sub_reader = get_multireference_video(vid, idx, i);
    implay(mat2gray(sub_reader.read_frames(1:sub_reader.frame_count)));
    fprintf("video %i with %i frames with idx %i to %i\n", ...
        i, sub_reader.frame_count, sub_reader.idx(1), sub_reader.idx(end));
end
