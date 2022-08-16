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
        'https://cloud.hiz-saarland.de/s/JpHyczRSMDbLwzP/download');
end

options = OF_options(...
    'input_file', input_file, ...
    'output_path', fullfile(output_folder, 'hdf5_comp_minimal'), ... 
    'output_format', 'HDF5', ...
    'alpha', 4, ... % choose a larger alpha to avoid registering the changing 
                ... % morphology of small structures (impact in that case)
    'quality_setting', 'balanced', ... % default behaviour is 'quality'
    'output_typename', '', ...
    'reference_frames', 1:10, ...
    'update_reference', true, ...
    'buffer_size', 10 ...
    );
compensate_recording(options);

vid_update_ref = get_video_file_reader('jupiter_demo/hdf5_comp_minimal/compensated.HDF5');
vid_update_ref = vid_update_ref.read_frames(1:vid_update_ref.frame_count);

options = OF_options(...
    'input_file', input_file, ...
    'output_path', fullfile(output_folder, 'hdf5_comp_minimal'), ... 
    'output_format', 'HDF5', ...
    'alpha', 4, ... % choose a larger alpha to avoid registering the changing 
                ... % morphology of small structures (impact in that case)
    'quality_setting', 'balanced', ... % default behaviour is 'quality'
    'output_typename', '', ...
    'reference_frames', 1:10 ...
    );
compensate_recording(options);

vid = get_video_file_reader('jupiter_demo/hdf5_comp_minimal/compensated.HDF5');
vid = vid.read_frames(1:vid.frame_count);

vid(:, end+1:end+size(vid, 2), :, :) = vid_update_ref;

implay(vid, 60);