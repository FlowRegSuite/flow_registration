% Author   : Philipp Flotho
% Copyright 2021 by Philipp Flotho, All rights reserved.

%% Example that downloads the jupiter demo data and simulates two channels with and without multiple files:

clear;
run('../set_path.m');

output_folder = 'jupiter_demo';
if ~isfolder(output_folder)
    mkdir(output_folder)
end
input_file = fullfile(output_folder, 'jupiter.tiff');
if (~exist(input_file, 'file'))
    websave(input_file, ...
        'https://drive.usercontent.google.com/download?id=12lEhahzKtOZsFgxLzwxnT8JsVBErvzJH&export=download&authuser=0');
end

%% preparing file writers and readers:
% creating the file reader (could also be MDF, maybe use:
% get_video_file_reader('input.mdf', 50, 5); to read frame batches of size 50
% with binning over 5 frames):

buffer_size = 50;
% In this example, we simulate a multichannel file by adding the same input
% file multiple times:
vid = get_video_file_reader({input_file, input_file}, buffer_size);

% creating the file writers, check get_video_file_writer for supported
% parameters. It can also take a string or cell (must fit the number of 
% channels!)as parameter to name the dataset, can contain * to indicate the
% position for the channel number:
hdf5_writer_single = get_video_file_writer(...
    fullfile(output_folder, 'jup_single.hdf5'), 'HDF5', 'dataset_names', 'ch*');
hdf5_writer_mult = get_video_file_writer(...
    fullfile(output_folder, 'jup_mult.hdf5'), 'MULTIFILE_HDF5', 'dataset_names', '/vid');

%% converting the tiff file to two different hdf formats:
while(vid.has_batch)
    tmp = vid.read_batch;
    hdf5_writer_mult.write_frames(tmp);
    hdf5_writer_single.write_frames(tmp);
end

%% motion compensation from the hdf5 file:
% different ways how to supply data to the OF_options file

% getting the hdf5_reader with custom parameters and passing it to the OF_options file:
hdf5_reader = get_video_file_reader(fullfile(output_folder, 'jup_single.hdf5'), ...
    [], [], 'dataset_names', {'ch1', '/ch2'});
% or:
hdf5_reader = get_video_file_reader(fullfile(output_folder, 'jup_single.hdf5'), ...
    [], [], 'dataset_names', 'ch*');
% or the following would work for a file where time is the first dimension, 
% then width, then height:
hdf5_reader = get_video_file_reader(fullfile(output_folder, 'jup_single.hdf5'), ...
    [], [], 'dataset_names', 'ch*', 'dimension_ordering', [3, 2, 1]);
% or simply (default ordering, datasets should have the same size):
hdf5_reader = get_video_file_reader(fullfile(output_folder, 'jup_single.hdf5'));
    
options = OF_options(...
    'input_file', hdf5_reader, ...
    'output_path', fullfile(output_folder, 'hdf5_comp'), ... 
    'output_format', 'HDF5', ...
    'bin_size', 1, ...
    'buffer_size', 24, ...
    'reference_frames', 100:200 ...
    );

% or of course file directly into OF_options:
options = OF_options(...
    'input_file', fullfile(output_folder, 'jup_single.hdf5'), ...
    'output_path', fullfile(output_folder, 'hdf5_comp'), ... 
    'output_format', 'HDF5', ...
    'alpha', 4, ... % choose a larger alpha to avoid registering the changing 
                ... % morphology of small structures (impact in that case)
    'min_level', 3, ... % instead of the quality setting, the min level for the 
                    ...   compensation can be set, the higher, the coarser the 
                    ...   resolution for the final solution
    'bin_size', 1, ...
    'buffer_size', 500, ...
    'reference_frames', 100:200 ...
    );

if (~exist(fullfile(output_folder, 'hdf5_comp', 'reference_frame.mat'), ...
        'file'))
    compensate_recording(options);
end

%% extracting and comparing the timecourses with and without motion compensation:
% resetting the file reader with the uncompensated video (after that,
% read_batch will return the very first batch again)
vid.reset;
vid_comp = get_video_file_reader(fullfile(output_folder, 'hdf5_comp', ...
    'compensated.hdf5'), buffer_size);

temporal_slice_comp = zeros(vid.get_height, vid.frame_count, vid.mat_data_type); 
temporal_slice_nocomp = zeros(vid.get_height, vid.frame_count, vid.mat_data_type);
time_course_comp = zeros(vid.frame_count, 1);
time_course_nocomp = zeros(vid.frame_count, 1);

video = zeros(vid.get_height, 2 * vid.get_width, 1, vid.frame_count, vid.mat_data_type);
imp_xy = [105, 220];

idx = 1;
while(vid.has_batch && vid_comp.has_batch)
    nocomp_buffer = vid.read_batch;
    comp_buffer = vid_comp.read_batch;
    n_frames = size(comp_buffer, 4);
    
    if idx == 1
        baseline_comp = mean(comp_buffer(:, :, :, 1:10), 4);
        baseline_nocomp = mean(nocomp_buffer(:, :, :, 1:10), 4);
        
        baseline_comp = squeeze(baseline_comp(imp_xy(2), imp_xy(1), 1, :));
        baseline_nocomp = squeeze(baseline_nocomp(imp_xy(2), imp_xy(1), 1, :));
    end
    
    time_course_comp(idx:idx+n_frames-1) = ...
        (double(squeeze(comp_buffer(imp_xy(2), imp_xy(1), 1, :))) - baseline_comp) / baseline_comp;
    time_course_nocomp(idx:idx+n_frames-1) = ...
        (double(squeeze(nocomp_buffer(imp_xy(2), imp_xy(1), 1, :))) - baseline_nocomp) / baseline_nocomp;
    
    temporal_slice_comp(:, idx:idx+n_frames-1) = squeeze(...
        comp_buffer(:, imp_xy(1), 1, :));    
    temporal_slice_nocomp(:, idx:idx+n_frames-1) = squeeze(...
        nocomp_buffer(:, imp_xy(1), 1, :));
    
    tmp = nocomp_buffer(:, :, 1, :);
    tmp(:, end+1:end+vid.get_width, :, :) = comp_buffer(:, :, 1,:);
    video(:, :, :, idx:idx+n_frames-1) = tmp;
    
    idx = idx + n_frames;
end

%% plotting and displaying the results:
implay(video, 60);
subplot(1, 3, 1);
imshow(temporal_slice_nocomp(95:270, :));
title('temporal slice, no motion compensation');
subplot(1, 3, 2);
imshow(temporal_slice_comp(95:270, :));
title('temporal slice, motion compensation');
subplot(1, 3, 3);
% info on fps taken from: https://www.chappelastro.com/astrophotography/solar_system/jupiter/2019/2019-08-07_04:10:36/
time = (1:length(time_course_nocomp)) / 66;
plot(time, time_course_nocomp);
hold on;
plot(time, time_course_comp);
grid on;
ylabel('relative intensity change');
xlabel('time (s)');
legend({'no motion compensation', 'motion compensation'});
title('time course of the impact event');
hold off