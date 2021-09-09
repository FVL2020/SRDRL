function generate_mod_LR()
%% matlab code to genetate mod images, downsampled LR, upsampled images.

%% set parameters
input_folder = '../datasets/downsampling/PolyU_cropped';

% downsampled LR
up_scale = 4;
LR_mod = 'bicubic'; %bicubic,bilinear,nearest
% save_LR_folder = [input_folder, '_', LR_mod];
save_LR_name = [LR_mod, '_'];

% mod images
mod_scale = 4;
% mod = '';
% save_mod_folder = '';
% save_mod_name = '';

% upsampled images
SR_mod = 'bicubic';
save_SR_folder = [input_folder, '_', SR_mod, 'Up'];
save_SR_name = ['_', SR_mod, 'Up',];

if exist('save_mod_folder', 'var')
    if exist(save_mod_folder, 'dir')
        disp(['It will cover ', save_mod_folder]);
    else
        mkdir(save_mod_folder);
    end
end
if exist('save_LR_folder', 'var')
    if exist(save_LR_folder, 'dir')
        disp(['It will cover ', save_LR_folder]);
    else
        mkdir(save_LR_folder);
    end
end
if exist('save_SR_folder', 'var')
    if exist(save_SR_folder, 'dir')
        disp(['It will cover ', save_SR_folder]);
    else
        mkdir(save_SR_folder);
    end
end

idx = 0;
filepaths = dir(fullfile(input_folder,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(input_folder, [imname, ext]));
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        if exist('save_mod_folder', 'var')
            imwrite(img, fullfile(save_mod_folder, [imname, '.png']));
        end
        % LR
        im_LR = imresize(img, 1/up_scale, LR_mod);
        if exist('save_LR_folder', 'var')
            imwrite(im_LR, fullfile(save_LR_folder, [save_LR_name, imname, '.png']));
        end
        % SR
        if exist('save_SR_folder', 'var')
            im_B = imresize(im_LR, up_scale, SR_mod);
            imwrite(im_B, fullfile(save_SR_folder, [save_LR_name, imname, save_SR_name, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
