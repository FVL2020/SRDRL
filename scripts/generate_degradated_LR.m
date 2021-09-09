function generate_degradated_LR()
%% matlab code to genetate mod images, degraded LR images, upsampled images.

%% set parameters
% downsample method
LR_mod = 'bicubic'; % bicubic,bilinear,nearest

% blur kernel width
%kernelwidth = 3.8*rand + 0.2; % [0.2, 4)
kernelwidth = 0.2; % 0.2£»1.3£»2.6

% noise level
%nlevel = 55*rand;  % [0, 55)
nlevel = 0; % 0; 5; 15; 25; 50
         
% comment the unnecessary line
input_folder = '../datasets/downsampling/Set14_HR';
save_LR_folder = [input_folder, '_', LR_mod, '_', num2str(kernelwidth), '_', num2str(nlevel)];
save_LR_name = [LR_mod, '_'];
%save_mod_folder = '';
%save_mod_name = '';

up_scale = 4;
mod_scale = 4;

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
        %kernelwidth = 3.8*rand + 0.2; % random blur on fly
        str_rlt = sprintf('blur:%f.\n', kernelwidth);
        fprintf(str_rlt);
        blury_img = imfilter(im2double(img),double(fspecial('gaussian',15, kernelwidth)),'replicate'); % add blur
        
        LR = imresize(blury_img, 1/up_scale, LR_mod); % downsample
        
        %nlevel = 55*rand; % random noise on fly
        str_rlt = sprintf('noise:%f.\n', nlevel);
        fprintf(str_rlt);
        %randn('seed',0);
        im_LR = LR + nlevel/255.*randn(size(LR)); % add noise
        %im_LR = LR; % noise-free
        
        if exist('save_LR_folder', 'var')
            imwrite(im_LR, fullfile(save_LR_folder, [save_LR_name, imname, '.png']));
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
