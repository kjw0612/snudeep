% This setup is tested under Ubuntu 14.04. CUDNN needs to be copied into local directory
addpath matlab
% download and extract CUDNN v2 rc2
if ~exist('./local', 'dir')
    fprintf('Download CUDNN v2 rc2\n');
    unzip('https://www.dropbox.com/s/s7wlgwt792z11ye/local.zip?dl=1');  
    fprintf('CUDNN sucessfully downloaded\n');
end
vl_compilenn('enableGPU', 1, 'cudaRoot', '/usr/local/cuda', 'cudaMethod', 'nvcc', 'enableCudnn', 1, 'cudnnRoot', 'local/');
