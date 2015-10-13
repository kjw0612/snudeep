function [y,maxmask] = vl_nnmaxout(x, varargin)
%VL_NNMAXOUT CNN maxout.
%   [Y,MASK] = VL_NNMAXOUT(X) applies maxout activation to the data X. MASK
%   is maxout mask for saving where 'max' came from.
%
%


% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.maxout_groupsize = 4;
opts.maxout_mask = [] ;

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
    dzdy = varargin{1} ;
    opts = vl_argparse(opts, varargin(2:end)) ;
else
    opts = vl_argparse(opts, varargin) ;
end

% determine mask
maxmask = opts.maxout_mask ;
g = opts.maxout_groupsize ;
channel = size(x,3) ;
if backMode && isempty(maxmask)
    warning('vl_nnmaxout: when using in backward mode, the mask should be specified') ;
end
if mod(size(x,3), g) ~= 0
    warning('vl_nnmaxout: group should divide the number of channels') ;
end


if isempty(maxmask)
    %     [i1,i2,i3,i4] = ind2sub(size(x),1:numel(x));
    %     i3 = ceil(i3/g);
    %     max_idx = accumarray([i1',i2',i3',i4'], x(:), [], @max) ;
    if isa(x,'gpuArray')
        max_idx = zeros([size(x,1), size(x,2), size(x,3)/g, size(x,4)],'gpuArray');
        for i=1:(channel/g)
            [~, max_idx(:,:,i,:)] = max(x(:,:,4*i-3:4*i,:),[],3);
        end
        maxmask = false(size(x),'gpuArray') ;
        for i=0:(channel/g-1)
            for j=1:g
                maxmask(:,:,4*i+j,:) = (max_idx(:,:,i+1,:)==j) ;
            end
        end
    else
        max_idx = zeros([size(x,1), size(x,2), size(x,3)/g, size(x,4)]);
        for i=1:(channel/g)
            [~, max_idx(:,:,i,:)] = max(x(:,:,4*i-3:4*i,:),[],3);
        end
        maxmask = false(size(x)) ;
        for i=0:(channel/g-1)
            for j=1:g
                maxmask(:,:,4*i+j,:) = (max_idx(:,:,i+1,:)==j) ;
            end
        end
    end
end

% do job

if ~backMode
    %     y = accumarray([i1',i2',i3',i4'], x(:), [], @max) ;
    y1 = bsxfun(@max, x(:,:,1:2:end,:), x(:,:,2:2:end,:));
    y = bsxfun(@max, y1(:,:,1:2:end,:), y1(:,:,2:2:end,:));
else
    if isa(x,'gpuArray')
        y = zeros(size(x),'gpuArray');
    else
        y = zeros(size(x));
    end
    for j=1:g
        y(:,:,j:g:end,:) = maxmask(:,:,j:g:end,:) .* dzdy;
    end
    y = single(y);
end
% 
% function ind = maxInd(x)
% [~, ind] = max(x) ;

