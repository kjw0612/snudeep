classdef BatchNorm < dagnn.ElementWise
  properties
    ndim = 1;
    mu = [];
    sigma = [];
    alpha = 0.1;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      if strcmp(obj.net.mode, 'test')
        outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, 'Moments', [obj.mu,obj.sigma]) ;
        return ;
      end
      outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnrelu(inputs{1}, derOutputs{1}) ;
      [derInputs{1}, derParams{1}, derParams{2}, moments] = ...
          vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}) ;
      obj.mu = (1 - obj.alpha) * obj.mu + obj.alpha * moments(:, 1);
      obj.sigma = (1 - obj.alpha) * obj.sigma + obj.alpha * moments(:, 2);
    end
    
    function params = initParams(obj)
      params{1} = ones(obj.ndim, 1, 'single');
      params{2} = zeros(obj.ndim, 1, 'single');
      obj.mu = zeros(obj.ndim, 1, 'single');
      obj.sigma = zeros(obj.ndim, 1, 'single');
    end
    
    function obj = BatchNorm(varargin)
      obj.load(varargin) ;
    end
  end
end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        