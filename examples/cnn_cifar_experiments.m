%% Experiment with the cnn_cifar.m

pbObj = pbNotify('wY8lenzuRIrOgfmcLNQPxfgzXMOPIPdC') ;
% usage : pbNotify('accessToken'); from https://www.pushbullet.com/account

expdir = 'data/cifar-maxout-init-3';

try
% [net_lenet, info_lenet] = cnn_cifar( ...
%   'modelType', 'lenet', 'expDir', 'data/cifar-lenet-1', 'gpus', [2], 'pushbullet', pbObj);

[net_maxout, info_maxout] = cnn_cifar( ...
  'modelType', 'maxout', 'expDir', expdir, 'gpus', [2], 'pushbullet', pbObj);
catch
    if ~isempty(pbObj)
        pbObj.notify('Error happened during execution');
    else
        fprintf('Error happened during execution');
    end
end

%%
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_maxout.val.objective, 'k') ; hold on ;
semilogy(info_lenet.val.objective, 'b') ;
xlabel('Training epochs'); ylabel('energy') ;
grid on ;
h=legend('maxout', 'maxpool') ;
set(h,'color','none');
title('objective') ;
subplot(1,2,2) ;
plot(info_maxout.val.error(1,:), 'k') ; hold on ; % first row for top1e
plot(info_maxout.val.error(2,:), 'k--') ; % second row for top5e
plot(info_lenet.val.error(1,:), 'b') ;
plot(info_lenet.val.error(2,:), 'b--') ;
h=legend(['maxout-val: ' num2str(min(info_maxout.val.error(1,:)))], ...
    ['maxout-val-5: ' num2str(min(info_maxout.val.error(2,:)))], ...
    ['lenet-val: ' num2str(min(info_lenet.val.error(1,:)))], ...
    ['lenet-val-5: ' num2str(min(info_lenet.val.error(1,:)))] );
grid on ;
xlabel('Trainig epochs'); ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;
saveas(gcf, fullfile(expdir, 'val_graph.png'));