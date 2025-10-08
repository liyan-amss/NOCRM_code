function [ACC] = ACC(gnd,res)
%ACC: get the ACC score for a clustering result
%   [ACC] = ACC(gnd,res);
%
%
%   

%==========
res = bestMap(gnd,res);
ACC = length(find(gnd == res))/length(gnd);