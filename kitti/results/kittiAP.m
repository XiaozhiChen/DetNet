function [ aps ] = kittiAP( data )
%KITTIAP Summary of this function goes here
%   Detailed explanation goes here

rec = data(:,1);
for i = 1 : 3
    prec = data(:,i+1);
    ap=0;
    for t=1:4:41
        ap=ap+prec(t)/11;
    end
    aps(i) = ap;
end


% rec = data(:,1);
% for i = 1 : 3
%     prec = data(:,i+1);
%     ap=0;
%     for t=0:0.1:1
%         p=max(prec(rec>=t));
%         if isempty(p)
%             p=0;
%         end
%         ap=ap+p/11;
%     end
%     aps(i) = ap;
% end

end

