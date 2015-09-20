function setlegend( labels, location )
%CUSTOMLEGEND Summary of this function goes here
%   Detailed explanation goes here


switch location
    case {'NorthEast', 'ne'}
        anchor = {'ne', 'ne'};
        buffer = [-5 -5];
    case {'NorthWest', 'nw'}
        anchor = {'nw', 'nw'};
        buffer = [5 -5];
    case {'SouthEast', 'se'}
        anchor = {'se', 'se'};
        buffer = [-5 5];
    case {'SouthWest', 'sw'}
        anchor = {'sw', 'sw'};
        buffer = [5 5];
    otherwise
        anchor = {'ne', 'ne'};
        buffer = [-5 -5];
end
        
legendflex(labels, 'xscale', 0.3, 'box', 'off', 'anchor', anchor, 'buffer', buffer, 'fontsize', 12);

end

