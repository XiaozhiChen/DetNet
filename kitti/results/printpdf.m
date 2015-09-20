function printpdf(fname)
%PRINTPDF Prints the current figure into a pdf document

% set(gca, 'LooseInset', get(gca, 'TightInset'));
% fname = [regexprep(fname, '^(.*)\.pdf$', '$1'), '.eps'];
% print('-depsc', fname) ;
% if ~system(['epstopdf ', fname])
%   system(['rm ', fname]);
% end

% if fname(end-3) < '0' || fname(end-3) > '9'
%     hei = 11;
%     wid = 13;
%     set(gcf, 'Units','centimeters', 'Position',[0 0 wid hei]);
%     export_fig(fname, '-pdf', '-transparent', '-r300');
% else
%     export_fig(fname, '-pdf', '-transparent', '-r300', '-q1');
% end

% export_fig(fname, '-pdf', '-transparent', '-r300');
export_fig(fname, '-png', '-transparent');
