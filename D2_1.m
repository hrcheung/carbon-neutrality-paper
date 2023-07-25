%%--------------------------------------------------------%%
%%                    TVP-VAR package                     %%
%%--------------------------------------------------------%%
%%
%%  [] = D2_1(vt, fldraw)
%%
%%  "D2_1" draws time-varying D2_8 response
%%
%%  [input]
%%   (fldraw = 1)
%%     vt:   m*1 vector of horizons to draw D2_8
%%   (fldraw = 0)
%%     vt:   m*1 vector of time points to draw D2_8
%%

function [] = D2_1(vt, fldraw)

global m_ns m_nk m_nl m_asvar;

ns = m_ns;  
nk = m_nk; 
nl = m_nl; 

mimpr = xlsread('tvpvar_imp.xlsx');
mimpm = mimpr(:, 3:end);
%floor
nimp = size(mimpm, 1) / m_ns;
mline = [0 .5 0; 0 0 1; 1 0 0; 0 .7 .7];
vline = {':', '--', '-', '-.'};
nline = size(vt, 2);

figure
for i = 1 : nk
  for j = 1 : nk
    id = (i-1)*nk + j;
    %1:nimp*m_ns,id
    mimp = reshape(mimpm(:, id), nimp, ns)';
    subplot(nk, nk, id);

    if fldraw == 1
        
      for k = 1 : nline
        plot(mimp(:, vt(k)+1), char(vline(k)), ...
             'Color', mline(k, :))
        hold on
      end
      vax = axis;
      axis([nl+1 ns+1 vax(3:4)])
      if vax(3) * vax(4) < 0
        line([nl+1, ns+1], [0, 0], 'Color', ones(1,3)*0.6)
      end
      if id == 1
        vlege = ['-period ahead'];
        for l = 2 : nline
          vlege = [vlege; '-period      '];
        end
        legend([num2str(vt') vlege])
      end

    else
        
      for k = 1 : nline
        plot(0:nimp-1, mimp(vt(k), :), char(vline(k)), ...
             'Color', mline(k, :))
        hold on
      end
      vax = axis;
      axis([0 nimp-1 vax(3:4)])
      if vax(3) * vax(4) < 0
        line([0, nimp-1], [0, 0], 'Color', ones(1,3)*0.6)
      end
      if id == 1
        vlege = ['t='];
        for l = 2 : nline
          vlege = [vlege; 't='];
        end
        legend([vlege num2str(vt')])
      end
        
    end

    hold off
    title(['$\varepsilon_{', char(m_asvar(i)), ...
           '}\uparrow\ \rightarrow\ ', ...
           char(m_asvar(j)), '$'], 'interpreter', 'latex')

  end
end
