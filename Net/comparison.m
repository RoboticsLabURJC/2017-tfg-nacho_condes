# 
# Created on May 15, 2017
#
# @author: dpascualhe
#

function comparison(path1, path2)

  more off;
  
  metrics_path1 = file_in_loadpath(path1);
  metrics_dict1 = load(metrics_path1).metrics;
  
  metrics_path2 = file_in_loadpath(path2);
  metrics_dict2 = load(metrics_path2).metrics;
  
  # Loss.
  figure('Units','normalized','Position',[0 0 1 1]);
  subplot(2,1,1)
  val_loss1 = metrics_dict1.("validation loss");
  val_loss2 = metrics_dict2.("validation loss");

  x1 = 1:0.001:length(val_loss1);
  x2 = 1:0.001:length(val_loss2);
  val_loss1 = interp1(val_loss1, x1);
  val_loss2 = interp1(val_loss2, x2);
  plot(x1, val_loss1, '.', x2, val_loss2, '.')
  set(gca,"ytick", 0:0.1:1, "ygrid", "on");
  title("Validation loss", "fontweight", "bold", "fontsize", 15);
  h = legend("Patience=2", "Patience=5", "location", "northeastoutside");
  set (h, "fontsize", 15);
  xlabel("Epoch number", "fontsize", 15);
  ylabel("Categorical crossentropy", "fontsize", 15);
  
  # Accuracy.
  subplot(2,1,2)
  val_acc1 = metrics_dict1.("validation accuracy");
  val_acc2 = metrics_dict2.("validation accuracy");

  x1 = 1:0.001:length(val_acc1);
  x2 = 1:0.001:length(val_acc2);
  val_acc1 = interp1(val_acc1, x1);
  val_acc2 = interp1(val_acc2, x2);
  plot(x1, val_acc1, ".", x2, val_acc2, ".")
  set(gca,"ytick", 0:0.1:1, "ygrid", "on");
  title("Validation accuracy", "fontweight", "bold", "fontsize", 15);
  h = legend("Patience=2", "Patience=5", "location", "northeastoutside");
  set (h, "fontsize", 15);
  xlabel("Epoch number", "fontsize", 15);
  ylabel("Accuracy", "fontsize", 15);
endfunction
