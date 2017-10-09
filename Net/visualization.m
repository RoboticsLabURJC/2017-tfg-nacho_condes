# 
# Created on Mar 28, 2017
#
# @author: dpascualhe
#

function benchmark(results_path)
# This function reads and plots a variety of measures, which evaluate the neural
# network performance, that have been saved like a structure (a Python
# dictionary) into a .mat file.

  more off;
  
  results_path = file_in_loadpath(results_path);
  results_dict = load(results_path).metrics;
  
  if results_dict.("training") == "y"
    # Loss.
    figure('Units','normalized','Position',[0 0 1 1]);
    subplot(2,1,1)
    train_loss = results_dict.("training loss");
    val_loss = results_dict.("validation loss");
    new_val_loss = NaN(size(train_loss));
    
    val_index = length(train_loss)/length(val_loss);
    for i = [val_index:val_index:length(train_loss); 1:length(val_loss)]
      if i(1) != 0
        new_val_loss(i(1)) = val_loss(i(2));
      endif
    endfor

    x = 1:length(train_loss);
    plot(x, train_loss, x, new_val_loss, "r.")
    set(gca,"ytick", 0:0.2:max(train_loss), "ygrid", "on");
    title("Loss after each batch during training", "fontweight",...
          "bold", "fontsize", 15);
    h = legend("Training", "Validation", "location", "northeastoutside");
    set (h, "fontsize", 15);
    xlabel("Batch number", "fontsize", 15);
    ylabel("Categorical crossentropy", "fontsize", 15);

    # Accuracy.
    subplot(2,1,2)
    train_acc = results_dict.("training accuracy");
    val_acc = results_dict.("validation accuracy");
    new_val_acc = NaN(size(train_acc));

    
    for i = [val_index:val_index:length(train_acc); 1:length(val_acc)]
      if i(1) != 0
        new_val_acc(i(1)) = val_acc(i(2));
      endif
    endfor

    x = 1:length(train_acc);
    plot(x, train_acc, x, new_val_acc, "r.")
    set(gca,"ytick", 0:0.2:max(train_acc), "ygrid", "on");
    title("Accuracy after each batch during training", "fontweight",...
          "bold", "fontsize", 15);
    h = legend("Training", "Validation", "location", "northeastoutside");
    set (h, "fontsize", 15);
    xlabel("Batch number", "fontsize", 15);
    ylabel("Accuracy", "fontsize", 15);

    printf("=======================VALIDATION RESULTS=======================\n")
    printf("\nValidation loss (after each epoch):\n")
    for i = 1:length(val_loss)
      printf("  Epoch %2i: %1.5f\n", i, val_loss(i))
    endfor
    printf("\nValidation accuracy (after each epoch):\n")
    for i = 1:length(val_acc)
      printf("  Epoch %2i: %1.5f\n", i, val_acc(i))
    endfor
  endif
  
  # Precision.
  figure;
  subplot(2,1,1)
  precision = results_dict.("precision");
  bar(precision)
  nb_classes = length(precision);
  set(gca,"ytick", 0:1/nb_classes:1, "ygrid", "on");
  title("Test precision", "fontweight", "bold", "fontsize", 15);
  xlabel("Class", "fontsize", 15);
  ylabel("Precision", "fontsize", 15);

  # Recall.
  subplot(2,1,2)
  recall = results_dict.("recall");
  bar(recall)
  set(gca,"ytick", 0:1/nb_classes:1, "ygrid", "on");
  title("Test recall", "fontweight", "bold", "fontsize", 15);
  xlabel("Class", "fontsize", 15);
  ylabel("Recall", "fontsize", 15);
  
  # Confusion matrix.
  figure;
  conf_mat = results_dict.("confusion matrix");
  new_conf_mat = NaN(size(conf_mat)+1);
  
  for i = 1:length(conf_mat)
    new_conf_mat(i, length(new_conf_mat)) = sum(conf_mat(i, :));
    new_conf_mat(length(new_conf_mat), i) = sum(conf_mat(:, i));
  endfor
  
  pred_samples = sum(new_conf_mat(1:length(conf_mat), length(new_conf_mat)));
  real_samples = sum(new_conf_mat(length(new_conf_mat), 1:length(conf_mat)));
  if pred_samples != real_samples
    printf("Number of predicted and real samples is not equal")
  endif

  new_conf_mat(1:size(conf_mat)(1), 1:size(conf_mat)(2)) = conf_mat;
  
  imagesc(double(new_conf_mat));
  colormap(gray);
  thresh = max(new_conf_mat(:))/2;
  for i = 1:length(new_conf_mat)
    for j = 1:length(new_conf_mat)
      h = text(i, j, num2str(new_conf_mat(i, j)), "horizontalalignment",...
               "center", "verticalalignment", "middle");
      if new_conf_mat(i, j) < thresh
        set(h, "Color", "white");
      endif
    endfor
    j = 1;
  endfor
  title(strcat("Confusion matrix -> Samples = ", num2str(real_samples)),...
        "fontweight", "bold", "fontsize", 15);
  xlabel("Predicted", "fontsize", 15);
  ylabel("Real", "fontsize", 15);
  set(gca,"XTick", 1:1:(length(conf_mat)+1), "XTickLabel",{"0","1","2","3",...
                                                           "4","5","6","7",...
                                                           "8", "9", "TOTAL"});
  set(gca,"YTick", 1:1:(length(conf_mat)+1), "YTickLabel",{"0","1","2","3",...
                                                           "4","5","6","7",...
                                                           "8", "9", "TOTAL"});
  
  # We print the results.  
  printf("\n==========================TEST RESULTS==========================\n")
  printf("\nPrecision:\n")
  for i = 1:length(precision)
    printf("  %2i: %1.5f\n", i, precision(i))
  endfor
  printf("\nRecall:\n")
  for i = 1:length(recall)
    printf("  %2i: %1.5f\n", i, recall(i))
  endfor
  printf("\nConfusion Matrix:\n")
  disp(conf_mat)
  printf(["\nTest loss:\n " num2str(results_dict.("loss"))])
  printf(["\nTest accuracy:\n " num2str(results_dict.("accuracy")) "\n"])
endfunction
