function J = cost_function(x, y, theta)
  m = size(x, 1);
  preds = x * theta;
  errs = (preds - y) .^ 2;
  J = 1 / (2 * m) * sum(errs)
endfunction
