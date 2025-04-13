function cost = costFunction(depth)
 if depth >= 0.95*3.2
     cost = 50000;
 elseif depth < 2/5
     cost = 0;
 else
     cost = 100 * (depth - 0.4);
 end
end