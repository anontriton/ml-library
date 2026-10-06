## ML FRAMEWORK
1) do math (w/ tensors)
2) find gradients
3) glue layer

## Notes
- ml is just input to outputf(x) --> y
    - ex: input could be some handwritten number and the output is some sort of probability distribution where the value at the num's index has a higher probability
- θ denotes parameters
- loss function C(x, y, θ)
    - take in some traning sample, input and some expected output and model parameters
    - quantifies how bad the model did
    - entire goal of ml is inimize cost by adjusting the parameters
- gradient of the cost function with respect to all the parameters tell us which direction to move

Differentiate:
1) numerical (f(x + h) - f(x)) / h
    - dealing with parameters with potentially thousands of numbers in them
    - to find the gradient, we would have to take a little step in each of those dimensions and compute the entire output
    - for that reason, we don't want to go with this method
2) symbolic "f(x) = x^2" -> "f'(x) = 2x"
    - you give it some sort of input string like above and it literally produces the symbols that give you something like f prime above
    - this works but theres some disadvantages
        - complicated to implement
        - a large number of symbols, especially when deriving multiple times
3) autograd
    - given some function, f(x) = mx + b, you define a computational graph for that function
    - you have some m and some x and let's say some intermediate variable z and we have b and it finally gives us f
        - from this, we're able to compute the gradients of inputs or even our parameters of m and b with respect to f by traversing this graph in certain ways
        - we'll be going with reverse mode, but there are other ways like forward mode

In a basic feedforward network, your output for some layer n is equal to ReLU of the weight in that layer times the output of the prev layer plus the bias in that layer
