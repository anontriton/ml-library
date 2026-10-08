# ml-library

## credits
https://github.com/Magicalbat/videos
followed this guy's youtube video but planning on adding more:
    - char-level tokenizer, map each char to an int
    - attention, a single-head causal self-attention block, then a small transformer
    - BPE(maybe?)
    will need some new pieces in the graph:
        - an embedding lookup op with gradients
        - transpose and scale ops
        - a causal mask
        - probably layer norm
        - fuze softmax + cross-entropy op

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

We have a cost function and parameters and the goal is the find the cost gradient with respect to those parameters

reverse mode automatic differentation

given some computational graph, the funtion program compile topologically sorted some DAG to give us a list of variables in the order that they need to be executed, and then model program compute loops each variable and computed its output, then model program compute grads went through in rever topo order, bottom to top of our graph, and computed the gradient of each function that requires the gradient with respect to the output variable assuming that our actual output is the sum of the output vector

## Jacobian Vector Product

let's say we have some function f(x) where f is a vector and also some function g(y) where g is also a vector
    - if we want to know f(g(y)) (the gradient of y with respect to f) we can say the gradient with respect to y of f is equal to the Jacobian of g times the gradient of x times g
    - what is Jacobian?
    - TLDR: kind of like a multi-dimensional chain rule
    - allows us to convert from the gradient with respect to one variable to a gradient wit respect to another
    - what we're doing is when we have our vectors we're chaining backwards, we're going to be computing its Jacobian vector product
        - if you pass a vector into ReLU, its inputs don't depend on each other, so the Jacobian for a ReLU function will only be diagonal
            - means it's trivial to compute the Jacobian at all
model_prog_compute_grads(model_program* prog)
    - loop through all our variables and clear their gradients
    - if our curr var does not require gradients, then the variable does not have a gradient whatsoever and we can just continue
    - if the curr var is a parameter, then we're not gonna clear it too
    - we know that the last variable in the program is the output and that is the variable we want to find our gradient with respect to
        - so we fill its gradient with 1 in other words that is giving us basically the gradient with respect to the sum of the final variable in our program, which in this case is fine but maybe to make it more robust is to explicitly have that last variable be just a single scalar but good enough for now
    - to find gradient, we go backwards, backprop, reverse topo sort i think?
    - if we only have one input and it doesn't require a gradient then we can just skip it, also check if cur does not require gradient
    - at each step we're going to add our cur gradient to our inputs gradient and possibly modify it
    - if we have just some function g(a, b) = a * b, our partial of g with respect to a is b and our partial of g with respect to b is just a
        - so in theory when we're computing our partial c w/ respect to a, we take our cur gradient and multiply it by b and vice versa for a
        - we have to transpose one of them bc it's matrix multiplication

in ReLU, there is no interdependence between variables in the input, so like the first variable in the output vector only depended on the input, but softmax depends on all the variables, so the Jacobian is not diagonal it has stuff everywhere, so let's just calculate the jacobian directly for softmax and then multiply by jacobian directly
    - output has to be a vector for this to work

## model training
stochastic gradient descent
    - take all of our training examples, split into batches, find gradients for parameters with respect to the cost for each example in the batch, avg it out, and directly subtract those gradients from our model
    - true gradient descent goes over everything but this is lighter and easier for this project
