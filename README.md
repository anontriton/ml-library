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

