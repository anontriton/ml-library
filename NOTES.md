# ml-library

## credits
https://github.com/Magicalbat/videos\
followed this guy's youtube video but planning on adding more:\
    - char-level tokenizer, map each char to an int\
    - attention, a single-head causal self-attention block, then a small transformer\
    - BPE(maybe?)\
    will need some new pieces in the graph:\
        - an embedding lookup op with gradients\
        - transpose and scale ops\
        - a causal mask\
        - probably layer norm\
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

let's say we have some function f(x) where f is a vector and also some function g(y) where g is also a vector\
    - if we want to know f(g(y)) (the gradient of y with respect to f) we can say the gradient with respect to y of f is equal to the Jacobian of g times the gradient of x times g\
    - what is Jacobian?\
    - TLDR: kind of like a multi-dimensional chain rule\
    - allows us to convert from the gradient with respect to one variable to a gradient wit respect to another\
    - what we're doing is when we have our vectors we're chaining backwards, we're going to be computing its Jacobian vector product\
        - if you pass a vector into ReLU, its inputs don't depend on each other, so the Jacobian for a ReLU function will only be diagonal\
            - means it's trivial to compute the Jacobian at all\
model_prog_compute_grads(model_program* prog)\
    - loop through all our variables and clear their gradients\
    - if our curr var does not require gradients, then the variable does not have a gradient whatsoever and we can just continue\
    - if the curr var is a parameter, then we're not gonna clear it too\
    - we know that the last variable in the program is the output and that is the variable we want to find our gradient with respect to\
        - so we fill its gradient with 1 in other words that is giving us basically the gradient with respect to the sum of the final variable in our program, which in this case is fine but maybe to make it more robust is to explicitly have that last variable be just a single scalar but good enough for now\
    - to find gradient, we go backwards, backprop, reverse topo sort i think?\
    - if we only have one input and it doesn't require a gradient then we can just skip it, also check if cur does not require gradient\
    - at each step we're going to add our cur gradient to our inputs gradient and possibly modify it\
    - if we have just some function g(a, b) = a * b, our partial of g with respect to a is b and our partial of g with respect to b is just a\
        - so in theory when we're computing our partial c w/ respect to a, we take our cur gradient and multiply it by b and vice versa for a\
        - we have to transpose one of them bc it's matrix multiplication

in ReLU, there is no interdependence between variables in the input, so like the first variable in the output vector only depended on the input, but softmax depends on all the variables, so the Jacobian is not diagonal it has stuff everywhere, so let's just calculate the jacobian directly for softmax and then multiply by jacobian directly\
    - output has to be a vector for this to work

## model training
stochastic gradient descent\
    - take all of our training examples, split into batches, find gradients for parameters with respect to the cost for each example in the batch, avg it out, and directly subtract those gradients from our model\
    - true gradient descent goes over everything but this is lighter and easier for this project

# POST VIDEO

## redesigning notes

- problem: graph and model were tangled together
  - mv_create took a model_context* and set model->input / output / cost based on flags, so just making a node secretly changed the model
  - model_prog_create also took the model but only actually used num_vars
  - if we split into files, graph.h needs model.h and model.h needs graph.h (model_context holds model_programs) -> circular include, won't compile
- options were:
  - a) shove model_context into the graph file, everything lives together
  - b) forward declare model_context to break the cycle
  - c) redesign so the graph doesn't need the model at all <- going w/ this one
- why c: the transformer is gonna need more than one input (token ids, maybe positions), and a model with exactly one input slot filled in by a flag doesn't scale to that
  - "cleanest" isn't a reason by itself, a real upcoming requirement is
- graph = just the index counter (num_vars)
  - not about speed, training runs the compiled programs and never touches the graph anyway
  - what we give up w/o a node list: listing every param, saving/loading weights in a stable order (topo order depends on how the sort happened to walk), printing the whole graph
  - add a node list when we do save/load weights, don't need it yet
- model is told its nodes explicitly instead of through flags
  - flags fail quietly: flag two nodes INPUT and the last one silently wins, forget the flag and model->input is NULL and it crashes at some memcpy way later
  - explicit means model_compile can check everything up front (input set, cost set, label shape == output shape) and fail loud w/ a real message
  - downside is one more line to forget, but the validation catches that
- drop INPUT, OUTPUT, DESIRED_OUTPUT, COST flags, keep REQUIRES_GRAD and PARAMETER
  - REQUIRES_GRAD / PARAMETER describe the math so they belong on the node, the others are roles which belong to the model
- training is its own layer
  - model = what to compute, training = how to fit it (batching, feeding data, optimizer step, eval)
  - they change for different reasons, e.g. swap SGD for Adam w/o touching the model
  - move model_train as is, don't generalize until the LM actually needs something different (rule of three, don't abstract until 2-3 real uses)
- dependency chain, arrows only go one way:
  - arena -> matrix -> graph -> model -> train -> examples
- order: redesign inside main.c FIRST, then split into files
  - refactor = move code w/o changing behavior, redesign = change types/signatures/ownership
  - doing both at once means if accuracy tanks you don't know which one broke it
  - in one file the compiler shows every call site that needs updating
- stuff that breaks when splitting a unity build:
  - no include guards -> header included twice -> "typedef redefinition" (anonymous structs count as a new type every time)
  - compiling arena.c separately AND still #including it -> duplicate symbol linker errors
  - static vars silently get one copy per file if a .c gets #included twice, no error at all
  - non-static functions are now global symbols, make internal helpers static
  - _DEFAULT_SOURCE has to be defined before ANY system header or it does nothing (linux + -std=c11 hides MAP_ANONYMOUS otherwise)
  - makefile needs -MMD -MP or editing a header won't rebuild the .o files that use it (stale build bugs that vanish after make clean)
  - link order: objects first, then libml.a, then -lm
  - measure training time before/after, separate files lose cross-file inlining (-flto gets it back if it matters)

## gradient-check tests

- this is where numerical differentiation from the top of the notes finally gets used
  - too slow to train with (one forward pass per number in every param) but perfect for checking autograd on tiny matrices
- if a gradient is wrong nothing crashes, the model just trains worse, and you can't tell a bug apart from bad hyperparams. so check every op
- use central difference instead of the one sided version:
  - (f(x + h) - f(x - h)) / 2h
  - error shrinks like h^2 instead of h, way more accurate for the same h
- how one check goes:
  - make tiny random inputs (like 3x4)
  - turn the op's output into one scalar loss L = sum(out * r) where r is a random matrix
    - why not just sum(out)? then the upstream grad is all 1s, and some bugs (like a transpose mixup) can hide behind that
  - run backprop to get the analytic grad
  - for every element of every input: nudge it +h, recompute L, nudge it -h, recompute L, put it back, numeric grad = (L+ - L-) / 2h
  - compare analytic vs numeric
- compare w/ relative error, not absolute: |a - n| / max(|a|, |n|, tiny)
- we're in f32 so precision is only ~7 digits
  - h too small -> round off error takes over, h too big -> the approximation is bad, h ~1e-3 is a good spot
  - expect rel error around 1e-3 to 1e-2 in f32, double would get way lower but the lib is all f32
- gotchas:
  - relu has a kink at 0, if a nudge crosses 0 the numeric grad is garbage, keep test inputs away from 0
  - cross entropy clamps q at eps, keep q away from the clamp
  - matmul needs all 4 transpose variants tested
  - grads are +=, so also check accumulation: run backprop twice w/o clearing and the grad should double
- these become the template for every new op later (embedding, layer norm, etc), write the check alongside the op

## char-level tokenizer

- models only work w/ numbers, a tokenizer turns text into ints and back
- char-level = every unique char in the corpus gets an id
  - tiny shakespeare has ~65 unique chars so vocab size ~65
  - two lookup tables: char -> id (encode) and id -> char (decode), since chars are bytes a 256 entry array works for char -> id
  - decode(encode(text)) == text, easy first test
- char-level vs BPE:
  - char: tiny vocab, never hits an unknown token, dead simple, but sequences are long and the model has to learn spelling itself
  - BPE: repeatedly merges the most common pair of tokens into a new token -> shorter sequences, bigger vocab, more work to build
- making training data:
  - block_size = context length, how many chars the model sees at once
  - take a chunk of block_size + 1 chars, input = first block_size, target = same thing shifted over by one
  - every position predicts its next char, so one chunk = block_size training examples at once
- train/val split: first 90% train, last 10% val, contiguous chunks not shuffled chars. val loss going up while train loss goes down = overfitting
- one-hot times a matrix just picks out a row, so an embedding lookup is literally row indexing
  - backward only adds grad into the rows that got used (scatter add)
  - way cheaper than building a 65 wide one-hot and doing a matmul
- sanity check: at init the model should be guessing uniformly so loss should be about -ln(1 / vocab) = ln(65) ~ 4.17. if it starts way higher something's off (usually init)

## attention and a small transformer

- goal: predict the next char from all the previous ones
  - an MLP on a fixed window treats each position rigidly, attention lets every position decide how much to pull from each earlier position
- shapes change: so far everything's been column vectors (N x 1), now inputs are (T x C) matrices, T = tokens in the context, C = embedding size, one token per row
  - means X * W instead of W * x, gotta pick a convention and stick w/ it
- input = token embedding + position embedding
  - attention by itself has no idea what order the tokens are in, position embedding is how it knows
- single head self-attention:
  - Q = X Wq (what am i looking for), K = X Wk (what do i have), V = X Wv (what i'll hand over)
  - scores = Q K^T / sqrt(d_k), (T x T), row i = how much token i cares about each token
  - divide by sqrt(d_k) bc dot products grow w/ dimension, big scores -> softmax gets super peaky -> tiny gradients
  - causal mask: set scores[i][j] = -inf for j > i before softmax so a token can't peek at the future
    - exp(-inf) = 0 so masked spots get 0 weight, still fine w/ max subtraction since the diagonal is never masked
  - softmax EACH ROW separately, our current mat_softmax treats the whole matrix as one distribution so we need a row-wise version
  - out = weights * V, (T x C)
- multi-head = several smaller heads side by side, concat, then one more linear. start w/ single head
- transformer block (pre-norm):
  - x = x + attention(layernorm(x))
  - x = x + mlp(layernorm(x)), mlp expands to 4x C, relu, back down to C
  - the x + ... parts are residuals, same idea as the skip connection in the mnist model
- layer norm:
  - per row: subtract the mean, divide by the std (+ small eps), then scale and shift by learned gamma and beta
  - keeps activations a sane size as blocks stack up
  - its backward pass is the trickiest one we'll write, gradient check it for sure
- output: layernorm -> linear to vocab size -> logits (T x vocab)
- fused softmax + cross entropy:
  - with a one-hot target, the grad wrt the logits is just softmax(logits) - onehot, no jacobian at all
  - our current softmax grad builds a (size x size) jacobian, fine for 10 classes but dumb at vocab size and for every row
  - forward computes log(sum(exp)) w/ max subtraction so it's stable, no eps clamp needed
- new graph pieces this needs: embedding lookup, transpose, scale, causal mask, row-wise softmax, layer norm, fused softmax + CE
- generating text:
  - feed the context, take the last row's logits, softmax, sample a char (temperature = divide logits before softmax, < 1 safer, > 1 wilder), append it, crop to the last block_size chars, repeat
- debugging tips:
  - overfit one tiny batch first, loss should go to ~0, if it can't something is broken
  - plain SGD is slow on transformers, probably want Adam, which goes in the training layer

## misc

big picture overview\
- arena owns al the memory. every node and matrix and program live in in
- graph hands out node ids. mv_create and the mv_* ops build nodes and give each one the next number
- model knows which nodes play which roles (input, output, label, cost). it compiles the prgorams and runs the forward pass
- train loops over the data, copies samples into the model's input nodes, runs the programs and updates the params

