# Assignment 2

## Lab 2

### Loop Unrolling
* How do we handle when there is not at least four elements remaining?
When there is not at least four elements remaining, we handle these remaining elements in a separate for loop. 
We're trying to optimize the common case so we're giving up the efficiency of at most operating on 3 values than the other 1000s of values that may come in.
Zero padding is another valid approach if we can constrain the input to always be divisible by 4.

* What is happening at the instruction level that may make this more efficient?
At the instruction level these four elements can be pulled into fast memory (registers) to be operated on 4 at a time instead of pulling one element at a time into one register. 
This helps by having elements quickly accessible to be operated on.

### Neon
* How do we handle when there is not at least four elements remaining?
Here I had an issue where if I used the remainder for loop from loop_unrolling, I introduce a floating point error and the dataset fails on #5.
I would use the zero-pad approach in these situations to avoid the floating point error. 
Luckily, the memory allocation provided enough buffer that indexing outside of the array pulled a zero which let this solution pass all datasets.

* What is happening at the hardware level that may make this more efficient?
Using neon intrinsics allows us to use the SIMD extension feature of ARM which lets us load memory address of multiple array elements into a neon register and apply an operation to them in parallel. 
Thus the acronym Single Instruction, Multiple Data.

## Homework
### Naive Matrix Multiply
The Naive Matrix Multiply solution averaged a sys time of 0.500s. Within ranges [0.475,0575]
real    0m0.823s
user    0m0.277s
sys     0m0.521s

### Block Matrix Multiply
The Block Matrix Multiply solution managed more sys times below 0.500s.
I averaged a sys time of 0.465
real    0m0.738s
user    0m0.310s
sys     0m0.411s

### Block Matrix Multiply Unrolled
There was no significant improvement, sys times were still consistently below 0.500s
I averaged a sys time of 0.455
real    0m0.736s
user    0m0.280s
sys     0m0.433s

### Block Matrix Multiply Neon
There was no significant improvement, sys times were still consistently below 0.500s
I averaged a sys time of 0.450
real    0m0.844s
user    0m0.338s
sys     0m0.481s