# What is Dias?

Dias is an automatic rewriter of `pandas` code for Jupyter (IPython) notebooks. It rewrites `pandas` code to semantically equivalent but faster versions, on-the-fly, transparently and correctly. Dias is extremely lightweight and it will incur virtually no extra runtime or memory overheads. At the same time, Dias can provide **100x or even 1000x speedups** (see example below).

Dias identifies rewrite opportunities automatically and leaves the rest of the code untouched, so you do not have to change a single line of your `pandas` code to use it.

## Quick Start

<table>
<tr>
<td>
<img src="https://baziotis.cs.illinois.edu/images/colab-icon.svg" style="width: 50px"/>
</td>

<td>
<a href="https://colab.research.google.com/drive/1Pe0N8pqfUReVWvYogXuvBD3YHQfhvlOR?usp=sharing">Quickstart Colab Notebook</a>
</td>
</tr>

</table>

The fastest way to get started is to play around with our [Quickstart Google Colab notebook](https://colab.research.google.com/drive/1Pe0N8pqfUReVWvYogXuvBD3YHQfhvlOR?usp=sharing). Otherwise, you can follow the documentation here to experiment locally.

<table>
<tr>
<th> <h3>Vanilla Pandas</h3> </th> <th> <h3>With Dias</h3> </th>
</tr>
<tr>
<td>


<table>
<!-- Add empty row to prevent zebra stripes -->
<tr></tr>

<tr>
  <td>

```python
import pandas as pd
import numpy as np
rand_arr = np.random.rand(2_500_000,20)
df = pd.DataFrame(rand_arr)
```

  </td>
</tr>

<!-- Add empty row to prevent zebra stripes -->
<tr></tr>

<tr>
  <td>

```python
%%time
def weighted_rating(x, m=50, C=5.6):
    v = x[0]
    R = x[9]
    return (v/(v+m) * R) + (m/(m+v) * C)
df.apply(weighted_rating, axis=1)
```

  </td>
</tr>


</table>

</td>

<td>


<table>
<!-- Add empty row to prevent zebra stripes -->
<tr></tr>

<tr>
  <td>

```python
import pandas as pd
# Import Dias. Keep everything 
# else the same.
import dias.rewriter
import numpy as np
rand_arr = np.random.rand(2_500_000,20)
df = pd.DataFrame(rand_arr)
```

  </td>
</tr>

<!-- Add empty row to prevent zebra stripes -->
<tr></tr>

<tr>
  <td>

```python
%%time
def weighted_rating(x, m=50, C=5.6):
    v = x[0]
    R = x[9]
    return (v/(v+m) * R) + (m/(m+v) * C)
df.apply(weighted_rating, axis=1)
```

  </td>
</tr>


</table>

</td>

</tr>

<tr>
<td>Original: 10.3s</td> <td>Rewritten: 48.4ms </td>
</tr>

<tr>
<td colspan="2">
<br/>
<h3>Speedup: 212x</h3>
<br/>
</td>
</tr>
</table>


### Installation

```
pip install dias
```

### Usage

Make sure that you are using a Jupyter/IPython notebook.

First import the package... That's it!
```python
import dias.rewriter
```

### Examples

Our [Quickstart notebook](https://colab.research.google.com/drive/1Pe0N8pqfUReVWvYogXuvBD3YHQfhvlOR?usp=sharing) contains many examples in a single place. You can also see our [examples directory](./examples/) which lists self-contained examples that showcase different use cases Dias.

## FAQ

### How lightweight is Dias?
Dias is extremely lightweight. In terms of memory overheads, anything that runs with vanilla `pandas`, runs with Dias enabled too. Dias is just a code rewriter, so it does not alter the way `pandas` stores data and its internal state is minimal.

Dias' runtime overheads are minimal too. In [our experiments](https://baziotis.cs.illinois.edu/papers/dias.pdf), the maximum overhead of Dias is 23ms. You may also want to take a look at [this example](https://github.com/ADAPT-uiuc/dias/blob/main/examples/crossing-boundaries.ipynb), where even though the original cell is quick, it is still worth using Dias.

### Can I inspect the rewritten version?

Yes. Dias' output is standard Python code, and so, for example, you do not need to know anything about Dias to know why you got a speedup. Similarly, you can just copy Dias' output and use it as any other Python code.

To inspect the rewritten version, add the comment `# DIAS_VERBOSE` at the beginning of your cell (right after any magic functions). See [this example](https://github.com/ADAPT-uiuc/dias/blob/main/examples/inspecting_output.ipynb).

### Is Dias a replacement for `pandas`?
No (which inherently means Dias does not suffer from lack of API support). Dias is a rewriter, which inspects and possibly rewrites `pandas` code.

### Does Dias work with a standard Python interpreter?
No. Dias currently uses IPython features.

### When does Dias rewrite code?

Dias looks for certain patterns, and upon recognizing one, it rewrites the code to a faster version. Thus, Dias will rewrite the code if it contains one of the patterns it is programmed to look for. Consider [this example](https://github.com/ADAPT-uiuc/dias/blob/main/examples/nsmallest.ipynb). One pattern Dias looks for is any expression followed by `sort_values()`, followed by `head()`. Upon recognizing this pattern, it rewrites the code to use `nsmallest()`. You can take a look at [the paper](https://baziotis.cs.illinois.edu/papers/dias.pdf) for more information.

Dias is still under early but active development, so expect more patterns to be added soon!

### Is Dias probabilistic? Is Dias an assistant?

No and no. Dias is not probabilistic; if it rewrites code, it is always correct (barring implementation bugs). Dias is also not intended to be an assistant. First, it's intended to be more quiet than an assistant. If Dias does its job correctly, then you should never have to think of it. Second, while you can inspect the rewritten code, Dias does not offer any explanations of why the rewritten version is faster.

## How to contribute

Dias is an ongoing research project by the [ADAPT group @ UIUC](https://github.com/ADAPT-uiuc). You can help us by sending us notebooks that you want to speed up and we will our best to make Dias do it automatically ([send us an email](mailto:sb54@illinois.edu) with either the notebook or Colab link)! Moreover, if you are aware of a pattern that can be rewritten to a faster version, please consider submitting an issue. You can use [our template](https://github.com/ADAPT-uiuc/dias/issues/new?assignees=&labels=pattern&template=pattern-request.md&title=%3CShort+description+of+the+original+and+the+rewritten%3E).

We also welcome feedback from all backgrounds, including industry specialists, data analysts and academics. Please reach out to sb54@illinois.edu to share your opinion!
