# feature list

## dependency graph

```mermaid
flowchart TD

c++ --> cuda

cuda --> rust-cuda


rust-core --> rust-cli
rust-core --> rust-cuda
rust-core --> python

python --> python-cuda
cuda --> python-cuda

c#
opencl

style c# fill         : #e884e8, stroke: #e884e8, stroke-width: 4px
style c++ fill        : #6da9ff, stroke: #6da9ff, stroke-width: 4px
style cuda fill       : #a8e64d, stroke: #a8e64d, stroke-width: 4px
style opencl fill     : #ff6f61, stroke: #ff6f61, stroke-width: 4px
style python fill     : #ffd343, stroke: #ffd343, stroke-width: 4px
style python-cuda fill: #ffd343, stroke: #a8e64d, stroke-width: 4px
style rust-cli fill   : #f7c9a8, stroke: #f7c9a8, stroke-width: 4px
style rust-core fill  : #f7c9a8, stroke: #f7c9a8, stroke-width: 4px
style rust-cuda fill  : #f7c9a8, stroke: #a8e64d, stroke-width: 4px
```

## dev features

| feature           | c++ | c#  | cuda | opencl | python | rust |
| ----------------- | --- | --- | ---- | ------ | ------ | ---- |
| format            | X   | O   | X    | X      | O      | O    |
| lint              | X   | O   | X    | X      | O      | O    |
| doc               | X   | O   | X    | X      | O      | O    |
| test              | X   | O   | X    | X      | O      | O    |
| bench             | X   | X   | X    | X      | X      | O    |
| ci                | O   | O   | O    | X      | O      | O    |
| update dependency | X   | O   | X    | X      | O      | O    |
| release           | O   | O   | O    | X      | O      | O    |
| publish           | X   | X   | X    | X      | O      | O    |

## features

| feature          | c++ | c#  | cuda | opencl | python | python(cuda) | rust | rust(cuda) |
| ---------------- | --- | --- | ---- | ------ | ------ | ------------ | ---- | ---------- |
| angle            | X   | X   | X    | X      | O      | X            | O    | X          |
| boundingbox      | X   | O   | X    | X      | X      | X            | X    | X          |
| crypto           | O   | X   | O    | X      | O      | O            | O    | O          |
| datum_compense   | O   | X   | O    | X      | O      | O            | O    | O          |
| gauss_projection | X   | X   | X    | X      | O      | X            | O    | X          |
| migrate          | X   | X   | X    | X      | O      | X            | O    | X          |
| space            | X   | X   | X    | O      | O      | X            | O    | X          |
