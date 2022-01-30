# Basic-Neural-Network-Library

This project is a neural network library. It consists of a bunch of nodes: operation and constant. All nodes cache their output and adjoint once computed,
but only constant nodes have a static value. Operation nodes have a dynamic output, since their output depends on their inputs.

I have implemented a FileIO and a DataPoint class. DataPoints contain a list of input data and a list of expected values. The FileIO returns a bunch of
DataPoints, which can be passed into a network in order to train it.

My tests are mainly to see if the nodes are behaving correctly, which you can test with "dotnet test". The AutomaticNetwork tests are currently incomplete,
since I ran out of time to implement the network correctly. 

You can create your own network by creating n amount of InputNodes, n amount of OperationNodes, and link them to n CostNodes.
In order to get the network to train, you also need to set ValueNodes for each OperationNode and CostNode, as weights and biases. You then set the
value for each InputNode (InputNode.Value = value) and compute the output of each CostNode (CostNode.getOutput()). To train the network, compute each
ValueNode's adjoint and scale it by some Gamma value (ValueNode.Value -= ValueNode.getAdjoint()*Gamma).

I have a basic structure in my Main file, which you can run with "dotnet run --project .\NeuralNetwork.Main\"

Code Requirements:
1) Class Definition
    -> AbstractNode.cs
2) Second Class Definition
    -> AbstractOperationNode.cs
3) Third Class Definition
    -> AbstractConstantNode.cs
4) Struct Definition
    -> ----
5) Enumerated Type
    -> ----
6) Inheritance
    -> AddNode.cs
7) Polymorphism
    -> CostNode.cs
8) Second Example of Polymorphism
    -> MultiplyNode.cs
9) Throwing and Catching Exception
    -> MyNetwork.cs
10) Generic Datatype
    -> DataPoint.cs
11) Properties
    -> AbstractNode.cs
12) Static Member Function
    -> MyFileIO.cs
13) Interface
    -> INode.cs
14) Second Interface
    -> INetwork.cs
15) Two Built-In Generic Collection Types
    -> MyNetwork.cs

Extra Credit:
1) Working With File IO
    -> MyFileIO.cs
2) Recursion
    -> AbstractNode.cs
3) Operation Overloading
    -> MyNetwork.cs