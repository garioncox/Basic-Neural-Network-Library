// Garion Cox
// Dec 13, 2021
// CS 1410 Final Project


using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class MyNetwork : INetwork
    // A Neural Network that holds a list of inputs, outputs, can train itself, and get an output from an input
    // This is very tightly coupled, but was written in half a day to try to get it to work
    {

        // REQUIREMENT #15: Two Built-In Generic Collection Data Types
        List<ValueNode> _valueNodes = new List<ValueNode>();
        List<InputNode> _inputs;
        List<CostNode> _costs;
        List<AbstractOperationNode> _predictionNodes = new List<AbstractOperationNode>();
        double _gamma;

        // Constructor with listed weights and biases
        public MyNetwork(List<InputNode> inputs, List<List<AbstractOperationNode>> hiddenLayers, List<CostNode> costs,
                         List<double> weightValues, List<double> biasValues, double gamma = 0.1) 
        {
            _inputs = inputs;
            _costs = costs;
            _gamma = gamma;

            int weightCount = 0;
            int biasCount = 0;

            // Link input nodes to hidden layer nodes
            foreach (AbstractOperationNode o in hiddenLayers[0]) {
                foreach (AbstractConstantNode i in inputs) {
                    MultiplyNode m = CreateWeightedMultiplyNode(weightValues[weightCount], i);
                    o.linkInputNode(m);

                    weightCount++;
                }
                
                ValueNode b = new ValueNode(biasValues[biasCount]);
                o.linkInputNode(b);
                _valueNodes.Add(b);

                biasCount++;
            }

            // Link all hidden layer nodes together
            for (int i = 1; i < hiddenLayers.Count; i++) {
                List<AbstractOperationNode> nextLayer = hiddenLayers[i];
                List<AbstractOperationNode> previousLayer = hiddenLayers[i-1];

                foreach (AbstractOperationNode nextNode in nextLayer) {
                    foreach (AbstractOperationNode previousNode in previousLayer) {
                        MultiplyNode m = CreateWeightedMultiplyNode(weightValues[weightCount], previousNode);
                        nextNode.linkInputNode(m);

                        weightCount++;
                    }

                    ValueNode b = new ValueNode(biasValues[biasCount]);
                    nextNode.linkInputNode(b);
                    _valueNodes.Add(b);

                    biasCount++;
                }
            }

            // Link last hidden layer nodes to cost nodes
            foreach (CostNode c in costs) {

                // Link last hidden layer nodes to prediction nodes
                AbstractOperationNode p = new AddNode();

                foreach (AbstractOperationNode lastLayerNode in hiddenLayers[hiddenLayers.Count - 1]) {
                    MultiplyNode m = CreateWeightedMultiplyNode(weightValues[weightCount], lastLayerNode);
                    weightCount++;

                    c.linkInputNode(m);

                    // the prediction node should have no influence on the other nodes, so the prediction node is hidden from the other nodes
                    p.addInputNode(m);
                }
                
                ValueNode b = new ValueNode(biasValues[biasCount]);
                _valueNodes.Add(b);
                biasCount++;

                c.linkInputNode(b);

                p.linkInputNode(b);
                _predictionNodes.Add(p);
            }
        }

        // Constructor with randomly initialized weights and biases (between -1, 1)

        // EXTRA CREDIT #3: Operator Overloading
        public MyNetwork(List<InputNode> inputs, List<List<AbstractOperationNode>> hiddenLayers, List<CostNode> costs, double gamma = 0.1) 
        {
            _inputs = inputs;
            _costs = costs;
            _gamma = gamma;

            Random r = new Random();

            int weightCount = 0;
            int biasCount = 0;

            // Link input nodes to hidden layer nodes
            foreach (AbstractOperationNode o in hiddenLayers[0]) {
                foreach (AbstractConstantNode i in inputs) {
                    MultiplyNode m = CreateWeightedMultiplyNode(r.NextDouble() * r.Next(-1, 1), i);
                    o.linkInputNode(m);

                    weightCount++;
                }
                
                ValueNode b = new ValueNode(r.NextDouble() * r.Next(-1, 1));
                o.linkInputNode(b);
                _valueNodes.Add(b);

                biasCount++;
            }

            // Link all hidden layer nodes together
            for (int i = 1; i < hiddenLayers.Count; i++) {
                List<AbstractOperationNode> nextLayer = hiddenLayers[i];
                List<AbstractOperationNode> previousLayer = hiddenLayers[i-1];

                foreach (AbstractOperationNode nextNode in nextLayer) {
                    foreach (AbstractOperationNode previousNode in previousLayer) {
                        MultiplyNode m = CreateWeightedMultiplyNode(r.NextDouble() * r.Next(-1, 1), previousNode);
                        nextNode.linkInputNode(m);

                        weightCount++;
                    }

                    ValueNode b = new ValueNode(r.NextDouble() * r.Next(-1, 1));
                    nextNode.linkInputNode(b);
                    _valueNodes.Add(b);

                    biasCount++;
                }
            }

            // Link last hidden layer nodes to cost nodes
            foreach (CostNode c in costs) {

                // Link last hidden layer nodes to prediction nodes
                AbstractOperationNode p = new AddNode();

                foreach (AbstractOperationNode lastLayerNode in hiddenLayers[hiddenLayers.Count - 1]) {
                    MultiplyNode m = CreateWeightedMultiplyNode(r.NextDouble() * r.Next(-1, 1), lastLayerNode);
                    weightCount++;

                    c.linkInputNode(m);

                    // the prediction node should have no influence on the other nodes, so the prediction node is hidden from the other nodes
                    p.addInputNode(m);
                }
                
                ValueNode b = new ValueNode(r.NextDouble() * r.Next(-1, 1));
                _valueNodes.Add(b);
                biasCount++;

                c.linkInputNode(b);

                p.linkInputNode(b);
                _predictionNodes.Add(p);
            }
        }

        // Creates a MultiplyNode with two inputs: the abstractnode and a value (weight) node.
        // This weights the output of the previous node, and is how the network trains itself.
        MultiplyNode CreateWeightedMultiplyNode(double weightValue, AbstractNode inputNode) {
            ValueNode weight = new ValueNode(weightValue);
            MultiplyNode m = new MultiplyNode();
            m.linkInputNode(weight);
            m.linkInputNode(inputNode);

            _valueNodes.Add(weight);

            return m;
        }

        // Returns the ouput of the network in list format [0, 0, 0, 1, 0...]
        public List<double> GetPrediction(List<double> inputValues) {
            updateInputValues(inputValues);
            foreach (AbstractConstantNode v in _valueNodes) {
                v.getAdjoint();
            }

            List<double> temp = new List<double>();
            foreach (AbstractOperationNode p in _predictionNodes) {
                p.HasComputedOutput = false;
                temp.Add(p.getOutput());
            }

            return temp;
        }

        // Trains the network on a single DataPoint
        public void Train(DataPoint<double> dataValue) {
            updateInputValues(dataValue.InputData);
            updateActualOutputValues(dataValue.ActualValues);

            // Compare the error for each output
            foreach (CostNode c in _costs) {
                c.getOutput();
            }

            // Scale the values accordingly
            foreach (AbstractConstantNode v in _valueNodes) {
                v.Value -= v.getAdjoint() * _gamma;
            }
        }

        // Update the expected values in order for the network to know how wrong it was, and adjust accordingly
        public void updateActualOutputValues(List<double> values) {
            if (values.Count != _costs.Count) {

                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Cost nodes != input count. \nNodes: {_costs.Count} \nValues: {values.Count}\n");
                Console.ResetColor();

                throw new IndexOutOfRangeException(); // TODO: Custom exception?
            }

            for (int i = 0; i < values.Count; i++) {
                _costs[i].ActualValue = values[i];
            }

        }
        
        // Update the input data points of the network
        public void updateInputValues(List<double> values) {

            // REQUIREMENT #9: Throwing and Catching Exceptions
            try {
                for (int i = 0; i < values.Count; i++) {
                    _inputs[i].Value = values[i];
                }
            } catch {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Input nodes != input count. \nNodes: {_inputs.Count} \nValues: {values.Count}\n");
                Console.ResetColor();

                throw new IndexOutOfRangeException();
            }
        }

    }
}